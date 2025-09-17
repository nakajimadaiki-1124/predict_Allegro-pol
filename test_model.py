#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test a trained NequIP/Allegro model on a dataset,
save predictions into an XYZ file, and generate parity plots
(DFT vs ML) for energy, polarization, polarizability, born charges.

- 学習は行わない（損失関数は無視）
- YAML のカスタムタグは安全に無視して読込
- force, stress は扱わない
- 予測はフレーム info / arrays に "predicted_*" として保存
- DFT(正解) と ML(予測) のパリティプロット PDF を出力
"""

import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from ase.io import read, write

# NequIP / Allegro
from nequip.utils import Config
from nequip.model import model_from_config
from nequip.data import AtomicData
from nequip.data.transforms import TypeMapper

# ==========
# 設定ここ
# ==========
CONFIG_PATH = "configs/BaTiO3.yaml"
MODEL_PATH  = "results/20250908/BaTiO3/best_model.pth"   # best_model.pth を使う
TEST_XYZ    = "data/BaTiO3.xyz"                          # DFT(正解)の extxyz
OUT_DIR     = "test_results"
ML_XYZ      = os.path.join(OUT_DIR, "BaTiO3-ML.xyz")     # 予測を書き出す extxyz
OUT_PDF     = os.path.join(OUT_DIR, "BaTiO3-parity.pdf") # パリティPDF

# =======================================================
# YAML のカスタムタグを無視して安全に読むためのローダー
# =======================================================
class SafeFullLoader(yaml.FullLoader):
    pass

def _unknown_constructor(loader, tag_suffix, node):
    # 未知の Python オブジェクトやタプル等は無視（None）
    return None

SafeFullLoader.add_multi_constructor("tag:yaml.org,2002:python/object:", _unknown_constructor)
SafeFullLoader.add_multi_constructor("tag:yaml.org,2002:python/tuple", _unknown_constructor)

def load_config_safely(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.load(f, Loader=SafeFullLoader)
    # 推論だけなので avg_num_neighbors=auto は None に潰す
    if cfg.get("avg_num_neighbors", None) == "auto":
        cfg["avg_num_neighbors"] = None
    return cfg

# ===================
# モデルのロード関数
# ===================
def load_model_from_config(cfg: dict, model_path: str, device: str):
    # NequIP の Config 経由でモデル組み立て
    model = model_from_config(Config.from_dict(cfg))

    # state_dict のキー形式を吸収
    sd = torch.load(model_path, map_location=device)
    if isinstance(sd, dict):
        if "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        elif "model_state_dict" in sd and isinstance(sd["model_state_dict"], dict):
            sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=False)

    model.to(device)
    model.eval()
    return model

# =========================================
# 予測 → XYZ 保存（force / stress は出さない）
# =========================================
def predict_and_save_xyz(model, test_file, out_file, device="cpu"):
    """テストデータを読み込み、モデル予測してXYZに保存"""
    from ase.io import read, write
    import torch
    from tqdm import tqdm

    frames = read(test_file, ":")  # 全フレーム読み込み
    results = []

    # モデルの dtype を取得（モデルに合わせる）
    try:
        model_dtype = next(p for p in model.parameters() if p.requires_grad).dtype
    except StopIteration:
        model_dtype = torch.float32

    for i, frame in enumerate(tqdm(frames, desc="Predicting")):
        try:
            # 勾配が必要な入力：pos（座標）
            pos = torch.tensor(
                frame.positions, dtype=model_dtype, device=device, requires_grad=True
            )
            # 勾配は不要：cell, pbc, atomic_numbers
            cell = torch.tensor(frame.cell.array, dtype=model_dtype, device=device)
            pbc = torch.tensor(frame.pbc, dtype=torch.bool, device=device)
            Z   = torch.tensor(frame.numbers, dtype=torch.long, device=device)

            inputs = {
                "pos": pos,
                "cell": cell,
                "pbc": pbc,
                "atomic_numbers": Z,
            }

            # ★ no_grad は使わない ★
            with torch.set_grad_enabled(True):
                pred = model(inputs)

            # 予測値を info に保存（force/stress は保存しない）
            info = {}
            if "total_energy" in pred:
                # スカラー（shape [1] や [[]]）を float に
                e = pred["total_energy"]
                info["predicted_total_energy"] = float(e.detach().cpu().reshape(-1)[0])
            if "polarization" in pred:
                P = pred["polarization"].detach().cpu().numpy()
                info["predicted_polarization"] = P.reshape(-1).tolist()
            if "polarizability" in pred:
                A = pred["polarizability"].detach().cpu().numpy()
                info["predicted_polarizability"] = A.reshape(-1).tolist()

            frame.info.update(info)
            results.append(frame)

        except Exception as e:
            tqdm.write(f"⚠️ frame {i} をスキップ: {e}")

    # 1件も残らないと extxyz が空ファイルになって ASE が怒るので回避
    if len(results) == 0:
        raise RuntimeError(
            "予測に成功したフレームが 0 件です。上のスキップ理由（最初の1〜2件）を確認してください。"
        )

    write(out_file, results, format="extxyz")
    print(f"✅ 予測XYZを書き出しました: {out_file}（{len(results)} フレーム）")

# ==============================
# extxyz から評価用データを抽出
# ==============================
def load_for_metrics(filename: str):
    """DFT/ML の extxyz から、評価に使う配列を取り出す。"""
    Fs = read(filename, ":")
    if not isinstance(Fs, list):
        Fs = [Fs]

    energy = []
    pol = []
    polz = []
    zstar = []  # list of (nat, 9)

    for at in Fs:
        # エネルギー
        if "total_energy" in at.info:
            energy.append(float(at.info["total_energy"]))
        elif "predicted_total_energy" in at.info:
            # ML 側の読み込みで使う
            energy.append(float(at.info["predicted_total_energy"]))
        else:
            energy.append(np.nan)

        # 分極
        if "polarization" in at.info:
            p = np.asarray(at.info["polarization"], dtype=float).reshape(3)
        elif "predicted_polarization" in at.info:
            p = np.asarray(at.info["predicted_polarization"], dtype=float).reshape(3)
        else:
            p = np.array([np.nan, np.nan, np.nan], dtype=float)
        pol.append(p)

        # 分極率 (3x3)
        if "polarizability" in at.info:
            a = np.asarray(at.info["polarizability"], dtype=float).reshape(9)
        elif "predicted_polarizability" in at.info:
            a = np.asarray(at.info["predicted_polarizability"], dtype=float).reshape(9)
        else:
            a = np.full(9, np.nan, dtype=float)
        polz.append(a)

        # Born 有効電荷 (nat,9) — arrays に入っている想定
        if "born_charge" in at.arrays:
            z = np.asarray(at.arrays["born_charge"], dtype=float)  # (nat,9) 仕様に合わせる
            # もし (nat,3,3) なら reshape
            if z.ndim == 3 and z.shape[1:] == (3, 3):
                z = z.reshape(z.shape[0], 9)
        elif "predicted_born_charge" in at.arrays:
            z = np.asarray(at.arrays["predicted_born_charge"], dtype=float)
            if z.ndim == 3 and z.shape[1:] == (3, 3):
                z = z.reshape(z.shape[0], 9)
        else:
            z = None
        zstar.append(z)

    energy = np.array(energy, dtype=float)
    pol = np.array(pol, dtype=float)              # (n,3)
    polz = np.array(polz, dtype=float)            # (n,9)
    # zstar は各フレームで nat が違い得るので list のまま返す
    return dict(energy=energy, polarization=pol, polarizability=polz, born_charge=zstar)

# ==========================
# パリティプロットのユーティリティ
# ==========================
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_absolute_error, mean_squared_error

def parity_plot(ax, y_true, y_pred, title):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t = y_true[mask].ravel()
    y_p = y_pred[mask].ravel()
    if y_t.size == 0:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
        ax.set_title(title)
        return
    mae = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    ax.scatter(y_t, y_p, alpha=0.6, s=16)
    lo, hi = min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("DFT")
    ax.set_ylabel("ML")
    ax.text(
        0.05, 0.95, f"MAE={mae:.4g}\nRMSE={rmse:.4g}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=9,
    )

def make_parity_pdf(dft_xyz: str, ml_xyz: str, out_pdf: str):
    dft = load_for_metrics(dft_xyz)
    ml  = load_for_metrics(ml_xyz)
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

    with PdfPages(out_pdf) as pdf:
        # Energy
        fig, ax = plt.subplots(figsize=(5, 5))
        parity_plot(ax, dft["energy"], ml["energy"], "Total Energy")
        pdf.savefig(fig); plt.close(fig)

        # Polarization: x,y,z
        labels = ["Px", "Py", "Pz"]
        for i in range(3):
            fig, ax = plt.subplots(figsize=(5, 5))
            parity_plot(ax, dft["polarization"][:, i], ml["polarization"][:, i], f"Polarization {labels[i]}")
            pdf.savefig(fig); plt.close(fig)

        # Polarizability: 3x3
        for i in range(3):
            for j in range(3):
                idx = 3 * i + j
                fig, ax = plt.subplots(figsize=(5, 5))
                parity_plot(ax, dft["polarizability"][:, idx], ml["polarizability"][:, idx], f"Polarizability α({i},{j})")
                pdf.savefig(fig); plt.close(fig)

        # Born charges: per-atom (全フレーム連結)
        dft_bc_all, ml_bc_all = [], []
        for d_z, m_z in zip(dft["born_charge"], ml["born_charge"]):
            if d_z is None or m_z is None:
                continue
            nat = min(d_z.shape[0], m_z.shape[0])
            dft_bc_all.append(d_z[:nat, :])  # (nat,9)
            ml_bc_all.append(m_z[:nat, :])
        if len(dft_bc_all) > 0:
            D = np.vstack(dft_bc_all)  # (Ntot,9)
            M = np.vstack(ml_bc_all)   # (Ntot,9)
            for i in range(3):
                for j in range(3):
                    idx = 3 * i + j
                    fig, ax = plt.subplots(figsize=(5, 5))
                    parity_plot(ax, D[:, idx], M[:, idx], f"Born charge Z*({i},{j})")
                    pdf.savefig(fig); plt.close(fig)

    print(f"✅ パリティプロットPDFを保存しました: {out_pdf}")

# ==========
# メイン
# ==========
def main():
    print("--- テストスクリプト開始 ---")
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")

    # Config 読込（カスタムタグ無視）
    cfg = load_config_safely(CONFIG_PATH)

    # モデルロード
    print("モデル構築・重み読込中...")
    model = load_model_from_config(cfg, MODEL_PATH, device)
    print("✅ モデル準備完了")

    # 予測 → XYZ 保存
    predict_and_save_xyz(model, cfg, TEST_XYZ, ML_XYZ, device=device)

    # パリティ PDF
    make_parity_pdf(TEST_XYZ, ML_XYZ, OUT_PDF)

    print("--- 全処理完了 ---")

if __name__ == "__main__":
    main()
