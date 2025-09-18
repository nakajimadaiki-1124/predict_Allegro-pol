#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml
import numpy as np
import torch
from torch import nn
from ase.io import read, write
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Allegro-pol のビルダーを登録（重要）
import allegro_pol  # noqa: F401

# NequIP / Allegro
from nequip.utils.config import Config
from nequip.model import model_from_config
from nequip.data import AtomicData
from nequip.data.transforms import TypeMapper


# ========= パスだけあなたの環境に合わせて =========
MODEL_PATH = Path("results/20250908/BaTiO3/best_model.pth")
TRAIN_YAML = Path("configs/BaTiO3.yaml")
INPUT_XYZ  = Path("data/BaTiO3.xyz")

OUTPUT_XYZ = Path("test_results/BaTiO3-ML.xyz")
OUTPUT_PDF = Path("test_results/BaTiO3-parity.pdf")
# ================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- YAML（カスタムタグ無視） ----------
def yaml_load_loose(path: Path) -> Dict[str, Any]:
    class _IgnoreUnknown(yaml.SafeLoader):
        pass
    def _ignore(loader, tag_suffix, node):
        if isinstance(node, yaml.ScalarNode):
            return loader.construct_scalar(node)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node)
        return None
    _IgnoreUnknown.add_multi_constructor("", _ignore)
    with open(path, "r") as f:
        data = yaml.load(f, Loader=_IgnoreUnknown)
    return data or {}


# ---------- best_model.pth の state_dict 取り出し ----------
def load_state_dict(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "model_state_dict", "state"):
            sd = ckpt.get(key)
            if isinstance(sd, dict):
                return sd
        if all(hasattr(v, "shape") for v in ckpt.values()):
            return ckpt  # テンソル辞書としてそのまま
    raise RuntimeError(
        f"未知のチェックポイント形式: {checkpoint_path}\n"
        f"トップレベル: {type(ckpt)} / {list(ckpt.keys()) if isinstance(ckpt, dict) else 'N/A'}"
    )


# ---------- モデル構築（YAML→ネットワーク、重みロード） ----------
def build_model(yaml_path: Path, weights_path: Path) -> nn.Module:
    cfg_dict = yaml_load_loose(yaml_path)

    # dtype は YAML に合わせる（あなたのyamlは model_dtype: float32）
    model_dtype = str(cfg_dict.get("model_dtype", "float32")).lower()
    if model_dtype in ("float64", "double"):
        torch.set_default_dtype(torch.float64)
        torch_dtype = torch.float64
    else:
        torch.set_default_dtype(torch.float32)
        torch_dtype = torch.float32
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    cfg = Config.from_dict(cfg_dict)
    cfg["initialize"] = False  # 推論のみ
    model = model_from_config(cfg, initialize=False, dataset=None, deploy=False)

    sd = load_state_dict(weights_path)
    model_keys = set(model.state_dict().keys())

    def try_strip(d: Dict[str, torch.Tensor], pref: str) -> Dict[str, torch.Tensor]:
        if all(k.startswith(pref) for k in d.keys()):
            return {k[len(pref):]: v for k, v in d.items()}
        return d

    for pref in ("model.", "module.", ""):
        trial = try_strip(sd, pref)
        if set(trial.keys()).issubset(model_keys):
            sd = trial
            break

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print("[warn] missing keys (ignored):", missing)
    if unexpected:
        print("[warn] unexpected keys (ignored):", unexpected)

    model.to(device=DEVICE, dtype=torch_dtype).eval()
    return model


# ---------- TypeMapper（chemical_symbol_to_type） ----------
def build_type_mapper(cfg_dict: Dict[str, Any]) -> TypeMapper:
    c2t = cfg_dict.get("chemical_symbol_to_type")
    if not isinstance(c2t, dict):
        raise KeyError("config.yaml に chemical_symbol_to_type がありません。")
    max_t = max(int(v) for v in c2t.values())
    type_names = [""] * (max_t + 1)
    for sym, t in c2t.items():
        type_names[int(t)] = sym
    return TypeMapper(type_names=type_names)


# ---------- ASE Atoms → AtomicData ----------
def atoms_to_atomicdata(atoms, r_max: float, type_mapper: TypeMapper) -> AtomicData:
    data = AtomicData.from_ase(atoms, r_max=r_max)
    data = type_mapper.transform(data)
    return data


# ---------- 推論（1フレーム） ----------
def predict_one(model: nn.Module, atoms, r_max: float, tm: TypeMapper) -> Tuple[float, np.ndarray, np.ndarray]:
    data = atoms_to_atomicdata(atoms, r_max=r_max, type_mapper=tm).to(device=DEVICE)
    with torch.no_grad():
        out = model(data)  # type: ignore

    # energy
    if "total_energy" in out:
        e = float(out["total_energy"].detach().cpu().item())
    elif "energy" in out:
        e = float(out["energy"].detach().cpu().item())
    else:
        raise KeyError("モデル出力に 'total_energy' も 'energy' も見つかりません。")

    # polarization
    pol_key = "polarization"
    if pol_key not in out:
        for alt in ("P", "dipole", "electric_polarization"):
            if alt in out:
                pol_key = alt
                break
        else:
            raise KeyError("モデル出力に 'polarization' が見つかりません。")

    # polarizability
    polz_key = "polarizability"
    if polz_key not in out:
        for alt in ("alpha", "dielectric_polarizability"):
            if alt in out:
                polz_key = alt
                break
        else:
            raise KeyError("モデル出力に 'polarizability' が見つかりません。")

    p = out[pol_key].detach().cpu().numpy().reshape(3)
    a = out[polz_key].detach().cpu().numpy().reshape(3, 3)
    return e, p, a


# ---------- フレーム見出し行から真値を抜く（頑丈なフォールバック） ----------
_NUM_RE = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

def _extract_labels_from_extxyz(xyz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    各フレームの2行目（コメント行）から
      total_energy=..., polarization="x y z", polarizability="9成分"
    を正規表現で抜き出して numpy 配列で返す。
    """
    energies: List[float] = []
    pols: List[np.ndarray] = []
    polzs: List[np.ndarray] = []

    with open(xyz_path, "r") as f:
        while True:
            na_line = f.readline()
            if not na_line:
                break  # EOF
            # atoms数行をスキップ
            try:
                n_atoms = int(na_line.strip())
            except Exception:
                # 変則ファイルなら抜ける
                break

            comment = f.readline()
            if not comment:
                break

            # energy
            m_e = re.search(rf"\btotal_energy\s*=\s*({_NUM_RE})", comment)
            if not m_e:
                raise KeyError("コメント行に total_energy= が見つかりません。")
            e = float(m_e.group(1))

            # polarization "x y y"
            m_p = re.search(rf'\bpolarization\s*=\s*"({_NUM_RE}\s+{_NUM_RE}\s+{_NUM_RE})"', comment)
            if not m_p:
                raise KeyError('コメント行に polarization="x y z" が見つかりません。')
            p = np.fromstring(m_p.group(1), sep=" ", dtype=float)
            if p.size != 3:
                raise ValueError("polarization の成分数が3ではありません。")

            # polarizability "9 numbers"
            m_a = re.search(
                rf'\bpolarizability\s*=\s*"(({_NUM_RE}\s+){{8}}{_NUM_RE})"', comment
            )
            if not m_a:
                raise KeyError('コメント行に polarizability="9成分" が見つかりません。')
            a = np.fromstring(m_a.group(1), sep=" ", dtype=float)
            if a.size != 9:
                raise ValueError("polarizability の成分数が9ではありません。")

            energies.append(e)
            pols.append(p)
            polzs.append(a.reshape(3, 3))

            # 原子行を読み飛ばす
            for _ in range(n_atoms):
                if not f.readline():
                    break

    if not energies:
        raise RuntimeError("XYZからフレームが抽出できませんでした。")

    return (
        np.array(energies, dtype=float),
        np.vstack(pols).astype(float),
        np.stack(polzs).astype(float),
    )


# ---------- パリティPDF ----------
def parity_pdf(e_true, e_pred, p_true, p_pred, a_true, a_pred, out_pdf: Path):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(out_pdf)) as pdf:
        # Energy
        fig = plt.figure()
        plt.scatter(e_true, e_pred, alpha=0.7)
        lo, hi = float(min(e_true.min(), e_pred.min())), float(max(e_true.max(), e_pred.max()))
        plt.plot([lo, hi], [lo, hi], linestyle="--")
        plt.xlabel("DFT total_energy (eV)")
        plt.ylabel("Predicted total_energy (eV)")
        plt.title("Parity: Total Energy")
        pdf.savefig(fig); plt.close(fig)

        # Polarization (components)
        fig = plt.figure()
        plt.scatter(p_true.reshape(-1), p_pred.reshape(-1), alpha=0.7)
        lo, hi = float(min(p_true.min(), p_pred.min())), float(max(p_true.max(), p_pred.max()))
        plt.plot([lo, hi], [lo, hi], linestyle="--")
        plt.xlabel("DFT polarization components")
        plt.ylabel("Predicted polarization components")
        plt.title("Parity: Polarization (per-component)")
        pdf.savefig(fig); plt.close(fig)

        # Polarizability (components)
        fig = plt.figure()
        a_t = a_true.reshape(len(a_true), -1); a_p = a_pred.reshape(len(a_pred), -1)
        plt.scatter(a_t.reshape(-1), a_p.reshape(-1), alpha=0.7)
        lo, hi = float(min(a_t.min(), a_p.min())), float(max(a_t.max(), a_p.max()))
        plt.plot([lo, hi], [lo, hi], linestyle="--")
        plt.xlabel("DFT polarizability components")
        plt.ylabel("Predicted polarizability components")
        plt.title("Parity: Polarizability (per-component)")
        pdf.savefig(fig); plt.close(fig)


# ---------- メイン ----------
def main():
    # YAML
    cfg = yaml_load_loose(TRAIN_YAML)
    if "r_max" not in cfg:
        raise KeyError("config.yaml に r_max がありません。")
    r_max = float(cfg["r_max"])
    tm = build_type_mapper(cfg)

    # モデル（デプロイ無し）
    model = build_model(TRAIN_YAML, MODEL_PATH)

    # 入力（構造本体は ASE で、真値は独自パーサで拾う）
    frames = read(str(INPUT_XYZ), ":")
    e_true, p_true, a_true = _extract_labels_from_extxyz(INPUT_XYZ)

    if len(frames) != len(e_true):
      raise RuntimeError(f"フレーム数不一致: ASE={len(frames)} / labels={len(e_true)}")

    # 推論
    e_pred, p_pred, a_pred = [], [], []
    for atoms in frames:
        e_p, p_p, a_p = predict_one(model, atoms, r_max, tm)
        # 予測を info に保存
        atoms.info["predicted_total_energy"] = float(e_p)
        atoms.info["predicted_polarization"] = p_p.astype(float).tolist()
        atoms.info["predicted_polarizability"] = a_p.reshape(-1).astype(float).tolist()

        e_pred.append(e_p); p_pred.append(p_p); a_pred.append(a_p)

    # 予測付き XYZ
    OUTPUT_XYZ.parent.mkdir(parents=True, exist_ok=True)
    write(str(OUTPUT_XYZ), frames)

    # パリティPDF
    parity_pdf(
        e_true=e_true,
        e_pred=np.array(e_pred),
        p_true=p_true,
        p_pred=np.vstack(p_pred),
        a_true=a_true,
        a_pred=np.stack(a_pred),
        out_pdf=OUTPUT_PDF,
    )

    print("=== Allegro-pol direct evaluation finished (no deploy) ===")
    print(f"Saved XYZ : {OUTPUT_XYZ}")
    print(f"Saved PDF : {OUTPUT_PDF}")
    print(f"Frames    : {len(frames)}")
    print(f"Device    : {DEVICE}")


if __name__ == "__main__":
    main()
