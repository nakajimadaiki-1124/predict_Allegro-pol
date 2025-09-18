#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, Any, Tuple

import yaml
import numpy as np
import torch
from torch import nn
from ase.io import read, write
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator

# Allegro-pol
import allegro_pol  # noqa: F401
from allegro_pol._keys import (
    POLARIZATION_KEY,
    POLARIZABILITY_KEY,
    BORN_CHARGE_KEY,
    EXTERNAL_ELECTRIC_FIELD_KEY,
)

# NequIP
from nequip.utils.config import Config
from nequip.model import model_from_config
from nequip.data import AtomicData, AtomicDataDict

TYPE_KEY = AtomicDataDict.ATOM_TYPE_KEY

# ====== パス ======
MODEL_PATH = Path("results/20250917/BaTiO3/best_model.pth")
TRAIN_YAML = Path("configs/BaTiO3.yaml")
INPUT_XYZ  = Path("data/BaTiO3_split/test_data.xyz")

OUTPUT_XYZ = Path("test_results/result_20250917/BaTiO3-ML.xyz")
OUTPUT_PDF = Path("test_results/result_20250917/BaTiO3-parity.pdf")
# =================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- YAML ---
def yaml_load_loose(path: Path) -> Dict[str, Any]:
    class _IgnoreUnknown(yaml.SafeLoader): pass
    def _ignore(loader, tag_suffix, node):
        if isinstance(node, yaml.ScalarNode):   return loader.construct_scalar(node)
        if isinstance(node, yaml.SequenceNode): return loader.construct_sequence(node)
        if isinstance(node, yaml.MappingNode):  return loader.construct_mapping(node)
        return None
    _IgnoreUnknown.add_multi_constructor("", _ignore)
    with open(path, "r") as f:
        return yaml.load(f, Loader=_IgnoreUnknown) or {}

# --- ckpt ロード ---
def load_state_dict(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "model_state_dict", "state"):
            sd = ckpt.get(k)
            if isinstance(sd, dict):
                return sd
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise RuntimeError(f"未知のチェックポイント形式: {checkpoint_path}")

# --- モデル構築 ---
def build_model(yaml_path: Path, weights_path: Path) -> nn.Module:
    cfg_dict = yaml_load_loose(yaml_path)
    model_dtype = str(cfg_dict.get("model_dtype", "float32")).lower()
    if model_dtype in ("float64", "double"):
        torch.set_default_dtype(torch.float64); torch_dtype = torch.float64
    else:
        torch.set_default_dtype(torch.float32); torch_dtype = torch.float32
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass

    cfg = Config.from_dict(cfg_dict)
    cfg["initialize"] = False
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
    if missing:   print("[warn] missing keys (ignored):", missing)
    if unexpected:print("[warn] unexpected keys (ignored):", unexpected)

    model.to(device=DEVICE, dtype=torch_dtype).eval()
    return model

# --- 記号→タイプID ---
def build_sym2type(cfg_dict: Dict[str, Any]) -> Dict[str, int]:
    c2t = cfg_dict.get("chemical_symbol_to_type")
    if not isinstance(c2t, dict):
        raise KeyError("config.yaml に chemical_symbol_to_type がありません。")
    return {str(sym): int(t) for sym, t in c2t.items()}

# --- ASE → AtomicData ---
def atoms_to_atomicdata_with_type(atoms, r_max: float, sym2type: Dict[str, int]) -> AtomicData:
    data = AtomicData.from_ase(atoms, r_max=r_max)

    Z_key = AtomicDataDict.ATOMIC_NUMBERS_KEY
    try:
        Z = data[Z_key]
    except Exception:
        Z = None
    if Z is None or not isinstance(Z, torch.Tensor):
        Z = torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    else:
        if Z.ndim == 2 and Z.shape[1] == 1:
            Z = Z.squeeze(1).contiguous()
        Z = Z.to(dtype=torch.long)
    data[Z_key] = Z

    types = torch.tensor([sym2type[s] for s in atoms.get_chemical_symbols()], dtype=torch.long)
    data[TYPE_KEY] = types

    if EXTERNAL_ELECTRIC_FIELD_KEY not in data:
        if AtomicDataDict.BATCH_PTR_KEY in data:
            ptr = data[AtomicDataDict.BATCH_PTR_KEY]
            num_batch = int(ptr.shape[0] - 1)
        elif AtomicDataDict.BATCH_KEY in data:
            batch = data[AtomicDataDict.BATCH_KEY]
            num_batch = int(batch.max().item() + 1) if batch.numel() > 0 else 1
        else:
            num_batch = 1
        elec_field = torch.zeros(
            (num_batch, 3),
            dtype=data[AtomicDataDict.POSITIONS_KEY].dtype,
        )
        data[EXTERNAL_ELECTRIC_FIELD_KEY] = elec_field
    return data

# --- 推論 ---
def predict_one(model: nn.Module, atoms, r_max: float, sym2type: Dict[str, int]) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    data = atoms_to_atomicdata_with_type(atoms, r_max=r_max, sym2type=sym2type)
    data = data.to(device=DEVICE)
    data_dict = data.to_dict()
    out = model(data_dict)  # type: ignore

    if "total_energy" in out:
        e = float(out["total_energy"].detach().cpu().item())
    elif "energy" in out:
        e = float(out["energy"].detach().cpu().item())
    else:
        raise KeyError("モデル出力に 'total_energy' が見つかりません。")

    if POLARIZATION_KEY not in out:
        for alt in ("P", "dipole", "electric_polarization"):
            if alt in out: pol_key = alt; break
        else: raise KeyError("polarization が見つかりません。")
    else: pol_key = POLARIZATION_KEY

    if POLARIZABILITY_KEY not in out:
        for alt in ("alpha", "dielectric_polarizability"):
            if alt in out: polz_key = alt; break
        else: raise KeyError("polarizability が見つかりません。")
    else: polz_key = POLARIZABILITY_KEY

    if BORN_CHARGE_KEY not in out:
        for alt in ("Zb", "borncharges"):
            if alt in out: born_key = alt; break
        else: raise KeyError("born_charge が見つかりません。")
    else: born_key = BORN_CHARGE_KEY

    p = out[pol_key].detach().cpu().numpy().reshape(3)
    a = out[polz_key].detach().cpu().numpy().reshape(3, 3)
    born = out[born_key].detach().cpu().numpy().reshape(len(atoms), 3, 3)
    return e, p, a, born

# --- ラベル抽出 ---
_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
def extract_labels_from_comment(xyz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    energies, pols, polzs, borns = [], [], [], []
    with open(xyz_path, "r") as f:
        while True:
            nline = f.readline()
            if not nline: break
            try: natoms = int(nline.strip())
            except Exception: break
            comment = f.readline() or ""

            m_e = re.search(rf"\btotal_energy\s*=\s*({_NUM})", comment)
            m_p = re.search(rf'\bpolarization\s*=\s*"({_NUM}\s+{_NUM}\s+{_NUM})"', comment)
            m_a = re.search(rf'\bpolarizability\s*=\s*"(({_NUM}\s+){{8}}{_NUM})"', comment)
            if not (m_e and m_p and m_a):
                raise KeyError("コメント行からラベル抽出失敗。")

            e = float(m_e.group(1))
            p = np.fromstring(m_p.group(1), sep=" ", dtype=float)
            a = np.fromstring(m_a.group(1), sep=" ", dtype=float).reshape(3, 3)

            frame_born = []
            for _ in range(natoms):
                line = f.readline()
                parts = line.split()
                born_vals = np.fromiter((float(x) for x in parts[4:13]), dtype=float, count=9)
                frame_born.append(born_vals.reshape(3, 3))

            energies.append(e)
            pols.append(p)
            polzs.append(a)
            borns.append(np.stack(frame_born))
    return (
        np.asarray(energies, float),
        np.vstack(pols),
        np.stack(polzs),
        np.stack(borns),
    )

# --- polarization ブランチ補正 ---
def wrap_polarization(pred_p: np.ndarray, ref_p: np.ndarray, atoms) -> np.ndarray:
    cell = atoms.cell
    vol = cell.volume
    P_quanta = np.array([np.linalg.norm(cell[i]) / vol for i in range(3)])
    corrected = np.copy(pred_p)
    for i in range(3):
        dq = P_quanta[i]
        if dq > 1e-8:
            delta = corrected[i] - ref_p[i]
            shift = np.round(delta / dq)
            corrected[i] -= shift * dq
    return corrected

# --- パリティプロット ---
# --- パリティプロット ---
def parity_pdf(e_true, e_pred, p_true, p_pred, a_true, a_pred, z_true, z_pred, out_pdf: Path):
    def _flatten(arr): return np.asarray(arr, dtype=float).reshape(-1)

    def _compute_r2(y_true, y_pred):
        residual = y_true - y_pred
        ss_res = float(np.sum(residual ** 2))
        mean_true = float(np.mean(y_true))
        ss_tot = float(np.sum((y_true - mean_true) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    def _compute_rmse(y_true, y_pred):
        residual = y_true - y_pred
        return float(np.sqrt(np.mean(residual ** 2)))

    def _parity_plot(true, pred, xlabel, ylabel, title, pdf):
        true_flat, pred_flat = _flatten(true), _flatten(pred)
        fig, ax = plt.subplots(figsize=(6, 6))

        # parity line
        ax.plot([true_flat.min(), true_flat.max()],
                [true_flat.min(), true_flat.max()], "r-", lw=2)

        # scatter
        ax.scatter(true_flat, pred_flat, s=15, alpha=0.6, color="tab:blue")

        # labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")

        # metrics
        r2 = _compute_r2(true_flat, pred_flat)
        rmse = _compute_rmse(true_flat, pred_flat)
        textstr = f"$R^2 = {r2:.3f}$\nRMSE = {rmse:.3f}"

        ax.text(0.98, 0.02, textstr,
                transform=ax.transAxes,
                ha="right", va="bottom",
                bbox=dict(facecolor="wheat", alpha=0.5))

        pdf.savefig(fig)
        plt.close(fig)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(out_pdf)) as pdf:
        _parity_plot(e_true, e_pred, "DFT Energy", "ML Energy", "Parity: Energy", pdf)
        _parity_plot(p_true, p_pred, "DFT Polarization", "ML Polarization", "Parity: Polarization", pdf)
        _parity_plot(a_true, a_pred, "DFT Polarizability", "ML Polarizability", "Parity: Polarizability", pdf)
        _parity_plot(z_true, z_pred, "DFT Born charge", "ML Born charge", "Parity: Born charge", pdf)


# --- メイン ---
def main():
    cfg = yaml_load_loose(TRAIN_YAML)
    r_max = float(cfg["r_max"])
    sym2type = build_sym2type(cfg)
    model = build_model(TRAIN_YAML, MODEL_PATH)

    frames = read(str(INPUT_XYZ), ":")
    e_true, p_true, a_true, z_true = extract_labels_from_comment(INPUT_XYZ)

    e_pred, p_pred, a_pred, z_pred = [], [], [], []
    for atoms, ref_p in zip(frames, p_true):
        e_p, p_p, a_p, z_p = predict_one(model, atoms, r_max, sym2type)
        # polarization 補正
        p_p_corr = wrap_polarization(p_p, ref_p, atoms)

        atoms.info["predicted_total_energy"]   = float(e_p)
        atoms.info["predicted_polarization"]   = p_p_corr.tolist()
        atoms.info["predicted_polarizability"] = a_p.reshape(-1).tolist()
        atoms.arrays["predicted_born_charge"]  = z_p.reshape(len(atoms), -1)

        e_pred.append(e_p); p_pred.append(p_p_corr); a_pred.append(a_p); z_pred.append(z_p)

    OUTPUT_XYZ.parent.mkdir(parents=True, exist_ok=True)
    write(str(OUTPUT_XYZ), frames)

    parity_pdf(
        e_true, np.asarray(e_pred),
        p_true, np.vstack(p_pred),
        a_true, np.stack(a_pred),
        z_true, np.stack(z_pred),
        OUTPUT_PDF
    )

    print("=== Allegro-pol evaluation finished (branch-corrected polarization) ===")
    print(f"Saved XYZ : {OUTPUT_XYZ}")
    print(f"Saved PDF : {OUTPUT_PDF}")
    print(f"Frames    : {len(frames)}")
    print(f"Device    : {DEVICE}")

if __name__ == "__main__":
    main()
