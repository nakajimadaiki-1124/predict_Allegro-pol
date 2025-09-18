#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, Any, Tuple, List

import yaml
import numpy as np
import torch
from torch import nn
from ase.io import read, write
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Allegro-pol をレジストリに登録
import allegro_pol  # noqa: F401

# NequIP
from nequip.utils.config import Config
from nequip.model import model_from_config
from nequip.data import AtomicData, AtomicDataDict

TYPE_KEY = AtomicDataDict.ATOM_TYPE_KEY

# ====== パス（ここだけ調整） ======
MODEL_PATH = Path("results/20250908/BaTiO3/best_model.pth")
TRAIN_YAML = Path("configs/BaTiO3.yaml")
INPUT_XYZ  = Path("data/BaTiO3.xyz")

OUTPUT_XYZ = Path("test_results/BaTiO3-ML.xyz")
OUTPUT_PDF = Path("test_results/BaTiO3-parity.pdf")
# =================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- YAML（カスタムタグ無視） ---
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

# --- ckpt ロード（形式差にロバスト） ---
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

# --- モデル構築＋重みロード ---
def build_model(yaml_path: Path, weights_path: Path) -> nn.Module:
    cfg_dict = yaml_load_loose(yaml_path)
    # dtype
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

# --- 記号→タイプID マップ ---
def build_sym2type(cfg_dict: Dict[str, Any]) -> Dict[str, int]:
    c2t = cfg_dict.get("chemical_symbol_to_type")
    if not isinstance(c2t, dict):
        raise KeyError("config.yaml に chemical_symbol_to_type がありません。")
    return {str(sym): int(t) for sym, t in c2t.items()}

# --- ASE → AtomicData（type を自前で付与、atomic_numbers も矯正） ---
def atoms_to_atomicdata_with_type(atoms, r_max: float, sym2type: Dict[str, int]) -> AtomicData:
    data = AtomicData.from_ase(atoms, r_max=r_max)

    # atomic_numbers を必ず LongTensor (N,) に矯正
    Z_key = AtomicDataDict.ATOMIC_NUMBERS_KEY  # "atomic_numbers"
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

    # ここを互換キーで
    types = torch.tensor([sym2type[s] for s in atoms.get_chemical_symbols()], dtype=torch.long)
    data[TYPE_KEY] = types
    return data

# --- 1フレーム推論 ---
def predict_one(model: nn.Module, atoms, r_max: float, sym2type: Dict[str, int]) -> Tuple[float, np.ndarray, np.ndarray]:
    data = atoms_to_atomicdata_with_type(atoms, r_max=r_max, sym2type=sym2type).to(device=DEVICE)
    with torch.no_grad():
        out = model(data)  # type: ignore

    # energy
    if "total_energy" in out:
        e = float(out["total_energy"].detach().cpu().item())
    elif "energy" in out:
        e = float(out["energy"].detach().cpu().item())
    else:
        raise KeyError("モデル出力に 'total_energy' も 'energy' も見つかりません。")

    # polarization / polarizability
    pol_key = "polarization"
    if pol_key not in out:
        for alt in ("P", "dipole", "electric_polarization"):
            if alt in out: pol_key = alt; break
        else:
            raise KeyError("モデル出力に 'polarization' が見つかりません。")

    polz_key = "polarizability"
    if polz_key not in out:
        for alt in ("alpha", "dielectric_polarizability"):
            if alt in out: polz_key = alt; break
        else:
            raise KeyError("モデル出力に 'polarizability' が見つかりません。")

    p = out[pol_key].detach().cpu().numpy().reshape(3)
    a = out[polz_key].detach().cpu().numpy().reshape(3, 3)
    return e, p, a

# --- コメント行から真値(energy, P, A)を抜く ---
_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
def extract_labels_from_comment(xyz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    energies, pols, polzs = [], [], []
    with open(xyz_path, "r") as f:
        while True:
            nline = f.readline()
            if not nline: break
            try:
                natoms = int(nline.strip())
            except Exception:
                break
            comment = f.readline() or ""

            m_e = re.search(rf"\btotal_energy\s*=\s*({_NUM})", comment)
            m_p = re.search(rf'\bpolarization\s*=\s*"({_NUM}\s+{_NUM}\s+{_NUM})"', comment)
            m_a = re.search(rf'\bpolarizability\s*=\s*"(({_NUM}\s+){{8}}{_NUM})"', comment)
            if not (m_e and m_p and m_a):
                raise KeyError("コメント行から total_energy / polarization / polarizability が抽出できませんでした。")

            e = float(m_e.group(1))
            p = np.fromstring(m_p.group(1), sep=" ", dtype=float)
            a = np.fromstring(m_a.group(1), sep=" ", dtype=float).reshape(3, 3)

            energies.append(e); pols.append(p); polzs.append(a)

            for _ in range(natoms):
                f.readline()
    if not energies:
        raise RuntimeError("XYZからフレームが抽出できませんでした。")
    return np.asarray(energies, float), np.vstack(pols), np.stack(polzs)

# --- パリティPDF ---
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

        # Polarization
        fig = plt.figure()
        plt.scatter(p_true.reshape(-1), p_pred.reshape(-1), alpha=0.7)
        lo, hi = float(min(p_true.min(), p_pred.min())), float(max(p_true.max(), p_pred.max()))
        plt.plot([lo, hi], [lo, hi], linestyle="--")
        plt.xlabel("DFT polarization components")
        plt.ylabel("Predicted polarization components")
        plt.title("Parity: Polarization (per-component)")
        pdf.savefig(fig); plt.close(fig)

        # Polarizability
        fig = plt.figure()
        a_t = a_true.reshape(len(a_true), -1); a_p = a_pred.reshape(len(a_pred), -1)
        plt.scatter(a_t.reshape(-1), a_p.reshape(-1), alpha=0.7)
        lo, hi = float(min(a_t.min(), a_p.min())), float(max(a_t.max(), a_p.max()))
        plt.plot([lo, hi], [lo, hi], linestyle="--")
        plt.xlabel("DFT polarizability components")
        plt.ylabel("Predicted polarizability components")
        plt.title("Parity: Polarizability (per-component)")
        pdf.savefig(fig); plt.close(fig)

# --- メイン ---
def main():
    cfg = yaml_load_loose(TRAIN_YAML)
    r_max = float(cfg["r_max"])
    sym2type = build_sym2type(cfg)

    model = build_model(TRAIN_YAML, MODEL_PATH)

    frames = read(str(INPUT_XYZ), ":")
    e_true, p_true, a_true = extract_labels_from_comment(INPUT_XYZ)
    if len(frames) != len(e_true):
        raise RuntimeError(f"フレーム数不一致: ASE={len(frames)} / labels={len(e_true)}")

    e_pred, p_pred, a_pred = [], [], []
    for atoms in frames:
        e_p, p_p, a_p = predict_one(model, atoms, r_max, sym2type)
        atoms.info["predicted_total_energy"]   = float(e_p)
        atoms.info["predicted_polarization"]   = p_p.tolist()
        atoms.info["predicted_polarizability"] = a_p.reshape(-1).tolist()
        e_pred.append(e_p); p_pred.append(p_p); a_pred.append(a_p)

    OUTPUT_XYZ.parent.mkdir(parents=True, exist_ok=True)
    write(str(OUTPUT_XYZ), frames)

    parity_pdf(
        e_true=e_true, e_pred=np.asarray(e_pred),
        p_true=p_true, p_pred=np.vstack(p_pred),
        a_true=a_true, a_pred=np.stack(a_pred),
        out_pdf=OUTPUT_PDF,
    )

    print("=== Allegro-pol evaluation finished (no deploy, no TypeMapper) ===")
    print(f"Saved XYZ : {OUTPUT_XYZ}")
    print(f"Saved PDF : {OUTPUT_PDF}")
    print(f"Frames    : {len(frames)}")
    print(f"Device    : {DEVICE}")

if __name__ == "__main__":
    main()
