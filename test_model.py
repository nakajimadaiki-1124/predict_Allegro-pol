#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run inference with a trained Allegro/NequIP model and generate parity plots.

This script follows the agreed workflow:

1. Load a configuration YAML file while ignoring any custom Python tags.
2. Build the model via :func:`nequip.model.model_from_config` and load the provided
   weights.
3. Predict total energy, polarization, and polarizability for each frame of a test
   extxyz file.  The predictions are stored in ``Atoms.info`` with a
   ``predicted_`` prefix and written to a new extxyz file.
4. Generate a parity plot PDF comparing the DFT reference values with the
   predictions for all available quantities.

<<<<<<< ours
=======
The input/output paths are configured via module-level constants so the script
can be executed without command-line arguments.

>>>>>>> theirs
Force and stress outputs are intentionally ignored.
"""

from __future__ import annotations

<<<<<<< ours
import argparse
=======
>>>>>>> theirs
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from ase import Atoms
from ase.io import read, write
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

from nequip.data import AtomicData
from nequip.data.transforms import TypeMapper
from nequip.model import model_from_config
from nequip.utils import Config

# -----------------------------------------------------------------------------
<<<<<<< ours
# YAML loader that safely ignores unsupported Python-specific tags
# -----------------------------------------------------------------------------


class SafeIgnoreUnknownLoader(yaml.SafeLoader):
    """YAML loader that replaces unknown python tags with ``None``."""


=======
# Default in-repository paths (edit here to point to your files)
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = Path("configs/BaTiO3.yaml")
WEIGHTS_PATH = Path("results/20250908/BaTiO3/best_model.pth")
TEST_XYZ_PATH = Path("data/BaTiO3.xyz")
OUTPUT_DIR = Path("test_results")
DEVICE_SPEC = "auto"
LOG_LEVEL = "INFO"

# -----------------------------------------------------------------------------
# YAML loader that safely ignores unsupported Python-specific tags
# -----------------------------------------------------------------------------


class SafeIgnoreUnknownLoader(yaml.SafeLoader):
    """YAML loader that replaces unknown python tags with ``None``."""


>>>>>>> theirs
def _ignore_unknown_constructor(loader: SafeIgnoreUnknownLoader, tag_suffix: str, node: yaml.Node):
    del tag_suffix  # unused
    del node  # unused
    return None


SafeIgnoreUnknownLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/object:", _ignore_unknown_constructor
)
SafeIgnoreUnknownLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/tuple", _ignore_unknown_constructor
)

# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------


def load_config_safely(path: Path) -> Dict:
    """Load a NequIP/Allegro configuration YAML while ignoring custom tags."""

    with path.open("r") as handle:
        cfg = yaml.load(handle, Loader=SafeIgnoreUnknownLoader)

    if cfg is None:
        raise ValueError(f"Configuration file {path} is empty or invalid")

    if cfg.get("avg_num_neighbors") == "auto":
        cfg["avg_num_neighbors"] = None

    required_keys = ["r_max", "chemical_symbol_to_type"]
    missing = [key for key in required_keys if key not in cfg]
    if missing:
        raise KeyError(f"Missing required configuration entries: {', '.join(missing)}")

    if not isinstance(cfg["chemical_symbol_to_type"], dict):
        raise TypeError("chemical_symbol_to_type must be a mapping from symbols to types")

    return cfg


def resolve_device(device_arg: str) -> torch.device:
    """Resolve the requested device string."""

    if device_arg.lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


<<<<<<< ours
=======
def resolve_runtime_path(path: Path | str, base_dir: Path) -> Path:
    """Resolve a path specification relative to ``base_dir``."""

    spec = Path(path)
    expanded = spec.expanduser()
    if expanded.is_absolute():
        return expanded
    return (base_dir / expanded).resolve()


>>>>>>> theirs
# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


def load_model_from_config(cfg: Dict, weights_path: Path, device: torch.device) -> torch.nn.Module:
    """Build the model and load the provided weights."""

    config = Config.from_dict(cfg)
    model = model_from_config(config)

    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict):
        if "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        elif "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
            state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    logging.info("Model loaded from %s", weights_path)
    return model


def infer_model_dtype(model: torch.nn.Module) -> torch.dtype:
    """Infer the floating point dtype used by the model parameters."""

    for param in model.parameters():
        return param.dtype
    return torch.get_default_dtype()


# -----------------------------------------------------------------------------
# Inference and extxyz writing
# -----------------------------------------------------------------------------


def predict_and_save_xyz(
    model: torch.nn.Module,
    cfg: Dict,
    test_xyz: Path,
    out_xyz: Path,
    device: torch.device,
) -> int:
    """Run inference for each frame and save predictions to an extxyz file."""

    if not test_xyz.exists():
        raise FileNotFoundError(f"Test structure file not found: {test_xyz}")

    frames_raw = read(str(test_xyz), ":")
    frames: List[Atoms]
    if isinstance(frames_raw, list):
        frames = frames_raw
    else:
        frames = [frames_raw]

    if len(frames) == 0:
        logging.warning("No frames found in %s; skipping prediction", test_xyz)
        return 0

    mapper = TypeMapper(chemical_symbol_to_type=cfg["chemical_symbol_to_type"])
    r_max = cfg["r_max"]
    model_dtype = infer_model_dtype(model)

    target_specs = {
        "total_energy": ("predicted_total_energy", 1),
        "polarization": ("predicted_polarization", 3),
        "polarizability": ("predicted_polarizability", 9),
    }

    logged_output_keys = False
    unavailable_targets: Dict[str, bool] = {key: False for key in target_specs}
    successful_frames: List[Atoms] = []

    for index, frame in enumerate(tqdm(frames, desc="Predicting", unit="frame")):
        try:
            data = AtomicData.from_ase(frame, r_max=r_max, type_mapper=mapper)
            data = data.to(device=device, dtype=model_dtype)

            with torch.no_grad():
                outputs = model(data.to_dict())

            if not logged_output_keys:
                logged_output_keys = True
                available = sorted(outputs.keys())
                logging.info("Model output keys: %s", ", ".join(available))
                for quantity in target_specs:
                    if quantity not in outputs:
                        logging.warning(
                            "Model output missing '%s'; predictions for this quantity will be skipped.",
                            quantity,
                        )
                        unavailable_targets[quantity] = True

            predictions: Dict[str, Iterable[float]] = {}
            for quantity, (info_key, expected_length) in target_specs.items():
                if unavailable_targets.get(quantity, False):
                    continue
                if quantity not in outputs:
                    unavailable_targets[quantity] = True
                    continue

                value = outputs[quantity]
                if not torch.is_tensor(value):
                    logging.warning(
                        "Output '%s' is not a tensor (frame %d); skipping future predictions for this quantity.",
                        quantity,
                        index,
                    )
                    unavailable_targets[quantity] = True
                    continue

                value_cpu = value.detach().cpu().reshape(-1)
                if expected_length == 1:
                    predictions[info_key] = float(value_cpu[0])
                else:
                    if value_cpu.numel() != expected_length:
                        logging.warning(
                            "Output '%s' has length %d (expected %d); skipping future predictions for this quantity.",
                            quantity,
                            value_cpu.numel(),
                            expected_length,
                        )
                        unavailable_targets[quantity] = True
                        continue
                    predictions[info_key] = value_cpu.tolist()

            if predictions:
                frame.info.update(predictions)

            successful_frames.append(frame)
        except Exception as exc:  # pylint: disable=broad-except
            logging.warning("Frame %d skipped due to error: %s", index, exc)
            continue

    if len(successful_frames) == 0:
        logging.warning("No frames were successfully predicted; skipping write to %s", out_xyz)
        return 0

    out_xyz.parent.mkdir(parents=True, exist_ok=True)
    write(str(out_xyz), successful_frames, format="extxyz")
    logging.info("Wrote predictions for %d frames to %s", len(successful_frames), out_xyz)
    return len(successful_frames)


# -----------------------------------------------------------------------------
# Parity plot utilities
# -----------------------------------------------------------------------------


def _extract_series(frames: List[Atoms], key: str, length: int) -> Optional[np.ndarray]:
    values: List[np.ndarray] = []
    for idx, frame in enumerate(frames):
        if key not in frame.info:
            logging.warning("Frame %d missing key '%s'", idx, key)
            return None
        raw = frame.info[key]
        arr = np.asarray(raw, dtype=float).reshape(-1)
        if arr.size == 0:
            logging.warning("Frame %d key '%s' is empty", idx, key)
            return None
        if length == 1:
            values.append(np.array([float(arr[0])], dtype=float))
        else:
            if arr.size != length:
                logging.warning(
                    "Frame %d key '%s' has length %d (expected %d)",
                    idx,
                    key,
                    arr.size,
                    length,
                )
                return None
            values.append(arr.astype(float))

    if not values:
        return None

    stacked = np.vstack(values)
    if length == 1:
        return stacked.reshape(-1)
    return stacked


def _prepare_quantity(
    dft_frames: List[Atoms],
    ml_frames: List[Atoms],
    quantity: str,
    length: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    dft_key = quantity
    ml_key = f"predicted_{quantity}"

    dft_values = _extract_series(dft_frames, dft_key, length)
    if dft_values is None:
        logging.warning("Skipping '%s' parity plot: missing DFT data", quantity)
        return None

    ml_values = _extract_series(ml_frames, ml_key, length)
    if ml_values is None:
        logging.warning("Skipping '%s' parity plot: missing predictions", quantity)
        return None

    if dft_values.shape[0] != ml_values.shape[0]:
        logging.warning(
            "Skipping '%s' parity plot: frame count mismatch (%d vs %d)",
            quantity,
            dft_values.shape[0],
            ml_values.shape[0],
        )
        return None

    if dft_values.shape[0] == 0:
        logging.warning("Skipping '%s' parity plot: no frames available", quantity)
        return None
<<<<<<< ours

    return dft_values, ml_values


=======

    return dft_values, ml_values


>>>>>>> theirs
def parity_plot(ax: plt.Axes, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t = y_true[mask]
    y_p = y_pred[mask]

    if y_t.size == 0:
        ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        return

    mae = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))

    ax.scatter(y_t, y_p, s=20, alpha=0.7)
    lo = min(float(y_t.min()), float(y_p.min()))
    hi = max(float(y_t.max()), float(y_p.max()))
    if lo == hi:
        delta = 1.0 if lo == 0.0 else 0.05 * abs(lo)
        lo -= delta
        hi += delta
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("DFT")
    ax.set_ylabel("ML")
    ax.text(
        0.05,
        0.95,
        f"MAE = {mae:.4g}\nRMSE = {rmse:.4g}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )


def make_parity_pdf(dft_xyz: Path, ml_xyz: Path, out_pdf: Path) -> None:
    if not ml_xyz.exists():
        logging.warning("Predicted XYZ file %s does not exist; skipping parity plot", ml_xyz)
        return

    dft_raw = read(str(dft_xyz), ":")
    ml_raw = read(str(ml_xyz), ":")

    dft_frames = dft_raw if isinstance(dft_raw, list) else [dft_raw]
    ml_frames = ml_raw if isinstance(ml_raw, list) else [ml_raw]
<<<<<<< ours

    plot_jobs: List[Tuple[str, np.ndarray, np.ndarray]] = []

=======

    plot_jobs: List[Tuple[str, np.ndarray, np.ndarray]] = []

>>>>>>> theirs
    energy_data = _prepare_quantity(dft_frames, ml_frames, "total_energy", 1)
    if energy_data is not None:
        plot_jobs.append(("Total Energy", energy_data[0], energy_data[1]))

    polarization_data = _prepare_quantity(dft_frames, ml_frames, "polarization", 3)
    if polarization_data is not None:
        labels = ["Px", "Py", "Pz"]
        for comp, label in enumerate(labels):
            plot_jobs.append(
                (f"Polarization {label}", polarization_data[0][:, comp], polarization_data[1][:, comp])
            )

    polarizability_data = _prepare_quantity(dft_frames, ml_frames, "polarizability", 9)
    if polarizability_data is not None:
        for i in range(3):
            for j in range(3):
                idx = 3 * i + j
                plot_jobs.append(
                    (
                        f"Polarizability α({i},{j})",
                        polarizability_data[0][:, idx],
                        polarizability_data[1][:, idx],
                    )
                )

    if not plot_jobs:
        logging.warning("No parity plots generated; skipping PDF creation")
        return

    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(out_pdf)) as pdf:
        for title, y_true, y_pred in plot_jobs:
            fig, ax = plt.subplots(figsize=(5, 5))
            parity_plot(ax, np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float), title)
            pdf.savefig(fig)
            plt.close(fig)

    logging.info("Parity plot PDF written to %s", out_pdf)


# -----------------------------------------------------------------------------
<<<<<<< ours
# Command-line interface
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to the model configuration YAML")
    parser.add_argument("--weights", required=True, type=Path, help="Path to the trained model weights (.pth)")
    parser.add_argument("--test", required=True, type=Path, help="Path to the test extxyz file")
    parser.add_argument("--outdir", required=True, type=Path, help="Directory to store prediction results")
    parser.add_argument(
        "--device",
        default="auto",
        help="Computation device (e.g., 'cpu', 'cuda', 'cuda:0', or 'auto' for automatic selection)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g., DEBUG, INFO, WARNING)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    device = resolve_device(args.device)
    logging.info("Using device: %s", device)

    cfg = load_config_safely(args.config)
=======
def main() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    device = resolve_device(DEVICE_SPEC)
    logging.info("Using device: %s", device)

    config_path = resolve_runtime_path(CONFIG_PATH, BASE_DIR)
    logging.info("Configuration file: %s", config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    cfg = load_config_safely(config_path)
>>>>>>> theirs
    logging.info(
        "Loaded config: r_max=%s, avg_num_neighbors=%s, elements=%s",
        cfg.get("r_max"),
        cfg.get("avg_num_neighbors"),
        ",".join(sorted(cfg["chemical_symbol_to_type"].keys())),
    )

<<<<<<< ours
    model = load_model_from_config(cfg, args.weights, device)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    base_name = args.test.stem
    ml_xyz = outdir / f"{base_name}-ML.xyz"
    out_pdf = outdir / f"{base_name}-parity.pdf"

    num_frames = predict_and_save_xyz(model, cfg, args.test, ml_xyz, device)
=======
    weights_path = resolve_runtime_path(WEIGHTS_PATH, BASE_DIR)
    logging.info("Weights file: %s", weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    model = load_model_from_config(cfg, weights_path, device)

    outdir = resolve_runtime_path(OUTPUT_DIR, BASE_DIR)
    logging.info("Output directory: %s", outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    test_xyz = resolve_runtime_path(TEST_XYZ_PATH, BASE_DIR)
    logging.info("Test structures: %s", test_xyz)
    base_name = test_xyz.stem
    ml_xyz = outdir / f"{base_name}-ML.xyz"
    out_pdf = outdir / f"{base_name}-parity.pdf"

    num_frames = predict_and_save_xyz(model, cfg, test_xyz, ml_xyz, device)
>>>>>>> theirs
    if num_frames == 0:
        logging.warning("No frames predicted; parity plot will not be generated")
        return

<<<<<<< ours
    make_parity_pdf(args.test, ml_xyz, out_pdf)
=======
    make_parity_pdf(test_xyz, ml_xyz, out_pdf)
>>>>>>> theirs


if __name__ == "__main__":
    main()
