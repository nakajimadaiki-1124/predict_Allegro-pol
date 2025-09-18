#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目的:
- KeyError: 'polarization' の原因切り分け
- extxyz として読めているか？
- atoms.info / atoms.arrays のどちらに入っているか？
- コメント行(2行目)からラベルを直接抽出できるか？

実行:
  (venv) python probe_info_labels.py
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import ase
from ase.io import read

# >>> ここだけ調整 <<<
INPUT_XYZ = Path("data/BaTiO3.xyz")

# ASE 版によって場所が違う: ase.io.formats.filetype を使う
def detect_filetype(path: Path) -> str | None:
    try:
        from ase.io.formats import filetype  # 新しめ
        return filetype(str(path))
    except Exception:
        return None  # 無くても致命ではない

NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

def read_comment_labels(xyz_path: Path, max_frames: int = 3):
    print("\n[1] コメント行からのラベル抽出（先頭最大3フレーム）")
    if not xyz_path.exists():
        print("  × ファイルが見つかりません:", xyz_path)
        return
    with open(xyz_path, "r") as f:
        frame = 0
        while frame < max_frames:
            nline = f.readline()
            if not nline:
                break
            try:
                natoms = int(nline.strip())
            except Exception:
                print("  ! 先頭行が整数(原子数)ではありません:", nline.strip()[:80])
                break
            comment = f.readline() or ""
            print(f"  - frame {frame} comment line:\n    {comment.strip()}")
            # 正規表現で抽出
            m_e = re.search(rf"\btotal_energy\s*=\s*({NUM})", comment)
            m_p = re.search(rf'\bpolarization\s*=\s*"({NUM}\s+{NUM}\s+{NUM})"', comment)
            m_a = re.search(rf'\bpolarizability\s*=\s*"(({NUM}\s+){{8}}{NUM})"', comment)

            e = float(m_e.group(1)) if m_e else None
            p = np.fromstring(m_p.group(1), sep=" ") if m_p else None
            a = np.fromstring(m_a.group(1), sep=" ") if m_a else None

            print(f"    -> parsed total_energy: {e}")
            print(f"    -> parsed polarization: {p if p is None else p.tolist()}")
            print(f"    -> parsed polarizability (len): {None if a is None else a.size}")

            # 原子行を読み飛ばし
            for _ in range(natoms):
                if not f.readline():
                    break
            frame += 1
        if frame == 0:
            print("  × フレームが読み出せませんでした（ファイル形式を確認）")

def dump_ase_read(xyz_path: Path, force_extxyz: bool = False, max_frames: int = 3):
    print(f"\n[2] ASE read での info/arrays の中身 (force_extxyz={force_extxyz})")
    if not xyz_path.exists():
        print("  × ファイルが見つかりません:", xyz_path)
        return
    try:
        frames = read(str(xyz_path), ":" if max_frames > 1 else 0,
                      format=("extxyz" if force_extxyz else None))
    except Exception as e:
        print("  × ASE read で例外:", repr(e))
        return

    if not isinstance(frames, list):
        frames = [frames]
    if not frames:
        print("  × フレームが0件です")
        return

    print(f"  -> 読み込んだフレーム数: {len(frames)} (表示は最初の{min(max_frames, len(frames))}件)")
    for i, at in enumerate(frames[:max_frames]):
        print(f"  --- frame {i} ---")
        info_keys = sorted(list(at.info.keys()))
        arr_keys  = sorted(list(at.arrays.keys()))
        print("    info.keys():", info_keys)
        for key in ("total_energy", "polarization", "polarizability"):
            if key in at.info:
                val = at.info[key]
                # 値が長すぎると見にくいので一部だけ
                s = str(val)
                if len(s) > 120: s = s[:117] + "..."
                print(f"    info[{key!r}] -> type={type(val).__name__}, value={s}")
            else:
                print(f"    info[{key!r}] -> (なし)")
        print("    arrays.keys():", arr_keys)
        for key in ("polarization", "polarizability"):
            if key in at.arrays:
                arr = at.arrays[key]
                print(f"    arrays[{key!r}] -> shape={arr.shape}, dtype={arr.dtype}")

def main():
    print("=== ENV ===")
    print("ASE ver:", ase.__version__)
    ftype = detect_filetype(INPUT_XYZ)
    print("filetype (ASE detection):", ftype)
    print("path exists:", INPUT_XYZ.exists())

    read_comment_labels(INPUT_XYZ, max_frames=3)
    dump_ase_read(INPUT_XYZ, force_extxyz=False, max_frames=3)
    dump_ase_read(INPUT_XYZ, force_extxyz=True,  max_frames=3)

if __name__ == "__main__":
    main()
