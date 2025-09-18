#!/usr/bin/env python3
from pathlib import Path
import torch
from ase.io import read

from nequip.data import AtomicData, AtomicDataDict
from nequip.data.transforms import TypeMapper

# --- パスを合わせてください ---
TRAIN_YAML = Path("configs/BaTiO3.yaml")
INPUT_XYZ  = Path("data/BaTiO3.xyz")
# --------------------------------

def yaml_load_loose(path: Path):
    import yaml
    class _IgnoreUnknown(yaml.SafeLoader): pass
    def _ignore(loader, tag_suffix, node):
        if isinstance(node, yaml.ScalarNode):   return loader.construct_scalar(node)
        if isinstance(node, yaml.SequenceNode): return loader.construct_sequence(node)
        if isinstance(node, yaml.MappingNode):  return loader.construct_mapping(node)
        return None
    _IgnoreUnknown.add_multi_constructor("", _ignore)
    with open(path, "r") as f:
        return yaml.load(f, Loader=_IgnoreUnknown) or {}

def list_keys_safe(ad):
    # AtomicData.keys が属性/メソッドのどちらでも対応
    k = getattr(ad, "keys", None)
    if callable(k):
        try:
            return list(k())
        except Exception:
            pass
    if k is not None:
        try:
            return list(k)
        except Exception:
            pass
    # fallback: 代表的なキー候補で存在チェック
    candidates = [
        AtomicDataDict.ATOMIC_NUMBERS_KEY,
        AtomicDataDict.NODE_TYPE_KEY,
        AtomicDataDict.POSITIONS_KEY,
        "edge_index", "edge_cell_shift", "cell", "pbc",
        "forces", "polarization", "stress",
    ]
    return [c for c in candidates if key_in(ad, c)]

def key_in(ad, key: str) -> bool:
    try:
        return key in ad  # 多くの版で __contains__ 実装済み
    except Exception:
        # __contains__ 無い版向けに、getitem で判定
        try:
            _ = ad[key]
            return True
        except Exception:
            return False

def get_val(ad, key: str):
    try:
        return ad[key]
    except Exception:
        return None

def build_typemapper_from_yaml(cfg_dict):
    c2t = cfg_dict["chemical_symbol_to_type"]
    type_names = [""] * (max(int(v) for v in c2t.values()) + 1)
    for s, t in c2t.items():
        type_names[int(t)] = s
    return TypeMapper(type_names=type_names)

def main():
    print("=== Probe AtomicData / TypeMapper ===")
    cfg = yaml_load_loose(TRAIN_YAML)
    r_max = float(cfg["r_max"])
    tm = build_typemapper_from_yaml(cfg)

    atoms = read(str(INPUT_XYZ), index=0)
    data: AtomicData = AtomicData.from_ase(atoms, r_max=r_max)

    print("Before transform:")
    keys0 = list_keys_safe(data)
    print("  keys:", keys0)

    Z_key = AtomicDataDict.ATOMIC_NUMBERS_KEY  # "atomic_numbers"
    Z = get_val(data, Z_key)
    print("  atomic_numbers present:", key_in(data, Z_key))
    if Z is not None:
        print("  atomic_numbers type :", type(Z).__name__)
        if isinstance(Z, torch.Tensor):
            print("  atomic_numbers dtype:", Z.dtype, "| shape:", tuple(Z.shape))
        else:
            print("  atomic_numbers repr :", repr(Z))

    # 壊れていてもいなくても、ここで LongTensor に矯正
    need_fix = (Z is None) or (not isinstance(Z, torch.Tensor)) or (Z.dtype != torch.long)
    if need_fix:
        print("  -> Fixing atomic_numbers to LongTensor from ASE…")
        z = torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        data[Z_key] = z
        Z = get_val(data, Z_key)
        print("  after fix: type:", type(Z).__name__, "dtype:", Z.dtype, "shape:", tuple(Z.shape))

    print("\nCalling TypeMapper.transform …")
    try:
        data2 = tm.transform(data)
        print("  transform OK")
        keys1 = list_keys_safe(data2)
        print("  after keys:", keys1)
        t_key = AtomicDataDict.NODE_TYPE_KEY  # "type"
        T = get_val(data2, t_key)
        print("  node type present:", key_in(data2, t_key))
        if isinstance(T, torch.Tensor):
            print("  node type dtype:", T.dtype, "| shape:", tuple(T.shape))
    except Exception as e:
        print("  transform FAILED:", repr(e))

if __name__ == "__main__":
    main()
