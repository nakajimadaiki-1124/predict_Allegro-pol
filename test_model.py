import copy
import importlib
import re
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def inference_module():
    """Import the inference script with lightweight stubs for heavy deps."""

    stubs: dict[str, object | None] = {}

    def install_stub(name: str, module: types.ModuleType) -> types.ModuleType:
        stubs[name] = sys.modules.get(name)
        sys.modules[name] = module
        return module

    # numpy stub -------------------------------------------------------------
    numpy_module = types.ModuleType("numpy")

    def _to_nested(value):
        if isinstance(value, FakeArray):
            return copy.deepcopy(value._data)
        if isinstance(value, (list, tuple)):
            return [_to_nested(item) for item in value]
        return float(value)

    def _copy_nested(value):
        if isinstance(value, list):
            return [_copy_nested(item) for item in value]
        return float(value)

    def _flatten(value):
        if isinstance(value, list):
            for item in value:
                yield from _flatten(item)
        else:
            yield float(value)

    def _apply_indices(data, keys):
        if not keys:
            return data
        key, *rest = keys
        if isinstance(key, slice):
            selection = data[key]
            return [_apply_indices(item, rest) for item in selection]
        if isinstance(key, int):
            return _apply_indices(data[key], rest)
        raise TypeError("unsupported index type")

    def _resolve_shape(shape, total_size):
        shape_list = list(shape)
        unknown_idx = None
        known_product = 1
        for idx, dim in enumerate(shape_list):
            if dim == -1:
                if unknown_idx is not None:
                    raise ValueError("only one unknown dimension allowed")
                unknown_idx = idx
            else:
                known_product *= int(dim)
        if unknown_idx is not None:
            if known_product == 0 or total_size % known_product != 0:
                raise ValueError("cannot infer shape")
            shape_list[unknown_idx] = total_size // known_product
        product = 1
        for dim in shape_list:
            product *= int(dim)
        if product != total_size:
            raise ValueError("shape mismatch")
        return tuple(int(dim) for dim in shape_list)

    def _reshape_from_flat(flat, shape):
        if len(shape) == 1:
            return [float(value) for value in flat]
        size = shape[0]
        step = len(flat) // size
        result = []
        for idx in range(size):
            start = idx * step
            end = (idx + 1) * step
            result.append(_reshape_from_flat(flat[start:end], shape[1:]))
        return result

    class FakeArray:
        def __init__(self, data):
            self._data = _to_nested(data)

        def reshape(self, *shape):
            if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
                shape = tuple(shape[0])
            shape = tuple(shape)
            flat = list(_flatten(self._data))
            if shape == (-1,):
                return FakeArray(flat)
            resolved = _resolve_shape(shape, len(flat))
            return FakeArray(_reshape_from_flat(flat, resolved))

        def min(self):
            return min(_flatten(self._data))

        def max(self):
            return max(_flatten(self._data))

        def tolist(self):
            return _copy_nested(self._data)

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

        def __getitem__(self, key):
            if isinstance(key, tuple):
                result = _apply_indices(self._data, list(key))
            else:
                result = _apply_indices(self._data, [key])
            if isinstance(result, list):
                return FakeArray(result)
            return float(result)

        def __float__(self):
            flat = list(_flatten(self._data))
            if len(flat) != 1:
                raise TypeError("cannot convert array with more than one element to float")
            return float(flat[0])

        @property
        def size(self) -> int:
            return len(list(_flatten(self._data)))

    def array(data, dtype=float):
        return FakeArray(data)

    def asarray(data, dtype=float):
        return array(data, dtype=dtype)

    def stack(sequence):
        converted = [
            item.tolist() if isinstance(item, FakeArray) else array(item).tolist()
            for item in sequence
        ]
        return FakeArray(converted)

    def arange(stop):
        stop_int = int(stop)
        return FakeArray([float(i) for i in range(stop_int)])

    def eye(size):
        n = int(size)
        data = []
        for i in range(n):
            row = [0.0] * n
            row[i] = 1.0
            data.append(row)
        return FakeArray(data)

    def mean(values):
        arr = asarray(values)
        flat = list(_flatten(arr._data))
        return sum(flat) / len(flat) if flat else 0.0

    def sqrt(value):
        return float(value) ** 0.5

    def isclose(a, b, atol=1e-8, rtol=1e-5):
        a_val = float(a)
        b_val = float(b)
        return abs(a_val - b_val) <= (atol + rtol * abs(b_val))

    numpy_module.array = array
    numpy_module.asarray = asarray
    numpy_module.stack = stack
    numpy_module.arange = arange
    numpy_module.eye = eye
    numpy_module.mean = mean
    numpy_module.sqrt = sqrt
    numpy_module.isclose = isclose
    numpy_module.isscalar = lambda obj: isinstance(obj, (int, float))
    numpy_module.bool_ = bool
    numpy_module.ndarray = FakeArray
    numpy_module.FakeArray = FakeArray

    install_stub("numpy", numpy_module)

    # yaml stub --------------------------------------------------------------
    yaml_module = types.ModuleType("yaml")

    class Node:
        def __init__(self, value):
            self.value = value

    class ScalarNode(Node):
        pass

    class SequenceNode(Node):
        pass

    class MappingNode(Node):
        pass

    class SafeLoader:
        constructors: dict = {}

        def __init__(self, stream=None):
            self.stream = stream

        @classmethod
        def add_constructor(cls, tag, constructor):
            cls.constructors[tag] = constructor

        def construct_scalar(self, node):
            return node.value

        def construct_sequence(self, node):
            return node.value

        def construct_mapping(self, node):
            return dict(node.value)

    def _parse_scalar(text: str):
        text = text.strip()
        if text == "":
            return ""
        lower = text.lower()
        try:
            if "." in text or "e" in lower:
                return float(text)
            return int(text)
        except ValueError:
            return text.strip('"\'')

    def _parse_value(text: str):
        text = text.strip()
        if text.startswith("[") and text.endswith("]"):
            inner = text[1:-1].strip()
            if not inner:
                return []
            return [_parse_value(part.strip()) for part in inner.split(",")]
        if text.startswith("{") and text.endswith("}"):
            inner = text[1:-1].strip()
            if not inner:
                return {}
            mapping = {}
            for part in inner.split(","):
                key, _, value = part.partition(":")
                mapping[key.strip()] = _parse_value(value.strip())
            return mapping
        return _parse_scalar(text)

    def _parse_mapping(lines: list[str], indent: int) -> dict:
        result: dict = {}
        while lines:
            raw_line = lines[0]
            if not raw_line.strip():
                lines.pop(0)
                continue
            current_indent = len(raw_line) - len(raw_line.lstrip(" "))
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError("invalid indentation")
            line = lines.pop(0).strip()
            key, _, value_part = line.partition(":")
            key = key.strip()
            value_part = value_part.strip()
            if not value_part:
                value = _parse_mapping(lines, indent + 2)
            else:
                value = _parse_value(value_part)
            result[key] = value
        return result

    def load(stream, Loader=None):
        if hasattr(stream, "read"):
            text = stream.read()
        else:
            text = str(stream)
        text = re.sub(r"!\S+\s*", "", text)
        lines = text.splitlines()
        return _parse_mapping(lines, 0)

    yaml_module.SafeLoader = SafeLoader
    yaml_module.ScalarNode = ScalarNode
    yaml_module.SequenceNode = SequenceNode
    yaml_module.MappingNode = MappingNode
    yaml_module.load = load

    install_stub("yaml", yaml_module)

    # torch stub -------------------------------------------------------------
    torch_stub = types.ModuleType("torch")

    class Device(str):
        pass

    torch_stub.device = lambda identifier=None: Device(str(identifier))

    class DummyCUDA:
        @staticmethod
        def is_available() -> bool:
            return False

    torch_stub.cuda = DummyCUDA()

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    torch_stub.no_grad = lambda: _NoGrad()
    torch_stub.load = lambda path, map_location=None: {}

    nn_module = types.ModuleType("torch.nn")

    class Module:
        def eval(self):
            return None

        def to(self, device):
            return self

    nn_module.Module = Module
    torch_stub.nn = nn_module

    install_stub("torch", torch_stub)
    install_stub("torch.nn", nn_module)

    # nequip package stubs ---------------------------------------------------
    nequip_module = types.ModuleType("nequip")
    nequip_module.__path__ = []  # mark as package
    install_stub("nequip", nequip_module)

    data_module = types.ModuleType("nequip.data")

    class AtomicData:
        def __init__(self, atoms, r_max):
            self.atoms = atoms
            self.r_max = r_max
            self.data_dict = {"atoms": atoms, "r_max": r_max}

        @staticmethod
        def from_ase(atoms, r_max):
            return AtomicData(atoms, r_max)

        def to(self, device):
            self.data_dict["device"] = device
            return self

        @staticmethod
        def to_AtomicDataDict(data):
            return dict(data.data_dict)

    class AtomicDataDict(dict):
        TOTAL_ENERGY_KEY = "energy"
        ALL_ENERGY_KEYS: list[str] = []
        Type = dict

    def register_fields(
        graph_fields=(), node_fields=(), cartesian_tensor_fields=None
    ) -> None:
        AtomicDataDict.ALL_ENERGY_KEYS.extend(list(graph_fields))

    data_module.AtomicData = AtomicData
    data_module.AtomicDataDict = AtomicDataDict
    data_module.register_fields = register_fields
    install_stub("nequip.data", data_module)
    nequip_module.data = data_module

    transforms_module = types.ModuleType("nequip.data.transforms")

    class TypeMapper:
        def __init__(self, chemical_symbol_to_type, type_names=None):
            self.mapping = dict(chemical_symbol_to_type)
            self.type_names = type_names

        def __call__(self, data):
            data.data_dict["type_mapping"] = self.mapping
            return data

    transforms_module.TypeMapper = TypeMapper
    install_stub("nequip.data.transforms", transforms_module)
    nequip_module.data.transforms = transforms_module

    model_module = types.ModuleType("nequip.model")

    def model_from_config(config=None, initialize=False):
        class DummyConfiguredModel(Module):
            def __init__(self, config):
                self.config = config

        return DummyConfiguredModel(config)

    model_module.model_from_config = model_from_config
    install_stub("nequip.model", model_module)
    nequip_module.model = model_module

    scripts_module = types.ModuleType("nequip.scripts")
    scripts_module.__path__ = []
    install_stub("nequip.scripts", scripts_module)
    nequip_module.scripts = scripts_module

    train_module = types.ModuleType("nequip.scripts.train")
    train_module.default_config = {
        "r_max": 1.0,
        "chemical_symbol_to_type": {"X": 0},
        "type_names": ["X"],
    }
    scripts_module.train = train_module
    install_stub("nequip.scripts.train", train_module)

    utils_module = types.ModuleType("nequip.utils")

    class Config(dict):
        @classmethod
        def from_dict(cls, data):
            return cls(dict(data))

    utils_module.Config = Config
    install_stub("nequip.utils", utils_module)
    nequip_module.utils = utils_module

    global_opts_module = types.ModuleType("nequip.utils._global_options")

    def _set_global_options(config):
        global_opts_module.last_config = config

    global_opts_module._set_global_options = _set_global_options
    utils_module._global_options = global_opts_module
    install_stub("nequip.utils._global_options", global_opts_module)

    train_pkg_module = types.ModuleType("nequip.train")
    train_pkg_module.__path__ = []
    install_stub("nequip.train", train_pkg_module)
    nequip_module.train = train_pkg_module

    train_key_module = types.ModuleType("nequip.train._key")
    train_key_module.ABBREV = {}
    install_stub("nequip.train._key", train_key_module)
    train_pkg_module._key = train_key_module

    # ase stub ---------------------------------------------------------------
    ase_module = types.ModuleType("ase")
    ase_module.__path__ = []
    io_module = types.ModuleType("ase.io")

    def fake_read(*args, **kwargs):
        return []

    def fake_write(*args, **kwargs):
        return None

    io_module.read = fake_read
    io_module.write = fake_write
    ase_module.io = io_module
    install_stub("ase", ase_module)
    install_stub("ase.io", io_module)

    # matplotlib stub -------------------------------------------------------
    class DummyAxes:
        def __init__(self):
            self.transAxes = object()

        def scatter(self, *args, **kwargs):
            return None

        def plot(self, *args, **kwargs):
            return None

        def set_xlim(self, *args, **kwargs):
            return None

        def set_ylim(self, *args, **kwargs):
            return None

        def set_xlabel(self, *args, **kwargs):
            return None

        def set_ylabel(self, *args, **kwargs):
            return None

        def set_title(self, *args, **kwargs):
            return None

        def set_aspect(self, *args, **kwargs):
            return None

        def text(self, *args, **kwargs):
            return None

        def axis(self, *args, **kwargs):
            return None

    class DummyFigure:
        def __init__(self, axes):
            self.axes = axes

        def tight_layout(self):
            return None

    class AxesGrid:
        def __init__(self, nrows, ncols):
            self._grid = [[DummyAxes() for _ in range(ncols)] for _ in range(nrows)]
            self.flat = [ax for row in self._grid for ax in row]

        def __getitem__(self, item):
            if isinstance(item, tuple):
                row, col = item
                return self._grid[row][col]
            return self._grid[item]

    matplotlib_module = types.ModuleType("matplotlib")
    matplotlib_module.__path__ = []
    pyplot_module = types.ModuleType("matplotlib.pyplot")

    def subplots(nrows, ncols, figsize=None):
        grid = AxesGrid(nrows, ncols)
        return DummyFigure(grid), grid

    pyplot_module.subplots = subplots
    pyplot_module.close = lambda fig: None
    matplotlib_module.pyplot = pyplot_module
    install_stub("matplotlib", matplotlib_module)
    install_stub("matplotlib.pyplot", pyplot_module)

    backends_module = types.ModuleType("matplotlib.backends")
    backends_module.__path__ = []
    install_stub("matplotlib.backends", backends_module)
    matplotlib_module.backends = backends_module

    backend_pdf_module = types.ModuleType("matplotlib.backends.backend_pdf")

    class PdfPages:
        def __init__(self, path):
            self.path = Path(path)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

        def savefig(self, fig):
            return None

        def close(self):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "wb") as handle:
                handle.write(b"%PDF-FAKE%")

    backend_pdf_module.PdfPages = PdfPages
    install_stub("matplotlib.backends.backend_pdf", backend_pdf_module)
    backends_module.backend_pdf = backend_pdf_module

    # sklearn.metrics stub ---------------------------------------------------
    sklearn_module = types.ModuleType("sklearn")
    sklearn_module.__path__ = []
    install_stub("sklearn", sklearn_module)

    metrics_module = types.ModuleType("sklearn.metrics")

    def mean_absolute_error(y_true, y_pred):
        true_flat = numpy_module.asarray(y_true).reshape(-1).tolist()
        pred_flat = numpy_module.asarray(y_pred).reshape(-1).tolist()
        diffs = [abs(a - b) for a, b in zip(true_flat, pred_flat)]
        return float(sum(diffs) / len(diffs)) if diffs else 0.0

    def mean_squared_error(y_true, y_pred):
        true_flat = numpy_module.asarray(y_true).reshape(-1).tolist()
        pred_flat = numpy_module.asarray(y_pred).reshape(-1).tolist()
        diffs = [(a - b) ** 2 for a, b in zip(true_flat, pred_flat)]
        return float(sum(diffs) / len(diffs)) if diffs else 0.0

    metrics_module.mean_absolute_error = mean_absolute_error
    metrics_module.mean_squared_error = mean_squared_error
    sklearn_module.metrics = metrics_module
    install_stub("sklearn.metrics", metrics_module)

    # Import the script under test now that dependencies are stubbed.
    module = importlib.import_module("scripts.run_inference_and_parity")

    yield module

    # Teardown: remove imported module and restore any prior modules.
    sys.modules.pop("scripts.run_inference_and_parity", None)
    for name, original in reversed(list(stubs.items())):
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


def test_load_yaml_ignore_tags(inference_module, tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        """
value: !Custom 1.23
nested:
  sequence: [1, !Other 2, 3]
  mapping: !Map {a: 5, b: 6}
""".strip(),
        encoding="utf-8",
    )

    data = inference_module.load_yaml_ignore_tags(yaml_path)

    assert data["value"] == pytest.approx(1.23)
    assert data["nested"]["sequence"] == [1, 2, 3]
    assert data["nested"]["mapping"] == {"a": 5, "b": 6}


def test_as_array_parses_inputs(inference_module):
    arr_from_string = inference_module._as_array("1, 2 3", length=3)
    assert arr_from_string.tolist() == pytest.approx([1.0, 2.0, 3.0])

    arr_from_list = inference_module._as_array([0, 0.5, 1.0], length=3)
    assert arr_from_list.tolist() == pytest.approx([0.0, 0.5, 1.0])

    with pytest.raises(ValueError):
        inference_module._as_array([1, 2], length=3)


def test_parity_plot_requires_data(inference_module):
    class Axis:
        def scatter(self, *args, **kwargs):
            return None

        def plot(self, *args, **kwargs):
            return None

        def set_xlim(self, *args, **kwargs):
            return None

        def set_ylim(self, *args, **kwargs):
            return None

        def set_xlabel(self, *args, **kwargs):
            return None

        def set_ylabel(self, *args, **kwargs):
            return None

        def set_title(self, *args, **kwargs):
            return None

        def set_aspect(self, *args, **kwargs):
            return None

        def text(self, *args, **kwargs):
            return None

    with pytest.raises(ValueError):
        inference_module.parity_plot(Axis(), [], [], "Empty")


def test_infer_predictions_updates_atoms(inference_module):
    np_module = inference_module.np

    class DummyTensor:
        def __init__(self, array):
            self._array = np_module.asarray(array)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._array

    class DummyModel:
        def __init__(self):
            self._call_index = 0

        def eval(self):
            return None

        def to(self, device):
            self.device = device
            return self

        def __call__(self, data_dict):
            idx = self._call_index
            self._call_index += 1
            energy = DummyTensor([5.0 + idx])
            polarization = DummyTensor([idx, idx + 1, idx + 2])
            base = [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
            ]
            polarizability = DummyTensor(
                [[value + idx for value in row] for row in base]
            )
            return {
                inference_module.AtomicDataDict.TOTAL_ENERGY_KEY: energy,
                inference_module.POLARIZATION_KEY: polarization,
                inference_module.POLARIZABILITY_KEY: polarizability,
            }

    class FakeAtoms:
        def __init__(self, total_energy, polarization, polarizability):
            self.info = {
                "total_energy": total_energy,
                "polarization": polarization,
                "polarizability": polarizability,
            }

    structures = [
        FakeAtoms(1.0, [0.0, 0.1, 0.2], list(range(9))),
        FakeAtoms(2.0, [0.3, 0.4, 0.5], [float(x) for x in range(9, 18)]),
    ]

    (
        energy_ref,
        energy_pred,
        pol_ref,
        pol_pred,
        polz_ref,
        polz_pred,
    ) = inference_module.infer_predictions(
        model=DummyModel(),
        type_mapper=inference_module.TypeMapper({"X": 0}),
        structures=structures,
        r_max=2.5,
        device="cpu",
    )

    assert energy_ref == [1.0, 2.0]
    assert energy_pred == [5.0, 6.0]

    assert pol_ref[0].tolist() == pytest.approx([0.0, 0.1, 0.2])
    assert pol_pred[1].tolist() == pytest.approx([1.0, 2.0, 3.0])

    assert polz_ref[0].tolist() == [
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
    ]
    assert polz_pred[1].tolist() == [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]

    assert structures[0].info["predicted_total_energy"] == pytest.approx(5.0)
    assert structures[1].info["predicted_polarization"] == [1.0, 2.0, 3.0]
    assert structures[0].info["predicted_polarizability"] == [float(x) for x in range(9)]


def test_create_parity_pdf_creates_file(inference_module, tmp_path):
    output_path = tmp_path / "parity.pdf"

    energy_ref = [0.0, 1.0]
    energy_pred = [0.1, 0.9]
    pol_ref = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    pol_pred = [[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]]
    polz_ref = [
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        [
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
    ]
    polz_pred = [
        [
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ],
        [
            [2.5, 0.0, 0.0],
            [0.0, 2.5, 0.0],
            [0.0, 0.0, 2.5],
        ],
    ]

    inference_module.create_parity_pdf(
        pdf_path=output_path,
        energy_ref=energy_ref,
        energy_pred=energy_pred,
        pol_ref=pol_ref,
        pol_pred=pol_pred,
        polz_ref=polz_ref,
        polz_pred=polz_pred,
    )

    assert output_path.is_file()
    assert output_path.read_bytes().startswith(b"%PDF")


def test_build_config_merges_defaults(inference_module):
    config = inference_module.build_config({"custom": 42})
    assert config["custom"] == 42
    assert config["r_max"] == 1.0
    assert config["chemical_symbol_to_type"] == {"X": 0}
