from nequip.data import register_fields, AtomicDataDict
from nequip.train._key import ABBREV

from typing import Final

# define keys for polarization and its associated properties
POLARIZATION_KEY: Final[str] = "polarization"
BORN_CHARGE_KEY: Final[str] = "born_charge"
EXTERNAL_ELECTRIC_FIELD_KEY: Final[str] = "external_electric_field"
POLARIZABILITY_KEY: Final[str] = "polarizability"

# register fields
register_fields(
    graph_fields=[POLARIZATION_KEY, EXTERNAL_ELECTRIC_FIELD_KEY, POLARIZABILITY_KEY],
    node_fields=[BORN_CHARGE_KEY],
    cartesian_tensor_fields={
        BORN_CHARGE_KEY: "ij",
        POLARIZABILITY_KEY: "ij=ji",
    },  # Born charge is not, in general, symmetric
)

# related by a linear function (differentiation) to the energy:
AtomicDataDict.ALL_ENERGY_KEYS.extend(
    [POLARIZATION_KEY, BORN_CHARGE_KEY, POLARIZABILITY_KEY]
)

ABBREV.update({POLARIZATION_KEY: "P", BORN_CHARGE_KEY: "Zb", POLARIZABILITY_KEY: "α"})
