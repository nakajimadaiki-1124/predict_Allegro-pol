from typing import Union

import torch

from e3nn import o3
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin

from .. import _keys


@compile_mode("script")
class SphericalHarmonicEdgeAttrs(GraphModuleMixin, torch.nn.Module):

    out_field: str
    electric_field_normalization: float

    def __init__(
        self,
        electric_field_normalization: float,
        irreps_edge_sh: Union[int, str, o3.Irreps],
        irreps_elec_field_sh: Union[int, str, o3.Irreps],
        edge_sh_normalization: str = "component",
        edge_sh_normalize: bool = True,
        irreps_in=None,
        out_field: str = AtomicDataDict.EDGE_ATTRS_KEY,
    ):
        super().__init__()
        self.out_field = out_field
        self.electric_field_normalization = electric_field_normalization

        if isinstance(irreps_edge_sh, int):
            self.irreps_edge_sh = o3.Irreps.spherical_harmonics(irreps_edge_sh)
        else:
            self.irreps_edge_sh = o3.Irreps(irreps_edge_sh)

        # additional lines for elec_field_sh
        if isinstance(irreps_elec_field_sh, int):
            self.irreps_elec_field_sh = o3.Irreps.spherical_harmonics(
                irreps_elec_field_sh
            )
        else:
            self.irreps_elec_field_sh = o3.Irreps(irreps_elec_field_sh)

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[_keys.EXTERNAL_ELECTRIC_FIELD_KEY],
            irreps_out={out_field: self.irreps_edge_sh + self.irreps_elec_field_sh},
        )

        self.sh = o3.SphericalHarmonics(
            self.irreps_edge_sh, edge_sh_normalize, edge_sh_normalization
        )
        # custom sh for electric field
        self.elec_field_sh = o3.SphericalHarmonics(
            self.irreps_elec_field_sh, False, edge_sh_normalization
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # PROCESS EDGES
        data = AtomicDataDict.with_batch(data)
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=False)
        edge_vec = data[AtomicDataDict.EDGE_VECTORS_KEY]
        edge_sh = self.sh(edge_vec)  # (Nedge, lm_edge)

        # PROCESS ELECTRIC FIELD
        elec_field = data[_keys.EXTERNAL_ELECTRIC_FIELD_KEY].div(
            self.electric_field_normalization
        )  # (Nbatch,)
        elec_field_sh_embed = self.elec_field_sh(elec_field)  # (Nbatch, lm_field)
        # map electric field sh embedding: (Nbatch, lm) -> (Nedge, lm)
        edge_center = torch.select(
            data[AtomicDataDict.EDGE_INDEX_KEY], 0, 0
        )  # (Nedge,)
        edge_batch = torch.index_select(
            data[AtomicDataDict.BATCH_KEY], 0, edge_center
        )  # (Nedge,)
        per_edge_field_sh_embed = torch.index_select(
            elec_field_sh_embed, 0, edge_batch
        )  # (Nedge, lm_field)

        data[self.out_field] = torch.cat(
            (edge_sh, per_edge_field_sh_embed), dim=-1
        )  # (Nedge, lm_edge + lm_field)
        return data
