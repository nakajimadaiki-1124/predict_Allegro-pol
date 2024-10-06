from typing import Optional
import logging

from nequip.data import AtomicDataDict, AtomicDataset
from nequip.nn import SequentialGraphNetwork, AtomwiseReduce, GraphModuleMixin
from nequip.utils import Config

from allegro.nn import (
    ProductTypeEmbedding,
    AllegroBesselBasis,
    EdgewiseEnergySum,
    Allegro_Module,
    ScalarMLP,
)
from allegro._keys import EDGE_FEATURES, EDGE_ENERGY
from allegro.model._allegro import _allegro_config_preprocess

from .._keys import EXTERNAL_ELECTRIC_FIELD_KEY
from ..nn import SphericalHarmonicEdgeAttrs
from ..nn import ForceStressPolarizationOutput as ForceStressPolarizationOutput_Module


def Allegro(config, initialize: bool, dataset: Optional[AtomicDataset] = None):
    logging.debug("Building Allegro model...")

    _allegro_config_preprocess(config, initialize=initialize, dataset=dataset)

    layers = {
        # -- Encode --
        # Get various edge invariants
        "radial_basis": AllegroBesselBasis,
        "typeembed": (
            ProductTypeEmbedding,
            dict(
                initial_scalar_embedding_dim=config.get(
                    "initial_scalar_embedding_dim",
                    # sane default to the MLP that comes next
                    config["two_body_latent_mlp_latent_dimensions"][0],
                ),
            ),
        ),
        # Get edge tensors
        "spharm": SphericalHarmonicEdgeAttrs,  # This is the custom one for including external electric field
        # The core allegro model:
        "allegro": (
            Allegro_Module,
            dict(
                field=AtomicDataDict.EDGE_ATTRS_KEY,  # initial input is the edge SH
                edge_invariant_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
            ),
        ),
        "edge_eng": (
            ScalarMLP,
            dict(field=EDGE_FEATURES, out_field=EDGE_ENERGY, mlp_output_dimension=1),
        ),
        # Sum edgewise energies -> per-atom energies:
        "edge_eng_sum": EdgewiseEnergySum,
        # Sum system energy:
        "total_energy_sum": (
            AtomwiseReduce,
            dict(
                reduce="sum",
                field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
            ),
        ),
    }

    model = SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
        irreps_in={EXTERNAL_ELECTRIC_FIELD_KEY: "1x1o"},
    )

    return model


def ForceStressPolarizationOutput(
    model: GraphModuleMixin,
    config: Config,
) -> ForceStressPolarizationOutput_Module:
    return ForceStressPolarizationOutput_Module(
        model,
        do_born_charge=config.get("do_born_charge", True),
    )
