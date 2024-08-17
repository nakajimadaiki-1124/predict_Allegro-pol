import torch
from nequip.data import AtomicDataDict


class FoldedPolLoss:
    """Custom loss function for polarization, accounting for multiplicity of
       polarization values using the minimum image convention.

    To use this in a training, set in your YAML file:

        loss_coeffs:
            polarization:
                - 0.1  (choose this)
                - !!python/object:nequip-pol-tmp.nequip_pol.pol_loss.FoldedPolLoss {}

    This funny syntax tells PyYAML to construct an object of this class and put it in the config.
    In this case, the `key` argument to `__call__` will be `"polarization"`.
    """

    def __call__(
        self,
        pred: AtomicDataDict.Type,
        ref: AtomicDataDict.Type,
        key: str,
        mean: bool,
    ):
        """
        Args:
            pred: output of the model (can be a batch)
            ref:  training data (can be a batch)
            key:  which key to compute loss on, from the `loss_coeffs` config shown above.
            mean: whether to return a single scalar loss value.  Can basically be ignored (`assert mean`)
                as long as you don't want to use this loss as a metric (if you do, return instead a
                per-(atom/graph/whatever) tensor of losses).
        """
        assert key == "polarization"

        # get variables
        cell = pred["_scaled_cell"].to(pred[key].dtype)  # (Nbatch, 3, 3)

        # number of atoms
        N = torch.bincount(ref[AtomicDataDict.BATCH_KEY])
        N = N.reshape((-1, 1))

        # difference between prediction and label
        pol_diff = pred[key] - ref[key]  # (Nbatch, 3)

        # map pol_diff to fractional coordinates
        frac_pol_diff = torch.einsum(
            "bi, bij -> bj", pol_diff, torch.linalg.inv(cell)
        )  # (Nbatch, 3)

        # fold difference into "unit cell" in fractional coordinates, i.e. 3 by 3 identity matrix
        frac_pol_diff = torch.remainder(frac_pol_diff, 1.0)

        # apply minimum image convention
        frac_pol_diff = torch.where(
            frac_pol_diff > 0.5, frac_pol_diff - 1.0, frac_pol_diff
        )
        # the following is likely redundant, possibly remove in the future
        frac_pol_diff = torch.where(
            frac_pol_diff < -0.5, frac_pol_diff + 1.0, frac_pol_diff
        )

        # map back from fractional to (scaled) Cartesian
        pol_diff = torch.einsum("bi, bij -> bj", frac_pol_diff, cell)  # (Nbatch, 3)

        if mean:  # used for loss
            return torch.mean(torch.square(pol_diff) / N / N)
        else:  # used for errors
            return torch.abs(pol_diff) / N
