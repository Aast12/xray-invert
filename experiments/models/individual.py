import jax

from chest_xray_sim.inverse.individual_gradient_optimizer import (
    IndividualGradientOptimizer,
)
from .common import DualBandForwardMixin


class IndividualSegmentationExperiment(DualBandForwardMixin, IndividualGradientOptimizer):
    """Segmentation experiment using individual gradient optimization."""

    def forward(self, txm, weights):
        """Batch forward processing with per-image weights."""
        result = jax.vmap(self._forward_single, in_axes=(0, 0))(txm, weights)
        assert result.ndim == 3, f"Expected 3D batch output, got shape {result.shape}"
        return result
