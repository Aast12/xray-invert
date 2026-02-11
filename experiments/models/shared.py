import jax

from chest_xray_sim.inverse.shared_weights_optimizer import SharedWeightsOptimizer
from .common import DualBandForwardMixin


class SharedWeightsSegmentationExperiment(DualBandForwardMixin, SharedWeightsOptimizer):
    """Segmentation experiment using shared weights optimization."""

    def __init__(self, inputs, hyperparams, w0, common_weights=True, **kwargs):
        super().__init__(inputs, hyperparams, w0, common_weights=common_weights, **kwargs)

    def forward(self, txm, weights):
        """Batch forward processing with shared weights."""
        result = jax.vmap(self._forward_single, in_axes=(0, None))(txm, weights)
        assert result.ndim == 3, f"Expected 3D batch output, got shape {result.shape}"
        return result
