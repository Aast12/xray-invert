from dataclasses import dataclass
import typing

from utils.tracking import ExperimentProtocol
from chest_xray_sim.pipeline.config import PipelineConfig


@dataclass()
class ExperimentArgs(ExperimentProtocol):
    n_steps: int
    total_variation: float
    prior_weight: float
    gmse_weight: float
    PRNGKey: int
    tm_init_params: tuple[str, tuple[float, float] | None]
    constant_weights: bool
    eps: float
    max_sigma: float
    max_enhancement: float
    smooth_metric: typing.Literal["tikonov", "tv"]
    use_band_similarity: bool
    windowing_type: typing.Literal["sigmoid", "linear"]
    data_fidelity: typing.Literal["l2", "ssim"]
    optimizer: str
    lr: float
    momentum: float
    weight_decay: float
    sigma: float
    high_sigma: float
    low_min_enhancement: float
    high_min_enhancement: float
    # Individual-only fields with defaults
    variant: str = ""
    segmentation_th: float = 0.6
    log_samples: int = 10

    def to_pipeline_config(self) -> PipelineConfig:
        """Convert experiment args to a PipelineConfig."""
        return PipelineConfig(
            n_steps=self.n_steps,
            lr=self.lr,
            optimizer=self.optimizer,
            eps=self.eps,
            sigma=self.sigma,
            high_sigma=self.high_sigma,
            windowing_type=self.windowing_type,
            total_variation=self.total_variation,
            prior_weight=self.prior_weight,
            smooth_metric=self.smooth_metric,
            data_fidelity=self.data_fidelity,
            max_enhancement=self.max_enhancement,
            low_min_enhancement=self.low_min_enhancement,
            high_min_enhancement=self.high_min_enhancement,
            PRNGKey=self.PRNGKey,
            tm_init_params=self.tm_init_params,
            constant_weights=self.constant_weights,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            segmentation_th=self.segmentation_th,
        )
