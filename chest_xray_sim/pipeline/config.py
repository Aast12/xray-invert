"""Pipeline configuration and presets."""

import typing
from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """All hyperparameters needed to run the inverse pipeline."""

    n_steps: int = 300
    lr: float = 0.01
    optimizer: str = "amsgrad"
    eps: float = 1e-7

    # Forward model
    sigma: float = 1.0
    high_sigma: float = 10.0
    windowing_type: typing.Literal["sigmoid", "linear"] = "sigmoid"

    # Regularization
    total_variation: float = 1e-7
    prior_weight: float = 0.0001
    smooth_metric: typing.Literal["tikonov", "tv", "tv2", "none"] = "tikonov"

    # Data fidelity
    data_fidelity: typing.Literal["l2", "ssim"] = "l2"

    # Projection bounds
    max_enhancement: float = 0.8
    low_min_enhancement: float = 0.1
    high_min_enhancement: float = 0.1

    # Initialization
    PRNGKey: int = 0
    tm_init_params: tuple[str, tuple[float, float] | None] = ("naive", None)
    constant_weights: bool = False

    # Optimizer-specific
    momentum: float = 0.9
    weight_decay: float = 0.0001

    # Misc
    segmentation_th: float = 0.6
    log_interval: int = 100


# Best variable-weights config (W&B run yiutl75n)
_DUAL_BAND_PRESET = PipelineConfig(
    n_steps=300,
    lr=0.006849,
    optimizer="amsgrad",
    eps=1e-7,
    sigma=1.0,
    high_sigma=10.0,
    windowing_type="sigmoid",
    total_variation=3.894e-7,
    prior_weight=0.0003118,
    smooth_metric="tikonov",
    data_fidelity="l2",
    max_enhancement=0.8,
    low_min_enhancement=0.1985,
    high_min_enhancement=0.2664,
    PRNGKey=0,
    tm_init_params=("naive", None),
    constant_weights=False,
    momentum=0.9,
    weight_decay=0.0001,
)

# Best constant-weights config (W&B run cuhsitz3)
_DUAL_BAND_CONSTANT_PRESET = PipelineConfig(
    n_steps=300,
    lr=0.01291,
    optimizer="amsgrad",
    eps=1e-6,
    sigma=1.0,
    high_sigma=10.0,
    windowing_type="linear",
    total_variation=4.775e-7,
    prior_weight=0.0004336,
    smooth_metric="tikonov",
    data_fidelity="l2",
    max_enhancement=0.8,
    low_min_enhancement=0.4478,
    high_min_enhancement=0.2397,
    PRNGKey=0,
    tm_init_params=("naive", None),
    constant_weights=True,
    momentum=0.9,
    weight_decay=0.0001,
)

PRESETS: dict[str, PipelineConfig] = {
    "dual_band": _DUAL_BAND_PRESET,
    "dual_band_constant": _DUAL_BAND_CONSTANT_PRESET,
}
