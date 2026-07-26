from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

# Shared information ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class NetworkConfig:
    architecture:  Tuple
    epochs:        int
    learning_rate: float
    decay_steps:   Optional[int] = None
    warmup_steps:  Optional[int] = None
    save_filepath: str           = "checkpoints/weights.pkl"
    load_filepath: Optional[str] = None

@dataclass(frozen=True)
class TrainingConfig:
    batch_size:      int
    val_split:       float = 0.8
    do_validation:   bool  = True
    lambda_reg:      float = 0.0
    use_auto_lambda: bool  = False


@dataclass(frozen=True)
class LearnableParam:
    value:     float
    learnable: bool = False

@dataclass(frozen=True)
class PhysicsParamsConfig:
    params: Any

# Global settings ──────────────────────────────────────────────
@dataclass(frozen=True)
class SystemConfig:
    q0:            Any
    p0:            Any
    period:        float
    seed:          int
    system_params: Dict[str, float]

# MLP ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MLPLossConfig:
    lambda_data: float = 1.0
    augment_trajectory_derivative: bool = False

@dataclass(frozen=True)
class MLPTrainConfig:
    data:           DataGenerationConfig
    evaluation:     EvalSpec
    network:        NetworkConfig
    training:       TrainingConfig
    loss:           MLPLossConfig
    system:         SystemConfig
    visualizations: VisualizationConfig

# HNN ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class HamiltonianConfig:
    n_dof:               int
    dissipation:         str                      = "none"
    coord_names:         Optional[Tuple[str, ...]] = None
    known_term:          Optional[Callable]        = None
    dissipation_term:    Optional[Callable]        = None
    describe:            Optional[Callable]        = None
    penalize_correction: bool                      = False

@dataclass(frozen=True)
class HNNLossConfig:
    lambda_eom:          float
    hamiltonian:         HamiltonianConfig
    lambda_energy_decay: float = 0.0
    lambda_correction:   float = 0.0
    physics:                   Optional[PhysicsParamsConfig] = None

@dataclass(frozen=True)
class HNNTrainConfig:
    data:           DataGenerationConfig
    evaluation:     EvalSpec
    network:        NetworkConfig
    training:       TrainingConfig
    loss:           HNNLossConfig
    system:         SystemConfig
    visualizations: VisualizationConfig

# LNN ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LagrangianConfig:
    n_dof:               int
    dissipation:         str                      = "none"
    coord_names:         Optional[Tuple[str, ...]] = None
    known_term:          Optional[Callable]        = None
    dissipation_term:    Optional[Callable]        = None
    describe:            Optional[Callable]        = None
    damping:             float                     = 1e-3
    penalize_correction: bool                      = False

@dataclass(frozen=True)
class LNNLossConfig:
    lambda_eom:          float
    lagrangian:          LagrangianConfig
    lambda_energy_decay: float = 0.0
    lambda_correction:   float = 0.0
    physics:             Optional[PhysicsParamsConfig] = None

@dataclass(frozen=True)
class LNNTrainConfig:
    data:           DataGenerationConfig
    evaluation:     EvalSpec
    network:        NetworkConfig
    training:       TrainingConfig
    loss:           LNNLossConfig
    system:         SystemConfig
    visualizations: VisualizationConfig

# PINN ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CollocationConfig:
    n_points: int
    x_min:    float
    x_max:    float

@dataclass(frozen=True)
class NormalisationConfig:
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0

@dataclass(frozen=True)
class PINNLossConfig:
    lambda_data:     float
    lambda_physics:  float
    lambda_boundary: float
    collocation:     CollocationConfig
    normalisation:   NormalisationConfig
    physics:         PhysicsParamsConfig
    residual_fn:     Any
    causal_epsilon_max:   float = 0.0
    causal_anneal_epochs: int   = 1
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)

@dataclass(frozen=True)
class PINNTrainConfig:
    data:           DataGenerationConfig
    evaluation:     EvalSpec
    network:        NetworkConfig
    training:       TrainingConfig
    loss:           PINNLossConfig
    system:         SystemConfig
    visualizations: VisualizationConfig

# Data generation ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DataGenerationConfig:
    make_data:         Optional[Callable]
    system:              SystemConfig
    type:                str                            = "field"
    # Field data
    q_range:             Optional[Tuple[float, float]]  = (-2.0, 2.0)
    p_range:             Optional[Tuple[float, float]]  = (-2.0, 2.0)
    field_N:             Optional[int]                  = 8192
    field_sigma:         Optional[float]                = 0.0
    coords:              Optional[str]                  = "canonical"
    # Trajectory data
    trajectory_N:        Optional[int]                  = 8192
    trajectory_sigma:    Optional[float]                = 0.0
    boundary_conditions: Optional[Tuple[float, ...]]    = None
    period:              Optional[float]                = None

# Visualizations ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AnimationConfig:
    horizon_multiplier: float         = 2.0
    n_points:           int           = 200
    fps:                int           = 20
    trail_len:          Optional[int] = None
    x_pad:              float         = 3.0
    n_x:                int           = 400
    scale:              float         = 1.0

@dataclass(frozen=True)
class LiveLearningConfig:
    enabled:         bool  = False
    t_reach:         float = 0.0
    n_points:        int   = 200
    interval_ms:     int   = 0
    log_scale:       bool  = True
    steps_per_frame: int   = 1

@dataclass(frozen=True)
class VisualizationConfig:
    animation:     AnimationConfig
    live_learning: LiveLearningConfig
    data:          DataGenerationConfig
    system:        SystemConfig
    make_vis:      Callable

# Evaluation ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class EvalSpec:
    q0:                 float
    p0:                 float
    period:             float
    radii:              Tuple[float, ...]       = (1.0, 2.0, 3.0, 5.0)
    horizons:           Tuple[int, ...]         = (2, 5, 10, 20, 50)
    pinn_horizons:      Tuple[int, ...]         = (1, 2, 5, 10)
    pinn_data_n:        int                     = 512
    points_per_period:  int                     = 100
    n_eval_points:      int                     = 200
    extra_plots:        Optional[Callable]      = None
    live_learning_plot: Optional[Callable]      = None
    visualizations:     Optional[VisualizationConfig] = None

# Models ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class InferenceConfig:
    system: SystemConfig

@dataclass(frozen=True)
class BenchmarkConfig:
    system: SystemConfig