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
    save_filepath: Optional[str] = None
    load_filepath: Optional[str] = None
    schedule:          str   = "cosine"
    end_value:         float = 1e-5
    plateau_factor:    float = 0.5
    plateau_patience:  int   = 50
    plateau_cooldown:  int   = 20
    plateau_rtol:      float = 1e-3
    plateau_min_scale: float = 1e-3

@dataclass(frozen=True)
class TrainingConfig:
    batch_size:      int
    val_split:       float = 0.8
    do_validation:   bool  = True
    lambda_reg:      float = 0.0
    use_auto_lambda: bool  = False
    live_metrics:    bool  = True
    static_metrics:  bool  = True

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
    visualizations: Optional[VisualizationConfig]  = None
    live_learning:  Optional[LiveLearningConfig]   = None

# HNN ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class HamiltonianConfig:
    n_dof:               int
    dissipation:         str                       = "none"
    integrator:          str                       = "rk4"
    known_term:          Optional[Callable]        = None
    dissipation_term:    Optional[Callable]        = None
    describe:            Optional[Callable]        = None
    penalize_correction: bool                      = False

@dataclass(frozen=True)
class HNNLossConfig:
    lambda_eom:          float
    hamiltonian:         HamiltonianConfig
    lambda_correction:   Optional[float] = None
    physics:             Optional[PhysicsParamsConfig] = None

@dataclass(frozen=True)
class HNNTrainConfig:
    data:           DataGenerationConfig
    evaluation:     EvalSpec
    network:        NetworkConfig
    training:       TrainingConfig
    loss:           HNNLossConfig
    system:         SystemConfig
    visualizations: Optional[VisualizationConfig]  = None
    live_learning:  Optional[LiveLearningConfig]   = None

# LNN ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LagrangianConfig:
    n_dof:               int
    dissipation:         str                       = "none"
    known_term:          Optional[Callable]        = None
    dissipation_term:    Optional[Callable]        = None
    describe:            Optional[Callable]        = None
    penalize_correction: bool                      = False

@dataclass(frozen=True)
class LNNLossConfig:
    lambda_eom:          float
    lagrangian:          LagrangianConfig
    lambda_correction:   Optional[float] = None
    physics:             Optional[PhysicsParamsConfig] = None

@dataclass(frozen=True)
class LNNTrainConfig:
    data:           DataGenerationConfig
    evaluation:     EvalSpec
    network:        NetworkConfig
    training:       TrainingConfig
    loss:           LNNLossConfig
    system:         SystemConfig
    visualizations: Optional[VisualizationConfig]  = None
    live_learning:  Optional[LiveLearningConfig]   = None

# PINN ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CollocationConfig:
    n_points: int
    x_min:    float
    x_max:    float
    bins:     int = 10

@dataclass(frozen=True)
class PINNLossConfig:
    collocation:     CollocationConfig
    physics:         PhysicsParamsConfig
    residual_fn:     Any
    lambda_data:     Optional[float] = None
    lambda_physics:  Optional[float] = None
    lambda_initial:  Optional[float] = None
    lambda_boundary: Optional[float] = None
    causal_epsilon:  float = 0.0
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)

@dataclass(frozen=True)
class PINNTrainConfig:
    data:           DataGenerationConfig
    evaluation:     EvalSpec
    network:        NetworkConfig
    training:       TrainingConfig
    loss:           PINNLossConfig
    system:         SystemConfig
    visualizations: Optional[VisualizationConfig] = None
    live_learning:  Optional[LiveLearningConfig]  = None

# Data generation ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DataGenerationConfig:
    make_data:           Optional[Callable]
    type:                str                            = "field"
    # Field data
    q_range:             Optional[Tuple[float, float]]  = (-2.0, 2.0)
    p_range:             Optional[Tuple[float, float]]  = (-2.0, 2.0)
    field_N:             Optional[int]                  = 8192
    field_sigma_y:         Optional[float]                = 0.0
    field_sigma_x:         Optional[float]                = 0.0
    coords:              Optional[str]                  = "canonical"
    # Trajectory data
    trajectory_N:        Optional[int]                  = 8192
    trajectory_sigma_y:    Optional[float]                = 0.0
    trajectory_sigma_x:    Optional[float]                = 0.0
    initial_conditions:  Optional[Tuple[float, ...]]    = None
    period:              Optional[float]                = None

# Visualizations ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class VisualizationConfig:
    enabled:         bool               = False
    horizon:         float              = 3.0 
    n_frames:        int                = 400
    interval:        int                = 25
    save_path:       bool               = False
    format:          str                = "mp4"
    dpi:             int                = 100
    make_vis:        Optional[Callable] = None

@dataclass(frozen=True)
class LiveLearningConfig:
    enabled:         bool               = False
    n_points:        int                = 200
    plot_every:      int                = 1
    horizon:         float              = 3.0
    log_scale:       bool               = True
    save_path:       bool               = False
    format:          str                = "mp4"
    make_live_plot:  Optional[Callable] = None

# Evaluation ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class EvalSpec:
    horizons:           Tuple[int, ...]         = (2, 5, 10, 20, 50)
    radii:              Tuple[float, ...]       = (0.5, 1.0, 1.5, 2.0, 3.0)
    horizon_tolerance:  float                   = 0.01
    horizon_window:     float                   = 0.25


# Tuning ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TuningConfig:
    check_every:        int   = 500
    collapse_fraction:  float = 0.5
    collapse_patience:  int   = 2
