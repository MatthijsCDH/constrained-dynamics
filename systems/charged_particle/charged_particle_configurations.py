import numpy as np

from configs.configurations import (
    NetworkConfig, TrainingConfig, MLPLossConfig, MLPTrainConfig,
    HNNLossConfig, HNNTrainConfig,
    LNNLossConfig, LNNTrainConfig,
    CollocationConfig, PhysicsParamsConfig, PINNLossConfig, PINNTrainConfig,
    LearnableParam, EvalSpec, DataGenerationConfig, SystemConfig, LiveLearningConfig,
    VisualizationConfig,
)
from model.physics_loss import inverse_bounded_param
from systems.charged_particle.data import make_data
from systems.charged_particle.charged_particle_visualizations import make_live_plot, make_animation
from systems.charged_particle import pinn as cp_pinn
from systems.charged_particle.hnn import (
    HAMILTONIAN, make_hamiltonian_parametric, make_hamiltonian_parametric_mechanical,
)
from systems.charged_particle.lnn import LAGRANGIAN, make_lagrangian_parametric


# Global ────────────────────────────────────────────
SYSTEM = SystemConfig(
    q0=np.array([1.0, 0.0]),
    p0=np.array([0.0, 0.6]),
    period=2 * np.pi,
    seed=0,
    system_params={"m": 1.0, "e": 1.0, "B": 1.0, "k": 1.0, "alpha": 0.6},
)

BOUNDS = {
    "m": (0.1, 5.0),
    "e": (0.1, 5.0),
    "B": (0.1, 5.0),
    "k": (0.1, 5.0),
    "alpha": (0.0, 3.0),
}
Q0_BOUNDS = (-3.0, 3.0)
P0_BOUNDS = (-3.0, 3.0)

HAMILTONIAN_PARAMETRIC            = make_hamiltonian_parametric(BOUNDS)
HAMILTONIAN_PARAMETRIC_MECHANICAL = make_hamiltonian_parametric_mechanical(BOUNDS)
LAGRANGIAN_PARAMETRIC             = make_lagrangian_parametric(BOUNDS)

PHYSICS_GUESS = PhysicsParamsConfig(params={
    "log_m": LearnableParam(value=inverse_bounded_param(SYSTEM.system_params["m"], *BOUNDS["m"]), learnable=False),
    "log_e": LearnableParam(value=inverse_bounded_param(SYSTEM.system_params["e"], *BOUNDS["e"]), learnable=False),
    "log_B": LearnableParam(value=inverse_bounded_param(0.4, *BOUNDS["B"]), learnable=True),
    "log_k": LearnableParam(value=inverse_bounded_param(0.4, *BOUNDS["k"]), learnable=True),
    "log_alpha": LearnableParam(value=inverse_bounded_param(0.2, *BOUNDS["alpha"]), learnable=True),
})

# Training presets ────────────────────────────────────────────────────────
TRAINING = TrainingConfig(
    batch_size      = 128,
    val_split       = 0.8,
    do_validation   = True,
    lambda_reg      = 8.4e-5,
    use_auto_lambda = False,
)

# Data generation ────────────────────────────────────────────────────────
DATA_GEN_FIELD = DataGenerationConfig(
    make_data        = make_data,
    type             = "field",
    coords           = "canonical",
    q_range          = (-2.0, 2.0),
    p_range          = (-2.0, 2.0),
    field_N          = 8192,
    field_sigma_y      = 0.00,
)

DATA_GEN_FIELD_MECHANICAL = DataGenerationConfig(
    make_data        = make_data,
    type             = "field",
    coords           = "mechanical",
    q_range          = (-2.0, 2.0),
    p_range          = (-2.0, 2.0),
    field_N          = 8192,
    field_sigma_y      = 0.00,
)

DATA_GEN_FIELD_VELOCITY = DataGenerationConfig(
    make_data        = make_data,
    type             = "field",
    coords           = "velocity",
    q_range          = (-2.0, 2.0),
    p_range          = (-2.0, 2.0),
    field_N          = 8192,
    field_sigma_y      = 0.00,
)

DATA_GEN_TRAJECTORY = DataGenerationConfig(
    make_data          = make_data,
    type               = "trajectory",
    coords             = "velocity",
    trajectory_N       = 8192,
    trajectory_sigma_y   = 0.00,
    initial_conditions = (SYSTEM.q0, SYSTEM.p0),
    period             = SYSTEM.period,
)

# Evaluations ────────────────────────────────────────────────────────
EVAL = EvalSpec()

# Animation ────────────────────────────────────────────────────────
VISUALIZATION = VisualizationConfig(enabled=True, horizon=3.0, n_frames=400, interval=25, save_path=True, format="mp4", make_vis=make_animation)

# Live learning ────────────────────────────────────────────────────────
LIVE_LEARNING = LiveLearningConfig(enabled=True, n_points=400, plot_every=100, horizon=3.0, save_path=True, format="mp4", make_live_plot=make_live_plot)


# MLP ─────────────────────────────────────────────────────────────────
MLP_LOSS = MLPLossConfig(augment_trajectory_derivative=True)

MLP_NETWORK = NetworkConfig(
    architecture = (
        {"type": "fc", "units": 128, "activation": "sin"},
        {"type": "fc", "units": 128, "activation": "tanh"},
        {"type": "fc", "units": 128, "activation": "tanh"},
        {"type": "fc", "units": 128, "activation": "sin"},
        {"type": "fc", "units": 4,   "activation": "linear"},
    ),
    epochs        = 5000,
    learning_rate = 1e-3,
)

MLP_NETWORK_TRAJECTORY = NetworkConfig(
    architecture = (
        {"type": "fc", "units": 128, "activation": "sin"},
        {"type": "fc", "units": 128, "activation": "tanh"},
        {"type": "fc", "units": 128, "activation": "tanh"},
        {"type": "fc", "units": 128, "activation": "sin"},
        {"type": "fc", "units": 2,   "activation": "linear"},
    ),
    epochs        = 5000,
    learning_rate = 1e-3,
    schedule      = "plateau",
)

# HNN ────────────────────────────────────────────────────────
HNN_LOSS = HNNLossConfig(
    lambda_eom  = 1.0,
    hamiltonian = HAMILTONIAN,
)

HNN_LOSS_PARAMETRIC = HNNLossConfig(
    lambda_eom        = 1.0,
    hamiltonian       = HAMILTONIAN_PARAMETRIC,
    lambda_correction = 0.05,
    physics           = PHYSICS_GUESS,
)

HNN_LOSS_PARAMETRIC_MECHANICAL = HNNLossConfig(
    lambda_eom        = 1.0,
    hamiltonian       = HAMILTONIAN_PARAMETRIC_MECHANICAL,
    lambda_correction = 0.05,
    physics           = PHYSICS_GUESS,
)

HNN_NETWORK = NetworkConfig(
    architecture = (
        {"type": "fc", "units": 128, "activation": "sin"},
        {"type": "fc", "units": 128, "activation": "tanh"},
        {"type": "fc", "units": 128, "activation": "tanh"},
        {"type": "fc", "units": 128, "activation": "sin"},
        {"type": "fc", "units": 1,   "activation": "linear"},
    ),
    epochs        = 5000,
    learning_rate = 1e-3,
)


# LNN ────────────────────────────────────────────────────────
LNN_LOSS = LNNLossConfig(
    lambda_eom = 1.0,
    lagrangian = LAGRANGIAN,
)

LNN_LOSS_PARAMETRIC = LNNLossConfig(
    lambda_eom        = 1.0,
    lagrangian        = LAGRANGIAN_PARAMETRIC,
    lambda_correction = 3.0,
    physics           = PHYSICS_GUESS,
)

LNN_NETWORK = NetworkConfig(
    architecture = (
        {"type": "fc", "units": 128, "activation": "softplus"},
        {"type": "fc", "units": 128, "activation": "softplus"},
        {"type": "fc", "units": 128, "activation": "softplus"},
        {"type": "fc", "units": 1,   "activation": "linear"},
    ),
    epochs        = 5000,
    learning_rate = 1e-3,
)

# PINN ────────────────────────────────────────────────────────
PINN_LOSS = PINNLossConfig(
    lambda_data    = 10.0,
    lambda_physics = 1.0,
    lambda_initial = 3.0,
    collocation    = CollocationConfig(n_points=400, x_min=0.0, x_max=EVAL.horizons[0] * SYSTEM.period),
    physics        = PhysicsParamsConfig(params={
        **PHYSICS_GUESS.params,
        "q0_1": LearnableParam(value=inverse_bounded_param(float(SYSTEM.q0[0]), *Q0_BOUNDS), learnable=True),
        "q0_2": LearnableParam(value=inverse_bounded_param(float(SYSTEM.q0[1]), *Q0_BOUNDS), learnable=True),
        "p0_1": LearnableParam(value=inverse_bounded_param(float(SYSTEM.p0[0]), *P0_BOUNDS), learnable=True),
        "p0_2": LearnableParam(value=inverse_bounded_param(float(SYSTEM.p0[1]), *P0_BOUNDS), learnable=True),
    }),
    residual_fn    = cp_pinn.ChargedParticlePINNLoss,
    causal_epsilon     = 0.0,
    bounds         = {**BOUNDS, "q0_1": Q0_BOUNDS, "q0_2": Q0_BOUNDS, "p0_1": P0_BOUNDS, "p0_2": P0_BOUNDS},
)

PINN_NETWORK = NetworkConfig(
    architecture = (
        {"type": "fc", "units": 128, "activation": "sin"},
        {"type": "fc", "units": 128, "activation": "tanh"},
        {"type": "fc", "units": 128, "activation": "tanh"},
        {"type": "fc", "units": 128, "activation": "sin"},
        {"type": "fc", "units": 2,   "activation": "linear"},
    ),
    epochs        = 5000,
    learning_rate = 1e-3,
    schedule      = "plateau",
)


# Experiments ──────────────────────────────────────────────────────────────
CONFIGS = {
    "mlp":                        MLPTrainConfig(data = DATA_GEN_FIELD,            evaluation = EVAL, network=MLP_NETWORK,            training=TRAINING, loss=MLP_LOSS,                       system=SYSTEM, visualizations=VISUALIZATION, live_learning=LIVE_LEARNING),
    "mlp_trajectory":             MLPTrainConfig(data = DATA_GEN_TRAJECTORY,       evaluation = EVAL, network=MLP_NETWORK_TRAJECTORY, training=TRAINING, loss=MLP_LOSS,                       system=SYSTEM, visualizations=VISUALIZATION, live_learning=LIVE_LEARNING),
    "hnn":                        HNNTrainConfig(data = DATA_GEN_FIELD,            evaluation = EVAL, network=HNN_NETWORK,            training=TRAINING, loss=HNN_LOSS,                       system=SYSTEM, visualizations=VISUALIZATION, live_learning=LIVE_LEARNING),
    "hnn_mechanical":             HNNTrainConfig(data = DATA_GEN_FIELD_MECHANICAL, evaluation = EVAL, network=HNN_NETWORK,            training=TRAINING, loss=HNN_LOSS,                       system=SYSTEM, visualizations=VISUALIZATION, live_learning=LIVE_LEARNING),
    "hnn_parametric":             HNNTrainConfig(data = DATA_GEN_FIELD,            evaluation = EVAL, network=HNN_NETWORK,            training=TRAINING, loss=HNN_LOSS_PARAMETRIC,            system=SYSTEM, visualizations=VISUALIZATION, live_learning=LIVE_LEARNING),
    "hnn_parametric_mechanical":  HNNTrainConfig(data = DATA_GEN_FIELD_MECHANICAL, evaluation = EVAL, network=HNN_NETWORK,            training=TRAINING, loss=HNN_LOSS_PARAMETRIC_MECHANICAL, system=SYSTEM, visualizations=VISUALIZATION, live_learning=LIVE_LEARNING),
    "lnn":                        LNNTrainConfig(data = DATA_GEN_FIELD_VELOCITY,   evaluation = EVAL, network=LNN_NETWORK,            training=TRAINING, loss=LNN_LOSS,                       system=SYSTEM, visualizations=VISUALIZATION, live_learning=LIVE_LEARNING),
    "lnn_parametric":             LNNTrainConfig(data = DATA_GEN_FIELD_VELOCITY,   evaluation = EVAL, network=LNN_NETWORK,            training=TRAINING, loss=LNN_LOSS_PARAMETRIC,            system=SYSTEM, visualizations=VISUALIZATION, live_learning=LIVE_LEARNING),
    "pinn":                       PINNTrainConfig(data = DATA_GEN_TRAJECTORY,      evaluation = EVAL, network=PINN_NETWORK,           training=TRAINING, loss=PINN_LOSS,                      system=SYSTEM, visualizations=VISUALIZATION, live_learning=LIVE_LEARNING),
}
