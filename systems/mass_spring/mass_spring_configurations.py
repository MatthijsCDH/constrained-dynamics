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
from systems.mass_spring.data import make_data
from systems.mass_spring.mass_spring_visualizations import make_live_plot, make_animation
from systems.mass_spring import pinn as ms_pinn
from systems.mass_spring.hnn import HAMILTONIAN, make_hamiltonian_parametric
from systems.mass_spring.lnn import LAGRANGIAN, make_lagrangian_parametric


# Global ────────────────────────────────────────────
SYSTEM = SystemConfig(
    q0=1.0,
    p0=0.0,
    period=4 * np.pi,
    seed=0,
    system_params={"omega": 2.1},
)

OMEGA_BOUNDS = (0.05, 5.0)
Q0_BOUNDS    = (-3.0, 3.0)
P0_BOUNDS    = (-3.0, 3.0)

HAMILTONIAN_PARAMETRIC = make_hamiltonian_parametric(OMEGA_BOUNDS)
LAGRANGIAN_PARAMETRIC  = make_lagrangian_parametric(OMEGA_BOUNDS)

# Training presets ────────────────────────────────────────────────────────
TRAINING = TrainingConfig(
    batch_size      = 128,
    val_split       = 0.8,
    do_validation   = True,
    lambda_reg      = 0.17414134181586206e-06,
    use_auto_lambda = False,
)

# Data generation ────────────────────────────────────────────────────────
DATA_GEN_FIELD = DataGenerationConfig(
    make_data        = make_data,
    type             = "field",
    q_range          = (-3.0, 3.0),
    p_range          = (-3.0, 3.0),
    field_N          = 8192 ,
    field_sigma_y    = 0.00,
)

DATA_GEN_TRAJECTORY = DataGenerationConfig(
    make_data           = make_data,
    type                = "trajectory",
    trajectory_N        = 8192,
    trajectory_sigma_y  = 0.40,
    initial_conditions  = (SYSTEM.q0, SYSTEM.p0),
    period              = SYSTEM.period,
)

# Evaluations ────────────────────────────────────────────────────────
EVAL = EvalSpec()

# Animation ────────────────────────────────────────────────────────
VISUALIZATION = VisualizationConfig(enabled=True, horizon=3.0, n_frames=1000, interval=10, save_path=True, format="mp4", make_vis=make_animation)

# Live learning ────────────────────────────────────────────────────────
LIVE_LEARNING = LiveLearningConfig(enabled=True, n_points=200, plot_every=100, horizon=3.0, save_path=True, format="mp4", make_live_plot=make_live_plot)


# MLP ─────────────────────────────────────────────────────────────────
MLP_LOSS = MLPLossConfig(augment_trajectory_derivative=True)

MLP_NETWORK = NetworkConfig(
    architecture = (
        {"type": "fc", "units": 64, "activation": "sin"},
        {"type": "fc", "units": 64, "activation": "tanh"},
        {"type": "fc", "units": 64, "activation": "tanh"},
        {"type": "fc", "units": 64, "activation": "sin"},
        {"type": "fc", "units": 2,  "activation": "linear"},
    ),
    epochs        = 5000,
    learning_rate = 4.6e-4,
)

MLP_NETWORK_TRAJECTORY = NetworkConfig(
    architecture = (
        {"type": "fc", "units": 64, "activation": "sin"},
        {"type": "fc", "units": 64, "activation": "tanh"},
        {"type": "fc", "units": 64, "activation": "tanh"},
        {"type": "fc", "units": 64, "activation": "sin"},
        {"type": "fc", "units": 1,  "activation": "linear"},
    ),
    epochs        = 5000,
    learning_rate = 1e-3,
    schedule      = "plateau"
)

# HNN ────────────────────────────────────────────────────────
HNN_LOSS = HNNLossConfig(
    lambda_eom  = 1.0,
    hamiltonian = HAMILTONIAN,
)

HNN_LOSS_PARAMETRIC = HNNLossConfig(
    lambda_eom  = 1.0,
    hamiltonian = HAMILTONIAN_PARAMETRIC,
    lambda_correction = 0.05,
    physics     = PhysicsParamsConfig(params={
        "log_omega": LearnableParam(value=inverse_bounded_param(0.5, *OMEGA_BOUNDS), learnable=True),
    }),
)

HNN_NETWORK = NetworkConfig(
    architecture = (
        {"type": "fc", "units": 64, "activation": "sin"},
        {"type": "fc", "units": 64, "activation": "tanh"},
        {"type": "fc", "units": 64, "activation": "tanh"},
        {"type": "fc", "units": 64, "activation": "sin"},
        {"type": "fc", "units": 1,  "activation": "linear"},
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
    lambda_eom = 1.0,
    lagrangian = LAGRANGIAN_PARAMETRIC,
    lambda_correction = 3.0,
    physics    = PhysicsParamsConfig(params={
        "log_omega": LearnableParam(value=inverse_bounded_param(0.5, *OMEGA_BOUNDS), learnable=True),
    }),
)

LNN_NETWORK = NetworkConfig(
    architecture = (
        {"type": "fc", "units": 64, "activation": "softplus"},
        {"type": "fc", "units": 64, "activation": "softplus"},
        {"type": "fc", "units": 64, "activation": "softplus"},
        {"type": "fc", "units": 1,  "activation": "linear"},
    ),
    epochs        = 5000,
    learning_rate = 1e-3,
)
# PINN ────────────────────────────────────────────────────────
PINN_LOSS = PINNLossConfig(
    lambda_data     = 0.10006913513545575,
    lambda_physics  = 0.10278118224157009,
    lambda_initial  = 0.0042770830499620704,
    collocation     = CollocationConfig(n_points=200, x_min=0.0, x_max=EVAL.horizons[0] * SYSTEM.period),
    physics         = PhysicsParamsConfig(params={
        "log_omega": LearnableParam(value=inverse_bounded_param(0.5, *OMEGA_BOUNDS), learnable=True),
        "q0":    LearnableParam(value=inverse_bounded_param(SYSTEM.q0, *Q0_BOUNDS), learnable=True),
        "p0":    LearnableParam(value=inverse_bounded_param(SYSTEM.p0, *P0_BOUNDS), learnable=True),
    }),
    residual_fn     = ms_pinn.MassSpringPINNLoss,
    causal_epsilon  = 0.3278726498335276,
    bounds          = {"omega": OMEGA_BOUNDS, "q0": Q0_BOUNDS, "p0": P0_BOUNDS},
)

PINN_NETWORK = NetworkConfig(
    architecture = (
        {"type": "fc", "units": 64, "activation": "sin"},
        {"type": "fc", "units": 64, "activation": "tanh"},
        {"type": "fc", "units": 64, "activation": "tanh"},
        {"type": "fc", "units": 64, "activation": "sin"},
        {"type": "fc", "units": 1,  "activation": "linear"},
    ),
    epochs        = 5000,
    learning_rate = 0.00248463381689824,
    schedule      = "plateau"
)


# Experiments ──────────────────────────────────────────────────────────────
CONFIGS = {
    "mlp":            MLPTrainConfig(data = DATA_GEN_FIELD,       evaluation = EVAL, network=MLP_NETWORK,            training=TRAINING, loss=MLP_LOSS,            system=SYSTEM, visualizations=VISUALIZATION, live_learning=LIVE_LEARNING),
    "mlp_trajectory": MLPTrainConfig(data = DATA_GEN_TRAJECTORY,  evaluation = EVAL, network=MLP_NETWORK_TRAJECTORY, training=TRAINING, loss=MLP_LOSS,            system=SYSTEM, visualizations=VISUALIZATION, live_learning=LIVE_LEARNING),
    "hnn":            HNNTrainConfig(data = DATA_GEN_FIELD,       evaluation = EVAL, network=HNN_NETWORK,            training=TRAINING, loss=HNN_LOSS,            system=SYSTEM, visualizations=VISUALIZATION, live_learning=LIVE_LEARNING),
    "hnn_parametric": HNNTrainConfig(data = DATA_GEN_FIELD,       evaluation = EVAL, network=HNN_NETWORK,            training=TRAINING, loss=HNN_LOSS_PARAMETRIC, system=SYSTEM, visualizations=VISUALIZATION, live_learning=LIVE_LEARNING),
    "lnn":            LNNTrainConfig(data = DATA_GEN_FIELD,       evaluation = EVAL, network=LNN_NETWORK,            training=TRAINING, loss=LNN_LOSS,            system=SYSTEM, visualizations=VISUALIZATION, live_learning=LIVE_LEARNING),
    "lnn_parametric": LNNTrainConfig(data = DATA_GEN_FIELD,       evaluation = EVAL, network=LNN_NETWORK,            training=TRAINING, loss=LNN_LOSS_PARAMETRIC, system=SYSTEM, visualizations=VISUALIZATION, live_learning=LIVE_LEARNING),
    "pinn":           PINNTrainConfig(data = DATA_GEN_TRAJECTORY, evaluation = EVAL, network=PINN_NETWORK,           training=TRAINING, loss=PINN_LOSS,           system=SYSTEM, visualizations=VISUALIZATION, live_learning=LIVE_LEARNING),
}
