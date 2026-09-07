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
from systems.mass_spring_damped.data import make_data
from systems.mass_spring_damped.mass_spring_damped_visualizations import make_live_plot, make_animation
from systems.mass_spring_damped import pinn as msd_pinn
from systems.mass_spring_damped.hnn import (
    HAMILTONIAN, make_hamiltonian_parametric, make_hamiltonian_port_hamiltonian,
)
from systems.mass_spring_damped.lnn import (
    LAGRANGIAN, make_lagrangian_parametric, make_lagrangian_learned_dissipation,
)


# Global ────────────────────────────────────────────
SYSTEM = SystemConfig(
    q0=1.0,
    p0=0.0,
    period=4 * np.pi,
    seed=0,
    system_params={"omega": 2.1, "gamma": 0.1323},
)

OMEGA_BOUNDS = (0.05, 5.0)
GAMMA_BOUNDS = (0.001, 2.0)
Q0_BOUNDS    = (-3.0, 3.0)
P0_BOUNDS    = (-3.0, 3.0)

HAMILTONIAN_PARAMETRIC         = make_hamiltonian_parametric(OMEGA_BOUNDS, GAMMA_BOUNDS)
HAMILTONIAN_PORT_HAMILTONIAN   = make_hamiltonian_port_hamiltonian(OMEGA_BOUNDS)
LAGRANGIAN_PARAMETRIC          = make_lagrangian_parametric(OMEGA_BOUNDS, GAMMA_BOUNDS)
LAGRANGIAN_LEARNED_DISSIPATION = make_lagrangian_learned_dissipation(OMEGA_BOUNDS)

OMEGA_GUESS = PhysicsParamsConfig(params={
    "log_omega": LearnableParam(value=inverse_bounded_param(0.5, *OMEGA_BOUNDS), learnable=True),
})

OMEGA_GAMMA_GUESS = PhysicsParamsConfig(params={
    "log_omega": LearnableParam(value=inverse_bounded_param(0.5, *OMEGA_BOUNDS), learnable=True),
    "log_gamma": LearnableParam(value=inverse_bounded_param(0.5, *GAMMA_BOUNDS), learnable=True),
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
    q_range          = (-3.0, 3.0),
    p_range          = (-3.0, 3.0),
    field_N          = 8192,
    field_sigma_y      = 0.00,
)

DATA_GEN_TRAJECTORY = DataGenerationConfig(
    make_data           = make_data,
    type                = "trajectory",
    trajectory_N        = 8192,
    trajectory_sigma_y    = 0.00,
    initial_conditions  = (SYSTEM.q0, SYSTEM.p0),
    period              = SYSTEM.period,
)

# Evaluations ────────────────────────────────────────────────────────
EVAL = EvalSpec()

# Animation ────────────────────────────────────────────────────────
VISUALIZATION = VisualizationConfig(enabled=True, horizon=3.0, n_frames=150, interval=25, save_path=True, format="mp4", make_vis=make_animation)

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
    lambda_eom        = 1.0,
    hamiltonian       = HAMILTONIAN_PARAMETRIC,
    lambda_correction = 0.05,
    physics           = OMEGA_GAMMA_GUESS,
)

HNN_LOSS_PORT_HAMILTONIAN = HNNLossConfig(
    lambda_eom        = 1.0,
    hamiltonian       = HAMILTONIAN_PORT_HAMILTONIAN,
    lambda_correction = 0.05,
    physics           = OMEGA_GUESS,
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

HNN_NETWORK_PORT_HAMILTONIAN = NetworkConfig(
    architecture = (
        {"type": "fc", "units": 64, "activation": "sin"},
        {"type": "fc", "units": 64, "activation": "tanh"},
        {"type": "fc", "units": 64, "activation": "tanh"},
        {"type": "fc", "units": 64, "activation": "sin"},
        {"type": "fc", "units": 4,  "activation": "linear"},
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
    physics           = OMEGA_GAMMA_GUESS,
)

LNN_LOSS_LEARNED_DISSIPATION = LNNLossConfig(
    lambda_eom        = 1.0,
    lagrangian        = LAGRANGIAN_LEARNED_DISSIPATION,
    lambda_correction = 3.0,
    physics           = OMEGA_GUESS,
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

LNN_NETWORK_LEARNED_DISSIPATION = NetworkConfig(
    architecture = (
        {"type": "fc", "units": 64, "activation": "softplus"},
        {"type": "fc", "units": 64, "activation": "softplus"},
        {"type": "fc", "units": 64, "activation": "softplus"},
        {"type": "fc", "units": 2,  "activation": "linear"},
    ),
    epochs        = 5000,
    learning_rate = 1e-3,
)

# PINN ────────────────────────────────────────────────────────
PINN_LOSS = PINNLossConfig(
    lambda_data     = 10.0,
    lambda_physics  = 1.0,
    lambda_initial  = 3.0,
    collocation     = CollocationConfig(n_points=200, x_min=0.0, x_max=EVAL.horizons[0] * SYSTEM.period),
    physics         = PhysicsParamsConfig(params={
        "log_omega": LearnableParam(value=inverse_bounded_param(0.5, *OMEGA_BOUNDS), learnable=True),
        "log_gamma": LearnableParam(value=inverse_bounded_param(0.5, *GAMMA_BOUNDS), learnable=True),
        "q0":    LearnableParam(value=inverse_bounded_param(SYSTEM.q0, *Q0_BOUNDS), learnable=True),
        "p0":    LearnableParam(value=inverse_bounded_param(SYSTEM.p0, *P0_BOUNDS), learnable=True),
    }),
    residual_fn     = msd_pinn.MassSpringDampedPINNLoss,
    causal_epsilon     = 0.0,
    bounds          = {"omega": OMEGA_BOUNDS, "gamma": GAMMA_BOUNDS, "q0": Q0_BOUNDS, "p0": P0_BOUNDS},
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
    learning_rate = 1e-3,
    schedule      = "plateau"
)


# Experiments ──────────────────────────────────────────────────────────────
CONFIGS = {
    "mlp":                     MLPTrainConfig(data = DATA_GEN_FIELD,      evaluation = EVAL, network=MLP_NETWORK,                     training=TRAINING, loss=MLP_LOSS,                        system=SYSTEM, visualizations=VISUALIZATION, live_learning=LIVE_LEARNING),
    "mlp_trajectory":          MLPTrainConfig(data = DATA_GEN_TRAJECTORY, evaluation = EVAL, network=MLP_NETWORK_TRAJECTORY,          training=TRAINING, loss=MLP_LOSS,                        system=SYSTEM, visualizations=VISUALIZATION, live_learning=LIVE_LEARNING),
    "hnn":                     HNNTrainConfig(data = DATA_GEN_FIELD,      evaluation = EVAL, network=HNN_NETWORK,                     training=TRAINING, loss=HNN_LOSS,                        system=SYSTEM, visualizations=VISUALIZATION, live_learning=LIVE_LEARNING),
    "hnn_parametric":          HNNTrainConfig(data = DATA_GEN_FIELD,      evaluation = EVAL, network=HNN_NETWORK,                     training=TRAINING, loss=HNN_LOSS_PARAMETRIC,             system=SYSTEM, visualizations=VISUALIZATION, live_learning=LIVE_LEARNING),
    "hnn_port_hamiltonian":    HNNTrainConfig(data = DATA_GEN_FIELD,      evaluation = EVAL, network=HNN_NETWORK_PORT_HAMILTONIAN,    training=TRAINING, loss=HNN_LOSS_PORT_HAMILTONIAN,       system=SYSTEM, visualizations=VISUALIZATION, live_learning=LIVE_LEARNING),
    "lnn":                     LNNTrainConfig(data = DATA_GEN_FIELD,      evaluation = EVAL, network=LNN_NETWORK,                     training=TRAINING, loss=LNN_LOSS,                        system=SYSTEM, visualizations=VISUALIZATION, live_learning=LIVE_LEARNING),
    "lnn_parametric":          LNNTrainConfig(data = DATA_GEN_FIELD,      evaluation = EVAL, network=LNN_NETWORK,                     training=TRAINING, loss=LNN_LOSS_PARAMETRIC,             system=SYSTEM, visualizations=VISUALIZATION, live_learning=LIVE_LEARNING),
    "lnn_learned_dissipation": LNNTrainConfig(data = DATA_GEN_FIELD,      evaluation = EVAL, network=LNN_NETWORK_LEARNED_DISSIPATION, training=TRAINING, loss=LNN_LOSS_LEARNED_DISSIPATION,    system=SYSTEM, visualizations=VISUALIZATION, live_learning=LIVE_LEARNING),
    "pinn":                    PINNTrainConfig(data = DATA_GEN_TRAJECTORY, evaluation = EVAL, network=PINN_NETWORK,                   training=TRAINING, loss=PINN_LOSS,                       system=SYSTEM, visualizations=VISUALIZATION, live_learning=LIVE_LEARNING),
}
