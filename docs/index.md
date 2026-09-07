# HNNs vs LNNs vs PINNs

A JAX framework for comparing three approaches to learning the dynamics of physical systems. The physics-informed neural network (PINN), Hamiltonian neural network (HNN) and Lagrangian neural network (LNN) each provide a systematic way of building the dynamics into a neural network. PINNs enforce the governing differential equation through the loss function, whereas HNNs and LNNs parameterize the Hamiltonian and Lagrangian respectively as the network output, from which the dynamics can be obtained. 

The goal is to quantify the advantages, disadvantages and tradeoffs for each framework by modeling physical systems and dynamical processes. 

<!-- BEGIN:hero -->
_No animation found for mass_spring / pinn. Run `train.py mass_spring pinn` to generate one._
<!-- END:hero -->

| Method | Physics is... |
|---|---|
| **[MLP](methods/mlp.md)**   | absent, pure data-driven baseline |
| **[PINN](methods/pinn.md)** | a soft loss penalty |
| **[HNN](methods/hnn.md)**   | a hard constraint on canonical coordinates `(q, p)` |
| **[LNN](methods/lnn.md)**   | a hard constraint on generalized coordinates `(q, q̇)` |

## Overview

**[MLP](methods/mlp.md)**. This is a standard feedforward network that directly maps the system state to its time derivative. It serves as a control group, providing a baseline for evaluating the benefits of incorporating the physical structure into a neural network.

**[PINN](methods/pinn.md)**. A PINN introduces a soft penalty into the loss function of a standard MLP, making it able to balance the data with known physics. The physics is encoded through a residual term containing the system's governing differential equation, which is evaluated at a set of collocation points. This allows PINNs to incorporate physical constraints even where no training data are available, making them particularly well suited to problems with sparse or incomplete data and enabling predictions beyond the temporal domain covered by the training data.  

**[HNN](methods/hnn.md)**. An HNN outputs a single scalar corresponding to the Hamiltonian `H(q, p)`, taking the canonical coordinates `(q, p)` as inputs. The system dynamics are then obtained from the symplectic gradient of `H`, ensuring exact energy conservation. The Hamiltonian formulation inherently preserves the learned energy for time-independent systems, providing a hard structural constraint on the learned dynamics. More generally, this makes HNNs particularly powerful for systems with known invariants or symmetries, as these constraints can be incorporated directly into the model rather than learned solely from data.

**[LNN](methods/lnn.md)**. Similar to an HNN, an LNN outputs a single scalar corresponding to the Lagrangian `L(q,q̇)`, taking the generalized coordinates and velocities as inputs, so no canonical momentum is required. The system dynamics are obtained from the Euler–Lagrange equations, embedding the variational structure of the system directly into the network. This formulation is particularly well suited to constrained systems, as constraints can be incorporated naturally through generalized coordinates or Lagrange multipliers. The main computational drawback is that evaluating the Euler–Lagrange equations requires higher-order derivatives of the network output, making LNNs generally more computationally expensive than HNNs.

### Field data and trajectory data

The methods do not all learn from the same type of data. 

**Field data** consists of independently sampled states within a phase space, with each state labeled by its true time derivative. Thus, the data takes the form $\bigl((q_i, p_i),\, (\dot q_i, \dot p_i)\bigr)$ and time never enters training. 

**Trajectory data** consists of a single solution sampled sequentially in time, giving pairs $(t_i,\, q_i)$ over a finite interval $t \in [0, T]$.


| Data | Methods | Network maps |
|---|---|---|
| Field | `mlp`, `hnn`, `hnn_parametric`, `lnn`, `lnn_parametric` | state $\rightarrow$ derivative (or scalar $H$ / $L$) |
| Trajectory | `mlp_trajectory`, `pinn` | $t \rightarrow$ state |

There are three main differences when comparing the results of a system per method. 

**The data requirements are not equivalent.** Field methods need derivative labels, which in practice means either a known model or numerically differentiated measurements. Trajectory methods need only positions over time. Thus, when an HNN outperforms a PINN, it is not a fair comparison as the HNN has access to additional information about the dynamics.

**The rollout is not equivalent.** Field methods do not output the next state directly; instead they map their input state to its corresponding time derivative, which then requires a numerical integrator to construct the trajectory over time. In contrast, the trajectory methods learn the trajectory of the system directly through mapping $t$ to the corresponding state $q(t)$.  

**The loss terms are not equivalent.** For field methods, each training sample directly constrains the dynamics within the sampled phase-space volume. In contrast, trajectory methods require initial or boundary conditions to find the solution, while collocation points enforce the governing dynamics within and beyond the observed data domain.

### Loss weighting

Every method minimizes a vector of loss terms rather than a single scalar, and the terms differ by how much physics the architecture encodes.

| Method | Loss terms |
|---|---|
| **MLP**  | `data`, `reg` |
| **PINN** | `data`, `physics`, `reg`, optional (`initial`, `boundary`) |
| **HNN**  | `eom`, `reg` (plus `correction` in hybrid mode) |
| **LNN**  | `eom`, `reg` (plus `correction` in hybrid mode) |

Each loss function is assigned a weight $\lambda_{i}$, which determines its relative contribution to the total loss during minimization. These weights can either be specified manually or learned automatically using `TrainingConfig.use_auto_lambda`. When enabled during training, Kendall uncertainty weighting is used, where each term carries a raw parameter $s_{i}$

$$
\lambda_i = \tfrac{1}{2}e^{-2 s_i}
$$

The total loss is then given by,

$$
\mathcal{L}_{\text{total}} = \frac{1}{\sum_j \lambda_j}\left( \sum_i \lambda_i \mathcal{L}_i \;+\; \sum_i s_i \right)
$$

The division by $\sum_j \lambda_j$ is applied under `stop_gradient`, so it rescales the gradient without changing its direction. The inclusion of the $\sum_i s_i$ term prevents the weights $\lambda_{i}$ from collapsing to zero and effectively switching off their corresponding loss functions. Therefore, the stationary point becomes,

$$
\frac{\partial}{\partial s_i}\Bigl(\lambda_i \mathcal{L}_i + s_i\Bigr) = -2\lambda_i \mathcal{L}_i + 1 = 0
\quad\Longrightarrow\quad
\lambda_i = \frac{1}{2\mathcal{L}_i}
$$

Thus, a term's weight ends up inversely proportional to its own loss. That is the intended behavior for multi-task learning, where a high-loss task is genuinely noisier. A physics residual ideally reaches zero, so decreasing the weight for a large physics residual may cause the optimizer to down-weight the physics term.

## How they compare

| | Physics guarantee | EOM knowledge | Applies to | Energy conservation | Coordinate requirement | Constraint handling | Data efficiency | Training stability | Tuning burden |
|---|---|---|---|---|---|---|---|---|---|
| **MLP**  | none | none | anything | none | none | none | lowest | stable | low |
| **PINN** | soft | the full equation, term by term | anything, incl. PDEs | approximate | none | soft penalties | high | $\lambda$-sensitive | high |
| **HNN**  | hard | only that `H` exists | Hamiltonian systems | exact in the learned `H` | canonical `(q, p)` | hard, through the coordinates | high | stable | low |
| **LNN**  | hard | only that `L` exists | Lagrangian systems | exact via Noether | generalized `(q, q̇)` | hard, through the coordinates | high | fragile Hessian solve | low |

## Results

Best-performing method per system by MSE, evaluated at $2\times$ the training horizon on clean data. Score is that MSE as a fraction of the zero-predictor, so 0 is perfect and 1 is no better than outputting nothing. Follow a system for the full per-method tables.

<!-- BEGIN:overview -->
| System | Best method | Score | MSE | Energy drift | Max horizon |
|---|---|---|---|---|---|
| [Charged Particle](systems/charged_particle/index.md) | — | — | — | — | — |
| [Double Pendulum](systems/double_pendulum/index.md) | — | — | — | — | — |
| [Double Pendulum (Damped)](systems/double_pendulum_damped/index.md) | — | — | — | — | — |
| [Mass Spring](systems/mass_spring/index.md) | — | — | — | — | — |
| [Mass Spring (Damped)](systems/mass_spring_damped/index.md) | — | — | — | — | — |
<!-- END:overview -->

## Getting started

The project uses [uv](https://docs.astral.sh/uv/) for dependency management and pins Python 3.10.

```bash
git clone https://github.com/MatthijsCDH/constrained-dynamics.git
cd constrained-dynamics
uv sync
```

The default dependency set includes `jax[cuda12]` and expects a CUDA-capable GPU.

Every entry point takes a system and a method, and `--list` shows what is available:

```bash
uv run python train.py --list
```

```
charged_particle: mlp, mlp_trajectory, hnn, hnn_mechanical, hnn_parametric, hnn_parametric_mechanical, lnn, lnn_parametric, pinn
double_pendulum: mlp, mlp_trajectory, hnn, hnn_parametric, lnn, lnn_parametric, pinn
double_pendulum_damped: mlp, mlp_trajectory, hnn, hnn_parametric, hnn_port_hamiltonian, lnn, lnn_parametric, lnn_learned_dissipation, pinn
mass_spring: mlp, mlp_trajectory, hnn, hnn_parametric, lnn, lnn_parametric, pinn
mass_spring_damped: mlp, mlp_trajectory, hnn, hnn_parametric, hnn_port_hamiltonian, lnn, lnn_parametric, lnn_learned_dissipation, pinn
```

There are three CLIs taking a system and a method, and each has `--list` to show the available pairs.

**`train.py`** trains one model once and shows it.

```bash
uv run python train.py mass_spring pinn --tuned
```

This run provides live loss curves and learned physics parameters when `LiveLearningConfig.enabled` is set to True, and generates an animation of the rollout against the ground truth when `VisualizationConfig.enabled` is set to True. Using `--tuned` loads hyperparameters previously found by `tune.py` instead of the defaults in the system's config file.

**`tune.py`** searches for hyperparameters by random search.

```bash
uv run python tune.py mass_spring pinn --n_trials 60 --n_seeds 5
```

Each trial samples a configuration from that system's search space and trains it on `--n_seeds` seeds, five by default. Each seed is scored by the [score](evaluation.md#score), the rollout MSE as a fraction of the zero-predictor, clipped at 1. A seed whose validation loss shows it has collapsed to the trivial solution is stopped early and assigned 1 outright, and the trial's value is the mean over its seeds. Scoring across seeds is what makes the search prefer a configuration that works reliably over one that works when lucky.

The default number of trials is set at 60 trials, which gives 95% confidence of landing in the top 5% of the search space. Results are written to `best_configs/<method>.json`, together with a `_diagnostics.json` file containing various diagnostic statistics per trial.

**`benchmark.py`** produces the numbers in the results tables.

```bash
uv run python benchmark.py mass_spring hnn --n_seeds 50 --sigmas 0.0 0.05 0.1 0.2 0.5
```

For each noise level $\sigma$, the model is trained from scratch independently for each new seed. Retraining occurs at each new seed, as the goal is to evaluate how reliably a method converges to a good solution. Each trained model is rolled out and scored for MSE against ground truth, split into the part inside the data window and the part beyond it, along with energy drift, max horizon, max radius, inference time and training time. All results are stored per method from which the plots and tables are generated. What each of those numbers measures, and how it is aggregated across seeds, is documented in the [evaluation reference](evaluation.md).

The intended order is `tune.py` first, then `benchmark.py`, as the tuned hyperparameters are automatically used within `benchmark.py`.

All the information about a system lives in one file `systems/<name>/<name>_configurations.py`. Configurations are built from frozen dataclasses, so settings are changed by editing this file.

Every field of every one of those dataclasses, what it does and what it defaults to, is documented in the [configuration reference](configurations.md). 

