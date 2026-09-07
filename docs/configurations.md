# Configuration reference

Every setting in the framework is a field on a frozen dataclass defined in `configs/configurations.py`. A run is fully described by the one `*TrainConfig` object it is handed.

Those objects are assembled per system in `systems/<name>/<name>_configurations.py`, where:

| Name | Type | Purpose |
|---|---|---|
| `SYSTEM` | `SystemConfig` | Constants shared by every file in the system |
| `EVAL` | `EvalSpec` | Evaluation horizons, extrapolation radii and the horizon threshold |
| `CONFIGS` | `dict[str, *TrainConfig]` | One entry per runnable method |

To change a setting, edit that file. To add a method, add an entry to `CONFIGS`. To add a system, copy the folder.

## Global

### `SystemConfig`

The physical constants of a system are defined once inside `SYSTEM` and referenced everywhere else.

| Setting | Type | Default | What it does |
|---|---|---|---|
| `q0` | float or array | required | Initial position, used for trajectory data, evaluation and visualization |
| `p0` | float or array | required | Initial momentum, used for trajectory data, evaluation and visualization |
| `period` | float | required | One characteristic period. Every horizon in the framework is a multiple of this, so `max_horizon = 2.0` means two periods |
| `seed` | int | required | Master seed for data generation, weight initialization and the train/validation split |
| `system_params` | dict | required | The *true* physical constants data is generated from, for example `{"omega": 2.1}`. |

---

## Data generation

### `DataGenerationConfig`

Controls what the network is trained on, where `type` selects the data type used for generation. 

| Setting | Type | Default | What it does |
|---|---|---|---|
| `make_data` | callable | required | Bound factory from the system's `data.py`, called as `config.data.make_data(config)` |
| `type` | str | `"field"` | `"field"` for state-to-derivative pairs, `"trajectory"` for time-to-state pairs. Also sets `NeuralNetwork.type`, which decides whether rollout integrates a vector field or evaluates the network directly |
| `coords` | str | `"canonical"` | `"canonical"` for `(q, p)`, `"velocity"` for `(q, q̇)`. Sets `NeuralNetwork.coords` and is passed to `true_trajectory`/`hamiltonian`. LNN needs `"velocity"`; it only matters once a system's canonical `p` and generalized `q̇` differ numerically |
| `q_range` | tuple | `(-2.0, 2.0)` | Field only. Box in position the sampled states are drawn from |
| `p_range` | tuple | `(-2.0, 2.0)` | Field only. Box in momentum |
| `field_N` | int | `8192` | Field only. Number of sampled states |
| `field_sigma_y` | float | `0.0` | Field only. Gaussian noise on the targets `(q̇, ṗ)`. **Relative**: the std used is `sigma × std(y)`, per component, so the same value means the same signal-to-noise ratio on every system. This is what `benchmark.py --sigmas` sweeps |
| `field_sigma_x` | float | `0.0` | Field only. The same, on the inputs `(q, p)`. This is errors-in-variables and it *biases* the learned vector field toward zero, which shows up as a frequency error. Off by default, sweep it as a separate experiment |
| `trajectory_N` | int | `8192` | Trajectory only. Number of time samples |
| `trajectory_sigma_y` | float | `0.0` | Trajectory only. Relative Gaussian noise on the target `q`, same convention as `field_sigma_y`. This is what `benchmark.py --sigmas` sweeps |
| `trajectory_sigma_x` | float | `0.0` | Trajectory only. The same, on the input `t`. Perturbing the timestamp *biases* the learned amplitude by $\exp(-\omega^2\sigma^2/2)$ while leaving $\omega$ unbiased, so `q0`/`p0` come out systematically small. Off by default |
| `initial_conditions` | tuple | `None` | Trajectory only. The `(q0, p0)` the solution starts from. Should reference `SYSTEM.q0`/`SYSTEM.p0` |
| `period` | float | `None` | Trajectory only. Length of the sampled window. Should reference `SYSTEM.period` |

Field and trajectory data are not interchangeable, and the difference drives most of the comparison on this site. See [Field data and trajectory data](index.md#field-data-and-trajectory-data).

---

## Network

### `NetworkConfig`

Architecture and optimizer schedule.

| Setting | Type | Default | What it does |
|---|---|---|---|
| `architecture` | tuple of dicts | required | Layer stack, for example `({"type": "fc", "units": 64, "activation": "tanh"}, ...)`. The output of the architecture must be carefully constructed according to the type of system and its degrees of freedom. |
| `epochs` | int | required | Training epochs |
| `learning_rate` | float | required | Peak learning rate, reached at the end of warmup |
| `schedule` | str | `"cosine"` | `"cosine"` for warmup then cosine decay to `end_value`. `"plateau"` for warmup then a constant rate with `optax.contrib.reduce_on_plateau` layered on |
| `end_value` | float | `1e-5` | Final learning rate of the cosine decay. Unused when `schedule="plateau"` |
| `decay_steps` | int | `None` | Cosine decay length in optimizer steps. `None` means `epochs × steps_per_epoch` |
| `warmup_steps` | int | `None` | Linear warmup length in steps. `None` means `max(200, decay_steps // 40)` |
| `plateau_factor` | float | `0.5` | Plateau only. Multiplier applied to the rate when progress stalls |
| `plateau_patience` | int | `50` | Plateau only. Epochs without improvement before reducing. `accumulation_size` is set to `steps_per_epoch`, so this really does count epochs rather than steps |
| `plateau_cooldown` | int | `20` | Plateau only. Epochs to wait after a reduction before measuring again |
| `plateau_rtol` | float | `1e-3` | Plateau only. Relative improvement below this counts as no progress |
| `plateau_min_scale` | float | `1e-3` | Plateau only. Floor on the cumulative reduction, so the rate cannot collapse to zero |
| `save_filepath` | str | `None` | Where to write weights after training. `None` disables saving |
| `load_filepath` | str | `None` | Weights to restore before training. `None` starts from a fresh initialization |

Gradient clipping at global norm `1.0` and AdamW weight decay of `1e-5` are applied unconditionally and are not configurable.

### `TrainingConfig`

| Setting | Type | Default | What it does |
|---|---|---|---|
| `batch_size` | int | required | Minibatch size. Must satisfy `val_split × N_data ≥ batch_size` |
| `val_split` | float | `0.8` | Fraction kept for **training**. `0.8` means 80% train and 20% validation |
| `do_validation` | bool | `True` | Whether to compute the validation pass, which calls `val_call` and so skips collocation and physics residuals |
| `lambda_reg` | float | `0.0` | Initial weight on the L2 penalty over network weights, always the last loss term |
| `use_auto_lambda` | bool | `False` | Whether the loss weights are learned by Kendall uncertainty weighting or frozen at their initial values |
| `live_metrics` | bool | `True` | Per-epoch progress line during training |
| `static_metrics` | bool | `True` | Summary banner printed at the start and end of the run |

---

## Loss

Each method has a loss config. Every `lambda_*` field is the **initial** value of that term's weight, converted internally to $s_i = -\tfrac{1}{2}\log(2\lambda_i)$. Whether the weights then move is controlled by `use_auto_lambda`. See [Loss weighting](index.md#loss-weighting).

A `lambda_*` field left at `None` means that loss term is not present for this run and no weight is allocated for it.

### `MLPLossConfig`

| Setting | Type | Default | What it does |
|---|---|---|---|
| `lambda_data` | float | `1.0` | Initial weight on the data term |
| `augment_trajectory_derivative` | bool | `False` | Trajectory MLPs only. When `True`, rollout reconstructs `q̇` by autodiff on the network's own output instead of reading it from a second output unit. Affects rollout only, not the training data |

### `HNNLossConfig`

| Setting | Type | Default | What it does |
|---|---|---|---|
| `lambda_eom` | float | required | Initial weight on the equation-of-motion term. Despite the name this is a supervised fit against the same derivative labels an MLP uses; the physics lives in the architecture, not in this term |
| `hamiltonian` | `HamiltonianConfig` | required | The architecture description below |
| `lambda_correction` | float | `None` | Hybrid mode only. Initial weight on a penalty holding the black-box correction small, so the known term is preferred where it fits |
| `physics` | `PhysicsParamsConfig` | `None` | Learnable or fixed constants used by `known_term` |

### `HamiltonianConfig`

| Setting | Type | Default | What it does |
|---|---|---|---|
| `n_dof` | int | required | Degrees of freedom. The network input is `2 × n_dof` |
| `dissipation` | str | `"none"` | `"none"` for conservative Hamilton's equations, `"rayleigh"` for a scalar damping term, `"port_hamiltonian"` for `ẋ = (J − R)∇H` with `R = LLᵀ` read off a second slice of the same network |
| `integrator` | str | `"rk4"` | Rollout integrator. `"leapfrog"` is symplectic and available only to HNN, since it needs the canonical split that LNN's coordinates do not provide |
| `known_term` | callable | `None` | `known_term(state, physics_params) -> scalar`. When set, `H = known_term(...) + network(...)`. When `None`, `H` is the pure black-box network |
| `dissipation_term` | callable | `None` | Known functional form for the dissipative part, same hybrid pattern as `known_term` |
| `describe` | callable | `None` | Formats learned physics parameters for printing after training |
| `penalize_correction` | bool | `False` | Enables the `correction` loss term. Ignored unless `known_term` is set |

### `LNNLossConfig`

| Setting | Type | Default | What it does |
|---|---|---|---|
| `lambda_eom` | float | required | Initial weight on the Euler-Lagrange term |
| `lagrangian` | `LagrangianConfig` | required | The architecture description below |
| `lambda_correction` | float | `None` | Hybrid mode only, as for HNN |
| `physics` | `PhysicsParamsConfig` | `None` | Learnable or fixed constants used by `known_term` |

### `LagrangianConfig`

| Setting | Type | Default | What it does |
|---|---|---|---|
| `n_dof` | int | required | Degrees of freedom. The network input is `2 × n_dof`, taken as `(q, q̇)` |
| `dissipation` | str | `"none"` | `"none"`, `"rayleigh"`, or `"learned_dissipation"` for `D(q, q̇) = ½ q̇ᵀR(q)q̇` with `R` obtained by evaluating the same network with the `q̇` slot zeroed |
| `known_term` | callable | `None` | Same hybrid pattern as HNN, applied to `L` |
| `dissipation_term` | callable | `None` | Known functional form for the dissipative part |
| `describe` | callable | `None` | Formats learned physics parameters for printing |
| `penalize_correction` | bool | `False` | Enables the `correction` loss term. Ignored unless `known_term` is set |

### `PINNLossConfig`

| Setting | Type | Default | What it does |
|---|---|---|---|
| `collocation` | `CollocationConfig` | required | Where the residual is enforced |
| `physics` | `PhysicsParamsConfig` | required | Constants appearing in the residual |
| `residual_fn` | callable | required | The system's own `PhysicsLoss` subclass, instantiated as `residual_fn(loss_config)` |
| `lambda_data` | float | `None` | Initial weight on the fit to observations |
| `lambda_physics` | float | `None` | Initial weight on the residual of the governing equation |
| `lambda_initial` | float | `None` | Initial weight on the initial-condition term |
| `lambda_boundary` | float | `None` | Initial weight on the spatial boundary term. Unused by the ODE systems; reserved for PDEs |
| `causal_epsilon` | float | `0.0` | Strength of causal weighting, which down-weights a collocation bin by how much residual is still unresolved before it. `0.0` disables it. Held constant during training: the curriculum anneals itself as the residuals fall, so no schedule is needed |
| `bounds` | dict | `{}` | Per-parameter `(min, max)` for learnable physics constants, applied through a sigmoid so a parameter cannot leave its range |

### `CollocationConfig`

| Setting | Type | Default | What it does |
|---|---|---|---|
| `n_points` | int | required | Number of collocation points, drawn once at construction |
| `x_min` | float | required | Start of the domain the residual is enforced on |
| `x_max` | float | required | End of that domain. Extending it past the data window is what lets a PINN extrapolate, and it also sets the normalization constant the network's input is divided by |
| `bins` | int | `10` | Number of equal-count time bins the residual is averaged into before causal weighting. Must divide `n_points`. Decouples `causal_epsilon` from `n_points`, so a tuned $\varepsilon$ transfers between systems with different collocation counts |

---

## Physics parameters

Shared by PINN, HNN and LNN. Every declared entry lands in `params["physics"][name]` whether or not it is learnable; a fixed entry is simply one the optimizer is never allowed to move, via `optax.set_to_zero()`, which is equivalent to hardcoding the constant.

### `LearnableParam`

| Setting | Type | Default | What it does |
|---|---|---|---|
| `value` | float | required | Initial value. When `bounds` apply, this is the *unconstrained* pre-sigmoid value, so build it with `inverse_bounded_param(target, lo, hi)` rather than writing the physical number directly |
| `learnable` | bool | `False` | Whether the optimizer may move it |

### `PhysicsParamsConfig`

| Setting | Type | Default | What it does |
|---|---|---|---|
| `params` | dict | required | `{name: LearnableParam(...)}`. Names must match what `known_term` or the residual reads |

---

## Evaluation

### `EvalSpec`

| Setting | Type | Default | What it does |
|---|---|---|---|
| `horizons` | tuple | `(2, 5, 10, 20, 50)` | Rollout lengths in periods. `horizons[0]` is the headline horizon used by `train.py` and by the results tables; `max(horizons)` is the ceiling `max_horizon` is censored at |
| `horizon_tolerance` | float | `0.01` | How far a rollout may degrade before it stops counting as reliable, as a fraction of the zero-predictor MSE. The threshold is `baseline + tolerance × zero_mse`, where `baseline` is the model's own mean error over the first period, so the column is comparable across systems and stays readable as $\sigma$ rises |
| `horizon_window` | float | `0.25` | Width, in periods, of the rolling window the error is averaged over before the threshold is applied. Smoothing makes the crossing robust to single-sample spikes; the cost is that nothing below `horizon_window` periods is reportable |
| `radii` | tuple | `(0.5, 1.0, 1.5, 2.0, 3.0)` | Initial-condition scalings used for state-space extrapolation. Each rollout starts from `(r·q0, r·p0)`, and `max_radius` is the largest `r` still tracking. Field methods only, since a trajectory method has its initial condition baked into its weights |

---

## Visualization

### `VisualizationConfig`

The rollout animations produced after training finishes.

| Setting | Type | Default | What it does |
|---|---|---|---|
| `enabled` | bool | `False` | Whether to build the animation at all |
| `horizon` | float | `3.0` | How many periods the rollout covers |
| `n_frames` | int | `400` | Frames rendered. The dominant cost, in both runtime and file size |
| `interval` | int | `25` | Milliseconds per frame on screen. Also sets the saved frame rate, as `1000 / interval` |
| `save_path` | bool | `False` | Whether to write the file. The path itself is derived from the system and method, landing in `systems/<system>/<family>/plots/` |
| `format` | str | `"mp4"` | `"mp4"` or `"gif"`. mp4 is roughly ten times smaller for the same content and gives the viewer playback controls |
| `dpi` | int | `100` | Resolution of the saved file |
| `make_vis` | callable | `None` | Bound factory from the system's visualizations module |

### `LiveLearningConfig`

The plot that drives training itself, one `train_epoch()` per animation frame, so the fit can be watched as it forms.

| Setting | Type | Default | What it does |
|---|---|---|---|
| `enabled` | bool | `False` | When `True`, `train.py` calls `live_learning_plot(model)` **instead of** `model.train()` |
| `n_points` | int | `200` | Resolution of the predicted curve |
| `plot_every` | int | `1` | Epochs between redraws. Redrawing every epoch throttles training to matplotlib's render speed, roughly 5 it/s against 300 it/s undecorated, so raise this for real runs |
| `horizon` | float | `3.0` | How far past the training domain the plot extends, in periods |
| `log_scale` | bool | `True` | Log axis on the loss panels |
| `save_path` | bool | `False` | Whether to write the result, again to a path derived from system and method |
| `format` | str | `"mp4"` | `"png"` saves a single snapshot of the final frame. `"gif"` or `"mp4"` captures every redrawn frame and writes an animation of the whole training run |
| `make_live_plot` | callable | `None` | Bound factory from the system's visualizations module |

Closing the live window stops training early and still saves.

---

## Assembling a run

The four `*TrainConfig` classes, `MLPTrainConfig`, `HNNTrainConfig`, `LNNTrainConfig` and `PINNTrainConfig`, differ only in which loss config they carry. The method is identified by the config type, so `HNNTrainConfig` is what selects the HNN code path.

| Setting | Type | Default | What it does |
|---|---|---|---|
| `data` | `DataGenerationConfig` | required | |
| `evaluation` | `EvalSpec` | required | |
| `network` | `NetworkConfig` | required | |
| `training` | `TrainingConfig` | required | |
| `loss` | one of the loss configs | required | Determines the method |
| `system` | `SystemConfig` | required | |
| `visualizations` | `VisualizationConfig` | `None` | `None` disables the post-hoc animation |
| `live_learning` | `LiveLearningConfig` | `None` | `None` falls back to a plain `model.train()` |

Presets are shared by reference across methods, so one edit propagates to every entry using it:

```python
CONFIGS = {
    "hnn":  HNNTrainConfig(data=DATA_GEN_FIELD, evaluation=EVAL, network=HNN_NETWORK,
                           training=TRAINING, loss=HNN_LOSS, system=SYSTEM,
                           visualizations=VISUALIZATION, live_learning=LIVE_LEARNING),
    "pinn": PINNTrainConfig(data=DATA_GEN_TRAJECTORY, evaluation=EVAL, network=PINN_NETWORK,
                            training=TRAINING, loss=PINN_LOSS, system=SYSTEM,
                            visualizations=VISUALIZATION, live_learning=LIVE_LEARNING),
}
```

Settings found by `tune.py` are not written back into this file. They are stored as JSON in `systems/<system>/<family>/best_configs/<method>.json` and reapplied on top of the config above when you pass `--tuned`, or automatically by `benchmark.py`. The per-method settings tables in the system pages report whichever of the two a run actually used.
