# Evaluation reference

Each trained model is rolled out from the true initial state to **2 periods**, set by `EvalSpec.horizons[0]`, and compared against the system's own `true_trajectory`. This page defines what each column measures. The results are given in the tables on the corresponding system page and come from running `benchmark.py`.

## MSE

The squared error between the rolled-out trajectory and the true one, averaged over the state components and then over the rollout,

```math
\text{MSE} = \frac{1}{N_t}\sum_{k=1}^{N_t} \bigl\lVert x_\theta(t_k) - x_{\text{true}}(t_k) \bigr\rVert^2 .
```

It compares positions at matched times, making it sensitive to phase shifts. 

## MSE in-domain and MSE beyond T

The same quantity, split at one period. **In-domain** averages over $t \le T$ and **beyond T** over $t > T$, using `Metrics.mse_split`.

For PINN and `mlp_trajectory`, the training data covers $[0, T]$, so beyond $T$ only the physics residual constrains the solution. For the field methods there is no temporal training domain at all, so the split is simply early against late rollout, and the two halves differ because integration error accumulates.

## Score

The score is a normalized mean squared error clipped to a maximum of `1`. The score shows how well the model reconstructs the true trajectory, where `0` is exact and `1` is no better than predicting zero.

```math
\text{score} = \min\left(\frac{\text{MSE}}{\overline{\lVert x_{\text{true}} \rVert^2}},\; 1\right),
\qquad
\overline{\lVert x_{\text{true}} \rVert^2} = \frac{1}{N_t}\sum_k \lVert x_{\text{true}}(t_k)\rVert^2 .
```

The denominator is a property of the system and horizon, not of the model, so the same number means the same thing on every system and at every noise level. This scoring is used in `tune.py` to measure the accuracy of a model.

## Energy drift

How far the model's trajectory drifts from the true energy at the same instant, normalized by the initial energy and averaged over the rollout,

```math
\Delta E = \frac{1}{N_t}\sum_{k=1}^{N_t} \frac{\bigl|E_\theta(t_k) - E_{\text{true}}(t_k)\bigr|}{\bigl|E_{\text{true}}(0)\bigr|} .
```

$E_\theta(t_k)$ is the *true* Hamiltonian evaluated on the states the model predicted, not the model's own learned Hamiltonian. For an HNN or LNN the learned Hamiltonian is conserved by construction, so it would not measure whether the trajectory stays on the correct energy surface.

On a conservative system $E_{\text{true}}(t_k) = E_{\text{true}}(0)$, so the two are the same number. For dissipative systems $E_{\text{true}}(t_k)$ decays and is no longer equal to the initial energy. 

## Max horizon

How far the rollout stays reliable. The pointwise error is averaged over a rolling window of `EvalSpec.horizon_window` periods, where the default is set at `0.25`, and the horizon is the first time that window mean exceeds the following threshold,

```math
\text{threshold} = \frac{1}{\bigl|{k : t_k \le T}\bigr|} \sum_{t_k \le T} \bigl\lVert x_\theta(t_k) - x_{\text{true}}(t_k) \bigr\rVert^2 + h \overline{\lVert x_{\text{true}} \rVert^2}
```

where the first term is the mean error over the first period. The second term corresponds to a fraction `horizon_tolerance` $h$ of the true mean. The window is aligned to its right edge, so the reported time is where the trailing quarter-period first exceeds the threshold. 

The threshold is relative, so a system whose signal is weak is not held to a tighter standard than one whose signal is strong. And it is measured against the model's own early accuracy, so a model at its noise floor is not declared broken for sitting there. At $\sigma$ = 0 the baseline term is negligible and the criterion reduces to a plain $h = 0.01$ of the zero-predictor.

The search runs over `EvalSpec.horizons`, `(2, 5, 10, 20, 50)`, extending the rollout only when the shorter one never crossed. 

## Max radius

The state-space extrapolation is only valid for methods that use `data_type = field`.

Field methods learn a vector field over the box set by `q_range` and `p_range`, and a rollout from the system's own initial condition may never leave that box. The max radius probes outside the box, where for each `r` in `EvalSpec.radii` the model is rolled out from $(r q_0,\ r p_0)$ and scored against the true trajectory from the same start, normalized by that trajectory's own zero-predictor. The reported value is the largest `r` whose score stays under `horizon_tolerance`.

PINN and `mlp_trajectory` report `n/a`, as these never learn a vector field; they map time directly to the state.

## Inference (ms)

Wall-clock time for one complete rollout, measured after warmup runs so that JIT compilation is excluded. This is the cost of the whole trajectory rather than of a single step.

## Train (s)

Wall-clock time to train one seed, including compilation. 

## Breakdown $\sigma$

The noise level at which a method stops being useful, defined as the $\sigma$ at which the score first exceeds `0.1`, interpolated in $\log_{10}$ between the two swept levels.

Reading it as "the noise level at which this method gets more than 10% of the way to useless" is exact. A `> 0.4` entry means the score never crossed within the swept range, so the value is a lower bound.

## Learned parameters

For methods with learnable physics parameters, the value recovered after training and its relative error against the true constant in `SystemConfig.system_params`,

```math
\text{error} = \frac{|\theta_{\text{learned}} - \theta_{\text{true}}|}{|\theta_{\text{true}}|} .
```

Unlike the other metrics, this one is not a property of the trajectory. Instead it is a direct measure on the physics predicted by the model.
