import numpy as np
import jax
import jax.numpy as jnp

from model.physics_loss import PhysicsLoss
from model.physics_loss import bounded_param


# PINN ────────────────────────────────────────────────────────────
class MassSpringPINNLoss(PhysicsLoss):
    def __init__(self, loss_config):
        self.loss_config = loss_config
        self.t_max = loss_config.collocation.x_max
        rng        = np.random.default_rng(0)
        self.t_collocation = np.sort(
            rng.uniform(loss_config.collocation.x_min, loss_config.collocation.x_max,
                        size=loss_config.collocation.n_points)
        ).astype(np.float32)
        self.omega_bounds = loss_config.bounds["omega"]
        self.q0_bounds     = loss_config.bounds["q0"]
        self.p0_bounds     = loss_config.bounds["p0"]
        self.causal_epsilon = loss_config.causal_epsilon
        self.bins           = loss_config.collocation.bins
        if loss_config.collocation.n_points % self.bins:
            raise ValueError("collocation.n_points must be divisible by collocation.bins")

    @property
    def loss_names(self):
        return ("data", "physics", "initial")

    def predict_q(self, params, forward_fn, t):
        t_hat = t / self.t_max
        return forward_fn(params["net"], t_hat[None, None])[0, 0]

    def q_ddot(self, q, physics):
        omega = bounded_param(physics["log_omega"], *self.omega_bounds)
        return -omega ** 2 * q

    def trajectory(self, params, forward_fn):
        def q(t):
            t_hat = t / self.t_max
            return forward_fn(params["net"], t_hat[None, None])[0]
        qdot = jax.jacfwd(q)
        def field(t):
            return jnp.concatenate([q(t), qdot(t)])
        return field

    def __call__(self, params, X, y, loss_options, forward_fn, epoch=0):
        physics = params["physics"]

        y_pred    = forward_fn(params["net"], X / self.t_max)
        data_loss = jnp.mean((y_pred - y) ** 2)

        def q(t):
            return self.predict_q(params, forward_fn, t)

        qdot  = jax.grad(q)
        qddot = jax.grad(qdot)

        def residual(t):
            return qddot(t) - self.q_ddot(q(t), physics)

        residuals_sq = jax.vmap(lambda t: residual(t) ** 2)(self.t_collocation)
        binned       = residuals_sq.reshape(self.bins, -1).mean(axis=1)
        cumulative   = jnp.cumsum(binned) - binned
        weights      = jax.lax.stop_gradient(jnp.exp(-self.causal_epsilon * cumulative))
        physics_loss = jnp.sum(weights * binned) / (jnp.sum(weights) + 1e-12)

        q0 = bounded_param(physics["q0"], *self.q0_bounds)
        p0 = bounded_param(physics["p0"], *self.p0_bounds)
        t0 = jnp.array(0.0)
        initial_loss = (q(t0) - q0) ** 2 + (qdot(t0) - p0) ** 2

        return jnp.array([data_loss, physics_loss, initial_loss])

    def val_call(self, params, X, y, loss_options, forward_fn, epoch=0):
        y_pred    = forward_fn(params["net"], X / self.t_max)
        data_loss = jnp.mean((y_pred - y) ** 2)
        return jnp.array([data_loss, 0.0, 0.0])

    def describe_physics(self, params):
        physics = params["physics"]
        omega   = float(bounded_param(physics["log_omega"], *self.omega_bounds))
        q0      = float(bounded_param(physics["q0"], *self.q0_bounds))
        p0      = float(bounded_param(physics["p0"], *self.p0_bounds))
        return {"omega": omega, "q0": q0, "p0": p0}
