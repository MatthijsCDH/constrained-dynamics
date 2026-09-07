import numpy as np
import jax
import jax.numpy as jnp

from model.physics_loss import PhysicsLoss
from model.physics_loss import bounded_param


# PINN ────────────────────────────────────────────────────────────
class ChargedParticlePINNLoss(PhysicsLoss):
    def __init__(self, loss_config):
        self.loss_config = loss_config
        self.t_max = loss_config.collocation.x_max
        rng        = np.random.default_rng(0)
        self.t_collocation = np.sort(
            rng.uniform(loss_config.collocation.x_min, loss_config.collocation.x_max,
                        size=loss_config.collocation.n_points)
        ).astype(np.float32)
        self.bounds = loss_config.bounds
        self.causal_epsilon = loss_config.causal_epsilon
        self.bins           = loss_config.collocation.bins
        if loss_config.collocation.n_points % self.bins:
            raise ValueError("collocation.n_points must be divisible by collocation.bins")

    @property
    def loss_names(self):
        return ("data", "physics", "initial")

    def system_params(self, physics):
        m = bounded_param(physics["log_m"], *self.bounds["m"])
        e = bounded_param(physics["log_e"], *self.bounds["e"])
        B = bounded_param(physics["log_B"], *self.bounds["B"])
        k = bounded_param(physics["log_k"], *self.bounds["k"])
        alpha = bounded_param(physics["log_alpha"], *self.bounds["alpha"])
        return m, e, B, k, alpha

    def predict_q(self, params, forward_fn, t):
        t_hat = t / self.t_max
        return forward_fn(params["net"], t_hat[None, None])[0]

    def acceleration(self, q, qdot, physics):
        m, e, B, k, alpha = self.system_params(physics)
        bz = B * (1.0 + alpha * (q[0] ** 2 + q[1] ** 2))
        ax = (e * bz * qdot[1] - k * q[0]) / m
        ay = (-e * bz * qdot[0] - k * q[1]) / m
        return jnp.stack([ax, ay])

    def eom_residual(self, q, qdot, qddot, physics):
        return qddot - self.acceleration(q, qdot, physics)

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

        qdot  = jax.jacfwd(q)
        qddot = jax.jacfwd(qdot)

        def residual(t):
            return self.eom_residual(q(t), qdot(t), qddot(t), physics)

        residuals_sq = jax.vmap(lambda t: jnp.sum(residual(t) ** 2))(self.t_collocation)
        binned       = residuals_sq.reshape(self.bins, -1).mean(axis=1)
        cumulative   = jnp.cumsum(binned) - binned
        weights      = jax.lax.stop_gradient(jnp.exp(-self.causal_epsilon * cumulative))
        physics_loss = jnp.sum(weights * binned) / (jnp.sum(weights) + 1e-12)

        q0 = jnp.stack([bounded_param(physics["q0_1"], *self.bounds["q0_1"]),
                        bounded_param(physics["q0_2"], *self.bounds["q0_2"])])
        p0 = jnp.stack([bounded_param(physics["p0_1"], *self.bounds["p0_1"]),
                        bounded_param(physics["p0_2"], *self.bounds["p0_2"])])
        t0 = jnp.array(0.0)
        initial_loss = jnp.sum((q(t0) - q0) ** 2) + jnp.sum((qdot(t0) - p0) ** 2)

        return jnp.array([data_loss, physics_loss, initial_loss])

    def val_call(self, params, X, y, loss_options, forward_fn, epoch=0):
        y_pred    = forward_fn(params["net"], X / self.t_max)
        data_loss = jnp.mean((y_pred - y) ** 2)
        return jnp.array([data_loss, 0.0, 0.0])

    def describe_physics(self, params):
        physics = params["physics"]
        m, e, B, k, alpha = self.system_params(physics)
        return {"m": float(m), "e": float(e), "B": float(B), "k": float(k), "alpha": float(alpha)}
