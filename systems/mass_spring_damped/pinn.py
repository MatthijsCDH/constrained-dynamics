import numpy as np
import jax
import jax.numpy as jnp

from model.physics_loss import PhysicsLoss
from model.physics_loss import bounded_param


class MassSpringDampedPINNLoss(PhysicsLoss):
    def __init__(self, loss_config):
        self.t_max = loss_config.collocation.x_max
        rng        = np.random.default_rng(0)
        self.t_collocation = np.sort(
            rng.uniform(loss_config.collocation.x_min, loss_config.collocation.x_max,
                        size=loss_config.collocation.n_points)
        ).astype(np.float32)
        self.omega_bounds = loss_config.bounds["omega"]
        self.gamma_bounds = loss_config.bounds["gamma"]
        self.q0_bounds     = loss_config.bounds["q0"]
        self.p0_bounds     = loss_config.bounds["p0"]

    @property
    def loss_names(self):
        return ("data", "physics", "boundary")

    def predict_q(self, params, forward_fn, t):
        t_hat = t / self.t_max
        return forward_fn(params["net"], t_hat[None, None])[0, 0]

    def q_ddot(self, q, qdot, physics):
        omega = bounded_param(physics["log_omega"], *self.omega_bounds)
        gamma = bounded_param(physics["log_gamma"], *self.gamma_bounds)
        return -omega ** 2 * q - gamma * qdot

    def trajectory(self, params, forward_fn):
        def q(t):
            t_hat = t / self.t_max
            return forward_fn(params["net"], t_hat[None, None])[0]
        qdot = jax.jacfwd(q)
        def field(t):
            return jnp.concatenate([q(t), qdot(t)])
        return field

    def __call__(self, params, X, y, aux_data, forward_fn, epoch=0):
        physics = params["physics"]

        y_pred    = forward_fn(params["net"], X / self.t_max)
        data_loss = jnp.mean((y_pred - y) ** 2)

        def q(t):
            return self.predict_q(params, forward_fn, t)

        qdot  = jax.grad(q)
        qddot = jax.grad(qdot)

        def residual(t):
            return qddot(t) - self.q_ddot(q(t), qdot(t), physics)

        residuals    = jax.vmap(residual)(self.t_collocation)
        physics_loss = jnp.mean(residuals ** 2)

        q0 = bounded_param(physics["q0"], *self.q0_bounds)
        p0 = bounded_param(physics["p0"], *self.p0_bounds)
        t0 = jnp.array(0.0)
        boundary_loss = (q(t0) - q0) ** 2 + (qdot(t0) - p0) ** 2

        return jnp.array([data_loss, physics_loss, boundary_loss])

    def val_call(self, params, X, y, aux_data, forward_fn, epoch=0):
        y_pred    = forward_fn(params["net"], X / self.t_max)
        data_loss = jnp.mean((y_pred - y) ** 2)
        return jnp.array([data_loss, 0.0, 0.0])

    def describe_physics(self, params):
        physics = params["physics"]
        omega   = float(bounded_param(physics["log_omega"], *self.omega_bounds))
        gamma   = float(bounded_param(physics["log_gamma"], *self.gamma_bounds))
        q0      = float(bounded_param(physics["q0"], *self.q0_bounds))
        p0      = float(bounded_param(physics["p0"], *self.p0_bounds))
        return f"omega={omega:.4f}  gamma={gamma:.4f}  q0={q0:.4f}  p0={p0:.4f}"
