import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp


def bounded_param(x, min_val, max_val):
    return min_val + (max_val - min_val) * jax.nn.sigmoid(x)


def inverse_bounded_param(value, min_val, max_val):
    normalized = (value - min_val) / (max_val - min_val)
    return math.log(normalized / (1.0 - normalized))


# General class for all type of models
class PhysicsLoss(ABC):
    @property
    def loss_names(self):
        raise NotImplementedError

    @property
    def aux_data(self):
        return None

    @abstractmethod
    def __call__(self, params, X, y, aux_data, forward_fn, epoch=0):
        return None

    def val_call(self, params, X, y, aux_data, forward_fn, epoch=0):
        return self(params, X, y, aux_data, forward_fn, epoch=epoch)

    def describe_physics(self, params):
        return None

    def trajectory(self, params, forward_fn):
        def q(t):
            return forward_fn(params["net"], t[None, None])[0]
        qdot = jax.jacfwd(q)
        def field(t):
            return jnp.concatenate([q(t), qdot(t)])
        return field

# MLP ───────────────────────────────────────────────────────────
class MLPLoss(PhysicsLoss):
    def __init__(self, loss_config):
        self.augment_trajectory_derivative = loss_config.augment_trajectory_derivative

    @property
    def loss_names(self):
        return ("data",)

    def __call__(self, params, X, y, aux_data, forward_fn, epoch=0):
        y_pred = forward_fn(params["net"], X)
        return jnp.array([jnp.mean((y_pred - y) ** 2)])

    def trajectory(self, params, forward_fn):
        if self.augment_trajectory_derivative:
            return super().trajectory(params, forward_fn)
        def field(t):
            return forward_fn(params["net"], t[None, None])[0]
        return field

    def dynamics(self, params, forward_fn):
        def field(state):
            return forward_fn(params["net"], state[None, :])[0]
        return field

# HNN ────────────────────────────────────────────────────────────
class HNNLoss(PhysicsLoss):
    def __init__(self, loss_config):
        ham = loss_config.hamiltonian
        if ham.dissipation not in ("none", "rayleigh"):
            raise NotImplementedError(f"HNN dissipation mode not implemented")
        self.n_dof               = ham.n_dof
        self.dissipation         = ham.dissipation
        self.known_term          = ham.known_term
        self.dissipation_term    = ham.dissipation_term
        self.describe            = ham.describe
        self.penalize_correction = ham.penalize_correction and ham.known_term is not None

    @property
    def loss_names(self):
        if self.penalize_correction:
            return ("eom", "correction")
        return ("eom",)

    def correction_fn(self, params, forward_fn):
        def correction(qp):
            return forward_fn(params["net"], qp[None])[0, 0]
        return correction

    def dynamics(self, params, forward_fn):
        n          = self.n_dof
        physics    = params.get("physics")
        correction = self.correction_fn(params, forward_fn)

        def hamiltonian(qp):
            c = correction(qp)
            if self.known_term is None:
                return c
            return self.known_term(qp, physics) + c

        grad_h = jax.grad(hamiltonian)

        def field(state):
            g  = grad_h(state)
            dq =  g[n:]
            dp = -g[:n]
            if self.dissipation == "rayleigh":
                q = state[:n]
                grad_d = jax.grad(lambda qdot: self.dissipation_term(q, qdot, physics))(dq)
                dp = dp - grad_d
            return jnp.concatenate([dq, dp])
        return field

    def __call__(self, params, X, y, aux_data, forward_fn, epoch=0):
        field       = self.dynamics(params, forward_fn)
        qp_dot_pred = jax.vmap(field)(X)
        eom_loss    = jnp.mean((qp_dot_pred - y) ** 2)

        if not self.penalize_correction:
            return jnp.array([eom_loss])

        correction      = self.correction_fn(params, forward_fn)
        correction_vals = jax.vmap(correction)(X)
        correction_loss = jnp.mean(correction_vals ** 2)
        return jnp.array([eom_loss, correction_loss])

    def describe_physics(self, params):
        if self.describe is not None and params.get("physics") is not None:
            return self.describe(params["physics"])
        return None

# LNN ────────────────────────────────────────────────────────────
class LNNLoss(PhysicsLoss):
    def __init__(self, loss_config):
        lag = loss_config.lagrangian
        if lag.dissipation not in ("none", "rayleigh"):
            raise NotImplementedError(f"LNN dissipation mode not implemented")
        self.n_dof              = lag.n_dof
        self.damping            = lag.damping
        self.dissipation        = lag.dissipation
        self.dissipation_term   = lag.dissipation_term
        self.known_term         = lag.known_term
        self.describe           = lag.describe
        self.penalize_correction = lag.penalize_correction and lag.known_term is not None

    @property
    def loss_names(self):
        if self.penalize_correction:
            return ("eom", "correction")
        return ("eom",)

    def correction_fn(self, params, forward_fn):
        def correction(q, qdot):
            qqdot = jnp.concatenate([q, qdot])
            return forward_fn(params["net"], qqdot[None])[0, 0]
        return correction

    def dynamics(self, params, forward_fn):
        n          = self.n_dof
        physics    = params.get("physics")
        correction = self.correction_fn(params, forward_fn)

        def lagrangian(q, qdot):
            c = correction(q, qdot)
            if self.known_term is None:
                return c
            return self.known_term(q, qdot, physics) + c

        def field(state):
            q, qdot = state[:n], state[n:]
            dL_dq = jax.grad(lagrangian, argnums=0)(q, qdot)
            hess  = jax.hessian(lagrangian, argnums=1)(q, qdot)
            cross = jax.jacfwd(jax.grad(lagrangian, argnums=1), argnums=0)(q, qdot)
            rhs   = dL_dq - cross @ qdot
            if self.dissipation == "rayleigh":
                grad_d = jax.grad(lambda qd: self.dissipation_term(q, qd, physics))(qdot)
                rhs = rhs - grad_d
            qddot = jnp.linalg.pinv(hess) @ rhs
            return jnp.concatenate([qdot, qddot])
        return field

    def __call__(self, params, X, y, aux_data, forward_fn, epoch=0):
        field          = self.dynamics(params, forward_fn)
        state_dot_pred = jax.vmap(field)(X)
        eom_loss       = jnp.mean((state_dot_pred - y) ** 2)

        if not self.penalize_correction:
            return jnp.array([eom_loss])

        n               = self.n_dof
        correction      = self.correction_fn(params, forward_fn)
        correction_vals = jax.vmap(lambda s: correction(s[:n], s[n:]))(X)
        correction_loss = jnp.mean(correction_vals ** 2)
        return jnp.array([eom_loss, correction_loss])

    def describe_physics(self, params):
        if self.describe is not None and params.get("physics") is not None:
            return self.describe(params["physics"])
        return None
