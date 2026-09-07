import math
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp


def bounded_param(x, min_val, max_val):
    return min_val + (max_val - min_val) * jax.nn.sigmoid(x)


def inverse_bounded_param(value, min_val, max_val):
    normalized = (value - min_val) / (max_val - min_val)
    return math.log(normalized / (1.0 - normalized))


def lower_triangular(vec, k):
    rows, cols = jnp.tril_indices(k)
    return jnp.zeros((k, k)).at[rows, cols].set(vec)


# General class for all type of models
class PhysicsLoss(ABC):
    @property
    def loss_names(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, params, X, y, loss_options, forward_fn, epoch=0):
        return None

    def val_call(self, params, X, y, loss_options, forward_fn, epoch=0):
        return self(params, X, y, loss_options, forward_fn, epoch=epoch)

    def describe_physics(self, params):
        return None

    def trajectory(self, params, forward_fn):
        def q(t):
            return forward_fn(params["net"], t[None, None])[0]
        qdot = jax.jacfwd(q)
        def field(t):
            return jnp.concatenate([q(t), qdot(t)])
        return field

    def cache_key(self):
        return getattr(self, "loss_config", self)

    def __eq__(self, other):
        return type(self) is type(other) and self.cache_key() is other.cache_key()

    def __hash__(self):
        return hash((type(self), id(self.cache_key())))

# MLP ───────────────────────────────────────────────────────────
class MLPLoss(PhysicsLoss):
    def __init__(self, loss_config):
        self.loss_config = loss_config
        self.augment_trajectory_derivative = loss_config.augment_trajectory_derivative

    @property
    def loss_names(self):
        return ("data",)

    def __call__(self, params, X, y, loss_options, forward_fn, epoch=0):
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
        self.loss_config = loss_config
        ham = loss_config.hamiltonian
        if ham.dissipation not in ("none", "rayleigh", "port_hamiltonian"):
            raise NotImplementedError(f"HNN dissipation mode not implemented")
        if ham.integrator not in ("rk4", "leapfrog"):
            raise NotImplementedError(f"HNN integrator not implemented")
        self.n_dof               = ham.n_dof
        self.dissipation         = ham.dissipation
        self.integrator          = ham.integrator
        self.known_term          = ham.known_term
        self.dissipation_term    = ham.dissipation_term
        self.describe            = ham.describe
        self.penalize_correction = ham.penalize_correction and ham.known_term is not None
        if self.dissipation == "port_hamiltonian":
            k = 2 * self.n_dof
            self.l_size = k * (k + 1) // 2

    @property
    def loss_names(self):
        if self.penalize_correction:
            return ("eom", "correction")
        return ("eom",)

    def correction_fn(self, params, forward_fn):
        def correction(qp):
            return forward_fn(params["net"], qp[None])[0, 0]
        return correction

    def hamiltonian_fn(self, params, forward_fn):
        physics    = params.get("physics")
        correction = self.correction_fn(params, forward_fn)
        def hamiltonian(qp):
            c = correction(qp)
            if self.known_term is None:
                return c
            return self.known_term(qp, physics) + c
        return hamiltonian

    def dissipation_matrix_fn(self, params, forward_fn):
        k = 2 * self.n_dof
        def R(qp):
            l_vec = forward_fn(params["net"], qp[None])[0, 1:1 + self.l_size]
            L = lower_triangular(l_vec, k)
            return L @ L.T
        return R

    def dynamics(self, params, forward_fn):
        n           = self.n_dof
        physics     = params.get("physics")
        hamiltonian = self.hamiltonian_fn(params, forward_fn)
        grad_h      = jax.grad(hamiltonian)

        if self.dissipation == "port_hamiltonian":
            dissipation_matrix = self.dissipation_matrix_fn(params, forward_fn)

            def field(state):
                g   = grad_h(state)
                dq  =  g[n:]
                dp  = -g[:n]
                R   = dissipation_matrix(state)
                R_g = R @ g
                dq  = dq - R_g[:n]
                dp  = dp - R_g[n:]
                return jnp.concatenate([dq, dp])
            return field

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

    def __call__(self, params, X, y, loss_options, forward_fn, epoch=0):
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
        self.loss_config = loss_config
        lag = loss_config.lagrangian
        if lag.dissipation not in ("none", "rayleigh", "learned_dissipation"):
            raise NotImplementedError(f"LNN dissipation mode not implemented")
        self.n_dof              = lag.n_dof
        self.dissipation        = lag.dissipation
        self.dissipation_term   = lag.dissipation_term
        self.known_term         = lag.known_term
        self.describe           = lag.describe
        self.penalize_correction = lag.penalize_correction and lag.known_term is not None
        if self.dissipation == "learned_dissipation":
            k = self.n_dof
            self.l_size = k * (k + 1) // 2

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

    def dissipation_matrix_fn(self, params, forward_fn):
        k = self.n_dof
        def R(q):
            qqdot = jnp.concatenate([q, jnp.zeros_like(q)])
            l_vec = forward_fn(params["net"], qqdot[None])[0, 1:1 + self.l_size]
            L = lower_triangular(l_vec, k)
            return L @ L.T
        return R

    def dynamics(self, params, forward_fn):
        n          = self.n_dof
        physics    = params.get("physics")
        correction = self.correction_fn(params, forward_fn)

        def lagrangian(q, qdot):
            c = correction(q, qdot)
            if self.known_term is None:
                return c
            return self.known_term(q, qdot, physics) + c

        dissipation_matrix = (
            self.dissipation_matrix_fn(params, forward_fn)
            if self.dissipation == "learned_dissipation" else None
        )

        def field(state):
            q, qdot = state[:n], state[n:]
            dL_dq = jax.grad(lagrangian, argnums=0)(q, qdot)
            hess  = jax.hessian(lagrangian, argnums=1)(q, qdot)
            cross = jax.jacfwd(jax.grad(lagrangian, argnums=1), argnums=0)(q, qdot)
            rhs   = dL_dq - cross @ qdot
            if self.dissipation == "rayleigh":
                grad_d = jax.grad(lambda qd: self.dissipation_term(q, qd, physics))(qdot)
                rhs = rhs - grad_d
            if dissipation_matrix is not None:
                rhs = rhs - dissipation_matrix(q) @ qdot
            qddot = jnp.linalg.pinv(hess) @ rhs
            return jnp.concatenate([qdot, qddot])
        return field

    def __call__(self, params, X, y, loss_options, forward_fn, epoch=0):
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
