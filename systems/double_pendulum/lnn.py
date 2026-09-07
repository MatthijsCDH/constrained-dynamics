import functools

import jax.numpy as jnp

from configs.configurations import LagrangianConfig
from model.physics_loss import bounded_param


def system_params(physics, bounds):
    m1 = bounded_param(physics["log_m1"], *bounds["m1"])
    m2 = bounded_param(physics["log_m2"], *bounds["m2"])
    l1 = bounded_param(physics["log_l1"], *bounds["l1"])
    l2 = bounded_param(physics["log_l2"], *bounds["l2"])
    g  = bounded_param(physics["log_g"],  *bounds["g"])
    return m1, m2, l1, l2, g


def known_term(q, qdot, physics, bounds):
    m1, m2, l1, l2, g = system_params(physics, bounds)
    q1, q2 = q[0], q[1]
    v1, v2 = qdot[0], qdot[1]

    delta = q1 - q2
    kinetic = (0.5 * (m1 + m2) * l1 ** 2 * v1 ** 2
               + 0.5 * m2 * l2 ** 2 * v2 ** 2
               + m2 * l1 * l2 * v1 * v2 * jnp.cos(delta))
    potential = -(m1 + m2) * g * l1 * jnp.cos(q1) - m2 * g * l2 * jnp.cos(q2)
    return kinetic - potential


def describe(physics, bounds):
    m1, m2, l1, l2, g = system_params(physics, bounds)
    return {"m1": float(m1), "m2": float(m2), "l1": float(l1), "l2": float(l2), "g": float(g)}


def make_lagrangian_parametric(bounds):
    return LagrangianConfig(
        n_dof=2,
        known_term=functools.partial(known_term, bounds=bounds),
        describe=functools.partial(describe, bounds=bounds),
        penalize_correction=True,
    )


LAGRANGIAN = LagrangianConfig(n_dof=2)
