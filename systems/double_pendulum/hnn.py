import functools

import jax.numpy as jnp

from configs.configurations import HamiltonianConfig
from model.physics_loss import bounded_param


def system_params(physics, bounds):
    m1 = bounded_param(physics["log_m1"], *bounds["m1"])
    m2 = bounded_param(physics["log_m2"], *bounds["m2"])
    l1 = bounded_param(physics["log_l1"], *bounds["l1"])
    l2 = bounded_param(physics["log_l2"], *bounds["l2"])
    g  = bounded_param(physics["log_g"],  *bounds["g"])
    return m1, m2, l1, l2, g


def known_term(qp, physics, bounds):
    m1, m2, l1, l2, g = system_params(physics, bounds)
    q1, q2, p1, p2 = qp[0], qp[1], qp[2], qp[3]

    delta = q1 - q2
    a = (m1 + m2) * l1 ** 2
    b = m2 * l1 * l2 * jnp.cos(delta)
    c = m2 * l2 ** 2
    det = a * c - b ** 2

    kinetic = 0.5 * (c * p1 ** 2 - 2.0 * b * p1 * p2 + a * p2 ** 2) / det
    potential = -(m1 + m2) * g * l1 * jnp.cos(q1) - m2 * g * l2 * jnp.cos(q2)
    return kinetic + potential


def describe(physics, bounds):
    m1, m2, l1, l2, g = system_params(physics, bounds)
    return {"m1": float(m1), "m2": float(m2), "l1": float(l1), "l2": float(l2), "g": float(g)}


def make_hamiltonian_parametric(bounds):
    return HamiltonianConfig(
        n_dof=2, dissipation="none",
        known_term=functools.partial(known_term, bounds=bounds),
        describe=functools.partial(describe, bounds=bounds),
        penalize_correction=True,
    )


HAMILTONIAN = HamiltonianConfig(n_dof=2, dissipation="none")
