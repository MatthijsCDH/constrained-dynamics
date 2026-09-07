import functools


from configs.configurations import LagrangianConfig
from model.physics_loss import bounded_param


def system_params(physics, bounds):
    m = bounded_param(physics["log_m"], *bounds["m"])
    e = bounded_param(physics["log_e"], *bounds["e"])
    B = bounded_param(physics["log_B"], *bounds["B"])
    k = bounded_param(physics["log_k"], *bounds["k"])
    alpha = bounded_param(physics["log_alpha"], *bounds["alpha"])
    return m, e, B, k, alpha


def known_term(q, qdot, physics, bounds):
    m, e, B, k, alpha = system_params(physics, bounds)
    x, y = q[0], q[1]
    vx, vy = qdot[0], qdot[1]
    sc = B * (0.5 + 0.25 * alpha * (x ** 2 + y ** 2))
    ax = -sc * y
    ay = sc * x
    kinetic = 0.5 * m * (vx ** 2 + vy ** 2)
    coupling = e * (vx * ax + vy * ay)
    return kinetic + coupling - 0.5 * k * (x ** 2 + y ** 2)


def describe(physics, bounds):
    m, e, B, k, alpha = system_params(physics, bounds)
    return {"m": float(m), "e": float(e), "B": float(B), "k": float(k), "alpha": float(alpha)}


def make_lagrangian_parametric(bounds):
    return LagrangianConfig(
        n_dof=2,
        known_term=functools.partial(known_term, bounds=bounds),
        describe=functools.partial(describe, bounds=bounds),
        penalize_correction=True,
    )


LAGRANGIAN = LagrangianConfig(n_dof=2)
