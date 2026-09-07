import functools


from configs.configurations import HamiltonianConfig
from model.physics_loss import bounded_param


def system_params(physics, bounds):
    m = bounded_param(physics["log_m"], *bounds["m"])
    e = bounded_param(physics["log_e"], *bounds["e"])
    B = bounded_param(physics["log_B"], *bounds["B"])
    k = bounded_param(physics["log_k"], *bounds["k"])
    alpha = bounded_param(physics["log_alpha"], *bounds["alpha"])
    return m, e, B, k, alpha


def known_term_canonical(qp, physics, bounds):
    m, e, B, k, alpha = system_params(physics, bounds)
    x, y, px, py = qp[0], qp[1], qp[2], qp[3]
    sc = B * (0.5 + 0.25 * alpha * (x ** 2 + y ** 2))
    ax = -sc * y
    ay = sc * x
    kinetic = ((px - e * ax) ** 2 + (py - e * ay) ** 2) / (2.0 * m)
    return kinetic + 0.5 * k * (x ** 2 + y ** 2)


def known_term_mechanical(qp, physics, bounds):
    m, e, B, k, alpha = system_params(physics, bounds)
    x, y, px, py = qp[0], qp[1], qp[2], qp[3]
    return (px ** 2 + py ** 2) / (2.0 * m) + 0.5 * k * (x ** 2 + y ** 2)


def describe(physics, bounds):
    m, e, B, k, alpha = system_params(physics, bounds)
    return {"m": float(m), "e": float(e), "B": float(B), "k": float(k), "alpha": float(alpha)}


def make_hamiltonian_parametric(bounds):
    return HamiltonianConfig(
        n_dof=2, dissipation="none",
        known_term=functools.partial(known_term_canonical, bounds=bounds),
        describe=functools.partial(describe, bounds=bounds),
        penalize_correction=True,
    )


def make_hamiltonian_parametric_mechanical(bounds):
    return HamiltonianConfig(
        n_dof=2, dissipation="none",
        known_term=functools.partial(known_term_mechanical, bounds=bounds),
        describe=functools.partial(describe, bounds=bounds),
        penalize_correction=True,
    )


HAMILTONIAN = HamiltonianConfig(n_dof=2, dissipation="none")
