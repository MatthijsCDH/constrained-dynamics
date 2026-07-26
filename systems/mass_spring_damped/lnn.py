import functools

from configs.configurations import LagrangianConfig
from model.physics_loss import bounded_param


def known_term(q, qdot, physics, omega_bounds):
    omega = bounded_param(physics["log_omega"], *omega_bounds)
    return 0.5 * qdot[0] ** 2 - 0.5 * omega ** 2 * q[0] ** 2


def dissipation_term(q, qdot, physics, gamma_bounds):
    gamma = bounded_param(physics["log_gamma"], *gamma_bounds)
    return 0.5 * gamma * qdot[0] ** 2


def describe(physics, omega_bounds, gamma_bounds):
    omega = float(bounded_param(physics["log_omega"], *omega_bounds))
    gamma = float(bounded_param(physics["log_gamma"], *gamma_bounds))
    return f"omega={omega:.4f}  gamma={gamma:.4f}"


def make_lagrangian_parametric(omega_bounds, gamma_bounds):
    return LagrangianConfig(
        n_dof=1, dissipation="rayleigh",
        known_term=functools.partial(known_term, omega_bounds=omega_bounds),
        dissipation_term=functools.partial(dissipation_term, gamma_bounds=gamma_bounds),
        describe=functools.partial(describe, omega_bounds=omega_bounds, gamma_bounds=gamma_bounds),
        penalize_correction=True,
    )


LAGRANGIAN = LagrangianConfig(n_dof=1)
