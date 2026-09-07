import functools

from configs.configurations import LagrangianConfig
from model.physics_loss import bounded_param


def known_term(q, qdot, physics, omega_bounds):
    omega = bounded_param(physics["log_omega"], *omega_bounds)
    return 0.5 * qdot[0] ** 2 - 0.5 * omega ** 2 * q[0] ** 2


def describe(physics, omega_bounds):
    omega = float(bounded_param(physics["log_omega"], *omega_bounds))
    return {"omega": omega}


def make_lagrangian_parametric(omega_bounds):
    return LagrangianConfig(
        n_dof=1,
        known_term=functools.partial(known_term, omega_bounds=omega_bounds),
        describe=functools.partial(describe, omega_bounds=omega_bounds),
        penalize_correction=True,
    )


LAGRANGIAN = LagrangianConfig(n_dof=1)
