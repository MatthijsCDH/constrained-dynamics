import functools

from configs.configurations import HamiltonianConfig
from model.physics_loss import bounded_param


def known_term(qp, physics, omega_bounds):
    q, p = qp[0], qp[1]
    omega = bounded_param(physics["log_omega"], *omega_bounds)
    return 0.5 * p ** 2 + 0.5 * omega ** 2 * q ** 2


def dissipation_term(q, qdot, physics, gamma_bounds):
    gamma = bounded_param(physics["log_gamma"], *gamma_bounds)
    return 0.5 * gamma * qdot[0] ** 2


def describe(physics, omega_bounds, gamma_bounds):
    omega = float(bounded_param(physics["log_omega"], *omega_bounds))
    gamma = float(bounded_param(physics["log_gamma"], *gamma_bounds))
    return {"omega": omega, "gamma": gamma}


def make_hamiltonian_parametric(omega_bounds, gamma_bounds):
    return HamiltonianConfig(
        n_dof=1, dissipation="rayleigh",
        known_term=functools.partial(known_term, omega_bounds=omega_bounds),
        dissipation_term=functools.partial(dissipation_term, gamma_bounds=gamma_bounds),
        describe=functools.partial(describe, omega_bounds=omega_bounds, gamma_bounds=gamma_bounds),
        penalize_correction=True,
    )


def describe_omega_only(physics, omega_bounds):
    omega = float(bounded_param(physics["log_omega"], *omega_bounds))
    return {"omega": omega}


def make_hamiltonian_port_hamiltonian(omega_bounds):
    return HamiltonianConfig(
        n_dof=1, dissipation="port_hamiltonian",
        known_term=functools.partial(known_term, omega_bounds=omega_bounds),
        describe=functools.partial(describe_omega_only, omega_bounds=omega_bounds),
        penalize_correction=True,
    )


HAMILTONIAN = HamiltonianConfig(n_dof=1, dissipation="none")
