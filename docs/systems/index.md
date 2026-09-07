# Systems

| System | Phase space | Canonical momentum | Energy | Ground truth | Extra methods |
|---|---|---|---|---|---|
| [Mass-Spring](mass_spring/index.md) | 2D $(q, p)$ | $p = \dot q$ | conserved | closed form | — |
| [Mass-Spring (Damped)](mass_spring_damped/index.md) | 2D $(q, p)$ | $p = \dot q$ | $\dot H = -\gamma p^2$ | closed form | port-Hamiltonian, learned dissipation |
| [Double Pendulum](double_pendulum/index.md) | 4D $(\theta, p)$ | $p = M(\theta)\dot\theta$ | conserved | Dopri5, tol $10^{-7}$ | — |
| [Double Pendulum (Damped)](double_pendulum_damped/index.md) | 4D $(\theta, p)$ | $p = M(\theta)\dot\theta$ | $\dot H = -\gamma\lVert\dot\theta\rVert^2$ | Dopri5, tol $10^{-7}$ | port-Hamiltonian, learned dissipation |
| [Charged Particle](charged_particle/index.md) | 4D $(r, p)$ | $p = mv + eA(r)$ | conserved | Dopri5, tol $10^{-7}$ | mechanical-coordinate HNN |

