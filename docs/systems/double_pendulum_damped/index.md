# Double Pendulum (Damped)

<!-- BEGIN:animation -->
<!-- END:animation -->

- **Phase space:** 4D, $(\theta_1, \theta_2, p_1, p_2)$
- **True parameters:** $m_1 = m_2 = 1$, $l_1 = l_2 = 1$, $g = 9.81$, $\gamma = 0.15$
- **Training domain:** $t \in [0, 2]$

## The system

The same two-link pendulum as [Double Pendulum](../double_pendulum/index.md), with a linear drag torque on each joint. The conservative part is unchanged, so the Lagrangian, the mass matrix $M(\theta)$, the canonical momentum $p = M(\theta)\dot\theta$ and the Hamiltonian

$$
H(\theta, p) = \tfrac{1}{2}p^{\mathsf{T}}M(\theta)^{-1}p + V(\theta)
$$

are all identical. What changes is the right-hand side of the Euler-Lagrange equations, which picks up a Rayleigh dissipation term,

$$
D(\dot\theta) = \tfrac{1}{2}\gamma\bigl(\dot\theta_1^2 + \dot\theta_2^2\bigr),
\qquad
M(\theta)\ddot\theta = f(\theta, \dot\theta) - \gamma\dot\theta
$$

and the energy is no longer conserved,

$$
\dot H = -\gamma\bigl(\dot\theta_1^2 + \dot\theta_2^2\bigr) \le 0 .
$$

Ground truth is integrated with `diffrax.Dopri5` at `rtol = atol = 1e-7`, as for the conservative case.

The system exists to separate two things that the conservative double pendulum conflates. Chaos limits how far any method can track; a structural inability to lose energy is a different failure and shows up differently. A vanilla HNN or LNN has $\dot H = 0$ built in and cannot represent the decay at all, so it should track the shape of the motion while sitting on the wrong energy surface. The [dissipative extensions](../../methods/hnn.md#dissipative-extensions) are what the system is here to exercise, on a problem hard enough that the damping is not the only difficulty.

## Results

<!-- BEGIN:results -->
<!-- END:results -->
