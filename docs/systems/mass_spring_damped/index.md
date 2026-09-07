# Mass-Spring (Damped)

<!-- BEGIN:animation -->
<!-- END:animation -->

- **Phase space:** 2D, $(q, p)$
- **True parameters:** $\omega = 2.1$, $\gamma = 0.1323$
- **Training domain:** $t \in [0, 4\pi]$

## The system

The same unit mass and linear spring as [Mass-Spring](../mass_spring/index.md), with a linear drag force added. The equation of motion is

$$
\ddot q = -\omega^2 q - \gamma\dot q
\qquad\Longleftrightarrow\qquad
\dot q = p, \quad \dot p = -\omega^2 q - \gamma p
$$

For $\gamma < 2\omega$ the motion is underdamped, with $\omega_d = \sqrt{\omega^2 - \gamma^2/4}$,

$$
q(t) = e^{-\gamma t/2}\bigl(A\cos\omega_d t + B\sin\omega_d t\bigr),
\qquad
A = q_0, \quad B = \frac{p_0 + \gamma q_0/2}{\omega_d}
$$

and $p = \dot q$ follows by differentiating. Ground truth comes from this expression, so again there is no solver error.

Damping is not derivable from a Lagrangian alone. The conservative part is unchanged and the drag is introduced through a Rayleigh dissipation function,

$$
L(q, \dot q) = \tfrac{1}{2}\dot q^{2} - \tfrac{1}{2}\omega^{2}q^{2},
\qquad
D(\dot q) = \tfrac{1}{2}\gamma\dot q^{2},
\qquad
\frac{d}{dt}\frac{\partial L}{\partial \dot q} = \frac{\partial L}{\partial q} - \frac{\partial D}{\partial \dot q}
$$

The Hamiltonian keeps the same form but is no longer conserved,

$$
H(q, p) = \tfrac{1}{2}p^{2} + \tfrac{1}{2}\omega^{2}q^{2},
\qquad
\dot H = -\gamma p^{2} \le 0 .
$$

This is what makes the system a test rather than a repeat of the conservative case. A vanilla HNN or LNN has $\dot H = 0$ built into its architecture and cannot represent energy decay at all, so it is expected to fail here in a specific and predictable way. The [dissipative extensions](../../methods/hnn.md#dissipative-extensions) are what the system exists to exercise.

## Results

<!-- BEGIN:results -->
_No benchmark results found for `mass_spring_damped`._
<!-- END:results -->


