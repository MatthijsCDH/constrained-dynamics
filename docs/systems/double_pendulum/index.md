# Double Pendulum (Conservative)

<!-- BEGIN:animation -->
<!-- END:animation -->

- **Phase space:** 4D, $(\theta_1, \theta_2, p_1, p_2)$
- **True parameters:** $m_1 = m_2 = 1$, $l_1 = l_2 = 1$, $g = 9.81$
- **Training domain:** $t \in [0, 2]$

## The system

Two point masses on rigid massless rods, the second hanging from the first. With $\Delta = \theta_1 - \theta_2$ the Lagrangian is

```math
L = \tfrac{1}{2}(m_1 + m_2)l_1^2\dot\theta_1^2
  + \tfrac{1}{2}m_2 l_2^2\dot\theta_2^2
  + m_2 l_1 l_2 \dot\theta_1\dot\theta_2\cos\Delta
  + (m_1 + m_2)g l_1\cos\theta_1
  + m_2 g l_2\cos\theta_2
```

The kinetic term is a quadratic form in the velocities with a configuration-dependent mass matrix,

```math
M(\theta) =
\begin{pmatrix}
(m_1 + m_2)l_1^2 & m_2 l_1 l_2\cos\Delta \\
m_2 l_1 l_2\cos\Delta & m_2 l_2^2
\end{pmatrix}
```

so the Euler-Lagrange equations are a coupled linear system for the accelerations, $M(\theta)\ddot\theta = f(\theta, \dot\theta)$, solved at every evaluation. The canonical momentum is

```math
p = M(\theta)\,\dot\theta,
\qquad
H(\theta, p) = \tfrac{1}{2}p^{\mathsf{T}}M(\theta)^{-1}p + V(\theta),
\qquad
V(\theta) = -(m_1 + m_2)g l_1\cos\theta_1 - m_2 g l_2\cos\theta_2 .
```

$H$ is conserved, but the motion is chaotic and there is no closed-form solution. Ground truth is integrated with `diffrax.Dopri5` at `rtol = atol = 1e-7`, the tightest tolerance float32 supports, so unlike the mass-spring systems the reference itself carries solver error. Nearby trajectories separate exponentially, which puts a hard ceiling on how far any method can track regardless of how well it learned the dynamics.

This is the system where the coordinate choice stops being cosmetic. $p = M(\theta)\dot\theta$ depends on the configuration, so $p$ and $\dot\theta$ are genuinely different quantities and an HNN needs the transformation derived before training can start. An LNN takes $(\theta, \dot\theta)$ directly and skips it.

Two of the five constants are held fixed in the parametric variants, `m1` and `g`, because the parameters are not all identifiable from the dynamics: scaling both masses leaves the equations of motion unchanged, and so does scaling both lengths against $g$. Fixing one mass and gravity removes both degeneracies and leaves `m2`, `l1` and `l2` recoverable.

## Results

<!-- BEGIN:results -->
<!-- END:results -->
