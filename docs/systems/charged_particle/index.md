# Charged Particle in a Magnetic Trap

<!-- BEGIN:animation -->
<!-- END:animation -->

- **Phase space:** 4D, $(x, y, p_x, p_y)$
- **True parameters:** $m = 1$, $e = 1$, $B_0 = 1$, $k = 1$, $\alpha = 0.6$
- **Training domain:** $t \in [0, 2\pi]$

## The system

A charged particle moving in a plane, in a magnetic field along $z$ and a harmonic trap. The field is deliberately non-uniform,

```math
B(r) = B_0\bigl(1 + \alpha\lVert r\rVert^2\bigr),
\qquad
A(r) = B_0\Bigl(\tfrac{1}{2} + \tfrac{1}{4}\alpha\lVert r\rVert^2\Bigr)(-y,\ x)
```

with $A$ chosen so that $\nabla \times A = B(r) \hat z$. A uniform field would give pure circular motion, which every method fits trivially; the $\alpha$ term makes the gyrofrequency depend on position and the orbits genuinely non-sinusoidal.

The Lagrangian, canonical momentum and Hamiltonian are

```math
L = \tfrac{1}{2}m\lVert v\rVert^2 + e\,v\cdot A(r) - \tfrac{1}{2}k\lVert r\rVert^2,
\qquad
p = m v + e A(r),
\qquad
H = \frac{\lVert p - eA(r)\rVert^2}{2m} + \tfrac{1}{2}k\lVert r\rVert^2 .
```

$H$ is conserved. The magnetic force does no work, so all of the energy is kinetic plus trap potential.

The point of this system is the middle expression. The canonical momentum differs from the mechanical momentum $mv$ by the vector potential, and $A$ depends on position, so the two are not related by a constant. In mechanical variables $(r, mv)$ the dynamics are **provably not** a Hamiltonian vector field for the canonical symplectic form, so an HNN handed mechanical momenta cannot represent them no matter how well it trains. That is not a tuning problem, it is a statement about what the architecture can express.

Both are benchmarked. `hnn` and `hnn_parametric` receive canonical $(r, p)$ and should work; `hnn_mechanical` and `hnn_parametric_mechanical` receive $(r, mv)$ and should fail. It is the cleanest available demonstration of what "requires canonical coordinates" actually costs when the transformation is not the identity.

Ground truth is integrated in velocity coordinates with `diffrax.Dopri5` at `rtol = atol = 1e-7` and converted to canonical form on output, which avoids the messier canonical $\dot p$ term.

Of the five constants, `m` and `e` are held fixed in the parametric variants. The dynamics depend only on the combinations $eB/m$ and $k/m$, so mass and charge are not separately identifiable from a trajectory; fixing them leaves `B`, `k` and `alpha` recoverable.

## Results

<!-- BEGIN:results -->
<!-- END:results -->
