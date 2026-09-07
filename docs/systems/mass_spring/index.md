# Mass-Spring (Simple Harmonic Oscillator)

<!-- BEGIN:animation -->
<!-- END:animation -->

- **Phase space:** 2D, $(q, p)$
- **True parameter:** $\omega = 2.1$
- **Training domain:** $t \in [0, 4 \pi]$

## The system

A unit mass on a linear spring. The equation of motion is

$$
\ddot q = -\omega^2 q
\qquad\Longleftrightarrow\qquad
\dot q = p, \quad \dot p = -\omega^2 q
$$

which has solution,

$$
q(t) = q_0\cos\omega t + \frac{p_0}{\omega}\sin\omega t,
\qquad
p(t) = -q_0\,\omega\sin\omega t + p_0\cos\omega t .
$$

Ground truth is evaluated from this expression rather than integrated numerically, so it carries no solver error. The Lagrangian and Hamiltonian are

$$
L(q, \dot q) = \tfrac{1}{2}\dot q^{2} - \tfrac{1}{2}\omega^{2}q^{2},
\qquad
H(q, p) = \tfrac{1}{2}p^{2} + \tfrac{1}{2}\omega^{2}q^{2} .
$$

With unit mass $p = \dot q$, so the canonical and generalized coordinates coincide and HNN and LNN see numerically identical inputs. That makes this the one system where the two can be compared without the canonical-momentum derivation getting in the way.

$H$ is conserved exactly, and every orbit is a closed ellipse in phase space, so any energy drift a method reports is its own error rather than a property of the system.

## Results

<!-- BEGIN:results -->
_No benchmark results found for `mass_spring`._
<!-- END:results -->

