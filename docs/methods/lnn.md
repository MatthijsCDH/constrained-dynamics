# LNN — Lagrangian Neural Network

An LNN is trained on field data $x_i = (q_i, \dot q_i)$, where $q_i$ and $\dot q_i$ are the generalized coordinates and velocities of the system respectively, and outputs a single scalar corresponding to the Lagrangian $L(q, \dot q)$. Like the HNN it never predicts the dynamics directly; instead they are recovered from derivatives of that scalar. Unlike the HNN, the recovery is not a gradient but a linear solve. 

Conservation of energy is an inherent property of the architecture, due to Noether's theorem. For a Lagrangian with no explicit time dependence, the conserved quantity is

$$
E = \dot q \cdot \frac{\partial L}{\partial \dot q} - L
$$

which is constant along the Euler-Lagrange flow for any $L(q, \dot q)$ the network represents. Moreover, because time does not explicitly enter the training, each sample in $(q, \dot q)$ space is independent, eliminating the need for trajectory-based training and reducing computational cost. Time evolution only enters after training, by numerically integrating the learned dynamics.

The difference with an HNN lies in the coordinates and in how the dynamics are recovered. An HNN requires the canonical momentum $p$, which is system-specific and has to be derived before training can start, whereas an LNN takes $(q, \dot q)$ directly. In return, Hamilton's equations are first order and yield the dynamics from a single gradient, while the Euler-Lagrange equation is second order and coupled, so the accelerations follow from a linear solve instead.

<p align="center">
  <img src="../images/LNN.svg" width="680" alt="LNN architecture: generalized coordinates and velocities in, scalar Lagrangian out, accelerations via the Euler-Lagrange solve">
</p>

$(q_i, \dot q_i)$ in, scalar $\mathcal{L}$ out. The accelerations follow from the Euler-Lagrange equation, which requires second derivatives of the network output rather than the single gradient an HNN needs.

## Loss function

At each state, the network does not predict the time derivatives, but instead recovers them from the Euler-Lagrange equation,

$$
\frac{d}{dt}\left(\frac{\partial L}{\partial \dot q}\right) = \frac{\partial L}{\partial q}
$$

Expanding the total time derivative with the chain rule turns this into a linear system for the accelerations,

$$
\underbrace{\frac{\partial^2 L_\theta}{\partial \dot q\, \partial \dot q^{\mathsf{T}}}}_{\text{Hessian}} \ddot q
\;=\;
\frac{\partial L_\theta}{\partial q} - \underbrace{\frac{\partial^2 L_\theta}{\partial \dot q\, \partial q^{\mathsf{T}}}}_{\text{cross term}} \dot q
$$

so a single evaluation needs three autodiff calls on the scalar output: a `jax.grad` for $\partial L_\theta/\partial q$, a `jax.hessian` for the Hessian, and a `jax.jacfwd` of `jax.grad` for the cross term. Nothing constrains the learned Hessian to be invertible, and early in training it frequently is not, so the system is solved with `jnp.linalg.pinv` rather than a direct inverse. The predicted dynamics are then

$$
f_\theta(q, \dot q) = \begin{pmatrix} \dot q \\[2pt] \ddot q \end{pmatrix},
\qquad
\ddot q = \left(\frac{\partial^2 L_\theta}{\partial \dot q\, \partial \dot q^{\mathsf{T}}}\right)^{+}\left(\frac{\partial L_\theta}{\partial q} - \frac{\partial^2 L_\theta}{\partial \dot q\, \partial q^{\mathsf{T}}}\dot q\right)
$$

which are compared directly with the labeled derivatives using the mean square error,

$$
\mathcal{L}_{\text{eom}} = \frac{1}{N} \sum_{i=1}^{N} \bigl\lVert f_\theta(q_i, \dot q_i) - \dot x_i \bigr\rVert^2 .
$$

This term is weighted by `LNNLossConfig.lambda_eom`. Together with an L2 regularization term $\mathcal{L}_{\text{reg}}$, this makes up the loss vector for an LNN. These loss functions are balanced by the Kendall uncertainty weighting, described in [Loss weighting](../index.md#loss-weighting). 


## Add-ons built for this project

### Hybrid known-term + network

The Lagrangian $L$ can be split into a known functional form plus a network correction: $L = L_{\text{known}}(q,\dot q;\theta) + L_{\text{net}}(q,\dot q)$, where $\theta$ are learnable physics parameters and the known term is supplied per system through `LagrangianConfig.known_term`. When `known_term` is `None` the Lagrangian is the pure black-box network. The network only has to learn whatever the known term misses, and the physics parameters come out of training as an actual number, not just a fitted black box.

However, this procedure provides no mechanism to prevent $L_{\text{net}}(q,\dot q)$ from absorbing the entire Lagrangian, rendering $L_{\text{known}}(q,\dot q;\theta)$ irrelevant. Consequently, the learnable parameter $\theta$ no longer has a unique solution, as many different $\theta$ can be paired with a compensating $L_{\text{net}}$ to reach the same minimal loss, leaving the loss landscape flat along $\theta$. Setting `LagrangianConfig.penalize_correction` adds a term to the loss to break the degeneracy,

$$
\mathcal{L}_{\text{correction}} = \frac{1}{N} \sum_{i=1}^{N} L_{\text{net}}(q_i, \dot q_i)^2
$$

This is a pure L2 penalty on the correction network's own output, weighted by `LNNLossConfig.lambda_correction`. Increasing $L_{\text{net}}$ to absorb more of the Lagrangian now has a penalty, which pushes the optimizer toward solutions where $L_{\text{known}}(q,\dot q;\theta)$ carries more of the dynamics and $L_{\text{net}}$ is used only to capture the remaining residual.

### Dissipative extensions

A standard LNN trained on damped data cannot represent systems with decaying energy, since $\dot{E}=0$ is an inherent property. Setting `LagrangianConfig.dissipation` to `"rayleigh"` or `"learned_dissipation"` instead of the default `"none"` offers two architectural changes that allow for dissipation, where the conservation guarantee is replaced by $\dot{E} \leq 0$. 

#### Rayleigh dissipation

To account for dissipation, the term $D(q, \dot q)$ is added to the Euler-Lagrange equation. The resulting equations of motion are,

$$
\frac{d}{dt}\left(\frac{\partial L}{\partial \dot q_i}\right) = \frac{\partial L}{\partial q_i} - \frac{\partial D}{\partial \dot q_i}
$$

which enters the solve as one more term on the right-hand side,

$$
\ddot q = \left(\frac{\partial^2 L}{\partial \dot q\, \partial \dot q^{\mathsf{T}}}\right)^{+}\left(\frac{\partial L}{\partial q} - \frac{\partial^2 L}{\partial \dot q\, \partial q^{\mathsf{T}}}\dot q - \frac{\partial D}{\partial \dot q}\right)
$$

With this additional term, the rate of change of the energy becomes,

$$
\dot E = -\sum_i \dot q_i \frac{\partial D}{\partial \dot q_i} \le 0
$$

The inequality is not an automatic guarantee, but must be enforced to ensure that the dissipative term only removes energy from the system. An example of a commonly used dissipation function is the quadratic form,

$$
D(q, \dot q) = \tfrac{1}{2}\gamma\,\dot q^{2}
\quad\Longrightarrow\quad
\dot E = -\gamma\,\dot q^{2} \le 0
$$

This form describes velocity-dependent damping, such as friction or air resistance. To satisfy the inequality the damping factor must be non-negative, $\gamma \ge 0$, which is enforced by bounding its learnable domain. Rayleigh is the dissipative counterpart of the hybrid known-term above and is supplied per system through `LagrangianConfig.dissipation_term`, and only its constants are learnable, given by `LearnableParam` entries in `LNNLossConfig.physics`. That makes $\gamma$ come out of training as a physically meaningful number, at the cost of having to know the damping structure in advance.


#### Learned dissipation


When the damping structure is not known, the whole dissipation operator can be learned instead. The dissipation is written as a quadratic form in the velocities,

$$
D(q, \dot q) = \tfrac{1}{2}\sum_{i,j=1}^{n} \dot q_i\,R_{ij}(q)\,\dot q_j
$$

where $R$ is the correction matrix that contains the dissipation of the system. In contrast to Rayleigh, the correction matrix $R$ can couple different degrees of freedom nonlinearly. The rate of change of the energy becomes,

$$
\dot E = -\sum_{i,j=1}^{n} R_{ij}\,\dot q_i\,\dot q_j \le 0
$$

where the inequality must be enforced rather than automatically guaranteed. To ensure this condition, the matrix $R$ must be symmetric positive semidefinite. However, since $R$ is learned by the network, it is not guaranteed that $R$ is positive semidefinite. To enforce positive semidefiniteness, the network instead learns a lower triangular matrix $L$, from which $R$ is constructed,

$$
R = L L^{\mathsf{T}}, \qquad L \ \text{lower triangular}
$$

This guarantees that $v^{\mathsf{T}} L L^{\mathsf{T}} v = \lVert L^{\mathsf{T}} v \rVert^2 \ge 0, \quad \forall v \in \mathbb{R}^n$. The matrix $L$ is read off a second slice of the same network's output, evaluated at the same input with the velocity slot forced to zero so that $R$ depends on $q$ alone, where index $0$ is the Lagrangian and the next $k(k+1)/2$ entries are the lower-triangular entries of $L$, where $k = n_{\text{dof}}$ and $n_{\text{dof}}$ is `LagrangianConfig.n_dof`. 

#### Choosing between them

| | Rayleigh | Learned dissipation |
|---|---|---|
| Damping form | supplied by the system | learned |
| Learned quantity | scalar constants | the $k(k+1)/2$ entries of a lower-triangular $L$, from which $R = LL^{\mathsf{T}}$ |
| Interpretability | high, $\gamma$ is a physical number | low, $R$ is a black box |
| Generality | a fixed functional form of $D$ | $R(q)$ varies with configuration, so it can couple degrees of freedom |
| Network output width | $1$ | $1 + k(k+1)/2$ |

Learned dissipation is strictly the more general of the two. Rayleigh is the better choice when the damping structure is actually known, because it recovers an interpretable parameter instead of a black box.

## Rollouts

In contrast to the Hamiltonian rollouts, the Lagrangian only uses [RK4](hnn.md#rk4-default) and so `LagrangianConfig` has no `integrator` field. The scheme itself is the same one described there: `NeuralNetwork.trajectories()` integrates whatever vector field `dynamics()` hands it, so only the field differs, being the Euler-Lagrange solve rather than the symplectic gradient. 


## Advantages

Similar to an HNN, conservation of energy is exact, but is represented by Noether's identity and holds only where the Euler-Lagrange equations are solved exactly. Although canonical momenta are not required, the information is not free here either. This is because the equations of motion require inverting the Hessian, which may be singular. In this framework, a pseudo-inverse is used, which equals the exact inverse wherever the Hessian is nonsingular and only weakens the conservation guarantee when it degenerates. Like HNN it carries one physics term rather than PINN's three, so the tuning burden is low. Moreover, other conservation laws may be inserted either via a PINN-like soft constraint in the loss function, or strictly enforced by transforming the inputs. 

## Disadvantages

Any system must be expressible as a Lagrangian, otherwise the learnable scalar has no meaning and the dynamics cannot be extracted. 

The cost is concentrated in one place and it is significant. The Hessian solve makes every evaluation more expensive than an HNN's gradient, training backpropagates through third derivatives, and the whole thing is numerically fragile enough to need a smooth activation and a pseudo-inverse solve before it will train at all. Like the HNN it applies only to systems expressible in this formalism, and it shares the requirement for labeled derivatives at every training point, with no equivalent of PINN's ability to work from sparse observations alone.
