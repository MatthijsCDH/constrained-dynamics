# Methods

| | Learns | Physics enters via |
|---|---|---|
| [MLP](mlp.md) | `state → state_dot` or `t → state` | Nowhere |
| [PINN](pinn.md) | `t → state` | A soft residual penalty in the loss |
| [HNN](hnn.md) | `(q, p) → H` | A symplectic gradient |
| [LNN](lnn.md) | `(q, q̇) → L` | An Euler-Lagrange solve |

