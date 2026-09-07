import diffrax
import jax
import jax.numpy as jnp
import numpy as np


class double_pendulum_data_generator:
    def __init__(self, config, seed):
        self.data_cfg   = config.data
        self.system_cfg = config.system
        params          = config.system.system_params
        self.m1         = params["m1"]
        self.m2         = params["m2"]
        self.l1         = params["l1"]
        self.l2         = params["l2"]
        self.g          = params["g"]
        self.seed       = seed
        if self.data_cfg.type not in ("field", "trajectory"):
            raise TypeError("Insert a known data generating type like field or trajectory.")

    def mass_matrix(self, q, xp=np):
        delta = q[..., 0] - q[..., 1]
        a = (self.m1 + self.m2) * self.l1 ** 2 * xp.ones_like(delta)
        b = self.m2 * self.l1 * self.l2 * xp.cos(delta)
        c = self.m2 * self.l2 ** 2 * xp.ones_like(delta)
        return a, b, c

    def potential(self, q, xp=np):
        return (-(self.m1 + self.m2) * self.g * self.l1 * xp.cos(q[..., 0])
                - self.m2 * self.g * self.l2 * xp.cos(q[..., 1]))

    def dL_dq(self, q, qdot, xp=np):
        delta = q[..., 0] - q[..., 1]
        cross = self.m2 * self.l1 * self.l2 * qdot[..., 0] * qdot[..., 1] * xp.sin(delta)
        d1 = -cross - (self.m1 + self.m2) * self.g * self.l1 * xp.sin(q[..., 0])
        d2 = cross - self.m2 * self.g * self.l2 * xp.sin(q[..., 1])
        return d1, d2

    def damping_torque(self, qdot, xp=np):
        return xp.zeros_like(qdot[..., 0]), xp.zeros_like(qdot[..., 1])

    def acceleration(self, q, qdot, xp=np):
        a, b, c = self.mass_matrix(q, xp)
        delta   = q[..., 0] - q[..., 1]
        sin_d   = xp.sin(delta)
        d1, d2  = self.dL_dq(q, qdot, xp)
        r1, r2  = self.damping_torque(qdot, xp)

        f1 = d1 + self.m2 * self.l1 * self.l2 * sin_d * qdot[..., 0] * qdot[..., 1] \
             - self.m2 * self.l1 * self.l2 * sin_d * qdot[..., 1] ** 2 - r1
        f2 = d2 - self.m2 * self.l1 * self.l2 * sin_d * qdot[..., 0] * qdot[..., 1] \
             + self.m2 * self.l1 * self.l2 * sin_d * qdot[..., 0] ** 2 - r2

        det = a * c - b ** 2
        return (c * f1 - b * f2) / det, (a * f2 - b * f1) / det

    def to_canonical(self, q, qdot, xp=np):
        a, b, c = self.mass_matrix(q, xp)
        return a * qdot[..., 0] + b * qdot[..., 1], b * qdot[..., 0] + c * qdot[..., 1]

    def to_velocity(self, q, p, xp=np):
        a, b, c = self.mass_matrix(q, xp)
        det = a * c - b ** 2
        return (c * p[..., 0] - b * p[..., 1]) / det, (a * p[..., 1] - b * p[..., 0]) / det

    def add_noise(self, rng, arr, sigma):
        scale = np.std(arr, axis=0, keepdims=True)
        return arr + rng.normal(0.0, sigma, size=arr.shape) * scale

    def generate_field_data(self):
        rng   = np.random.default_rng(self.seed)
        n     = self.data_cfg.field_N
        q     = np.stack([rng.uniform(*self.data_cfg.q_range, size=n),
                          rng.uniform(*self.data_cfg.q_range, size=n)], axis=-1)
        qdot  = np.stack([rng.uniform(*self.data_cfg.p_range, size=n),
                          rng.uniform(*self.data_cfg.p_range, size=n)], axis=-1)

        qddot1, qddot2 = self.acceleration(q, qdot)

        if self.data_cfg.coords == "velocity":
            X = np.concatenate([q, qdot], axis=-1)
            y = np.stack([qdot[..., 0], qdot[..., 1], qddot1, qddot2], axis=-1)
        else:
            p1, p2 = self.to_canonical(q, qdot)
            pdot1, pdot2 = self.dL_dq(q, qdot)
            r1, r2 = self.damping_torque(qdot)
            X = np.stack([q[..., 0], q[..., 1], p1, p2], axis=-1)
            y = np.stack([qdot[..., 0], qdot[..., 1], pdot1 - r1, pdot2 - r2], axis=-1)

        X = self.add_noise(rng, X, self.data_cfg.field_sigma_x)
        y = self.add_noise(rng, y, self.data_cfg.field_sigma_y)
        return X, y

    def generate_trajectory_data(self):
        q0, p0 = self.data_cfg.initial_conditions
        rng = np.random.default_rng(self.seed)
        t   = np.sort(rng.uniform(0.0, self.data_cfg.period, size=self.data_cfg.trajectory_N))

        traj = self.true_trajectory(q0, p0, t, coords=self.data_cfg.coords)
        X = t[:, None]
        y = traj[:, :2]

        X = self.add_noise(rng, X, self.data_cfg.trajectory_sigma_x)
        y = self.add_noise(rng, y, self.data_cfg.trajectory_sigma_y)
        return X, y

    def build_training_data(self):
        if self.data_cfg.type == "field":
            return self.generate_field_data()
        elif self.data_cfg.type == "trajectory":
            return self.generate_trajectory_data()
        else:
            raise TypeError("Insert a known data generating type like field or trajectory.")

    def velocity_eom(self, t, state, args):
        q    = state[:2]
        qdot = state[2:]
        qddot1, qddot2 = self.acceleration(q, qdot, xp=jnp)
        return jnp.stack([qdot[0], qdot[1], qddot1, qddot2])

    def true_trajectory(self, q0, p0, t, coords="canonical"):
        q0 = np.atleast_1d(np.asarray(q0, dtype=np.float32))
        p0 = np.atleast_1d(np.asarray(p0, dtype=np.float32))
        t  = np.asarray(t, dtype=np.float32)

        if coords == "velocity":
            qdot0 = p0
        else:
            qdot0 = np.stack(self.to_velocity(q0, p0), axis=-1)

        state0 = jnp.asarray(np.concatenate([q0, np.atleast_1d(qdot0)]), dtype=jnp.float32)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.velocity_eom),
            diffrax.Dopri5(),
            t0=float(t[0]), t1=float(t[-1]), dt0=None, y0=state0,
            stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-7),
            saveat=diffrax.SaveAt(ts=jnp.asarray(t)),
            max_steps=1_000_000,
        )
        states = np.asarray(sol.ys)
        q, qdot = states[:, :2], states[:, 2:]

        if coords == "velocity":
            return np.concatenate([q, qdot], axis=-1)
        p1, p2 = self.to_canonical(q, qdot)
        return np.stack([q[:, 0], q[:, 1], p1, p2], axis=-1)

    def hamiltonian(self, state, coords="canonical"):
        state = np.asarray(state)
        q = state[..., :2]
        if coords == "velocity":
            qdot = state[..., 2:]
        else:
            qdot = np.stack(self.to_velocity(q, state[..., 2:]), axis=-1)
        a, b, c = self.mass_matrix(q)
        kinetic = 0.5 * (a * qdot[..., 0] ** 2
                         + 2.0 * b * qdot[..., 0] * qdot[..., 1]
                         + c * qdot[..., 1] ** 2)
        return kinetic + self.potential(q)


def make_data(config):
    return double_pendulum_data_generator(config, config.system.seed)
