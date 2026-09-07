import diffrax
import jax
import jax.numpy as jnp
import numpy as np

class charged_particle_data_generator:
    def __init__(self, config, seed):
        self.data_cfg   = config.data
        self.system_cfg = config.system
        params          = config.system.system_params
        self.m          = params["m"]
        self.e          = params["e"]
        self.B          = params["B"]
        self.k          = params["k"]
        self.alpha      = params["alpha"]
        self.seed       = seed
        if self.data_cfg.type not in ("field", "trajectory"):
            raise TypeError("Insert a known data generating type like field or trajectory.")
        if self.data_cfg.coords not in ("canonical", "mechanical", "velocity"):
            raise TypeError("Insert a known coordinate system: canonical, mechanical or velocity.")

    def field_strength(self, r, xp=np):
        return self.B * (1.0 + self.alpha * (r[..., 0] ** 2 + r[..., 1] ** 2))

    def vector_potential(self, r, xp=np):
        s = self.B * (0.5 + 0.25 * self.alpha * (r[..., 0] ** 2 + r[..., 1] ** 2))
        return -s * r[..., 1], s * r[..., 0]

    def acceleration(self, r, v, xp=np):
        bz = self.field_strength(r, xp)
        ax = (self.e * bz * v[..., 1] - self.k * r[..., 0]) / self.m
        ay = (-self.e * bz * v[..., 0] - self.k * r[..., 1]) / self.m
        return ax, ay

    def to_canonical(self, r, v, xp=np):
        ax, ay = self.vector_potential(r, xp)
        return self.m * v[..., 0] + self.e * ax, self.m * v[..., 1] + self.e * ay

    def to_velocity(self, r, p, xp=np):
        ax, ay = self.vector_potential(r, xp)
        return (p[..., 0] - self.e * ax) / self.m, (p[..., 1] - self.e * ay) / self.m

    def add_noise(self, rng, arr, sigma):
        scale = np.std(arr, axis=0, keepdims=True)
        return arr + rng.normal(0.0, sigma, size=arr.shape) * scale

    def generate_field_data(self):
        rng = np.random.default_rng(self.seed)
        n   = self.data_cfg.field_N
        r   = np.stack([rng.uniform(*self.data_cfg.q_range, size=n),
                        rng.uniform(*self.data_cfg.q_range, size=n)], axis=-1)
        v   = np.stack([rng.uniform(*self.data_cfg.p_range, size=n),
                        rng.uniform(*self.data_cfg.p_range, size=n)], axis=-1)

        ax, ay = self.acceleration(r, v)

        if self.data_cfg.coords == "velocity":
            X = np.concatenate([r, v], axis=-1)
            y = np.stack([v[..., 0], v[..., 1], ax, ay], axis=-1)
        elif self.data_cfg.coords == "mechanical":
            X = np.stack([r[..., 0], r[..., 1], self.m * v[..., 0], self.m * v[..., 1]], axis=-1)
            y = np.stack([v[..., 0], v[..., 1], self.m * ax, self.m * ay], axis=-1)
        else:
            px, py = self.to_canonical(r, v)
            x, yy = r[..., 0], r[..., 1]
            sc = self.B * (0.5 + 0.25 * self.alpha * (x ** 2 + yy ** 2))
            ds_dx = 0.5 * self.B * self.alpha * x
            ds_dy = 0.5 * self.B * self.alpha * yy
            pdot_x = self.e * (v[..., 0] * (-yy * ds_dx) + v[..., 1] * (sc + x * ds_dx)) - self.k * x
            pdot_y = self.e * (v[..., 0] * (-sc - yy * ds_dy) + v[..., 1] * (x * ds_dy)) - self.k * yy
            X = np.stack([r[..., 0], r[..., 1], px, py], axis=-1)
            y = np.stack([v[..., 0], v[..., 1], pdot_x, pdot_y], axis=-1)

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
        r = state[:2]
        v = state[2:]
        ax, ay = self.acceleration(r, v, xp=jnp)
        return jnp.stack([v[0], v[1], ax, ay])

    def true_trajectory(self, q0, p0, t, coords="canonical"):
        q0 = np.atleast_1d(np.asarray(q0, dtype=np.float32))
        p0 = np.atleast_1d(np.asarray(p0, dtype=np.float32))
        t  = np.asarray(t, dtype=np.float32)

        if coords == "canonical":
            v0 = np.stack(self.to_velocity(q0, p0), axis=-1)
        elif coords == "mechanical":
            v0 = p0 / self.m
        else:
            v0 = p0

        state0 = jnp.asarray(np.concatenate([q0, np.atleast_1d(v0)]), dtype=jnp.float32)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.velocity_eom),
            diffrax.Dopri5(),
            t0=float(t[0]), t1=float(t[-1]), dt0=None, y0=state0,
            stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-7),
            saveat=diffrax.SaveAt(ts=jnp.asarray(t)),
            max_steps=1_000_000,
        )
        states = np.asarray(sol.ys)
        r, v = states[:, :2], states[:, 2:]

        if coords == "velocity":
            return np.concatenate([r, v], axis=-1)
        if coords == "mechanical":
            return np.concatenate([r, self.m * v], axis=-1)
        px, py = self.to_canonical(r, v)
        return np.stack([r[:, 0], r[:, 1], px, py], axis=-1)

    def hamiltonian(self, state, coords="canonical"):
        state = np.asarray(state)
        r = state[..., :2]
        if coords == "canonical":
            v = np.stack(self.to_velocity(r, state[..., 2:]), axis=-1)
        elif coords == "mechanical":
            v = state[..., 2:] / self.m
        else:
            v = state[..., 2:]
        kinetic   = 0.5 * self.m * (v[..., 0] ** 2 + v[..., 1] ** 2)
        potential = 0.5 * self.k * (r[..., 0] ** 2 + r[..., 1] ** 2)
        return kinetic + potential


def make_data(config):
    return charged_particle_data_generator(config, config.system.seed)
