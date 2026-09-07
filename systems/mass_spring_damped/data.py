import numpy as np


class mass_spring_damped_data_generator:
    def __init__(self, config, seed):
        self.data_cfg   = config.data
        self.system_cfg = config.system
        self.omega      = config.system.system_params["omega"]
        self.gamma      = config.system.system_params["gamma"]
        self.seed       = seed
        if self.data_cfg.type not in ("field", "trajectory"):
            raise TypeError("Insert a known data generating type like field or trajectory.")

    def add_noise(self, rng, arr, sigma):
        scale = np.std(arr, axis=0, keepdims=True)
        return arr + rng.normal(0.0, sigma, size=arr.shape) * scale

    def generate_field_data(self):
        rng = np.random.default_rng(self.seed)
        q = rng.uniform(*self.data_cfg.q_range, size=self.data_cfg.field_N)
        p = rng.uniform(*self.data_cfg.p_range, size=self.data_cfg.field_N)

        X = np.stack([q, p], axis=-1)
        y = np.stack([p, -(self.omega ** 2) * q - self.gamma * p], axis=-1)

        X = self.add_noise(rng, X, self.data_cfg.field_sigma_x)
        y = self.add_noise(rng, y, self.data_cfg.field_sigma_y)
        return X, y

    def generate_trajectory_data(self):
        q0, p0 = self.data_cfg.initial_conditions
        rng = np.random.default_rng(self.seed)
        t   = np.sort(rng.uniform(0.0, self.data_cfg.period, size=self.data_cfg.trajectory_N))

        traj = self.true_trajectory(q0, p0, t, coords="canonical")
        X = t[:, None]
        y = traj[:, :1]

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

    def true_trajectory(self, q0, p0, t, coords="canonical"):
        t = np.asarray(t, dtype=np.float32)

        omega_d = np.sqrt(self.omega ** 2 - (self.gamma / 2) ** 2)
        exp_decay = np.exp(-self.gamma * t / 2)
        cos = np.cos(omega_d * t)
        sin= np.sin(omega_d * t)

        A = q0
        B = (p0 + self.gamma * q0 / 2) / omega_d

        q = exp_decay * (A * cos + B * sin)
        p = exp_decay * ((omega_d * B - self.gamma / 2 * A) * cos + (-omega_d * A - self.gamma / 2 * B) * sin)

        return np.stack([q, p], axis=-1)

    def hamiltonian(self, state, coords="canonical"):
        q, p = state[..., 0], state[..., 1]
        return 0.5 * (p ** 2 + (self.omega ** 2) * q ** 2)


def make_data(config):
    return mass_spring_damped_data_generator(config, config.system.seed)
