import numpy as np


class mass_spring_damped_data_generator:
    def __init__(self, config, seed):
        self.config = config
        self.omega  = config.system.system_params["omega"]
        self.gamma  = config.system.system_params["gamma"]
        self.seed   = seed
        if config.type not in ("field", "trajectory"):
            raise TypeError("Insert a known data generating type like field or trajectory.")

    def add_noise(self, rng, arr, sigma):
        return (arr + rng.normal(0.0, sigma, size=arr.shape))

    def generate_field_data(self):
        rng = np.random.default_rng(self.seed)
        q = rng.uniform(*self.config.q_range, size=self.config.field_N)
        p = rng.uniform(*self.config.p_range, size=self.config.field_N)

        X = np.stack([q, p], axis=-1)
        y = np.stack([p, -(self.omega ** 2) * q - self.gamma * p], axis=-1)

        X = self.add_noise(rng, X, self.config.field_sigma)
        y = self.add_noise(rng, y, self.config.field_sigma)
        return X, y

    def generate_trajectory_data(self):
        q0, p0 = self.config.boundary_conditions
        rng = np.random.default_rng(self.seed)
        t   = np.sort(rng.uniform(0.0, self.config.period, size=self.config.trajectory_N))

        traj = self.true_trajectory(q0, p0, t, coords="canonical")
        X = t[:, None]
        y = traj[:, :1]

        X = self.add_noise(rng, X, self.config.trajectory_sigma)
        y = self.add_noise(rng, y, self.config.trajectory_sigma)
        return X, y

    def build_training_data(self):
        if self.config.type == "field":
            return self.generate_field_data()
        elif self.config.type == "trajectory":
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
