import numpy as np


class mass_spring_data_generator:
    def __init__(self, config, seed):
        self.config = config
        self.omega  = config.system.system_params["omega"]
        self.seed = seed
        if config.type not in ("field", "trajectory"):
            raise TypeError("Insert a known data generating type like field or trajectory.")

    def add_noise(self, rng, arr, sigma):
        return (arr + rng.normal(0.0, sigma, size=arr.shape))

    def generate_field_data(self):
        rng = np.random.default_rng(self.seed)
        q = rng.uniform(*self.config.q_range, size=self.config.field_N)
        p = rng.uniform(*self.config.p_range, size=self.config.field_N)

        X = np.stack([q, p], axis=-1)
        y = np.stack([p, -(self.omega ** 2) * q], axis=-1)

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
        q = q0 * np.cos(self.omega * t) + (p0 / self.omega) * np.sin(self.omega * t)
        p = -q0 * self.omega * np.sin(self.omega * t) + p0 * np.cos(self.omega * t)
        return np.stack([q, p], axis=-1)

    def hamiltonian(self, state, coords="canonical"):
        q, p = state[..., 0], state[..., 1]
        return 0.5 * (p ** 2 + (self.omega ** 2) * q ** 2)


def make_data(config):
    return mass_spring_data_generator(config, config.system.seed)
