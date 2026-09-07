import argparse
import dataclasses
import importlib
import pkgutil
import json
import os
import pickle 
import numpy as np
import time
import sys
import optuna

from plots import (plot_mse_band_over_time, plot_drift_band_over_time)

def family_of(method):
    return method.split("_")[0]

def load_config_module(system):
    return importlib.import_module(f"systems.{system}.{system}_configurations")

def list_experiments():
    import systems
    found = False
    for info in sorted(pkgutil.iter_modules(systems.__path__), key=lambda m: m.name):
        try:
            config_list = load_config_module(info.name)
        except Exception:
            continue
        if hasattr(config_list, "CONFIGS"):
            found = True
            print(f"{info.name}: {', '.join(config_list.CONFIGS)}")
    if not found:
        print("No tuning modules found (need a systems/<name>/<name>_tune.py with a SEARCH_SPACES dict).")

class Metrics:
    def __init__(self, model, config, horizon, n_warmups = 3):
        self.model  = model
        self.config = config
        self.system = config.data.make_data(config)
        self.horizon = horizon
        self.elapsed = self.trajectories(horizon, n_warmups)
        
    def trajectories(self, horizon, n_warmups = 3):
        self.t = np.linspace(0, horizon * self.config.system.period, int(500 * horizon))
        self.true_states = self.system.true_trajectory(
            self.config.system.q0, self.config.system.p0, self.t, coords=self.model.coords
        )

        if self.model.type == "trajectory":
            call = lambda: self.model.trajectories(self.t)
        else:
            call = lambda: self.model.trajectories(self.t, state0=self.true_states[0])

        for _ in range(n_warmups):
            call()

        start = time.perf_counter()
        self.learned_states = call()
        elapsed = time.perf_counter() - start
        return elapsed

    def mse_over_time(self):
        mse_series = np.mean((self.learned_states - self.true_states) ** 2, axis=-1)
        return mse_series, float(np.mean(mse_series))

    def mse_split(self, t_split):
        sq       = np.mean((self.learned_states - self.true_states) ** 2, axis=-1)
        inside   = self.t <= t_split
        outside  = ~inside
        in_mean  = float(np.mean(sq[inside]))  if np.any(inside)  else float("nan")
        out_mean = float(np.mean(sq[outside])) if np.any(outside) else float("nan")
        return in_mean, out_mean

    def energy_drift_over_time(self):
        E_true    = np.asarray(self.system.hamiltonian(self.true_states, coords=self.model.coords))
        E_learned = np.asarray(self.system.hamiltonian(self.learned_states, coords=self.model.coords))

        drift_series = np.abs(E_learned - E_true) / (np.abs(E_true[0]) + 1e-12)
        return drift_series, float(np.mean(drift_series))

    def accuracy_horizon(self, mse_series, tolerance, window_periods=0.25):
        period    = self.config.system.period
        zero_mse  = float(np.mean(np.asarray(self.true_states) ** 2))
        baseline  = float(np.mean(mse_series[self.t <= period]))
        threshold = baseline + tolerance * zero_mse

        w = max(1, int(round(window_periods * period / (self.t[1] - self.t[0]))))
        if w > len(mse_series):
            return None

        cumulative = np.concatenate([[0.0], np.cumsum(mse_series)])
        rolling    = (cumulative[w:] - cumulative[:-w]) / w

        exceeded = np.where(rolling > threshold)[0]
        if len(exceeded) == 0:
            return None
        return float(self.t[w - 1 + exceeded[0]]) / period

    def relative_learned_parameter_error(self):
        learned_parameters = self.model.physics_loss.describe_physics(self.model.state.params)
        if not learned_parameters:
            return {}

        true_parameters = dict(self.config.system.system_params)
        for name, value in (("q0", self.config.system.q0), ("p0", self.config.system.p0)):
            if name not in true_parameters and np.ndim(value) == 0:
                true_parameters[name] = float(value)

        errors = {}
        for name, learned in learned_parameters.items():
            if name not in true_parameters:
                continue
            true = true_parameters[name]
            diff = abs(float(learned) - true)
            errors[name] = float(diff / abs(true)) if abs(true) > 1e-8 else float(diff)
        return errors


class Evaluation:
    def __init__(self, system, method, n_seeds, sigmas, epochs):
        self.system  = system
        self.method  = method
        self.n_seeds = n_seeds
        self.seeds   = list(range(n_seeds))
        self.sigmas  = sigmas
        self.epochs  = epochs

        self.config  = load_config_module(system)
        self.radius_truth = {}
        self.horizon = self.config.CONFIGS[method].evaluation.horizons[0]

    def find_max_horizon(self, model, config, metrics):
        tolerance = config.evaluation.horizon_tolerance
        window    = config.evaluation.horizon_window

        mse_series, _ = metrics.mse_over_time()
        max_horizon   = metrics.accuracy_horizon(mse_series, tolerance, window)
        if max_horizon is not None:
            return max_horizon

        for horizon in config.evaluation.horizons:
            if horizon <= metrics.horizon:
                continue
            metrics       = Metrics(model, config, horizon, n_warmups=0)
            mse_series, _ = metrics.mse_over_time()
            max_horizon   = metrics.accuracy_horizon(mse_series, tolerance, window)
            if max_horizon is not None:
                return max_horizon
        return None
    
    def find_max_radius(self, model, config, metrics):
        if model.type == "trajectory":
            return None
        tolerance = config.evaluation.horizon_tolerance
        q0 = np.atleast_1d(np.asarray(config.system.q0, dtype=float))
        p0 = np.atleast_1d(np.asarray(config.system.p0, dtype=float))

        best = None
        for radius in config.evaluation.radii:
            key = (radius, model.coords, len(metrics.t))
            if key not in self.radius_truth:
                system = config.data.make_data(config)
                self.radius_truth[key] = np.asarray(
                    system.true_trajectory(radius * q0, radius * p0, metrics.t, coords=model.coords)
                )
            true    = self.radius_truth[key]
            learned = np.asarray(model.trajectories(metrics.t, state0=true[0]))
            if not np.all(np.isfinite(learned)):
                break
            score = float(np.mean((learned - true) ** 2)) / float(np.mean(true ** 2))
            if score > tolerance:
                break
            best = radius
        return best

    def seed_sweep(self, config):
        from model.network import NeuralNetwork
        results = []
        start_time = time.time()

        print("\n" * 4)
        for seed in self.seeds:
            system_config   = dataclasses.replace(config.system, seed=seed)
            training_config = dataclasses.replace(config.training, live_metrics=False, static_metrics=False)
            seed_config     = dataclasses.replace(config, system=system_config, training = training_config)

            gen   = seed_config.data.make_data(seed_config)
            X, y  = gen.build_training_data()
            model = NeuralNetwork(seed_config, X, y)
            train_start = time.perf_counter()
            model.train()
            train_time = time.perf_counter() - train_start

            metrics = Metrics(model, seed_config, self.horizon)
            mse_series, mse_mean = metrics.mse_over_time()
            drift_series, drift_mean = metrics.energy_drift_over_time()
            param_errors = metrics.relative_learned_parameter_error()
            physics_params = model.physics_loss.describe_physics(model.state.params) or {}
            max_horizon =  self.find_max_horizon(model, seed_config, metrics)
            max_radius  =  self.find_max_radius(model, seed_config, metrics)
            mse_early, mse_late = metrics.mse_split(seed_config.system.period)

            results.append({
                "seed": seed, "t": metrics.t,
                "mse_series": mse_series, "mse_mean": mse_mean,
                "mse_early": mse_early, "mse_late": mse_late,
                "drift_series": drift_series, "drift_mean": drift_mean,
                "physics_params": physics_params, "physics_params_error": param_errors,
                "inference_time": metrics.elapsed, "max_horizon": max_horizon,
                "max_radius": max_radius, "train_time": train_time,
                "zero_mse": float(np.mean(np.asarray(metrics.true_states) ** 2)),
            })

            self.print_live(seed, results, start_time)
        return results

    def load_tuned_config(self, method):
        base_config = self.config.CONFIGS[method]
        base_config = dataclasses.replace(base_config, network=dataclasses.replace(base_config.network, epochs=self.epochs))
        path = os.path.join("systems", self.system, family_of(method), "best_configs", f"{method}.json")
        self.load_tuned = True
        if not os.path.exists(path):
            self.load_tuned = False
            return base_config

        with open(path) as f:
            best_params = json.load(f)

        tune_mod = importlib.import_module(f"systems.{self.system}.{self.system}_tune")
        search_space = tune_mod.SEARCH_SPACES[method]
        trial = optuna.trial.FixedTrial(best_params)

        return search_space(trial, base_config)

    def sigma_sweep(self, method, resume=False):
        base_config = self.load_tuned_config(method)
        sigma_results = {}
        if resume:
            try:
                sigma_results = self.load_results(method)
                print(f"Resuming: {len(sigma_results)} sigma(s) already saved {sorted(sigma_results)}")
            except FileNotFoundError:
                pass

        for sigma in self.sigmas:
            if sigma in sigma_results:
                print(f"Skipping sigma {sigma}, already present")
                continue

            data_config  = dataclasses.replace(base_config.data, field_sigma_y=sigma, trajectory_sigma_y=sigma)
            sigma_config = dataclasses.replace(base_config, data=data_config)

            results = self.seed_sweep(sigma_config)
            summary = self.summarise_seeds(results, sigma_config)
            sigma_results[sigma] = {"seeds": results, "summary": summary}
            self.save_results(sigma_results, method, quiet=True)
            self.print_results(summary, sigma)
        return sigma_results

    def summarise_seeds(self, results, config):
        mse_series_mean   = np.stack([r["mse_series"] for r in results])
        drift_series_mean = np.stack([r["drift_series"] for r in results])

        mse_means         = np.array([r["mse_mean"] for r in results])
        mse_early_means   = np.array([r["mse_early"] for r in results])
        mse_late_means    = np.array([r["mse_late"] for r in results])
        drift_means       = np.array([r["drift_mean"] for r in results])
        inference_times   = np.array([r["inference_time"] for r in results])
        train_times       = np.array([r["train_time"] for r in results])

        ceiling = max(config.evaluation.horizons)
        horizons = np.array([ceiling if r["max_horizon"] is None else r["max_horizon"] for r in results])
        n_censored = sum(r["max_horizon"] is None for r in results)
        exceeded_horizon = n_censored > len(results) / 2
        max_horizon = float(np.median(horizons))
        max_horizon_std = float(np.std(horizons))
        radii = [r["max_radius"] for r in results if r["max_radius"] is not None]

        param_names = results[0]["physics_params"].keys()
        physics_summary = {
            name: {
                "mean": float(np.mean([r["physics_params"][name] for r in results])),
                "std":  float(np.std([r["physics_params"][name] for r in results])),
            }
            for name in param_names
        }

        error_names = results[0]["physics_params_error"].keys()
        physics_error_summary = {
            name: {
                "mean": float(np.mean([r["physics_params_error"][name] for r in results])),
                "std":  float(np.std([r["physics_params_error"][name] for r in results])),
            }
            for name in error_names
        }
        return {
            "t":    results[0]["t"],
            "mse_series_mean":      np.mean(mse_series_mean, axis=0),
            "mse_series_std":       np.std(mse_series_mean, axis=0),
            "mse_mean_mean":        float(np.mean(mse_means)),
            "mse_mean_std":         float(np.std(mse_means)),
            "mse_early_mean":       float(np.mean(mse_early_means)),
            "mse_early_std":        float(np.std(mse_early_means)),
            "mse_late_mean":        float(np.mean(mse_late_means)),
            "mse_late_std":         float(np.std(mse_late_means)),
            "drift_series_mean":    np.mean(drift_series_mean, axis=0),
            "drift_series_std":     np.std(drift_series_mean, axis=0),
            "drift_mean_mean":      float(np.mean(drift_means)),
            "drift_mean_std":       float(np.std(drift_means)),
            "physics_params":       physics_summary,
            "physics_params_error": physics_error_summary,
            "inference_time_mean":  float(np.mean(inference_times)),
            "inference_time_std":   float(np.std(inference_times)),
            "train_time_mean":      float(np.mean(train_times)),
            "train_time_std":       float(np.std(train_times)),
            "max_horizon":          max_horizon,
            "max_horizon_std":      max_horizon_std,
            "max_horizons":         [r["max_horizon"] for r in results],
            "n_censored":           int(n_censored),
            "exceeded_horizon":     exceeded_horizon,
            "max_radius":           float(np.median(radii)) if radii else None,
            "max_radius_std":       float(np.std(radii)) if radii else None,
            "zero_mse":             float(results[0]["zero_mse"]),
        }
        
    def save_results(self, sigma_results, method = None, quiet = False):
        method = method or self.method
        out_dir = os.path.join("systems", self.system, family_of(method), "evaluation")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{method}.pkl")
        with open(path, "wb") as f:
            pickle.dump(sigma_results, f)
        if not quiet:
            print(f"Saved results to {path}")
        return path

    def load_results(self, method):
        path = os.path.join("systems", self.system, family_of(method), "evaluation", f"{method}.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)

    def run(self, resume=False):
        self.load_tuned_config(self.method)
        self.print_static_ui()
        results = self.sigma_sweep(self.method, resume=resume)
        self.save_results(results)

        plots_dir = os.path.join("systems", self.system, family_of(self.method), "plots")
        os.makedirs(plots_dir, exist_ok=True)
        for sigma, data in results.items():
            plot_mse_band_over_time(
                data["summary"], sigma, self.system, self.method,
                save_path=os.path.join(plots_dir, f"mse_{self.method}_sigma{sigma}.png"),
                show=False,
            )
            plot_drift_band_over_time(
                data["summary"], sigma, self.system, self.method,
                save_path=os.path.join(plots_dir, f"drift_{self.method}_sigma{sigma}.png"),
                show=False,
            )
        return results

    @staticmethod
    def format_time(seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        elif m > 0:
            return f"{m}m {s}s"
        else:
            return f"{s}s"

    @staticmethod
    def build_bar(progress, width=40):
        filled = int(progress * width)
        return "█" * filled + "░" * (width - filled)
    
    def print_static_ui(self):
        print(f"\n{'='*60}")
        print(f"{'Evaluation Sweep':^60}")
        print(f"\n{'='*60}")
        print(f" System         : {self.system}")
        print(f" Method         : {self.method}")
        print(f" Number of seeds: {self.n_seeds}")
        print(f" Sigmas         : {self.sigmas}")
        print(f" Epochs         : {self.epochs}")
        print(f" Horizon        : {self.horizon:.4g} x period")
        print(f" Tuned params   : {self.load_tuned}")
        print(f"\n{'='*60}")

    def print_live(self, seed, results, start_time):
        elapsed = time.time() - start_time
        it_s    = (seed + 1) / elapsed if elapsed > 0 else 0
        eta     = (self.n_seeds - (seed + 1)) / it_s if it_s > 0 else 0
        bar     = self.build_bar((seed + 1) / self.n_seeds)

        mse     = results[-1]["mse_mean"]
        drift   = results[-1]["drift_mean"]
        lines = [f" Seed {seed + 1}/{self.n_seeds}  {bar}"]
        lines.append(f" {'MSE':<14}: {mse:.3e}")
        lines.append(f" {'drift':<14}: {drift:.3e}")
        lines.append(
            f" speed={it_s:.2f} it/s  "
            f"elapsed={self.format_time(elapsed)}  "
            f"ETA={self.format_time(eta)}"
        )
        
        n = len(lines)
        sys.stdout.write(f"\033[{n}F\033[J")
        print("\n".join(lines))

    def print_results(self, results, sigma):
        title = f"{self.system} / {self.method}  (sigma={sigma})"
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}")
        print(f" {'MSE':<16}: {results['mse_mean_mean']:.3e} ± {results['mse_mean_std']:.3e}")
        print(f" {'Energy drift':<16}: {results['drift_mean_mean']:.3e} ± {results['drift_mean_std']:.3e}")
        print(f" {'Inference time':<16}: {results['inference_time_mean']:.3e} ± {results['inference_time_std']:.3e}")
        print(f" {'Training time':<16}: {results['train_time_mean']:.3e} ± {results['train_time_std']:.3e}")

        horizon_str = f"{results['max_horizon']:.3g}x period" + ("+" if results["exceeded_horizon"] else "")
        print(f" {'Max horizon':<16}: {horizon_str}")

        if results["physics_params"]:
            print(f" {'physics_params':<16}:")
            for name, stat in results["physics_params"].items():
                err = results["physics_params_error"].get(name)
                err_str = f"  (rel err={err['mean']:.3g} ± {err['std']:.3g})" if err else ""
                print(f" {name:<14}: {stat['mean']:.4g} ± {stat['std']:.4g}{err_str}")
    
def main(system, method, n_seeds, sigmas, epochs, resume=False):
    evaluation = Evaluation(system, method, n_seeds, sigmas, epochs)
    results = evaluation.run(resume=resume)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation of a (system, method)")
    parser.add_argument("system", nargs="?", help="System name")
    parser.add_argument("method", nargs="?", help="Method name")
    parser.add_argument("--n_seeds", type=int, default= 50, help="Number of seeds")
    parser.add_argument("--sigmas", type=float, nargs="+", default= [0.0, 0.05, 0.1, 0.2, 0.3, 0.4], help="Noise in the data to sweep")
    parser.add_argument("--epochs", type=int, default= 5000, help="Number of epochs per trial")
    parser.add_argument("--list", action="store_true", help="List of available (system, method) pairs")
    parser.add_argument("--resume", action="store_true", help="Skip sigmas already saved in <method>.pkl")
    args = parser.parse_args()

    if args.list or args.system is None or args.method is None:
        list_experiments()
    else:
        config_list = load_config_module(args.system)
        if args.method not in config_list.CONFIGS:
            raise SystemExit(f"Unknown method in configurations")
        main(args.system, args.method, args.n_seeds, args.sigmas, args.epochs, args.resume)