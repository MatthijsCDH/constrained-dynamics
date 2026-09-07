import argparse
import dataclasses
import importlib
import pkgutil
import json
import os
import time
import numpy as np
import jax
import optuna

from benchmark import family_of
from configs.configurations import LNNTrainConfig, TuningConfig

def load_config_module(system):
    return importlib.import_module(f"systems.{system}.{system}_configurations")

def load_config_tune(system):
    return importlib.import_module(f"systems.{system}.{system}_tune")

def list_experiments():
    import systems
    found = False
    for info in sorted(pkgutil.iter_modules(systems.__path__), key=lambda m: m.name):
        try:
            config_list = load_config_module(info.name)
            tune_list   = load_config_tune(info.name)
        except Exception:
            continue
        if hasattr(config_list, "CONFIGS") and hasattr(tune_list, "SEARCH_SPACES"):
            found = True
            print(f"{info.name}: {', '.join(tune_list.SEARCH_SPACES)}")
    if not found:
        print("No tuning modules found (need a systems/<name>/<name>_tune.py with a SEARCH_SPACES dict).")


class Tuner:
    def __init__(self, system, method, n_trials, epochs, n_seeds, tuning):
        self.system   = system
        self.method   = method
        self.n_trials = n_trials
        self.n_seeds  = n_seeds
        self.tuning   = tuning
        self.study    = None

        self.base_config  = load_config_module(system).CONFIGS[method]
        self.search_space = load_config_tune(system).SEARCH_SPACES[method]
        self.epochs       = epochs if epochs is not None else self.base_config.network.epochs

        config       = self.base_config
        coords       = "velocity" if isinstance(config, LNNTrainConfig) else "canonical"
        horizon      = config.evaluation.horizons[0]
        self.rollout = config.data.type != "trajectory"
        generator    = config.data.make_data(config)
        self.t       = np.linspace(0.0, horizon * config.system.period, int(500 * horizon))
        self.true    = np.asarray(
            generator.true_trajectory(config.system.q0, config.system.p0, self.t, coords=coords)
        )
        self.zero    = float(np.mean(self.true ** 2))

    def probe(self, model):
        learned = model.trajectories(self.t, state0=self.true[0]) if self.rollout else model.trajectories(self.t)
        learned = np.asarray(learned)
        if not np.all(np.isfinite(learned)):
            return 1.0
        pointwise = np.mean((learned - self.true) ** 2, axis=-1)
        return min(float(np.mean(pointwise)) / self.zero, 1.0)

    def fit(self, model):
        history = model.history["val"] if model.do_validation else model.history["train"]
        primary = float(np.mean(np.asarray(history[-1]), axis=0)[0])
        target  = np.asarray(model.y_val_batched if model.do_validation else model.y_train_batched)
        scale   = float(np.mean(target ** 2))
        if not np.isfinite(primary) or scale <= 0.0:
            return 1.0
        return min(primary / scale, 1.0)

    def seed_config(self, config, seed):
        network  = dataclasses.replace(config.network, epochs=self.epochs, save_filepath=None)
        training = dataclasses.replace(config.training, live_metrics=False, static_metrics=False)
        system   = dataclasses.replace(config.system, seed=seed)
        return dataclasses.replace(config, network=network, training=training, system=system)

    def train_per_seed(self, config):
        from model.network import NeuralNetwork
        gen   = config.data.make_data(config)
        X, y  = gen.build_training_data()
        model = NeuralNetwork(config, X, y)
        model.train_setup()

        total = config.network.epochs
        fit, consecutive, stopped = 1.0, 0, 0
        for epoch in range(1, total + 1):
            model.train_epoch(epoch)
            if epoch % self.tuning.check_every and epoch != total:
                continue
            fit         = self.fit(model)
            stopped     = epoch
            consecutive = consecutive + 1 if fit >= self.tuning.collapse_fraction else 0
            if consecutive >= self.tuning.collapse_patience:
                break

        collapsed = bool(consecutive >= self.tuning.collapse_patience)
        return {
            "seed":       config.system.seed,
            "score":      1.0 if collapsed else self.probe(model),
            "fit":        fit,
            "collapsed":  collapsed,
            "stopped_at": stopped,
            "physics":    model.physics_loss.describe_physics(model.state.params) or {},
        }

    def record(self, trial, seeds, start, abandoned):
        trial.set_user_attr("seeds",       seeds)
        trial.set_user_attr("n_collapsed", int(sum(s["collapsed"] for s in seeds)))
        trial.set_user_attr("score_std",   float(np.std([s["score"] for s in seeds])))
        trial.set_user_attr("abandoned",   abandoned)
        trial.set_user_attr("train_time",  float(time.perf_counter() - start))

    def objective(self, trial):
        config    = self.search_space(trial, self.base_config)
        done      = trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        incumbent = min((t.value for t in done), default=float("inf"))

        seeds, start = [], time.perf_counter()
        for seed in range(self.n_seeds):
            seeds.append(self.train_per_seed(self.seed_config(config, seed)))
            floor     = sum(s["score"] for s in seeds) / self.n_seeds
            abandoned = floor >= incumbent
            self.record(trial, seeds, start, abandoned)

            last = seeds[-1]
            print(f"  trial {trial.number:3d} seed {seed}: score={last['score']:.3e} fit={last['fit']:.3e} "
                  f"collapsed={last['collapsed']} stopped_at={last['stopped_at']}", flush=True)

            if abandoned:
                jax.clear_caches()
                print(f"  trial {trial.number:3d} abandoned, floor {floor:.4f} >= incumbent {incumbent:.4f}", flush=True)
                raise optuna.TrialPruned()

        jax.clear_caches()
        return float(np.mean([s["score"] for s in seeds]))

    def run(self):
        self.study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0), direction="minimize")
        self.study.optimize(self.objective, n_trials=self.n_trials)
        self.save()
        return self.study

    def data_summary(self):
        d = self.base_config.data
        if d.type == "trajectory":
            return {"type": d.type, "N": d.trajectory_N,
                    "sigma_y": d.trajectory_sigma_y, "sigma_x": d.trajectory_sigma_x}
        return {"type": d.type, "N": d.field_N,
                "sigma_y": d.field_sigma_y, "sigma_x": d.field_sigma_x}

    def save(self):
        out_dir = os.path.join("systems", self.system, family_of(self.method), "best_configs")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{self.method}.json"), "w") as f:
            json.dump(self.study.best_params, f, indent=2)

        best = self.study.best_trial
        with open(os.path.join(out_dir, f"{self.method}_diagnostics.json"), "w") as f:
            json.dump({
                "score":       best.value,
                "best_params": best.params,
                "best_number": best.number,
                "n_seeds":     self.n_seeds,
                "epochs":      self.epochs,
                "n_trials":    len(self.study.trials),
                "data":        self.data_summary(),
                "true_params": dict(self.base_config.system.system_params),
                "tuning":      dataclasses.asdict(self.tuning),
                "user_attrs":  best.user_attrs,
                "all_trials": [
                    {"number": t.number, "state": t.state.name, "value": t.value, "params": t.params, **t.user_attrs}
                    for t in self.study.trials
                ],
            }, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune a (system, method) using multi-seed random search")
    parser.add_argument("system", nargs="?", help="System name")
    parser.add_argument("method", nargs="?", help="Method name")
    parser.add_argument("--n_trials", type=int, default= 60, help="Number of random trials, default set at 60 for a 95% confidence level")
    parser.add_argument("--n_seeds", type=int, default= 5, help="Number of seeds per config")
    parser.add_argument("--epochs", type=int, default= None, help="Number of epochs per seed, defaults to the config's own value")
    parser.add_argument("--check_every", type=int, default= TuningConfig.check_every, help="Number of epochs between collapse checks")
    parser.add_argument("--collapse_fraction", type=float, default= TuningConfig.collapse_fraction, help="Validation loss threshold as a fraction of the zero-predictor validation loss")
    parser.add_argument("--collapse_patience", type=int, default= TuningConfig.collapse_patience, help="Consecutive positive checks before a seed is stopped")
    parser.add_argument("--list", action="store_true", help="List of available (system, method) pairs")
    args = parser.parse_args()

    if args.list or args.system is None or args.method is None:
        list_experiments()
    else:
        config_list = load_config_module(args.system)
        tune_list   = load_config_tune(args.system)
        if args.method not in config_list.CONFIGS:
            raise SystemExit(f"Unknown method in configurations")
        if args.method not in tune_list.SEARCH_SPACES:
            raise SystemExit(f"Unknown method in search spaces")
        tuning = TuningConfig(args.check_every, args.collapse_fraction, args.collapse_patience)
        Tuner(args.system, args.method, args.n_trials, args.epochs, args.n_seeds, tuning).run()
