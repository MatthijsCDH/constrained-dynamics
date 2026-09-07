
import argparse
import importlib
import json
import os
import pkgutil
import optuna

from plots import plot_mse_over_time, plot_energy_drift_over_time, plot_training_history, plot_lambdas_history

def family_of(method):
    return method.split("_")[0]

def load_config_module(system):
    return importlib.import_module(f"systems.{system}.{system}_configurations")


def load_tuned_config(system, method, base_config):
    path = os.path.join("systems", system, family_of(method), "best_configs", f"{method}.json")
    if not os.path.exists(path):
        print(f"No tuned config found at {path}, using configurations.py default")
        return base_config

    with open(path) as f:
        best_params = json.load(f)

    tune_mod = importlib.import_module(f"systems.{system}.{system}_tune")
    search_space = tune_mod.SEARCH_SPACES[method]
    trial = optuna.trial.FixedTrial(best_params)
    print(f"Using tuned config for {method} from {path}")
    return search_space(trial, base_config)


def list_experiments():
    import systems
    found = False
    for info in sorted(pkgutil.iter_modules(systems.__path__), key=lambda m: m.name):
        try:
            mod = load_config_module(info.name)
        except Exception:
            continue
        if hasattr(mod, "CONFIGS"):
            found = True
            print(f"{info.name}: {', '.join(mod.CONFIGS)}")
    if not found:
        print("No experiment modules found (need a systems/<name>/<name>_configurations.py with a CONFIGS dict).")


def main(system, method, config):
    import os
    from model.network import NeuralNetwork
    from benchmark import Metrics

    plots_dir = os.path.join("systems", system, family_of(method), "plots")

    gen   = config.data.make_data(config)
    X, y  = gen.build_training_data()
    model = NeuralNetwork(config, X, y)
    save_weights = False
    try:
        if config.live_learning is not None and config.live_learning.enabled:
            live_plot = config.live_learning.make_live_plot(
                config, os.path.join(plots_dir, f"live_{method}.{config.live_learning.format}"))
            live_plot.live_learning_plot(model)
        else:
            model.train()
        print("Training complete")
        physics_desc = model.physics_loss.describe_physics(model.state.params)
        if physics_desc:
            print(f"Learned physics parameters: {physics_desc}")

        horizon = config.evaluation.horizons[0]
        metrics = Metrics(model, config, horizon)

        mse_series, mse_mean = metrics.mse_over_time()
        print(f"MSE at {horizon}x period: {mse_mean:.6e}")
        plot_mse_over_time(metrics.t, mse_series)

        drift_series, drift_mean = metrics.energy_drift_over_time()
        print(f"Energy drift at {horizon}x period: mean={drift_mean:.6e}")
        plot_energy_drift_over_time(metrics.t, drift_series)
        
    except KeyboardInterrupt:
        if save_weights:
            model.save_weights()
    finally:
        plot_training_history(model.history, model.loss_names, log_scale=False)
        plot_lambdas_history(model.history, model.loss_names, log_scale=False)
        if config.visualizations is not None and config.visualizations.enabled:
            vis = config.visualizations.make_vis(
                config, os.path.join(plots_dir, f"animation_{method}.{config.visualizations.format}"))
            vis.animate(model)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a (system, method) experiment.")
    parser.add_argument("system", nargs="?", help="system name, e.g. mass-spring")
    parser.add_argument("method", nargs="?", help="method name, e.g. mlp / hnn / hnn_parametric / lnn / pinn")
    parser.add_argument("--list", action="store_true", help="list available system/method pairs and exit")
    parser.add_argument("--tuned", action="store_true", help="use tuned hyperparameters from best_configs/ if available")
    args = parser.parse_args()

    if args.list or args.system is None or args.method is None:
        list_experiments()
    else:
        conf_mod = load_config_module(args.system)
        if args.method not in conf_mod.CONFIGS:
            raise SystemExit(
                f"Unknown method {args.method!r} for {args.system!r}; "
                f"available: {', '.join(conf_mod.CONFIGS)}"
            )
        config = conf_mod.CONFIGS[args.method]
        if args.tuned:
            config = load_tuned_config(args.system, args.method, config)
        main(args.system, args.method, config)

