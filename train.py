
import argparse
import importlib
import pkgutil

from configs.configurations import (PINNTrainConfig, HNNTrainConfig, LNNTrainConfig, MLPTrainConfig)


def load_config_module(system):
    return importlib.import_module(f"systems.{system}.{system}_configurations")


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


def main(config):
    import os
    import numpy as np
    from model.network import NeuralNetwork

    gen   = config.data.make_data(config.data)
    X, y  = gen.build_training_data()
    model = NeuralNetwork(config, X, y)
    save_weights = False
    try:
        if config.visualizations.live_learning.enabled:
            raise NotImplementedError
        else:
            model.train()
        print("Training complete")
        physics_desc = model.physics_loss.describe_physics(model.state.params)
        if physics_desc:
            print(f"Learned physics parameters: {physics_desc}")
    except KeyboardInterrupt:
        if save_weights:
            model.save_weights()
    finally:
        model.plot_training_history(log_scale=False)
        model.plot_lambdas_history()
        if not config.visualizations.live_learning.enabled:
            raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a (system, method) experiment.")
    parser.add_argument("system", nargs="?", help="system name, e.g. qho")
    parser.add_argument("method", nargs="?", help="method name, e.g. mlp / hnn / hnn_parametric / lnn / pinn")
    parser.add_argument("--list", action="store_true", help="list available system/method pairs and exit")
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
        main(conf_mod.CONFIGS[args.method])

