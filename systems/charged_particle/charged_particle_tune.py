import dataclasses

def search_space_mlp(trial, base_config):
    network  = dataclasses.replace(base_config.network, learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True))
    training = dataclasses.replace(base_config.training, lambda_reg   = trial.suggest_float("lambda_reg", 1e-6, 1e-2, log=True))
    return dataclasses.replace(base_config, network=network, training=training)

def search_space_hnn(trial, base_config):
    config = search_space_mlp(trial, base_config)
    training = dataclasses.replace(config.training, use_auto_lambda=False)
    return dataclasses.replace(config, training=training)

def search_space_hnn_parametric(trial, base_config):
    config = search_space_hnn(trial, base_config)
    loss  = dataclasses.replace(config.loss, lambda_correction=trial.suggest_float("lambda_correction",1e-3,1.0, log=True))
    return dataclasses.replace(config, loss=loss)

def search_space_lnn(trial, base_config):
    config = search_space_mlp(trial, base_config)
    training = dataclasses.replace(config.training, use_auto_lambda=False)
    return dataclasses.replace(config, training=training)

def search_space_lnn_parametric(trial, base_config):
    config = search_space_lnn(trial, base_config)
    training = dataclasses.replace(config.training, use_auto_lambda=False)
    loss  = dataclasses.replace(config.loss, lambda_correction=trial.suggest_float("lambda_correction",1e-3,1.0, log=True))
    return dataclasses.replace(config, loss=loss, training=training)

def search_space_pinn(trial, base_config):
    network  = dataclasses.replace(
        base_config.network,
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True))
    training = dataclasses.replace(
        base_config.training,
        lambda_reg      = trial.suggest_float("lambda_reg", 1e-8, 1e-1, log=True),
        use_auto_lambda = False)
    loss = dataclasses.replace(
        base_config.loss,
        lambda_data    = 1.0,
        lambda_physics = trial.suggest_float("lambda_physics", 1e-3, 1e3, log=True),
        lambda_initial = trial.suggest_float("lambda_initial", 1e-3, 1e3, log=True),
        causal_epsilon = trial.suggest_float("causal_epsilon", 1e-1, 1e3, log=True),
    )
    return dataclasses.replace(base_config, network=network, training=training, loss=loss)

SEARCH_SPACES = {
    "mlp":                       search_space_mlp,
    "mlp_trajectory":            search_space_mlp,
    "hnn":                       search_space_hnn,
    "hnn_mechanical":            search_space_hnn,
    "hnn_parametric":            search_space_hnn_parametric,
    "hnn_parametric_mechanical": search_space_hnn_parametric,
    "lnn":                       search_space_lnn,
    "lnn_parametric":            search_space_lnn_parametric,
    "pinn":                      search_space_pinn,
}