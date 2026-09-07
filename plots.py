import os

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history, loss_names, log_scale=False, save_path=None, show=True):
    if not history["train"]:
        print("No training history to plot.")
        return None

    loss_names  = loss_names
    train_stack = np.stack([np.mean(np.asarray(epoch), axis=0) for epoch in history["train"]])
    has_val     = history["val"][0] is not None
    if has_val:
        val_stack = np.stack([np.mean(np.asarray(epoch), axis=0) for epoch in history["val"]])
    epochs = np.arange(1, train_stack.shape[0] + 1)

    total_train = np.array(history["total_train"])
    if has_val:
        total_val = np.array(history["total_val"])

    n_losses = len(loss_names)
    fig, axes = plt.subplots(n_losses + 1, 1, figsize=(8, 3 * (n_losses + 1)), squeeze=False)

    ax = axes[0, 0]
    ax.plot(epochs, total_train, label="train")
    if has_val:
        ax.plot(epochs, total_val, label="val")
    if log_scale:
        ax.set_yscale("log")
    ax.set_title("total")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(alpha=0.3)

    for i, name in enumerate(loss_names):
        ax = axes[i + 1, 0]
        ax.plot(epochs, train_stack[:, i], label="train")
        if has_val:
            ax.plot(epochs, val_stack[:, i], label="val")
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    return fig

def plot_lambdas_history(history, loss_names, log_scale=False, save_path=None, show=True):
    if not history["eff_lambdas"]:
        print("No lambda history to plot.")
        return None

    loss_names     = loss_names
    eff_lambdas    = np.stack(history["eff_lambdas"])
    epochs         = np.arange(1, eff_lambdas.shape[0] + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, name in enumerate(loss_names):
        ax.plot(epochs, eff_lambdas[:, i], label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("effective λ")
    if log_scale:
        ax.set_yscale("log")
    ax.set_title("Kendall loss weights")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    return fig

def plot_mse_over_time(t, mse_series, log_scale=True, save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, mse_series)
    ax.set_xlabel("t")
    ax.set_ylabel("MSE")
    ax.set_title("MSE over time")
    ax.grid(True)
    fig.tight_layout()
    if log_scale:
        ax.set_yscale("log")
    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()

def plot_energy_drift_over_time(t, drift_series, log_scale=True, save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, drift_series)
    ax.set_xlabel("t")
    ax.set_ylabel("Energy drift")
    ax.set_title("Energy drift over time")
    ax.grid(True)
    fig.tight_layout()
    if log_scale:
        ax.set_yscale("log")
    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()

def plot_mse_band_over_time(results, sigma, system, method, log_scale=True, save_path=None, show=True):
    t = results["t"]
    mean = np.array(results["mse_series_mean"])
    std  = np.array(results["mse_series_std"])
    lower = np.maximum(mean - std, mean * 1e-2) if log_scale else mean - std

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(t, lower, mean + std, alpha = 0.2)
    ax.plot(t, mean)
    ax.set_xlabel("t")
    ax.set_ylabel("MSE")
    ax.set_title(f"{system}, {method}, {sigma} - MSE over time  ")
    ax.grid(True)
    fig.tight_layout()
    if log_scale:
        ax.set_yscale("log")
    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()

def plot_drift_band_over_time(results, sigma, system, method, log_scale=True, save_path=None, show=True):
    t = results["t"]
    mean = np.array(results["drift_series_mean"])
    std  = np.array(results["drift_series_std"])
    lower = np.maximum(mean - std, mean * 1e-2) if log_scale else mean - std

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, mean)
    ax.fill_between(t, lower, mean + std, alpha = 0.2)
    ax.set_xlabel("t")
    ax.set_ylabel("Energy drift")
    ax.set_title(f"{system}, {method}, {sigma} - Energy drift over time  ")
    ax.grid(True)
    fig.tight_layout()
    if log_scale:
        ax.set_yscale("log")
    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()
