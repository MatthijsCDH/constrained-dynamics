import contextlib
import os
import threading
import warnings

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import matplotlib.gridspec as gridspec

from systems.mass_spring_damped import data as msd_data


@contextlib.contextmanager
def suppress_fork_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"os\.fork\(\) was called",
                                category=RuntimeWarning)
        yield


def make_writer(fmt, fps):
    if fmt == "mp4":
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        return FFMpegWriter(fps=fps, bitrate=1800)
    if fmt == "gif":
        return PillowWriter(fps=fps)
    raise ValueError(f"Unsupported animation format {fmt!r}; expected 'gif' or 'mp4'")


def write_frames(frames, save_path, fmt, fps):
    if not frames:
        return
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    if fmt == "gif":
        from PIL import Image
        images = [Image.fromarray(f) for f in frames]
        images[0].save(save_path, save_all=True, append_images=images[1:],
                       duration=max(1, int(1000 / fps)), loop=0, optimize=True)
    elif fmt == "mp4":
        import imageio_ffmpeg
        height, width = frames[0].shape[:2]
        pad_h, pad_w  = (-height) % 16, (-width) % 16
        writer = imageio_ffmpeg.write_frames(save_path, (width + pad_w, height + pad_h),
                                             fps=fps, quality=7)
        with suppress_fork_warning():
            writer.send(None)
            for frame in frames:
                if pad_h or pad_w:
                    frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=255)
                writer.send(np.ascontiguousarray(frame).tobytes())
            writer.close()
    else:
        raise ValueError(f"Unsupported animation format {fmt!r}; expected 'gif' or 'mp4'")

    print(f"Saved live learning animation to {save_path}")


class MassSpringDampedVisualizations:
    def __init__(self, config, save_path=None):
        self.config    = config
        self.save_path = save_path
        self.data   = msd_data.make_data(config)
        self.q0     = config.system.q0
        self.p0     = config.system.p0
        self.period = config.system.period

    def live_learning_plot(self, model, show=True):
        live_cfg = self.config.live_learning
        model.train_setup()

        if not show:
            for epoch in range(1, model.epochs + 1):
                model.train_epoch(epoch)
            if model.save_filepath is not None:
                model.save_weights()
            return None

        t_max       = live_cfg.horizon * self.period
        t           = np.linspace(0.0, t_max, live_cfg.n_points)
        true_states = self.data.true_trajectory(self.q0, self.p0, t, coords=model.coords)
        q_true      = true_states[:, 0]

        loss_names = model.loss_names
        n_losses   = len(loss_names)

        fig = plt.figure(figsize=(14, 7))
        gs  = gridspec.GridSpec(nrows=max(n_losses, 1), ncols=2, width_ratios=[3, 1])

        ax_pred = fig.add_subplot(gs[:, 0])
        ax_pred.plot(t, q_true, color="blue", label="true q(t)")
        line_pred, = ax_pred.plot([], [], color="red", linestyle="--", linewidth=2, label="predicted q(t)")
        ax_pred.axvline(self.period, color="blue", linestyle=":", linewidth=2, label="training domain end")
        collocation_x_max = getattr(model.physics_loss, "t_max", None)
        if collocation_x_max is not None and abs(collocation_x_max - self.period) > 1e-9:
            ax_pred.axvline(collocation_x_max, color="purple", linestyle="-.", linewidth=2, label="collocation domain end")
        ax_pred.set_xlim(t[0], t[-1])
        ax_pred.set_ylim(float(q_true.min()) - 0.5, float(q_true.max()) + 0.5)
        ax_pred.set_xlabel("t")
        ax_pred.set_ylabel("q")
        ax_pred.legend(loc="upper right")
        ax_pred.grid(True)

        loss_lines = {}
        loss_axes  = {}
        for i, name in enumerate(loss_names):
            ax = fig.add_subplot(gs[i, 1])
            loss_axes[name] = ax
            line_train, = ax.plot([], [], color="blue", label="train")
            loss_lines[f"train_{name}"] = line_train
            if model.do_validation:
                line_val, = ax.plot([], [], color="orange", label="val")
                loss_lines[f"val_{name}"] = line_val
            ax.set_ylabel(name)
            ax.legend(loc="upper right")
            ax.grid(True)
            if live_cfg.log_scale:
                ax.set_yscale("log")
        loss_axes[loss_names[-1]].set_xlabel("epoch")

        closed = {"flag": False}
        def on_close(event):
            closed["flag"] = True
            finish(early=True)
        fig.canvas.mpl_connect("close_event", on_close)
        stop_training = threading.Event()
        training_done = threading.Event()

        def train_loop():
            for epoch in range(1, model.epochs + 1):
                if stop_training.is_set():
                    break
                model.train_epoch(epoch)
            training_done.set()

        trainer = threading.Thread(target=train_loop, daemon=True)
        trainer.start()

        frames   = []
        record   = live_cfg.save_path and self.save_path and live_cfg.format != "png"
        finished = {"flag": False}
        def finish(early):
            if finished["flag"]:
                return
            finished["flag"] = True
            if anim.event_source is not None:
                anim.event_source.stop()
            stop_training.set()
            trainer.join(timeout=5)
            if model.save_filepath is not None:
                model.save_weights()
            if live_cfg.save_path and self.save_path:
                if live_cfg.format == "png":
                    os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
                    fig.savefig(self.save_path, dpi=110)
                    print(f"Saved live learning plot to {self.save_path}")
                else:
                    write_frames(frames, self.save_path, live_cfg.format, fps=8)
            if early:
                print(f"Live plot closed at epoch {len(model.history['train'])}/{model.epochs} -- stopping training.")
            else:
                timer = fig.canvas.new_timer(interval=2000)
                timer.single_shot = True
                timer.add_callback(plt.close, fig)
                timer.start()

        last_drawn = {"epoch": 0}

        def update(frame):
            if closed["flag"]:
                finish(early=True)
                return (line_pred, *loss_lines.values())

            epoch = len(model.history["train"])
            due   = epoch - last_drawn["epoch"] >= live_cfg.plot_every
            if epoch > 0 and (due or training_done.is_set()):
                last_drawn["epoch"] = epoch
                if model.type == "trajectory":
                    learned_states = model.trajectories(t)
                else:
                    learned_states = model.trajectories(t, state0=true_states[0])
                line_pred.set_data(t, learned_states[:, 0])

                epochs      = np.arange(1, epoch + 1)
                train_stack = np.stack([np.mean(np.asarray(e), axis=0) for e in model.history["train"][:epoch]])
                for i, name in enumerate(loss_names):
                    loss_lines[f"train_{name}"].set_data(epochs, train_stack[:, i])
                if model.do_validation:
                    val_stack = np.stack([np.mean(np.asarray(e), axis=0) for e in model.history["val"][:epoch]])
                    for i, name in enumerate(loss_names):
                        loss_lines[f"val_{name}"].set_data(epochs, val_stack[:, i])
                for ax in loss_axes.values():
                    ax.relim()
                    ax.autoscale_view()

                ax_pred.set_title(f"Epoch {epoch}/{model.epochs}")

                if record:
                    fig.canvas.draw()
                    frames.append(np.asarray(fig.canvas.buffer_rgba())[::2, ::2, :3].copy())

            if training_done.is_set():
                finish(early=False)

            return (line_pred, *loss_lines.values())

        def init():
            return (line_pred, *loss_lines.values())

        anim = FuncAnimation(fig, update, init_func=init, interval=100,
                              blit=False, repeat=False, cache_frame_data=False)

        plt.show()
        plt.ioff()
        return fig


def make_live_plot(config, save_path=None):
    return MassSpringDampedVisualizations(config, save_path)


class MassSpringDampedAnimation:
    def __init__(self, config, save_path=None):
        self.config    = config
        self.save_path = save_path
        self.data   = msd_data.make_data(config)
        self.q0     = config.system.q0
        self.p0     = config.system.p0
        self.period = config.system.period

    def spring_path(self, x_wall, x_mass, n_coils=12, amplitude=0.12, n_points=200):
        s = np.linspace(0.0, 1.0, n_points)
        x = x_wall + s * (x_mass - x_wall)
        y = amplitude * np.sin(2 * np.pi * n_coils * s)
        y[:8]  = 0.0
        y[-8:] = 0.0
        return x, y

    def rollout(self, model, horizon, n_frames):
        t = np.linspace(0.0, horizon * self.period, n_frames)
        true_states = np.asarray(
            self.data.true_trajectory(self.q0, self.p0, t, coords=model.coords)
        )
        if model.type == "trajectory":
            learned_states = np.asarray(model.trajectories(t))
        else:
            learned_states = np.asarray(model.trajectories(t, state0=true_states[0]))
        return t, true_states, learned_states

    def animate(self, model, horizon=None, n_frames=None, interval=None, show=True):
        vis       = self.config.visualizations
        horizon   = horizon  if horizon  is not None else vis.horizon
        n_frames  = n_frames if n_frames is not None else vis.n_frames
        interval  = interval if interval is not None else vis.interval
        save_path = self.save_path if vis.save_path else None

        t, true_states, learned_states = self.rollout(model, horizon, n_frames)
        q_true, q_pred = true_states[:, 0], learned_states[:, 0]
        E_true = np.asarray(self.data.hamiltonian(true_states, coords=model.coords))
        E_pred = np.asarray(self.data.hamiltonian(learned_states, coords=model.coords))

        span   = float(np.max(np.abs(np.concatenate([q_true, q_pred])))) + 0.4
        x_wall = -span - 0.3

        fig = plt.figure(figsize=(11, 7.5))
        gs  = gridspec.GridSpec(nrows=4, ncols=1, height_ratios=[1, 1, 2, 2], hspace=0.55)

        ax_true = fig.add_subplot(gs[0])
        ax_pred = fig.add_subplot(gs[1])
        ax_q    = fig.add_subplot(gs[2])
        ax_E    = fig.add_subplot(gs[3])

        springs, masses = {}, {}
        for ax, label, colour in ((ax_true, "true", "tab:blue"), (ax_pred, "learned", "tab:red")):
            ax.set_xlim(x_wall - 0.2, span + 0.2)
            ax.set_ylim(-0.6, 0.6)
            ax.set_yticks([])
            ax.set_ylabel(label, color=colour)
            ax.axvline(x_wall, color="black", linewidth=4)
            ax.axhline(0.0, color="lightgrey", linewidth=1, zorder=0)
            springs[label], = ax.plot([], [], color=colour, linewidth=1.5)
            masses[label]   = ax.add_patch(
                plt.Rectangle((0, -0.22), 0.44, 0.44, facecolor=colour, alpha=0.75, zorder=3)
            )

        ax_q.plot(t, q_true, color="tab:blue", label="true q(t)")
        ax_q.plot(t, q_pred, color="tab:red", linestyle="--", label="learned q(t)")
        ax_q.axvline(self.period, color="grey", linestyle=":", label="training domain end")
        marker_true, = ax_q.plot([], [], "o", color="tab:blue", markersize=7)
        marker_pred, = ax_q.plot([], [], "o", color="tab:red", markersize=7)
        ax_q.set_xlim(t[0], t[-1])
        ax_q.set_ylabel("q")
        ax_q.grid(alpha=0.3)
        ax_q.legend(loc="upper right", fontsize=8)
        ax_E.plot(t, E_true, color="tab:blue", label="true E(t)")
        ax_E.plot(t, E_pred, color="tab:red", linestyle="--", label="learned E(t)")
        ax_E.axvline(self.period, color="grey", linestyle=":")
        marker_E_true, = ax_E.plot([], [], "o", color="tab:blue", markersize=6)
        marker_E_pred, = ax_E.plot([], [], "o", color="tab:red", markersize=6)
        ax_E.set_xlim(t[0], t[-1])
        ax_E.set_xlabel("t")
        ax_E.set_ylabel("energy")
        ax_E.grid(alpha=0.3)
        ax_E.legend(loc="upper right", fontsize=8)

        def draw(frame):
            for label, q in (("true", q_true[frame]), ("learned", q_pred[frame])):
                xs, ys = self.spring_path(x_wall, q - 0.22)
                springs[label].set_data(xs, ys)
                masses[label].set_x(q - 0.22)
            marker_true.set_data([t[frame]], [q_true[frame]])
            marker_pred.set_data([t[frame]], [q_pred[frame]])
            marker_E_true.set_data([t[frame]], [E_true[frame]])
            marker_E_pred.set_data([t[frame]], [E_pred[frame]])
            ax_true.set_title(f"t = {t[frame]:.2f}   ({t[frame] / self.period:.2f} periods)", fontsize=10)
            return (*springs.values(), *masses.values(),
                    marker_true, marker_pred, marker_E_true, marker_E_pred)

        anim = FuncAnimation(fig, draw, frames=n_frames, interval=interval,
                             blit=False, repeat=True)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fps = max(1, int(round(1000 / interval)))
            with suppress_fork_warning():
                anim.save(save_path, writer=make_writer(vis.format, fps), dpi=vis.dpi)
            print(f"Saved animation to {save_path}")
        if show:
            plt.show()
        return anim


def make_animation(config, save_path=None):
    return MassSpringDampedAnimation(config, save_path)
