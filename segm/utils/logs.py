import json
from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt
import click
from collections import OrderedDict

from segm.utils.lines import Lines


def plot_logs(logs, x_key, y_key, size, vmin, vmax, epochs):
    m = np.inf
    M = -np.inf
    domains = []
    lines = []
    y_keys = y_key.split("/")
    for name, log in logs.items():
        logs[name] = log[:epochs]
    for name, log in logs.items():
        domain = [x[x_key] for x in log if y_keys[0] in x]
        if y_keys[0] not in log[0]:
            continue
        log_plot = [x[y_keys[0]] for x in log if y_keys[0] in x]
        for y_key in y_keys[1:]:
            if y_key in log_plot[0]:
                log_plot = [x[y_key] for x in log_plot if y_key in x]
        domains.append(domain)
        lines.append(np.array(log_plot)[:, None])
        m = np.min((m, min(log_plot)))
        M = np.max((M, max(log_plot)))
    if vmin is not None:
        m = vmin
    if vmax is not None:
        M = vmax
    delta = 0.1 * (M - m)

    ratio = 0.6
    figsizes = {"tight": (4, 3), "large": (16 * ratio, 10 * ratio)}
    figsize = figsizes[size]

    # plot parameters
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    plot_lines = Lines(resolution=50, smooth=0.0)
    plot_lines.LEGEND["loc"] = "upper left"
    # plot_lines.LEGEND["fontsize"] = "large"
    plot_lines.LEGEND["bbox_to_anchor"] = (0.75, 0.2)
    labels_logs = list(logs.keys())
    colors = plot_lines(ax, domains, lines, labels_logs)
    ax.grid(True, alpha=0.5)
    ax.set_ylim(m - delta, M + delta)

    plt.show()
    fig.savefig(
        "plot.png", bbox_inches="tight", pad_inches=0.1, transparent=False, dpi=300
    )
    plt.close(fig)


def print_logs(logs, x_key, y_key, last_log_idx=None):
    delim = "   "
    s = ""
    keys = []
    y_keys = y_key.split("/")
    for name, log in logs.items():
        log_idx = last_log_idx
        if log_idx is None:
            log_idx = len(log) - 1
        while y_keys[0] not in log[log_idx]:
            log_idx -= 1
        last_log = log[log_idx]
        log_x = last_log[x_key]
        log_y = last_log[y_keys[0]]
        for y_key in y_keys[1:]:
            log_y = log_y[y_key]
        s += f"{name}:\n"
        # s += f"{delim}{x_key}: {log_x}\n"
        s += f"{delim}{y_key}: {log_y:.4f}\n"
        keys += list(last_log.keys())
    keys = list(set(keys))
    keys = ", ".join(keys)
    s = f"keys: {keys}\n" + s
    print(s)


def read_logs(root, logs_path):
    logs = {}
    for name, path in logs_path.items():
        path = root / path
        if not path.exists():
            print(f"Skipping {name} that has no log file")
            continue
        logs[name] = []
        with open(path, "r") as f:
            for line in f.readlines():
                d = json.loads(line)
                logs[name].append(d)
    return logs


@click.command()
@click.argument("log_path", type=str)
@click.option("--x-key", default="epoch", type=str)
@click.option("--y-key", default="val_mean_iou", type=str)
@click.option("-s", "--size", default="large", type=str)
@click.option("-ep", "--epoch", default=-1, type=int)
@click.option("-plot", "--plot/--no-plot", default=True, is_flag=True)
def main(log_path, x_key, y_key, size, epoch, plot):
    abs_path = Path(__file__).parent / log_path
    if abs_path.exists():
        log_path = abs_path
    config = yaml.load(open(log_path, "r"), Loader=yaml.FullLoader)
    root = Path(config["root"])
    logs_path = OrderedDict(config["logs"])
    vmin = config.get("vmin", None)
    vmax = config.get("vmax", None)
    epochs = config.get("epochs", None)

    logs = read_logs(root, logs_path)
    if not logs:
        return
    print_logs(logs, x_key, y_key, epoch)
    if plot:
        plot_logs(logs, x_key, y_key, size, vmin, vmax, epochs)


if __name__ == "__main__":
    main()
