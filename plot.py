import matplotlib.pyplot as plt
import numpy as np

from nary import NaryFunc
from typing import Tuple

def plot_2d_trajectory(func: NaryFunc, trajectory: list) -> None:
    trajectory = np.array(trajectory)
    xs = trajectory.flatten()
    fs = np.array([func(np.array([x])) for x in xs])

    plt.figure(figsize=(10, 6))
    x_min, x_max = xs.min(), xs.max()
    pad = max(1.0, (x_max - x_min) * 0.2)

    x_grid = np.linspace(x_min - pad, x_max + pad, 200)
    f_grid = np.array([func(np.array([x])) for x in x_grid])

    plt.plot(x_grid, f_grid, "b-", label="Функция")
    plt.plot(xs, fs, "ro-", markersize=4, linewidth=1.5, label="Траектория")
    plt.scatter(xs[0], fs[0], c="green", marker="s", s=100, label="Начало")
    plt.scatter(xs[-1], fs[-1], c="blue", marker="*", s=150, label="Конец")

    plt.xlabel("x", fontsize=12)
    plt.ylabel("f(x)", fontsize=12)
    plt.title("Траектория градиентного спуска (1D)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("gd.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_3d_func(
    func: NaryFunc,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    steps: int,
    trajectory: list,
    name: str,
) -> None:
    x_min, x_max = x_range
    y_min, y_max = y_range

    x_grid = np.linspace(x_min, x_max, steps)
    y_grid = np.linspace(y_min, y_max, steps)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7)

    trajectory = np.array(trajectory)
    xs = trajectory[:, 0]
    ys = trajectory[:, 1]
    zs = np.array([func(np.array([x, y])) for x, y in zip(xs, ys)])

    ax.plot(xs, ys, zs, color="red", label="Линия траектории")
    ax.scatter(xs[1:-2], ys[1:-2], zs[1:-2], marker="o", s=2, color="blue", label="Точки траектории")
    ax.scatter(xs[0], ys[0], zs[0], marker="s", s=20, color="black", label="Начало")
    ax.scatter(xs[-1], ys[-1], zs[-1], marker="*", s=50, color="cyan", label="Конец", zorder=10)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")

    ax.set_title(name)
    plt.savefig("2a3d.png", dpi=600, bbox_inches="tight")
    plt.close()

def plot_2d_projection_trajectory(
    func: NaryFunc,
    trajectory: list,
    name: str
) -> None:
    xs = [point[0] for point in trajectory]
    ys = [point[1] for point in trajectory]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_range = x_max - x_min
    y_range = y_max - y_min

    pad_x = x_range * 0.1 if x_range != 0 else 1
    pad_y = y_range * 0.1 if y_range != 0 else 1

    x_min, x_max = x_min - pad_x, x_max + pad_x
    y_min, y_max = y_min - pad_y, y_max + pad_y

    x_grid = np.linspace(x_min, x_max, 50)
    y_grid = np.linspace(y_min, y_max, 50)

    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    plt.figure()
    cp = plt.contour(X, Y, Z, levels=30, colors='black')
    plt.clabel(cp, inline=True, fontsize=8)

    plt.plot(xs, ys, color="red", label="Линия траектории")
    plt.scatter(xs, ys, marker="o", s=5, color="blue", label="Точки траектории", zorder=10)
    plt.scatter(xs[0], ys[0], marker="s", s=30, color="black", label="Начало", zorder=10)
    plt.scatter(xs[-1], ys[-1], marker="*", s=80, color="green", label="Конец", zorder=10)

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title(name)
    plt.legend()
    plt.savefig("2a2d.png", dpi=600, bbox_inches="tight")
    plt.close()

def plot_gradient(
    func: NaryFunc,
    draw2d: bool,
    draw3d: bool,
    trajectory: list = None,
    name: str = "Unnamed"
) -> None:
    if trajectory is None:
        return
    if draw2d:
        plot_2d_trajectory(func, trajectory)
    if draw3d:
        xs = [pt[0] for pt in trajectory]
        ys = [pt[1] for pt in trajectory]
        pad = max(1.0, max(np.ptp(xs), np.ptp(ys))) * 0.2
        x_range = (min(xs) - pad, max(xs) + pad)
        y_range = (min(ys) - pad, max(ys) + pad)
        plot_2d_projection_trajectory(func, trajectory, name=name)
        plot_3d_func(func, x_range, y_range, steps=50, trajectory=trajectory, name=name)
