from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from vartools.animator import Animator


@dataclass
class SecondOrderConstraints:
    x_lim = 2

    def is_free(self, position):
        if abs(position[0]) > self.x_lim:
            return False
        else:
            return True

    def g(self, xx) -> float:
        return abs(xx[0]) - self.x_lim

    def gradient_g(self, xx):
        return -1 * np.array([np.copysign(1, xx[0]), 0])

    def get_time_derivative(self, control_input: float = 0):
        return np.array([-self.v - self.x - control_input, self.v])

    def get_langragian_multiplier(self, xx: np.ndarray) -> float:
        lagr = (-1) * self.gradient_g(xx).T @ DivergentFlowField.evaluate(xx)
        return max(lagr, 0.0)


class DivergentFlowField:
    @staticmethod
    def evaluate(xx: np.ndarray, control_input: float = 0) -> np.ndarray:
        return np.array([-xx[1] - xx[0] - control_input, xx[1]])


def plot_border(ax, y_lim):
    color = "black"
    linewidth = 2
    ax.plot([-2, -2], y_lim, color, linewidth=linewidth)
    ax.plot([2, 2], y_lim, color, linewidth=linewidth)


def plot_bw_field(constraints, ax, n_resolution, x_lim, y_lim):
    nx = ny = n_resolution
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx),
        np.linspace(y_lim[0], y_lim[1], ny),
    )
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    collision_free = np.zeros(positions.shape[1], dtype=bool)

    for pp in range(positions.shape[1]):
        collision_free[pp] = constraints.is_free(positions[:, pp])

    black_white_map = [[0, 0, 0], [1, 1, 1]]
    bw_cmap = ListedColormap(black_white_map)
    levels = np.linspace(0.0, 1.0, 3)
    cont = ax.contourf(
        positions[0, :].reshape(nx, ny),
        positions[1, :].reshape(nx, ny),
        collision_free.reshape(nx, ny),
        levels=levels,
        zorder=-2,
        # cmap="Greys_r",
        cmap=bw_cmap,
        alpha=0.2,
    )

    plot_border(ax=ax, y_lim=y_lim)


def main(
    n_resolution=10,
    global_name="secondorder_",
    save_figure=False,
    x_lim=[-3, 3],
    y_lim=[-3, 3],
    figsize=(5, 4.5),
):
    constraints = SecondOrderConstraints()

    # Initial Dynamics

    fig, ax = plt.subplots(figsize=figsize)

    n_grid = n_resolution
    nx = ny = n_resolution
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx),
        np.linspace(y_lim[0], y_lim[1], ny),
    )
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    velocities = np.zeros_like(positions)
    collision_free = np.zeros(positions.shape[1], dtype=bool)

    for pp in range(positions.shape[1]):
        velocities[:, pp] = DivergentFlowField.evaluate(positions[:, pp])
        collision_free[pp] = constraints.is_free(positions[:, pp])

    ax.streamplot(
        positions[0, :].reshape(n_grid, n_grid),
        positions[1, :].reshape(n_grid, n_grid),
        velocities[0, :].reshape(n_grid, n_grid),
        velocities[1, :].reshape(n_grid, n_grid),
        # color="black",
        color="black",
        # color="#414141",
        density=1,
        zorder=0,
    )

    plot_bw_field(
        constraints, ax=ax, n_resolution=n_resolution, x_lim=x_lim, y_lim=y_lim
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if save_figure:
        figname = "dynamics_with_binary_g_value"
        plt.savefig("figures/" + global_name + figname + filetype, bbox_inches="tight")


def plot_boundary_repulsion(constraints, ax, y_lim, n_points=30):
    # def plot_repulsion_field(figsize=(5, 4.5), x_lim=[-10, 10], y_lim=[-10, 10]):
    plot_border(ax, y_lim=y_lim)

    border_xx = [-2, 2]
    for xx in border_xx:
        boundary = np.vstack(
            (np.ones(n_points) * xx, np.linspace(y_lim[0], y_lim[1], n_points))
        )

        lambda_arrow = np.zeros_like(boundary)
        for pp in range(boundary.shape[1]):
            langr_mult = constraints.get_langragian_multiplier(boundary[:, pp])
            gradient = constraints.gradient_g(boundary[:, pp])
            lambda_arrow[:, pp] = langr_mult * gradient

        ax.quiver(
            boundary[0, :],
            boundary[1, :],
            lambda_arrow[0, :],
            lambda_arrow[1, :],
            color="red",
            scale=10,
        )


def plot_contrained_second_order(
    resolution_vf=20,
    resolution_contour=100,
    figsize=(5, 4.5),
    x_lim=[-3, 3],
    y_lim=[-3, 3],
    global_name="secondorder_",
    save_figure=False,
    constraints=None,
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if constraints is None:
        constraints = SecondOrderConstraints()

    plot_boundary_repulsion(ax=ax, y_lim=y_lim, constraints=constraints)
    plot_bw_field(
        constraints, ax=ax, n_resolution=resolution_contour, x_lim=x_lim, y_lim=y_lim
    )

    nx = ny = resolution_vf
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx),
        np.linspace(y_lim[0], y_lim[1], ny),
    )
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    velocities = np.zeros_like(positions)
    collision_free = np.zeros(positions.shape[1], dtype=bool)
    for pp in range(positions.shape[1]):
        velocities[:, pp] = DivergentFlowField.evaluate(positions[:, pp])
        collision_free[pp] = constraints.is_free(positions[:, pp])

    constrained_dynamics = []
    # for pp in range(constrained_dynamics.shape[1]):
    #     if collision_free[pp]:
    #         continue
    positions_free = positions[:, collision_free]
    constrained_dynamics = velocities[:, collision_free]

    ax.quiver(
        positions_free[0, :],
        positions_free[1, :],
        constrained_dynamics[0, :],
        constrained_dynamics[1, :],
        # color="black",
        color="black",
        # scale=0.1,
        # color="#414141",
        zorder=0,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if save_figure:
        figname = "constrained_dynamics"
        plt.savefig("figures/" + global_name + figname + filetype, bbox_inches="tight")


if (__name__) == "__main__":
    filetype = ".pdf"
    plt.ion()
    plt.close("all")
    main(n_resolution=100, save_figure=True)
    plot_contrained_second_order(resolution_vf=20, save_figure=True)
