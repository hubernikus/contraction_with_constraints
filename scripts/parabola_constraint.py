import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


class ConstraintSystem:
    @staticmethod
    def is_free(xx: np.ndarray) -> bool:
        return ConstraintSystem.g(xx) < 0

    def g(xx: np.ndarray) -> float:
        return xx[0] * xx[0] + xx[1]

    @staticmethod
    def gradient_g(xx: np.ndarray) -> np.ndarray:
        return np.array([2 * xx[0], 1])


class DivergentFlowField:
    @staticmethod
    def evaluate(xx: np.ndarray) -> np.ndarray:
        return np.array([xx[0], xx[1] + 1])

    @staticmethod
    def langragian_multiplier(xx: np.ndarray) -> float:
        return -(xx[0] ** 2 + 1) / (4 * xx[0] ** 2 + 1)


def get_parabola(xx1):
    return -xx1 * xx1


def plot_parabola(ax, x_lim, n_resolution=100):
    x_vals = np.linspace(x_lim[0], x_lim[1], n_resolution)
    y_vals = get_parabola(x_vals)
    ax.plot(x_vals, y_vals, color="k", linewidth=3)


def main(
    n_resolution=10,
    save_figure=False,
    x_lim=[-10, 10],
    y_lim=[-10, 10],
):
    # Initial Dynamics
    figsize = (5, 4.5)
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
    g_value = np.zeros(positions.shape[1])

    for pp in range(positions.shape[1]):
        velocities[:, pp] = DivergentFlowField.evaluate(positions[:, pp])
        collision_free[pp] = ConstraintSystem.is_free(positions[:, pp])
        g_value[pp] = ConstraintSystem.g(positions[:, pp])

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
    # ax.contour(cont, levels=[0, 1], colors="black")
    plot_parabola(ax=ax, x_lim=x_lim)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if save_figure:
        figname = "dynamics_with_binary_g_value"
        plt.savefig("figures/" + figname + filetype, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=figsize)
    levels = np.linspace(-20.0, 20.0, 20)
    cont = ax.contourf(
        positions[0, :].reshape(nx, ny),
        positions[1, :].reshape(nx, ny),
        g_value.reshape(nx, ny),
        levels=levels,
        extend="both",
        zorder=-2,
        cmap="seismic",
        alpha=0.5,
    )
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
    # ax.contour(cont, levels=[0, 1], colors="black")
    plot_parabola(ax=ax, x_lim=x_lim)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    boundary = get_visible_parabola(x_lim, y_lim)
    lambda_arrow = np.zeros_like(boundary)

    for pp in range(boundary.shape[1]):
        langr_mult = DivergentFlowField.langragian_multiplier(boundary[:, pp])
        gradient = ConstraintSystem.gradient_g(boundary[:, pp])
        lambda_arrow[:, pp] = langr_mult * gradient

    if save_figure:
        figname = "dynamics_with_smooth_g_value"
        plt.savefig("figures/" + figname + filetype, bbox_inches="tight")


def setup_plot(ax, x_lim, y_lim):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)


def get_visible_parabola(x_lim, y_lim):
    # def plot_repulsion_field(figsize=(5, 4.5), x_lim=[-10, 10], y_lim=[-10, 10]):
    n_repulsion_arrows = 50

    x_vals = np.linspace(x_lim[0], x_lim[1], n_repulsion_arrows)
    y_vals = get_parabola(x_vals)

    ind_good = y_vals > y_lim[0]
    x_vals = x_vals[ind_good]
    y_vals = y_vals[ind_good]

    return np.vstack((x_vals, y_vals))


def plot_contour_gray(ax, resolution_contour, x_lim, y_lim):
    nx = ny = resolution_contour
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx),
        np.linspace(y_lim[0], y_lim[1], ny),
    )
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    collision_free = np.zeros(positions.shape[1], dtype=bool)
    for pp in range(positions.shape[1]):
        collision_free[pp] = ConstraintSystem.is_free(positions[:, pp])

    plot_parabola(ax=ax, x_lim=x_lim)

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


def plot_constraint_dynamcis(
    save_figure,
    figsize=(5, 4.5),
    resolution_contour=100,
    resolution_vf=20,
    x_lim=[-10, 10],
    y_lim=[-10, 10],
):
    fig, ax = plt.subplots(figsize=figsize)

    plot_contour_gray(
        ax=ax, resolution_contour=resolution_contour, x_lim=x_lim, y_lim=y_lim
    )

    n_grid = resolution_vf
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
        collision_free[pp] = ConstraintSystem.is_free(positions[:, pp])

    boundary = get_visible_parabola(x_lim, y_lim)
    lambda_arrow = np.zeros_like(boundary)

    for pp in range(boundary.shape[1]):
        langr_mult = DivergentFlowField.langragian_multiplier(boundary[:, pp])
        gradient = ConstraintSystem.gradient_g(boundary[:, pp])
        lambda_arrow[:, pp] = langr_mult * gradient

    ax.quiver(
        boundary[0, :],
        boundary[1, :],
        lambda_arrow[0, :],
        lambda_arrow[1, :],
        color="red",
    )

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
        # color="#414141",
        zorder=0,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if save_figure:
        figname = "constrained_dynamics"
        plt.savefig("figures/" + figname + filetype, bbox_inches="tight")


if (__name__) == "__main__":
    filetype = ".pdf"
    plt.ion()
    plt.close("all")
    # main(n_resolution=100, save_figure=False)
    plot_constraint_dynamcis(save_figure=False)
    # plot_repulsion_field()
