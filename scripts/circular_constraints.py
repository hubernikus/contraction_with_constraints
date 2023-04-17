from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from vartools.animator import Animator


@dataclass
class CircularConstraints:
    position: np.ndarray = field(default_factory=lambda: np.array([1.0, 1]))
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0]))

    @property
    def a(self):
        return self.position[0]

    @property
    def a_dot(self):
        return self.velocity[0]

    @property
    def b(self):
        return self.position[1]

    @property
    def b_dot(self):
        return self.velocity[1]

    def is_free(self, xx: np.ndarray) -> bool:
        return self.g(xx) < 0

    def g(self, xx: np.ndarray) -> float:
        return -((xx[0] - self.a) ** 2) - (xx[1] - self.b) ** 2 + 1

    def gradient_g(self, xx: np.ndarray) -> np.ndarray:
        return -2 * np.array([xx[0] - self.a, xx[1] - self.b])

    def get_langragian_multiplier(self, xx: np.ndarray) -> float:
        value = -0.5 * np.array([xx[0] - self.a, xx[1] - self.b]) @ xx
        value -= 0.5 * (xx[0] - self.a) * self.a_dot
        value -= 0.5 * (xx[1] - self.b) * self.b_dot
        return min(value, 0)

    def get_surface_points(self, n_resolution=100):
        angs = np.linspace(0, 2 * np.pi, n_resolution)
        x_vals = np.cos(angs) + self.a
        y_vals = np.sin(angs) + self.b
        return np.vstack([x_vals, y_vals])


class DivergentFlowField:
    @staticmethod
    def evaluate(xx: np.ndarray) -> np.ndarray:
        return (-1) * np.array([xx[0], xx[1]])


def get_parabola(xx1):
    return -xx1 * xx1


def plot_circle(ax, circle, n_resolution=100):
    angs = np.linspace(0, 2 * np.pi, n_resolution)
    x_vals = np.cos(angs) + circle.a
    y_vals = np.sin(angs) + circle.b
    ax.plot(x_vals, y_vals, color="k", linewidth=3)

    # Plot center velocity
    if np.linalg.norm(circle.velocity):
        ax.arrow(
            circle.position[0],
            circle.position[1],
            circle.velocity[0],
            circle.velocity[1],
            color="blue",
            width=0.05,
        )


def main(
    n_resolution=10,
    global_name="circular_",
    save_figure=False,
    x_lim=[-3, 3],
    y_lim=[-3, 3],
):
    constraints = CircularConstraints()

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
        collision_free[pp] = constraints.is_free(positions[:, pp])
        g_value[pp] = constraints.g(positions[:, pp])

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
    plot_circle(ax=ax, circle=constraints)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if save_figure:
        figname = "dynamics_with_binary_g_value"
        plt.savefig("figures/" + global_name + figname + filetype, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=figsize)
    levels = np.linspace(-5.0, 5.0, 20)
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
    plot_boundary_repulsion(ax=ax, constraints=constraints)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if save_figure:
        figname = "dynamics_with_smooth_g_value"
        plt.savefig("figures/" + global_name + figname + filetype, bbox_inches="tight")


def plot_boundary_repulsion(constraints, ax, n_repulsion_arrows=50):
    # def plot_repulsion_field(figsize=(5, 4.5), x_lim=[-10, 10], y_lim=[-10, 10]):
    plot_circle(ax=ax, circle=constraints)

    boundary = constraints.get_surface_points(n_repulsion_arrows)
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
    )


def plot_bw_circular(constraints, ax, n_grid, x_lim, y_lim):
    nx = ny = n_grid
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


def plot_contrained_circular(
    resolution_vf=20,
    resolution_contour=100,
    figsize=(5, 4.5),
    x_lim=[-3, 3],
    y_lim=[-3, 3],
    global_name="circular_",
    save_figure=False,
    constraints=None,
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if constraints is None:
        constraints = CircularConstraints()

    plot_boundary_repulsion(ax=ax, constraints=constraints)
    plot_bw_circular(
        constraints=constraints,
        ax=ax,
        n_grid=resolution_contour,
        x_lim=x_lim,
        y_lim=y_lim,
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

    attractor = np.array([0, 0.0])
    ax.plot(attractor[0], attractor[1], "k*")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if save_figure:
        figname = "constrained_dynamics"
        plt.savefig("figures/" + global_name + figname + filetype, bbox_inches="tight")


def moving_circular_several_plot(figsize=(5, 4.5), save_figure=False):
    dt = 1.6
    n_plots = 4

    constraints = CircularConstraints(
        position=np.array([-0.8, -1.6]),
        velocity=np.array([0.2, 0.6])
        # position=np.array([1, 1]),
        # velocity=np.array([1, 3.0]),
    )

    for ii in range(n_plots):
        _, ax = plt.subplots(figsize=figsize)

        plot_contrained_circular(
            resolution_vf=20,
            resolution_contour=100,
            x_lim=[-3, 3],
            y_lim=[-3, 3],
            constraints=constraints,
            ax=ax,
        )

        if save_figure:
            global_name = "circular_"
            figname = f"constrained_dynamics_dt_{ii}"
            plt.savefig(
                "figures/" + global_name + figname + filetype, bbox_inches="tight"
            )

        # Update position
        constraints.position = constraints.position + constraints.velocity * dt


class CircleAnimator(Animator):
    def setup(self, figsize=(5, 4.5)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.resolution_vf = 20
        self.resolution_contour = 100
        self.x_lim = [-3, 3]
        self.y_lim = [-3, 3]

        self.constraints = CircularConstraints(
            position=np.array([-1.4, -2.6]),
            velocity=np.array([0.2, 0.6])
            # position=np.array([1, 1]),
            # velocity=np.array([1, 3.0]),
        )

    def update_step(self, ii: int) -> None:
        if not ii % 10:
            print(f"ii: {ii}")

        self.ax.clear()
        plot_contrained_circular(
            resolution_vf=self.resolution_vf,
            resolution_contour=self.resolution_contour,
            x_lim=self.x_lim,
            y_lim=self.y_lim,
            constraints=self.constraints,
            ax=self.ax,
        )
        # Update position
        self.constraints.position = (
            self.constraints.position + self.constraints.velocity * self.dt_simulation
        )


def run_animation():
    animator = CircleAnimator(
        it_max=80,
        dt_simulation=0.1,
        dt_sleep=0.1,
        file_type=".gif",
        animation_name="circular_constraints",
    )
    animator.setup()
    animator.run(save_animation=True)


if (__name__) == "__main__":
    filetype = ".pdf"
    plt.ion()
    plt.close("all")
    # main(n_resolution=100, save_figure=True)
    # plot_contrained_circular(resolution_vf=20, save_figure=False)
    # moving_circular_several_plot(save_figure=True)
    # run_animation()
