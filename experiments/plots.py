import matplotlib.pyplot as plt
import numpy as np


def plot_solution(
    x,
    numerical,
    exact=None,
    title="",
    xlabel="x",
    ylabel="u"
):
    """
    Plot numerical solution against analytical solution.
    """

    plt.figure(figsize=(7, 4))

    if exact is not None:
        plt.plot(
            x,
            exact,
            "--",
            linewidth=2,
            label="Analytical"
        )

    plt.plot(
        x,
        numerical,
        linewidth=2,
        label="WENO9"
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_convergence(
    Ns,
    errors,
    order=9,
    title="Spatial convergence"
):
    """
    Log-log convergence plot.
    """

    Ns = np.asarray(Ns)
    errors = np.asarray(errors)

    h = 1.0 / Ns

    plt.figure(figsize=(6, 4))

    plt.loglog(
        h,
        errors,
        "o-",
        linewidth=2,
        label="WENO9"
    )

    reference = errors[0] * (h / h[0]) ** order

    plt.loglog(
        h,
        reference,
        "--",
        linewidth=2,
        label=f"$O(h^{order})$"
    )

    plt.xlabel(r"$h$")
    plt.ylabel(r"$L_2$ error")

    plt.title(title)

    plt.grid(True, which="both")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_shock(
    x,
    numerical,
    x_shock,
    left_state=1.0,
    right_state=0.0,
    title="Shock capturing"
):
    """
    Plot numerical solution together with exact shock.
    """

    exact = np.where(
        np.asarray(x) < x_shock,
        left_state,
        right_state
    )

    plt.figure(figsize=(7, 4))

    plt.plot(
        x,
        exact,
        "--",
        linewidth=2,
        label="Analytical"
    )

    plt.plot(
        x,
        numerical,
        linewidth=2,
        label="WENO9"
    )

    plt.axvline(
        x_shock,
        color="k",
        linestyle=":",
        linewidth=1,
        alpha=0.6
    )

    plt.xlabel("x")
    plt.ylabel("u")
    plt.title(title)

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()