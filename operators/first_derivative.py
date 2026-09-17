from operators.flux_divergence import flux_divergence


def first_derivative(
    u,
    dx,
    axis,
    flux,
    ng=4,
):
    return flux_divergence(
        u,
        dx,
        axis,
        flux,
        ng,
    )