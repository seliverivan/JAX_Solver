from experiments.weno_cl.advection import (
    test_linear_advection_smooth,
    test_linear_advection_step
)

from experiments.weno_cl.burgers import (
    test_burgers_shock
)


if __name__ == "__main__":

    test_linear_advection_smooth()

    test_linear_advection_step()

    test_burgers_shock()