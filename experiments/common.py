import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ==========================================================
# Utils
# ==========================================================

def banner(title):

    print("="*70)
    print(title)
    print("="*70)



def compute_L2(u, exact, dx):

    return float(
        jnp.sqrt(
            dx*jnp.sum(
                (u-exact)**2
            )
        )
    )



def compute_TV(u):

    return float(
        jnp.sum(
            jnp.abs(
                jnp.roll(u,-1)-u
            )
        )
    )