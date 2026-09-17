from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from rk.rk5 import rk5


@partial(
    jax.jit,
    static_argnames=(
        "rhs",
        "compute_dt",
        "max_steps",
        "num_frames",
    ),
)
def integrate(
    u0,
    t_end,
    rhs,
    compute_dt,
    max_steps,
    num_frames=2,
):
    """
    Generic adaptive time integrator.

    Parameters
    ----------
    u0 : jax.Array
        Initial condition.

    t_end : float
        Final simulation time.

    rhs : callable
        Right-hand side function.

    compute_dt : callable
        Adaptive timestep function.

    max_steps : int
        Maximum number of timesteps.

    num_frames : int
        Number of states stored in history.

        The first frame is always t = 0.
        The last frame is always t = t_end.

        Memory usage therefore depends only on
        num_frames, not on max_steps.

    Returns
    -------
    u : jax.Array
        Final solution.

    history : jax.Array
        Saved solution states.

        Shape:
            (num_frames,) + u0.shape

    times : jax.Array
        Requested/saved frame times.

        Shape:
            (num_frames,)

    save_id : jax.Array
        Number of frames written.

    step : jax.Array
        Number of RK steps performed.
    """

    # ------------------------------------------------------
    # Validate number of frames
    #
    # We need at least:
    #
    #   frame 0 -> initial state
    #   frame 1 -> final state
    # ------------------------------------------------------

    num_frames = max(
        int(num_frames),
        2,
    )

    # ------------------------------------------------------
    # Target frame times
    #
    # Example:
    #
    # num_frames = 5
    #
    # times:
    #
    # 0
    # 0.25 t_end
    # 0.50 t_end
    # 0.75 t_end
    # t_end
    # ------------------------------------------------------

    frame_times = jnp.linspace(
        0.0,
        t_end,
        num_frames,
        dtype=u0.dtype,
    )

    # ------------------------------------------------------
    # History
    #
    # IMPORTANT:
    #
    # Memory is now:
    #
    #     num_frames * u0.size
    #
    # and NOT:
    #
    #     max_steps / save_every * u0.size
    # ------------------------------------------------------

    history = jnp.zeros(
        (num_frames,) + u0.shape,
        dtype=u0.dtype,
    )

    history = history.at[0].set(u0)

    # ------------------------------------------------------
    # Saved frame counter
    #
    # Frame 0 already contains u0.
    #
    # Therefore the next frame to write is 1.
    # ------------------------------------------------------

    save_id = jnp.int32(1)

    # ------------------------------------------------------
    # Integration state
    # ------------------------------------------------------

    state = (
        u0,
        jnp.asarray(
            0.0,
            dtype=u0.dtype,
        ),
        jnp.int32(0),
        save_id,
        history,
    )

    # ------------------------------------------------------
    # Loop condition
    # ------------------------------------------------------

    def cond(state):

        (
            _,
            t,
            step,
            _,
            _,
        ) = state

        return jnp.logical_and(
            t < t_end,
            step < max_steps,
        )

    # ------------------------------------------------------
    # Loop body
    # ------------------------------------------------------

    def body(state):

        (
            u,
            t,
            step,
            save_id,
            history,
        ) = state

        # --------------------------------------------------
        # Adaptive timestep
        # --------------------------------------------------

        dt = compute_dt(u)

        dt = jnp.minimum(
            dt,
            t_end - t,
        )

        # --------------------------------------------------
        # RK5
        # --------------------------------------------------

        u_new = rk5(
            u,
            dt,
            rhs,
        )

        t_new = t + dt

        step_new = step + 1

        # --------------------------------------------------
        # Save frames
        #
        # The next requested frame time is:
        #
        #     frame_times[save_id]
        #
        # If the current RK step crossed that time,
        # store the current solution.
        # --------------------------------------------------

        has_frames_left = (
            save_id < num_frames
        )

        crossed_frame = jnp.logical_and(
            has_frames_left,
            t_new >= frame_times[save_id],
        )

        # --------------------------------------------------
        # Write current solution into the requested frame.
        #
        # In normal CFL operation one RK step will not
        # cross many output times.
        #
        # Therefore this writes at most one frame per step.
        # --------------------------------------------------

        history = lax.cond(
            crossed_frame,
            lambda h: h.at[save_id].set(u_new),
            lambda h: h,
            history,
        )

        save_id = (
            save_id
            + crossed_frame.astype(jnp.int32)
        )

        # --------------------------------------------------
        # Final state
        #
        # If we reached t_end, explicitly put the final
        # numerical solution into the last history slot.
        #
        # This guarantees that history[-1] is the actual
        # final solution.
        # --------------------------------------------------

        reached_final = (
            t_new >= t_end
        )

        history = lax.cond(
            reached_final,
            lambda h: h.at[num_frames - 1].set(
                u_new
            ),
            lambda h: h,
            history,
        )

        # --------------------------------------------------
        # If final state was written and the frame counter
        # has not reached the end, mark all frame slots
        # logically as available.
        #
        # The solver still slices using save_id, so update
        # it to num_frames at the final step.
        # --------------------------------------------------

        save_id = lax.cond(
            reached_final,
            lambda _: jnp.int32(num_frames),
            lambda _: save_id,
            operand=None,
        )

        return (
            u_new,
            t_new,
            step_new,
            save_id,
            history,
        )

    # ------------------------------------------------------
    # Integrate
    # ------------------------------------------------------

    (
        u,
        t,
        step,
        save_id,
        history,
    ) = lax.while_loop(
        cond,
        body,
        state,
    )

    # ------------------------------------------------------
    # If max_steps was reached before t_end,
    # save_id may be smaller than num_frames.
    #
    # The caller can use history[:save_id].
    # ------------------------------------------------------

    # ------------------------------------------------------
    # Times corresponding to the allocated frames
    # ------------------------------------------------------

    times = frame_times

    return (
        u,
        history,
        times,
        save_id,
        step,
    )