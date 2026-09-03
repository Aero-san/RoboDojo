"""Episode termination policy shared by RoboDojo evaluation modes."""

from __future__ import annotations


def resolve_episode_status(
    *,
    reward: float,
    success: bool,
    steps: int,
    max_steps: int,
    fixed_horizon: bool,
) -> tuple[bool, bool]:
    """Return ``(success, should_end)`` for the current environment state."""

    reached_horizon = steps >= max_steps
    # Environment failure queries are terminal labels even when fixed-horizon
    # collection keeps the simulator running for more observations/actions.
    if not success:
        return False, reached_horizon or not fixed_horizon
    if reward > 1 - 1e-3:
        return True, reached_horizon or not fixed_horizon
    if reached_horizon:
        return False, True
    return success, False
