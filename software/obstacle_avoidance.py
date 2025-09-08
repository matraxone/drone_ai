"""Obstacle avoidance logic using range sensors and simple reactive control."""

from typing import Optional


def compute_avoidance_command(distance_m: Optional[float], min_clearance_m: float = 1.5) -> float:
    """Return a yaw rate command [-1, 1] to steer away if too close.

    This is a placeholder. Positive means turn right, negative left.
    """
    if distance_m is None:
        return 0.0
    if distance_m < min_clearance_m:
        return 0.5
    return 0.0


