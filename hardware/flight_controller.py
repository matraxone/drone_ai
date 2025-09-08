"""Flight controller logic with PID stabilization scaffolding."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class PidGains:
    kp: float
    ki: float
    kd: float


class PidController:
    def __init__(self, gains: PidGains) -> None:
        self.gains = gains
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error: float, dt: float) -> float:
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        return (
            self.gains.kp * error + self.gains.ki * self.integral + self.gains.kd * derivative
        )


class AttitudeController:
    def __init__(self, roll_gains: PidGains, pitch_gains: PidGains, yaw_gains: PidGains) -> None:
        self.roll_pid = PidController(roll_gains)
        self.pitch_pid = PidController(pitch_gains)
        self.yaw_pid = PidController(yaw_gains)

    def compute_motor_mix(
        self, roll_error: float, pitch_error: float, yaw_error: float, dt: float
    ) -> Tuple[float, float, float, float]:
        roll_out = self.roll_pid.compute(roll_error, dt)
        pitch_out = self.pitch_pid.compute(pitch_error, dt)
        yaw_out = self.yaw_pid.compute(yaw_error, dt)
        # Placeholder mix for an X quad layout
        return (
            0.5 + roll_out - pitch_out + yaw_out,
            0.5 - roll_out - pitch_out - yaw_out,
            0.5 - roll_out + pitch_out + yaw_out,
            0.5 + roll_out + pitch_out - yaw_out,
        )


