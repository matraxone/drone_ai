"""Motor control module for ESC and brushless motors.

This module will provide PWM signal generation and ramping logic
to control four ESCs connected to the flight controller or GPIO via
an intermediate microcontroller. Implementations will be platform-specific.
"""

from typing import Protocol, List


class MotorOutput(Protocol):
    def set_duty_cycle(self, channel_index: int, duty_cycle: float) -> None:
        """Set duty cycle (0.0-1.0) for a specific motor channel."""


class MotorController:
    """High-level controller for four motors."""

    def __init__(self, pwm_driver: MotorOutput, motor_channels: List[int] | None = None) -> None:
        self.pwm_driver = pwm_driver
        self.motor_channels = motor_channels or [0, 1, 2, 3]

    def arm(self) -> None:
        for index in self.motor_channels:
            self.pwm_driver.set_duty_cycle(index, 0.0)

    def set_throttles(self, throttles: List[float]) -> None:
        if len(throttles) != len(self.motor_channels):
            raise ValueError("Throttle list length must match motor channels")
        for index, throttle in zip(self.motor_channels, throttles):
            self.pwm_driver.set_duty_cycle(index, max(0.0, min(1.0, throttle)))


