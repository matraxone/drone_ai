"""Sensor interfaces for IMU (MPU6050), GPS, and range sensors."""

from dataclasses import dataclass
from typing import Protocol, Optional


@dataclass
class ImuReading:
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


@dataclass
class GpsReading:
    latitude: float
    longitude: float
    altitude_m: float
    fix_quality: int


class Imu(Protocol):
    def read(self) -> ImuReading: ...


class Gps(Protocol):
    def read(self) -> Optional[GpsReading]: ...


class Rangefinder(Protocol):
    def read_distance_m(self) -> Optional[float]: ...


