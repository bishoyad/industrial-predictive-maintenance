"""
Pydantic schemas for request / response validation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    """
    A single sensor reading snapshot for one engine at one cycle.
    Sensor values match the NASA C-MAPSS naming after dropping low-variance sensors.
    """
    unit_id: int = Field(..., ge=1, description="Engine unit identifier")
    cycle: int = Field(..., ge=1, description="Operational cycle number")
    op_setting_1: float = Field(0.0, description="Operational setting 1")
    op_setting_2: float = Field(0.0, description="Operational setting 2")
    op_setting_3: float = Field(0.0, description="Operational setting 3")
    sensor_2: float = Field(...)
    sensor_3: float = Field(...)
    sensor_4: float = Field(...)
    sensor_7: float = Field(...)
    sensor_8: float = Field(...)
    sensor_9: float = Field(...)
    sensor_11: float = Field(...)
    sensor_12: float = Field(...)
    sensor_13: float = Field(...)
    sensor_14: float = Field(...)
    sensor_15: float = Field(...)
    sensor_17: float = Field(...)
    sensor_20: float = Field(...)
    sensor_21: float = Field(...)

    model_config = {"json_schema_extra": {
        "example": {
            "unit_id": 1,
            "cycle": 150,
            "op_setting_1": -0.0007,
            "op_setting_2": -0.0004,
            "op_setting_3": 100.0,
            "sensor_2": 641.21,
            "sensor_3": 1583.76,
            "sensor_4": 1404.22,
            "sensor_7": 554.36,
            "sensor_8": 2388.02,
            "sensor_9": 9044.54,
            "sensor_11": 47.47,
            "sensor_12": 521.66,
            "sensor_13": 2388.09,
            "sensor_14": 8138.62,
            "sensor_15": 8.4195,
            "sensor_17": 392,
            "sensor_20": 38.86,
            "sensor_21": 23.3735,
        }
    }}


class RULRequest(BaseModel):
    """Sliding window of sensor readings for one engine."""
    readings: List[SensorReading] = Field(
        ...,
        min_length=1,
        description="Time-ordered sequence of sensor readings (at least 1 cycle).",
    )


class RULResponse(BaseModel):
    unit_id: int
    predicted_rul: float = Field(..., description="Predicted Remaining Useful Life in cycles")
    health_status: str = Field(..., description="HEALTHY | WARNING | CRITICAL")
    confidence_pct: Optional[float] = Field(None, description="Model confidence estimate (0-100)")


class AnomalyRequest(BaseModel):
    """Single or batch sensor readings for anomaly detection."""
    readings: List[SensorReading]


class AnomalyResponse(BaseModel):
    unit_id: int
    cycle: int
    is_anomaly: bool
    anomaly_score: float = Field(..., description="Higher = more anomalous")
    message: str
