from dataclasses import dataclass
from typing import Optional


@dataclass
class TelemetryFrame:
    tick: int
    link_active: bool
    vision_confidence: float
    joint_temp_c: float
    actuator_current_a: float
    fault_injected: Optional[str] = None


class TelemetryStream:
    """Produces deterministic or fault-injected telemetry ticks for BT evaluation."""

    MIN_VISION_CONFIDENCE = 0.80
    MAX_SAFE_TEMP_C = 75.0
    MAX_SAFE_CURRENT_A = 12.0

    def __init__(self, frames: list[TelemetryFrame]):
        self._frames = frames
        self._index = 0

    def next_frame(self) -> Optional[TelemetryFrame]:
        if self._index >= len(self._frames):
            return None
        frame = self._frames[self._index]
        self._index += 1
        return frame

    @classmethod
    def is_frame_healthy(cls, frame: TelemetryFrame) -> tuple[bool, Optional[str]]:
        if not frame.link_active:
            return False, "FAULT_TELEMETRY_BLACKOUT"
        if frame.joint_temp_c > cls.MAX_SAFE_TEMP_C:
            return False, "FAULT_THERMAL_EXCURSION"
        if frame.actuator_current_a > cls.MAX_SAFE_CURRENT_A:
            return False, "FAULT_ACTUATOR_OVERCURRENT"
        if frame.vision_confidence < cls.MIN_VISION_CONFIDENCE:
            return False, "DEGRADED_VISION_GLARE"
        return True, None