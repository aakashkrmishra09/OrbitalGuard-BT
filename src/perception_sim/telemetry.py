from dataclasses import dataclass
from typing import Optional

MIN_SAFE_VISION_CONFIDENCE = 0.70
MAX_SAFE_TEMP_C = 75.0
MAX_SAFE_CURRENT_A = 8.0


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


def is_frame_healthy(frame: TelemetryFrame) -> tuple[bool, Optional[str], str]:
    """
    Deterministic frame health check used by the Behaviour Tree before/alongside
    advisory ML checks.
    Returns: (is_healthy, fault_code, detail_message)
    """
    if not frame.link_active or frame.fault_injected == "LINK_LOSS":
        return (
            False,
            "FAULT_TELEMETRY_BLACKOUT",
            "Telemetry link inactive or link-loss fault injected",
        )

    if frame.joint_temp_c > MAX_SAFE_TEMP_C or frame.fault_injected == "THERMAL_EXCURSION":
        return (
            False,
            "FAULT_THERMAL_EXCURSION",
            f"Joint temperature {frame.joint_temp_c:.1f} C exceeded safe limit ({MAX_SAFE_TEMP_C:.1f} C)",
        )

    if (
        frame.actuator_current_a > MAX_SAFE_CURRENT_A
        or frame.fault_injected == "ACTUATOR_OVERCURRENT"
    ):
        return (
            False,
            "FAULT_ACTUATOR_OVERCURRENT",
            f"Actuator current {frame.actuator_current_a:.2f} A exceeded safe limit ({MAX_SAFE_CURRENT_A:.2f} A)",
        )

    if frame.vision_confidence < MIN_SAFE_VISION_CONFIDENCE or frame.fault_injected == "SOLAR_GLARE":
        return (
            False,
            "FAULT_LOW_VISION_CONFIDENCE",
            f"Vision confidence {frame.vision_confidence:.2f} below threshold ({MIN_SAFE_VISION_CONFIDENCE:.2f})",
        )

    return True, None, "Nominal telemetry frame"