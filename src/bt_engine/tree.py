from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.mission_planner.validator import MissionCommand
from src.perception_sim.telemetry import TelemetryFrame, TelemetryStream


class NodeStatus(str, Enum):
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class SystemMode(str, Enum):
    IDLE = "IDLE"
    NOMINAL_INSPECTION = "NOMINAL_INSPECTION"
    HOLD_REACQUIRE = "HOLD_REACQUIRE"
    SAFE_HOLD = "SAFE_HOLD"
    COMPLETED = "COMPLETED"


@dataclass
class TickRecord:
    tick: int
    mode: SystemMode
    active_node: str
    status: NodeStatus
    reason: str
    vision_confidence: float
    link_active: bool


@dataclass
class BehaviourTreeController:
    mission: MissionCommand
    target_ticks: int = 5
    mode: SystemMode = SystemMode.IDLE
    completed_inspection_ticks: int = 0
    reacquire_attempts: int = 0
    max_reacquire_attempts: int = 3
    history: list[TickRecord] = field(default_factory=list)
    last_fault_code: Optional[str] = None

    def step(self, frame: TelemetryFrame) -> TickRecord:
        if self.mode == SystemMode.SAFE_HOLD:
            record = TickRecord(
                tick=frame.tick,
                mode=SystemMode.SAFE_HOLD,
                active_node="SafeHoldLatch",
                status=NodeStatus.RUNNING,
                reason=f"Latched in SAFE_HOLD ({self.last_fault_code or 'CRITICAL_FAULT'})",
                vision_confidence=frame.vision_confidence,
                link_active=frame.link_active,
            )
            self.history.append(record)
            return record

        healthy, condition = TelemetryStream.is_frame_healthy(frame)

        if condition in {
            "FAULT_TELEMETRY_BLACKOUT",
            "FAULT_THERMAL_EXCURSION",
            "FAULT_ACTUATOR_OVERCURRENT",
        }:
            self.mode = SystemMode.SAFE_HOLD
            self.last_fault_code = condition
            record = TickRecord(
                tick=frame.tick,
                mode=SystemMode.SAFE_HOLD,
                active_node="CriticalFaultSafeHold",
                status=NodeStatus.FAILURE,
                reason=f"Entered SAFE_HOLD due to {condition}",
                vision_confidence=frame.vision_confidence,
                link_active=frame.link_active,
            )
            self.history.append(record)
            return record

        if condition == "DEGRADED_VISION_GLARE":
            self.reacquire_attempts += 1
            if self.reacquire_attempts > self.max_reacquire_attempts:
                self.mode = SystemMode.SAFE_HOLD
                self.last_fault_code = "EXCEEDED_GLARE_REACQUIRE_RETRIES"
                record = TickRecord(
                    tick=frame.tick,
                    mode=SystemMode.SAFE_HOLD,
                    active_node="GlareEscalationSafeHold",
                    status=NodeStatus.FAILURE,
                    reason="Max vision re-acquisition attempts exceeded under glare",
                    vision_confidence=frame.vision_confidence,
                    link_active=frame.link_active,
                )
                self.history.append(record)
                return record

            self.mode = SystemMode.HOLD_REACQUIRE
            record = TickRecord(
                tick=frame.tick,
                mode=SystemMode.HOLD_REACQUIRE,
                active_node="HoldAndAdjustPose",
                status=NodeStatus.RUNNING,
                reason=(
                    f"Low vision confidence ({frame.vision_confidence:.2f}); "
                    f"holding pose (attempt {self.reacquire_attempts}/{self.max_reacquire_attempts})"
                ),
                vision_confidence=frame.vision_confidence,
                link_active=frame.link_active,
            )
            self.history.append(record)
            return record

        assert healthy
        self.reacquire_attempts = 0
        self.completed_inspection_ticks += 1

        if self.completed_inspection_ticks >= self.target_ticks:
            self.mode = SystemMode.COMPLETED
            status = NodeStatus.SUCCESS
            reason = f"Completed target inspection ({self.completed_inspection_ticks}/{self.target_ticks} ticks)"
        else:
            self.mode = SystemMode.NOMINAL_INSPECTION
            status = NodeStatus.RUNNING
            reason = (
                f"Inspecting sector {self.mission.target_sector} "
                f"({self.completed_inspection_ticks}/{self.target_ticks} ticks)"
            )

        record = TickRecord(
            tick=frame.tick,
            mode=self.mode,
            active_node="SectorInspectionSequence",
            status=status,
            reason=reason,
            vision_confidence=frame.vision_confidence,
            link_active=frame.link_active,
        )
        self.history.append(record)
        return record