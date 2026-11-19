from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from src.anomaly_engine.detector import SimplexAnomalyDetector
from src.mission_planner.validator import MissionCommand
from src.perception_sim.telemetry import TelemetryFrame, is_frame_healthy


class SystemMode(str, Enum):
    NOMINAL_INSPECTION = "NOMINAL_INSPECTION"
    HOLD_REACQUIRE = "HOLD_REACQUIRE"
    PREEMPTIVE_HOLD = "PREEMPTIVE_HOLD"
    SAFE_HOLD = "SAFE_HOLD"
    COMPLETED = "COMPLETED"


@dataclass
class StepRecord:
    tick: int
    mode: str
    active_node: str
    status: str
    vision_confidence: float
    link_active: bool
    reason: str
    anomaly_severity: float = 0.0
    anomaly_driver: str = "Nominal"


@dataclass
class BehaviourTreeController:
    mission: MissionCommand
    target_ticks: int = 3
    max_reacquire_retries: int = 2
    mode: SystemMode = SystemMode.NOMINAL_INSPECTION
    completed_ticks: int = 0
    reacquire_attempts: int = 0
    last_fault_code: str | None = None
    history: list[StepRecord] = field(default_factory=list)
    detector: SimplexAnomalyDetector = field(default_factory=SimplexAnomalyDetector)
    _anomaly_window: deque[bool] = field(default_factory=lambda: deque(maxlen=3))

    def step(self, frame: TelemetryFrame) -> StepRecord:
        # 1. LATCHED SAFE_HOLD OVERRIDE (Non-bypassable)
        if self.mode == SystemMode.SAFE_HOLD:
            rec = StepRecord(
                tick=frame.tick,
                mode=self.mode.value,
                active_node="SafeHoldLatchNode",
                status="RUNNING",
                vision_confidence=frame.vision_confidence,
                link_active=frame.link_active,
                reason=f"Latched in SAFE_HOLD ({self.last_fault_code})",
                anomaly_severity=1.0,
                anomaly_driver=self.last_fault_code or "SAFE_HOLD",
            )
            self.history.append(rec)
            return rec

        # 2. DETERMINISTIC HARD SAFETY EVALUATION (Authoritative Layer 1)
        healthy, fault_code, detail = is_frame_healthy(frame)

        # Critical hardware/link faults immediately latch SAFE_HOLD
        if not healthy and fault_code in {
            "FAULT_TELEMETRY_BLACKOUT",
            "FAULT_THERMAL_EXCURSION",
            "FAULT_ACTUATOR_OVERCURRENT",
        }:
            self.mode = SystemMode.SAFE_HOLD
            self.last_fault_code = fault_code
            rec = StepRecord(
                tick=frame.tick,
                mode=self.mode.value,
                active_node="CriticalFaultSafeHoldNode",
                status="FAILURE",
                vision_confidence=frame.vision_confidence,
                link_active=frame.link_active,
                reason=detail,
                anomaly_severity=1.0,
                anomaly_driver=fault_code,
            )
            self.history.append(rec)
            return rec

        # Transient vision confidence loss -> bounded HOLD_REACQUIRE
        if not healthy and fault_code == "FAULT_LOW_VISION_CONFIDENCE":
            self.reacquire_attempts += 1
            self.last_fault_code = fault_code
            if self.reacquire_attempts > self.max_reacquire_retries:
                self.mode = SystemMode.SAFE_HOLD
                rec = StepRecord(
                    tick=frame.tick,
                    mode=self.mode.value,
                    active_node="ReacquireBudgetExceededNode",
                    status="FAILURE",
                    vision_confidence=frame.vision_confidence,
                    link_active=frame.link_active,
                    reason="Exceeded vision re-acquisition retry budget; latching SAFE_HOLD",
                    anomaly_severity=1.0,
                    anomaly_driver=fault_code,
                )
            else:
                self.mode = SystemMode.HOLD_REACQUIRE
                rec = StepRecord(
                    tick=frame.tick,
                    mode=self.mode.value,
                    active_node="HoldAndReacquireNode",
                    status="RUNNING",
                    vision_confidence=frame.vision_confidence,
                    link_active=frame.link_active,
                    reason=f"{detail} (retry {self.reacquire_attempts}/{self.max_reacquire_retries})",
                    anomaly_severity=0.75,
                    anomaly_driver=fault_code,
                )
            self.history.append(rec)
            return rec

        # 3. SIMPLEX ADVISORY ML LAYER (Layer 2: Multi-sensor drift inside safe bounds)
        assessment = self.detector.score_frame(frame)
        self._anomaly_window.append(assessment.is_anomalous)

        # M-of-N hysteresis: require 2 of last 3 ticks anomalous to trigger PREEMPTIVE_HOLD
        if sum(self._anomaly_window) >= 2:
            self.mode = SystemMode.PREEMPTIVE_HOLD
            self.last_fault_code = "ADVISORY_MULTI_SENSOR_DRIFT"
            rec = StepRecord(
                tick=frame.tick,
                mode=self.mode.value,
                active_node="SimplexAnomalyPreemptiveHoldNode",
                status="RUNNING",
                vision_confidence=frame.vision_confidence,
                link_active=frame.link_active,
                reason=(
                    f"Early multi-sensor drift intercepted before hard threshold breach "
                    f"(p={assessment.p_value}, driver: {assessment.primary_driver})"
                ),
                anomaly_severity=assessment.severity,
                anomaly_driver=assessment.primary_driver,
            )
            self.history.append(rec)
            return rec

        # 4. NOMINAL EXECUTION
        self.reacquire_attempts = 0
        self.completed_ticks += 1
        if self.completed_ticks >= self.target_ticks:
            self.mode = SystemMode.COMPLETED
            node = "MissionCompleteNode"
            status = "SUCCESS"
            reason = f"Completed {self.completed_ticks}/{self.target_ticks} nominal ticks"
        else:
            self.mode = SystemMode.NOMINAL_INSPECTION
            node = "NominalInspectionStepNode"
            status = "RUNNING"
            reason = f"Nominal inspection progress ({self.completed_ticks}/{self.target_ticks})"

        rec = StepRecord(
            tick=frame.tick,
            mode=self.mode.value,
            active_node=node,
            status=status,
            vision_confidence=frame.vision_confidence,
            link_active=frame.link_active,
            reason=reason,
            anomaly_severity=assessment.severity,
            anomaly_driver=assessment.primary_driver if assessment.is_anomalous else "Nominal",
        )
        self.history.append(rec)
        return rec