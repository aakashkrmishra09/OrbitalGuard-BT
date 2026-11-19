import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.anomaly_engine.detector import SimplexAnomalyDetector
from src.bt_engine.tree import BehaviourTreeController, SystemMode
from src.mission_planner.validator import (
    DeterministicSafetyGuard,
    MissionCommand,
    SafetyErrorCode,
    SafetyValidationError,
)
from src.perception_sim.telemetry import TelemetryFrame
from src.telemetry_api.main import app


def _valid_mission() -> MissionCommand:
    return MissionCommand(
        mission_id="MSN-SOLAR-B4",
        target_sector="SOLAR_B4",
        target_joint_deg=65.0,
        max_joint_velocity_deg_s=3.5,
        battery_soc_pct=82.0,
        restricted_zones=["KEEPOUT_K2", "THERMAL_SHIELD_A1"],
    )


def test_safety_guard_approves_valid_mission():
    guard = DeterministicSafetyGuard()
    result = guard.validate(_valid_mission())
    assert result["status"] == "APPROVED"
    assert result["enforced_velocity_limit_deg_s"] == 3.5


def test_safety_guard_rejects_keepout_and_joint_limits():
    guard = DeterministicSafetyGuard()

    keepout_cmd = MissionCommand(
        mission_id="MSN-BAD-ZONE",
        target_sector="keepout_k2",
        target_joint_deg=30.0,
        max_joint_velocity_deg_s=2.0,
        battery_soc_pct=90.0,
        restricted_zones=["KEEPOUT_K2"],
    )
    with pytest.raises(SafetyValidationError) as exc_info:
        guard.validate(keepout_cmd)
    assert exc_info.value.code == SafetyErrorCode.KEEP_OUT_ZONE_VIOLATION

    over_joint_cmd = MissionCommand(
        mission_id="MSN-BAD-JOINT",
        target_sector="SOLAR_B4",
        target_joint_deg=160.0,
        max_joint_velocity_deg_s=2.0,
        battery_soc_pct=90.0,
    )
    with pytest.raises(SafetyValidationError) as joint_exc:
        guard.validate(over_joint_cmd)
    assert joint_exc.value.code == SafetyErrorCode.JOINT_LIMIT_EXCEEDED


def test_bt_recovers_from_transient_glare_and_completes():
    ctrl = BehaviourTreeController(mission=_valid_mission(), target_ticks=3)
    frames = [
        TelemetryFrame(tick=1, link_active=True, vision_confidence=0.95, joint_temp_c=42.0, actuator_current_a=4.1),
        TelemetryFrame(tick=2, link_active=True, vision_confidence=0.55, joint_temp_c=43.0, actuator_current_a=4.2),
        TelemetryFrame(tick=3, link_active=True, vision_confidence=0.92, joint_temp_c=43.5, actuator_current_a=4.0),
        TelemetryFrame(tick=4, link_active=True, vision_confidence=0.94, joint_temp_c=44.0, actuator_current_a=3.9),
    ]
    for f in frames:
        ctrl.step(f)

    assert ctrl.history[1].mode == SystemMode.HOLD_REACQUIRE
    assert ctrl.mode == SystemMode.COMPLETED


def test_bt_latches_safe_hold_on_telemetry_blackout():
    ctrl = BehaviourTreeController(mission=_valid_mission(), target_ticks=4)
    frames = [
        TelemetryFrame(tick=1, link_active=True, vision_confidence=0.95, joint_temp_c=40.0, actuator_current_a=3.5),
        TelemetryFrame(tick=2, link_active=False, vision_confidence=0.95, joint_temp_c=40.0, actuator_current_a=3.5),
        TelemetryFrame(tick=3, link_active=True, vision_confidence=0.95, joint_temp_c=40.0, actuator_current_a=3.5),
    ]
    for f in frames:
        ctrl.step(f)

    assert ctrl.mode == SystemMode.SAFE_HOLD
    assert ctrl.last_fault_code == "FAULT_TELEMETRY_BLACKOUT"


def test_fastapi_simulation_endpoint():
    client = TestClient(app)
    payload = {
        "mission": _valid_mission().model_dump(),
        "target_ticks": 2,
        "frames": [
            {"tick": 1, "link_active": True, "vision_confidence": 0.91, "joint_temp_c": 41.0, "actuator_current_a": 3.8},
            {"tick": 2, "link_active": True, "vision_confidence": 0.93, "joint_temp_c": 42.0, "actuator_current_a": 3.7},
        ],
    }
    resp = client.post("/simulation/run", json=payload)
    assert resp.status_code == 200
    assert resp.json()["final_mode"] == "COMPLETED"


def test_root_and_preset_simulation_endpoints():
    client = TestClient(app)

    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "OrbitalGuard-BT" in root_resp.text

    info_resp = client.get("/api/info")
    assert info_resp.status_code == 200
    assert info_resp.json()["service"] == "OrbitalGuard-BT"

    glare_resp = client.post("/simulation/run-preset/glare_recovery")
    assert glare_resp.status_code == 200
    assert glare_resp.json()["final_mode"] == "COMPLETED"

    blackout_resp = client.post("/simulation/run-preset/blackout_safe_hold")
    assert blackout_resp.status_code == 200
    assert blackout_resp.json()["final_mode"] == "SAFE_HOLD"


def test_simplex_ml_intercepts_subtle_bearing_drift_before_hard_limit():
    client = TestClient(app)
    resp = client.post("/simulation/run-preset/subtle_bearing_drift")
    assert resp.status_code == 200
    data = resp.json()
    assert data["final_mode"] == "PREEMPTIVE_HOLD"
    assert data["last_fault_code"] == "ADVISORY_MULTI_SENSOR_DRIFT"
    assert any(step["mode"] == "PREEMPTIVE_HOLD" for step in data["history"])


def test_conformal_false_alarm_rate_stays_bounded_on_nominal_frames():
    detector = SimplexAnomalyDetector(alpha=0.01, seed=42)
    rng = np.random.default_rng(123)

    temp = 41.0
    anomalies = 0
    total_ticks = 200

    for tick in range(1, total_ticks + 1):
        temp = float(np.clip(temp + rng.normal(0.18, 0.08), 39.5, 47.0))
        current = float(np.clip(3.25 + 0.016 * temp + rng.normal(0.0, 0.12), 3.4, 4.4))
        vision = float(np.clip(rng.normal(0.93, 0.018), 0.86, 0.99))
        frame = TelemetryFrame(
            tick=tick,
            link_active=True,
            vision_confidence=vision,
            joint_temp_c=temp,
            actuator_current_a=current,
        )
        assessment = detector.score_frame(frame)
        if assessment.is_anomalous:
            anomalies += 1

    empirical_false_alarm_rate = anomalies / total_ticks
    assert empirical_false_alarm_rate <= 0.05


def test_m_of_n_hysteresis_ignores_isolated_single_tick_spike():
    ctrl = BehaviourTreeController(mission=_valid_mission(), target_ticks=4)

    frames = [
        TelemetryFrame(tick=1, link_active=True, vision_confidence=0.94, joint_temp_c=41.0, actuator_current_a=3.85),
        # Single isolated current spike inside hard safe bounds (8.0 A)
        TelemetryFrame(tick=2, link_active=True, vision_confidence=0.93, joint_temp_c=41.5, actuator_current_a=5.65),
        # Subsequent nominal ticks should keep 2-of-3 count below threshold and reach COMPLETED
        TelemetryFrame(tick=3, link_active=True, vision_confidence=0.94, joint_temp_c=41.8, actuator_current_a=3.88),
        TelemetryFrame(tick=4, link_active=True, vision_confidence=0.95, joint_temp_c=42.1, actuator_current_a=3.90),
    ]

    for frame in frames:
        ctrl.step(frame)

    assert ctrl.history[1].mode != SystemMode.PREEMPTIVE_HOLD
    assert ctrl.mode == SystemMode.COMPLETED


def test_deterministic_hard_fault_overrides_preemptive_hold():
    ctrl = BehaviourTreeController(mission=_valid_mission(), target_ticks=5)

    frames = [
        TelemetryFrame(tick=1, link_active=True, vision_confidence=0.94, joint_temp_c=41.0, actuator_current_a=3.85),
        TelemetryFrame(tick=2, link_active=True, vision_confidence=0.93, joint_temp_c=41.4, actuator_current_a=3.90),
        TelemetryFrame(tick=3, link_active=True, vision_confidence=0.92, joint_temp_c=43.8, actuator_current_a=5.15),
        TelemetryFrame(tick=4, link_active=True, vision_confidence=0.91, joint_temp_c=46.5, actuator_current_a=5.65),
        TelemetryFrame(tick=5, link_active=True, vision_confidence=0.90, joint_temp_c=78.5, actuator_current_a=5.80),
    ]

    for frame in frames:
        ctrl.step(frame)

    assert ctrl.history[3].mode == SystemMode.PREEMPTIVE_HOLD
    assert ctrl.mode == SystemMode.SAFE_HOLD
    assert ctrl.last_fault_code == "FAULT_THERMAL_EXCURSION"