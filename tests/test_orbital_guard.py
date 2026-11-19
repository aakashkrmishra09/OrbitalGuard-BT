import pytest
from fastapi.testclient import TestClient

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