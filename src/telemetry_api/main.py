from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.bt_engine.tree import BehaviourTreeController
from src.mission_planner.validator import (
    DeterministicSafetyGuard,
    MissionCommand,
    SafetyValidationError,
)
from src.perception_sim.telemetry import TelemetryFrame, TelemetryStream

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

app = FastAPI(
    title="OrbitalGuard-BT Telemetry & Verification API",
    description=(
        "Deterministic pre-execution safety guard, Behaviour Tree fault-recovery "
        "controller, and telemetry fault-injection verification harness."
    ),
    version="0.3.0",
)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

guard = DeterministicSafetyGuard()


class SimulationRequest(BaseModel):
    mission: MissionCommand
    target_ticks: int = 3
    frames: list[TelemetryFrame]


def _default_mission() -> MissionCommand:
    return MissionCommand(
        mission_id="MSN-LUNAR-ARM-04",
        target_sector="SOLAR_ARRAY_B4",
        target_joint_deg=65.0,
        max_joint_velocity_deg_s=3.5,
        battery_soc_pct=84.0,
        restricted_zones=["KEEPOUT_K2", "THERMAL_SHIELD_A1"],
    )


def _preset_frames(preset_name: str) -> tuple[int, list[TelemetryFrame]]:
    key = preset_name.strip().lower()
    if key == "nominal":
        return 3, [
            TelemetryFrame(tick=1, link_active=True, vision_confidence=0.94, joint_temp_c=41.0, actuator_current_a=3.9),
            TelemetryFrame(tick=2, link_active=True, vision_confidence=0.92, joint_temp_c=42.2, actuator_current_a=4.0),
            TelemetryFrame(tick=3, link_active=True, vision_confidence=0.95, joint_temp_c=43.0, actuator_current_a=3.8),
        ]
    if key == "glare_recovery":
        return 3, [
            TelemetryFrame(tick=1, link_active=True, vision_confidence=0.93, joint_temp_c=41.5, actuator_current_a=4.0),
            TelemetryFrame(tick=2, link_active=True, vision_confidence=0.58, joint_temp_c=42.5, actuator_current_a=4.1, fault_injected="SOLAR_GLARE"),
            TelemetryFrame(tick=3, link_active=True, vision_confidence=0.91, joint_temp_c=43.0, actuator_current_a=3.9),
            TelemetryFrame(tick=4, link_active=True, vision_confidence=0.94, joint_temp_c=43.5, actuator_current_a=3.8),
        ]
    if key == "blackout_safe_hold":
        return 4, [
            TelemetryFrame(tick=1, link_active=True, vision_confidence=0.94, joint_temp_c=40.5, actuator_current_a=3.7),
            TelemetryFrame(tick=2, link_active=False, vision_confidence=0.94, joint_temp_c=41.0, actuator_current_a=3.7, fault_injected="LINK_LOSS"),
            TelemetryFrame(tick=3, link_active=True, vision_confidence=0.94, joint_temp_c=41.2, actuator_current_a=3.6),
        ]
    if key == "subtle_bearing_drift":
        return 5, [
            TelemetryFrame(tick=1, link_active=True, vision_confidence=0.94, joint_temp_c=41.0, actuator_current_a=3.85),
            TelemetryFrame(tick=2, link_active=True, vision_confidence=0.93, joint_temp_c=41.4, actuator_current_a=3.90),
            TelemetryFrame(tick=3, link_active=True, vision_confidence=0.92, joint_temp_c=43.8, actuator_current_a=5.15, fault_injected="BEARING_FRICTION_DRIFT"),
            TelemetryFrame(tick=4, link_active=True, vision_confidence=0.91, joint_temp_c=46.5, actuator_current_a=5.65, fault_injected="BEARING_FRICTION_DRIFT"),
            TelemetryFrame(tick=5, link_active=True, vision_confidence=0.91, joint_temp_c=49.2, actuator_current_a=5.90, fault_injected="BEARING_FRICTION_DRIFT"),
        ]
    raise HTTPException(
        status_code=404,
        detail=f"Unknown preset '{preset_name}'. Choose from: nominal, glare_recovery, blackout_safe_hold.",
    )


def _execute_simulation(
    mission: MissionCommand, target_ticks: int, frames: list[TelemetryFrame]
) -> dict:
    try:
        validation = guard.validate(mission)
    except SafetyValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error_code": exc.code.value, "message": exc.message},
        ) from exc

    controller = BehaviourTreeController(mission=mission, target_ticks=target_ticks)
    stream = TelemetryStream(frames)
    while True:
        frame = stream.next_frame()
        if frame is None:
            break
        controller.step(frame)

    return {
        "validation": validation,
        "final_mode": controller.mode.value,
        "last_fault_code": controller.last_fault_code,
        "ticks_executed": len(controller.history),
        "history": [item.__dict__ for item in controller.history],
    }


@app.get("/")
def root() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/info")
def api_info() -> dict:
    return {
        "service": "OrbitalGuard-BT",
        "version": "0.3.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/api/info",
            "/mission/validate",
            "/simulation/run",
            "/simulation/run-preset/{preset_name}",
        ],
        "presets": ["nominal", "glare_recovery", "blackout_safe_hold"],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "OrbitalGuard-BT"}


@app.post("/mission/validate")
def validate_mission(cmd: MissionCommand) -> dict:
    try:
        return guard.validate(cmd)
    except SafetyValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error_code": exc.code.value, "message": exc.message},
        ) from exc


@app.post("/simulation/run")
def run_simulation(req: SimulationRequest) -> dict:
    return _execute_simulation(req.mission, req.target_ticks, req.frames)


@app.post("/simulation/run-preset/{preset_name}")
def run_preset_simulation(preset_name: str) -> dict:
    target_ticks, frames = _preset_frames(preset_name)
    return _execute_simulation(_default_mission(), target_ticks, frames)