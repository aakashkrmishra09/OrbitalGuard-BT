from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.bt_engine.tree import BehaviourTreeController
from src.mission_planner.validator import (
    DeterministicSafetyGuard,
    MissionCommand,
    SafetyValidationError,
)
from src.perception_sim.telemetry import TelemetryFrame, TelemetryStream

app = FastAPI(title="OrbitalGuard-BT Telemetry & Verification API", version="0.1.0")
guard = DeterministicSafetyGuard()


class SimulationRequest(BaseModel):
    mission: MissionCommand
    target_ticks: int = 3
    frames: list[TelemetryFrame]


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
    try:
        validation = guard.validate(req.mission)
    except SafetyValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error_code": exc.code.value, "message": exc.message},
        ) from exc

    controller = BehaviourTreeController(
        mission=req.mission, target_ticks=req.target_ticks
    )
    stream = TelemetryStream(req.frames)
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