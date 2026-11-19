# OrbitalGuard-BT

Deterministic safety validation, Behaviour Tree fault recovery, and automated verification harness for autonomous orbital and ground-inspection systems (inspired by safety-critical space robotics and telemetry workflows).

```mermaid
flowchart LR
    CMD[MissionCommand JSON] --> GUARD[DeterministicSafetyGuard]
    GUARD -->|Approved| BT[BehaviourTreeController]
    GUARD -->|Rejected 422| ERR[Explicit SafetyErrorCode]
    SIM[TelemetryStream / Fault Injection] --> BT
    BT -->|Nominal / Glare Retry| INSP[SectorInspection / HOLD_REACQUIRE]
    BT -->|Blackout / Thermal / Overcurrent| SAFE[Latched SAFE_HOLD]
    BT --> API[FastAPI Telemetry & Verification API]