from enum import Enum
from pydantic import BaseModel, Field, field_validator


class SafetyErrorCode(str, Enum):
    JOINT_LIMIT_EXCEEDED = "ERR_JOINT_LIMIT_EXCEEDED"
    VELOCITY_LIMIT_EXCEEDED = "ERR_VELOCITY_LIMIT_EXCEEDED"
    INSUFFICIENT_BATTERY = "ERR_INSUFFICIENT_BATTERY"
    KEEP_OUT_ZONE_VIOLATION = "ERR_KEEP_OUT_ZONE"


class SafetyValidationError(Exception):
    def __init__(self, code: SafetyErrorCode, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code.value}: {message}")


class MissionCommand(BaseModel):
    mission_id: str = Field(..., min_length=3, max_length=64)
    target_sector: str = Field(..., min_length=2, max_length=32)
    target_joint_deg: float = Field(..., ge=-180.0, le=180.0)
    max_joint_velocity_deg_s: float = Field(..., gt=0.0, le=15.0)
    battery_soc_pct: float = Field(..., ge=0.0, le=100.0)
    restricted_zones: list[str] = Field(default_factory=list)

    @field_validator("target_sector")
    @classmethod
    def normalize_sector(cls, value: str) -> str:
        return value.strip().upper()


class DeterministicSafetyGuard:
    """Enforces hard kinematic, power, and keep-out preconditions prior to execution."""

    MAX_SAFE_JOINT_DEG = 135.0
    MAX_SAFE_VELOCITY_DEG_S = 5.0
    MIN_REQUIRED_BATTERY_PCT = 20.0

    def validate(self, cmd: MissionCommand) -> dict:
        if abs(cmd.target_joint_deg) > self.MAX_SAFE_JOINT_DEG:
            raise SafetyValidationError(
                SafetyErrorCode.JOINT_LIMIT_EXCEEDED,
                f"Target angle {cmd.target_joint_deg} deg exceeds +/-{self.MAX_SAFE_JOINT_DEG} deg limit.",
            )

        if cmd.max_joint_velocity_deg_s > self.MAX_SAFE_VELOCITY_DEG_S:
            raise SafetyValidationError(
                SafetyErrorCode.VELOCITY_LIMIT_EXCEEDED,
                f"Velocity {cmd.max_joint_velocity_deg_s} deg/s exceeds safe limit {self.MAX_SAFE_VELOCITY_DEG_S} deg/s.",
            )

        if cmd.battery_soc_pct < self.MIN_REQUIRED_BATTERY_PCT:
            raise SafetyValidationError(
                SafetyErrorCode.INSUFFICIENT_BATTERY,
                f"Battery SoC {cmd.battery_soc_pct}% below minimum {self.MIN_REQUIRED_BATTERY_PCT}% for mission.",
            )

        blocked = {z.strip().upper() for z in cmd.restricted_zones}
        if cmd.target_sector in blocked:
            raise SafetyValidationError(
                SafetyErrorCode.KEEP_OUT_ZONE_VIOLATION,
                f"Target sector '{cmd.target_sector}' is inside a restricted keep-out zone.",
            )

        return {
            "status": "APPROVED",
            "mission_id": cmd.mission_id,
            "sector": cmd.target_sector,
            "enforced_velocity_limit_deg_s": min(
                cmd.max_joint_velocity_deg_s, self.MAX_SAFE_VELOCITY_DEG_S
            ),
        }