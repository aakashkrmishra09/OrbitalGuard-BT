import numpy as np

from src.anomaly_engine.features import SlidingFeatureExtractor
from src.perception_sim.telemetry import TelemetryFrame


def generate_nominal_feature_matrix(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    extractor = SlidingFeatureExtractor(window_size=3)
    rows: list[np.ndarray] = []

    # Regime 1: Nominal inspection (200 ticks)
    extractor.reset()
    temp = 41.0
    for t in range(1, 201):
        temp = float(np.clip(temp + rng.normal(0.25, 0.18), 39.0, 50.0))
        current = float(np.clip(3.25 + 0.016 * temp + rng.normal(0.0, 0.18), 3.4, 4.6))
        vision = float(np.clip(rng.normal(0.93, 0.025), 0.85, 0.99))
        frame = TelemetryFrame(
            tick=t,
            link_active=True,
            vision_confidence=vision,
            joint_temp_c=temp,
            actuator_current_a=current,
        )
        rows.append(extractor.extract(frame))

    # Regime 2: Active slew / higher load (150 ticks)
    extractor.reset()
    temp = 42.0
    for t in range(1, 151):
        temp = float(np.clip(temp + rng.normal(0.30, 0.20), 40.0, 52.0))
        current = float(np.clip(3.35 + 0.016 * temp + rng.normal(0.0, 0.20), 3.5, 4.8))
        vision = float(np.clip(rng.normal(0.90, 0.03), 0.82, 0.98))
        frame = TelemetryFrame(
            tick=t,
            link_active=True,
            vision_confidence=vision,
            joint_temp_c=temp,
            actuator_current_a=current,
        )
        rows.append(extractor.extract(frame))

    # Regime 3: Station keeping / low load (150 ticks)
    extractor.reset()
    temp = 39.5
    for t in range(1, 151):
        temp = float(np.clip(temp + rng.normal(0.10, 0.15), 37.5, 46.0))
        current = float(np.clip(3.15 + 0.015 * temp + rng.normal(0.0, 0.16), 3.2, 4.3))
        vision = float(np.clip(rng.normal(0.95, 0.02), 0.87, 0.99))
        frame = TelemetryFrame(
            tick=t,
            link_active=True,
            vision_confidence=vision,
            joint_temp_c=temp,
            actuator_current_a=current,
        )
        rows.append(extractor.extract(frame))

    return np.vstack(rows)