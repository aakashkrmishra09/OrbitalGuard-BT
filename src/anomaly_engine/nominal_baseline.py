import numpy as np

from src.anomaly_engine.features import SlidingFeatureExtractor
from src.perception_sim.telemetry import TelemetryFrame


def generate_nominal_feature_matrix(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    extractor = SlidingFeatureExtractor(window_size=3)
    rows: list[np.ndarray] = []

    # Regime 1: Nominal inspection (200 ticks)
    extractor.reset()
    temp = 40.5
    for t in range(1, 201):
        temp = min(46.0, temp + float(rng.normal(0.18, 0.08)))
        current = float(np.clip(3.2 + 0.016 * temp + rng.normal(0.0, 0.12), 3.4, 4.4))
        vision = float(np.clip(rng.normal(0.93, 0.018), 0.86, 0.99))
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
        temp = min(48.0, temp + float(rng.normal(0.22, 0.09)))
        current = float(np.clip(3.35 + 0.016 * temp + rng.normal(0.0, 0.14), 3.6, 4.6))
        vision = float(np.clip(rng.normal(0.90, 0.022), 0.84, 0.97))
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
        temp = max(38.0, temp + float(rng.normal(0.05, 0.06)))
        current = float(np.clip(3.1 + 0.016 * temp + rng.normal(0.0, 0.10), 3.2, 4.1))
        vision = float(np.clip(rng.normal(0.95, 0.015), 0.88, 0.99))
        frame = TelemetryFrame(
            tick=t,
            link_active=True,
            vision_confidence=vision,
            joint_temp_c=temp,
            actuator_current_a=current,
        )
        rows.append(extractor.extract(frame))

    return np.vstack(rows)