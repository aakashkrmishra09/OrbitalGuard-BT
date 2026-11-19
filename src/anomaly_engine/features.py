from collections import deque
from dataclasses import dataclass
import numpy as np

from src.perception_sim.telemetry import TelemetryFrame


FEATURE_NAMES = [
    "vision_confidence",
    "joint_temp_c",
    "actuator_current_a",
    "thermal_gradient_c_per_tick",
    "current_temp_coupling",
    "vision_jitter_std",
]


@dataclass
class SlidingFeatureExtractor:
    window_size: int = 3

    def __post_init__(self) -> None:
        self._window: deque[TelemetryFrame] = deque(maxlen=self.window_size)

    def reset(self) -> None:
        self._window.clear()

    def extract(self, frame: TelemetryFrame) -> np.ndarray:
        self._window.append(frame)
        frames = list(self._window)

        if len(frames) >= 2:
            temp_grad = frames[-1].joint_temp_c - frames[-2].joint_temp_c
        else:
            temp_grad = 0.20

        # Expected nominal thermal-current baseline (~3.9A around 42-44C)
        expected_current = 3.25 + 0.015 * frame.joint_temp_c
        current_coupling = frame.actuator_current_a - expected_current

        # Only include frames with nominal vision (>= 0.70) so glare recovery doesn't create artificial jitter
        healthy_vis = [f.vision_confidence for f in frames if f.vision_confidence >= 0.70]
        if len(healthy_vis) >= 2:
            vis_jitter = float(np.std(healthy_vis))
        else:
            vis_jitter = 0.015

        return np.array(
            [
                frame.vision_confidence,
                frame.joint_temp_c,
                frame.actuator_current_a,
                temp_grad,
                current_coupling,
                vis_jitter,
            ],
            dtype=float,
        )