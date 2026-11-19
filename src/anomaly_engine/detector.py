from dataclasses import dataclass
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.anomaly_engine.features import FEATURE_NAMES, SlidingFeatureExtractor
from src.anomaly_engine.nominal_baseline import generate_nominal_feature_matrix
from src.perception_sim.telemetry import TelemetryFrame


HUMAN_LABELS = {
    "vision_confidence": "Vision Confidence",
    "joint_temp_c": "Joint Temperature",
    "actuator_current_a": "Actuator Current",
    "thermal_gradient_c_per_tick": "Thermal Gradient (dT/dt)",
    "current_temp_coupling": "Current-Thermal Coupling",
    "vision_jitter_std": "Vision Jitter",
}


@dataclass(frozen=True)
class AnomalyAssessment:
    is_anomalous: bool
    severity: float
    p_value: float
    log_likelihood: float
    primary_driver: str


class SimplexAnomalyDetector:
    """
    Advisory unsupervised anomaly detector using a 3-regime GMM calibrated
    via Split Conformal Prediction. Never overrides hard deterministic limits.
    """

    def __init__(self, alpha: float = 0.01, seed: int = 42) -> None:
        self.alpha = alpha
        self.extractor = SlidingFeatureExtractor(window_size=3)
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(
            n_components=3,
            covariance_type="full",
            random_state=seed,
        )

        data = generate_nominal_feature_matrix(seed=seed)
        split_idx = int(len(data) * 0.8)
        train_x, cal_x = data[:split_idx], data[split_idx:]

        train_scaled = self.scaler.fit_transform(train_x)
        self.gmm.fit(train_scaled)

        # Conformal calibration: nonconformity score = negative log-likelihood
        cal_scaled = self.scaler.transform(cal_x)
        self._cal_nonconformity = np.sort(-self.gmm.score_samples(cal_scaled))
        self._threshold = float(np.quantile(self._cal_nonconformity, 1.0 - self.alpha))

    def reset(self) -> None:
        self.extractor.reset()

    def score_frame(self, frame: TelemetryFrame) -> AnomalyAssessment:
        vec = self.extractor.extract(frame).reshape(1, -1)
        scaled = self.scaler.transform(vec)
        log_lik = float(self.gmm.score_samples(scaled)[0])
        nonconformity = -log_lik

        # Empirical conformal p-value
        n_cal = len(self._cal_nonconformity)
        count_ge = int(np.sum(self._cal_nonconformity >= nonconformity))
        p_value = float((count_ge + 1) / (n_cal + 1))

        # Calibrated severity in [0.0, 1.0]
        ratio = nonconformity / max(self._threshold, 1e-6)
        severity = float(np.clip(ratio / 1.5, 0.0, 1.0))

        # Feature attribution via standardized z-score magnitude
        z_scores = np.abs(scaled[0])
        top_idx = int(np.argmax(z_scores))
        top_feature = FEATURE_NAMES[top_idx]
        driver = f"{HUMAN_LABELS[top_feature]} (+{z_scores[top_idx]:.1f} sigma)"

        return AnomalyAssessment(
            is_anomalous=nonconformity > self._threshold,
            severity=round(severity, 3),
            p_value=round(p_value, 4),
            log_likelihood=round(log_lik, 3),
            primary_driver=driver,
        )