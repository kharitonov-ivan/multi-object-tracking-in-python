import numpy as np

from src.common.gaussian_density import GaussianDensity
from src.common.hypothesis_reduction import HypothesisReduction
from src.common.normalize_log_weights import normalize_log_weights
from src.configs import SensorModelConfig
from src.measurement_models import MeasurementModel
from src.motion_models import BaseMotionModel
from src.trackers.single_object_trackers.base_single_object_tracker import (
    SingleObjectTracker,
)


class ProbabilisticDataAssociationTracker(SingleObjectTracker):
    def __init__(
        self,
        w_min: float,
        gating_size: float,
        meas_model: MeasurementModel,
        sensor_model: SensorModelConfig,
        motion_model: BaseMotionModel,
        *args,
        **kwargs,
    ):
        self.meas_model = meas_model
        self.sensor_model = sensor_model
        self.motion_model = motion_model
        self.w_min = w_min
        self.gating_size = gating_size
        self.initial_state = GaussianDensity()
        super().__init__()

    @property
    def name(self):
        return "PDA SOT"

    def step(self, measurements: np.ndarray):
        """Tracks a single object using probabilistic data association

        For each filter recursion iteration implemented next steps:
        0) prediction
        1) gating
        2) create missed detection hypothesis
        3) create object detection hypothesis for each detection inside the gate
        4) normalise hypothesis weights
        5) prune hypothesis with small weights, and then re-normalise the weights
        6) merge different hypotheses using  GaussianDensity moment matching
        7) extract object state estimate
        """

        prev_state = initial_state
        estimations = [None for x in range(len(measurements))]
        for timestep, measurements_in_scene in enumerate(measurements):
            estimations[timestep] = self.estimation_step(
                predicted_state=prev_state,
                current_measurements=np.array(measurements_in_scene),
            )
            prev_state = GaussianDensity.predict(state=estimations[timestep], motion_model=self.motion_model)
        return tuple(estimations)

    def estimation_step(self, predicted_state: GaussianDensity, current_measurements: np.ndarray):
        # 1. Gating
        (meas_in_gate, _) = GaussianDensity.ellipsoidal_gating(
            predicted_state,
            current_measurements,
            self.meas_model,
            self.gating_size,
        )
        if meas_in_gate.size == 0:  # number of hypothesis
            current_step_state = predicted_state

        else:
            # 2. Create miss detection hypothesis
            multi_hypotheses = [predicted_state]  # no detection hypothesis
            for z_ingate in meas_in_gate:
                multi_hypotheses.append(
                    GaussianDensity.update(
                        initial_state=predicted_state,
                        measurements=z_ingate,
                        model_measurement=self.meas_model,
                    )
                )

            predicted_likelihood = GaussianDensity.predicted_likelihood(
                state_pred=predicted_state,
                z=meas_in_gate,
                measurement_model=self.meas_model,
            )

            # Hypothesis evaluation
            # detection
            w_theta_factor = np.log(self.sensor_model.P_D / self.sensor_model.intensity_c)
            w_theta_k = predicted_likelihood + w_theta_factor
            # misdetection
            w_theta_0 = 1 - self.sensor_model.P_D

            hypotheses_weights_log = [np.log(w_theta_0), w_theta_k]
            log_w, log_sum_ = normalize_log_weights(hypotheses_weights_log)
            hypotheses_weight_log, multi_hypotheses = HypothesisReduction.prune(
                hypotheses_weights=log_w,
                multi_hypotheses=multi_hypotheses,
                threshold=self.w_min,
            )

            log_w, log_sum_ = normalize_log_weights(hypotheses_weights_log)
            # def moment_matching(weights: List[float], states: List[ GaussianDensity]) ->  GaussianDensity:
            current_step_state = GaussianDensity.moment_matching(weights=log_w, states=multi_hypotheses)

        estimation = current_step_state
        return estimation
