import logging

import numpy as np

from src.common.gaussian_density import GaussianDensity
from src.configs import SensorModelConfig
from src.measurement_models import MeasurementModel
from src.motion_models import BaseMotionModel

from .base_single_object_tracker import SingleObjectTracker


class NearestNeighbourTracker(SingleObjectTracker):
    def __init__(
        self,
        gating_size: float,
        meas_model: MeasurementModel,
        sensor_model: SensorModelConfig,
        motion_model: BaseMotionModel,
        initial_state: GaussianDensity,
    ) -> None:
        self.meas_model = meas_model
        self.sensor_model = sensor_model
        self.motion_model = motion_model
        self.gating_size = gating_size
        self.state: GaussianDensity = initial_state
        self.timestep = 0
        super().__init__()

    @property
    def name(self):
        return "Nearest Neighbout SOT"

    def predict(self):
        self.state = GaussianDensity.predict(self.state, self.motion_model, dt=1.0)

    def estimate(self):
        return {i: self.state[i] for i in range(len(self.state))}

    def step(self, measurements):
        """Tracks a single object using nearest neighbour association

        For each filter recursion iteration implemented next steps:
        0) prediction
        1) gating
        2) calculates the predicted likelihood for each measurement in the gate
        3) find the nearest neighbour measurement
        4) compares the weight of the missed detection hypotheses and
           the weight of the object detection hypothesis created using
           the nearest neigbour measurement
        5) if the object detection hypothesis using the nearest neighbour
           measurement has the hightes weight, perform Kalman update
        6) extract object state estimate

        """
        self.timestep += 1
        self.predict()
        logging.error(self.state.means[..., :2])
        self.update(measurements)
        return self.estimate()

    def update(self, measurements: np.ndarray):
        # 1. Gating
        (meas_in_gate, _) = GaussianDensity.ellipsoidal_gating(
            self.state,
            measurements,
            self.meas_model,
            self.gating_size,
        )

        if meas_in_gate.sum() == 0:  # number of possible hypothesis
            return  # no detection hypothesis

        # 2. Calculate the predicted likelihood for each measurement in the gate
        predicted_likelihood = GaussianDensity.predict_loglikelihood(
            state_pred=self.state,
            measurements=measurements,
            measurement_model=self.meas_model,
        )

        # misdetection
        weight_missdetection = 1 - self.sensor_model.P_D

        # detection
        w_theta_factor = np.log(self.sensor_model.P_D / self.sensor_model.intensity_c)
        weight_detections = predicted_likelihood + w_theta_factor

        # 3. Compare the weight of the missed detection
        # hypothesis and the weight of the object detection hypothesis
        # using the nearest neighbour measurement
        max_weight_idx = np.argmax(weight_detections, axis=1)
        max_weight = weight_detections[0, max_weight_idx]

        if weight_missdetection > max_weight:
            return  # nothing update because missdetection hypothesis has the highest weight

        # Update state with nearest neighbour measurement
        next_means, next_covs, _ = GaussianDensity.update(self.state, measurements[max_weight_idx], self.meas_model)
        self.state = GaussianDensity(means=next_means[0], covs=next_covs[0])
