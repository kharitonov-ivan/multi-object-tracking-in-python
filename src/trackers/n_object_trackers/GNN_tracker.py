import copy

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
from tqdm import tqdm as tqdm

from src.common.gaussian_density import GaussianDensity
from src.common.state import Gaussian
from src.configs import SensorModelConfig
from src.measurement_models import MeasurementModel
from src.motion_models import MotionModel

from .base_n_object_tracker import KnownObjectTracker


class GlobalNearestNeighboursTracker(KnownObjectTracker):
    def __init__(
        self,
        meas_model: MeasurementModel,
        sensor_model: SensorModelConfig,
        motion_model: MotionModel,
        M,
        merging_threshold,
        P_G,
        w_min,
        n,
        initial_state: Gaussian,
        *args,
        **kwargs,
    ) -> None:
        self.meas_model = meas_model
        self.sensor_model = sensor_model
        self.motion_model = motion_model
        self.w_min = w_min
        self.P_G = P_G
        self.gating_size = chi2.ppf(P_G, df=self.meas_model.dim)
        self.M = M
        self.n = n
        self.merging_threshold = merging_threshold
        self.hypotheses_weight = [np.log(1.0)]
        self.multi_hypotheses_bank = initial_state
        self.timestep = 0.0
        super().__init__()

    @property
    def method(self) -> str:
        return "GNN"

    def step(self, measurements, dt):
        self.timestep += 1
        self.predict(dt)
        if measurements.size > 0:
            self.update(measurements)
        return self.estimate()

    def predict(self, dt):
        self.multi_hypotheses_bank = [GaussianDensity.predict(state, self.motion_model, dt=1.0) for state in self.multi_hypotheses_bank]

    def estimate(self):
        """
        For each filter recursion iteration implemented next steps:
        1) for each prior perform prediction
        2) for each hypothesis, perform ellipsoidal gating
           and only create object detection hypotheses for detections
        3) construct 2D cost matrix of size
           (number of objects x number of z_ingate + number of objects)
        4) find best assignment using a 2D assignment solver
        5) create new local hypotheses accotding to the best assgnment matrix obtained
        6) get obhect state estimates
        7) preform prefict for each local hypotheses
        """
        return {idx: self.multi_hypotheses_bank[idx] for idx in range(len(self.multi_hypotheses_bank))}

    def update(self, current_measurements: np.ndarray):
        # 1) elipsoidal gating separately for each object
        num_of_measurements = current_measurements.shape[0]
        indices_of_objects_in_gate = np.zeros((self.n, num_of_measurements), dtype=bool)

        for object_state_idx, object_state in enumerate(self.multi_hypotheses_bank):
            z_ingate, indices_in_gate = GaussianDensity.ellipsoidal_gating(
                object_state,
                current_measurements,
                measurement_model=self.meas_model,
                gating_size=self.gating_size,
            )
            indices_of_objects_in_gate[object_state_idx, :] = indices_in_gate

        # Disconsider measurements which do not fall inside any object gate
        mask_to_keep = np.sum(indices_of_objects_in_gate, axis=0, dtype=bool)
        source_to_considered_ids = np.where(mask_to_keep)[0]  # TODO fix it
        considered_to_source_ids = np.sort(source_to_considered_ids)
        filtered_measurements = current_measurements[source_to_considered_ids]
        num_of_filtered_measurements = filtered_measurements.shape[0]

        #   3) construct 2D cost matrix of size
        #    (number of objects x number of z_ingate + number of objects)
        cost_matrix = np.full((self.n, num_of_filtered_measurements + self.n), np.inf)
        w_theta_factor = np.log(self.sensor_model.P_D / self.sensor_model.intensity_c)
        w_theta_0 = np.log(1 - self.sensor_model.P_D)  # misdetection
        for idx_object in range(self.n):
            for enum_meas, idx_meas in enumerate(source_to_considered_ids):
                S_i_h = (
                    self.meas_model.H(self.multi_hypotheses_bank[idx_object])
                    @ self.multi_hypotheses_bank[idx_object].P
                    @ self.meas_model.H(self.multi_hypotheses_bank[idx_object]).T
                )
                z_bar_i_h = self.meas_model.h(self.multi_hypotheses_bank[idx_object].x)
                vec_diff = current_measurements[idx_meas] - z_bar_i_h
                mahl = 0.5 * vec_diff @ np.linalg.inv(S_i_h) @ vec_diff.T
                factor = 0.5 * np.log(np.linalg.det(2 * np.pi * S_i_h))
                cost = mahl + factor - w_theta_factor

                cost_matrix[idx_object, enum_meas] = cost
            cost_matrix[idx_object, num_of_filtered_measurements + idx_object] = w_theta_0

        # 4) find best assignment using a 2D assignment solver
        #  4) find best assignment using a 2D assignment solver
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 5) create new local hypotheses accotding to the best assgnment matrix obtained
        for object_idx in range(self.n):
            if col_ind[object_idx] >= num_of_filtered_measurements:
                # Misdetection
                pass
            else:
                # Detection
                measurement_idx = considered_to_source_ids[col_ind[object_idx]]
                state = GaussianDensity.update(
                    self.multi_hypotheses_bank[object_idx],
                    current_measurements[measurement_idx],
                    self.meas_model,
                )

                self.multi_hypotheses_bank[object_idx] = copy.copy(state)
