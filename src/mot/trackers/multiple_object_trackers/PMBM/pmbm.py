import logging as lg

import numpy as np
import scipy, scipy.stats
from mot import (
    GaussianDensity,
    GaussianMixture,
    MeasurementModel,
    MotionModel,
    SensorModelConfig,
)

from mot.trackers.multiple_object_trackers.PMBM.common.birth_model import (
    BirthModel,
)

from .common import (
    GlobalHypothesis,
    MultiBernouilliMixture,
    PoissonRFS,
    Association,
    AssignmentSolver,
)


class PMBM:
    """
    1. Prediction of Bernoulli component.
    2. Misdetection update of Bernoulli component.
    3. Object detection update of Bernoulli component.
    4. Prediction of Poisson Point Process (PPP).
    5. Misdetection update of PPP.
    6. Object detection update of PPP.
    7. PMBM prediction.
    8. PMBM update.
    9. Object states extraction.
    """

    def __init__(
        self,
        meas_model: MeasurementModel,
        sensor_model: SensorModelConfig,
        motion_model: MotionModel,
        birth_model: GaussianMixture,
        max_number_of_hypotheses: int,
        gating_percentage : float,
        w_min,
        detection_probability,
        survival_probability: float,
        density: GaussianDensity = GaussianDensity,
        *args,
        **kwargs,
    ):
        self.timestep = 0
        self.density = density
        self.meas_model = meas_model
        self.sensor_model = sensor_model
        self.motion_model = motion_model
        self.birth_model = birth_model
        self.survival_probability = survival_probability  # models death of an object (aka P_S)

        # Interval for ellipsoidal gating (aka P_G)
        self.gating_percentage = gating_percentage
        self.gating_size = scipy.stats.chi2.ppf(self.gating_percentage,
                                                df=self.meas_model.d)
        self.w_min = w_min  # in log domain
        self.detection_probability = detection_probability  # models detection (and missdetection) of an object (aka P_)
        self.merging_threshold = merging_threshold
        self.desired_num_global_hypotheses = 10
        self.max_number_of_hypotheses = max_number_of_hypotheses

        self.PPP = PoissonRFS(initial_intensity=copy.deepcopy(birth_model))
        self.MBM = MultiBernouilliMixture()

    def __repr__(self) -> str:
        return self.__class__.__name__ + (
            f"(current_timestep={self.timestep}, "
            f"MBM components ={len(self.MBM.tracks)}, "
            f"Global hypotheses={len(self.MBM.global_hypotheses)}, "
            f"PPP components={len(self.PPP.intensity)}, ")

    def increment_timestep(self):
        self.timestep += 1

    def predict(
        self,
        birth_model: GaussianMixture,
        motion_model: MotionModel,
        survival_probability: float,
        dt: float,
        density: GaussianDensity,
    ) -> None:
        """Performs PMBM preidction step"""

        # MBM predict
        self.MBM.predict(motion_model, survival_probability, density, dt)
        # PPP predict
        self.PPP.predict(motion_model, copy.deepcopy(birth_model), survival_probability, dt)

    def update(self, measurements: np.ndarray) -> None:
        if len(measurements) == 0:
            lg.debug(f"\n no measurements!")
            return
        lg.debug(f"\n===current timestep: {self.timestep}===")
        lg.debug(f"\n   global hypotheses {self.MBM.global_hypotheses}")
        lg.debug(f"\n   MBM tracks {self.MBM.tracks} \n")

        gating_matrix_undetected, used_meas_undetected = self.PPP.gating(
            measurements, self.density, self.meas_model, self.gating_size
        )

        gating_matrix_detected, used_meas_detected = self.MBM.gating(
            measurements, self.density, self.meas_model, self.gating_size
        )

        self.MBM.update(
            self.detection_probability,
            measurements,
            gating_matrix_detected,
            self.meas_model,
            self.density,
        )

        new_tracks = self.PPP.get_targets_detected_for_first_time(
            measurements,
            gating_matrix_undetected,
            self.sensor_model.intensity_c,
            self.meas_model,
            self.detection_probability,
        )

        # Update of PPP intensity for undetected objects that remain undetected
        self.PPP.undetected_update(self.detection_probability)

        if not self.MBM.global_hypotheses:
            self.MBM.tracks.update(new_tracks)

            hypo_list = []
            for track_id, track in self.MBM.tracks.items():
                for sth_id, sth in track.single_target_hypotheses.items():
                    hypo_list.append(Association(track_id, sth_id))

            self.MBM.global_hypotheses.append(
                GlobalHypothesis(log_weight=0.0, associations=hypo_list)
            )

        else:

            new_global_hypotheses = []
            for global_hypothesis in self.MBM.global_hypotheses:
                assigner = AssignmentSolver(
                    global_hypothesis=global_hypothesis,
                    old_tracks=self.MBM.tracks,
                    new_tracks=new_tracks,
                    measurements=measurements,
                    num_of_desired_hypotheses=self.max_number_of_hypotheses,
                )
                next_global_hypothesis = assigner.solve()
                new_global_hypotheses.extend(next_global_hypothesis)

            self.update_tree()
            self.MBM.tracks.update(new_tracks)

            self.MBM.global_hypotheses = new_global_hypotheses
            self.MBM.normalize_global_hypotheses_weights()

            for new_track in new_tracks:
                self.MBM.add_track(new_track)

        # Update of PPP intensity for undetected objects that remain undetected
        self.PPP.undetected_update(self.detection_probability)
        
    def rebuild_tree(self):
        """1. Move children to upper lever."""
        for track in self.MBM.tracks.values():
            track.cut_tree()

    def estimator(self):
        estimates = self.MBM.estimator()
        logging.debug(f'estimates = {estimates}')
        return estimates

    def reduction(self) -> None:
        self.PPP.prune(threshold=-10)
        # self.MBM.prune_global_hypotheses(threshold=np.log(0.001))
        self.MBM.cap_global_hypothesis(self.max_number_of_hypotheses)
        # self.MBM.remove_unused_bernoullies()

    def estimation_step(self, current_measurements: np.ndarray):
        # PMBM prediction
        self.PMBM_predict()

        # PMBM update
        self.update(current_measurements)

        # Extract state estimates from the PPP
        estimates = self.PMBM_estimator()

        return estimates
