import numpy as np


class MeasurementsGenerator:
    """Generates object_generated measurements and clutter"""

    def __init__(
        self,
        object_data,
        sensor_model,
        meas_model,
        random_state=None,
    ):
        """Generates object generated measurement and clutter

        Args:
            object_data (list of lists of Gaussians): a structure contains object data
            sensor_model (SensorModel): a structure specifies sensor model parameters
            meas_model (MeasurementModel): a structure specifies the measurement model
                                           parameters

        Attributes:


        Returns:
            meas_data ([]): cell array of size (total_tracking_time, 1)
                            each cell stores measurements of size
                            (measurement dimension) x (number of measurements
                            at corresponding timestep)
        """
        self._generator = np.random.RandomState(random_state)
        self.object_data = object_data
        self.sensor_model = sensor_model
        self.meas_model = meas_model
        self.timestep = 0

    def generate(self):
        """Generates measurements and clutter

        Returns:
            measurements (np.ndarray): (measurement dimension) x (number of measurements)
            sources (list of int): source of each measurement
        """
        measurements = []
        sources = []
        observations = self.generate_observations()
        if observations:
            object_observations, object_sources = observations
            measurements.append(object_observations)
            sources.append(object_sources)
        clutter = self.generate_clutter()
        if clutter:
            clutter_observations, clutter_sources = clutter
            measurements.append(clutter_observations)
            sources.append(clutter_sources)
        return (
            self.timestep,
            np.concatenate(measurements, axis=0),
            np.concatenate(sources, axis=0),
        )

    def generate_observations(self):
        objects_in_scene = self.object_data[self.timestep]
        if len(objects_in_scene) > 0:
            # Generate misses
            detection_mask = self._generator.uniform(size=len(objects_in_scene)) < self.sensor_model.P_D
            observed_objects = [(obj_id, obj) for is_observed, (obj_id, obj) in zip(detection_mask, self.object_data[self.timestep].items()) if is_observed]

            # Generate measurement
            if not observed_objects:
                return
            gt_state_means = np.array([obj.means[0] for _, obj in observed_objects])
            object_observations = self.meas_model.observe(gt_state_means)
            sources = [obj_id for obj_id, _ in observed_objects]
            return object_observations, sources

    def generate_clutter(self):
        # Number of clutter measurements
        N_c = self._generator.poisson(lam=self.sensor_model.lambda_c)
        # Generate clutter
        # TODO keep in mind that pos of object can be vary
        clutter_min_coord, clutter_max_coord = np.diag(self.sensor_model.range_c)
        clutter_observations = np.random.uniform(clutter_min_coord, clutter_max_coord, [N_c, self.meas_model.dim])
        clutter_sources = [-1] * N_c
        return clutter_observations, clutter_sources

    def __iter__(self):
        return self

    def __next__(self):
        if self.timestep >= len(self.object_data):
            raise StopIteration
        else:
            result = self.generate()
            self.timestep += 1
            return result
