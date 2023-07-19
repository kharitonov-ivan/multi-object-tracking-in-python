from typing import Optional, Tuple

import hydra
from mot.configs import GroundTruthConfig
from mot.simulator import MeasurementsGenerator, ObjectData
from omegaconf import DictConfig
from src.common.gaussian_density import GaussianDensity as GaussianDensity


def train(cfg: DictConfig) -> Tuple[dict, dict]:
    # set seed for random number generators in pytorch, numpy and python.random

    scenario_motion_model = hydra.utils.instantiate(cfg.scenario_motion_model)
    scenario_sensor_model = hydra.utils.instantiate(cfg.scenario_sensor_model)
    scenario_measurement_model = hydra.utils.instantiate(cfg.scenario_measurement_model)
    ground_truth_config = hydra.utils.instantiate(cfg.ground_truth_config)

    object_data = ObjectData(
        ground_truth_config=ground_truth_config,
        motion_model=scenario_motion_model,
        if_noisy=False,
    )
    meas_gen = MeasurementsGenerator(
        object_data=object_data, sensor_model=sensor_model, meas_model=meas_model
    )
    GroundTruthConfig(object_motion_fixture, total_time=simulation_steps)

    tracker = hydra.utils.instantiate(cfg.tracker)
    tracker_estimations = []
    for timestep, measurements, sources in meas_data:
        estimations = tracker.step(measurements)
        tracker_estimations.append(estimations)
    import pdb

    pdb.set_trace()
    # log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    # datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # train_metrics = trainer.callback_metrics

    return None


@hydra.main(version_base="1.3", config_path="./configs", config_name="default.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    metric_dict, _ = train(cfg)

    return None


if __name__ == "__main__":
    main()
