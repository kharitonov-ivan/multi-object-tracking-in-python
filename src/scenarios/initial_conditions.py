from dataclasses import dataclass

import numpy as np

from src.common.state import Gaussian
from src.configs import Object
from src.motion_models import (
    ConstantVelocityMotionModel,
    CoordinateTurnMotionModel,
    MotionModel,
)


BIRTH_TIME = 10
DEATH_TIME = 80
default_nonlinear_P = np.diag(np.power([1e2, 1e2, 1e1, 1.0 * np.pi / 180.0, 1 * np.pi / 180], 2))  # # non-linear state vector (X-pos, Y-pos, velocity, heading, turn-rate)


motion_scenarios = {
    ConstantVelocityMotionModel: {
        "static": [
            [Object(initial=Gaussian(x=np.array([0.0, 0.0, 10.0, 10.0]), P=np.eye(4)), t_birth=0, t_death=100)],
            [
                Object(initial=Gaussian(x=np.array([-100.0, 100.0, 0.0, 0.0]), P=np.eye(4)), t_birth=BIRTH_TIME, t_death=DEATH_TIME),
                Object(initial=Gaussian(x=np.array([100.0, 100.0, 0.0, 0.0]), P=np.eye(4)), t_birth=BIRTH_TIME, t_death=DEATH_TIME),
            ],
            [
                Object(initial=Gaussian(x=np.array([-250.0, 250.0, 0.0, 0.0]), P=np.eye(4)), t_birth=BIRTH_TIME, t_death=DEATH_TIME),
                Object(initial=Gaussian(x=np.array([250.0, 250.0, 0.0, 0.0]), P=np.eye(4)), t_birth=BIRTH_TIME, t_death=DEATH_TIME),
                Object(initial=Gaussian(x=np.array([0.0, -250.0, 0.0, 0.0]), P=np.eye(4)), t_birth=BIRTH_TIME, t_death=DEATH_TIME),
            ],
        ],
        "dynamic": [
            [Object(initial=Gaussian(x=np.array([0.0, 0.0, 10.0, 10.0]), P=np.eye(4)), t_birth=BIRTH_TIME, t_death=DEATH_TIME)],
            [
                Object(initial=Gaussian(x=np.array([-100.0, 10.0, 10.0, 10.0]), P=np.eye(4)), t_birth=BIRTH_TIME, t_death=DEATH_TIME),
                Object(initial=Gaussian(x=np.array([10.0, -10.0, 0.0, -10.0]), P=np.eye(4)), t_birth=BIRTH_TIME, t_death=DEATH_TIME),
            ],
            [
                Object(initial=Gaussian(x=np.array([10.0, 10.0, 0.0, 10.0]), P=np.eye(4)), t_birth=5, t_death=69),
                Object(initial=Gaussian(x=np.array([10.0, -10.0, 0.0, -10.0]), P=np.eye(4)), t_birth=0, t_death=69),
            ],
        ],
        "many": [
            [
                Object(initial=Gaussian(x=np.array([0.0, 0.0, 0.0, -10.0]), P=np.eye(4)), t_birth=0, t_death=69),
                Object(initial=Gaussian(x=np.array([400.0, -600.0, -10.0, 5.0]), P=np.eye(4)), t_birth=0, t_death=99),
                Object(initial=Gaussian(x=np.array([-800.0, -200.0, 20.0, -5.0]), P=np.eye(4)), t_birth=0, t_death=69),
                Object(initial=Gaussian(x=np.array([400.0, -600.0, -7.0, -4.0]), P=np.eye(4)), t_birth=19, t_death=99),
                Object(initial=Gaussian(x=np.array([400.0, -600.0, -2.5, 10.0]), P=np.eye(4)), t_birth=19, t_death=99),
                Object(initial=Gaussian(x=np.array([0.0, 0.0, 7.5, -5.0]), P=np.eye(4)), t_birth=19, t_death=99),
                Object(initial=Gaussian(x=np.array([-800.0, -200.0, 12.0, 7.0]), P=np.eye(4)), t_birth=39, t_death=99),
                Object(initial=Gaussian(x=np.array([-200.0, 800.0, 15.0, -10.0]), P=np.eye(4)), t_birth=39, t_death=99),
                Object(initial=Gaussian(x=np.array([-800.0, -200.0, 3.0, 15.0]), P=np.eye(4)), t_birth=59, t_death=99),
                Object(initial=Gaussian(x=np.array([-200.0, 800.0, -3.0, -15.0]), P=np.eye(4)), t_birth=59, t_death=99),
                Object(initial=Gaussian(x=np.array([0.0, 0.0, -20.0, -15.0]), P=np.eye(4)), t_birth=79, t_death=99),
                Object(initial=Gaussian(x=np.array([-200.0, 800.0, 15.0, -5.0]), P=np.eye(4)), t_birth=79, t_death=99),
            ]
        ],
    },
    CoordinateTurnMotionModel: {
        "static": [[Object(initial=Gaussian(x=np.array([0.0, 0.0, 10.0, 0.0, np.pi / 180.0]), P=default_nonlinear_P), t_birth=0, t_death=400)]],
        "dynamic": [
            [
                Object(initial=Gaussian(x=np.array([0.0, 0.0, 5.0, 0.0, np.pi / 180]), P=default_nonlinear_P), t_birth=0, t_death=99),
                Object(initial=Gaussian(x=np.array([20.0, -20.0, -20.0, 0.0, np.pi / 90]), P=default_nonlinear_P), t_birth=0, t_death=99),
                Object(initial=Gaussian(x=np.array([-20.0, 10.0, -10.0, 0.0, np.pi / 360]), P=default_nonlinear_P), t_birth=0, t_death=99),
                Object(initial=Gaussian(x=np.array([-10.0, 10.0, 8.0, 0.0, np.pi / 270]), P=default_nonlinear_P), t_birth=0, t_death=99),
            ]
        ],
    },
}


@dataclass
class Scenario:
    motion_model: MotionModel
    movement_type: str
    object_configs: str


all_object_scenarios = []
for motion_model in motion_scenarios:
    for movement_type in motion_scenarios[motion_model]:
        for scenario in motion_scenarios[motion_model][movement_type]:
            all_object_scenarios.append(Scenario(motion_model, movement_type, scenario))
