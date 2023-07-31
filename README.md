
# Multi-Object-Tracking-in-python

Python implementation of this course [youtube 📺](https://www.youtube.com/channel/UCa2-fpj6AV8T6JK1uTRuFpw/featured):


# Installation in virtualenv

```
pip install .
```

As a depencies package manager this project use PDM.

# Single Object tracking

`pytest tests/SOT/`

![](https://raw.githubusercontent.com/kharitonov-ivan/Multi-Object-Tracking-in-python/main/tests/SOT/.images/NearestNeighbourTracker-SOT-linear-case-(CV).gif)

# Tracking with fixed number of objects

`pytest tests/MOT`

![](https://github.com/kharitonov-ivan/Multi-Object-Tracking-in-python/blob/8298718bc4fe7b1abf76a9974bf4147084f804cb/tests/MOT/.images/GlobalNearestNeighboursTracker-n%20MOT%20linear%20(CV).gif?raw=true)


# Multi object tracking with Probability Hypothesis Density filter

`pytest tests/PHD`

![](https://github.com/kharitonov-ivan/Multi-Object-Tracking-in-python/blob/main/tests/PHD/.images/GMPHD-n_MOT_linear_CV.gif?raw=true)

# Multi object tracking with Poisson Multi Bernoulli Mixture filter

`pytest tests/PMBM`

![](https://github.com/kharitonov-ivan/Multi-Object-Tracking-in-python/blob/main/tests/PMBM/.images/PMBM-many_objects_linear_motion_delayed-P_S=0.99-P_D=0.9-lambda_c=10.0.gif?raw=true)

![](https://raw.githubusercontent.com/kharitonov-ivan/Multi-Object-Tracking-in-python/main/tests/PMBM/.images/PMBM-many_objects_linear_motion_delayed-P_S%3D0.99-P_D%3D0.9-lambda_c%3D10.0.png)
