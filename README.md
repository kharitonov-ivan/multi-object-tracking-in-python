
# Multi-Object-Tracking-in-python

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

This repository contains implementation of various multi-object trackers. 

In addition, it includes is tutorial with goal to demonstrate principles of work this trackers in educational proposes. It is unfluenced by the Multiple Object Tracking course in Chalmers and Edx [youtube 📺](https://www.youtube.com/channel/UCa2-fpj6AV8T6JK1uTRuFpw/featured) and MATLAB open sourced implementations.

You can dive into in tutorial notebook (locally or in colab). Or explore code explicitly.
<a target="_blank" href="https://colab.research.google.com/github/kharitonov-ivan/multi-object-tracking-in-python/blob/main/tutorial-mot.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Using in virtualenv

Firstly, you need install Eigen3 in your system.

```bash
python3.10 -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
python -m pip install -e .
```

# Development

As a dependencies package manager this project use PDM.

Install pre-commit

```bash
pre-commit install
```

# Single Object tracking

`pytest tests/SOT/`

![](https://raw.githubusercontent.com/kharitonov-ivan/Multi-Object-Tracking-in-python/main/images/scenario-gifs/SOT_PDA.gif)

# Tracking with fixed number of objects

`pytest tests/MOT`

![](https://raw.githubusercontent.com/kharitonov-ivan/Multi-Object-Tracking-in-python/main/images/scenario-gifs/NMOT-GNN.gif)


# Multi object tracking with Probability Hypothesis Density filter

`pytest tests/PHD`

# Multi object tracking with Poisson Multi Bernoulli Mixture filter

`pytest tests/PMBM`

![](https://raw.githubusercontent.com/kharitonov-ivan/Multi-Object-Tracking-in-python/main/images/scenario-gifs/MOT-PMBM.gif)

![](https://raw.githubusercontent.com/kharitonov-ivan/Multi-Object-Tracking-in-python/main/tests/PMBM/.images/PMBM-many_objects_linear_motion_delayed-P_S%3D0.99-P_D%3D0.9-lambda_c%3D10.0.png)
