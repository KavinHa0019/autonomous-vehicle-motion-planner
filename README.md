# Safe Motion Planning for Complex Autonomous Driving Using Hybrid A* and Spline Functions
This repository contains the source code for two path planning algorithms, A* and hybrid A*. Both algorithms aim to identify the shortest path between two points on a grid that avoids collisions with any obstacles. The hybrid A* code includes a path smoother and velocity planner, which guarantee safety and enhance comfort for an autonomous vehicle.

### Requirements
The project was developed and tested in Python 3.10.7, which includes the following required modules in its standard library:
* heapq
* random
* time.time
* warnings.filterwarnings

The project also requires the following external modules, whose libaries they belong on need to be installed separately:
* numpy (NumPy v1.23.0)
* matplotlib.pyplot (Matplotlib v3.5.2)
* scipy.interpolate (SciPy v1.8.1)
* scipy.integrate.simpson (SciPy v1.8.1)
