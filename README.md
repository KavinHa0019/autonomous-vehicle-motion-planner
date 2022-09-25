# Safe Motion Planning for Complex Autonomous Driving Using A*, Hybrid A* and Spline Functions
This repository contains the source code for two path planning algorithms, A* and hybrid A*. Both algorithms aim to identify the shortest path between two points on a grid that avoids collisions with any obstacles. The hybrid A* code includes a path smoother and velocity planner, which guarantee safety and enhance comfort for an autonomous vehicle.

### Project Requirements
The project was developed and tested in Python 3.10.7, which requires the following built-in modules:
* heapq
* random
* time.time
* warnings.filterwarnings

The project also requires the following external modules, whose parent libaries need to be installed separately:
* numpy (NumPy v1.23.0)
* matplotlib.pyplot (Matplotlib v3.5.2)
* scipy.interpolate (SciPy v1.8.1)
* scipy.integrate.simpson (SciPy v1.8.1)

### How the Project Works
Each program begins by generating the environment where the pathfinding problem takes place in, by defining its boundaries and adding road boundaries (also known as "kerbs") to simulate one of two road scenarios: a highway and an intersection. This environment takes the form of a 50x50 square grid, consisting of 2500 evenly sized square cells. Despite this, the algorithm is only permitted to explore a fraction of these cells, which are all located in between the kerbs. The program then spawns a random number of obstacles which the pathfinding algorithm is programmed to avoid. Appropriate start and goal positions are specified and the algorithm proceeds to explore the grid in order to reach the goal.

### Instructions for Use
Both programs can be used immediately after installation, assuming the libraries mentioned above have also been installed along with Python 3.10.7. However, the user may switch between road scenarios. By default, both programs simulate the highway scenario.

### Credit
This project was based on a similar project, created by jvirdi2:
https://github.com/jvirdi2/A_star_and_Hybrid_A_star
