# Safe Motion Planning for Complex Autonomous Driving Using A*, Hybrid A* and Spline Functions
This repository contains the source code for two path planning algorithms, (conventional) A* and hybrid A*. Both algorithms aim to identify the shortest path between two points on a grid that avoids collisions with any obstacles. The hybrid A* code includes a path smoother and velocity planner, which guarantee safety and enhance comfort for an autonomous vehicle.

### Project Requirements
The project was developed and tested in Python 3.10.7, and it requires the following built-in modules:
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

The program then initiates the pathfinding algorithm, which explores nodes by identifying the location of their neighbours and assigning an f-cost to each. Neighbours in conventional A* are defined as nodes that are orthogonally or diagonally adjacent to the node currently being explored. Hybrid A*, meanwhile, computes neighbour locations using a kinematic model, which takes into account the velocity and steering inputs of the vehicle required to get to them from the node it is currently on.

Explored nodes are compared using a g-cost function and heuristic. The g-cost function is defined based on several criteria, such as the distance from the start node and nearby obstacles, and the velocity and steering inputs required to reach each node (exclusive to hybrid A*). Meanwhile, the heuristic provides an estimate of the shortest distance required to travel from each node to arrive at the goal node. Each node is then given an f-cost, which is the sum of its g-cost and heuristic value.

f(n) = g(n) + h(n)
* f(n) = f-cost of a given node n
* g(n) = g-cost of node n
* h(n) = Heuristic value of node n

Exploration ends when the algortihm has reached the goal node; the program then traces the shortest path between the start and goal nodes. The algorithm determines the node with the next lowest f-cost and adds it to the shortest path. As a result, the increase in f-cost is minimised along the shortest path, and thus it is always optimal. Once the shortest path has been found, the program plots the path on a graph representing the search grid. Afterwards, the vehicle's heading at each node along the planned path are determined and plotted as a heading profile. The purpose of these heading profiles is to compare the smoothness, and thus comfort (and safety to a lesser extent), of paths generated by both conventional and hybrid A*.

The conventional A* program then terminates, while the hybrid A* program executes a path smoother that smoothens the generated path using cubic B-splines, providing better passenger comfort. The velocity planner then generates velocity, acceleration and jerk profiles indicating desired values of velocity, acceleration and jerk at every point along the smoothed path. These profiles guarantee passenger safety and may enhance passenger comfort, as long as it does not compromise safety in the process. The hybrid A* program finally terminates by plotting these profiles on three separate graphs.

### Instructions for Use
Both programs can be used immediately after installation, assuming the libraries mentioned above have also been installed along with Python 3.10.7. Modules have already been imported and the programs do not require any user intervention beyond running them in the terminal. However, the user may decide which road scenario to work with. By default, both programs simulate the highway scenario.

To switch to the intersection scenario:
* Comment out lines 194-204 and in lines 208-225 (A*)
* Comment out lines 405-417 and in lines 421-440 (Hybrid A*)

To revert back to the highway scenario:
* Comment out lines 208-225 and in lines 194-204 (A*)
* Comment out lines 421-440 and in lines 405-417 (Hybrid A*)

In any case, one and only one road scenario can be selected.

### Outputs
* Terminal (A*):
Displays coordinates of start and goal nodes, number of obstacles, scenario, motion planner status and execution time

* Terminal (Hybrid A*):
Displays the same information as the terminal in A*

* Shortest path plot (A*, highway scenario):
Graphical representation of the entire search grid, which contains kerbs, obstacles, the shortest path and every node along it (including the start and goal nodes)

* Shortest path plot (Hybrid A*, intersection scenario):
Same as the shortest path plot for A*, except it also contains the smoothed path

* Velocity profile plots:
Two profiles plotted on one graph: the maximum velocity profile which indicates the maximum allowable velocity the vehicle can travel at to guarantee safety, and the final/desired velocity profile which enhances passenger comfort

* Acceleration profile plots:
Three profiles plotted on one graph: the acceleration profile obtained from the maximum velocity profile, the same profile but clipped to fit acceleration constraints, and the final/desired acceleration profile which enhances passenger comfort 

* Jerk profile plots:
Three profiles plotted on one graph: the acceleration profile obtained from the maximum velocity profile, the same profile but clipped to fit jerk constraints, and the final/desired jerk profile which enhances passenger comfort 

### Credit
This project is based on a similar project, created by jvirdi2:
https://github.com/jvirdi2/A_star_and_Hybrid_A_star
