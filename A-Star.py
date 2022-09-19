# Source code written and finalised by Kavin Ha
# Source code based off of jvirdi2's hybrid A* implementation: https://github.com/jvirdi2/A_star_and_Hybrid_A_star

import heapq as hq
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as itp
from scipy.integrate import simpson
import random
from time import time
from warnings import filterwarnings


class AStar:
    """AStar() class defines search grid boundaries, kerbs and obstacles"""
    def __init__(self, min_x, max_x, min_y, max_y, kerbs, obstacles, spacing):
        self.min_x = min_x              # Minimum x-value in grid
        self.max_x = max_x              # Maximum x-value in grid
        self.min_y = min_y              # Minimum y-value in grid
        self.max_y = max_y              # Maximum y-value in grid
        self.kerbs = kerbs              # Kerbs - boundaries used to close off parts of grid to form a road
        self.obstacles = obstacles      # Obstacles - must be avoided at all costs
        self.spacing = spacing          # Spacing - determines number of lanes in road

    def find_path(self, start, end):
        """Pathfinding algorithm - uses A* to find shortest path from start to end"""
        # Assign class attributes to new variables - reduces execution time by avoiding the need to search through class
        x_min = self.min_x
        x_max = self.max_x
        y_min = self.min_y
        y_max = self.max_y
        kerbs = self.kerbs
        obstacles = self.obstacles

        # Discrete coordinates (_d) - located at centre of cell
        # Continuous coordinates (_c) - located anywhere within cell
        start = (cell(start[0]), cell(start[1]))        # Discrete coordinates of start node
        end = (cell(end[0]), cell(end[1]))              # Discrete coordinates of goal node

        open_heap = []          # Priority queue ranks nodes based on f        [(f, node_d), ...]
        open_dict = {}          # Contain all nodes about to be explored       {node_d: (f, parent_d), ...}
        closed_dict = {}        # Contain all nodes already explored           {node_d: (f, parent_d), ...}
        g = 0                                   # g(start) = 0
        f = g + euclidean(start, end)           # Heuristic h(n) = euclidean(n, end)

        hq.heappush(open_heap, (f, start))      # Begin pathfinding by exploring start node first
        open_dict[start] = (f, start)           # Include start node in open dictionary - will explore start node first

        def obs_cost(n):
            """Computes obstacle cost based on distances between neighbours and all obstacles in grid"""
            spacing = self.spacing  # Number of cells above max_x/2
            lanes = (2 * spacing - 1) / cw  # Number of lanes (i.e. cells between parallel kerbs)

            # Obstacle cost formula
            g_obs_i = lanes * sum(list(map((lambda m: 1 / euclidean(n, m) ** 2), obstacles)))
            return g_obs_i

        # While priority queue is not empty (i.e. there are nodes yet to be explored)
        while len(open_heap) > 0:

            # open_heap[0] = Node in priority queue with lowest f-cost
            # Make node with lowest f-cost the current node
            current_d = open_heap[0][1]         # Discrete coordinates of current node
            f_current = open_heap[0][0]         # f-cost of current node

            closed_dict[current_d] = open_dict[current_d]      # Current node will already be explored after one loop

            # If goal node is one of current node's neighbours (i.e. within one cell width away from current node)
            if euclidean(current_d, end) < cw:
                print("\nGoal node found - tracing shortest path...")
                rev_final_path = [end]          # Trace shortest path beginning with goal node
                node = current_d                # Include current node in shortest path following goal node

                while node != start:
                    open_node = closed_dict[node]       # Most recent node along shortest path list
                    parent = open_node[1]               # Identify parent node of "node"
                    rev_final_path.append(parent)       # Add parent to shortest path
                    node = open_node[1]                 # Now parent is most recent node - make parent = "node"

                return rev_final_path[::-1]             # Reverse shortest path list - make start node first element

            hq.heappop(open_heap)                              # Remove current node from priority node after exploring

            # Nodes adjacent to current node (orthogonally or diagonally) are neighbours
            neighbours = [(current_d[0], current_d[1] + cw),            # Up
                          (current_d[0] + cw, current_d[1] + cw),       # Up right
                          (current_d[0] + cw, current_d[1]),            # Right
                          (current_d[0] + cw, current_d[1] - cw),       # Down right
                          (current_d[0], current_d[1] - cw),            # Down
                          (current_d[0] - cw, current_d[1] - cw),       # Down left
                          (current_d[0] - cw, current_d[1]),            # Left
                          (current_d[0] - cw, current_d[1] + cw)]       # Up left

            # Neighbours must be in grid and not occupied by kerb nor obstacle (such nodes known as "free neighbours")
            # List comprehension filters out neighbours that do not satisfy above criteria
            neighbours_free = list(filter((lambda n: x_min <= n[0] <= x_max and y_min <= n[1] <= y_max and
                                          n not in (kerbs or obstacles)), [m for m in neighbours]))

            # If current node has no neighbours, skip it and move to next node in priority queue
            if len(neighbours) == 0:
                pass

            # Otherwise, compute neighbouring nodes and their corresponding costs
            else:
                g_current = list(map((lambda n: f_current - euclidean(current_d, end)), neighbours_free))
                g_to_neighbour = list(map((lambda n: euclidean(current_d, n)), neighbours_free))
                g_obs = list(map(obs_cost, neighbours_free))
                # g_current = g-cost to get from start to current node
                # g_to_neighbour = g-cost to get from current node to neighbour
                # g_obs = Obstacle cost term

                g = np.array(g_current) + np.array(g_to_neighbour) + np.array(g_obs)        # Total g-cost
                h = list(map((lambda n: euclidean(n, end)), neighbours_free))               # Heuristic value
                f = g + np.array(h)                                                         # Total f-cost

                # For all free neighbouring nodes (i.e. nodes that vehicle can occupy), determine what to do with...
                # ...f-cost of each neighbour
                # Compares most recent f-cost with f-cost in open dictionary or both dictionaries
                # Algorithm always assigns lowest f-cost to every node

                for i in range(len(neighbours_free)):
                    # If neighbour already exists in open dictionary and has lower f-cost, ignore most recent f-cost
                    if neighbours_free[i] in open_dict and f[i] >= open_dict[neighbours_free[i]][0]:
                        pass

                    # If neighbour already exists in both dictionaries and has lower f-cost, ignore most recent f-cost
                    elif neighbours_free[i] in (open_dict and closed_dict) and f[i] >= \
                            closed_dict[neighbours_free[i]][0]:
                        pass

                    # Otherwise, consider f-cost and discrete coordinates of neighbour
                    else:
                        hq.heappush(open_heap, (f[i], neighbours_free[i]))      # Add neighbour to priority queue
                        open_dict[neighbours_free[i]] = (f[i], current_d)       # Neighbour about to be explored

        # End pathfinding algorithm if it is impossible to arrive at goal node - return empty list
        # Empty list will cause program to run into error and terminate as algorithm will have nothing to work with
        print("\nGoal node cannot be found.")
        return []


def cell(q):
    """Converts continuous coordinates into discrete coordinates by identifying the cell the coordinates lie in and
    outputting the centre coordinates of that cell"""
    if abs(q - int(q)) > cw/2:          # If continuous coordinates are to the right of cell centre
        return round(q) - cw/2          # Round up then subtract by half cell width - shift to the left
    elif abs(q - int(q)) < cw/2:        # If continuous coordinates are to the left of cell centre
        return round(q) + cw/2          # Round down then add by half cell width - shift to the right
    elif abs(q - int(q)) == cw/2:       # If continuous coordinates lie exactly at cell centre
        return q                        # Keep coordinates the same


def euclidean(position, target):
    """Mainly used as heuristic but can be used to find Euclidean distance between any two cells; in the case that
    "target" is the goal node, this function computes the heuristic value of the "position" node"""
    dist = np.sqrt((position[0] - target[0])**2 + (position[1] - target[1])**2)
    return dist


cw = 1      # Cell width


def main():
    """The function used to execute the entire program, including the definition of the search area, the execution of
    the pathfinding algorithm and the generation of graphs describing the shortest path"""

    print("\nStarting A* path planner...")
    # Runtime warnings may arise if division by zero may be performed - no errors encountered so far regarding this...
    # ...issue so can safely ignore these messages
    filterwarnings("ignore", category=RuntimeWarning)

    min_x, max_x, min_y, max_y = 0, 50, 0, 50    # Grid boundaries - used to determine whether a cell lies out of bounds

    def generate_kerbs(scenario, kerb):
        """Generates kerbs to roughly simulate different road scenarios; so far only two scenarios are available:
        highway and intersection"""

        # 0 = Highway, 1 = Intersection
        if scenario == 0:
            return np.random.uniform(max_y / 2 - spacing, max_y / 2 + spacing)
        elif scenario == 1:
            if kerb < max_x / 2 - spacing or kerb > max_x / 2 + spacing:
                return np.random.uniform(max_y / 2 - spacing, max_y / 2 + spacing)
            elif max_x / 2 - spacing <= kerb <= max_x / 2 + spacing:
                return np.random.uniform(min_y, max_y)

    spacing = 3

    """Kerbs for highway and intersection scenarios; comment lines 190-199 for intersection scenario, or comment lines
    203-219 for highway scenario; one of the two must be commented out at all times"""

    # HIGHWAY ----------------------------------------------------------------------------------------------------------

    x_start, y_start = min_x, random.uniform(max_x/2 - spacing, max_x/2 + spacing)
    x_goal, y_goal = max_x - 1, random.uniform(max_x/2 - 2, max_x/2 + 2)
    print("> Start: (x=" + str(x_start) + ", y=" + str(y_start) + ")")
    print("> Goal:  (x=" + str(x_goal) + ", y=" + str(y_goal) + ")")
    print("> Road Scenario: Highway")

    x_kerb = np.tile(np.linspace(min_x, max_x - 1, max_x), 2)
    y_kerb = np.concatenate(((max_x/2 - spacing) * np.ones(max_x), (max_x/2 + spacing) * np.ones(max_x)))

    x_obs = list(set(np.random.uniform(min_x, max_x, np.random.randint(0, 10))))
    y_obs = np.ravel(list(map((lambda n: generate_kerbs(scenario=0, kerb=n)), x_obs)))

    # INTERSECTION -----------------------------------------------------------------------------------------------------

    # x_start, y_start = min_x, random.uniform(max_x/2 - spacing, max_x/2 + spacing)
    # x_goal, y_goal = random.uniform(max_x/2 - spacing, max_x/2 + spacing), max_x - 1
    # print("\nStart: (x=" + str(x_start) + ", y=" + str(y_start) + ")")
    # print("Goal:  (x=" + str(x_goal) + ", y=" + str(y_goal)+ ")")
    # print("> Road Scenario: Intersection")
    #
    # x_kerb = np.concatenate((np.tile(np.linspace(min_x, max_x - 1, max_x), 2),
    #                         (max_y/2 - spacing) * np.ones(max_y - 2*spacing + 1),
    #                         (max_y/2 + spacing) * np.ones(max_y - 2*spacing + 1)))
    # y_kerb = np.concatenate(((max_x/2 - spacing) * np.ones(max_x - 2*spacing + 1),
    #                         (max_x/2 + spacing) * np.ones(max_x - 2*spacing + 1),
    #                         np.tile(np.linspace(min_y, max_y - 1, max_y), 2)))
    #
    # x_kerb = list(filter((lambda xx: not (max_x/2 - spacing < xx < max_x/2 + spacing)), x_kerb))
    # y_kerb = list(filter((lambda yy: not (max_y/2 - spacing < yy < max_y/2 + spacing)), y_kerb))
    #
    # x_obs = list(set(np.random.uniform(min_x, max_x, np.random.randint(0, 10))))
    # y_obs = np.ravel(list(map((lambda n: generate_kerbs(scenario=1, kerb=n)), x_obs)))

    # OBSTACLE GENERATION AND PATHFINDING ------------------------------------------------------------------------------

    cell_x_kerb = list(map(cell, x_kerb))               # Discrete x-coordinates of all kerbs
    cell_y_kerb = list(map(cell, y_kerb))               # Discrete y-coordinates of all kerbs
    kerbs = set(list(zip(cell_x_kerb, cell_y_kerb)))    # Set of discrete coordinates of all kerbs

    cell_x_obs = list(map(cell, x_obs))                 # Discrete x-coordinates of all obstacles
    cell_y_obs = list(map(cell, y_obs))                 # Discrete y-coordinates of all obstacles

    # List of obstacles within grid boundaries and not occupied by a kerb nor obstacle
    obstacles = list(filter((lambda n: n != (x_start, y_start) and n != (x_goal, y_goal) and n not in kerbs),
                            set(list(zip(cell_x_obs, cell_y_obs)))))
    print("> Number of Obstacles: %s" % len(obstacles))

    a_star_start = time()       # Start time of A* execution - used to calculate A* execution time

    # Defines search grid boundaries and locations of all kerbs and obstacles to prepare for pathfinding algorithm
    a_star = AStar(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, kerbs=kerbs, obstacles=obstacles,
                   spacing=spacing)
    path = a_star.find_path((x_start, y_start), (x_goal, y_goal))       # Shortest path generated by A*
    a_star_end = time() - a_star_start                                  # End time (i.e. duration) of A* execution
    print("> A* Execution Time: %s seconds" % a_star_end)

    final_path = []                       # List of discrete coordinates of nodes along shortest path
    [final_path.append(node) for node in path if node not in final_path]        # Update "final_path"

    # Two lists, one containing discrete x-coordinates of nodes along shortest path, and the other containing...
    # ...discrete y-coordinates
    x_path, y_path = map(list, zip(*final_path))

    # List of previous nodes along shortest path - used to calculate distance between a node and its parent
    final_path_prev = final_path[:1] + final_path[:-1]

    # List of distances between each node along shortest path and its corresponding parent node - used to determine...
    # ...heading at each node
    node_dist = list(map(tuple, np.subtract(final_path, final_path_prev)))

    # List of headings at each node along shortest path (within range [0, 360) degrees)
    theta_path = list(map((lambda n: np.rad2deg(np.arctan2(n[1], n[0])) % 360), [m for m in node_dist]))

    # Headings mapped to range (-180, 180] degrees
    # Steering clockwise should not result in heading change > 180 degrees, hence range is limited to 180 degrees max
    # Instead, anticlockwise steering should result in negative change in heading, hence range includes negative values
    theta_mapped = [n if 0 <= n <= 180 else n - 360 for n in list(theta_path)]
    print("> Min Heading: " + str(min(theta_mapped)) + " degrees \n> Max Heading: " + str(max(theta_mapped)) +
          " degrees \n> Heading Range: " + str(max(theta_mapped) - min(theta_mapped)) + " degrees")

    # Linear parametric B-spline interpolation to find path distance at long shortest path from start to goal node
    # B-spline is identical to original path generated by A*
    tck, u = itp.splprep([x_path, y_path], s=0, k=1)

    # tck = List of tuples, where each tuple contains:...
    # ...(t = vector of knots, c = B-spline coefficients, k = B-spline degree = 1)

    # u = B-spline parameter proportional to spline length with range [0, 1], where 0 and 1 corresponds to start and...
    # ...goal nodes, respectively

    # First derivatives of x- and y-coordinates with respect to "u" - used to calculate path distance
    dx_du = itp.splev(u, tck, der=1)[0]
    dy_du = itp.splev(u, tck, der=1)[1]

    integrand = np.sqrt(dx_du ** 2 + dy_du ** 2)        # To be integrated with respect to u to obtain path distance
    s_path = simpson(integrand, u) * u                  # List of path distances at each node along shortest path

    # Only print final element in s_path (i.e. total path distance)
    print("> Total Distance: %s cells" % str(s_path[-1]))

    # Linear parametric B-spline interpolation to find heading along shortest path from start to goal node
    tck_head, u_head = itp.splprep([s_path, theta_mapped], s=0, k=1)

    # "tck_head" and "u_head" are analogous to tck and u for path distance B-spline, but are unique to this spline...
    # ...(likewise, "tck" and "u" are only used in path distance B-spline, not this spline)

    s_spl, theta_spl = itp.splev(u_head, tck_head)
    # s_spl = Path distance at 1000 evenly spaced points along shortest path from start to goal node
    # theta_spl = Heading at 1000 evenly spaced points along shortest path from start to goal node

    dtheta_ds = np.divide(itp.splev(u_head, tck_head, der=1)[1], itp.splev(u_head, tck_head, der=1)[0])
    # dtheta_ds = Rate of change of heading at 1000 evenly spaced points along shortest path from start to goal node
    # d(theta)/ds = (dy/ds)/(dx/ds) = itp.splev(u_head, tck_head, der=1)[1] / itp.splev(u_head, tck_head, der=1)[0]

    print("> Mean d(theta)/ds: " + str(np.mean(dtheta_ds)) + " degrees \n> Var d(theta)/ds: " +
          str(np.var(dtheta_ds)) + " degrees^2")
    # Mean d(theta)/ds usually lower for conventional A* than hybrid A*
    # Variance of d(theta)/ds usually much higher for conventional A* than hybrid A*

    # A* SHORTEST PATH IN CARTESIAN COORDINATE SYSTEM ------------------------------------------------------------------
    """Generate a graph of discrete y-coordinates against x-coordinates, then plot the coordinates of all nodes
    along shortest path, joining all of them with a straight line; this graph provides a visualisation of the
    shortest path from start to goal node according to A*"""

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x_path, y_path, "-", color="darkred", label="Shortest Path")
    ax.plot(x_path, y_path, "x", color="purple", label="Waypoints")
    ax.plot(x_path[0], y_path[0], "x", color="orange", label="Start Node")
    ax.plot(x_path[-1], y_path[-1], "x", color="mediumturquoise", label="Goal Node")
    ax.plot(cell_x_obs, cell_y_obs, "s", color="k", label="Obstacles")
    ax.plot(cell_x_kerb, cell_y_kerb, "s", color="darkgrey", label="Kerbs")
    ax.text(x_path[0], y_path[0], "S", fontsize=12)
    ax.text(x_path[-1], y_path[-1], "G", fontsize=12)
    plt.title("A* Shortest Path in Cartesian Coordinate System")
    plt.xlabel("x [cell widths]")
    plt.ylabel("y [cell heights]")
    plt.legend(loc='best', fancybox=True, shadow=True)

    major_ticks = np.arange(min_x, max_x, cw * 5)
    minor_ticks = np.arange(min_x, max_x, cw)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])

    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.25)
    ax.grid(which='major', alpha=0.5)

    # HEADING PROFILE --------------------------------------------------------------------------------------------------
    """Plots heading and rate of change of heading against path distance"""

    fig_head, ax = plt.subplots(figsize=(8, 6))
    ax.plot(s_path, theta_mapped, "--", color="darkred", label=r"$\theta$", linewidth=0.5)
    ax.plot(s_spl, dtheta_ds, "-", color="orangered", label=r"d$\theta$/ds")
    plt.title("Heading Change Comparison Along Shortest Path")
    plt.xlabel("Distance s [cells]")
    plt.ylabel(r"Heading $\theta$ [deg],   Rate of Change of Heading d$\theta$/ds [deg/cells]")
    plt.legend(loc='best', fancybox=True, shadow=True)

    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.35)
    ax.grid(which='major', alpha=0.7)
    plt.minorticks_on()

    print("\nDisplaying results... \n> Program Execution Time: %s seconds" % (time() - start_time))
    plt.show()


if __name__ == '__main__':
    start_time = time()         # Start time of program execution (from definition of search grid to plotting results)
    main()                      # Execute entire program util results have been found
