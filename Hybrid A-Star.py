# Source code written and finalised by Kavin Ha: https://github.com/KavinHa0019/autonomous-vehicle-motion-planner
# Source code based off of jvirdi2's hybrid A* implementation: https://github.com/jvirdi2/A_star_and_Hybrid_A_star

import heapq as hq
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as itp
from scipy.integrate import simpson
import random
from time import time
from warnings import filterwarnings


class HybridAStar:
    """HybridAStar() class defines search grid boundaries, kerbs and obstacles"""
    def __init__(self, min_x, max_x, min_y, max_y, kerbs, obstacles, spacing):
        self.min_x = min_x                  # Minimum x-value in grid
        self.max_x = max_x                  # Maximum x-value in grid
        self.min_y = min_y                  # Minimum y-value in grid
        self.max_y = max_y                  # Maximum y-value in grid
        self.kerbs = kerbs                  # Kerbs - boundaries used to close off parts of grid to form a road
        self.obstacles = obstacles          # Obstacles - must be avoided at all costs
        self.spacing = spacing              # Spacing - determines number of lanes in road

    def find_path(self, start, end, l_f, l_r, dt):
        """Pathfinding algorithm - uses A* to find shortest path from start to end"""
        # Assign class attributes to new variables - reduces execution time by avoiding the need to search through class
        x_min = self.min_x
        x_max = self.max_x
        y_min = self.min_y
        y_max = self.max_y
        kerbs = self.kerbs
        obstacles = self.obstacles

        nodes_per_v = 9
        # nodes_per_v = Number of neighbouring nodes for each unique value of v
        # Only two unique values of v (1 and -1) so total number of neighbouring nodes is double nodes_per_v
        # Must be odd so that delta can take on value of 0

        # List of values for v and delta for every possible neighbouring node
        v = np.concatenate((-np.ones(nodes_per_v), np.ones(nodes_per_v)))
        delta = np.tile(np.linspace(-40, 40, nodes_per_v), 2)

        # v = Velocity input - can be 1 (forward driving) or -1 (reverse driving)
        # delta = Steering input - can be 0 +/- 40 degrees

        # Discrete coordinates (_d) - located at centre of cell
        # Continuous coordinates (_c) - located anywhere within cell
        start_c = (start[0], start[1], start[2])                        # Continuous coordinates of start node
        start_d = (cell(start[0]), cell(start[1]), round(start[2]))     # Discrete coordinates of start node
        end_c = (end[0], end[1], end[2])                                # Continuous coordinates of goal node
        end_d = (cell(end[0]), cell(end[1]), round(end[2]))             # Discrete coordinates of goal node

        open_heap = []
        # Priority queue ranks nodes based on f         [(f, node_d), ...]
        open_dict = {}
        # Contain all nodes about to be explored        {node_d: (f, node_c, (parent_d, parent_c), v), ...}
        closed_dict = {}
        # Contain all nodes already explored            {node_d: (f, node_c, (parent_d, parent_c), v), ...}

        g = 0                                           # g(start) = 0
        f = g + euclidean(start_d, end_d, True)         # Heuristic h(n) = euclidean(n, end, True)

        # Begin pathfinding by exploring start node first
        hq.heappush(open_heap, (f, start_d))
        # Include start node in open dictionary - will explore start node first
        open_dict[start_d] = (f, start_d, (start_d, start_c), 0)

        def neighbour_node(v_i, delta_i):
            """Continuous coordinates of neighbouring nodes are calculated using a kinematic bicycle model of the
            vehicle; velocity and steering g-cost terms for each neighbour are also calculated here - they depend on
            the input [v, delta] used to travel to their corresponding neighbour"""

            beta = np.rad2deg(np.arctan((l_f / (l_f + l_r)) * np.tan(np.deg2rad(delta_i)))) * dt
            neighbour_x_c = current_c[0] + (v_i * np.cos(np.deg2rad(current_c[2] + beta))) * dt
            neighbour_y_c = current_c[1] + (v_i * np.sin(np.deg2rad(current_c[2] + beta))) * dt
            neighbour_theta_c = np.rad2deg(np.deg2rad(current_c[2]) +
                                           (v_i * np.sin(np.deg2rad(beta)) / l_r) * dt) % 360
            # beta = Sideslip angle [deg]
            # neighbour_x_c = Continuous x-coordinates of neighbouring node
            # neighbour_y_c = Continuous y-coordinates of neighbouring node
            # neighbour_theta_c = Heading coordinate of neighbouring node
            # current_c = Continuous coordinates (current_x_c, current_y_c, current_theta_c) of current node

            # Discrete coordinates of neighbouring node
            neighbour_x_d = cell(neighbour_x_c)
            neighbour_y_d = cell(neighbour_y_c)
            neighbour_theta_d = round(neighbour_theta_c)

            v_cost = 1 if v_i < 0 else 0        # Cost of arriving at node with velocity input v
            delta_cost = abs(delta_i) / 40      # Cost of arriving at node with steering input delta

            # Ideally, vehicle should arrive with v = 1 and delta = 0
            # Therefore, nodes that require v = 1 and delta = 0 have v_cost = 0 and delta_cost = 0, respectively
            # Reverse driving is undesirable so nodes that encourage it have a v_cost of 1
            # Similarly, sharp steering should be discouraged so delta_cost is proportional to delta - the smaller...
            # ...the steering angle, the better

            neighbour = ((neighbour_x_d, neighbour_y_d, neighbour_theta_d),
                         (neighbour_x_c, neighbour_y_c, neighbour_theta_c), v_i, delta_i, v_cost, delta_cost)
            # Tuple contains information about a particular neighbouring node in the following order:
            # ...(discrete coordinates, continuous coordinates, velocity input, steering input, velocity cost term, ...
            # ...steering cost term)
            return neighbour

        def obs_cost(n):
            """Computes obstacle cost based on distances between neighbours and all obstacles in grid"""
            spacing = self.spacing                  # Number of cells above max_x/2
            lanes = (2*spacing - 1) / cw            # Number of lanes (i.e. cells between parallel kerbs)

            # Obstacle cost formula
            g_obs_i = lanes * sum(list(map((lambda m: 1 / euclidean(n, m, False)**2), obstacles)))
            return g_obs_i

        # While priority queue is not empty (i.e. there are nodes yet to be explored)
        while len(open_heap) > 0:

            # open_heap[0] = Node in priority queue with lowest f-cost
            # Make node with lowest f-cost the current node
            current_d = open_heap[0][1]                 # Discrete coordinates of current node
            current_c = open_dict[current_d][1]         # Continuous coordinates of current node
            f_current = open_heap[0][0]                 # f-cost of current node

            closed_dict[current_d] = open_dict[current_d]       # Current node will already be explored after one loop

            # If goal node is one of current node's neighbours (i.e. within one cell width away from current node)
            if euclidean(current_d, end_d, True) < cw:
                print("\nGoal node found - tracing shortest path...")
                rev_final_path = [end_d]            # Trace shortest path beginning with goal node
                node = current_d                    # Include current node in shortest path following goal node
                direction = []
                # Direction of motion (i.e. forward or reverse) at every node along shortest path

                while node != start_d:
                    open_node = closed_dict[node]           # Most recent node along shortest path list
                    parent = open_node[2][1]                # Identify (continuous) parent node of "node"
                    rev_final_path.append(parent)           # Add parent to shortest path
                    direct = open_node[3]                   # Direction of motion at parent node
                    direction.append(direct)                # Append "direct" to "direction"
                    node = open_node[2][0]                  # Now parent is most recent node - make parent = "node"

                direction.append(open_dict[start_d][3])     # Append direction of motion at start node to "direction"
                direction[-1] = direction[-2]
                # Assume direction at goal node = direction at penultimate node along shortest path (i.e. vehicle...
                # ...does not change direction from when algorithm sees goal node as neighbour to when it reaches it)

                # Reverse order of shortest path and direction lists - make start node first element in both
                return rev_final_path[::-1], direction[::-1]

            hq.heappop(open_heap)                               # Remove current node from priority node after exploring

            # Nodes that vehicle can arrive at according to kinematic model are neighbours
            neighbours = list(map(neighbour_node, v, delta))

            # Neighbours must be in grid and not occupied by kerb nor obstacle (such nodes known as "free neighbours")
            # List comprehension filters out neighbours that do not satisfy above criteria
            neighbours_free = list(filter((lambda n: x_min <= n[0][0] <= x_max and y_min <= n[0][1] <= y_max and
                                          (n[0][0], n[0][1]) not in (kerbs or obstacles)), [m for m in neighbours]))

            # If current node has no neighbours, skip it and move to next node in priority queue
            if len(neighbours_free) == 0:
                pass

            # Otherwise, compute neighbouring nodes and their corresponding costs
            else:
                # Zip neighbours_free into 6 lists, each containing one piece of information about all free neighbours
                nfree_d, nfree_c, v_node, delta_node, g_v, g_delta = [list(n) for n in zip(*neighbours_free)]
                nfree_theta_c = [n[2] for n in nfree_c]

                # nfree_d = Discrete coordinates of free neighbours
                # nfree_c = Continuous coordinates of free neighbours
                # nfree_theta_c = Heading coordinates of free neighbours - used to determine heading change cost term
                # v_node = Velocity inputs used to arrive at free neighbours
                # delta_node = Steering inputs used to arrive at free neighbours
                # g_v = Velocity cost terms of free neighbours
                # g_delta = Steering cost terms of free neighbours

                # Obstacle cost terms for every free neighbour
                # Only distance between discrete coordinates of neighbour and obstacles are considered
                # Free neighbours are likely to share cells (especially if there are more than 8 of them), so if...
                # ...discrete distances are considered, some neighbours will share same obstacle cost
                # Because of this, computing obstacle cost term individually for every free neighbour is redundant...
                # ...and requires more computational resources
                # Instead, algorithm only computes obstacle cost for all unique discrete coordinates in nfree_d...
                # ...and records them in dictionary
                # If free neighbour occupies same cell as another free neighbour whose obstacle cost is in...
                # ...dictionary, former will immediately share same obstacle cost as latter - no calculation required

                # To get all unique coordinates in nfree_d, algorithm creates set to remove duplicates in nfree_d
                g_obs_all = list(map(obs_cost, list(set(nfree_d))))

                # Obstacle costs are assigned to unique coordinates and stored in dictionary
                obs_costs = dict(zip(list(set(nfree_d)), g_obs_all))

                # g_obs = Obstacle cost term
                # For every free neighbour, get corresponding obstacle cost from obs_costs dictionary
                g_obs = list(map((lambda n: obs_costs[n]), nfree_d))

                # g_head = Heading change cost term
                # Penalises sharp changes in steering angle which may lead to unsafe/uncomfortable manoeuvres
                g_head = np.divide((abs(180 - abs((np.array(nfree_theta_c) -
                                    current_c[2] * np.ones(len(nfree_theta_c))) - 180))), 180)

                g_current = (f_current - euclidean(current_c, end_c, True)) * np.ones(len(nfree_c))
                g_to_neighbour = list(map((lambda n: euclidean(current_c, n, True)), nfree_c))
                # g_current = g-cost to get from start to current node
                # g_to_neighbour = g-cost to get from current node to neighbour

                g = np.array(g_current) + np.array(g_to_neighbour) + \
                    np.array(g_v) + np.array(g_delta) + np.array(g_head) + np.array(g_obs)      # Total g-cost
                h = list(map((lambda n: euclidean(n, end_c, True)), nfree_c))                   # Heuristic value
                f = g + np.array(h)                                                             # Total f-cost

                # For all free neighbouring nodes (i.e. nodes that vehicle can occupy), determine what to do with...
                # ...f-cost of each neighbour
                # Compares most recent f-cost with f-cost in open dictionary or both dictionaries
                # Algorithm always assigns lowest f-cost to every node

                for i in range(len(nfree_d)):
                    # If neighbour already exists in open dictionary and has lower f-cost, ignore most recent f-cost
                    if nfree_d[i] in open_dict and f[i] >= open_dict[nfree_d[i]][0]:
                        pass

                    # If neighbour already exists in both dictionaries and has lower f-cost, ignore most recent f-cost
                    elif nfree_d[i] in (open_dict and closed_dict) and f[i] >= closed_dict[nfree_d[i]][0]:
                        pass

                    # Otherwise, consider f-cost and discrete coordinates of neighbour
                    else:
                        hq.heappush(open_heap, (f[i], nfree_d[i]))
                        open_dict[nfree_d[i]] = (f[i], nfree_c[i], (current_d, current_c), v_node[i] / abs(v_node[i]))

        # End pathfinding algorithm if it is impossible to arrive at goal node - return empty list
        # Empty list will cause program to run into error and terminate as algorithm will have nothing to work with
        print("\nGoal node cannot be found.")
        return []


def cell(q):
    """Converts continuous coordinates into discrete coordinates by identifying the cell the coordinates lie in and
    outputting the centre coordinates of that cell"""
    if abs(q - int(q)) > cw / 2:            # If continuous coordinates are to the right of cell centre
        return round(q) - cw / 2            # Round up then subtract by half cell width - shift to the left
    elif abs(q - int(q)) < cw / 2:          # If continuous coordinates are to the left of cell centre
        return round(q) + cw / 2            # Round down then add by half cell width - shift to the right
    elif abs(q - int(q)) == cw / 2:         # If continuous coordinates lie exactly at cell centre
        return q                            # Keep coordinates the same


def euclidean(position, target, cont):
    """Mainly used as heuristic but can be used to find Euclidean distance between any two cells; in the case that
    "target" is the goal node, this function computes the heuristic value of the "position" node"""

    # "cont" is a Boolean variable that determines whether distances should be taken between discrete or continuous...
    # ...coordinates of cells
    # True = continuous distance (includes headings of "position" and "target" nodes)
    # False = discrete distance (headings not considered) - only used when determining obstacle cost

    if cont:        # cont = True
        dist = np.sqrt((position[0] - target[0]) ** 2 + (position[1] - target[1]) ** 2 +
                       (np.deg2rad(position[2]) - np.deg2rad(target[2])) % np.pi ** 2)
    else:           # cont = False
        dist = np.sqrt((position[0] - target[0]) ** 2 + (position[1] - target[1]) ** 2)
    return dist


def velocity_profile(final_path, direction, a_lat_max, tck_spl3, u_spl3):
    """Generates velocity profile (with distance s as domain and velocity v as range) for shortest path based on
    reference velocity, velocity of leading vehicle and maximum allowable velocity due to path curvature;
    usually generates comfortable profiles but will always guarantee passenger safety, even if at the cost of comfort"""

    v_max = []          # Maximum allowable velocity that vehicle can travel along shortest path
    v_ref = []          # Reference velocity (speed limit or 0 if direction of motion changes)
    v_lead = []         # Velocity of leading vehicle (or leading velocity - slightly below v_ref +/- small disturbance)
    v_curve = []        # Curvature-dependent velocity (maximum allowable velocity to remain on curved path)

    prev_direction = direction[-1:] + direction[:-1]
    # Direction of motion at parents of nodes along shortest path - used to determine whether there is a change in...
    # ...direction at any given node

    prev_direction[0], prev_direction[-1] = 0, 0
    # Usually "direction" can only either be 1 or -1, but at start and goal nodes where vehicle is assumed to be...
    # ...stationary, a dummy value of 0 is assigned to "direction"
    # Only these two nodes have a "direction" value of 0

    diff_direction = list(np.array(direction) - np.array(prev_direction))
    # Compares direction at any given node to direction at its parent
    # If "diff_direction" = 0, direction of motion is the same as for its parent node
    # Otherwise, (if "diff_direction" != 0) direction of motion for this node is different from its parent node...
    # ...indicating a change in direction (and thus zero velocity) at that node itself

    change_direction = list(np.nonzero(diff_direction)[0])
    # Nodes in "change_direction" have a non-zero "diff_direction" value
    # Since no other node has a dummy value for "direction" expect start and goal nodes, "diff_direction" for both...
    # ...those nodes will always be non-zero and therefore have zero velocity

    # First derivatives of x- and y-coordinates with respect to "u" - used to calculate smoothed path distance
    dx_du = itp.splev(u_spl3, tck_spl3, der=1)[0]
    dy_du = itp.splev(u_spl3, tck_spl3, der=1)[1]
    integrand = np.sqrt(dx_du ** 2 + dy_du ** 2)        # To be integrated with respect to u to obtain path distance
    s = simpson(integrand, u_spl3) * u_spl3             # List of path distances at each node along shortest path

    # Second derivatives of x- and y-coordinates with respect to "u" - used to calculate curvature along smoothed path
    d2x_du2 = itp.splev(u_spl3, tck_spl3, der=2)[0]
    d2y_du2 = itp.splev(u_spl3, tck_spl3, der=2)[1]

    curve = abs(dx_du * d2y_du2 - dy_du * d2x_du2) / ((dx_du ** 2 + dy_du ** 2) ** 1.5)
    # List of curvatures at each node along shortest path

    speed_limit = 20        # Speed limit used as reference velocity

    # For every node along shortest path
    for i in range(len(final_path)):
        if i in change_direction:           # If there is a change in direction
            v_ref_i = 0                     # Reference velocity = 0
        else:                               # Otherwise, if direction is constant
            v_ref_i = speed_limit           # Reference velocity = Speed limit

        v_lead_i = 0.75 * speed_limit + np.random.uniform(-1, 1)
        # Velocity of leading vehicle (equal to 75% of speed limit +/- 1)
        # Disturbance included to account for slight changes in leading vehicle movement, which does not travel...
        # ...at exactly constant speed at all times

        try:
            v_curve_i = np.sqrt(a_lat_max / curve[i])       # Curvature-dependent velocity
        except ZeroDivisionError:
            v_curve_i = float("Inf")                        # If curvature is zero, assume infinite velocity

        v_max_i = min(abs(v_ref_i), abs(v_lead_i), abs(v_curve_i))
        # Maximum allowable velocity defined as minimum of reference velocity "v_ref", leading velocity "v_lead" and...
        # ...curvature-dependent velocity "v_curve"
        # Even if "v_curve" = infinity, since "v_ref" and "v_lead" can never be infinite, "v_max" can also only take...
        # ...on a finite value

        v_ref.append(v_ref_i)
        v_lead.append(v_lead_i)
        v_curve.append(v_curve_i)
        v_max.append(v_max_i)

    # PCHIP smoothing for both velocity profiles (with and without direction)
    # PCHIP preferred over B-spline in this case as PCHIP avoids overshoots
    # Overshoots can be misleading as they could imply a change in direction (if spline crosses x-axis) even if...
    # ..."direction is constant
    s_spl3 = np.linspace(0, max(s), 1000)
    v_spl = itp.pchip_interpolate(s, v_max, s_spl3)

    # Function returns tuple consisting of 9 lists:
    # (shortest path total distance, smoothed path total distance, max allowable velocity at each node, ...
    # ...max allowable velocity along smoothed path, reference velocity at each node, leading velocity at each node, ...
    # ...curvature-dependent velocity at each node)
    return s, s_spl3, v_max, v_spl, v_ref, v_lead, v_curve


def d2v_ds2_sign(aa, cc, dd):
    """Using the signs of two coefficients -j and a^2 of the equation v^3 d2v/ds2 - jv + a^2 = 0, this function can
    determine the sign of d2v/ds2; the cubic equation is used to determine values of velocity v that correspond to
    desired values of acceleration a, jerk j, and d2v/ds2"""

    if np.sign(cc) + np.sign(dd) > 0:                   # If cc and dd are both positive
        return min(-aa, aa)                             # Make aa negative
    elif np.sign(cc) + np.sign(dd) < 0:                 # If cc and dd are both negative
        return max(-aa, aa)                             # Make aa positive
    else:
        if np.sign(cc) == 0 and np.sign(dd) == 0:       # If cc and dd are both zero
            return 0                                    # Make aa zero
        else:                                           # If cc and dd have opposite signs
            return max(-aa, aa)                         # Make aa positive


cw = 1          # Cell width


def main():
    """The function used to execute the entire program, including the definition of the search area, the execution of
    the pathfinding algorithm and the generation of graphs describing the shortest path"""

    print("\nStarting hybrid A* path planner...")
    # Runtime warnings may arise if division by zero may be performed - no errors encountered so far regarding this...
    # ...issue so can safely ignore these messages
    filterwarnings("ignore", category=RuntimeWarning)

    min_x, max_x, min_y, max_y = 0, 50, 0, 50   # Grid boundaries - used to determine whether a cell lies out of bounds
    a_lon_max, a_lat_max, j_max = 2, 1, 0.9     # Comfort constraints (maximum allowable acceleration and jerk)

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

    """Kerbs for highway and intersection scenarios; comment lines 405-417 for intersection scenario, or comment lines
    421-440 for highway scenario; one of the two must be commented out at all times"""

    # HIGHWAY ----------------------------------------------------------------------------------------------------------

    x_start, y_start, theta_start = min_x, random.uniform(max_x/2 - spacing, max_x/2 + spacing), \
        random.uniform(-10, 10) % 360
    x_goal, y_goal, theta_goal = max_x - 1, random.uniform(max_x/2 - 2, max_x/2 + 2), \
        random.uniform(-10, 10) % 360
    print("> Start: (x=" + str(x_start) + ", y=" + str(y_start) + ", theta=" + str(theta_start) + ")")
    print("> Goal:  (x=" + str(x_goal) + ", y=" + str(y_goal) + ", theta=" + str(theta_goal) + ")")
    print("> Road Scenario: Highway")

    x_kerb = np.tile(np.linspace(min_x, max_x - 1, max_x), 2)
    y_kerb = np.concatenate(((max_x/2 - spacing) * np.ones(max_x), (max_x/2 + spacing) * np.ones(max_x)))

    x_obs = list(set(np.random.uniform(min_x, max_x, np.random.randint(5, 11))))
    y_obs = np.ravel(list(map((lambda n: generate_kerbs(scenario=0, kerb=n)), x_obs)))

    # INTERSECTION -----------------------------------------------------------------------------------------------------

    # x_start, y_start, theta_start = min_x, random.uniform(max_x/2 - spacing, max_x/2 + spacing), \
    #     random.uniform(-10, 10) % 360
    # x_goal, y_goal, theta_goal = random.uniform(max_x/2 - 2, max_x/2 + 2), max_x - 1, \
    #     random.uniform(80, 100) % 360
    # print("> Start: (x=" + str(x_start) + ", y=" + str(y_start) + ", theta=" + str(theta_start) + ")")
    # print("> Goal:  (x=" + str(x_goal) + ", y=" + str(y_goal) + ", theta=" + str(theta_goal) + ")")
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
    # x_obs = list(set(np.random.uniform(min_x, max_x, np.random.randint(15, 21))))
    # y_obs = np.ravel(list(map((lambda n: generate_kerbs(scenario=1, kerb=n)), x_obs)))

    # OBSTACLE GENERATION AND PATHFINDING ------------------------------------------------------------------------------

    cell_x_kerb = list(map(cell, x_kerb))                   # Discrete x-coordinates of all kerbs
    cell_y_kerb = list(map(cell, y_kerb))                   # Discrete y-coordinates of all kerbs
    kerbs = set(list(zip(cell_x_kerb, cell_y_kerb)))        # Set of discrete coordinates of all kerbs

    cell_x_obs = list(map(cell, x_obs))                     # Discrete x-coordinates of all obstacles
    cell_y_obs = list(map(cell, y_obs))                     # Discrete y-coordinates of all obstacles

    # List of obstacles within grid boundaries and not occupied by a kerb nor obstacle
    obstacles = list(filter((lambda n: n != (x_start, y_start) and n != (x_goal, y_goal) and n not in kerbs),
                            set(list(zip(cell_x_obs, cell_y_obs)))))
    print("> Number of Obstacles: %s" % len(obstacles))

    hy_a_star_start = time()        # Start time of hybrid A* execution - used to calculate hybrid A* execution time

    # Defines search grid boundaries and locations of all kerbs and obstacles to prepare for pathfinding algorithm
    hy_a_star = HybridAStar(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, kerbs=kerbs, obstacles=obstacles,
                            spacing=spacing)
    path, direction = hy_a_star.find_path((x_start, y_start, theta_start), (x_goal, y_goal, theta_goal), l_f=2, l_r=2.5,
                                          dt=cw)
    # path = Shortest path generated by hybrid A*
    # direction = Direction of motion at each node along shortest path

    hy_a_star_end = time() - hy_a_star_start                # End time (i.e. duration) of A* execution
    print("> Hybrid A* Execution Time: %s seconds" % hy_a_star_end)

    final_path = []                       # List of discrete coordinates of nodes along shortest path
    [final_path.append(node) for node in path if node not in final_path]        # Update "final_path"

    # Three lists, one containing discrete x-coordinates of nodes along shortest path, another containing...
    # ...discrete y-coordinates, and the last containing heading coordinates
    x_path, y_path, theta_path = map(list, zip(*final_path))

    # Headings mapped to range (-180, 180] degrees
    # Steering clockwise should not result in heading change > 180 degrees, hence range is limited to 180 degrees max
    # Instead, anticlockwise steering should result in negative change in heading, hence range includes negative values
    theta_mapped = [n if 0 <= n <= 180 else n - 360 for n in list(theta_path)]
    print("> Min Heading: " + str(min(theta_mapped)) + " degrees \n> Max Heading: " + str(max(theta_mapped)) +
          " degrees \n> Heading Range: " + str(max(theta_mapped) - min(theta_mapped)) + " degrees")

    # Linear parametric B-spline interpolation to find path distance at long shortest path from start to goal node
    # B-spline is identical to original path generated by hybrid A*
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

    # Cubic (degree 3) B-spline interpolation in order to smoothen shortest path, improving passenger comfort
    # Curvature depends on d2x/du2 and d2y/du2 - degree of cubic spline is high enough so that curvature is...
    # ...continuous along B-spline smoothed path
    # Meanwhile, shortest paths generated conventional and hybrid A* are modelled as linear functions; degree of
    # linear functions is not high enough, so curvature is discontinuous (either 0 or infinity)

    tck_spl, u_spl = itp.splprep([x_path, y_path], s=0, k=3)
    # "tck_spl" and "u_spl" are analogous to tck and u for path distance B-spline, but are unique to this spline

    x_spl3, y_spl3 = itp.splev(np.linspace(0, 1, 1000), tck_spl)
    # Continuous x- and y- coordinates for 1000 evenly spaced points along smoothed path from start to goal node

    print("\nShortest path found - starting velocity planner...")
    velo_profile_start = time()
    # Start time of velocity planner execution - used to calculate velocity planner execution time

    # Generates velocity profile for B-spline smoothed path
    velocity = velocity_profile(final_path, direction, a_lat_max, tck_spl, u_spl)
    s, s_spl3, v_max, v_spl, v_ref, v_lead, v_curve = velocity

    # Acceleration: a = v dv/ds, Jerk: j = v(v d2v/ds2 + (dv/ds)^2)
    # Acceleration and jerk profiles for B-spline smoothed path generated using above formulas
    # To meet comfort constraints, both profiles were clipped so that they can never exceed "a_lon_max", ...
    # ..."a_lat_max" and "j_max" - these profiles represent the DESIRED acceleration and jerk

    a_spl = np.clip(v_spl * itp.pchip_interpolate(s, v_max, s_spl3, der=1), -a_lon_max, a_lon_max)
    j_spl = np.clip(v_spl * ((v_spl * itp.pchip_interpolate(s, v_max, s_spl3, der=2)) +
                    (itp.pchip_interpolate(s, v_max, s_spl3, der=1))**2), -j_max, j_max)
    # a_spl = Clipped longitudinal acceleration profile for B-spline smoothed path
    # j_spl = Clipped jerk profile for B-spline smoothed path

    a = itp.pchip_interpolate(s_spl3, a_spl, s)
    j = itp.pchip_interpolate(s_spl3, j_spl, s)
    # a = Desired longitudinal acceleration at each node along shortest path
    # j = Desired jerk at each node along shortest path
    
    # Desired acceleration and jerk set to zero at goal node
    # This ensures no change in velocity so vehicle can remain stationary when it arrives at goal node
    a[-1], j[-1] = 0, 0

    # Unlike acceleration and jerk profiles generated from original velocity profile, these new splines do not need...
    # ...to be clipped; they will always satisfy comfort constraints for all points along smoothed path
    # These new profiles generally exhibit more gradual changes in acceleration/jerk compared to their original...
    # ...counterparts, which involve much larger spikes that must be clipped

    a_spl_new = itp.pchip_interpolate(s, a, s_spl3)
    j_spl_new = itp.pchip_interpolate(s, j, s_spl3)
    # a_spl_new = PCHIP spline interpolation of a
    # j_spl_new = PCHIP spline interpolation of j

    d2v_ds2_spl = itp.pchip_interpolate(s, v_max, s_spl3, der=2)     # Second derivative of velocity profile
    d2v_ds2 = itp.pchip_interpolate(s_spl3, d2v_ds2_spl, s)          # d2v/ds2 at each node along shortest path

    d2v_ds2 = np.clip(list(map(d2v_ds2_sign, d2v_ds2, -j, a**2)), np.minimum(0, -4 * -j ** 3 / (27 * a ** 4)),
                      np.maximum(0, -4 * -j ** 3 / (27 * a ** 4)))
    # Clipped d2v/ds2 function at each node along shortest path
    # To ensure all solutions for v are real, d2v_ds2 must be between min(0, 4j^3 / 27a^4) and max(0, 4j^3 / 27a^4)...
    # ...for all nodes along shortest path
    # These limits correspond to the determinant of the cubic equation (seen in "d2v_ds2_sign()") >= 0, which...
    # ...generates real solutions for v

    d2v_ds2 = [0 if np.isnan(n) else n for n in d2v_ds2]
    # For any given node, if -j and a^2 = 0, d2v_ds2 = 0 so that v can take any value (usually value of v for...
    # ...previous node - constant velocity ensures maximum comfort)
    # If d2v_ds2 != 0, v = 0 meaning the vehicle would unnecessarily stop at that node, even though "v_max" permits...
    # ...it to keep moving

    v = list((map((lambda aa, bb, cc, dd: list(np.polynomial.polynomial.polyroots([dd, cc, bb, aa]))),
                  d2v_ds2, np.ravel(np.zeros(len(d2v_ds2))), -j, a**2)))
    # For each node along shortest path, cubic equation is generated (in the form shown in "d2v_ds2_sign") using...
    # ...values of d2v/ds2, -j and a^2 corresponding to that node
    # Equations are then solved to obtain all possible solutions of v corresponding to each node

    v_real = list(map((lambda solution: list(np.array(solution).real)), v))
    # Although all solutions are real, some solutions may include negligible (but non-zero) imaginary parts, ...
    # ...resulting in Python recognising them as complex values
    # Imaginary parts are therefore removed manually, leaving behind real parts

    v_valid = [[m for m in n if 0 < m <= v_max[v_real.index(n)]] for n in v_real]
    # For each node, any solutions that do not exceed the node's corresponding "v_max" are termed valid solutions, ...
    # ...and are thus kept by the velocity planner; those that exceed this threshold value are rejected

    v_valid[0], v_valid[-1] = [0], [0]
    # Velocity at start and goal nodes should be zero already according to "v_ref" (which "v_max" depends on)
    # In case solutions to v at start and goal nodes are non-zero (replacing previous values of v), this line resets...
    # ...them back to zero

    v_chosen = [max(n) if len(n) != 0 else "nan" for n in v_valid]
    # If there are multiple solutions for a given node, select the largest one
    # If there are no solutions, add node to "empty" list below

    empty = list(filter((lambda n: v_chosen[n] == "nan"), [m[0] for m in list(enumerate(v_valid))]))

    # For each node that does not have valid solutions for v <= "v_max"
    for i in range(len(empty)):
        v_chosen[empty[i]] = v_max[empty[i] - 1]        # Velocity of node = "v_max" at parent node

    v_chosen = list(np.minimum(v_chosen, v_max))
    # In case all solutions still exceed "v_max" for a given node, velocity at that node will be capped at "v_max"...
    # ...(i.e. vehicle will travel at maximum allowable velocity at that node)
    # Otherwise, largest solution for v below "v_max" will be selected as velocity at that node

    # For each node n from start node to penultimate node along shortest path
    for n in range(len(v_chosen) - 2):
        if v_chosen[n+1] < v_chosen[n] <= v_max[n+1]:
            # If velocity at child node < velocity at node n <= "v_max" at child node (i.e. if vehicle travels...
            # ...slower at next node compared to this node)

            v_chosen[n+1] = v_chosen[n]
            # Make velocity at child node equal to velocity at node n
            # Do not slow down unless necessary to remain below or equal to "v_max"
            # Avoids slowing down too much which reduces dv/ds and d2v/ds2 (and thus acceleration and jerk) while...
            # ...encouraging the vehicle to travel faster if it is safe to do so

    v_chosen_spl = itp.pchip_interpolate(s, v_chosen, s_spl3)
    # Final velocity profile generated via PCHIP spline interpolation using velocity values in "v_chosen"
    # In cases where two consecutive nodes (i.e. a node and its child) share the same velocity, the profile usually...
    # ...assigns the same velocity for all points along the spline between the nodes
    # Essentially, the velocity planner assumes the vehicle travels at constant velocity between these two nodes, ...
    # ... which means dv/ds and d2v/ds2 (and thus acceleration and jerk) are zero between them

    velo_chosen = np.multiply(direction, v_chosen)
    # Same as "v_chosen" except direction of motion at each node has been taken account
    # Some values are positive (indicating forward driving) while others are negative (indicating reverse driving)
    # Magnitude of all values of "velo_chosen" are identical to "v_chosen"

    velo_chosen_spl = itp.pchip_interpolate(s, velo_chosen, s_spl3)
    # PCHIP spline interpolation of velo_chosen - result is final velocity profile with direction taken into account

    velo_profile_end = time() - velo_profile_start      # End time (i.e. duration) of velocity planner execution
    print("> Velocity Planner Execution Time: %s seconds" % velo_profile_end)

    # HYBRID A* SHORTEST PATH IN CARTESIAN COORDINATE SYSTEM -----------------------------------------------------------
    """Generate a graph of continuous y-coordinates against x-coordinates, then plot the coordinates of all nodes
    along shortest path, joining all of them with a straight line and cubic B-spline; this graph provides a 
    visualisation of the shortest path from start to goal node according to hybrid A*"""

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x_path, y_path, "-", color="darkgreen", label="Shortest Path")
    ax.plot(x_spl3, y_spl3, "-", color="lime", label="Smoothed Path")
    ax.plot(x_path, y_path, "x", color="purple", label="Waypoints")
    ax.plot(x_path[0], y_path[0], "x", color="orange", label="Start Node")
    ax.plot(x_path[-1], y_path[-1], "x", color="mediumturquoise", label="Goal Node")
    ax.plot(cell_x_obs, cell_y_obs, "s", color="k", label="Obstacles")
    ax.plot(cell_x_kerb, cell_y_kerb, "s", color="darkgrey", label="Kerbs")
    ax.text(x_path[0], y_path[0], "S", fontsize=12)
    ax.text(x_path[-1], y_path[-1], "G", fontsize=12)
    plt.title("Hybrid A* Shortest Path in Cartesian Coordinate System")
    plt.xlabel("x [cells]")
    plt.ylabel("y [cells]")
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

    # SPEED PROFILE (Speed v vs Distance s) ----------------------------------------------------------------------------

    plt.figure()
    plt.plot(s, v_max, color="darkred", label="Maximum Velocity")
    plt.plot(s, v_max, ".", color="k")
    plt.plot(s_spl3, v_spl, color="orangered", label="Smoothed Maximum Velocity")
    plt.plot(s, v_chosen, "x", color="purple")
    plt.plot(s, v_chosen, color="darkgreen", label="Final Velocity Profile")
    plt.plot(s_spl3, v_chosen_spl, color="lime", label="Smoothed Velocity Profile")
    plt.plot(s, v_ref, "--", color="r", label="Reference Velocity", linewidth=0.5)
    plt.plot(s, v_lead, "--", color="g", label="Leading Vehicle Velocity", linewidth=0.5)
    plt.plot(s, v_curve, "--", color="b", label="Curvature-Dependent Velocity", linewidth=0.5)
    plt.plot(s[0], v_chosen[0], "x", color="orange")
    plt.plot(s[-1], v_chosen[-1], "x", color="mediumturquoise")
    plt.title("Speed Profile (Speed vs Path Distance)")
    plt.xlabel("Path Distance s [cells]")
    plt.ylabel("Speed v [cells/s]")
    plt.legend(loc='best', fancybox=True, shadow=True)

    plt.grid(which='both')
    plt.grid(which='minor', alpha=0.25)
    plt.grid(which='major', alpha=0.5)
    plt.fill_between(s_spl3, v_chosen_spl, v_spl, color="lightcyan")
    plt.ylim([min(v_max) - (0.1 * min(v_max)), max(v_max) + (0.1 * max(v_max))])

    # VELOCITY PROFILE (Velocity v vs Distance s) ----------------------------------------------------------------------

    plt.figure()
    plt.plot(s, v_chosen, "x", color="purple")
    plt.plot(s[0], velo_chosen[0], "x", color="orange")
    plt.plot(s[-1], velo_chosen[-1], "x", color="mediumturquoise")
    plt.plot(s_spl3, velo_chosen_spl, "-", color="lime")
    plt.title("Velocity Profile (Velocity vs Path Distance)")
    plt.xlabel("Path Distance s [cells]")
    plt.ylabel("Velocity v [cells/s]")

    plt.grid(which='both')
    plt.grid(which='minor', alpha=0.25)
    plt.grid(which='major', alpha=0.5)
    plt.ylim([min(velo_chosen) - (0.1 * min(velo_chosen)), max(velo_chosen) + (0.1 * max(velo_chosen))])

    # ACCELERATION PROFILE (Acceleration a vs Distance s) --------------------------------------------------------------

    plt.figure()
    plt.plot(s, a, color="darkgreen", label="Acceleration")
    plt.plot(s, a, "x", color="purple")
    plt.plot(s[0], a[0], "x", color="orange")
    plt.plot(s[-1], a[-1], "x", color="mediumturquoise")
    plt.plot(s_spl3, a_spl, "--", color="b", label="Old Interpolated Acceleration", linewidth=0.5)
    plt.plot(s_spl3, a_spl_new, color="lime", label="New Interpolated Acceleration")
    plt.hlines(y=a_lon_max, xmin=0, xmax=max(s), color="red", linestyles="--", label="Max Comfort Acceleration",
               linewidth=0.5)
    plt.hlines(y=-a_lon_max, xmin=0, xmax=max(s), color="red", linestyles="--", label="Min Comfort Acceleration",
               linewidth=0.5)
    plt.title("Acceleration Profile (Acceleration vs Path Distance)")
    plt.xlabel("Path Distance s [cells]")
    plt.ylabel("Acceleration a [cells/s^2]")
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(which='both')
    plt.grid(which='minor', alpha=0.25)
    plt.grid(which='major', alpha=0.5)
    plt.ylim([1.1 * -a_lon_max, 1.1 * a_lon_max])

    # JERK PROFILE (Jerk j vs Distance s) ------------------------------------------------------------------------------

    plt.figure()
    plt.plot(s, j, color="darkgreen", label="Jerk")
    plt.plot(s, j, "x", color="purple")
    plt.plot(s[0], j[0], "x", color="orange")
    plt.plot(s[-1], j[-1], "x", color="mediumturquoise")
    plt.plot(s_spl3, j_spl, "--", color="b", label="Old Interpolated Jerk", linewidth=0.5)
    plt.plot(s_spl3, j_spl_new, color="lime", label="New Interpolated Jerk")
    plt.hlines(y=j_max, xmin=0, xmax=max(s), color="red", linestyles="--", label="Max Comfort Jerk",
               linewidth=0.5)
    plt.hlines(y=-j_max, xmin=0, xmax=max(s), color="red", linestyles="--", label="Min Comfort Jerk",
               linewidth=0.5)
    plt.title("Jerk Profile (Jerk vs Path Distance)")
    plt.xlabel("Path Distance s [cells]")
    plt.ylabel("Jerk j [cells/s^3]")
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(which='both')
    plt.grid(which='minor', alpha=0.25)
    plt.grid(which='major', alpha=0.5)
    plt.ylim([1.1 * -j_max, 1.1 * j_max])

    print("\nDisplaying results... \n> Program Execution Time: %s seconds" % (time() - start_time))
    plt.show()


if __name__ == '__main__':
    start_time = time()
    main()
