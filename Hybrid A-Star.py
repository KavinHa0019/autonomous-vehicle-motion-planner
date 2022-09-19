# Source code originally written by jvirdi2
# Source code modified by Kavin Ha

import heapq as hq
import math
import matplotlib.pyplot as plt
import numpy as np
import random

# total cost f(n) = actual cost g(n) + heuristic cost h(n)


class HybridAStar:
    def __init__(self, x_min, x_max, y_min, y_max, obstacles, resolution, l_f, l_r):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.obstacles = obstacles
        self.resolution = resolution
        self.l_f = l_f
        self.l_r = l_r


    def h(self, position, target):
        output = np.sqrt(((position[0] - target[0]) ** 2) + ((position[1] - target[1]) ** 2) + (
                    math.radians(position[2]) - math.radians(target[2])) ** 2)
        return float(output)


    """
    For each node n, we need to store:
    (discrete_x, discrete_y, heading angle theta),
    (continuous x, continuous y, heading angle theta)
    cost g, f,
    path [(continuous x, continuous y, continuous theta),...]
    start: discrete (x, y, theta)
    end: discrete (x, y, theta)
    sol_path = [(x1,y1,theta1),(x2,y2,theta2), ...]
    """

    def find_path(self, start, end, l_f, l_r, dt):
        v = [-1, 1]
        v_cost = [1, 0]

        delta = [-40, 0, 40]
        delta_cost = [0.1, 0, 0.1]

        start = (float(start[0]), float(start[1]), float(start[2]))
        end = (float(end[0]), float(end[1]), float(end[2]))
        # The above 2 are in discrete coordinates

        open_heap = []  # element of this list is like (cost,node_d)
        open_dict = {}  # element of this is like node_d:(cost,node_c,(parent_d,parent_c))

        closed_dict = {}  # element of this is like node_d:(cost,node_c,(parent_d,parent_c))

        obstacles = set(self.obstacles)
        cost_to_neighbour_from_start = 0

        # Here a heap is chosen. (cost, path) is a tuple which is pushed in
        # open_set_sorted. The main advantage is that as more (cost, path) are
        # added to this open_set, heap automatically sorts it and this first
        # element is automatically the lowest cost one
        # Here path is [(),()....] where each () has (discrete,continuous) for a node
        # for path normal appending is done. If you use heap there, the elements
        # get sorted and we don't want that. We want to preserve the order in
        # which we move for start to destination node

        hq.heappush(open_heap, (cost_to_neighbour_from_start + self.h(start, end), start))

        open_dict[start] = (cost_to_neighbour_from_start + self.h(start, end), start, (start, start))

        while len(open_heap) > 0:

            # choose the node that has minimum total cost for exploration
            # print(open_set_sorted)

            chosen_d_node = open_heap[0][1]
            chosen_node_total_cost = open_heap[0][0]
            chosen_c_node = open_dict[chosen_d_node][1]

            closed_dict[chosen_d_node] = open_dict[chosen_d_node]

            # print(self.euc_dist(chosen_path_last_element[0],end))

            if self.h(chosen_d_node, end) < 1:

                rev_final_path = [end]  # reverse of final path
                node = chosen_d_node
                m = 1
                while m == 1:
                    open_node_contents = closed_dict[node]  # (cost,node_c,(parent_d,parent_c))
                    parent_of_node = open_node_contents[2][1]

                    rev_final_path.append(parent_of_node)
                    node = open_node_contents[2][0]
                    if node == start:
                        rev_final_path.append(start)
                        break

                print(rev_final_path[::-1])
                return rev_final_path[::-1]

            hq.heappop(open_heap)

            for i in range(len(v)):
                for j in range(len(delta)):
                    cost_to_neighbour_from_start = chosen_node_total_cost - self.h(chosen_d_node, end)

                    beta = np.rad2deg(np.arctan(l_r/(l_f + l_r) * np.tan(np.deg2rad(delta[j]))))
                    neighbour_x_cts = chosen_c_node[0] + v[i] * np.cos(np.deg2rad(chosen_c_node[2] + beta)) * dt
                    neighbour_y_cts = chosen_c_node[1] + v[i] * np.sin(np.deg2rad(chosen_c_node[2] + beta)) * dt
                    neighbour_theta_cts = np.rad2deg(np.deg2rad(chosen_c_node[2]) +
                                                     v[i]/l_r * np.sin(np.deg2rad(beta)) * dt) % 360

                    #print(neighbour_x_cts, neighbour_y_cts, neighbour_theta_cts)

                    neighbour_x_d = round(neighbour_x_cts)
                    neighbour_y_d = round(neighbour_y_cts)
                    neighbour_theta_d = round(neighbour_theta_cts)

                    neighbour = ((neighbour_x_d, neighbour_y_d, neighbour_theta_d),
                                 (neighbour_x_cts, neighbour_y_cts, neighbour_theta_cts))

                    if (((neighbour_x_d, neighbour_y_d) not in obstacles) and
                            (neighbour_x_d >= self.x_min) and (neighbour_x_d <= self.x_max) and
                            (neighbour_y_d >= self.y_min) and (neighbour_y_d <= self.y_max)):

                        heurestic = self.h((neighbour_x_d, neighbour_y_d, neighbour_theta_d), end)
                        cost_to_neighbour_from_start = abs(v[i]) + cost_to_neighbour_from_start + \
                                                       v_cost[i] + delta_cost[j]

                        # print(heurestic,cost_to_neighbour_from_start)
                        total_cost = heurestic + cost_to_neighbour_from_start

                        # If the cost of going to this successor happens to be more
                        # than an already existing path in the open list to this successor,
                        # skip this successor

                        skip = 0
                        # print(open_set_sorted)
                        # If the cost of going to this successor happens to be more
                        # than an already existing path in the open list to this successor,
                        # skip this successor
                        found_lower_cost_path_in_open = 0

                        if neighbour[0] in open_dict:

                            if total_cost > open_dict[neighbour[0]][0]:
                                skip = 1

                            elif neighbour[0] in closed_dict:

                                if total_cost > closed_dict[neighbour[0]][0]:
                                    found_lower_cost_path_in_open = 1

                        if skip == 0 and found_lower_cost_path_in_open == 0:
                            hq.heappush(open_heap, (total_cost, neighbour[0]))
                            open_dict[neighbour[0]] = (total_cost, neighbour[1], (chosen_d_node, chosen_c_node))
            # a=a+1
            # print(open_set_sorted)
        print("Did not find the goal - it's unattainable.")
        return []


def main():
    print(__file__ + " start!!")

    # start and goal position
    # (x, y, theta) in meters, meters, degrees
    x_start, y_start, theta_start = -2, -15, 0
    x_goal, y_goal, theta_goal = 15, 4, 180

    # create obstacles
    obstacles = []

    for i in range(10):
        obstacles.extend(((0, i), (0, -i), (i, 0), (-i, 0), (i, i), (i, -i), (-i, -i), (-i, i)))

    x_obs, y_obs = [], []
    for (x, y) in obstacles:
        x_obs.append(x)
        y_obs.append(y)

    plt.plot(x_obs, y_obs, ".k")
    plt.plot(x_start, y_start, "xr")
    plt.plot(x_goal, y_goal, "xb")
    plt.grid(True)
    plt.axis("equal")

    hybrid_a_star = HybridAStar(x_min=-20, x_max=20, y_min=-20, y_max=20, obstacles=obstacles, resolution=1, l_f=2, l_r=2.5)
    path = hybrid_a_star.find_path((x_start, y_start, theta_start), (x_goal, y_goal, theta_goal), l_f=2, l_r=2.5, dt=1)

    x_path, y_path = [], []
    for node in path:
        x_path.append(node[0])
        y_path.append(node[1])

    plt.plot(x_path, y_path, "-r")
    plt.show()


if __name__ == '__main__':
    main()
