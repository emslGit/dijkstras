import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from math import tan, pi, hypot, sqrt
from scipy.sparse import csr_matrix, csgraph
from scipy.spatial import cKDTree
from matplotlib.collections import LineCollection


# control_panel:
RADIUS = 0.0025  # radius to check distances against in part 3.
START_CITY = 1573  # HU: 311 DE: 1573
END_CITY = 10584  # HU 702 DE: 10584
FILE_DIR = "lecturefiles/GermanyCities.txt"
start_time = time.time()
print_time = True  # if set to True, prints time taken for every function.


def read_coordinate_file(filename):
    """
    Part 1, reads and extracts coordinates.

    :param filename: File with data to convert to coordinates.
    :return: Array with coordinates
    """
    file_one = open(filename, "r")
    coordinates_list = []
    for line in file_one:
        y, x = line.strip("{}\n").split(",")
        y_mercator = np.log(tan(pi / 4 + (pi * float(y)) / 360))
        x_mercator = (pi * float(x)) / 180  # mercator projection formula.
        coordinates_list.append([x_mercator, y_mercator])
    coordinates_array = np.array(coordinates_list)
    file_one.close()
    if print_time:
        print("Ran part 1, current time: {:7.4f} seconds".format(time.time() - start_time))
    return coordinates_array


def construct_graph_connections(coord_list, radius):
    """
    part 3, constructs connections.

    :param coord_list: Coordinates to check, [[x0, y0], [x1, y1], ... ].
    :param radius: Maximum radius between cities to be stored.
    :return: List of node-indices and distances between corresponding nodes.
    """
    distances_list = []
    cities_list = []
    for n, outer in enumerate(coord_list):
        for j in range(n + 1, len(coord_list)):
            inner = coord_list[j]
            distance = hypot(outer[0] - inner[0], outer[1] - inner[1])  # calculating distances with pythagoras.
            if distance < radius:  # ignores irrelevant radii.
                distances_list.append(distance)
                cities_list.append([n, j])
    distances_array = np.array(distances_list)  # distances between indices/nodes.
    cities_array = np.array(cities_list)  # pairs of indices, from and to.

    if print_time:
        print("Ran part 3, current time: {:7.4f} seconds".format(time.time() - start_time))
    return [cities_array, distances_array]


def construct_fast_graph_connections(coord_list, radius):
    """
    part 10, constructs connections.
    :param coord_list: Coordinates to check, [[x0, y0], [x1, y1], ... ].
    :param radius: Maximum radius between cities to be stored.
    :return: List of node-indices and distances between corresponding nodes.
    """
    tree = cKDTree(coord_list)
    tree_qbp = tree.query_ball_point(coord_list, radius)
    distances_list = []
    tree_indices = []

    for i, reach in enumerate(tree_qbp):  # fills list with cities (indices).
        for j in reach:
            if i < j:
                distance = hypot(coord_list[i][0] - coord_list[j][0], coord_list[i][1] - coord_list[j][1])
                distances_list.append(distance)
                tree_indices.append((i, j))
    distances_array = np.array(distances_list)
    cities_array = np.array(tree_indices)

    if print_time:
        print("Ran part 10, current time: {:7.4f} seconds".format(time.time() - start_time))
    return [cities_array, distances_array]


def construct_graph(indices, distances, n):
    """
    part 4, constructs sparse graph.

    :param indices: List of nodes in graph, index form.
    :param distances: List of distances between nodes (where applicable) in graph.
    :param n: Number of nodes.
    :return: CSR matrix.
    """
    d = distances
    i = indices[:, 0]
    j = indices[:, 1]
    connections_matrix = csr_matrix((d, (i, j)), shape=(n, n))
    if print_time:
        print("Ran part 4, current time: {:7.4f} seconds".format(time.time() - start_time))
    return connections_matrix


def dijkstra(connections_matrix, start_node):
    """
    pt6, computes shortest path between any given start_node and the end node defined at the control panel section.

    :param start_node: start node from which to compute the shortest path to our given end node.
    :param connections_matrix: A sparse matrix containing the node connections.
    :return: Dijkstra matrix for start node, Predecessor matrix for start node.
    """
    return csgraph.dijkstra(connections_matrix, directed=False, indices=start_node, return_predecessors=True)


def compute_path(predecessor_matrix, start_node, end_node):
    """
    pt7, outputs indices of quickest path.

    :param predecessor_matrix: Predecessor matrix.
    :param start_node: Start node, [x0, y0].
    :param end_node: End node.
    :return: List of paths between nodes.
    """
    path_list = [end_node]
    A = predecessor_matrix
    i = start_node
    j = end_node
    while A[j] >= 0:  # finds path by looping through the predecessor_matrix backwards.
        path_list.append(A[j])
        j = A[j]
    if print_time:
        print("Ran part 6 + 7, current time: {:7.4f} seconds".format(time.time() - start_time))
    return path_list[::-1]


def plot_points(coord_list, connections, path):
    """
    Part 2, 5 and 8, plots nodes/vertices.

    :param coord_list: Coordinates to plot, [[x0, y0], [x1, y1], ... ].
    :param connections: List of distances between nodes (where applicable) in graph.
    :param path: List of paths between nodes.
    :return: String: shortest distance and path from start to end node.
    """
    cities_segment = [[coord_list[node[0]], coord_list[node[1]]] for node in connections]

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#fbfbf1")
    ax.scatter(coord_list[:, 0], coord_list[:, 1], s=4, color="#e20049")  # plots nodes/cities.
    if print_time:
        print("Ran part 2, current time: {:7.4f} seconds".format(time.time() - start_time))

    pt5_segments = LineCollection(cities_segment, linewidths=0.25, colors="k", alpha=1)
    ax.add_collection(pt5_segments)
    if print_time:
        print("Ran part 5, current time: {:7.4f} seconds".format(time.time() - start_time))

    pt7_segments = LineCollection([coord_list[path]], linewidths=2, colors="#3073ff", alpha=0.75)
    ax.add_collection(pt7_segments)

    if print_time:
        print("Ran part 8, current time: {:7.4f} seconds".format(time.time() - start_time))
    plt.show()
    return "Shortest distance from nodes {} to {} is {}\nPath: {}".format(START_CITY, END_CITY,
                                                                          pt6_predecessor[0][END_CITY], path)


pt1_coordinates = read_coordinate_file(FILE_DIR)
pt3_index_distance = construct_graph_connections(pt1_coordinates, RADIUS)
pt10_index_distance = construct_fast_graph_connections(pt1_coordinates, RADIUS)
pt4_matrix = construct_graph(pt3_index_distance[0], pt3_index_distance[1], len(pt1_coordinates[:, 1]))
pt6_predecessor = dijkstra(pt4_matrix, START_CITY)
pt7_path = compute_path(pt6_predecessor[1], START_CITY, END_CITY)
print(plot_points(pt1_coordinates, pt3_index_distance[0], pt7_path))
