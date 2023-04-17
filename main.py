import operator
from collections import deque

import networkx as nx
import numpy as np


def construct_graph(graph_data):
    """
  Code for constructing the graph
  Used the dataset provided
  """
    graph_data = open("data.txt", 'r')
    G = nx.Graph()
    for line in graph_data.readlines():
        line = line.strip().split(" ")
        G.add_edge(int(line[0]), int(line[1]))
    return G


def between_centrality(graph, num):
    """
  Here is your code for calculating the node between centralities
  The betweennes centrality for node x
  is the probability that a shortest path passes through x
  1) Go through all the nodes
  2) Implement BFS, forward step
  3) Perform the backward step

  """
    nodes = graph
    # Dict for the end results, in order to keep track
    results = dict.fromkeys(G, 0.0)

    for root in nodes:
        predecessors, sigma, visited = breadth_first_search(graph, root)
        # Dict for the delta, used for the backwards step, all 0 initially
        delta = dict.fromkeys(visited, 0.0)

        while visited:
            # Going through the list, Last In First Out
            node = visited.pop(-1)

            # Used for the computation of the Brandes algorithm
            coefficient = (1 + delta[node]) / sigma[node]

            # Going through all the predecessors
            for pred in predecessors[node]:
                # Formula from the algorithm
                delta[pred] += (sigma[pred] * coefficient)

            if node != root:
                results[node] += delta[node]

    sorted_results = sort_dict(results, num)
    write_to_file(sorted_results)
    return sorted_results


def sort_dict(results, num):
    "helping method to sort list"
    sorted_bc = dict(reversed(sorted(results.items(), key=lambda c: c[1])))
    return list(sorted_bc.keys())[:num]


def write_to_file(results, value="w"):
    f = open("47888249.txt", value)
    if value == "a":
        # Appending the file
        f.write("\n")
    for node in results:
        f.write(str(node) + " ")
    f.close()
    print("Results written to file! :) good job!")


def pagerank_centrality(graph, aplha, beta):
    a = adjacency_matrix(graph)
    d = inverse_degree_matrix(graph, a)

    count = 0


def adjacency_matrix(graph):
    """
    Function for creating the adjacency matrix.
    :param graph: from the dataset
    :return: Adjacency matrix
    """
    # n is the dimension of the matrix

    n = len(graph)
    # sorted nodes
    nodes = sorted(list(graph.nodes()))

    # Initliaze the matrix with zeros, n x n matrix
    a = {}
    for node in graph:
        a[node] = [0 for i in range(len(nodes))]
        for neighbor in graph.adj[node]:
            a[node][neighbor] = 1
    print(a)
    return a


def inverse_degree_matrix(graph, adjacency_matrix):
    """
    Function for creating the degree matrix
    :param graph: from the dataset
    :return: Degree matrix
    """
    D = dict(graph.degree)
    print(D)
    for node in D.keys():
        D[node] = 1 / D[node]
    print(D)

    return D


def breadth_first_search(graph, root):
    """
    Method for the forward step which is BFS
    Return: sigma, predecessors and visited nodes
    """
    # Creates a dictionary for predecessors so we easily can look this up
    predecessors = {}

    for node in graph:
        # initialize empty list so every entry has a predecessor list
        predecessors[node] = []

    # Dict for the distances, all 0 initially
    distances = {}

    # Dict for the sigma, all 0 initially
    sigma = dict.fromkeys(graph, 0.0)

    # Creates a list for the visited nodes so we have an overview
    visited = []

    # Set the distance from the root to root 0
    distances[root] = 0.0
    sigma[root] = 1.0

    queue = deque([root])

    while len(queue) > 0:

        # Could also use queue.pop(0) but this is faster:
        node = queue.popleft()

        # Adds the node to the visited list since we have now been here
        visited.append(node)

        for neighbor in graph[node]:

            if neighbor not in distances:  # not visited node
                queue.append(neighbor)

                # Computing the distance for the neighbor node
                distances[neighbor] = 1 + distances[node]

            # Check if it is a predecessor, this we need to keep track of in order to go backwards
            if distances[neighbor] == distances[node] + 1:
                # The neighbor comes right after the node which makes the node a predecessor of the neighbor
                predecessors[neighbor].append(node)
                sigma[neighbor] += sigma[node]

    return predecessors, sigma, visited


def main():
    pass


if __name__ == "__main__":
    #G = construct_graph('data.txt')
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4])

    G.add_edge(0, 1)
    G.add_edge(0, 3)
    G.add_edge(0, 4)
    G.add_edge(1, 2)
    G.add_edge(1, 4)
    G.add_edge(2, 3)
    G.add_edge(2, 4)
    # print(G)
    graph = {
        '5': ['3', '7'],
        '3': ['2', '4'],
        '7': ['8'],
        '2': [],
        '4': ['8'],
        '8': []
    }

    # between_centrality(G, 10)
    pagerank_centrality(G, 0.5, 0.5)
