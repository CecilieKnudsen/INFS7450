import operator
from collections import deque

import networkx as nx
import numpy as np


def construct_graph(graph_data):
    """
    Function for constructing the graph
    Used the dataset provided
    :param graph_data: txt file
    :return: The graph
    """
    graph_data = open("data.txt", 'r')
    G = nx.Graph()
    for line in graph_data.readlines():
        line = line.strip().split(" ")
        G.add_edge(int(line[0]), int(line[1]))
    return G


def between_centrality(graph, num):
    """
  Function for calculating the node between centralities
  The betweennes centrality for node x
  is the probability that a shortest path passes through x
  1) Go through all the nodes
  2) Implement BFS, forward step
  3) Perform the backward step
  :param: graph: from the dataset
   :param: num: number of nodes wanted
  :return: Top 10 nodes

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
    """
    Function to sort the top results
    :param results: needs sorting
    :param num: top values
    :return: The extrracted top values
    """
    sorted_bc = dict(reversed(sorted(results.items(), key=lambda c: c[1])))
    return list(sorted_bc.keys())[:num]


def write_to_file(results, value="w"):
    """
    Function to write to file
    :param results: written to file
    :param value: write or append
    :return: none
    """
    f = open("47888249.txt", value)
    if value == "a":
        # Appending the file
        f.write("\n")
    for node in results:
        f.write(str(node) + " ")
    f.close()
    print("Results written to file! :) good job!")



def pagerank_centrality(graph, alpha, beta, eps=1e-6):
    """
    Function for computing the pagerank values
    :param graph: from the dataset
    :param alpha: given in the task
    :param beta: given in the task
    :param eps: epsilon value
    :return: Top 10 nodes
    """
    a = adjacency_matrix(graph)
    d = inverse_degree_matrix(graph)


    count = 0
    n = len(a)
    # The start vector, all PageRank values are 1 initally
    # This is the "C" in the power iteration formula
    # PR = dict.fromkeys(a, 1.0)
    print(n)
    PR = dict.fromkeys(a, 1.0/n)

    # Printing out for checking
    print("This is the start PageRrank: " + str(PR))
    print("This is the start A: " + str(a))
    print("This is the start D: " + str(d))

    # Power iteration method is used for computing PageRank
    # Error value first set to 1 just for initialisation

    err = 1
    while err > n * eps: # Condition where we need to break

        # A print to check the first values
        if count < 10:
            count = count + 1
            print("Iteration {}: the value of the pagerank: {}".format(count, PR))
        # Exceeding the number of iterations that is sufficient
        # We then break the while loop
        if count == 100:
            break

        # Setting the PR to the PR it was previously
        prev_PR = PR
        PR = dict.fromkeys(prev_PR, 0)

        # Going through each element in the PageRank vector
        # In order to complete the matrix multiplication
        for element in PR:

            # Going through each row in the adjacency matrix
            for value in a[element]:

                # Multiplication with the inverse degree matrix as well
                # Setting the new PR value
                # The inverse degree multiplication only have values along the diagonal
                PR[value] += (alpha*d[element]*prev_PR[element])
            # Beta is added to the final value
            PR[element] += beta

        # New error value is computed
        # This needs to be checked in order to know if we need to break
        err = (np.linalg.norm([PR[n] - prev_PR[n] for n in PR]))


        if err < n * eps:
            # Sorting the top 10 values
            Old_PR = PR
            PR = sort_dict(PR, 10)
            print("This is the top 10: " + str(PR))
            write_to_file(PR, "a")
            return PR

    print("No convergence happened in the 100 iterations")
    return PR


def adjacency_matrix(graph):
    """
    Function for creating the adjacency matrix.
    :param graph: from the dataset
    :return: Adjacency matrix
    """
    a = nx.to_dict_of_dicts(graph)
    print(a)
    return a


def inverse_degree_matrix(graph):
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


"""

    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4])

    G.add_edge(0, 1)
    G.add_edge(0, 3)
    G.add_edge(0, 4)
    G.add_edge(1, 2)
    G.add_edge(1, 4)
    G.add_edge(2, 3)
    G.add_edge(2, 4)

"""

if __name__ == "__main__":
    G = construct_graph('data.txt')
    between_centrality(G, 10)
    pagerank_centrality(G, 0.85, 0.15)
