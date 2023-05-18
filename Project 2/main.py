"""
Project 2 Social Media Analytics
Cecilie Knudsen 4788249
Link Prediction
"""
import networkx as nx


def construct_graph(file=None, list =None):
    """
    Function for constructing the graph
    Used the dataset provided
    :param graph_data: txt file
    :return: The graph
    """
    G = nx.Graph()
    # Based on if the input is a list or a file we construct the graph
    if list:
        G.add_edges_from(list)
    elif file:
        file = open(file, 'r')
        for line in file.readlines():
            line = line.strip().split(" ")
            G.add_edge(int(line[0]), int(line[1]))
    return G

def get_neighbors(graph):
    """
    Finding the neighbors
    Necessary for the neighborhood-methods
    """
    neighbors_dict = dict()
    # Method for finding all the neighbors
    for node in graph:
        neighbors_dict[node] = set(graph[node])
    return neighbors_dict

def compute_jaccard():
    return

def read_training_data(file):
    data = []


