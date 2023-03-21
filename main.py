import networkx as nx
import numpy as np



def construct_graph(graph_data):
  """
  Code for constructing the graph
  """
  graph_data = open("data.txt", 'r')
  G = nx.Graph()
  for line in graph_data.readlines():
    line = line.strip().split(" ")
    G.add_edge(int(line[0]), int(line[1]))
  return G

def between_centrality(graph, top):
  """
  Here is your code for calculating the node between centralities
  The betweennes centrality for node x
  is the probability that a shortest path passes through x
  TODO: Implement the Brandes Algorithm
  1) Go through all the nodes
  2) Implement BFS, first step
  3) Go through all nodes calculate betweenness centrality
  4) Normalize results
  """
  nodes = graph
  betweenness = dict.fromkeys(graph, 0.0)
  for source in nodes:
    BFS(graph, source)



def pagerank_centrality(graph):
  """
  Here is your main function
  """
  pass

def BFS(graph, source):
  """
  TODO: write code for BFS
  :param graph:
  :return:
  """
def normalize(graph):
  """
  TODO: code for normalization
  :param graph:
  :return:
  """
if __name__ == "__main__":
  G = construct_graph('data.txt')
  print(G)
  between_centrality(G, 10)

  """
  Here you callback you code to load data, and generate the results
  make sure to run the code via a clikc 
  """