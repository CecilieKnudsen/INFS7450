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




def validation_set_pairs(positive_file, negative_file):

    data_set = []
    labels = []
    with open(positive_file, 'r') as file:
        for line in file.readlines():
            line = line.strip().split(" ")
            data_set.append((line[0], line[1]))
            labels.append(1)
    with open(negative_file, 'r') as file:
        for line in file.readlines():
            line = line.strip().split(" ")
            data_set.append((line[0], line[1]))
            labels.append(0)
    # print("This is the data set: " + str(data_set))
    # print("These are the labels: " + str(labels))
    return data_set, labels

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

def compute_jaccard(node_pairs, neighbors):
    """
    Formula for computing the Jaccard similiarity
    :param node1:
    :param node2:
    :param neighbors:
    :return:
    """

    score_dict = dict()
    for pair in node_pairs:
        author1 = int(pair[0])
        author2 = int(pair[1])

        score = len(neighbors[author1].intersection(neighbors[author2]))
        score = float(score)/len(neighbors[author1].union(neighbors[author2]))
        score_dict[str(author1) + " and " + str(author2)] = score

    # Checking how many values over zero we have
    values_over_zero = 0
    for values in score_dict.values():
        if values >0:
            values_over_zero+=1
    print("This is the Jaccard score: " + str(score_dict))
    print("The number of score over zero: " + str(values_over_zero))
    return score_dict

def sort_dict(results, num):
    """
    Function to sort the top results
    :param results: needs sorting
    :param num: top values
    :return: The extrracted top values
    """
    sorted_bc = dict(reversed(sorted(results.items(), key=lambda c: c[1])))
    return list(sorted_bc.keys())[:num]



def main():
    G_training = construct_graph("training.txt")
    neighbors = get_neighbors(G_training)
    validation_pairs, _ = validation_set_pairs(
        "val_positive.txt", "val_negative.txt")

    # Computing the jaccard similarity
    jaccard_score = compute_jaccard(validation_pairs, neighbors)
    top_100_values = sort_dict(jaccard_score, 100)
    print("The top 100 pairs: " + str(top_100_values))





if __name__ == '__main__':
    main()