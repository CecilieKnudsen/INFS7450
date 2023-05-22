"""
Project 2 Social Media Analytics
Cecilie Knudsen 47888249
Link Prediction
"""
import networkx as nx


def construct_graph(file):
    """
    Function for constructing the graph
    Used the dataset provided
    :param graph_data: txt file
    :return: The graph
    """
    G = nx.Graph()
    # Construct the graph from the given file
    file = open(file, 'r')
    for line in file.readlines():
        line = line.strip().split(" ")
        G.add_edge(line[0], line[1])
    return G


def validation_set_pairs(positive_file, negative_file):
    """
    Method for creating the validation pairs that are
    used for the training of the algorithm
    :param positive_file: positive edges (exist)
    :param negative_file: negative edges (does not exist)
    :return: list of the pairs
    """
    data_set = []
    with open(positive_file, 'r') as file:
        for line in file.readlines():
            line = line.strip().split(" ")
            data_set.append((line[0], line[1]))
    with open(negative_file, 'r') as file:
        for line in file.readlines():
            line = line.strip().split(" ")
            data_set.append((line[0], line[1]))
    return data_set


def set_pairs(file):
    """
    Method for creating pairs
    Used for the test data
    :param file: file we want to pair
    :return: list of the pairs
    """
    data_set = []
    with open(file, 'r') as file:
        for line in file.readlines():
            line = line.strip().split(" ")
            data_set.append((line[0], line[1]))
    return data_set


def get_neighbors(graph):
    """
    Finding the neighbors
    Necessary for the neighborhood-methods
    :param graph: graph we want pairs from
    :return: dictonary with pairs
    """
    neighbors_dict = dict()
    # Method for finding all the neighbors
    for node in graph:
        neighbors_dict[node] = set(graph[node])
    return neighbors_dict


def reverse_pair(pair):
    """
    The pair needs to be reversed because it is the same
    the opposite way as well
    :param pair: pair we want to reverse
    :return: the reversed pair
    """
    list_pair = pair[0].split(" & ")
    return list_pair[1] + " & " + list_pair[0]


def compute_jaccard(node_pairs, neighbors):
    """
    Method for computing the Jaccard Similiarity

    :param node_pairs: Pairs of nodes to compute
    :param neighbors: The dictonary of neighbors from graph
    :return: Dictonary of jaccard scores
    """

    score_dict = dict()
    for pair in node_pairs:
        author1 = pair[0]
        author2 = pair[1]
        score = len(neighbors[author1].intersection(neighbors[author2]))
        score = float(score) / len(neighbors[author1].union(neighbors[author2]))
        score_dict[str(author1) + " & " + str(author2)] = score

    # Checking how many values over zero we have
    values_over_zero = 0
    for values in score_dict.values():
        if values > 0.0:
            values_over_zero += 1
    print("The number of score over zero: " + str(values_over_zero))
    return score_dict


def jaccard_with_library(G, node_pairs):
    """
    Computing Jaccard Similiarty utilizing
    the NetworkX library
    :param G: Graph to compute for
    :param node_pairs: node pairs used
    :return: The dictonary of scores
    """
    score_dict = dict()
    jaccard = nx.jaccard_coefficient(G, node_pairs)
    for u, v, p in jaccard:
        score_dict[u + " & " + v] = p
    return score_dict


def compute_adamic_adar(G, edges):
    """
    Computing Adamic Adar utilizing
    the NetworkX library
    :param G: Graph to compute for
    :param edges: The relevant edges
    :return: The dictonary of scores
    """
    score_dict = dict()
    adamic_adar = nx.adamic_adar_index(G, edges)
    for u, v, p in adamic_adar:
        score_dict[u + " & " + v] = p
    return score_dict


def compute_preferential_attachement(G, edges):
    """
        Computing Preferential Attachment utilizing
        the NetworkX library
        :param G: Graph to compute for
        :param edges: The relevant edges
        :return: The dictonary of scores
    """
    score_dict = dict()
    preferential_attachement = nx.preferential_attachment(G, edges)
    for u, v, p in preferential_attachement:
        score_dict[u + " & " + v] = p
    return score_dict


def compute_resource_allocation_index(G, edges):
    """
        Computing Resource Allocation Index utilizing
        the NetworkX library
        :param G: Graph to compute for
        :param edges: The relevant edges
        :return: The dictonary of scores
        """
    score_dict = dict()
    resource_allocation_index = nx.resource_allocation_index(G, edges)
    for u, v, p in resource_allocation_index:
        score_dict[u + " & " + v] = p
    return score_dict


def compute_common_neighbor_centrality(G, edges, alpha=0.8):
    """
        Computing Common Neighbor Centrality utilizing
        the NetworkX library
        :param G: Graph to compute for
        :param edges: The relevant edges
        :param alpha: choosen Alpha value
        :return: The dictonary of scores
        """
    score_dict = dict()
    common_neighbor_centrality = nx.common_neighbor_centrality(G, edges, alpha)
    for u, v, p in common_neighbor_centrality:
        score_dict[u + " & " + v] = p
    return score_dict


def calculate_score(results, positive, measure_method):
    """
    Method for computing the proximity score
    used to evaluate which algorithm is the
    best for link prediction
    :param results: Results to calculate for
    :param positive: The positive edges
    :param measure_method: Which algorithm used
    :return: The proximity score
    """
    score = 0
    positive_pairs = dict.fromkeys(pair[0] + " & " + str(pair[1]) for pair in positive.edges())
    for pair in results:
        reverse_the_pair = reverse_pair(pair)
        if pair[0] in positive_pairs or reverse_the_pair in positive_pairs:
            score += 1
    accuracy_score = (score / len(positive_pairs)) * 100
    print("This is the accuracy score for %s: %f" % (measure_method, accuracy_score) + "%")
    return accuracy_score


def sort_dict(results, num):
    """
    Function to sort the top results
    :param results: needs sorting
    :param num: top values needed
    :return: The extracted top values in a list
    """
    return sorted(results.items(), key=lambda c: c[1], reverse=True)[:num]


def format_fix(list):
    """
    Method for fixing the format
    so it is in the desired way
    for the results file
    :param list: Result list written to file
    :return: List in the wanted format
    """
    results = []
    for pair in list:
        results.append(pair[0].replace(" &", "") + "\n")
    return results


def write_to_file(results, measure, value="w"):
    """
    Function to write to file
    :param results: written to file
    :param value: write or append, default write
    :return: none
    """
    f = open("results.txt", value)
    f.writelines(results)
    f.close()
    print("The results is written to file with measurement %s, good job! " % (measure))


def main():
    G_training = construct_graph("training.txt")
    G_positive = construct_graph("val_positive.txt")
    test_data = set_pairs("test.txt")

    neighbors = get_neighbors(G_training)
    validation_pairs= validation_set_pairs("val_positive.txt", "val_negative.txt")

    # Computing the Jaccard Similarity
    jaccard = compute_jaccard(validation_pairs, neighbors)
    top_100_values = sort_dict(jaccard, 100)
    jaccard_score = calculate_score(top_100_values, G_positive, "Jaccard Similarity")

    # Computing the Jaccard simialirty using networkx
    jaccard_nx = jaccard_with_library(G_training, validation_pairs)
    top_100_jaccard_nx = sort_dict(jaccard_nx, 100)
    jaccard_score_nx = calculate_score(top_100_jaccard_nx, G_positive, "Jaccard Similarity Networkx")

    # Computing Adamic Adar
    adamic_adar = compute_adamic_adar(G_training, validation_pairs)
    top_100_adamic = sort_dict(adamic_adar, 100)
    adamic_score = calculate_score(top_100_adamic, G_positive, "Adamic Adar")

    # Computing Preferential Attachement
    preferential_attachement = compute_preferential_attachement(G_training, validation_pairs)
    top_100_pa = sort_dict(preferential_attachement, 100)
    pa_score = calculate_score(top_100_pa, G_positive, "Preferential Attachement")

    # Computing Resource Allocation Index
    resource_allocation_index = compute_resource_allocation_index(G_training, validation_pairs)
    top_100_rai = sort_dict(resource_allocation_index, 100)
    rai_score = calculate_score(top_100_rai, G_positive, "Resource Allocation Index")

    # Computing Common Neighbor Centrality
    common_neigbor_centrality = compute_resource_allocation_index(G_training, validation_pairs)
    top_100_cmc = sort_dict(common_neigbor_centrality, 100)
    cmc_score = calculate_score(top_100_cmc, G_positive, "Common Neighbor Centrality")

    # TEST Set, from the results, Resource Allocation and Common Neighbor Centrality gave the best score

    resource_allocation_index_test = compute_resource_allocation_index(G_training, test_data)
    top_100_test = sort_dict(resource_allocation_index_test, 100)
    final_results = format_fix(top_100_test)
    write_to_file(final_results, "Resource Allocation Index")


if __name__ == '__main__':
    main()
