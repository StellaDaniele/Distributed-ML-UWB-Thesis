import json
import os

from ensemble_aggregator import aggregator

folder_path = "../Others/prova/"

def extract_data(folder_name):
    path = folder_path+folder_name
    json_file_names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.json')]

    sorted_json_file_names = sorted(json_file_names)
    nodes_data = []

    for json_file_name in sorted_json_file_names:
        json_file = open(path+"/"+json_file_name, "r")
        nodes_data.append(json.load(json_file))
        json_file.close()
        print(json_file_name)

    n_nodes = len(nodes_data)

    centroids = []
    correctly_classified_before = []
    accuracies_before = []

    for node in nodes_data:
        first_iteration = node["pipeline_iterations"][0]
        centroids.append(first_iteration["final_centroids"])
        correctly_classified_before.append(first_iteration["correctly_classified_samples"])
        accuracies_before.append(first_iteration["accuracy"])

    n_tests = len(nodes_data[0]["pipeline_iterations"][0]["test_data"])
    n_neighbors = len(nodes_data[0]["pipeline_iterations"][0]["test_data"][0]["neighbors"])
    neighbors = [[None] * n_neighbors for _ in range(len(nodes_data))]
    scores = [[None] * n_neighbors for _ in range(len(nodes_data))]
    correctly_classified = 0

    for test in range(n_tests):
        test_coordinates = nodes_data[0]["pipeline_iterations"][0]["test_data"][test]["test_coordinates"]
        #print(test_coordinates)
        for node in range(n_nodes):
            for neighbor in range(n_neighbors):
                neighbors[node][neighbor] = nodes_data[0]["pipeline_iterations"][0]["test_data"][test]["neighbors"][neighbor]["coordinates"]
                scores[node][neighbor] = nodes_data[0]["pipeline_iterations"][0]["test_data"][test]["neighbors"][neighbor]["score"]
        correctly_classified += int(aggregator(n_nodes, centroids, n_neighbors, neighbors, test, test_coordinates, scores))

    accuracy = correctly_classified/n_tests

    print("Before aggregation:")
    print("\tCorrectly classified = "+str(correctly_classified_before))
    print("\tAccuracy="+str(accuracies_before))
    print("After aggregation:")
    print("\tCorrectly classified = "+str(correctly_classified))
    print("\tAccuracy="+str(accuracy))


    print()


if __name__ == "__main__":
    # Get all the subdirs
    for subfolder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder_name)
        if os.path.isdir(subfolder_path):
            extract_data(subfolder_name)
