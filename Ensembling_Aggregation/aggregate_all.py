import json
import os
import csv

from ensemble_aggregator import aggregator_scores
from ensemble_aggregator import aggregator_coordinates
from ensemble_aggregator import aggregator_coordinates_normalization

folder_path = "../Simulation/Logs/"

def extract_data(folder_name, writer):
    try:
        path = folder_path + folder_name
        json_file_names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.json')]
        print(folder_name)
        sorted_json_file_names = sorted(json_file_names)
        nodes_data = []

        for json_file_name in sorted_json_file_names:
            with open(path + "/" + json_file_name, "r") as json_file:
                nodes_data.append(json.load(json_file))

        n_nodes = len(nodes_data)
        n_tests = len(nodes_data[0]["pipeline_iterations"][0]["test_data"])
        n_neighbors = len(nodes_data[0]["pipeline_iterations"][0]["test_data"][0]["neighbors"])

        centroids = []
        correctly_classified_before = []
        accuracies_before = []
        centroids_to_swap = []
        counter = 0
        for node in nodes_data:
            first_iteration = node["pipeline_iterations"][0]
            centroids.append(first_iteration["final_centroids"])
            correctly_classified_before.append(first_iteration["correctly_classified_samples"])
            accuracies_before.append(first_iteration["accuracy"])
            centroids_to_swap.append(first_iteration["CENTROIDS_SWAPPED"])
            if centroids_to_swap[counter]:
                centroids[counter][0], centroids[counter][1] = centroids[counter][1], centroids[counter][0]
            counter += 1


        neighbors = [[[None] * n_neighbors for _ in range(n_nodes)] for _ in range(n_tests)]
        scores = [[[None] * n_neighbors for _ in range(n_nodes)] for _ in range(n_tests)]
        neighbors_labels = [[[None] * n_neighbors for _ in range(n_nodes)] for _ in range(n_tests)]

        correctly_classified_scores = 0
        test_coordinates = []

        for test in range(n_tests):
            test_coordinates.append(nodes_data[0]["pipeline_iterations"][0]["test_data"][test]["test_coordinates"])
            for node in range(n_nodes):
                for neighbor in range(n_neighbors):
                    neighbors[test][node][neighbor] = nodes_data[node]["pipeline_iterations"][0]["test_data"][test]["neighbors"][neighbor]["coordinates"]
                    scores[test][node][neighbor] = nodes_data[node]["pipeline_iterations"][0]["test_data"][test]["neighbors"][neighbor]["score"]
                    if centroids_to_swap[node]:
                        neighbors_labels[test][node][neighbor] = 1 - nodes_data[node]["pipeline_iterations"][0]["test_data"][test]["neighbors"][neighbor]["label"]
                    else:
                        neighbors_labels[test][node][neighbor] = nodes_data[node]["pipeline_iterations"][0]["test_data"][test]["neighbors"][neighbor]["label"]

            correctly_classified_scores += int(aggregator_scores(n_nodes, n_neighbors, neighbors[test], test_coordinates, test, scores[test], neighbors_labels[test]))
        accuracy_score = correctly_classified_scores / n_tests

        correctly_classified_coordinates_test, correctly_classified_coordinates_centroids = aggregator_coordinates(n_nodes, centroids, n_neighbors, neighbors, test_coordinates, scores, neighbors_labels)
        accuracy_coordinates_test = correctly_classified_coordinates_test / n_tests
        accuracy_coordinates_centroids = correctly_classified_coordinates_centroids / n_tests

        correctly_classified_coordinates_normalization_test, correctly_classified_coordinates_normalization_centroids = aggregator_coordinates_normalization(n_nodes, centroids, n_neighbors, neighbors, test_coordinates, scores, neighbors_labels)
        accuracy_coordinates_normalization_test = correctly_classified_coordinates_normalization_test / n_tests
        accuracy_coordinates_normalization_centroids = correctly_classified_coordinates_normalization_centroids / n_tests


        # Print on CSV the folder name, the settings (1 per column), and the accuracies
        writer.writerow([folder_name] + folder_name.split('_')[1:] +
                        [accuracies_before, accuracy_score, accuracy_coordinates_test,accuracy_coordinates_centroids,
                        accuracy_coordinates_normalization_test,accuracy_coordinates_normalization_centroids])
    except Exception as e:
        print(f"Error processing folder {folder_name}: {e}")
        return


if __name__ == "__main__":
    output = open("algorithms_comparison.csv", "w", newline="")
    writer = csv.writer(output)
    writer.writerow(["Folder_name","N_NODES","K_NEIGHBOR","MEMORY_SIZE","CONFIDENCE","CONFIDENCE_THR","FILTER",
                     "ONE_SHOT","INITIAL_THR","UPDATE_THR","K","ITERATION","N_TRAIN","N_TRAIN_USED","N_TEST",
                     "N_TEST_USED","original","accuracy_scores","accuracy_coordinates_test",
                     "accuracy_coordinates_centroids","accuracy_coordinates_normalization_test","accuracy_coordinates_normalization_centroids"])
    # Get all the subdirs
    for subfolder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder_name)
        if os.path.isdir(subfolder_path):
            extract_data(subfolder_name, writer)
