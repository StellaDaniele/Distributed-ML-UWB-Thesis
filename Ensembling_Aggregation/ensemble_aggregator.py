import numpy as np
from math import sqrt

y_test = [0,1,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,1,1,0,0,1,1,0,0,0,0,0,1,1,1,1,1,1,1,0,1,0,1,0,1,0,0,0,0,0,0,0,1,1,0,1,0,1,0,0,0,0,0,1,0,1,1,0,0,0,1,0,0,1,0,1,0,0,0,1,1,1,0,0,0,0,1,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,1,0,1,1]

# When the standard deviation is 0, avoid dividing by 0
def safe_divide(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

def aggregator_scores(n_nodes, n_neighbors, neighbors, test_coordinates, test_id, scores, neighbors_labels):
    all_neighbors_data = []
    for node in range(n_nodes):
        # print("Node="+str(node))
        # print("\tNeighbors:")
        for neighbor in range(n_neighbors):
            all_neighbors_data.append((neighbors[node][neighbor], scores[node][neighbor], neighbors_labels[node][neighbor]))
            # print(neighbors[node][neighbor], end="\t")
            # print(scores[node][neighbor], end="\t")
            # print(neighbors_labels[node][neighbor])
    sorted_neighbors = sorted(all_neighbors_data, key=lambda pair:pair[1], reverse=True)
    #print(sorted_neighbors)

    weighted_votes = {}
    k = min(n_neighbors, 5)
    for i in range(k):
        _, score, label = sorted_neighbors[i]
        if label not in weighted_votes:
            weighted_votes[label] = 0
        weighted_votes[label] += score
    predicted_label = max(weighted_votes, key=weighted_votes.get)

    # print("test_coordinates="+str(test_coordinates[test_id]),end="\t")
    # print("test_label="+str(y_test[test_id]))
    # print("predicted_label=\t"+str(predicted_label))
    return predicted_label == y_test[test_id]


def aggregator_coordinates(n_nodes, centroids, n_neighbors, neighbors, test_coordinates, scores, neighbors_labels):
    correctly_classified_counter_test = 0 # Using distance to the test data
    correctly_classified_counter_centroids = 0 # Using distance to the centroids
    averaged_centroids = [[sum(inner_lists[i][j] for inner_lists in centroids) / len(centroids) for j in range(len(centroids[0][0]))] for i in range(len(centroids[0]))]
    # print("centroids:")
    # print(centroids)
    # print("New centroids")
    # print(averaged_centroids)
    # print()
    for test in range(len(y_test)):
        all_neighbors_data = []
        for node in range(n_nodes):
            for neighbor in range(n_neighbors):
                all_neighbors_data.append((neighbors[test][node][neighbor]))
        # Euclidean distance neighbors
        new_neighbors_test = [] # Closest to the test data
        new_neighbors_centroids = [] # Closest to the centroids
        for neighbor in range(len(all_neighbors_data)):
            distance0 = 0
            distance1 = 0
            for i in range(len(all_neighbors_data[neighbor])):
                distance0 += (all_neighbors_data[neighbor][i] - averaged_centroids[0][i]) ** 2
                distance1 += (all_neighbors_data[neighbor][i] - averaged_centroids[1][i]) ** 2
            distance0 = sqrt(distance0)
            distance1 = sqrt(distance1)
            # print(all_neighbors_data[neighbor],end="\t")
            # print("distance0="+str(distance0), end="\t")
            # print("distance1="+str(distance1))
            distance_from_centroid = min(distance0,distance1)
            distance_from_test = 0
            for i in range(len(all_neighbors_data[neighbor])):
                distance_from_test += (all_neighbors_data[neighbor][i] - test_coordinates[test][i]) ** 2
            distance_from_test = sqrt(distance_from_test)
            new_neighbors_test.append([all_neighbors_data[neighbor],
                                 distance_from_test,
                                 1 if distance1 < distance0 else 0])
            new_neighbors_centroids.append([all_neighbors_data[neighbor],
                                 distance_from_centroid,
                                 1 if distance1 < distance0 else 0])

        sorted_neighbors_test = sorted(new_neighbors_test, key=lambda pair:pair[1], reverse=False)
        sorted_neighbors_centroids = sorted(new_neighbors_centroids, key=lambda pair:pair[1], reverse=False)
        # print("sorted_neighbors:")
        # print(sorted_neighbors)
        weighted_votes = {}
        k = min(n_neighbors, 5)
        for i in range(k):
            _, distance, label = sorted_neighbors_test[i]
            if label not in weighted_votes:
                weighted_votes[label] = 0
            weighted_votes[label] += distance
        predicted_label = min(weighted_votes, key=weighted_votes.get)
        # print(predicted_label)
        correctly_classified_counter_test += int(predicted_label == y_test[test])

        weighted_votes = {}
        k = min(n_neighbors, 5)
        for i in range(k):
            _, distance, label = sorted_neighbors_centroids[i]
            if label not in weighted_votes:
                weighted_votes[label] = 0
            weighted_votes[label] += distance
        predicted_label = min(weighted_votes, key=weighted_votes.get)
        # print(predicted_label)
        correctly_classified_counter_centroids += int(predicted_label == y_test[test])
    # print()
    return correctly_classified_counter_test,correctly_classified_counter_centroids


def aggregator_coordinates_normalization(n_nodes, centroids, n_neighbors, neighbors, tests_coordinates, scores, neighbors_labels):
    # Initialization
    np_centroids = []
    np_neighbors = []
    np_scores = []
    np_labels = []
    for node in range(n_nodes):
        np_centroids.append(np.array(centroids[node]))
        np_neighbors.append(np.array(neighbors[node]))
        np_scores.append(np.array(scores[node]))
        np_labels.append(np.array(neighbors_labels[node]))

    np_test = []
    for test_id in range(len(y_test)):
        np_test.append(np.array(tests_coordinates[test_id]))

    # Normalization and aggregation
    normalized_centroids = []
    normalized_neighbors = []

    for node in range(n_nodes):
        normalized_centroids.append(safe_divide(np_centroids[node] - np.mean(centroids[node], axis=0), np.std(np_centroids[node], axis=0)))
        #normalized_neighbors.append((np_neighbors[node] - np.mean(neighbors[node], axis=0)) / np.std(neighbors[node], axis=0))
        normalized_neighbors.append(safe_divide(np.array(neighbors[node]) - np.mean(neighbors[node], axis=0), np.std(neighbors[node], axis=0)))

    all_normalized_centroids = np.vstack(normalized_centroids)
    all_normalized_neighbors = np.vstack(normalized_neighbors)

    mean_aggregated_centroid = np.mean(all_normalized_centroids, axis=0)
    std_dev_aggregated_centroid = np.std(all_normalized_centroids, axis=0)


    normalized_tests = []
    for test_id in range(len(y_test)):
        normalized_tests.append(safe_divide(np_test[test_id] - mean_aggregated_centroid, std_dev_aggregated_centroid))
    #normalized_test_datum = safe_divide(np_test - mean_aggregated_centroid, std_dev_aggregated_centroid)

    # print("Shape of all_normalized_neighbors:", all_normalized_neighbors.shape)
    # print("Shape of labels array:", np.zeros(n_nodes).shape)

    # KNN Classification using all neighbors from all nodes
    #k = max(n_nodes*n_neighbors, 5)  # Number of neighbors for KNN
    k = min(n_nodes*n_neighbors, 5)

    # Predict the label for the normalized test datum
    predicted_labels = 0
    correctly_classified = 0
    for i in range(len(y_test)):
        if(predicted_labels[i] == y_test[i]):
            correctly_classified += 1

    return correctly_classified

def example():
    # Examples come form:
    #    2_1_10_0_0_0_CONF_1_0_0_2_50_614_307_154_154_1.exe
    #    2_1_10_307_0_0_CONF_1_0_0_2_50_614_307_154_154_2.exe

    node1_centroids = np.array([
        [123.333336, 26.5, 30.175001, 31.833334],
        [186.5, 28.5, 36.324997, 42.25]
    ])

    node2_centroids = np.array([
        [117.599998, 28.200001, 30.439999, 34.599998],
        [162.600006, 24.200001, 30.26, 32.0]
    ])

    node1_neighbors = np.array([
        [118.0, 36.0, 33.299999, 23.0]
    ])

    node2_neighbors = np.array([
        [158.0, 30.0, 35.5, 35.0]
    ])

    # Example test coordinates (first one)
    test_coordinates = np.array([
        [115.0, 21.0, 24.0, 34.0]
    ])

    # Example scores (right one is the score in the JSON)
    node1_score = 1-0.112587
    node2_score = 1-0.066117

    # Normalize the centroids for each node
    normalized_node1_centroids = (node1_centroids - np.mean(node1_centroids, axis=0)) / np.std(node1_centroids, axis=0)
    normalized_node2_centroids = (node2_centroids - np.mean(node2_centroids, axis=0)) / np.std(node2_centroids, axis=0)

    print("Normalized Centroids for Node 1:")
    print(normalized_node1_centroids)

    print("\nNormalized Centroids for Node 2:")
    print(normalized_node2_centroids)

    # Normalize the neighbor coordinates for each node
    normalized_node1_neighbors = (node1_neighbors - np.mean(node1_centroids, axis=0)) / np.std(node1_centroids, axis=0)
    normalized_node2_neighbors = (node2_neighbors - np.mean(node2_centroids, axis=0)) / np.std(node2_centroids, axis=0)

    print("\nNormalized Neighbor Coordinates for Node 1:")
    print(normalized_node1_neighbors)

    print("\nNormalized Neighbor Coordinates for Node 2:")
    print(normalized_node2_neighbors)

    # Normalize the test coordinates
    normalized_test_datum = (test_coordinates - np.mean(node1_centroids, axis=0)) / np.std(node1_centroids, axis=0)

    print("\nNormalized Test Coordinates:")
    print(normalized_test_datum)

    # Scores
    print("\nNode 1 score: " + str(node1_score))
    print("Node 2 score: " + str(node2_score))



    # Combine normalized centroids and neighbors
    all_normalized_centroids = np.vstack((normalized_node1_centroids, normalized_node2_centroids))
    all_normalized_neighbors = np.vstack((normalized_node1_neighbors, normalized_node2_neighbors))

    # Aggregated centroid
    aggregated_centroid = np.mean(all_normalized_centroids, axis=0)

    # Calculate distances for each node's neighbors and the test datum
    distances_node1 = np.linalg.norm(all_normalized_neighbors - aggregated_centroid, axis=1)
    distances_node2 = np.linalg.norm(all_normalized_neighbors - aggregated_centroid, axis=1)
    distance_test_datum = np.linalg.norm(normalized_test_datum - aggregated_centroid)

    # Weighted distances using scores
    weighted_distances_node1 = distances_node1 * node1_score
    weighted_distances_node2 = distances_node2 * node2_score
    weighted_distance_test_datum = distance_test_datum  # No score weighting for test datum

    # Voting using weighted distances and test datum
    votes = np.zeros(2)
    if np.sum(weighted_distances_node1) + weighted_distance_test_datum > np.sum(weighted_distances_node2) + weighted_distance_test_datum:
        votes[0] += node1_score
    else:
        votes[1] += node2_score

    # Inference based on votes
    inference = np.argmax(votes)

    print("Inference for Test Datum:", inference)

if __name__ == "__main__":
    example()
