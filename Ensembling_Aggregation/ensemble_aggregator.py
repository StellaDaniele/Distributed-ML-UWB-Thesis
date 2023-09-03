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
    correctly_classified_counter_test = 0 # Using distance to the test data
    correctly_classified_counter_centroids = 0 # Using distance to the centroids
    all_neighbors = []
    for test in range(len(y_test)):
        for node in range(n_nodes):
            for neighbor in range(n_neighbors):
                all_neighbors.append((neighbors[test][node][neighbor]))
    np_all_neighbors = np.array(all_neighbors)

    mean_coordinates = np.mean(np_all_neighbors, axis=0)
    std_coordinates = np.std(np_all_neighbors, axis=0)

    # print(str(mean_coordinates)+"\t"+str(std_coordinates))
    normalized_centroid = []
    for centroid in centroids:
        normalized_centroid.append(safe_divide(centroid - mean_coordinates, std_coordinates))
    # print(normalized_centroid)

    np_data = np.array(normalized_centroid)
    averaged_centroids = np.mean(np_data, axis=0)
    # print(averaged_centroids)

    normalized_test_coordinates = []
    for test_coordinate in tests_coordinates:
        normalized_test_coordinates.append(safe_divide(test_coordinate - mean_coordinates, std_coordinates))
    # print(normalized_test_coordinates)
    for test_id in range(len(y_test)):
        np_neighbors = []
        # np_scores = []
        # np_labels = []
        normalized_neighbors = []


        for node in range(n_nodes):
            np_neighbors.append(np.array(neighbors[test_id][node]))
            # np_scores.append(np.array(scores[test_id][node]))
            # np_labels.append(np.array(neighbors_labels[test_id][node]))
            normalized_neighbors.append(safe_divide(np.array(neighbors[test_id][node]) -
                                                    mean_coordinates,
                                                    std_coordinates))
        all_normalized_neighbors = np.vstack(normalized_neighbors)

        new_neighbors_test = [] # Closest to the test data
        new_neighbors_centroids = [] # Closest to the centroids
        for neighbor in range(len(all_normalized_neighbors)):
            distance0 = 0
            distance1 = 0
            for i in range(len(all_normalized_neighbors[neighbor])):
                distance0 += (all_normalized_neighbors[neighbor][i] - averaged_centroids[0][i]) ** 2
                distance1 += (all_normalized_neighbors[neighbor][i] - averaged_centroids[1][i]) ** 2
            distance0 = sqrt(distance0)
            distance1 = sqrt(distance1)
            # print(all_neighbors_data[neighbor],end="\t")
            # print("distance0="+str(distance0), end="\t")
            # print("distance1="+str(distance1))
            distance_from_centroid = min(distance0,distance1)
            distance_from_test = 0
            for i in range(len(all_normalized_neighbors[neighbor])):
                distance_from_test += (all_normalized_neighbors[neighbor][i] - normalized_test_coordinates[test_id][i]) ** 2
            distance_from_test = sqrt(distance_from_test)
            new_neighbors_test.append([all_normalized_neighbors[neighbor],
                                 distance_from_test,
                                 1 if distance1 < distance0 else 0])
            new_neighbors_centroids.append([all_normalized_neighbors[neighbor],
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
        correctly_classified_counter_test += int(predicted_label == y_test[test_id])

        weighted_votes = {}
        k = min(n_neighbors, 5)
        for i in range(k):
            _, distance, label = sorted_neighbors_centroids[i]
            if label not in weighted_votes:
                weighted_votes[label] = 0
            weighted_votes[label] += distance
        predicted_label = min(weighted_votes, key=weighted_votes.get)
        # print(predicted_label)
        correctly_classified_counter_centroids += int(predicted_label == y_test[test_id])
    return correctly_classified_counter_test,correctly_classified_counter_centroids
