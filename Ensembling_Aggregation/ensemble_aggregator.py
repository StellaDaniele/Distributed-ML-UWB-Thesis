import numpy as np

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

    # When the standard deviation is 0, avoid dividing by 0
    def safe_divide(numerator, denominator):
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

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
