import numpy as np

def initialize_centroids_forgy(data, k):
    # TODO implement random initialization
    centroid_indices = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[centroid_indices]
    return centroids

def initialize_centroids_kmeans_pp(data, k):
    # TODO implement kmeans++ initizalization
    num_samples, num_features = data.shape
    centroids = np.empty((k, num_features))
    
    first_centroid_idx = np.random.choice(num_samples)
    centroids[0] = data[first_centroid_idx]
    
    for i in range(1, k):
        distances = np.array([min(np.linalg.norm(centroid - x) ** 2 for centroid in centroids[:i]) for x in data])
        probabilities = distances / distances.sum()
        next_centroid_idx = np.random.choice(num_samples, p=probabilities)
        centroids[i] = data[next_centroid_idx]
    
    return centroids

def assign_to_cluster(data, centroids):
    return np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)

def update_centroids(data, assignments):
    num_samples, num_features = data.shape
    k = np.max(assignments) + 1
    centroids = np.empty((k, num_features))

    for i in range(k):
        cluster_points = data[assignments == i]
        centroids[i] = np.mean(cluster_points, axis=0)

    return centroids

def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :])**2))

def k_means(data, num_centroids, kmeansplusplus= False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else: 
        centroids = initialize_centroids_forgy(data, num_centroids)

    
    assignments  = assign_to_cluster(data, centroids)
    for i in range(100): # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)         

