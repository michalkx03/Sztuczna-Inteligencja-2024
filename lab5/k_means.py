import numpy as np

def initialize_centroids_forgy(data, k):
    # TODO implement random initialization
    centroid_indices = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[centroid_indices]
    return centroids

def initialize_centroids_kmeans_pp(data, k):
    # TODO implement kmeans++ initizalization

    data_index = np.random.choice(len(data))
    centroids = []
    centroids.append(data[data_index])

    for _ in range(k-1):
        max_dist = -1
        best = None
        for point in data:
            if not any(np.array_equal(point, centroid) for centroid in centroids):
                dist = sum([np.sum((centroid - point)**2) for centroid in centroids])
                if dist > max_dist:
                    max_dist = dist
                    best = point
        centroids.append(best)

    return np.array(centroids)

def assign_to_cluster(data, centroids):
    assigned = []
    for point in data:
        dist = [np.linalg.norm(point - centroid) for centroid in centroids]
        assigned_cluster = np.argmin(dist)
        assigned.append(assigned_cluster)
    return np.array(assigned)


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
        #print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)         

