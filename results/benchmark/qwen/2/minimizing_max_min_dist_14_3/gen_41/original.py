# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import SphericalVoronoi
from scipy.spatial.distance import cdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)

    def fibonacci_sphere(n):
        """Generate n points distributed approximately uniformly on a sphere."""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        for i in range(n):
            z = 1 - (i / (n - 1)) * 2  # z goes from 1 to -1
            radius = np.sqrt(1 - z*z)

            theta = np.arctan2(np.sin(i * 2 * np.pi / phi), np.cos(i * 2 * np.pi / phi))
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)

    def normalize_to_unit_sphere(points):
        """Normalize points to lie on unit sphere."""
        norms = np.linalg.norm(points, axis=1)
        return points / norms[:, np.newaxis]

    def compute_distance_matrix(points):
        """Efficiently compute distance matrix."""
        return cdist(points, points, 'euclidean')

    def compute_min_max_ratio(dist_matrix):
        """Calculate min/max distance ratio, ignoring diagonal."""
        np.fill_diagonal(dist_matrix, np.inf)
        dmin = np.min(dist_matrix)
        dmax = np.max(dist_matrix)
        return dmin / dmax if dmax > 0 else 0

    def compute_spherical_voronoi_stats(points):
        """Compute statistics from spherical Voronoi tessellation."""
        # Project points to unit sphere (should already be there)
        points_normalized = normalize_to_unit_sphere(points)

        # Create SphericalVoronoi object
        sv = SphericalVoronoi(points_normalized, radius=1.0)

        # Compute Voronoi cell areas
        cell_areas = sv.calculate_areas()

        # Get the vertices of Voronoi cells
        voronoi_vertices = sv.vertices

        return sv, cell_areas, voronoi_vertices

    def balance_distances(points, max_iter=1000):
        """Main optimization loop using Voronoi-based distance balancing."""
        current_points = points.copy()
        best_points = points.copy()
        best_ratio = 0.0

        for iteration in range(max_iter):
            # Compute distance matrix
            dist_matrix = compute_distance_matrix(current_points)

            # Calculate current ratio
            current_ratio = compute_min_max_ratio(dist_matrix)

            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = current_points.copy()

            # Apply Voronoi-based adjustment
            try:
                sv, cell_areas, _ = compute_spherical_voronoi_stats(current_points)

                # Compute centroid of each Voronoi cell
                centroids = np.zeros((len(current_points), 3))
                for i in range(len(current_points)):
                    # Get indices of points belonging to this Voronoi cell
                    # For simplicity, we'll use a heuristic approach
                    pass  # Using a simpler direct approach below

                # Simple distance-based adjustment: move points to balance their distances
                adjusted_points = current_points.copy()

                # For each point, compute average distance to others and adjust accordingly
                for i in range(len(current_points)):
                    # Compute distances from point i to all others
                    distances = dist_matrix[i]
                    distances[i] = np.inf  # Ignore self-distance

                    # Find nearest and furthest points
                    nearest_idx = np.argmin(distances)
                    furthest_idx = np.argmax(distances)

                    # Move point closer to nearest and further from furthest
                    # This increases the min distance and decreases the max distance
                    if distances[nearest_idx] < distances[furthest_idx]:
                        # Move towards nearest point to increase min distance
                        direction = current_points[nearest_idx] - current_points[i]
                        move_amount = 0.001 * np.linalg.norm(direction)
                        if move_amount > 0:
                            direction = direction / np.linalg.norm(direction)
                            adjusted_points[i] += direction * move_amount

                    # Also move away from furthest to decrease max distance
                    direction = current_points[i] - current_points[furthest_idx]
                    move_amount = 0.0005 * np.linalg.norm(direction)
                    if move_amount > 0:
                        direction = direction / np.linalg.norm(direction)
                        adjusted_points[i] -= direction * move_amount

                # Ensure points stay within unit cube [0,1]^3
                adjusted_points = np.clip(adjusted_points, 0, 1)

                # Use the adjusted points for next iteration
                current_points = adjusted_points

            except Exception as e:
                # Fallback to simple gradient-like approach if Voronoi fails
                adjusted_points = current_points.copy()
                for i in range(len(current_points)):
                    # Simple gradient ascent based on neighbor distances
                    for j in range(len(current_points)):
                        if i != j:
                            diff = current_points[j] - current_points[i]
                            dist = np.linalg.norm(diff)
                            if dist > 0:
                                adjustment = 0.001 * diff / dist
                                adjusted_points[i] += adjustment

                adjusted_points = np.clip(adjusted_points, 0, 1)
                current_points = adjusted_points

        return best_points, best_ratio

    # Phase 1: Initialize with Fibonacci sphere placement
    initial_points = fibonacci_sphere(14)

    # Normalize to unit sphere
    initial_points = normalize_to_unit_sphere(initial_points)

    # Map to unit cube [0,1]^3
    initial_points = (initial_points + 1) / 2

    # Phase 2: Optimize using distance balancing
    final_points, ratio = balance_distances(initial_points, 1000)

    return final_points

# EVOLVE-BLOCK-END