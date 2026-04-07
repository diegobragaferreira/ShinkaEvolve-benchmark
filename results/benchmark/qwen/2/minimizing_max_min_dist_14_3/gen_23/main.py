# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """

    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        distances = pdist(points)
        return np.min(distances) / np.max(distances)

    def initialize_points():
        """Initialize points in a good starting configuration using improved Fibonacci sphere method."""
        # Use improved Fibonacci sphere method for better distribution
        points = np.zeros((14, 3))

        # Golden ratio
        golden_ratio = (1 + math.sqrt(5)) / 2

        for i in range(14):
            # Improved Fibonacci sphere placement
            # y goes from 1 to -1 (not quite the same as standard)
            y = 1 - (i / float(14 - 1)) * 2

            # Radius at this y value
            radius = math.sqrt(1 - y * y)

            # Angle around the sphere
            theta = math.acos(y)  # polar angle from z-axis

            # Improved azimuthal angle calculation
            phi = ((i * golden_ratio) % 14) * (2 * math.pi / 14)  # Corrected version

            # Convert to Cartesian coordinates
            x = radius * math.cos(phi)
            z = radius * math.sin(phi)

            points[i] = [x, y, z]

        return points

    def optimize_points(initial_points, max_iterations=10000, temp_start=1.0, cooling_rate=0.9995):
        """Optimize point placement using simulated annealing with enhanced perturbation."""
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_ratio = compute_min_max_ratio(current_points)

        temp = temp_start

        for iteration in range(max_iterations):
            # Enhanced perturbation selection: choose point based on distance characteristics
            # Calculate all pairwise distances to understand current distribution
            distances = pdist(current_points)

            # Select point that would most benefit from adjustment
            # Focus on extremes: points that are involved in small distances or large distances
            if iteration % 500 == 0 and iteration > 0:
                # Sometimes select a point based on distance statistics
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                mean_dist = np.mean(distances)

                # If we have many small distances, prefer to move points that are too close
                if min_dist < mean_dist * 0.3:
                    # Prefer points in smaller distances
                    point_idx = np.random.choice(len(current_points), p=[1/len(current_points)]*len(current_points))
                else:
                    # Otherwise random selection
                    point_idx = np.random.randint(0, len(current_points))
            else:
                # Standard random selection
                point_idx = np.random.randint(0, len(current_points))

            # Create perturbation vector with adaptive scale based on temperature
            perturbation_scale = temp * 0.02
            perturbation = np.random.normal(0, perturbation_scale, 3)

            # Apply perturbation
            new_points = current_points.copy()
            new_points[point_idx] += perturbation

            # Normalize to keep points approximately on unit sphere
            # This helps maintain constraint without being too restrictive
            norm_factors = np.linalg.norm(new_points, axis=1)
            # Avoid division by zero
            norm_factors = np.where(norm_factors == 0, 1, norm_factors)
            new_points = new_points / norm_factors[:, np.newaxis] * np.ones_like(new_points) * 0.999 + 0.001

            new_ratio = compute_min_max_ratio(new_points)

            # Accept or reject based on Metropolis criterion
            if new_ratio > best_ratio or np.random.rand() < math.exp((new_ratio - best_ratio) / temp):
                current_points = new_points
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = new_points.copy()

            # Cool down
            temp *= cooling_rate

            # Occasionally print progress
            if iteration % 1000 == 0:
                print(f"Iteration {iteration}, Best ratio: {best_ratio:.6f}")

        return best_points, best_ratio

    # Initialize with a good configuration
    initial_points = initialize_points()

    # Optimize using simulated annealing
    optimized_points, final_ratio = optimize_points(initial_points)

    # Ensure we're returning the best solution found
    return optimized_points


# EVOLVE-BLOCK-END