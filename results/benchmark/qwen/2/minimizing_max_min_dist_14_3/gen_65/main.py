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
        """Initialize points in a good starting configuration using enhanced Fibonacci sphere method."""
        # Generate points on a unit sphere using an improved approach
        # Based on mathematical insights for 14-point distributions on sphere
        points = np.zeros((14, 3))

        # Golden ratio
        golden_ratio = (1 + math.sqrt(5)) / 2

        # Use a more sophisticated approach for better distribution
        # This creates a configuration that's closer to optimal
        for i in range(14):
            # Modified Fibonacci approach with better spacing
            y = 1 - (i / float(14 - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y

            # Use adjusted phi calculation for better uniformity
            # Apply a correction factor based on point index to avoid clustering
            phi = (i * golden_ratio) % 14 * (2 * math.pi / 14)

            # Add a small correction to improve distribution
            correction = 0.1 * math.sin(0.5 * i * math.pi / 7)
            phi += correction

            # Convert to Cartesian coordinates
            x = radius * math.cos(phi)
            z = radius * math.sin(phi)

            points[i] = [x, y, z]

        # Apply small random perturbations but with more strategic approach
        # This helps avoid local minima during optimization while preserving good structure
        np.random.seed(42)  # For reproducibility
        perturbation_magnitude = 0.03  # Reduced magnitude for more controlled perturbation
        points += np.random.normal(0, perturbation_magnitude, points.shape)

        # Normalize to unit sphere to ensure all points are exactly on the surface
        norms = np.linalg.norm(points, axis=1)
        safe_norms = np.where(norms == 0, 1.0, norms)
        points = points / safe_norms[:, np.newaxis]

        # Apply additional geometric refinement
        # Rotate slightly to break any remaining symmetries
        rotation_angle = np.pi / 7  # Small rotation angle
        cos_a = math.cos(rotation_angle)
        sin_a = math.sin(rotation_angle)

        # Simple rotation around the z-axis
        for i in range(14):
            x, y, z = points[i]
            points[i] = [x * cos_a - y * sin_a, x * sin_a + y * cos_a, z]

        return points

    def optimize_points(initial_points, max_iterations=10000, temp_start=1.0, cooling_rate=0.9995):
        """Optimize point placement using simulated annealing."""
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_ratio = compute_min_max_ratio(current_points)

        temp = temp_start

        for iteration in range(max_iterations):
            # Make a small random perturbation to one random point
            point_idx = np.random.randint(0, len(current_points))
            # Create perturbation vector
            perturbation = np.random.normal(0, temp * 0.01, 3)
            # Apply perturbation
            new_points = current_points.copy()
            new_points[point_idx] += perturbation

            # Normalize to keep points on unit sphere (this helps maintain the constraint)
            # But we don't strictly enforce this to allow for better optimization
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