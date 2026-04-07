# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def calculate_ratio(points):
        """Calculate min/max distance ratio"""
        if len(points) < 2:
            return 0

        # Compute pairwise distances efficiently
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances

        if distances.size == 0:
            return 0

        d_min = np.min(distances)
        d_max = np.max(distances)

        if d_max <= 0:
            return 0

        return d_min / d_max

    def create_hexagonal_initialization():
        """Create initial configuration based on hexagonal packing"""
        # For 16 points in a roughly hexagonal pattern
        # We'll make a 4x4 triangular lattice (which approximates hexagonal)
        points = []

        # Hexagonal packing parameters
        spacing_x = 1.0 / 3.0  # horizontal spacing
        spacing_y = spacing_x * np.sqrt(3) / 2  # vertical spacing

        # Create triangular lattice
        rows = 4
        cols = 4

        for i in range(rows):
            for j in range(cols):
                x = j * spacing_x
                # Offset every other row
                if i % 2 == 1:
                    x += spacing_x / 2
                y = i * spacing_y

                # Keep within bounds
                if x <= 1 and y <= 1:
                    points.append([x, y])

        points = np.array(points)

        # If we have too many points, take first 16
        if len(points) > 16:
            points = points[:16]
        # If we have too few, pad with random points
        elif len(points) < 16:
            # Add random points in a way that maintains some structure
            additional_points = []
            for _ in range(16 - len(points)):
                x = np.random.uniform(0, 1)
                y = np.random.uniform(0, 1)
                additional_points.append([x, y])
            points = np.vstack([points, additional_points])

        # Ensure we have exactly 16 points
        points = points[:16]

        # Add small symmetric perturbations to break degeneracy
        noise_scale = 0.02
        np.random.seed(42)
        points += np.random.normal(0, noise_scale, points.shape)

        # Ensure points stay within bounds
        points[:, 0] = np.clip(points[:, 0], 0, 1)
        points[:, 1] = np.clip(points[:, 1], 0, 1)

        return points

    def simulated_annealing_optimization(initial_points, max_iter=10000):
        """Optimize using simulated annealing with better cooling schedule"""
        current_points = initial_points.copy()
        current_ratio = calculate_ratio(current_points)

        # Initial temperature and parameters
        T = 0.5  # Higher initial temperature for better exploration
        cooling_rate = 0.9995  # Slower cooling for better convergence
        min_temp = 1e-8

        best_points = current_points.copy()
        best_ratio = current_ratio

        # Track recent improvements for adaptive cooling
        recent_improvements = []
        improvement_window = 100

        for iteration in range(max_iter):
            # Adaptive cooling based on progress
            if len(recent_improvements) > improvement_window:
                recent_improvements.pop(0)

            # Cool slower if no improvements recently
            if len(recent_improvements) > 0 and sum(recent_improvements[-50:]) == 0:
                T *= 0.9999  # Even slower cooling
            else:
                T *= cooling_rate

            if T < min_temp:
                break

            # Create new candidate point configuration
            new_points = current_points.copy()

            # Choose a random point to perturb
            perturb_idx = np.random.randint(len(current_points))

            # Perturb with larger step size initially, smaller later
            perturbation_magnitude = T * 0.05

            # Apply symmetric exponential perturbations to both coordinates
            dx = np.random.exponential(perturbation_magnitude)
            dy = np.random.exponential(perturbation_magnitude)

            # Randomly decide direction
            if np.random.random() > 0.5:
                dx = -dx
            if np.random.random() > 0.5:
                dy = -dy

            # Apply perturbation
            new_points[perturb_idx, 0] += dx
            new_points[perturb_idx, 1] += dy

            # Boundary handling with reflection (more sophisticated than clipping)
            for i in range(len(new_points)):
                if new_points[i, 0] < 0:
                    new_points[i, 0] = -new_points[i, 0]
                elif new_points[i, 0] > 1:
                    new_points[i, 0] = 2 - new_points[i, 0]
                if new_points[i, 1] < 0:
                    new_points[i, 1] = -new_points[i, 1]
                elif new_points[i, 1] > 1:
                    new_points[i, 1] = 2 - new_points[i, 1]

            # Calculate new ratio
            new_ratio = calculate_ratio(new_points)

            # Accept or reject using Metropolis criterion
            if new_ratio > current_ratio or np.random.rand() < np.exp((new_ratio - current_ratio) / T):
                current_points = new_points
                current_ratio = new_ratio

                # Track improvement
                recent_improvements.append(1 if new_ratio > current_ratio else 0)

                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = current_points.copy()
            else:
                recent_improvements.append(0)

        return best_points, best_ratio

    # Create initial hexagonal configuration
    initial_points = create_hexagonal_initialization()

    # Optimize using simulated annealing
    optimized_points, final_ratio = simulated_annealing_optimization(initial_points)

    return optimized_points


# EVOLVE-BLOCK-END