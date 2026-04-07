# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def calculate_ratio(points):
        """Calculate the min/max distance ratio for given points."""
        if len(points) < 2:
            return 0

        # Compute distance matrix
        distances = squareform(pdist(points))

        # Zero out diagonal
        np.fill_diagonal(distances, np.inf)

        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return 0

        return min_dist / max_dist

    def create_hexagonal_initialization():
        """Create a better hexagonal grid initialization."""
        # Use a 4x4 hexagonal grid with proper spacing
        points = []

        # Hexagon radius (distance between centers)
        radius = 0.25  # Adjusted for better distribution in unit square

        # Create hexagonal lattice
        rows = 4
        cols = 4

        for i in range(rows):
            for j in range(cols):
                # Offset every other row
                x_offset = (i % 2) * 0.5
                x = j + x_offset
                y = i * np.sqrt(3) / 2

                # Scale to fit in [0,1] square
                x = x / (cols - 1) * 0.9 + 0.05  # Leave some margin
                y = y / ((rows - 1) * np.sqrt(3) / 2) * 0.9 + 0.05

                points.append([x, y])

        # Add small perturbations to break symmetry
        points = np.array(points)
        noise_scale = 0.02
        points += np.random.normal(0, noise_scale, points.shape)

        # Ensure points stay within bounds
        points[:, 0] = np.clip(points[:, 0], 0, 1)
        points[:, 1] = np.clip(points[:, 1], 0, 1)

        return points

    def simulated_annealing_optimization(initial_points, max_iter=10000):
        """Optimize using simulated annealing with adaptive cooling."""
        current_points = initial_points.copy()
        current_ratio = calculate_ratio(current_points)

        # Parameters for simulated annealing
        T = 0.5  # Initial temperature
        cooling_rate = 0.9995
        min_temp = 1e-6

        best_points = current_points.copy()
        best_ratio = current_ratio

        # Track recent improvements for adaptive cooling
        improvement_threshold = 50
        recent_improvements = 0

        for iteration in range(max_iter):
            # Adapt cooling rate based on recent improvements
            if recent_improvements > improvement_threshold:
                # Cool faster if we're making progress
                T *= cooling_rate * 1.05
            else:
                # Cool normally
                T *= cooling_rate

            if T < min_temp:
                break

            # Create a new candidate solution by perturbing one point
            new_points = current_points.copy()
            perturb_idx = np.random.randint(len(current_points))

            # Perturbation with exponential distribution for long tails
            delta_x = np.random.exponential(T * 0.1)
            delta_y = np.random.exponential(T * 0.1)

            # Random signs
            if np.random.random() > 0.5:
                delta_x = -delta_x
            if np.random.random() > 0.5:
                delta_y = -delta_y

            # Apply perturbation
            new_points[perturb_idx, 0] += delta_x
            new_points[perturb_idx, 1] += delta_y

            # Boundary handling with reflection
            if new_points[perturb_idx, 0] < 0:
                new_points[perturb_idx, 0] = -new_points[perturb_idx, 0]
            elif new_points[perturb_idx, 0] > 1:
                new_points[perturb_idx, 0] = 2 - new_points[perturb_idx, 0]

            if new_points[perturb_idx, 1] < 0:
                new_points[perturb_idx, 1] = -new_points[perturb_idx, 1]
            elif new_points[perturb_idx, 1] > 1:
                new_points[perturb_idx, 1] = 2 - new_points[perturb_idx, 1]

            # Calculate new ratio
            new_ratio = calculate_ratio(new_points)

            # Accept or reject using Metropolis criterion
            if new_ratio > current_ratio or np.random.rand() < np.exp((new_ratio - current_ratio) / T):
                current_points = new_points
                current_ratio = new_ratio

                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = current_points.copy()
                    recent_improvements = 0
                else:
                    recent_improvements += 1
            else:
                recent_improvements += 1

        return best_points, best_ratio

    # Multi-start optimization to avoid local optima
    best_overall_points = None
    best_overall_ratio = 0

    # Run multiple optimizations from different starting points
    num_starts = 5
    for start_idx in range(num_starts):
        # Create different initializations
        np.random.seed(42 + start_idx)  # Different seeds for variety

        # Try hexagonal initialization
        initial_points = create_hexagonal_initialization()

        # Optimize
        optimized_points, final_ratio = simulated_annealing_optimization(initial_points, 8000)

        if final_ratio > best_overall_ratio:
            best_overall_ratio = final_ratio
            best_overall_points = optimized_points.copy()

    return best_overall_points


# EVOLVE-BLOCK-END