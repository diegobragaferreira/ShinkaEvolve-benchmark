# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    np.random.seed(42)
    n = 16
    d = 2

    # Create structured initial lattice configuration
    def create_lattice_initialization():
        # Create a 4x4 grid with slight perturbations
        rows, cols = 4, 4
        points = []

        for i in range(rows):
            for j in range(cols):
                x = j + 0.5 * (i % 2)
                y = i * np.sqrt(3)/2
                points.append([x, y])

        points = np.array(points)

        # Normalize to [0,1] x [0,1]
        max_x = cols - 0.5
        max_y = (rows - 1) * np.sqrt(3)/2

        points[:, 0] = points[:, 0] / max_x
        points[:, 1] = points[:, 1] / max_y

        # Add structured perturbations to break symmetry
        noise_scale = 0.015
        points += np.random.normal(0, noise_scale, points.shape)

        # Ensure points stay within bounds
        points[:, 0] = np.clip(points[:, 0], 0, 1)
        points[:, 1] = np.clip(points[:, 1], 0, 1)

        return points

    # Calculate min/max distance ratio efficiently
    def calculate_ratio(points):
        if len(points) < 2:
            return 0

        # Compute pairwise distances
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances

        if distances.size == 0:
            return 0

        d_min = np.min(distances)
        d_max = np.max(distances)

        if d_max <= 0:
            return 0

        return d_min / d_max

    # Enhanced optimization using improved approach
    def optimize_points(initial_points, max_iter=10000):
        current_points = initial_points.copy()
        current_ratio = calculate_ratio(current_points)

        # Improved parameters for this approach
        T = 0.3  # Higher initial temperature
        cooling_rate = 0.9997  # Faster cooling rate
        min_temp = 1e-6

        best_points = current_points.copy()
        best_ratio = current_ratio

        # Track recent improvements for adaptive cooling
        recent_improvements = []
        improvement_window = 50

        # Simplified perturbation approach
        def get_targeted_perturbation(points, temperature, iteration):
            # Select a random point to perturb
            perturb_idx = np.random.randint(len(points))

            # Adjust perturbation magnitude based on iteration progress
            base_magnitude = temperature * 0.08
            adaptive_magnitude = base_magnitude * (1.0 - iteration / max_iter * 0.5)

            # Create perturbation with exponential distribution
            new_points = points.copy()

            # Try several candidates and pick the best one
            best_candidate_ratio = current_ratio
            best_candidate_pos = points[perturb_idx].copy()

            # Try multiple perturbation attempts
            num_samples = 15
            for _ in range(num_samples):
                # Create perturbation with exponential distribution for longer tails
                dx = np.random.exponential(adaptive_magnitude)
                dy = np.random.exponential(adaptive_magnitude)

                # Randomly decide direction
                if np.random.random() > 0.5:
                    dx = -dx
                if np.random.random() > 0.5:
                    dy = -dy

                candidate_point = points[perturb_idx].copy()
                candidate_point[0] += dx
                candidate_point[1] += dy

                # Apply boundary reflection
                if candidate_point[0] < 0:
                    candidate_point[0] = -candidate_point[0]
                elif candidate_point[0] > 1:
                    candidate_point[0] = 2 - candidate_point[0]
                if candidate_point[1] < 0:
                    candidate_point[1] = -candidate_point[1]
                elif candidate_point[1] > 1:
                    candidate_point[1] = 2 - candidate_point[1]

                # Test this move
                test_points = new_points.copy()
                test_points[perturb_idx] = candidate_point

                test_ratio = calculate_ratio(test_points)
                if test_ratio > best_candidate_ratio:
                    best_candidate_ratio = test_ratio
                    best_candidate_pos = candidate_point.copy()

            # Update the point with the best candidate
            new_points[perturb_idx] = best_candidate_pos

            return new_points, best_candidate_ratio

        for iteration in range(max_iter):
            # Adaptive cooling based on recent improvements
            if len(recent_improvements) > improvement_window:
                recent_improvements.pop(0)

            # Cooling schedule with adaptation
            if len(recent_improvements) > 0 and sum(recent_improvements[-10:]) == 0:
                # If no improvements recently, cool slower
                T *= 0.9999
            else:
                T *= cooling_rate

            if T < min_temp:
                break

            # Get targeted perturbation
            new_points, new_ratio = get_targeted_perturbation(current_points, T, iteration)

            # Accept or reject the new solution using Metropolis criterion
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

    # Create multiple diverse initial configurations
    def create_multiple_initializations():
        initializations = []

        # 1. Original lattice initialization
        initializations.append(create_lattice_initialization())

        # 2. Random initialization
        np.random.seed(42)
        rand_points = np.random.rand(16, 2)
        initializations.append(rand_points)

        # 3. Perturbed hexagonal grid
        hex_points = create_lattice_initialization()
        hex_points += np.random.normal(0, 0.02, hex_points.shape)
        # Clip to bounds to keep within valid range
        hex_points[:, 0] = np.clip(hex_points[:, 0], 0, 1)
        hex_points[:, 1] = np.clip(hex_points[:, 1], 0, 1)
        initializations.append(hex_points)

        # 4. Alternative structured initialization
        alt_points = []
        for i in range(4):
            for j in range(4):
                x = i * 0.25 + np.random.normal(0, 0.01)
                y = j * 0.25 + np.random.normal(0, 0.01)
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                alt_points.append([x, y])
        initializations.append(np.array(alt_points))

        return initializations

    # Multi-start optimization
    initial_configs = create_multiple_initializations()

    best_final_points = None
    best_ratio = -np.inf

    # Run optimization from each initial configuration
    for i, initial_config in enumerate(initial_configs):
        print(f"Starting optimization run {i+1}...")
        final_points, ratio = optimize_points(initial_config)

        if ratio > best_ratio:
            best_ratio = ratio
            best_final_points = final_points

    # Return the best result found
    return best_final_points

# EVOLVE-BLOCK-END