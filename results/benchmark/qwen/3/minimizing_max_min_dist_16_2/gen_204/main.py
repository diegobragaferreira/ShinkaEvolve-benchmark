# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
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

    # Create structured hexagonal initial lattice configuration with enhanced asymmetry
    def create_hexagonal_initialization():
        # Create a 4x4 hexagonal grid pattern
        points = []
        sqrt3 = np.sqrt(3)

        # Hexagonal grid parameters
        spacing_x = 1.0 / 3.0
        spacing_y = sqrt3 / 4.0

        for i in range(4):
            for j in range(4):
                x = j * spacing_x
                y = i * spacing_y

                # Offset odd rows for hexagonal pattern
                if i % 2 == 1:
                    x += spacing_x / 2

                # Add systematic asymmetry based on position to break symmetry
                # This creates more effective symmetry breaking than simple random noise
                position_factor = (i * 7 + j * 3) % 10
                noise_scale = 0.015 + position_factor * 0.003

                # Use more structured noise patterns with directional bias
                x += np.random.normal(0, noise_scale * 0.7)
                y += np.random.normal(0, noise_scale * 0.7)

                # Add slight directional bias to encourage better distribution
                if i % 3 == 0:
                    x += np.random.normal(0, noise_scale * 0.2)
                if j % 3 == 0:
                    y += np.random.normal(0, noise_scale * 0.2)

                points.append([x, y])

        points = np.array(points)

        # Normalize to [0,1] x [0,1] properly
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

        if x_max > x_min:
            points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
        if y_max > y_min:
            points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)

        # Ensure all points are within bounds
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

    # Enhanced optimization using improved simulated annealing with adaptive cooling
    def optimize_points(initial_points, max_iter=15000):
        current_points = initial_points.copy()
        current_ratio = calculate_ratio(current_points)

        # Improved parameters for this approach
        T = 0.4  # Higher initial temperature for better exploration
        cooling_rate = 0.9997  # Slightly faster cooling rate
        min_temp = 1e-6
        best_points = current_points.copy()
        best_ratio = current_ratio

        # Track recent improvements for adaptive cooling
        recent_improvements = []
        improvement_window = 100

        # Track global progress
        improvement_history = []

        # Neighborhood-based perturbation strategy
        def get_neighborhood_perturbation(points, temperature, iteration):
            # Create a copy of the point array
            new_points = points.copy()

            # Select a random point to perturb
            perturb_idx = np.random.randint(len(points))

            # Adjust perturbation magnitude based on iteration progress
            base_magnitude = temperature * 0.15
            adaptive_magnitude = base_magnitude * (1.0 - iteration / max_iter * 0.3)

            # Try several candidates based on local neighborhood
            best_candidate_ratio = current_ratio
            best_candidate_pos = points[perturb_idx].copy()

            # Sample many candidates for better selection
            num_samples = 30  # Increased sampling for better selection

            # Create perturbations using multiple strategies
            for _ in range(num_samples):
                # Strategy 1: Gaussian perturbation with adaptive variance
                # Based on how close we are to the target (smaller variance when close to optimal)
                gaussian_variance = adaptive_magnitude * 0.5  # Reduced influence of temperature
                dx = np.random.normal(0, gaussian_variance)
                dy = np.random.normal(0, gaussian_variance)

                candidate_point = points[perturb_idx].copy()
                candidate_point[0] += dx
                candidate_point[1] += dy

                # Apply boundary handling with rejection sampling for better distribution
                # Instead of reflection, we check if point is valid, otherwise generate new one
                attempts = 0
                while (candidate_point[0] < 0 or candidate_point[0] > 1 or
                       candidate_point[1] < 0 or candidate_point[1] > 1) and attempts < 5:
                    # Generate completely new point in valid region with some bias toward original
                    candidate_point = points[perturb_idx].copy()
                    # Add a bit of Gaussian noise to avoid getting stuck exactly where we were
                    candidate_point[0] += np.random.normal(0, gaussian_variance * 0.5)
                    candidate_point[1] += np.random.normal(0, gaussian_variance * 0.5)
                    attempts += 1

                # If we still have issues, just clip to prevent infinite loops
                candidate_point[0] = np.clip(candidate_point[0], 0, 1)
                candidate_point[1] = np.clip(candidate_point[1], 0, 1)

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

        # Main optimization loop
        for iteration in range(max_iter):
            # Adaptive cooling based on recent improvements and overall trend
            if len(recent_improvements) > improvement_window:
                recent_improvements.pop(0)

            # Track global progress
            improvement_history.append(current_ratio)
            if len(improvement_history) > 50:
                improvement_history.pop(0)

            # Dynamic cooling adjustment
            if len(recent_improvements) > 0:
                recent_improvement_rate = sum(recent_improvements[-20:]) / min(20, len(recent_improvements))
                # If we haven't improved much recently, slow down cooling
                if recent_improvement_rate < 0.1:
                    T *= 0.9998  # Very slow cooling
                elif recent_improvement_rate > 0.3:
                    # If we're improving quickly, speed up cooling a bit
                    T *= (cooling_rate * 0.999)
                else:
                    # Normal cooling rate
                    T *= cooling_rate
            else:
                T *= cooling_rate

            # Additional safety check: if we're in a plateau state, cool slower
            if len(improvement_history) > 20:
                recent_changes = np.diff(improvement_history[-20:])
                avg_change = np.mean(np.abs(recent_changes))
                if avg_change < 1e-6:
                    T *= 0.9999  # Very slow cooling during stagnation

            if T < min_temp:
                break

            # Get targeted perturbation
            new_points, new_ratio = get_neighborhood_perturbation(current_points, T, iteration)

            # Accept or reject the new solution using Metropolis criterion
            if new_ratio > current_ratio or np.random.rand() < np.exp((new_ratio - current_ratio) / T):
                current_points = new_points
                current_ratio = new_ratio

                # Track improvement
                recent_improvements.append(1 if new_ratio > current_ratio else 0)

                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = current_points.copy()
                    # Reset stagnation counter when we find a better solution
            else:
                recent_improvements.append(0)

        return best_points, best_ratio

    # Multi-start optimization with diverse initial configurations
    def create_multiple_initializations():
        initializations = []

        # 1. Enhanced hexagonal grid with systematic asymmetry
        initializations.append(create_hexagonal_initialization())

        # 2. Random initialization
        rand_points = np.random.rand(16, 2)
        initializations.append(rand_points)

        # 3. Grid initialization with different spacing
        grid_points = []
        for i in range(4):
            for j in range(4):
                x = i * 0.25 + np.random.normal(0, 0.015)
                y = j * 0.25 + np.random.normal(0, 0.015)
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                grid_points.append([x, y])
        initializations.append(np.array(grid_points))

        # 4. Perturbed hexagonal grid
        hex_points = create_hexagonal_initialization()
        hex_points += np.random.normal(0, 0.025, hex_points.shape)
        hex_points[:, 0] = np.clip(hex_points[:, 0], 0, 1)
        hex_points[:, 1] = np.clip(hex_points[:, 1], 0, 1)
        initializations.append(hex_points)

        # 5. Triangular lattice pattern with different spacing
        tri_points = []
        sqrt3 = np.sqrt(3)
        spacing_x = 1.0 / 3.0
        spacing_y = sqrt3 / 4.0

        for i in range(4):
            for j in range(4):
                x = j * spacing_x
                y = i * spacing_y

                if i % 2 == 1:
                    x += spacing_x / 2

                # Add noise with systematic variations
                x += (np.random.random() - 0.5) * 0.018
                y += (np.random.random() - 0.5) * 0.018

                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                tri_points.append([x, y])

        initializations.append(np.array(tri_points))

        # 6. Spiral pattern initialization (known to work well for dispersion)
        spiral_points = []
        center = np.array([0.5, 0.5])
        radius = 0.45
        angle_step = 2 * np.pi / 16
        for i in range(16):
            angle = i * angle_step
            r = radius * (i / 15.0)  # Gradually increase radius
            x = center[0] + r * np.cos(angle)
            y = center[1] + r * np.sin(angle)
            # Add small noise to break symmetry
            x += np.random.normal(0, 0.01)
            y += np.random.normal(0, 0.01)
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            spiral_points.append([x, y])
        initializations.append(np.array(spiral_points))

        # 7. Concentric rings pattern with random variation
        ring_points = []
        for i in range(16):
            # Distribute along concentric circles
            circle_radius = 0.4 * (i % 4 + 1) / 4.0  # 4 concentric circles
            angle = 2 * np.pi * i / 16.0
            x = 0.5 + circle_radius * np.cos(angle)
            y = 0.5 + circle_radius * np.sin(angle)
            # Add noise to prevent regularity
            x += np.random.normal(0, 0.015)
            y += np.random.normal(0, 0.015)
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            ring_points.append([x, y])
        initializations.append(np.array(ring_points))

        return initializations

    # Run multiple optimizations from different starting points
    initial_configs = create_multiple_initializations()

    best_final_points = None
    best_ratio = -np.inf

    # Run optimization from each initial configuration with increased iterations
    for i, initial_config in enumerate(initial_configs):
        print(f"Starting optimization run {i+1}...")
        final_points, ratio = optimize_points(initial_config, max_iter=12000)

        if ratio > best_ratio:
            best_ratio = ratio
            best_final_points = final_points

    # Final refinement with the best configuration using a slightly different approach
    if best_final_points is not None:
        # Try one more optimization run with the best configuration but with a different cooling schedule
        final_points, final_ratio = optimize_points(best_final_points, max_iter=5000)
        return final_points
    else:
        # Fallback to hexagonal initialization
        return create_hexagonal_initialization()

# EVOLVE-BLOCK-END