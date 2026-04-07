# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time
import random

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distances between all point pairs."""
        if len(points) < 2:
            return 0.0

        # Compute pairwise distances
        distances = pdist(points)

        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return 0.0

        return min_dist / max_dist

    def compute_min_max_ratio_with_penalty(points, penalty_factor=1000.0):
        """Compute the ratio with boundary penalties to avoid edge violations."""
        # Apply boundary penalty: points too close to edges get penalized heavily
        penalty = 0.0
        for point in points:
            # Penalize if point is within 0.01 of any boundary
            if (point[0] < 0.01 or point[0] > 0.99 or
                point[1] < 0.01 or point[1] > 0.99):
                penalty += penalty_factor

        ratio = compute_min_max_ratio(points)
        return ratio - penalty / len(points)

    def generate_multiple_initial_configs():
        """Generate several different initial configurations with symmetry breaking."""
        configs = []

        # Configuration 1: Hexagonal grid with deterministic symmetry-breaking perturbations
        points1 = []
        for i in range(4):
            for j in range(4):
                # Base hexagonal position
                x_base = j * 0.25 + (i % 2) * 0.125
                y_base = i * 0.25

                # Apply symmetry-breaking perturbation based on position
                symmetry_factor = (i * 7 + j * 3) % 10
                x_pert = np.sin(symmetry_factor * 0.5) * 0.005
                y_pert = np.cos(symmetry_factor * 0.3) * 0.005

                x = x_base + x_pert + np.random.normal(0, 0.003)
                y = y_base + y_pert + np.random.normal(0, 0.003)
                points1.append([x, y])
        points1 = np.array(points1)
        points1 = np.clip(points1, 0, 1)
        configs.append(points1)

        # Configuration 2: More clustered arrangement with asymmetry
        points2 = []
        for i in range(4):
            for j in range(4):
                x_base = j * 0.25 + (i % 2) * 0.125
                y_base = i * 0.25

                # Different symmetry breaking for second config
                x_pert = np.sin(i * 0.7) * 0.008 * (1 + j * 0.1)
                y_pert = np.cos(j * 0.5) * 0.008 * (1 + i * 0.1)

                x = x_base + x_pert + np.random.normal(0, 0.007)
                y = y_base + y_pert + np.random.normal(0, 0.007)
                points2.append([x, y])
        points2 = np.array(points2)
        points2 = np.clip(points2, 0, 1)
        configs.append(points2)

        # Configuration 3: Random but structured with controlled asymmetry
        points3 = []
        # Create a more complex base pattern
        for i in range(4):
            for j in range(4):
                # Apply non-uniform spacing and asymmetry
                x = 0.1 + j * 0.22 + (i % 2) * 0.11 + np.sin(i * 1.3) * 0.01
                y = 0.1 + i * 0.22 + np.cos(j * 1.7) * 0.01
                points3.append([x, y])
        points3 = np.array(points3) + np.random.normal(0, 0.01, (16, 2))
        points3 = np.clip(points3, 0, 1)
        configs.append(points3)

        # Configuration 4: Spiral-like pattern with deterministic asymmetry
        points4 = []
        angle = 0
        radius = 0
        for i in range(16):
            # Add deterministic asymmetry to spiral
            angle_offset = np.sin(i * 0.3) * 0.1
            radius_offset = np.cos(i * 0.2) * 0.01
            x = 0.5 + (radius + radius_offset) * np.cos(angle + angle_offset)
            y = 0.5 + (radius + radius_offset) * np.sin(angle + angle_offset)
            points4.append([x, y])
            angle += 0.5 + np.sin(i * 0.1) * 0.05
            radius += 0.01 + np.cos(i * 0.2) * 0.005
        points4 = np.array(points4)
        points4 = np.clip(points4, 0, 1)
        configs.append(points4)

        # Configuration 5: Checkerboard with systematic asymmetry
        points5 = []
        for i in range(4):
            for j in range(4):
                x_base = j * 0.25 + (i % 2) * 0.125
                y_base = i * 0.25

                # Systematic asymmetry based on position
                asym_x = ((i + 1) * (j + 1)) % 5 * 0.003
                asym_y = ((i + 1) * (j + 1)) % 7 * 0.002

                x = x_base + asym_x + np.random.normal(0, 0.005)
                y = y_base + asym_y + np.random.normal(0, 0.005)
                points5.append([x, y])
        points5 = np.array(points5)
        points5 = np.clip(points5, 0, 1)
        configs.append(points5)

        # Configuration 6: Random uniform distribution with slight clustering
        points6 = np.random.rand(16, 2) * 0.9 + 0.05
        configs.append(points6)

        # Configuration 7: Regular grid with small random jitter
        points7 = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + (i % 2) * 0.125 + np.random.normal(0, 0.005)
                y = i * 0.25 + np.random.normal(0, 0.005)
                points7.append([x, y])
        points7 = np.array(points7)
        points7 = np.clip(points7, 0, 1)
        configs.append(points7)

        return configs

    def objective_function(params):
        """Objective function to minimize (negative of min/max ratio)."""
        # Reshape parameters back to points array
        points = params.reshape(-1, 2)
        ratio = compute_min_max_ratio(points)
        # Return negative because we want to maximize the ratio
        return -ratio

    def optimize_with_lbfgs(initial_points):
        """Optimize using L-BFGS-B method for local refinement."""
        # Flatten for optimization
        initial_params = initial_points.flatten()

        # Set up bounds for each coordinate (0 to 1 for both x and y)
        bounds = [(0, 1)] * 32  # 16 points * 2 coordinates each

        # Optimize using L-BFGS-B
        try:
            result = minimize(
                objective_function,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-10},
                callback=None
            )

            # Extract optimized points
            optimized_points = result.x.reshape(-1, 2)

            # Make sure they're within bounds
            optimized_points = np.clip(optimized_points, 0, 1)

            return optimized_points
        except Exception:
            return initial_points

    def optimize_points(initial_points, max_time=175):
        """Optimize point positions using enhanced simulated annealing with hybrid approach."""
        start_time = time.time()

        # Normalize initial points to [0,1] x [0,1]
        points = np.clip(initial_points, 0, 1)
        current_ratio = compute_min_max_ratio_with_penalty(points)

        # Enhanced parameters for optimization
        temperature = 1.0
        # Adaptive cooling - starts fast, then slows down
        cooling_rate = 0.9999
        min_temperature = 1e-8
        max_iterations = 500000
        iteration = 0

        best_points = points.copy()
        best_ratio = current_ratio

        # Track recent improvements for early stopping
        recent_improvements = []
        patience = 0
        max_patience = 1000

        while temperature > min_temperature and iteration < max_iterations and (time.time() - start_time) < max_time:
            # Create candidate solution using cluster-based moves for better exploration
            candidate_points = points.copy()

            # Choose move type: single point move (70%), cluster move (20%), or global perturbation (10%)
            move_type = random.random()
            if move_type < 0.2:
                # Cluster move: move 2-4 nearby points together
                num_points_to_move = random.randint(2, 4)
                selected_indices = random.sample(range(len(points)), num_points_to_move)

                # Calculate centroid of selected points
                centroid = np.mean(candidate_points[selected_indices], axis=0)

                # Move centroid and adjust all points relative to it
                move_vector = np.random.normal(0, 0.015, 2)
                new_centroid = np.clip(centroid + move_vector, 0, 1)
                delta = new_centroid - centroid

                for idx in selected_indices:
                    candidate_points[idx] += delta
            elif move_type < 0.3:
                # Global perturbation: perturb all points
                candidate_points += np.random.normal(0, 0.01, candidate_points.shape)
            else:
                # Single point move (standard approach)
                idx = np.random.randint(0, len(points))
                # Larger perturbation for better exploration
                candidate_points[idx] += np.random.normal(0, 0.02, 2)

            # Keep within bounds
            candidate_points = np.clip(candidate_points, 0, 1)

            # Calculate acceptance probability
            candidate_ratio = compute_min_max_ratio_with_penalty(candidate_points)

            # Accept or reject based on Metropolis criterion
            if candidate_ratio > current_ratio or np.random.rand() < np.exp((candidate_ratio - current_ratio) / temperature):
                points = candidate_points
                current_ratio = candidate_ratio

                # Update best solution
                if current_ratio > best_ratio:
                    best_points = points.copy()
                    best_ratio = current_ratio
                    recent_improvements = []
                    patience = 0
                else:
                    patience += 1
                    recent_improvements.append(current_ratio)
                    if len(recent_improvements) > 50:
                        recent_improvements.pop(0)
            else:
                patience += 1

            # Early stopping if no improvement for too long
            if patience > max_patience:
                if len(recent_improvements) > 10:
                    recent_avg = np.mean(recent_improvements[-10:])
                    if recent_avg > 0.99 * best_ratio:
                        break

            # Cool down with adaptive rate
            if temperature > 0.1:
                temperature *= cooling_rate  # Faster cooling initially
            else:
                temperature *= 0.99999  # Slower cooling in later stages

            iteration += 1

        # Final local refinement with L-BFGS
        final_points = optimize_with_lbfgs(best_points)
        final_ratio = compute_min_max_ratio_with_penalty(final_points)

        if final_ratio > best_ratio:
            return final_points
        else:
            return best_points

    # Generate multiple initial configurations
    np.random.seed(42)
    initial_configs = generate_multiple_initial_configs()

    # Run optimization from each configuration
    best_final_points = None
    best_final_ratio = -np.inf

    for i, initial_config in enumerate(initial_configs):
        # Use a slightly reduced time budget per run to allow for multiple runs
        config_points = optimize_points(initial_config, max_time=175 / len(initial_configs))
        config_ratio = compute_min_max_ratio_with_penalty(config_points)

        if config_ratio > best_final_ratio:
            best_final_ratio = config_ratio
            best_final_points = config_points.copy()

    # Final validation
    if best_final_points is None:
        # Fallback to a simple hexagonal arrangement
        fallback_points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25
                fallback_points.append([x, y])
        best_final_points = np.array(fallback_points)
        best_final_points = np.clip(best_final_points + np.random.normal(0, 0.01, (16, 2)), 0, 1)

    return best_final_points

# EVOLVE-BLOCK-END