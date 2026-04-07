# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import time
from typing import Tuple

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses an enhanced Voronoi-based evolutionary approach with multi-stage optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    np.random.seed(42)

    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances"""
        if len(points) < 2:
            return 0
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max <= 0:
            return 0
        return d_min / d_max

    def create_diverse_initial_configurations():
        """Create 12 diverse initial configurations for robust optimization"""
        configurations = []

        # Strategy 1: Base hexagonal grid
        points = []
        rows = 4
        cols = 4

        spacing_x = 1.0
        spacing_y = np.sqrt(3) / 2

        for i in range(rows):
            for j in range(cols):
                x = j * spacing_x + (i % 2) * spacing_x * 0.5
                y = i * spacing_y
                points.append([x, y])

        points = np.array(points)

        # Normalize to [0,1] x [0,1]
        max_x = (cols - 1) + 0.5
        max_y = (rows - 1) * spacing_y

        points[:, 0] = points[:, 0] / max_x
        points[:, 1] = points[:, 1] / max_y

        configurations.append(points.copy())

        # Strategy 2: Hexagonal grid with strong noise
        noise = np.random.normal(0, 0.02, points.shape)
        noisy_points = points + noise
        noisy_points = np.clip(noisy_points, 0, 1)
        configurations.append(noisy_points.copy())

        # Strategy 3: Hexagonal grid with moderate noise
        noise = np.random.normal(0, 0.01, points.shape)
        noisy_points = points + noise
        noisy_points = np.clip(noisy_points, 0, 1)
        configurations.append(noisy_points.copy())

        # Strategy 4: Random points with boundary awareness
        random_points = np.random.rand(16, 2)
        for i in range(len(random_points)):
            # Boundary correction
            if random_points[i, 0] < 0.01:
                random_points[i, 0] = 0.01 + np.random.rand() * 0.01
            elif random_points[i, 0] > 0.99:
                random_points[i, 0] = 0.99 - np.random.rand() * 0.01

            if random_points[i, 1] < 0.01:
                random_points[i, 1] = 0.01 + np.random.rand() * 0.01
            elif random_points[i, 1] > 0.99:
                random_points[i, 1] = 0.99 - np.random.rand() * 0.01
        configurations.append(random_points.copy())

        # Strategy 5: Perturbed uniform grid
        uniform_points = []
        for i in range(4):
            for j in range(4):
                uniform_points.append([i/3, j/3])
        uniform_points = np.array(uniform_points[:16])
        perturbations = np.random.normal(0, 0.015, (16, 2))
        perturbed_uniform = uniform_points + perturbations
        perturbed_uniform = np.clip(perturbed_uniform, 0, 1)
        configurations.append(perturbed_uniform.copy())

        # Strategy 6: Triangular lattice with noise
        triangular_points = []
        rows = 4
        cols = 4
        spacing_x = 1.0
        spacing_y = np.sqrt(3)/2

        for i in range(rows):
            for j in range(cols):
                x = j * spacing_x + (i % 2) * spacing_x * 0.5
                y = i * spacing_y
                triangular_points.append([x, y])

        triangular_points = np.array(triangular_points)
        max_x = (cols - 1) + 0.5
        max_y = (rows - 1) * spacing_y
        triangular_points[:, 0] = triangular_points[:, 0] / max_x
        triangular_points[:, 1] = triangular_points[:, 1] / max_y
        noise = np.random.normal(0, 0.01, triangular_points.shape)
        triangular_points = triangular_points + noise
        triangular_points = np.clip(triangular_points, 0, 1)
        configurations.append(triangular_points.copy())

        # Strategy 7: Random with seed 123
        np.random.seed(123)
        configurations.append(np.random.rand(16, 2).copy())

        # Strategy 8: Random with seed 456
        np.random.seed(456)
        configurations.append(np.random.rand(16, 2).copy())

        # Strategy 9: Spherical arrangement (inspired by optimal point distributions)
        # Arrange points on a circle and then distribute in 2D
        theta = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radius = 0.4  # smaller radius to keep within bounds
        spherical_points = np.column_stack([radius * np.cos(theta) + 0.5, radius * np.sin(theta) + 0.5])
        configurations.append(spherical_points.copy())

        # Strategy 10: Perturbed spherical arrangement
        noise = np.random.normal(0, 0.01, spherical_points.shape)
        perturbed_spherical = spherical_points + noise
        perturbed_spherical = np.clip(perturbed_spherical, 0, 1)
        configurations.append(perturbed_spherical.copy())

        # Strategy 11: Random points with higher boundary avoidance
        random_points_high = np.random.rand(16, 2)
        # Push points away from boundaries with stronger force
        for i in range(len(random_points_high)):
            if random_points_high[i, 0] < 0.02:
                random_points_high[i, 0] = 0.02 + np.random.rand() * 0.02
            elif random_points_high[i, 0] > 0.98:
                random_points_high[i, 0] = 0.98 - np.random.rand() * 0.02

            if random_points_high[i, 1] < 0.02:
                random_points_high[i, 1] = 0.02 + np.random.rand() * 0.02
            elif random_points_high[i, 1] > 0.98:
                random_points_high[i, 1] = 0.98 - np.random.rand() * 0.02
        configurations.append(random_points_high.copy())

        # Strategy 12: Grid-like with more randomization
        grid_like_points = []
        for i in range(4):
            for j in range(4):
                grid_like_points.append([i/3 + np.random.normal(0, 0.01), j/3 + np.random.normal(0, 0.01)])
        grid_like_points = np.array(grid_like_points[:16])
        grid_like_points = np.clip(grid_like_points, 0, 1)
        configurations.append(grid_like_points.copy())

        return configurations

    def adaptive_simulated_annealing(initial_points, max_iter=2000):
        """Enhanced simulated annealing with adaptive parameters"""
        current_points = initial_points.copy()
        current_ratio = compute_min_max_ratio(current_points)

        # Adaptive cooling schedule
        T = 0.3  # Initial temperature
        cooling_rate = 0.9995  # Cooling rate
        min_temp = 1e-6

        best_points = current_points.copy()
        best_ratio = current_ratio

        # Track recent improvements for adaptive cooling
        recent_improvements = []
        improvement_window = 50
        stagnation_counter = 0
        max_stagnation = 200

        # Early stopping
        last_improvement = 0
        patience = 1000

        for iteration in range(max_iter):
            # Adaptive cooling based on progress
            if len(recent_improvements) >= improvement_window:
                avg_improvement = np.mean(recent_improvements[-improvement_window:])
                if avg_improvement < 1e-6:
                    stagnation_counter += 1
                    if stagnation_counter > max_stagnation:
                        T *= 0.95  # Cool faster when stuck
                        stagnation_counter = 0
                else:
                    stagnation_counter = 0
                    # Warm up occasionally if there's consistent improvement
                    if avg_improvement > 1e-4 and np.random.rand() < 0.02:
                        T = min(T * 1.05, 0.5)  # Warm up

            # Standard cooling
            T *= cooling_rate

            if T < min_temp:
                break

            # Choose perturbation strategy
            if np.random.rand() < 0.3:
                # Neighborhood-based perturbation (more coordinated)
                new_points = current_points.copy()

                # Select two nearby points with improved selection criteria
                candidates = list(range(len(current_points)))
                np.random.shuffle(candidates)

                selected_pair = None
                # Look for a closer pair of points to ensure meaningful coordination
                for i in range(min(len(candidates), 8)):  # Limit search scope
                    idx1 = candidates[i]
                    for j in range(i+1, min(len(candidates), 12)):  # Limit second search
                        idx2 = candidates[j]
                        dist = np.sqrt(np.sum((current_points[idx1] - current_points[idx2])**2))
                        if dist < 0.2:  # Tighter distance constraint for coordination
                            selected_pair = (idx1, idx2)
                            break
                    if selected_pair:
                        break

                if selected_pair:
                    idx1, idx2 = selected_pair
                    perturbation_magnitude = T * 0.08

                    # Move both points together with some correlation
                    delta = np.random.normal(0, perturbation_magnitude, 2)
                    correlation = 0.4
                    delta1 = delta * (1 - correlation) + np.random.normal(0, perturbation_magnitude, 2) * correlation
                    delta2 = delta * (1 - correlation) + np.random.normal(0, perturbation_magnitude, 2) * correlation

                    new_points[idx1] += delta1
                    new_points[idx2] += delta2

                    # Boundary enforcement with improved handling
                    for idx in [idx1, idx2]:
                        new_points[idx, 0] = np.clip(new_points[idx, 0], 0, 1)
                        new_points[idx, 1] = np.clip(new_points[idx, 1], 0, 1)
                        # Prevent sticking to boundaries (more aggressive)
                        if new_points[idx, 0] < 0.005:
                            new_points[idx, 0] = 0.005 + np.random.rand() * 0.01
                        elif new_points[idx, 0] > 0.995:
                            new_points[idx, 0] = 0.995 - np.random.rand() * 0.01
                        if new_points[idx, 1] < 0.005:
                            new_points[idx, 1] = 0.005 + np.random.rand() * 0.01
                        elif new_points[idx, 1] > 0.995:
                            new_points[idx, 1] = 0.995 - np.random.rand() * 0.01
                else:
                    # Fallback to single point (improved boundary handling)
                    idx = np.random.randint(len(current_points))
                    new_points = current_points.copy()
                    perturbation_magnitude = T * 0.1
                    new_points[idx, 0] += np.random.normal(0, perturbation_magnitude)
                    new_points[idx, 1] += np.random.normal(0, perturbation_magnitude)
                    new_points[:, 0] = np.clip(new_points[:, 0], 0, 1)
                    new_points[:, 1] = np.clip(new_points[:, 1], 0, 1)
                    # Prevent sticking to boundaries with stronger correction
                    if new_points[idx, 0] < 0.005:
                        new_points[idx, 0] = 0.005 + np.random.rand() * 0.01
                    elif new_points[idx, 0] > 0.995:
                        new_points[idx, 0] = 0.995 - np.random.rand() * 0.01
                    if new_points[idx, 1] < 0.005:
                        new_points[idx, 1] = 0.005 + np.random.rand() * 0.01
                    elif new_points[idx, 1] > 0.995:
                        new_points[idx, 1] = 0.995 - np.random.rand() * 0.01
            else:
                # Single point perturbation
                idx = np.random.randint(len(current_points))
                new_points = current_points.copy()
                perturbation_magnitude = T * 0.1

                # Adjust based on local density estimation
                local_density = 0
                for i in range(len(current_points)):
                    if i != idx:
                        dist = np.sqrt(np.sum((current_points[idx] - current_points[i])**2))
                        if dist < 0.2:
                            local_density += 1

                # Smaller perturbations in dense regions
                if local_density > 4:
                    perturbation_magnitude *= 0.5
                elif local_density < 2:
                    perturbation_magnitude *= 1.5

                new_points[idx, 0] += np.random.normal(0, perturbation_magnitude)
                new_points[idx, 1] += np.random.normal(0, perturbation_magnitude)

                # Boundary enforcement with soft correction for dense regions
                new_points[:, 0] = np.clip(new_points[:, 0], 0, 1)
                new_points[:, 1] = np.clip(new_points[:, 1], 0, 1)

                # Correct points that would stick to boundaries
                if new_points[idx, 0] < 0.01:
                    new_points[idx, 0] = 0.01 + np.random.rand() * 0.01
                elif new_points[idx, 0] > 0.99:
                    new_points[idx, 0] = 0.99 - np.random.rand() * 0.01
                if new_points[idx, 1] < 0.01:
                    new_points[idx, 1] = 0.01 + np.random.rand() * 0.01
                elif new_points[idx, 1] > 0.99:
                    new_points[idx, 1] = 0.99 - np.random.rand() * 0.01

            # Evaluate new solution
            new_ratio = compute_min_max_ratio(new_points)

            # Track improvements
            if new_ratio > current_ratio:
                recent_improvements.append(new_ratio - current_ratio)
                if len(recent_improvements) > improvement_window * 2:
                    recent_improvements.pop(0)

            # Accept or reject with Metropolis criterion
            if new_ratio > current_ratio or np.random.rand() < np.exp((new_ratio - current_ratio) / T):
                current_points = new_points
                current_ratio = new_ratio

                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = current_points.copy()
                    last_improvement = iteration

            # Early stopping if no improvement
            if iteration - last_improvement > patience:
                # Additional check: if we haven't improved in a long time,
                # also stop if our temperature is very low
                if T < min_temp * 10:
                    break

        return best_points, best_ratio

    def multi_scale_optimization(initial_points):
        """Multi-scale optimization with coarse to fine approach"""
        # Coarse scale optimization
        coarse_points = initial_points.copy()
        for _ in range(500):
            idx = np.random.randint(len(coarse_points))
            coarse_points[idx, 0] += np.random.normal(0, 0.05, 1)
            coarse_points[idx, 1] += np.random.normal(0, 0.05, 1)
            coarse_points[:, 0] = np.clip(coarse_points[:, 0], 0, 1)
            coarse_points[:, 1] = np.clip(coarse_points[:, 1], 0, 1)

        # Medium scale optimization
        medium_points = coarse_points.copy()
        for _ in range(1000):
            idx = np.random.randint(len(medium_points))
            medium_points[idx, 0] += np.random.normal(0, 0.01, 1)
            medium_points[idx, 1] += np.random.normal(0, 0.01, 1)
            medium_points[:, 0] = np.clip(medium_points[:, 0], 0, 1)
            medium_points[:, 1] = np.clip(medium_points[:, 1], 0, 1)

        # Fine scale optimization
        fine_points = medium_points.copy()
        for _ in range(1500):
            idx = np.random.randint(len(fine_points))
            fine_points[idx, 0] += np.random.normal(0, 0.002, 1)
            fine_points[idx, 1] += np.random.normal(0, 0.002, 1)
            fine_points[:, 0] = np.clip(fine_points[:, 0], 0, 1)
            fine_points[:, 1] = np.clip(fine_points[:, 1], 0, 1)

        return fine_points

    def enhanced_local_refinement(points):
        """Enhanced local refinement with adaptive perturbations"""
        best_points = points.copy()
        best_ratio = compute_min_max_ratio(best_points)

        # Multiple phases with decreasing step sizes
        step_sizes = [0.02, 0.01, 0.005, 0.002]

        for phase, step_size in enumerate(step_sizes):
            # More iterations in earlier phases
            iterations = 2000 // (phase + 1)

            for _ in range(iterations):
                # Select multiple points to perturb
                num_perturbations = min(5, len(best_points))
                indices = np.random.choice(len(best_points), num_perturbations, replace=False)

                new_points = best_points.copy()

                # Apply perturbations with adaptive magnitudes
                for idx in indices:
                    # Estimate local density
                    local_density = 0
                    for i in range(len(best_points)):
                        if i != idx:
                            dist = np.sqrt(np.sum((best_points[idx] - best_points[i])**2))
                            if dist < 0.2:
                                local_density += 1

                    # Adjust step size based on density
                    adaptive_step = step_size * (1.0 if local_density < 3 else 0.5)

                    new_points[idx, 0] += np.random.normal(0, adaptive_step)
                    new_points[idx, 1] += np.random.normal(0, adaptive_step)

                    # Enforce boundaries
                    new_points[idx, 0] = np.clip(new_points[idx, 0], 0, 1)
                    new_points[idx, 1] = np.clip(new_points[idx, 1], 0, 1)

                    # Prevent sticking to boundaries
                    if new_points[idx, 0] < 0.01:
                        new_points[idx, 0] = 0.01 + np.random.rand() * 0.01
                    elif new_points[idx, 0] > 0.99:
                        new_points[idx, 0] = 0.99 - np.random.rand() * 0.01
                    if new_points[idx, 1] < 0.01:
                        new_points[idx, 1] = 0.01 + np.random.rand() * 0.01
                    elif new_points[idx, 1] > 0.99:
                        new_points[idx, 1] = 0.99 - np.random.rand() * 0.01

                new_ratio = compute_min_max_ratio(new_points)

                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = new_points.copy()

        return best_points

    # Generate diverse initial configurations
    initial_configurations = create_diverse_initial_configurations()

    best_ratio = -np.inf
    best_points = None

    # Multi-start optimization using diverse initial points (increased from 8 to 12)
    for i, initial_config in enumerate(initial_configurations):
        try:
            # Apply multi-scale optimization first
            scaled_points = multi_scale_optimization(initial_config)

            # Apply adaptive simulated annealing with increased iterations
            optimized_points, ratio = adaptive_simulated_annealing(scaled_points, max_iter=1500)

            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()

        except Exception as e:
            continue

    # Final refinement
    if best_points is not None:
        refined_points = enhanced_local_refinement(best_points)
        final_ratio = compute_min_max_ratio(refined_points)

        if final_ratio > best_ratio:
            best_ratio = final_ratio
            best_points = refined_points.copy()

    # Fallback to first configuration if nothing worked
    if best_points is None:
        initial_configurations = create_diverse_initial_configurations()
        best_points = initial_configurations[0].copy()

    return best_points

# EVOLVE-BLOCK-END