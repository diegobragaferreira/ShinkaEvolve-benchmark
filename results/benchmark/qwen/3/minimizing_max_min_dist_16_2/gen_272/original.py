# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining hexagonal initialization with adaptive optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances"""
        distances = pdist(points)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return np.min(distances) / max_dist

    def create_hexagonal_grid():
        """Create initial configuration using an optimized hexagonal grid pattern"""
        # Create a more sophisticated hexagonal grid for 16 points
        points = []
        sqrt3 = np.sqrt(3)

        # Arrange points in a hexagonal pattern that better fills the unit square
        # Using 4 rows with alternating column offsets for true hexagonal packing
        rows = 4
        cols = 4

        # Calculate proper hexagonal spacing for optimal packing in unit square
        spacing_x = 1.0  # Will be adjusted
        spacing_y = sqrt3 / 2.0  # Height of hexagon

        for i in range(rows):
            for j in range(cols):
                x = j * spacing_x + (i % 2) * spacing_x * 0.5
                y = i * spacing_y

                # Add subtle perturbations to break perfect symmetry
                # These values are chosen to be small but effective for symmetry breaking
                if (i + j) % 3 == 0:
                    x += 0.005 * np.random.randn()
                    y += 0.005 * np.random.randn()
                elif (i + j) % 3 == 1:
                    x -= 0.003 * np.random.randn()
                    y += 0.003 * np.random.randn()

                points.append([x, y])

        points = np.array(points[:16])  # Take first 16 points

        # Normalize to fit within [0,1] x [0,1] properly
        # Find the bounding box
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

        # Avoid division by zero
        if (x_max - x_min) > 0:
            points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
        if (y_max - y_min) > 0:
            points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)

        # Center the points in the unit square to avoid edge effects
        x_center = np.mean(points[:, 0])
        y_center = np.mean(points[:, 1])
        points[:, 0] -= x_center - 0.5
        points[:, 1] -= y_center - 0.5

        # Clamp to ensure all points are within bounds
        points = np.clip(points, 0, 1)

        return points

    def create_diverse_initial_configurations():
        """Create diverse initial configurations"""
        configurations = []

        # Strategy 1: Base hexagonal grid
        configurations.append(create_hexagonal_grid())

        # Strategy 2: Random with boundary awareness
        np.random.seed(42)
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

        # Strategy 3: Perturbed hexagonal grid
        hex_points = create_hexagonal_grid()
        np.random.seed(43)
        perturbations = np.random.normal(0, 0.01, (16, 2))
        configurations.append(np.clip(hex_points + perturbations, 0, 1))

        # Strategy 4: Uniform grid with noise
        uniform_grid = []
        for i in range(4):
            for j in range(4):
                uniform_grid.append([i/3, j/3])
        uniform_points = np.array(uniform_grid[:16])
        np.random.seed(44)
        noise = np.random.normal(0, 0.015, (16, 2))
        configurations.append(np.clip(uniform_points + noise, 0, 1))

        # Strategy 5: Triangular lattice
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
        # Normalize triangular lattice
        max_x = (cols - 1) + 0.5
        max_y = (rows - 1) * spacing_y
        triangular_points[:, 0] = triangular_points[:, 0] / max_x
        triangular_points[:, 1] = triangular_points[:, 1] / max_y
        configurations.append(np.clip(triangular_points[:16], 0, 1))

        # Strategy 6: Random with different seed
        np.random.seed(123)
        configurations.append(np.random.rand(16, 2).copy())

        return configurations

    def adaptive_optimization(initial_points, max_iter=2000):
        """Enhanced optimization with adaptive parameters"""
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
        improvement_window = 30
        stagnation_counter = 0
        max_stagnation = 200

        # Early stopping
        last_improvement = 0
        patience = 500

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

                # Select two nearby points
                candidates = list(range(len(current_points)))
                np.random.shuffle(candidates)

                selected_pair = None
                for i in range(len(candidates)-1):
                    idx1 = candidates[i]
                    for j in range(i+1, len(candidates)):
                        idx2 = candidates[j]
                        dist = np.sqrt(np.sum((current_points[idx1] - current_points[idx2])**2))
                        if dist < 0.2:  # Within reasonable distance
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
                    # Prevent sticking to boundaries
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

    # Multi-start optimization using diverse initial points
    for i, initial_config in enumerate(initial_configurations):
        try:
            # Apply multi-scale optimization first
            scaled_points = multi_scale_optimization(initial_config)

            # Then apply adaptive optimization
            optimized_points, ratio = adaptive_optimization(scaled_points, max_iter=1500)

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