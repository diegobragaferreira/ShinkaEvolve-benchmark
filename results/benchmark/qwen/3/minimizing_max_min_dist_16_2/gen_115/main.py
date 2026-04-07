# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

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

    def compute_penalized_ratio(points):
        """Compute the ratio with boundary penalties"""
        ratio = compute_min_max_ratio(points)

        # Add boundary penalty - points too close to boundaries reduce the ratio significantly
        boundary_penalty = 0
        boundary_threshold = 0.01

        # Count how many points are near boundaries
        near_boundary_count = 0
        for point in points:
            if (point[0] < boundary_threshold or point[0] > 1-boundary_threshold or
                point[1] < boundary_threshold or point[1] > 1-boundary_threshold):
                near_boundary_count += 1

        # Apply penalty based on how many points are near boundaries
        if near_boundary_count > 0:
            # More penalty for more boundary points
            boundary_penalty = near_boundary_count * 0.01 * ratio

        return max(0, ratio - boundary_penalty)

    def create_enhanced_hexagonal_grid():
        """Create enhanced hexagonal grid with better spacing and symmetry breaking"""
        # Arrange 16 points in a 4x4 grid with proper hexagonal spacing
        rows = 4
        cols = 4
        points = []

        spacing_x = 1.0
        spacing_y = np.sqrt(3) / 2

        for i in range(rows):
            for j in range(cols):
                # Offset every other row for hexagonal packing
                x = j * spacing_x + (i % 2) * spacing_x * 0.5
                y = i * spacing_y
                points.append([x, y])

        # Convert to numpy array
        points = np.array(points)

        # Normalize to [0,1] x [0,1]
        max_x = (cols - 1) + 0.5  # Account for offset in last row
        max_y = (rows - 1) * spacing_y

        points[:, 0] = points[:, 0] / max_x
        points[:, 1] = points[:, 1] / max_y

        # Add strategic perturbations to break symmetry with better randomness
        np.random.seed(42)
        noise = np.random.normal(0, 0.015, points.shape)

        # More sophisticated symmetry breaking: use a mathematical pattern
        for i in range(len(points)):
            # Apply a pattern based on index to create better asymmetry
            pattern_factor = (i * 7 + (i//4) * 3) % 10
            noise[i] *= (1.0 + 0.1 * (pattern_factor / 10.0))

        points += noise
        points = np.clip(points, 0, 1)

        return points

    def create_alternative_configurations():
        """Generate multiple alternative initial configurations"""
        configs = []

        # Configuration 1: Enhanced hexagonal grid
        configs.append(create_enhanced_hexagonal_grid())

        # Configuration 2: Random but constrained
        np.random.seed(42)
        configs.append(np.random.rand(16, 2))

        # Configuration 3: Grid with perturbations
        grid_points = create_enhanced_hexagonal_grid()
        np.random.seed(43)
        perturbations = np.random.normal(0, 0.05, (16, 2))
        configs.append(np.clip(grid_points + perturbations, 0, 1))

        # Configuration 4: Simple 4x4 uniform grid with slight perturbation
        uniform_grid = []
        for i in range(4):
            for j in range(4):
                uniform_grid.append([i/3, j/3])
        np.random.seed(44)
        perturbations = np.random.normal(0, 0.02, (16, 2))
        configs.append(np.clip(np.array(uniform_grid[:16]) + perturbations, 0, 1))

        return configs

    def multi_scale_optimization(initial_points):
        """Multi-scale optimization approach for better exploration"""
        best_points = initial_points.copy()
        best_ratio = compute_penalized_ratio(best_points)

        # Scale 1: Coarse optimization with large steps
        points = initial_points.copy()
        for iter_coarse in range(500):
            new_points = points.copy()
            # Large step size for broad search
            idx = np.random.randint(len(points))
            # Adaptive magnitude based on current best ratio
            adaptive_magnitude = 0.05 * (1.0 + best_ratio)
            new_points[idx, 0] += np.random.normal(0, adaptive_magnitude, 1)
            new_points[idx, 1] += np.random.normal(0, adaptive_magnitude, 1)
            new_points[:, 0] = np.clip(new_points[:, 0], 0, 1)
            new_points[:, 1] = np.clip(new_points[:, 1], 0, 1)

            new_ratio = compute_penalized_ratio(new_points)
            if new_ratio > best_ratio or np.random.rand() < np.exp((new_ratio - best_ratio) * 10):
                points = new_points.copy()
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = points.copy()

        # Scale 2: Medium optimization with medium steps
        for iter_medium in range(1000):
            new_points = points.copy()
            # Medium step size for local refinement
            idx = np.random.randint(len(points))
            # Adaptive magnitude based on current best ratio
            adaptive_magnitude = 0.01 * (1.0 + best_ratio)
            new_points[idx, 0] += np.random.normal(0, adaptive_magnitude, 1)
            new_points[idx, 1] += np.random.normal(0, adaptive_magnitude, 1)
            new_points[:, 0] = np.clip(new_points[:, 0], 0, 1)
            new_points[:, 1] = np.clip(new_points[:, 1], 0, 1)

            new_ratio = compute_penalized_ratio(new_points)
            if new_ratio > best_ratio or np.random.rand() < np.exp((new_ratio - best_ratio) * 50):
                points = new_points.copy()
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = points.copy()

        # Scale 3: Fine optimization with small steps
        for iter_fine in range(1500):
            new_points = points.copy()
            # Small step size for fine adjustment
            idx = np.random.randint(len(points))
            # Adaptive magnitude based on current best ratio
            adaptive_magnitude = 0.002 * (1.0 + best_ratio)
            new_points[idx, 0] += np.random.normal(0, adaptive_magnitude, 1)
            new_points[idx, 1] += np.random.normal(0, adaptive_magnitude, 1)
            new_points[:, 0] = np.clip(new_points[:, 0], 0, 1)
            new_points[:, 1] = np.clip(new_points[:, 1], 0, 1)

            new_ratio = compute_penalized_ratio(new_points)
            if new_ratio > best_ratio or np.random.rand() < np.exp((new_ratio - best_ratio) * 100):
                points = new_points.copy()
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = points.copy()

        return best_points

    def enhanced_simulated_annealing(initial_points, max_iter=3000):
        """Enhanced simulated annealing with adaptive cooling and mixed perturbations"""
        current_points = initial_points.copy()
        current_ratio = compute_penalized_ratio(current_points)

        # Better cooling schedule with adaptive parameters
        T = 0.2  # Higher initial temperature for extensive exploration
        cooling_rate = 0.9992  # Moderate cooling rate
        min_temp = 1e-6

        best_points = current_points.copy()
        best_ratio = current_ratio

        # Track recent improvements for adaptive cooling
        recent_improvements = []
        improvement_window = 50

        for iteration in range(max_iter):
            # Adaptive cooling based on recent performance
            if len(recent_improvements) >= improvement_window:
                avg_improvement = np.mean(recent_improvements[-improvement_window:])
                if avg_improvement < 1e-5:  # Stagnation detected
                    T *= 0.95  # Cool faster if stagnating
                elif avg_improvement > 1e-4:  # Good progress
                    T *= 1.01  # Warm up occasionally to escape local minima

            T *= cooling_rate

            if T < min_temp:
                break

            # Try different types of perturbations for diversity
            perturbation_type = np.random.choice(['single', 'neighborhood', 'cluster'], p=[0.6, 0.25, 0.15])

            new_points = current_points.copy()
            accepted = False

            if perturbation_type == 'single':
                # Single point perturbation with adaptive magnitude
                idx = np.random.randint(len(current_points))
                # Adaptively adjust magnitude based on local density and ratio
                local_density = estimate_local_density(current_points, idx, 0.2)
                adaptive_magnitude = T * 0.1 * (1.0 + 0.5 * (1.0 / (local_density + 1)))

                new_points[idx, 0] += np.random.normal(0, adaptive_magnitude)
                new_points[idx, 1] += np.random.normal(0, adaptive_magnitude)

                # Enforce boundaries with reflection
                for i in range(len(new_points)):
                    for j in range(2):
                        if new_points[i, j] < 0:
                            new_points[i, j] = -new_points[i, j]  # Reflect
                        elif new_points[i, j] > 1:
                            new_points[i, j] = 2 - new_points[i, j]  # Reflect

                accepted = True

            elif perturbation_type == 'neighborhood':
                # Neighborhood-based perturbation for coordinated moves
                # Select two points that are relatively close
                candidates = list(range(len(current_points)))
                np.random.shuffle(candidates)

                # Find a pair of points that are reasonably close
                selected_pair = None
                for i in range(len(candidates)-1):
                    idx1 = candidates[i]
                    for j in range(i+1, len(candidates)):
                        idx2 = candidates[j]
                        dist = np.sqrt(np.sum((current_points[idx1] - current_points[idx2])**2))
                        if dist < 0.25:  # Only consider nearby pairs
                            selected_pair = (idx1, idx2)
                            break
                    if selected_pair:
                        break

                if selected_pair is not None:
                    idx1, idx2 = selected_pair
                    perturbation_magnitude = T * 0.05

                    # Coordinate the movement of both points
                    delta1 = np.random.normal(0, perturbation_magnitude, 2)
                    delta2 = np.random.normal(0, perturbation_magnitude, 2)

                    # Add some correlation to make movements more meaningful
                    correlation_factor = 0.3
                    delta1 = delta1 * (1 - correlation_factor) + delta2 * correlation_factor
                    delta2 = delta2 * (1 - correlation_factor) + delta1 * correlation_factor

                    new_points[idx1, :] += delta1
                    new_points[idx2, :] += delta2

                    # Enforce boundaries with reflection
                    for i in range(len(new_points)):
                        for j in range(2):
                            if new_points[i, j] < 0:
                                new_points[i, j] = -new_points[i, j]  # Reflect
                            elif new_points[i, j] > 1:
                                new_points[i, j] = 2 - new_points[i, j]  # Reflect

                    accepted = True
                else:
                    # Fall back to single point
                    idx = np.random.randint(len(current_points))
                    perturbation_magnitude = T * 0.1
                    new_points[idx, 0] += np.random.normal(0, perturbation_magnitude)
                    new_points[idx, 1] += np.random.normal(0, perturbation_magnitude)

                    # Enforce boundaries with reflection
                    for i in range(len(new_points)):
                        for j in range(2):
                            if new_points[i, j] < 0:
                                new_points[i, j] = -new_points[i, j]  # Reflect
                            elif new_points[i, j] > 1:
                                new_points[i, j] = 2 - new_points[i, j]  # Reflect

                    accepted = True

            else:  # cluster perturbation
                # Perturb several points together to explore larger neighborhoods
                num_cluster_points = min(3, len(current_points) // 4)
                cluster_indices = np.random.choice(len(current_points), num_cluster_points, replace=False)

                for idx in cluster_indices:
                    perturbation_magnitude = T * 0.03
                    new_points[idx, 0] += np.random.normal(0, perturbation_magnitude)
                    new_points[idx, 1] += np.random.normal(0, perturbation_magnitude)

                # Enforce boundaries with reflection
                for i in range(len(new_points)):
                    for j in range(2):
                        if new_points[i, j] < 0:
                            new_points[i, j] = -new_points[i, j]  # Reflect
                        elif new_points[i, j] > 1:
                            new_points[i, j] = 2 - new_points[i, j]  # Reflect

                accepted = True

            if accepted:
                # Accept or reject the new solution
                new_ratio = compute_penalized_ratio(new_points)

                # Track recent improvements
                if new_ratio > current_ratio:
                    recent_improvements.append(new_ratio - current_ratio)
                    if len(recent_improvements) > improvement_window * 2:
                        recent_improvements.pop(0)

                # Metropolis criterion
                if new_ratio > current_ratio or np.random.rand() < np.exp((new_ratio - current_ratio) / T):
                    current_points = new_points
                    current_ratio = new_ratio

                    if current_ratio > best_ratio:
                        best_ratio = current_ratio
                        best_points = current_points.copy()

        return best_points, best_ratio

    def estimate_local_density(points, center_idx, radius):
        """Estimate local density around a point"""
        count = 0
        center_point = points[center_idx]

        for i, point in enumerate(points):
            if i != center_idx:
                distance = np.sqrt(np.sum((center_point - point)**2))
                if distance <= radius:
                    count += 1

        return count

    # Generate multiple initial configurations
    initial_configs = create_alternative_configurations()

    best_ratio = -np.inf
    best_points = None

    # Try each initial configuration with optimization
    for i, initial_points in enumerate(initial_configs):
        # Clip initial points to valid bounds
        initial_points = np.clip(initial_points, 0, 1)

        # Apply multi-scale optimization first
        optimized_points = multi_scale_optimization(initial_points)
        ratio = compute_min_max_ratio(optimized_points)

        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()

    # Final optimization with enhanced simulated annealing
    if best_points is not None:
        final_points, final_ratio = enhanced_simulated_annealing(best_points)
        if final_ratio > best_ratio:
            best_ratio = final_ratio
            best_points = final_points.copy()

    # If no optimization succeeded, return the best initial configuration
    if best_points is None:
        # Fallback to simple enhanced hexagonal configuration
        best_points = create_enhanced_hexagonal_grid()

    return best_points

# EVOLVE-BLOCK-END