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

    def compute_min_max_ratio(points):
        """Computes the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0

        # Compute pairwise distances
        distances = pdist(points)
        dmin = np.min(distances)
        dmax = np.max(distances)

        # Handle edge case where all points are identical
        if dmax == 0:
            return 0.0

        return dmin / dmax

    def compute_boundary_penalty(points, penalty_weight=15.0):
        """Computes penalty for points near boundaries with logarithmic scaling for smoother transitions."""
        penalty = 0
        for point in points:
            # Penalty for being close to any boundary
            dist_to_boundaries = [
                point[0],  # distance to left boundary
                1 - point[0],  # distance to right boundary
                point[1],  # distance to bottom boundary
                1 - point[1]   # distance to top boundary
            ]
            min_dist = min(dist_to_boundaries)
            if min_dist < 0.01:  # Only penalize if very close to boundary
                # Use logarithmic penalty for smooth transition
                penalty += penalty_weight * np.log(1 + 100 * (0.01 - min_dist))
        return penalty

    def evaluate_with_penalty(points, penalty_weight=10.0):
        """Evaluate ratio with boundary penalty applied."""
        ratio = compute_min_max_ratio(points)
        penalty = compute_boundary_penalty(points, penalty_weight)
        return ratio - penalty

    def generate_hexagonal_grid():
        """Generate a hexagonal grid pattern."""
        points = []
        sqrt3 = np.sqrt(3)

        # 4x4 hexagonal pattern inspired by L-BFGS approach
        for i in range(4):
            for j in range(4):
                x = j + 0.5 * (i % 2)
                y = i * sqrt3 / 2
                points.append([x, y])

        points = np.array(points)

        # Normalize to [0,1] x [0,1] with better aspect ratio preservation
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])

        if x_range > 0:
            points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
        if y_range > 0:
            points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range

        # Scale to fit nicely within [0.05, 0.95] x [0.05, 0.95]
        points[:, 0] = points[:, 0] * 0.9 + 0.05
        points[:, 1] = points[:, 1] * 0.9 + 0.05

        return points

    def initialize_multiple_strategies():
        """Initialize multiple starting configurations with enhanced diversity."""
        strategies = []

        # Strategy 1: Standard hexagonal grid with moderate perturbation
        base_grid = generate_hexagonal_grid()
        np.random.seed(42)
        perturbed = base_grid + np.random.normal(0, 0.01, base_grid.shape)
        perturbed = np.clip(perturbed, 0, 1)
        strategies.append(("hex", perturbed))

        # Strategy 2: Hexagonal with higher noise
        np.random.seed(123)
        perturbed_high = base_grid + np.random.normal(0, 0.02, base_grid.shape)
        perturbed_high = np.clip(perturbed_high, 0, 1)
        strategies.append(("hex_high", perturbed_high))

        # Strategy 3: Random initialization
        np.random.seed(456)
        random_points = np.random.rand(16, 2)
        strategies.append(("random", random_points))

        # Strategy 4: Triangular lattice variation
        triangular_points = []
        sqrt3 = np.sqrt(3)
        for i in range(4):
            for j in range(4):
                x = j + 0.5 * (i % 2)
                y = i * sqrt3 / 2
                triangular_points.append([x, y])
        triangular_points = np.array(triangular_points[:16])

        # Normalize triangular points properly
        x_range = np.max(triangular_points[:, 0]) - np.min(triangular_points[:, 0])
        y_range = np.max(triangular_points[:, 1]) - np.min(triangular_points[:, 1])
        if x_range > 0:
            triangular_points[:, 0] = (triangular_points[:, 0] - np.min(triangular_points[:, 0])) / x_range * 0.9 + 0.05
        if y_range > 0:
            triangular_points[:, 1] = (triangular_points[:, 1] - np.min(triangular_points[:, 1])) / y_range * 0.9 + 0.05
        strategies.append(("triangular", triangular_points))

        # Strategy 5: Hexagonal with lower noise
        np.random.seed(789)
        perturbed_low = base_grid + np.random.normal(0, 0.005, base_grid.shape)
        perturbed_low = np.clip(perturbed_low, 0, 1)
        strategies.append(("hex_low", perturbed_low))

        # Strategy 6: Perturbed hexagonal with position-dependent noise
        np.random.seed(999)
        position_dependent = base_grid.copy()
        for i in range(len(position_dependent)):
            # Different noise levels based on position to break symmetry more effectively
            row = i // 4
            col = i % 4
            noise_level = 0.01 + 0.005 * (row + col) / 8
            position_dependent[i] += np.random.normal(0, noise_level, 2)
        position_dependent = np.clip(position_dependent, 0, 1)
        strategies.append(("hex_position", position_dependent))

        # Strategy 7: Slightly different random initialization
        np.random.seed(333)
        random_points2 = np.random.rand(16, 2)
        strategies.append(("random2", random_points2))

        # Strategy 8: Mirror-symmetric configuration
        mirror_points = []
        base_pattern = [
            [0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8],
            [0.4, 0.3], [0.6, 0.3], [0.4, 0.7], [0.6, 0.7],
            [0.3, 0.4], [0.3, 0.6], [0.7, 0.4], [0.7, 0.6],
            [0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5]
        ]
        for point in base_pattern:
            mirror_points.append(point)
        mirror_points = np.array(mirror_points)
        # Add small noise
        np.random.seed(555)
        mirror_points += np.random.normal(0, 0.01, mirror_points.shape)
        mirror_points = np.clip(mirror_points, 0, 1)
        strategies.append(("mirror", mirror_points))

        return strategies

    def neighborhood_move(current_points, point_indices, step_size=0.01, momentum=0.7):
        """Performs a coordinated move on a cluster of points with momentum preservation."""
        new_points = current_points.copy()

        # Calculate centroid of selected points
        centroid = np.mean(current_points[point_indices], axis=0)

        # Generate movement vector with momentum
        if hasattr(neighborhood_move, 'last_move'):
            # Preserve some momentum from previous move
            move_vector = momentum * neighborhood_move.last_move + (1 - momentum) * np.random.normal(0, step_size, 2)
        else:
            move_vector = np.random.normal(0, step_size, 2)

        # Store for next iteration
        neighborhood_move.last_move = move_vector

        # Apply movement to selected points
        for idx in point_indices:
            new_points[idx] = current_points[idx] + move_vector

            # Boundary handling with reflection and momentum conservation
            for dim in range(2):
                if new_points[idx, dim] < 0:
                    # Reflect and preserve momentum
                    new_points[idx, dim] = -new_points[idx, dim] + 0.1 * move_vector[dim]
                elif new_points[idx, dim] > 1:
                    # Reflect and preserve momentum
                    new_points[idx, dim] = 2 - new_points[idx, dim] - 0.1 * move_vector[dim]

        return new_points

    def adaptive_simulated_annealing(initial_points, max_iterations=5000, initial_temp=0.15):
        """Enhanced simulated annealing with adaptive cooling and neighborhood moves."""
        current_points = initial_points.copy()
        current_ratio = evaluate_with_penalty(current_points)

        best_points = current_points.copy()
        best_ratio = current_ratio

        temperature = initial_temp
        cooling_rate = 0.9995
        min_temp = 1e-8

        # Track recent improvements for adaptive cooling
        recent_improvements = []
        improvement_threshold = 50
        stagnation_counter = 0

        # Track progress for adaptive cooling
        last_best_ratio = current_ratio
        progress_stagnation = 0
        max_progress_stagnation = 100

        # Phase-based parameters for adaptive cooling
        phase = 0  # Phase tracking for adaptive cooling
        phase_params = [
            {'cooling': 0.9995, 'step_factor': 0.15},  # Phase 0: Aggressive
            {'cooling': 0.9992, 'step_factor': 0.12},  # Phase 1: Moderate
            {'cooling': 0.9990, 'step_factor': 0.08},  # Phase 2: Fine tuning
        ]

        for iteration in range(max_iterations):
            # Adaptive temperature adjustment based on progress and phase
            if iteration % 100 == 0 and iteration > 0:
                # Determine current phase based on progress and iteration
                if iteration < max_iterations * 0.3:
                    phase = 0
                elif iteration < max_iterations * 0.7:
                    phase = 1
                else:
                    phase = 2

                # Check for improvement
                ratio_diff = current_ratio - last_best_ratio
                if ratio_diff > 1e-8:
                    # There was an improvement - speed up cooling slightly
                    cooling_rate = min(phase_params[phase]['cooling'] * 1.005, 0.9998)
                    last_best_ratio = current_ratio
                    progress_stagnation = 0
                else:
                    # No improvement - slow down cooling and track stagnation
                    cooling_rate = max(phase_params[phase]['cooling'] * 0.99, 0.999)
                    progress_stagnation += 1

                    # If stagnating for too long, restart with higher temperature
                    if progress_stagnation > max_progress_stagnation:
                        temperature = min(temperature * 2.0, 0.5)
                        progress_stagnation = 0

            # Determine step size based on temperature and phase
            step_size = temperature * phase_params[phase]['step_factor']

            # Decide between single point or neighborhood move
            if np.random.random() < 0.7:  # 70% chance of neighborhood move
                # Select random subset of points for neighborhood move
                num_selected = np.random.randint(2, 6)  # 2 to 5 points
                point_indices = np.random.choice(len(current_points), size=num_selected, replace=False)

                # Perform neighborhood move with momentum
                new_points = neighborhood_move(current_points, point_indices, step_size=step_size, momentum=0.7)
            else:
                # Single point move (traditional approach)
                new_points = current_points.copy()
                point_idx = np.random.randint(len(current_points))
                delta = np.random.normal(0, step_size, 2)
                new_points[point_idx] = current_points[point_idx] + delta

                # Boundary handling with reflection
                for dim in range(2):
                    if new_points[point_idx, dim] < 0:
                        new_points[point_idx, dim] = -new_points[point_idx, dim]
                    elif new_points[point_idx, dim] > 1:
                        new_points[point_idx, dim] = 2 - new_points[point_idx, dim]

            # Evaluate new solution
            new_ratio = evaluate_with_penalty(new_points)

            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio:
                current_points = new_points
                current_ratio = new_ratio

                if new_ratio > best_ratio:
                    best_points = new_points.copy()
                    best_ratio = new_ratio
                    stagnation_counter = 0
            else:
                if np.random.random() < np.exp((new_ratio - current_ratio) / temperature):
                    current_points = new_points
                    current_ratio = new_ratio
                else:
                    stagnation_counter += 1

            # Adaptive cooling: If no improvement for a while, cool faster
            if stagnation_counter > 50:
                cooling_rate = min(cooling_rate * 0.99, 0.9999)
                stagnation_counter = 0

            # Cool down temperature
            temperature *= cooling_rate
            if temperature < min_temp:
                temperature = min_temp

            # Early stopping if we're not improving much
            recent_improvements.append(1 if new_ratio > current_ratio else 0)
            if len(recent_improvements) > improvement_threshold:
                recent_improvements.pop(0)
                if sum(recent_improvements) < 2 and iteration > 1000:
                    break

        return best_points, best_ratio

    # Generate multiple initializations
    strategies = initialize_multiple_strategies()

    # Run optimization from each starting point
    best_result = None
    best_score = -np.inf

    for strategy_name, initial_points in strategies:
        try:
            optimized_points, score = adaptive_simulated_annealing(
                initial_points, max_iterations=3000, initial_temp=0.05
            )

            if score > best_score:
                best_score = score
                best_result = optimized_points

        except Exception as e:
            continue  # Skip failed runs

    # Final refinement with the best result
    if best_result is not None:
        try:
            final_points, _ = adaptive_simulated_annealing(
                best_result, max_iterations=2000, initial_temp=0.02
            )
            return final_points
        except:
            pass

    # Fallback to best found if optimization fails
    if best_result is not None:
        return best_result

    # Last resort: return a basic hexagonal grid with small perturbation
    base_grid = generate_hexagonal_grid()
    np.random.seed(42)
    fallback = base_grid + np.random.normal(0, 0.005, base_grid.shape)
    return np.clip(fallback, 0, 1)

# EVOLVE-BLOCK-END