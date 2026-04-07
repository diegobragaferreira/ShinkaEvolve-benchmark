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

    def compute_min_max_ratio(points):
        """Computes the ratio of minimum to maximum pairwise distances using efficient cdist."""
        if len(points) < 2:
            return 0.0

        # Use cdist for more efficient pairwise distance calculation
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances

        if distances.size == 0:
            return 0

        d_min = np.min(distances)
        d_max = np.max(distances)

        # Handle edge case where all points are identical
        if d_max == 0:
            return 0.0

        return d_min / d_max

    def compute_boundary_penalty(points, penalty_weight=10.0):
        """Computes penalty for points near boundaries with improved scaling."""
        penalty = 0.0
        for point in points:
            # Penalty for being close to any boundary
            dist_to_boundaries = [
                point[0],  # distance to left boundary
                1 - point[0],  # distance to right boundary
                point[1],  # distance to bottom boundary
                1 - point[1]   # distance to top boundary
            ]
            min_dist = min(dist_to_boundaries)

            # Soft penalty that increases smoothly as points approach boundaries
            if min_dist < 0.01:
                # Use exponential penalty for smoother transition
                penalty += penalty_weight * np.exp(-min_dist / 0.005) * (0.01 - min_dist)
        return penalty

    def evaluate_with_penalty(points, penalty_weight=10.0):
        """Evaluate ratio with boundary penalty applied."""
        ratio = compute_min_max_ratio(points)
        penalty = compute_boundary_penalty(points, penalty_weight)
        return ratio - penalty

    def generate_hexagonal_grid():
        """Generate a hexagonal grid pattern with improved mathematical foundation."""
        points = []
        sqrt3 = np.sqrt(3)

        # 4x4 hexagonal pattern with optimized spacing
        base_spacing_x = 1.0
        base_spacing_y = base_spacing_x * sqrt3 / 2.0

        for i in range(4):
            for j in range(4):
                x = j * base_spacing_x + (i % 2) * base_spacing_x / 2.0
                y = i * base_spacing_y
                points.append([x, y])

        points = np.array(points)

        # Normalize to [0,1] x [0,1] with better aspect ratio preservation
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])

        if x_range > 0:
            points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
        if y_range > 0:
            points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range

        # Scale to fit nicely within [0.05, 0.95] x [0.05, 0.95] for better interior distribution
        points[:, 0] = points[:, 0] * 0.9 + 0.05
        points[:, 1] = points[:, 1] * 0.9 + 0.05

        return points

    def enhance_symmetry_breaking(points, seed=42):
        """Apply enhanced symmetry breaking to the point configuration with mathematical functions."""
        np.random.seed(seed)

        # Apply more sophisticated perturbations based on position
        enhanced_points = points.copy()

        # Position-dependent perturbations with mathematical functions
        for i in range(len(enhanced_points)):
            row = i // 4
            col = i % 4

            # Create structured perturbations that break symmetry but maintain hexagonal character
            # Use trigonometric functions for mathematical structure
            pos_dependent_pert = np.array([
                0.005 * np.sin(row * 0.5) * np.cos(col * 0.3),
                0.005 * np.cos(row * 0.3) * np.sin(col * 0.5)
            ])

            # Add some random noise as well
            random_pert = np.random.normal(0, 0.003, 2)

            # Apply combined perturbation
            enhanced_points[i] += pos_dependent_pert + random_pert

            # Ensure points stay within bounds
            enhanced_points[i] = np.clip(enhanced_points[i], 0, 1)

        return enhanced_points

    def generate_triangular_lattice():
        """Generate triangular lattice pattern with proper normalization."""
        points = []
        sqrt3 = np.sqrt(3)
        
        # Create triangular lattice with 4x4 grid
        spacing_x = 1.0 / 3.0
        spacing_y = spacing_x * sqrt3 / 2
        
        for i in range(4):
            for j in range(4):
                x = j * spacing_x
                if i % 2 == 1:
                    x += spacing_x / 2
                y = i * spacing_y
                points.append([x, y])
        
        points = np.array(points[:16])
        
        # Normalize to [0,1] x [0,1]
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

        # Strategy 1: Standard hexagonal grid with enhanced symmetry breaking
        base_grid = generate_hexagonal_grid()
        enhanced_points = enhance_symmetry_breaking(base_grid, seed=42)
        strategies.append(("hex_enhanced", enhanced_points))

        # Strategy 2: Hexagonal with higher noise and enhanced symmetry breaking
        np.random.seed(123)
        perturbed_high = base_grid + np.random.normal(0, 0.02, base_grid.shape)
        perturbed_high = np.clip(perturbed_high, 0, 1)
        enhanced_points_high = enhance_symmetry_breaking(perturbed_high, seed=123)
        strategies.append(("hex_high_enhanced", enhanced_points_high))

        # Strategy 3: Random initialization
        np.random.seed(456)
        random_points = np.random.rand(16, 2)
        strategies.append(("random", random_points))

        # Strategy 4: Triangular lattice variation with enhanced symmetry breaking
        triangular_points = generate_triangular_lattice()
        enhanced_triangular = enhance_symmetry_breaking(triangular_points, seed=456)
        strategies.append(("triangular_enhanced", enhanced_triangular))

        # Strategy 5: Hexagonal with lower noise and enhanced symmetry breaking
        np.random.seed(789)
        perturbed_low = base_grid + np.random.normal(0, 0.005, base_grid.shape)
        perturbed_low = np.clip(perturbed_low, 0, 1)
        enhanced_points_low = enhance_symmetry_breaking(perturbed_low, seed=789)
        strategies.append(("hex_low_enhanced", enhanced_points_low))

        # Strategy 6: Alternative hexagonal with sine/cosine perturbations
        np.random.seed(999)
        base_grid_alt = generate_hexagonal_grid()
        enhanced_points_alt = base_grid_alt.copy()
        for i in range(len(enhanced_points_alt)):
            row = i // 4
            col = i % 4
            perturbation = np.array([
                0.008 * np.sin(row * 0.7) * np.cos(col * 0.4),
                0.008 * np.cos(row * 0.4) * np.sin(col * 0.7)
            ])
            noise = np.random.normal(0, 0.003, 2)
            enhanced_points_alt[i] += perturbation + noise
            enhanced_points_alt[i] = np.clip(enhanced_points_alt[i], 0, 1)
        strategies.append(("hex_alt_pert", enhanced_points_alt))

        return strategies

    def neighborhood_move(current_points, point_indices, step_size=0.01):
        """Performs a coordinated move on a cluster of points with adaptive selection."""
        new_points = current_points.copy()

        # Calculate centroid of selected points for better movement coordination
        centroid = np.mean(current_points[point_indices], axis=0)

        # Generate movement vector (same for all selected points)
        move_vector = np.random.normal(0, step_size, 2)

        # Apply movement to selected points
        for idx in point_indices:
            new_points[idx] = current_points[idx] + move_vector

            # Boundary handling with more sophisticated reflection
            for dim in range(2):
                if new_points[idx, dim] < 0.01:
                    # Push away from boundary with small random component
                    new_points[idx, dim] = 0.01 + np.random.uniform(0, 0.005)
                elif new_points[idx, dim] > 0.99:
                    # Push away from boundary with small random component
                    new_points[idx, dim] = 0.99 - np.random.uniform(0, 0.005)

        return new_points

    def adaptive_simulated_annealing(initial_points, max_iterations=5000, initial_temp=0.1):
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

        # Progress tracking variables
        last_best_ratio = current_ratio
        progress_stagnation = 0
        max_progress_stagnation = 100

        for iteration in range(max_iterations):
            # Adaptive temperature adjustment based on progress and iteration stage
            if iteration % 100 == 0 and iteration > 0:
                # Check for improvement
                ratio_diff = current_ratio - last_best_ratio
                if ratio_diff > 1e-8:
                    # There was an improvement - speed up cooling slightly
                    cooling_rate = min(cooling_rate * 1.005, 0.9998)
                    last_best_ratio = current_ratio
                    progress_stagnation = 0
                else:
                    # No improvement - slow down cooling and track stagnation
                    cooling_rate = max(cooling_rate * 0.99, 0.999)
                    progress_stagnation += 1

                    # If stagnating for too long, restart with higher temperature
                    if progress_stagnation > max_progress_stagnation:
                        temperature = min(temperature * 2.0, 0.5)
                        progress_stagnation = 0

                # Position-dependent temperature adjustment
                # More aggressive cooling in later phases
                if iteration > max_iterations * 0.7:
                    cooling_rate = min(cooling_rate * 0.995, 0.9995)
                elif iteration > max_iterations * 0.5:
                    cooling_rate = min(cooling_rate * 0.998, 0.9997)

            # Decide between single point or neighborhood move with probability
            if np.random.random() < 0.7:  # 70% chance of neighborhood move
                # Select random subset of points for neighborhood move
                num_selected = np.random.randint(2, 5)  # 2 to 4 points
                point_indices = np.random.choice(len(current_points), size=num_selected, replace=False)

                # Perform neighborhood move
                new_points = neighborhood_move(current_points, point_indices, step_size=temperature * 0.1)
            else:
                # Single point move with enhanced strategy
                new_points = current_points.copy()
                point_idx = np.random.randint(len(current_points))

                # Choose perturbation type adaptively based on temperature
                if temperature > 0.05:
                    # High temperature: use larger, more diverse perturbations
                    perturbation_type = np.random.choice(['normal', 'uniform', 'exponential'])
                    if perturbation_type == 'normal':
                        delta = np.random.normal(0, temperature * 0.15, 2)
                    elif perturbation_type == 'uniform':
                        delta = np.random.uniform(-temperature * 0.2, temperature * 0.2, 2)
                    else:  # exponential
                        delta = np.random.exponential(temperature * 0.1, 2)
                        if np.random.random() > 0.5:
                            delta[0] = -delta[0]
                        if np.random.random() > 0.5:
                            delta[1] = -delta[1]
                else:
                    # Low temperature: use smaller, precise perturbations
                    delta = np.random.normal(0, temperature * 0.05, 2)

                new_points[point_idx] = current_points[point_idx] + delta

                # Boundary handling with more careful approach
                for dim in range(2):
                    if new_points[point_idx, dim] < 0.01:
                        # Push away from left boundary
                        new_points[point_idx, dim] = 0.01 + np.random.uniform(0, 0.005)
                    elif new_points[point_idx, dim] > 0.99:
                        # Push away from right boundary
                        new_points[point_idx, dim] = 0.99 - np.random.uniform(0, 0.005)

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
            if stagnation_counter > 100:
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