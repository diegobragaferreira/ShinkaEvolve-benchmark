# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math
import warnings

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

    def compute_boundary_penalty(points, penalty_weight=20.0):
        """Computes penalty for points near boundaries using quadratic scaling."""
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
                # Quadratic penalty for smooth but strong boundary avoidance
                penalty += penalty_weight * (0.01 - min_dist)**2
        return penalty

    def evaluate_with_penalty(points, penalty_weight=20.0):
        """Evaluate ratio with boundary penalty applied."""
        ratio = compute_min_max_ratio(points)
        penalty = compute_boundary_penalty(points, penalty_weight)
        return ratio - penalty

    def generate_hexagonal_grid_with_variations():
        """Generate multiple hexagonal grid variations with different symmetries."""
        # Base hexagonal grid with different offsets and rotations
        grids = []
        
        # Regular hexagonal grid
        points = []
        sqrt3 = np.sqrt(3)
        for i in range(4):
            for j in range(4):
                x = j + 0.5 * (i % 2)
                y = i * sqrt3 / 2
                points.append([x, y])
        
        points = np.array(points[:16])
        
        # Normalize
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        if x_range > 0:
            points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
        if y_range > 0:
            points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
            
        # Scale to [0.05, 0.95]
        points[:, 0] = points[:, 0] * 0.9 + 0.05
        points[:, 1] = points[:, 1] * 0.9 + 0.05
        
        grids.append(("hex_base", points))
        
        # Rotated hexagonal grid
        rotation_angle = np.pi / 6  # 30 degrees
        cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_points = np.dot(points, rotation_matrix.T)
        
        # Normalize after rotation
        x_range_rot = np.max(rotated_points[:, 0]) - np.min(rotated_points[:, 0])
        y_range_rot = np.max(rotated_points[:, 1]) - np.min(rotated_points[:, 1])
        if x_range_rot > 0:
            rotated_points[:, 0] = (rotated_points[:, 0] - np.min(rotated_points[:, 0])) / x_range_rot
        if y_range_rot > 0:
            rotated_points[:, 1] = (rotated_points[:, 1] - np.min(rotated_points[:, 1])) / y_range_rot
            
        # Scale to [0.05, 0.95]
        rotated_points[:, 0] = rotated_points[:, 0] * 0.9 + 0.05
        rotated_points[:, 1] = rotated_points[:, 1] * 0.9 + 0.05
        grids.append(("hex_rotated", rotated_points))
        
        # Offset hexagonal grid
        offset_points = points.copy()
        offset_points += np.random.normal(0, 0.005, offset_points.shape)
        offset_points = np.clip(offset_points, 0, 1)
        grids.append(("hex_offset", offset_points))
        
        # Variationally perturbed grid
        var_points = points.copy()
        np.random.seed(42)
        for i in range(len(var_points)):
            row = i // 4
            col = i % 4
            noise = np.random.normal(0, 0.01 * (1 + 0.1 * (row + col)), 2)
            var_points[i] += noise
        var_points = np.clip(var_points, 0, 1)
        grids.append(("hex_var", var_points))
        
        return grids

    def generate_triangular_lattice():
        """Generate triangular lattice pattern."""
        points = []
        sqrt3 = np.sqrt(3)
        
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
            
        # Scale to [0.05, 0.95] range
        points[:, 0] = points[:, 0] * 0.9 + 0.05
        points[:, 1] = points[:, 1] * 0.9 + 0.05
        
        return [("triangular", points)]

    def generate_polar_arrangement():
        """Generate points in polar arrangement."""
        points = []
        n = 16
        
        # Arrange in rings with different radii and angles
        radii = [0.15, 0.35, 0.55, 0.75]
        angles_per_ring = [4, 4, 4, 4]  # Number of points per ring
        
        for i, (radius, angle_count) in enumerate(zip(radii, angles_per_ring)):
            if len(points) >= n:
                break
            for j in range(angle_count):
                if len(points) >= n:
                    break
                angle = 2 * np.pi * j / angle_count
                x = 0.5 + radius * np.cos(angle) * 0.4
                y = 0.5 + radius * np.sin(angle) * 0.4
                points.append([x, y])
        
        # Fill remaining points with random variation
        while len(points) < n:
            x = 0.5 + (np.random.rand() - 0.5) * 0.8
            y = 0.5 + (np.random.rand() - 0.5) * 0.8
            points.append([x, y])
        
        points = np.array(points[:n])
        
        # Apply small perturbations to break symmetries
        np.random.seed(42)
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        return [("polar", points)]

    def generate_grid_with_randomization():
        """Generate grid with systematic randomization."""
        points = []
        n_per_side = 4
        
        # Create regular grid
        x = np.linspace(0.1, 0.9, n_per_side)
        y = np.linspace(0.1, 0.9, n_per_side)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])[:16]
        
        # Add systematic perturbations
        np.random.seed(42)
        for i in range(len(points)):
            # Vary perturbation strength based on position
            row = i // n_per_side
            col = i % n_per_side
            strength = 0.004 + 0.003 * (row + col) / (n_per_side * 2)
            points[i] += np.random.normal(0, strength, 2)
        
        points = np.clip(points, 0, 1)
        return [("grid", points)]

    def generate_random_arrangement():
        """Generate random points."""
        np.random.seed(42)
        points = np.random.rand(16, 2)
        return [("random", points)]

    def initialize_multiple_strategies():
        """Initialize multiple starting configurations with enhanced variety."""
        strategies = []

        # Add hexagonal grids with variations
        try:
            hex_grids = generate_hexagonal_grid_with_variations()
            strategies.extend(hex_grids)
        except Exception as e:
            warnings.warn(f"Hexagonal grid generation failed: {str(e)}")
            strategies.append(("fallback_hex", generate_hexagonal_grid_with_variations()[0][1]))

        # Add triangular lattice
        try:
            tri_grids = generate_triangular_lattice()
            strategies.extend(tri_grids)
        except Exception as e:
            warnings.warn(f"Triangular lattice generation failed: {str(e)}")
            strategies.append(("fallback_tri", generate_triangular_lattice()[0][1]))

        # Add polar arrangement
        try:
            polar_grids = generate_polar_arrangement()
            strategies.extend(polar_grids)
        except Exception as e:
            warnings.warn(f"Polar arrangement generation failed: {str(e)}")
            strategies.append(("fallback_polar", generate_polar_arrangement()[0][1]))

        # Add grid with randomization
        try:
            grid_grids = generate_grid_with_randomization()
            strategies.extend(grid_grids)
        except Exception as e:
            warnings.warn(f"Grid with randomization generation failed: {str(e)}")
            strategies.append(("fallback_grid", generate_grid_with_randomization()[0][1]))

        # Add random
        try:
            rand_grids = generate_random_arrangement()
            strategies.extend(rand_grids)
        except Exception as e:
            warnings.warn(f"Random arrangement generation failed: {str(e)}")
            strategies.append(("fallback_rand", generate_random_arrangement()[0][1]))

        return strategies

    def neighborhood_move(current_points, point_indices, step_size=0.01):
        """Performs a coordinated move on a cluster of points."""
        new_points = current_points.copy()
        
        # Calculate centroid of selected points
        centroid = np.mean(current_points[point_indices], axis=0)
        
        # Generate movement vector
        move_vector = np.random.normal(0, step_size, 2)
        
        # Apply movement to selected points with boundary reflection
        for idx in point_indices:
            new_points[idx] = current_points[idx] + move_vector
            
            # Boundary handling with reflection
            for dim in range(2):
                if new_points[idx, dim] < 0.01:
                    new_points[idx, dim] = 0.01 + np.random.uniform(0, 0.005)
                elif new_points[idx, dim] > 0.99:
                    new_points[idx, dim] = 0.99 - np.random.uniform(0, 0.005)
                    
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
        
        # Progress tracking
        last_best_ratio = current_ratio
        progress_stagnation = 0
        max_progress_stagnation = 100
        
        # Phase-based cooling adjustments
        phase = 0
        phase_thresholds = [max_iterations * 0.2, max_iterations * 0.5, max_iterations * 0.8]
        
        for iteration in range(max_iterations):
            # Adaptive temperature adjustment based on progress and phase
            if iteration % 100 == 0 and iteration > 0:
                # Adjust phase based on iteration count
                if iteration < phase_thresholds[0]:
                    phase = 0  # Aggressive exploration
                    phase_cooling = 0.9995
                elif iteration < phase_thresholds[1]:
                    phase = 1  # Balanced exploration/exploitation
                    phase_cooling = 0.9992
                else:
                    phase = 2  # Fine-tuning
                    phase_cooling = 0.9990
                
                # Check for improvement
                ratio_diff = current_ratio - last_best_ratio
                if ratio_diff > 1e-8:
                    # There was an improvement
                    cooling_rate = min(phase_cooling * 1.005, 0.9998)
                    last_best_ratio = current_ratio
                    progress_stagnation = 0
                else:
                    # No improvement
                    cooling_rate = max(phase_cooling * 0.99, 0.999)
                    progress_stagnation += 1
                    
                    # Restart with higher temp if stagnating
                    if progress_stagnation > max_progress_stagnation:
                        temperature = min(temperature * 2.0, 0.5)
                        progress_stagnation = 0
            
            # Determine step size based on phase and temperature
            step_size = temperature * 0.1
            
            # Decide between single point or neighborhood move
            if np.random.random() < 0.7:  # 70% chance of neighborhood move
                # Select random subset of points for neighborhood move
                num_selected = np.random.randint(2, 6)  # 2 to 5 points
                point_indices = np.random.choice(len(current_points), size=num_selected, replace=False)
                
                # Perform neighborhood move
                new_points = neighborhood_move(current_points, point_indices, step_size=step_size)
            else:
                # Single point move
                new_points = current_points.copy()
                point_idx = np.random.randint(len(current_points))
                delta = np.random.normal(0, step_size, 2)
                new_points[point_idx] = current_points[point_idx] + delta
                
                # Boundary handling with reflection
                for dim in range(2):
                    if new_points[point_idx, dim] < 0.01:
                        new_points[point_idx, dim] = 0.01 + np.random.uniform(0, 0.005)
                    elif new_points[point_idx, dim] > 0.99:
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
            if stagnation_counter > 50:
                cooling_rate = min(cooling_rate * 0.99, 0.9999)
                stagnation_counter = 0
            
            # Cool down temperature
            temperature *= cooling_rate
            if temperature < min_temp:
                temperature = min_temp
            
            # Early stopping if improvement is minimal
            recent_improvements.append(1 if new_ratio > current_ratio else 0)
            if len(recent_improvements) > improvement_threshold:
                recent_improvements.pop(0)
                if sum(recent_improvements) < 2 and iteration > 1000:
                    break
        
        return best_points, best_ratio

    # Generate multiple initializations using our enhanced strategies
    strategies = initialize_multiple_strategies()

    # Run optimization from each starting point
    best_result = None
    best_score = -np.inf
    optimization_runs = 0

    # Keep track of which strategies have been tried
    tried_strategies = set()
    
    for strategy_name, initial_points in strategies:
        # Skip strategies already tried
        if strategy_name in tried_strategies:
            continue
        tried_strategies.add(strategy_name)
        
        try:
            # Use different optimization parameters based on strategy type
            if "random" in strategy_name or "grid" in strategy_name:
                # Use more iterations for these potentially harder-to-optimize cases
                optimized_points, score = adaptive_simulated_annealing(
                    initial_points, max_iterations=5000, initial_temp=0.12
                )
            else:
                # Standard parameters
                optimized_points, score = adaptive_simulated_annealing(
                    initial_points, max_iterations=5000, initial_temp=0.15
                )
            
            if score > best_score:
                best_score = score
                best_result = optimized_points.copy()
                optimization_runs += 1
                
        except Exception as e:
            continue  # Skip failed runs

    # Final refinement with the best result if we have one
    if best_result is not None:
        try:
            # Use more intensive refinement
            final_points, _ = adaptive_simulated_annealing(
                best_result, max_iterations=2500, initial_temp=0.03
            )
            return final_points
        except Exception as e:
            pass

    # Fallback to best found if optimization fails
    if best_result is not None:
        return best_result

    # Last resort: return a basic hexagonal grid with small perturbation
    base_grid = generate_hexagonal_grid_with_variations()[0][1]
    np.random.seed(42)
    fallback = base_grid + np.random.normal(0, 0.005, base_grid.shape)
    return np.clip(fallback, 0, 1)

# EVOLVE-BLOCK-END