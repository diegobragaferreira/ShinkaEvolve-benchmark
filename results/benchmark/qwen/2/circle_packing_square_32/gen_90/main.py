# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

def compute_local_density(circles, point, k=5):
    """Compute local density around a point using k-nearest neighbors"""
    if len(circles) <= 1:
        return 0.0
    
    # Calculate distances to all other circles
    distances = np.sqrt(np.sum((circles[:, :2] - point)**2, axis=1))
    
    # Get indices of k nearest neighbors (excluding self)
    sorted_indices = np.argsort(distances)
    nearest_indices = sorted_indices[1:k+1] if len(sorted_indices) > 1 else sorted_indices
    
    # Return average distance to neighbors
    if len(nearest_indices) == 0:
        return 0.0
    return np.mean(distances[nearest_indices])

def initialize_adaptive_hexagonal(n_circles, padding=0.05):
    """Initialize circles using adaptive hexagonal grid with density-aware sizing"""
    # Determine grid dimensions
    rows = int(np.ceil(np.sqrt(n_circles)))
    cols = int(np.ceil(n_circles / rows))
    
    # Create hexagonal pattern
    spacing_x = (1 - 2*padding) / cols
    spacing_y = (1 - 2*padding) / rows
    
    # Adjust spacing for hexagonal arrangement
    hex_spacing_x = spacing_x
    hex_spacing_y = spacing_y * np.sqrt(3)/2
    
    circles = []
    circle_count = 0
    
    # Pre-allocate positions for density analysis
    all_positions = []
    
    for i in range(rows):
        for j in range(cols):
            if circle_count >= n_circles:
                break
                
            # Hexagonal offset
            x_offset = (j if i % 2 == 0 else j + 0.5) * hex_spacing_x + padding
            y_offset = i * hex_spacing_y + padding
            
            # Add some randomness to avoid perfect grid
            x = max(padding, min(1-padding, x_offset + np.random.normal(0, 0.005*hex_spacing_x)))
            y = max(padding, min(1-padding, y_offset + np.random.normal(0, 0.005*hex_spacing_y)))
            
            all_positions.append([x, y])
            circle_count += 1
            
        if circle_count >= n_circles:
            break
    
    # Compute local densities for each position
    all_positions = np.array(all_positions)
    densities = []
    
    # Use cKDTree for efficient neighbor queries
    if len(all_positions) > 1:
        tree = cKDTree(all_positions)
        # Find 5 nearest neighbors for each point to estimate density
        _, indices = tree.query(all_positions, k=min(6, len(all_positions)), 
                               distance_upper_bound=2*max(hex_spacing_x, hex_spacing_y))
        
        for i, idxs in enumerate(indices):
            if len(idxs) > 1:  # Exclude self
                # Remove invalid indices (should not happen with proper query)
                valid_idxs = idxs[idxs != i] 
                if len(valid_idxs) > 0:
                    # Calculate mean distance to neighbors
                    neighbor_positions = all_positions[valid_idxs]
                    distances = np.sqrt(np.sum((all_positions[i] - neighbor_positions)**2, axis=1))
                    avg_distance = np.mean(distances)
                    densities.append(avg_distance)
                else:
                    densities.append(0.0)
            else:
                densities.append(0.0)
    else:
        densities = [0.0] * len(all_positions)
    
    # Set initial radii based on local density (denser areas get smaller circles)
    circles = []
    for i, (x, y) in enumerate(all_positions):
        base_radius = min(hex_spacing_x, hex_spacing_y) * 0.35
        # Adjust radius inversely with local density (higher density = smaller circles)
        density_factor = max(0.2, min(1.0, 1.0 - densities[i] / (2 * max(hex_spacing_x, hex_spacing_y))))
        adjusted_radius = base_radius * density_factor
        
        # Ensure reasonable radius limits
        adjusted_radius = max(0.01, min(0.4, adjusted_radius))
        circles.append([x, y, adjusted_radius])
    
    # Ensure exactly n_circles
    while len(circles) < n_circles:
        # Add random circles in valid positions
        x = np.random.uniform(padding, 1-padding)
        y = np.random.uniform(padding, 1-padding)
        # Radius based on proximity to edges
        r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
        circles.append([x, y, r])
        
    return np.array(circles[:n_circles])

def is_valid_placement(circles, threshold=1e-6):
    """Check if circle configuration is valid with improved tolerance"""
    n = len(circles)
    if n == 0:
        return False
        
    # Check boundary constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    
    # Check overlap constraints using KDTree for efficiency
    if n > 1:
        positions = circles[:, :2]
        tree = cKDTree(positions)
        
        # Query pairs within 2*(max_radius) distance
        max_radius = np.max(circles[:, 2])
        pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')
        
        for i, j in pairs:
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if distance < r1 + r2 - threshold:
                return False
                
    return True

def calculate_penalty(circle_config, weight_boundary=1000, weight_overlap=1000, penalty_scale=1.0):
    """Calculate penalty for constraint violations with adaptive scaling"""
    penalty = 0.0
    n = len(circle_config)

    # Boundary penalties with adaptive scaling based on violation severity
    for i in range(n):
        x, y, r = circle_config[i]
        # Calculate boundary violations
        left_violation = max(0, r - x)
        right_violation = max(0, x + r - 1)
        bottom_violation = max(0, r - y)
        top_violation = max(0, y + r - 1)

        # Apply quadratic penalty scaling (stronger penalties for larger violations)
        penalty += weight_boundary * penalty_scale * (left_violation**2 + right_violation**2 +
                                                     bottom_violation**2 + top_violation**2)

    # Overlap penalties with adaptive scaling based on violation severity
    if n > 1:
        positions = circle_config[:, :2]
        radii = circle_config[:, 2]
        tree = cKDTree(positions)

        # Query pairs within 2*(max_radius) distance
        max_radius = np.max(radii)
        pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')

        for i, j in pairs:
            x1, y1, r1 = circle_config[i]
            x2, y2, r2 = circle_config[j]
            distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if distance < r1 + r2:
                # Adaptive penalty based on violation magnitude with exponential scaling
                violation = r1 + r2 - distance
                # Use quadratic scaling for overlap penalties to emphasize severe overlaps
                penalty += weight_overlap * penalty_scale * (violation**2) * np.exp(10 * violation)

    return penalty

def evaluate_fitness(circle_config):
    """Evaluate fitness with dual considerations: sum of radii and constraint satisfaction"""
    # Convert to array if needed
    if not isinstance(circle_config, np.ndarray):
        circle_config = np.array(circle_config)
    
    # Sum of radii (negative because we minimize in optimization)
    sum_radii = -np.sum(circle_config[:, 2])
    
    # Constraint penalty
    penalty = calculate_penalty(circle_config, penalty_scale=1.0)
    
    return sum_radii + penalty

def optimize_with_constraints(initial_circles, max_iter=1000, tolerance=1e-8):
    """Optimize a configuration with enhanced constraint handling"""
    n = len(initial_circles)
    
    # Prepare flattened parameter vector (x, y, r for each circle)
    initial_params = initial_circles.flatten()
    
    # Define bounds for optimization (x,y in [0.05, 0.95], r in [0.01, 0.4])
    bounds = []
    for _ in range(n):
        bounds.extend([(0.05, 0.95), (0.05, 0.95), (0.01, 0.4)])
    
    try:
        # Use L-BFGS-B for constrained optimization
        result = minimize(
            evaluate_fitness,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': tolerance, 'gtol': tolerance}
        )
        
        # Extract optimized parameters
        optimized_params = result.x
        optimized_circles = optimized_params.reshape(-1, 3)
        
        return optimized_circles
        
    except Exception as e:
        # Return original if optimization fails
        return initial_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    max_iter = 1500
    tolerance = 1e-8
    
    best_sum_radii = 0.0
    best_circles = None
    
    # Stage 1: Multi-start optimization with adaptive hexagonal initialization
    num_starts = 20
    stage1_results = []
    
    for start_idx in range(num_starts):
        # Initialize with adaptive hexagonal grid
        np.random.seed(start_idx * 12345)  # Fixed seed for reproducibility
        circles = initialize_adaptive_hexagonal(n)
        
        # First optimization pass with reduced tolerance for coarse exploration
        coarse_circles = optimize_with_constraints(circles, max_iter=500, tolerance=1e-4)
        
        # Validate this coarse solution
        if is_valid_placement(coarse_circles):
            current_sum = np.sum(coarse_circles[:, 2])
            stage1_results.append((current_sum, coarse_circles.copy()))
    
    # Select best from stage 1 results
    if stage1_results:
        stage1_results.sort(key=lambda x: x[0], reverse=True)
        best_from_stage1 = stage1_results[0][1]
        
        # Stage 2: Fine-grained optimization starting from the best found so far
        fine_optimized = optimize_with_constraints(best_from_stage1, max_iter=max_iter, tolerance=tolerance)
        
        if is_valid_placement(fine_optimized):
            final_sum = np.sum(fine_optimized[:, 2])
            if final_sum > best_sum_radii:
                best_sum_radii = final_sum
                best_circles = fine_optimized.copy()
    
    # If no valid solution found from stage 1, fallback to a robust initialization
    if best_circles is None:
        # Try a different initialization strategy
        np.random.seed(42)
        rows = 6
        cols = 6
        padding = 0.05
        spacing_x = (1 - 2*padding) / cols
        spacing_y = (1 - 2*padding) / rows
        hex_spacing_x = spacing_x
        hex_spacing_y = spacing_y * np.sqrt(3)/2
        
        circles = []
        circle_count = 0
        
        for i in range(rows):
            for j in range(cols):
                if circle_count >= n:
                    break
                    
                x_offset = (j if i % 2 == 0 else j + 0.5) * hex_spacing_x + padding
                y_offset = i * hex_spacing_y + padding
                
                x = max(padding, min(1-padding, x_offset))
                y = max(padding, min(1-padding, y_offset))
                
                base_radius = min(hex_spacing_x, hex_spacing_y) * 0.35
                circles.append([x, y, base_radius])
                circle_count += 1
                
            if circle_count >= n:
                break
        
        # Fill remaining circles randomly but strategically
        while len(circles) < n:
            x = np.random.uniform(padding, 1-padding)
            y = np.random.uniform(padding, 1-padding)
            # Radius based on proximity to edges
            r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
            circles.append([x, y, r])
            
        best_circles = np.array(circles[:n])
        
        # Optimize the fallback solution
        best_circles = optimize_with_constraints(best_circles, max_iter=max_iter, tolerance=tolerance)
    
    # Final validation and correction
    if not is_valid_placement(best_circles):
        # Reset to a robust hexagonal grid if final configuration is invalid
        np.random.seed(42)
        rows = 6
        cols = 6
        padding = 0.05
        spacing_x = (1 - 2*padding) / cols
        spacing_y = (1 - 2*padding) / rows
        hex_spacing_x = spacing_x
        hex_spacing_y = spacing_y * np.sqrt(3)/2
        
        circles = []
        circle_count = 0
        
        for i in range(rows):
            for j in range(cols):
                if circle_count >= n:
                    break
                    
                x_offset = (j if i % 2 == 0 else j + 0.5) * hex_spacing_x + padding
                y_offset = i * hex_spacing_y + padding
                
                x = max(padding, min(1-padding, x_offset))
                y = max(padding, min(1-padding, y_offset))
                
                base_radius = min(hex_spacing_x, hex_spacing_y) * 0.35
                circles.append([x, y, base_radius])
                circle_count += 1
                
            if circle_count >= n:
                break
        
        # Fill remaining circles randomly but strategically
        while len(circles) < n:
            x = np.random.uniform(padding, 1-padding)
            y = np.random.uniform(padding, 1-padding)
            # Radius based on proximity to edges
            r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
            circles.append([x, y, r])
            
        best_circles = np.array(circles[:n])
        
        # Final optimization
        best_circles = optimize_with_constraints(best_circles, max_iter=max_iter, tolerance=tolerance)
    
    # Ensure all circles are within bounds and valid
    for i in range(len(best_circles)):
        x, y, r = best_circles[i]
        # Constrain positions to valid range
        best_circles[i][0] = max(r, min(1-r, x))
        best_circles[i][1] = max(r, min(1-r, y))
        # Constrain radii to valid range
        best_circles[i][2] = max(0.01, min(0.4, r))
        
    return best_circles

# EVOLVE-BLOCK-END