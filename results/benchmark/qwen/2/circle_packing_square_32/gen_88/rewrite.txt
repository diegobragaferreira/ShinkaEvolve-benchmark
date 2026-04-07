# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import random

def initialize_circles_hexgrid_adaptive(n=32):
    """Initialize circle positions with adaptive hexagonal grid and k-NN density-aware radius assignment."""
    
    # Adaptive hexagonal grid generation
    # Calculate optimal grid dimensions based on target density
    packing_density = 0.866  # Hexagonal packing density
    ideal_rows_cols = np.sqrt(n / packing_density)
    
    # Round to nearest integers that maintain sufficient cells
    rows = int(np.ceil(ideal_rows_cols))
    cols = int(np.ceil(n / rows))
    
    # Ensure we have enough positions
    while rows * cols < n:
        rows += 1
        
    # Create hexagonal grid with proper spacing
    positions = []
    
    # Calculate spacing to fit within unit square
    hex_spacing_x = 1.0 / cols
    hex_spacing_y = hex_spacing_x * 0.866  # sqrt(3)/2
    
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            # Offset every other row for hexagonal packing
            x_offset = (i % 2) * (hex_spacing_x / 2)
            x = (j + 0.5 + x_offset) * hex_spacing_x
            y = (i + 0.5) * hex_spacing_y
            
            # Clamp to valid range with small margin
            x = np.clip(x, 0.01, 0.99)
            y = np.clip(y, 0.01, 0.99)
            positions.append([x, y])
    
    # Trim to exact count
    positions = positions[:n]
    
    # Build KDTree for efficient neighbor queries
    positions_array = np.array(positions)
    tree = cKDTree(positions_array)
    
    # Assign initial radii with k-NN density estimation (k=5 as per recommendations)
    initial_radii = []
    k = 5
    
    for i in range(n):
        x, y = positions[i]
        
        # Calculate distance to boundaries for containment constraint
        dist_to_boundaries = min(x, 1-x, y, 1-y)
        
        # Find k nearest neighbors (excluding self)
        distances, indices = tree.query(positions_array[i], k=k+1)
        # Remove self-distance (distance 0)
        distances = distances[1:]
        
        # Compute average distance to k nearest neighbors for density estimation
        avg_neighbor_distance = np.mean(distances) if len(distances) > 0 else 0.1
        
        # Calculate appropriate radius based on available space
        # Larger radius in sparse regions, smaller in dense regions
        # Scale inverse with density (average neighbor distance)
        radius_based_on_boundaries = dist_to_boundaries * 0.4
        radius_based_on_density = avg_neighbor_distance * 0.15  # Reduced weight compared to boundary
        
        # Take the minimum of both constraints to ensure feasibility
        base_radius = min(radius_based_on_boundaries, radius_based_on_density)
        
        # Apply additional constraints to ensure reasonable initial values
        final_radius = max(0.005, min(0.15, base_radius))
        
        # Add slight randomness to avoid symmetric solutions
        final_radius *= np.random.uniform(0.95, 1.05)
        
        initial_radii.append(final_radius)
    
    # Combine into circles array
    circles = np.column_stack([positions, initial_radii])
    return circles

def evaluate_fitness_smooth(circles):
    """Evaluate fitness with smooth penalty functions for better optimization."""
    n = len(circles)
    total_radius = np.sum(circles[:, 2])

    # Smooth penalty for containment violations using exponential functions
    penalty = 0
    
    # Containment constraints with smooth penalties
    for i in range(n):
        x, y, r = circles[i]
        # Exponential penalty for boundary violations
        if x - r < 0:
            penalty += 1000 * np.exp(10 * (x - r))
        if x + r > 1:
            penalty += 1000 * np.exp(10 * (x + r - 1))
        if y - r < 0:
            penalty += 1000 * np.exp(10 * (y - r))
        if y + r > 1:
            penalty += 1000 * np.exp(10 * (y + r - 1))

    # Smooth penalty for overlap constraints using exponential functions
    positions = circles[:, :2]
    radii = circles[:, 2]
    
    # Compute pairwise distances
    distances = cdist(positions, positions)
    
    # Create mask for pairs that would overlap (excluding diagonal)
    overlap_mask = distances < (radii[:, None] + radii[None, :])
    np.fill_diagonal(overlap_mask, False)
    
    # Apply smooth penalty for overlaps
    overlap_distances = (radii[:, None] + radii[None, :] - distances)
    overlap_penalty = np.sum(1000 * np.exp(10 * overlap_distances[overlap_mask]))
    penalty += overlap_penalty

    return total_radius - penalty

def optimize_circles_multistart(circles, num_starts=5):
    """Optimize circle positions using multi-start approach with perturbed initial configurations."""
    best_circles = circles.copy()
    best_fitness = evaluate_fitness_smooth(circles)
    
    # Run multiple optimization starts with different perturbations
    for start_idx in range(num_starts):
        # Create perturbed version of the initial configuration
        perturbed_circles = circles.copy()
        
        # Apply small random perturbations to positions
        for i in range(len(perturbed_circles)):
            # Slightly perturb positions (±0.05 range)
            perturbed_circles[i, 0] += np.random.uniform(-0.05, 0.05)
            perturbed_circles[i, 1] += np.random.uniform(-0.05, 0.05)
            
            # Clamp to valid range
            perturbed_circles[i, 0] = np.clip(perturbed_circles[i, 0], 0.01, 0.99)
            perturbed_circles[i, 1] = np.clip(perturbed_circles[i, 1], 0.01, 0.99)
        
        # Optimize this perturbed version
        def objective(params):
            circles_flat = params.reshape(-1, 3)
            return -evaluate_fitness_smooth(circles_flat)

        # Flatten the circles array for optimization
        initial_params = perturbed_circles.flatten()

        # Define bounds for optimization
        bounds = []
        for i in range(len(initial_params)):
            if i % 3 == 2:  # radius parameter
                bounds.append((0.001, 0.5))  # Radius between 0.001 and 0.5
            else:  # x and y parameters
                bounds.append((0.001, 0.999))  # Position within the unit square

        try:
            # Use L-BFGS-B which handles bounds well
            result = minimize(objective, initial_params, method='L-BFGS-B',
                             bounds=bounds, options={'maxiter': 2000})

            if result.success:
                optimized_circles = result.x.reshape(-1, 3)
                fitness = evaluate_fitness_smooth(optimized_circles)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_circles = optimized_circles.copy()
        except Exception as e:
            continue  # Skip this start if optimization fails

    return best_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    circles = np.zeros((n, 3))

    # Initialize with enhanced heuristic approach
    circles = initialize_circles_hexgrid_adaptive(n)

    # Optimize the configuration with multi-start approach
    circles = optimize_circles_multistart(circles, num_starts=5)

    return circles

# EVOLVE-BLOCK-END