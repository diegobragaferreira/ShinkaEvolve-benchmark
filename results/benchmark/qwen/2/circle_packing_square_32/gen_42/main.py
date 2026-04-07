# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import math
from typing import Tuple, List

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def compute_local_density(positions: np.ndarray, k: int = 5) -> np.ndarray:
    """Compute local density around each point using k-nearest neighbors."""
    if len(positions) <= 1:
        return np.ones(len(positions))
    
    tree = cKDTree(positions)
    distances, _ = tree.query(positions, k=min(k+1, len(positions)), p=2)
    # Average distance to k nearest neighbors (excluding self)
    avg_distances = np.mean(distances[:, 1:], axis=1)
    # Density is inverse of average distance (higher density = lower average distance)
    densities = 1.0 / (avg_distances + 1e-10)
    return densities

def initialize_hexagonal_grid(n_circles: int) -> np.ndarray:
    """Initialize circle positions using a hexagonal grid pattern with adaptive sizing."""
    # Find dimensions of hexagonal grid
    cols = int(math.ceil(math.sqrt(n_circles)))
    rows = int(math.ceil(n_circles / cols))
    
    # Create a hexagonal grid
    positions = []
    for i in range(rows):
        for j in range(cols):
            # Offset every other row
            x_offset = j + 0.5 * (i % 2)
            y_offset = i * math.sqrt(3) / 2
            positions.append([x_offset, y_offset])

    if len(positions) > n_circles:
        positions = positions[:n_circles]

    positions = np.array(positions)
    
    if len(positions) == 0:
        return np.zeros((n_circles, 2))
        
    # Normalize to fit in [0.05, 0.95] x [0.05, 0.95] to leave margin
    min_x, min_y = positions.min(axis=0)
    max_x, max_y = positions.max(axis=0)
    
    if max_x - min_x > 0 and max_y - min_y > 0:
        scale_x = 0.9 / (max_x - min_x)
        scale_y = 0.9 / (max_y - min_y)
        positions[:, 0] = (positions[:, 0] - min_x) * scale_x + 0.05
        positions[:, 1] = (positions[:, 1] - min_y) * scale_y + 0.05
    
    return positions

def compute_radius_at_position(pos: Tuple[float, float], circles: np.ndarray, 
                             min_radius: float = 0.001, max_radius: float = 0.5) -> float:
    """Compute maximum possible radius at given position without overlapping existing circles."""
    x, y = pos
    
    # Initial estimate of radius based on distance to nearest boundary
    r_boundary = min(x, 1-x, y, 1-y)
    
    if r_boundary <= min_radius:
        return min_radius

    # Check overlap with existing circles
    min_dist = float('inf')
    for circ in circles:
        if len(circ) >= 3:  # Ensure it has the expected format
            circ_x, circ_y, circ_r = circ[0], circ[1], circ[2]
            dist = np.sqrt((x - circ_x)**2 + (y - circ_y)**2)
            min_dist = min(min_dist, dist)
    
    # Maximum radius is limited by both boundary and existing circles
    if min_dist != float('inf'):
        r = min(r_boundary, min_dist/2 - 0.001)
    else:
        r = r_boundary
        
    return max(min_radius, min(max_radius, r))

def smooth_penalty_constraint_violation(distance: float, min_distance: float, 
                                      penalty_weight: float = 1000.0) -> float:
    """Apply smooth penalty for constraint violations."""
    if distance >= min_distance:
        return 0.0
    else:
        # Exponential penalty that smoothly increases as violation grows
        violation = min_distance - distance
        return penalty_weight * (1.0 - np.exp(-violation * 10.0))

def evaluate_fitness_smooth(circles_flat: np.ndarray, n_circles: int, 
                          penalty_weight: float = 1000.0) -> float:
    """Evaluate the fitness of a circle configuration with smooth penalties."""
    # Reshape flat array back into circles
    circles = circles_flat.reshape((n_circles, 3))
    
    # Calculate sum of radii (this is what we want to maximize)
    total_radius = np.sum(circles[:, 2])
    
    penalty = 0.0
    
    # Boundary penalties using smooth exponential function
    for i in range(n_circles):
        x, y, r = circles[i]
        # Penalties for boundary violations using smooth exponential
        if x - r < 0:
            penalty += smooth_penalty_constraint_violation(x - r, 0.0, penalty_weight)
        if x + r > 1:
            penalty += smooth_penalty_constraint_violation(-(x + r - 1), 0.0, penalty_weight)
        if y - r < 0:
            penalty += smooth_penalty_constraint_violation(y - r, 0.0, penalty_weight)
        if y + r > 1:
            penalty += smooth_penalty_constraint_violation(-(y + r - 1), 0.0, penalty_weight)
    
    # Overlap penalties using smooth exponential function with spatial indexing for efficiency
    if len(circles) > 1:
        # Build k-d tree for fast neighbor search
        positions = circles[:, :2]
        tree = cKDTree(positions)
        
        # Search for nearby points to reduce pairwise comparisons
        for i in range(n_circles):
            x1, y1, r1 = circles[i]
            
            # Query nearby circles (within 4*(r1 + r_max) distance)
            r_max = np.max(circles[:, 2]) if len(circles) > 0 else 0.5
            query_radius = 4 * (r1 + r_max)
            
            neighbors = tree.query_ball_point([x1, y1], query_radius)
            
            for j in neighbors:
                if i != j:
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    # Smooth penalty for overlap
                    penalty += smooth_penalty_constraint_violation(
                        distance, r1 + r2, penalty_weight)
    
    # Return negative because we minimize in scipy but want to maximize radius sum
    return -(total_radius - penalty)

def generate_multiple_initializations(n_circles: int, n_init: int = 5) -> List[np.ndarray]:
    """Generate multiple initial configurations for robust optimization."""
    initial_configs = []
    
    # Base hexagonal grid with random perturbations
    base_positions = initialize_hexagonal_grid(n_circles)
    
    for i in range(n_init):
        # Create variation by adding random noise
        positions = base_positions.copy()
        np.random.seed(i)  # For deterministic variations
        noise_scale = 0.05
        positions += np.random.uniform(-noise_scale, noise_scale, positions.shape)
        
        # Ensure positions stay within bounds
        positions = np.clip(positions, 0.05, 0.95)
        
        # Generate initial radii using density-aware method
        radii = np.zeros(n_circles)
        for j in range(n_circles):
            radii[j] = compute_radius_at_position(positions[j], 
                                                np.column_stack([positions[:j], radii[:j]]))
        
        # Ensure minimum radius
        radii = np.maximum(radii, 0.005)
        
        config = np.column_stack([positions, radii]).flatten()
        initial_configs.append(config)
    
    return initial_configs

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    circles = np.zeros((n, 3))
    
    # Generate multiple initial configurations
    initial_configs = generate_multiple_initializations(n, n_init=3)
    
    best_result = None
    best_sum = -float('inf')
    
    # Try different initializations
    for i, initial_solution in enumerate(initial_configs):
        try:
            # Define bounds for optimization (x, y, r for each circle)
            bounds = []
            for j in range(n):
                bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.5)])  # x, y, r bounds
            
            # Optimize using scipy minimize with smooth penalties
            result = minimize(
                evaluate_fitness_smooth, 
                initial_solution, 
                args=(n,), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6},
                callback=lambda x: None  # No callback needed
            )
            
            if result.success:
                # Evaluate final fitness
                final_fitness = evaluate_fitness_smooth(result.x, n)
                # Convert from negative fitness back to positive sum of radii
                sum_radii = -final_fitness
                
                if sum_radii > best_sum:
                    best_sum = sum_radii
                    best_result = result.x.copy()
                    
        except Exception as e:
            print(f"Optimization attempt {i} failed: {e}")
            continue
    
    # If we have a valid result, return it
    if best_result is not None:
        final_circles = best_result.reshape((n, 3))
        return final_circles
    else:
        # Fallback: return initial configuration from first attempt
        initial_positions = initialize_hexagonal_grid(n)
        initial_radii = np.full(n, 0.02)
        return np.column_stack([initial_positions, initial_radii])

# EVOLVE-BLOCK-END