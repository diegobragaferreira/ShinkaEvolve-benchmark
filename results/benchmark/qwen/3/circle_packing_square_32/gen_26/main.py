# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import differential_evolution
import warnings

def generate_hexagonal_grid(n_circles, padding=0.05):
    """Generate initial circle positions on a hexagonal grid."""
    # Calculate grid parameters
    rows = int(np.ceil(np.sqrt(n_circles)))
    cols = int(np.ceil(n_circles / rows))
    
    # Create hexagonal grid
    y_positions = np.linspace(padding, 1-padding, rows)
    x_positions = np.linspace(padding, 1-padding, cols)
    
    # Offset every other row
    x_offsets = np.arange(cols) * (1-padding*2) / cols
    y_offsets = np.arange(rows) * (1-padding*2) / rows
    
    # Generate positions
    positions = []
    for i, y in enumerate(y_positions):
        for j, x in enumerate(x_positions):
            if len(positions) >= n_circles:
                break
            offset = 0.5 * (i % 2)  # Offset every other row
            actual_x = x + offset * (1-padding*2) / cols
            positions.append([actual_x, y])
        if len(positions) >= n_circles:
            break
            
    return np.array(positions[:n_circles])

def calculate_radius_at_position(positions, radii, idx, min_radius=0.001):
    """Calculate maximum possible radius at given position considering overlaps."""
    min_dist = float('inf')
    
    # Check distance to all other circles
    for i, pos in enumerate(positions):
        if i != idx:
            dist = np.sqrt(np.sum((positions[idx] - pos)**2))
            min_dist = min(min_dist, dist)
            
    # Return minimum of available space and max possible radius
    max_radius = min(min_dist/2.0, 0.5)
    return max(max_radius, min_radius)

def check_constraints(circles, min_radius=0.001):
    """Check if all circles satisfy containment and non-overlap constraints."""
    n = len(circles)
    
    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if r <= min_radius or x <= r or x >= 1-r or y <= r or y >= 1-r:
            return False
    
    # Check overlap constraints using KDTree for efficiency
    positions = circles[:, :2]
    tree = cKDTree(positions)
    
    for i in range(n):
        x, y, r = circles[i]
        # Find nearby points (within 2*r distance)
        neighbors = tree.query_ball_point([x, y], 2*r + 1e-8)
        
        # Check overlap with each neighbor
        for j in neighbors:
            if i != j:
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x-x2)**2 + (y-y2)**2)
                if dist < r + r2 + 1e-8:
                    return False
                    
    return True

def evaluate_fitness(circles):
    """Evaluate fitness as the sum of all radii."""
    return np.sum(circles[:, 2])

def optimize_circles(initial_circles, max_iter=1000):
    """Optimize circle positions and radii using constrained optimization."""
    
    def objective(params):
        # Reshape params back to circles array
        circles = params.reshape(-1, 3)
        return -evaluate_fitness(circles)  # Negative because we want to maximize
    
    def constraint_func(params):
        circles = params.reshape(-1, 3)
        if not check_constraints(circles):
            return -1  # Violated constraint
        return 1   # Valid constraint
    
    # Flatten initial circles for optimization
    flat_initial = initial_circles.flatten()
    
    # Set bounds for optimization (positions and radii)
    bounds = []
    for i in range(len(flat_initial)):
        if i % 3 == 2:  # Radius parameter
            bounds.append((0.001, 0.5))  # Radius bounds
        else:  # Position parameters
            bounds.append((0.001, 0.999))  # Position bounds
    
    # Initial constraint test
    if not check_constraints(initial_circles):
        raise ValueError("Initial configuration violates constraints")
    
    # Use differential evolution for robust optimization
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=max_iter,
            popsize=15,
            tol=1e-6,
            recombination=0.7,
            seed=42,
            callback=lambda x, convergence=None: print(f"Fitness: {-objective(x)}")
        )
        
        optimized_circles = result.x.reshape(-1, 3)
        
        # Final validation
        if check_constraints(optimized_circles):
            return optimized_circles
        else:
            # If optimization failed, return original
            warnings.warn("Optimization did not produce valid result. Returning initial.")
            return initial_circles
            
    except Exception as e:
        warnings.warn(f"Optimization failed with error: {str(e)}. Returning initial configuration.")
        return initial_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For deterministic results
    
    # Phase 1: Hexagonal initialization
    initial_positions = generate_hexagonal_grid(32, padding=0.05)
    
    # Phase 2: Initialize with maximum possible radii
    circles = np.zeros((32, 3))
    for i in range(32):
        circles[i] = [
            initial_positions[i][0],
            initial_positions[i][1],
            calculate_radius_at_position(initial_positions, None, i)
        ]
    
    # Phase 3: Local optimization
    optimized_circles = optimize_circles(circles, max_iter=500)
    
    # Phase 4: Final refinement
    if check_constraints(optimized_circles):
        return optimized_circles
    else:
        # Fallback to initial configuration
        return circles

# EVOLVE-BLOCK-END
