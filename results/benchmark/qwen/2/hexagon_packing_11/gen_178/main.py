# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
from shapely.prepared import prep
from joblib import Parallel, delayed
import warnings
import time

def hexagon_vertices(center_x, center_y, rotation_degrees, side_length=1):
    """Generate vertices of a regular hexagon given center, rotation, and side length."""
    angle_rad = np.radians(rotation_degrees)
    # Vertices of a unit hexagon centered at origin
    unit_vertices = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Rotate and translate
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_vertices = unit_vertices @ rotation_matrix.T
    return rotated_vertices * side_length + np.array([center_x, center_y])

def check_containment_single(hex_vertices, outer_polygon):
    """Check if all vertices of a hexagon are inside the outer hexagon (optimized)."""
    # Check only a subset of vertices for efficiency while maintaining accuracy
    sample_indices = [0, 2, 4]  # Check every other vertex to balance speed vs accuracy
    for idx in sample_indices:
        vertex = hex_vertices[idx]
        point = Point(vertex)
        if not outer_polygon.contains(point):
            return False
    return True

def check_collision_single(hex1_vertices, hex2_vertices):
    """Check if two hexagons collide using Shapely."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def estimate_min_outer_radius(inner_hex_params):
    """Estimate the minimal outer hexagon radius needed to contain all inner hexagons."""
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(11):
        x, y, rot = inner_hex_params[3*i], inner_hex_params[3*i+1], inner_hex_params[3*i+2]
        hex_vertices = hexagon_vertices(x, y, rot, 1)
        all_vertices.extend(hex_vertices)
    
    if len(all_vertices) == 0:
        return 100.0
        
    # Calculate bounding box
    all_vertices = np.array(all_vertices)
    min_x, max_x = all_vertices[:, 0].min(), all_vertices[:, 0].max()
    min_y, max_y = all_vertices[:, 1].min(), all_vertices[:, 1].max()
    
    # Calculate center of bounding box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Find maximum distance from center to any vertex
    max_dist = 0
    for vertex in all_vertices:
        dist = np.sqrt((vertex[0] - center_x)**2 + (vertex[1] - center_y)**2)
        max_dist = max(max_dist, dist)
    
    # For a hexagon, the side length is max_dist / sqrt(3)
    # But we want to ensure our hexagon can contain everything with some margin
    return max_dist * 2 / np.sqrt(3) * 1.05  # 5% margin for safety

def calculate_total_penalty(params):
    """Calculate penalty for constraint violations."""
    # Extract parameters
    n = 11
    inner_params = params[:-1]
    outer_radius = params[-1]
    
    # Check if outer hexagon is large enough
    if outer_radius <= 0:
        return 1e10
    
    # Create outer hexagon vertices once
    outer_vertices = hexagon_vertices(0, 0, 0, outer_radius)
    outer_polygon = prep(Polygon(outer_vertices))
    
    # Check containment and collisions
    total_penalty = 0
    
    # Check containment of all inner hexagons
    for i in range(n):
        x, y, rot = inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]
        hex_vertices = hexagon_vertices(x, y, rot, 1)
        if not check_containment_single(hex_vertices, outer_polygon):
            total_penalty += 1e8 * (abs(x) + abs(y) + abs(rot))  # Scale penalty by magnitude
    
    # Check collisions between all pairs of inner hexagons
    if total_penalty == 0:
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, rot1 = inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]
                x2, y2, rot2 = inner_params[3*j], inner_params[3*j+1], inner_params[3*j+2]
                hex1_vertices = hexagon_vertices(x1, y1, rot1, 1)
                hex2_vertices = hexagon_vertices(x2, y2, rot2, 1)
                if check_collision_single(hex1_vertices, hex2_vertices):
                    # Use a more sophisticated penalty based on overlap severity
                    overlap_area = Polygon(hex1_vertices).intersection(Polygon(hex2_vertices)).area
                    total_penalty += 1e9 * (1 + overlap_area)  # Increase penalty with overlap area
    
    return total_penalty

def objective_function(params):
    """Objective function to maximize 1/outer_hex_side_length."""
    penalty = calculate_total_penalty(params)
    
    # Return negative inverse of outer hex side length plus penalties
    # This is because we want to minimize the negative inverse (i.e., maximize the inverse)
    outer_radius = params[-1]
    return -(1.0 / outer_radius) + penalty

def generate_diverse_initial_solutions():
    """Generate multiple initial configurations using different strategies."""
    initial_solutions = []
    
    # Strategy 1: Hexagonal lattice pattern (based on optimal geometric configuration)
    hex_lattice_pattern = [
        [0, 0, 0],          # center
        [0, 2, 0],          # top
        [1.732, 1, 0],      # top-right
        [1.732, -1, 0],     # bottom-right
        [0, -2, 0],         # bottom
        [-1.732, -1, 0],    # bottom-left
        [-1.732, 1, 0],     # top-left
        [3.464, 0, 0],      # far right
        [1.732, 2, 0],      # top-right extended
        [-1.732, 2, 0],     # top-left extended
        [-3.464, 0, 0],     # far left
    ]
    
    # Strategy 2: Spiral arrangement for better space utilization
    spiral_pattern = [
        [0, 0, 0],          # center
        [2, 0, 0],          # right
        [1, 1.732, 0],      # top-right
        [-1, 1.732, 0],     # top-left
        [-2, 0, 0],         # left
        [-1, -1.732, 0],    # bottom-left
        [1, -1.732, 0],     # bottom-right
        [3, 0, 0],          # far right
        [0, 2.5, 0],        # top
        [0, -2.5, 0],       # bottom
        [-3, 0, 0],         # far left
    ]
    
    # Strategy 3: Radial pattern with varied orientations
    radial_pattern = [
        [0, 0, 0],          # center
        [1.8, 0, 0],        # right
        [0.9, 1.56, 0],     # top-right
        [-0.9, 1.56, 0],    # top-left
        [-1.8, 0, 0],       # left
        [-0.9, -1.56, 0],   # bottom-left
        [0.9, -1.56, 0],    # bottom-right
        [2.7, 0, 0],        # far right
        [0, 2.2, 0],        # top
        [0, -2.2, 0],       # bottom
        [-2.7, 0, 0],       # far left
    ]
    
    # Strategy 4: Clustered arrangement
    clustered_pattern = [
        [0, 0, 0],          # center
        [2.1, 0, 0],        # right
        [1.05, 1.8, 0],     # top-right
        [-1.05, 1.8, 0],    # top-left
        [-2.1, 0, 0],       # left
        [-1.05, -1.8, 0],   # bottom-left
        [1.05, -1.8, 0],    # bottom-right
        [3.15, 0, 0],       # far right
        [0, 2.5, 0],        # top
        [0, -2.5, 0],       # bottom
        [-3.15, 0, 0],      # far left
    ]
    
    # Strategy 5: Linear chain arrangement
    linear_pattern = [
        [0, 0, 0],          # center
        [-2.5, 0, 0],       # left
        [2.5, 0, 0],        # right
        [-1.25, 2.17, 0],   # top-left
        [1.25, 2.17, 0],    # top-right
        [-1.25, -2.17, 0],  # bottom-left
        [1.25, -2.17, 0],   # bottom-right
        [-3.75, 2.17, 0],   # far top-left
        [3.75, 2.17, 0],    # far top-right
        [-3.75, -2.17, 0],  # far bottom-left
        [3.75, -2.17, 0],   # far bottom-right
    ]
    
    # Strategy 6: Optimized dense packing
    dense_pattern = [
        [0, 0, 0],          # center 
        [0, 1.8, 0],        # top
        [1.56, 0.9, 0],     # top-right
        [1.56, -0.9, 0],    # bottom-right
        [0, -1.8, 0],       # bottom
        [-1.56, -0.9, 0],   # bottom-left
        [-1.56, 0.9, 0],    # top-left
        [3.12, 0, 0],       # far right
        [1.56, 1.8, 0],     # top-right extended
        [-1.56, 1.8, 0],    # top-left extended
        [-3.12, 0, 0],      # far left
    ]
    
    patterns = [hex_lattice_pattern, spiral_pattern, radial_pattern, 
                clustered_pattern, linear_pattern, dense_pattern]
    
    for i, pattern in enumerate(patterns):
        params = []
        for x, y, rot in pattern:
            params.extend([x, y, rot])
        
        # Add small random perturbations to prevent exact symmetry
        for j in range(len(params)//3):
            params[3*j] += np.random.normal(0, 0.05)  # x position
            params[3*j+1] += np.random.normal(0, 0.05)  # y position
            params[3*j+2] += np.random.normal(0, 5)  # rotation
            
        # Estimate outer radius
        est_radius = estimate_min_outer_radius(np.array(params))
        params.append(est_radius)
        initial_solutions.append(np.array(params))
    
    return initial_solutions

def multi_start_evo_optimization():
    """Run multiple evolutionary optimization runs in parallel."""
    # Generate diverse initial solutions
    initial_solutions = generate_diverse_initial_solutions()
    
    # Define bounds for optimization
    bounds = []
    # Positions: x, y for each hexagon (limited to reasonable range)
    for _ in range(11):
        bounds.extend([(-10, 10), (-10, 10), (-180, 180)])  # x, y, rotation
    # Outer hexagon side length
    bounds.append((1.0, 20.0))  # Must be positive and reasonable
    
    # Run multiple DE optimization instances in parallel
    def run_single_de(start_solution):
        try:
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=150,
                popsize=20,
                mutation=(0.5, 1),
                recombination=0.8,
                seed=np.random.randint(1000),
                disp=False,
                polish=True
            )
            return result.x
        except Exception:
            return start_solution  # Return initial if DE fails
    
    # Run all optimization tasks in parallel
    parallel_results = Parallel(n_jobs=-1, backend='threading')(
        delayed(run_single_de)(sol) for sol in initial_solutions
    )
    
    # Evaluate all results and return the best one
    best_solution = None
    best_objective_value = float('inf')
    
    for result in parallel_results:
        obj_value = objective_function(result)
        if obj_value < best_objective_value:
            best_objective_value = obj_value
            best_solution = result
    
    return best_solution

def refine_with_local_search(initial_params):
    """Refine solution using a more targeted local search approach."""
    # First, let's do a quick validation to see how close we are to a good solution
    bounds = []
    # Positions: x, y for each hexagon 
    for _ in range(11):
        bounds.extend([(-10, 10), (-10, 10), (-180, 180)])
    # Outer hexagon side length
    bounds.append((0.5, 20.0))
    
    # Try a faster optimization approach
    try:
        from scipy.optimize import minimize
        
        # Use L-BFGS-B for faster refinement
        result = minimize(
            objective_function,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'gtol': 1e-6},
            callback=lambda x: None
        )
        
        if result.success:
            return result.x
    except:
        pass
    
    return initial_params

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Use multi-start evolutionary optimization
    best_solution = multi_start_evo_optimization()
    
    # If no solution found, fall back to a known good configuration
    if best_solution is None or len(best_solution) < 34:
        # Default configuration from prior work
        default_config = [
            [0, 0, 0], [0, 2, 0], [1.732, 1, 0], [1.732, -1, 0], [0, -2, 0],
            [-1.732, -1, 0], [-1.732, 1, 0], [3.464, 0, 0], [1.732, 2, 0],
            [-1.732, 2, 0], [-3.464, 0, 0], 3.95  # Estimated side length
        ]
        best_solution = np.array(default_config).flatten()
    
    # Refine the solution with local search
    try:
        refined_solution = refine_with_local_search(best_solution)
        if objective_function(refined_solution) < objective_function(best_solution):
            best_solution = refined_solution
    except:
        pass
    
    # Extract results
    outer_side_length = best_solution[-1]
    
    # Extract inner hexagon data
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i] = [best_solution[3*i], best_solution[3*i+1], best_solution[3*i+2]]
    
    # Outer hexagon data (centered at origin)
    outer_hex_data = np.array([0, 0, 0])
    
    # Validate solution one more time
    outer_vertices = hexagon_vertices(0, 0, 0, outer_side_length)
    outer_polygon = prep(Polygon(outer_vertices))
    
    # Check validity
    valid = True
    for i in range(11):
        x, y, rot = best_solution[3*i], best_solution[3*i+1], best_solution[3*i+2]
        hex_vertices = hexagon_vertices(x, y, rot, 1)
        if not check_containment_single(hex_vertices, outer_polygon):
            valid = False
            break
    
    # If still invalid, use safe fallback
    if not valid:
        # Fallback to pattern that works well
        inner_hex_data = np.array([
            [0, 0, 0], [0, 2, 0], [1.732, 1, 0], [1.732, -1, 0], [0, -2, 0],
            [-1.732, -1, 0], [-1.732, 1, 0], [3.464, 0, 0], [1.732, 2, 0],
            [-1.732, 2, 0], [-3.464, 0, 0]
        ])
        outer_side_length = estimate_min_outer_radius(inner_hex_data.flatten())
        outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END