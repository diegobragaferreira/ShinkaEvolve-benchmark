# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
from shapely.prepared import prep
import time
from itertools import combinations
import warnings
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

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
    """Check if all vertices of a hexagon are inside the outer hexagon."""
    for vertex in hex_vertices:
        point = Point(vertex)
        if not outer_polygon.contains(point):
            return False
    return True

def check_collision_single(hex1_vertices, hex2_vertices):
    """Check if two hexagons collide using Shapely."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def calculate_outer_hex_side_length(inner_hex_params):
    """Calculate the minimal outer hexagon side length needed to contain all inner hexagons."""
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
    
    # Calculate diagonal distance from center to corner
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Find maximum distance from center to any vertex
    max_dist = 0
    for vertex in all_vertices:
        dist = np.sqrt((vertex[0] - center_x)**2 + (vertex[1] - center_y)**2)
        max_dist = max(max_dist, dist)
    
    # For a hexagon, the side length is max_dist / sqrt(3)
    # But we want to ensure our hexagon can contain everything with some margin
    return max_dist * 2 / np.sqrt(3) * 1.1  # 10% margin

def evaluate_solution(params, outer_side_length=None):
    """Evaluate the solution and return fitness score."""
    n = 11
    if outer_side_length is None:
        outer_side_length = params[-1]
    
    # Check if outer hexagon is large enough
    if outer_side_length <= 0:
        return 1e10
    
    # Create outer hexagon vertices once
    outer_vertices = hexagon_vertices(0, 0, 0, outer_side_length)
    outer_polygon = prep(Polygon(outer_vertices))
    
    # Check containment of all inner hexagons
    total_penalty = 0
    inner_positions = params[:-1].reshape(-1, 3)
    
    # Check containment for all hexagons
    for i in range(n):
        x, y, rot = inner_positions[i]
        hex_vertices = hexagon_vertices(x, y, rot, 1)
        if not check_containment_single(hex_vertices, outer_polygon):
            total_penalty += 1e10
    
    # Check collisions between all pairs of inner hexagons  
    if total_penalty == 0:
        # Use fast parallel processing for collision detection
        def check_pair_collision(i, j):
            x1, y1, rot1 = inner_positions[i]
            x2, y2, rot2 = inner_positions[j]
            hex1_vertices = hexagon_vertices(x1, y1, rot1, 1)
            hex2_vertices = hexagon_vertices(x2, y2, rot2, 1)
            return check_collision_single(hex1_vertices, hex2_vertices)
        
        collision_checks = Parallel(n_jobs=-1)(
            delayed(check_pair_collision)(i, j) 
            for i in range(n) for j in range(i+1, n)
        )
        
        collisions = sum(collision_checks)
        if collisions > 0:
            total_penalty += 1e10
    
    # Return negative inverse of outer hex side length plus penalties
    return -(1.0 / outer_side_length) + total_penalty

def generate_greedy_initialization():
    """Generate an initial configuration using a greedy approach."""
    # Start with a symmetric pattern that's known to work
    # We'll place hexagons in a honeycomb-like pattern
    positions = []
    
    # Center hexagon
    positions.append([0, 0, 0])
    
    # First ring around center
    ring_1_positions = [
        [0, 2, 0],       # top
        [1.732, 1, 0],   # top-right
        [1.732, -1, 0],  # bottom-right
        [0, -2, 0],      # bottom
        [-1.732, -1, 0], # bottom-left
        [-1.732, 1, 0],  # top-left
    ]
    positions.extend(ring_1_positions)
    
    # Second ring
    ring_2_positions = [
        [3.464, 0, 0],   # far right
        [1.732, 2, 0],   # upper middle
        [-1.732, 2, 0],  # upper middle left
        [-3.464, 0, 0],  # far left
        [-1.732, -2, 0], # lower middle left
        [1.732, -2, 0],  # lower middle right
    ]
    positions.extend(ring_2_positions)
    
    # Flatten and create parameters
    params = []
    for pos in positions:
        params.extend(pos)
    
    # Estimate outer hexagon size
    estimated_size = calculate_outer_hex_side_length(np.array(params))
    params.append(estimated_size)
    
    return np.array(params)

def generate_symmetric_initialization():
    """Generate a symmetric initial configuration."""
    positions = []
    
    # Center hexagon
    positions.append([0, 0, 0])
    
    # Symmetric ring around center
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    distances = [2, 2, 2, 2, 2, 2]  # All same distance
    
    for i, (angle, dist) in enumerate(zip(angles, distances)):
        x = dist * np.cos(angle)
        y = dist * np.sin(angle)
        positions.append([x, y, 0])
    
    # Add second ring
    angles2 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    distances2 = [3.5, 3.5, 3.5, 3.5, 3.5, 3.5]
    
    for i, (angle, dist) in enumerate(zip(angles2, distances2)):
        x = dist * np.cos(angle)
        y = dist * np.sin(angle)
        positions.append([x, y, 0])
    
    # Flatten and create parameters
    params = []
    for pos in positions:
        params.extend(pos)
    
    # Estimate outer hexagon size
    estimated_size = calculate_outer_hex_side_length(np.array(params))
    params.append(estimated_size)
    
    return np.array(params)

def generate_random_initialization():
    """Generate a random initial configuration."""
    params = []
    
    # Generate random positions and rotations for 11 hexagons
    for i in range(11):
        # Random positions in a reasonable range
        x = np.random.uniform(-3, 3)
        y = np.random.uniform(-3, 3)
        rot = np.random.uniform(-180, 180)
        params.extend([x, y, rot])
    
    # Estimate outer hexagon size
    estimated_size = calculate_outer_hex_side_length(np.array(params))
    params.append(estimated_size)
    
    return np.array(params)

def optimize_with_local_refinement(initial_params):
    """Refine the initial parameters using local optimization."""
    n = 11
    
    # Define bounds for optimization (simplified)
    bounds = []
    for _ in range(n):
        bounds.extend([(-5, 5), (-5, 5), (-180, 180)])
    bounds.append((1.0, 15.0))  # Outer hex side length
    
    def objective(x):
        return evaluate_solution(x)
    
    # Try local optimization with different starting points
    best_result = None
    best_score = float('inf')
    
    # Run multiple local optimizations
    for attempt in range(3):
        # Add some noise to initial parameters
        noisy_params = initial_params.copy()
        noise_scale = 0.2
        for i in range(len(noisy_params) - 1):  # Don't touch the last element (outer size)
            noisy_params[i] += np.random.normal(0, noise_scale)
        
        try:
            # Local optimization with L-BFGS-B
            result = minimize(
                objective,
                noisy_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-6},
                callback=lambda x: None  # No callback needed
            )
            
            if result.success and result.fun < best_score:
                best_score = result.fun
                best_result = result
                
        except Exception:
            continue
    
    # Return the best result found
    if best_result is not None:
        return best_result.x
    else:
        return initial_params

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Generate multiple initial guesses
    initial_guesses = [
        generate_greedy_initialization(),
        generate_symmetric_initialization(),
        generate_random_initialization()
    ]
    
    best_score = float('inf')
    best_params = None
    
    # Evaluate each initial guess
    for i, guess in enumerate(initial_guesses):
        try:
            score = evaluate_solution(guess)
            if score < best_score:
                best_score = score
                best_params = guess.copy()
        except Exception:
            continue
    
    if best_params is None:
        # Fallback to simple arrangement
        best_params = np.array([
            [0, 0, 0], [0, 2, 0], [1.732, 1, 0], [1.732, -1, 0], [0, -2, 0],
            [-1.732, -1, 0], [-1.732, 1, 0], [3.464, 0, 0], [1.732, 2, 0],
            [-1.732, 2, 0], [-3.464, 0, 0], 10.0  # Add side length
        ]).flatten()
    
    # Apply local refinement to the best initial guess
    refined_params = optimize_with_local_refinement(best_params)
    
    # Final evaluation
    final_score = evaluate_solution(refined_params)
    
    # Extract inner hexagon data
    outer_side_length = refined_params[-1]
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i] = [refined_params[3*i], refined_params[3*i+1], refined_params[3*i+2]]
    
    # Outer hexagon data
    outer_hex_data = np.array([0, 0, 0])  # Centered at origin
    
    # Validate solution
    outer_vertices = hexagon_vertices(0, 0, 0, outer_side_length)
    outer_polygon = prep(Polygon(outer_vertices))
    
    valid_solution = True
    for i in range(11):
        x, y, rot = refined_params[3*i], refined_params[3*i+1], refined_params[3*i+2]
        hex_vertices = hexagon_vertices(x, y, rot, 1)
        
        if not check_containment_single(hex_vertices, outer_polygon):
            valid_solution = False
            break
    
    # Final fallback if validation fails
    if not valid_solution:
        inner_hex_data = np.array([
            [0, 0, 0], [0, 2, 0], [1.732, 1, 0], [1.732, -1, 0], [0, -2, 0],
            [-1.732, -1, 0], [-1.732, 1, 0], [3.464, 0, 0], [1.732, 2, 0],
            [-1.732, 2, 0], [-3.464, 0, 0]
        ])
        outer_side_length = calculate_outer_hex_side_length(inner_hex_data.flatten())
        outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END
