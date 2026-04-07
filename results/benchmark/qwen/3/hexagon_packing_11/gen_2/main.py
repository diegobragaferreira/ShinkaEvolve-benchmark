# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
from numba import jit
import time
import random

@jit(nopython=True)
def hexagon_vertices(x, y, angle_deg, side_length=1):
    """Compute vertices of a regular hexagon given position and rotation."""
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Vertices of a unit hexagon centered at origin
    base_vertices = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Scale and rotate
    scaled = base_vertices * side_length
    
    # Rotate and translate
    rotated = np.zeros_like(scaled)
    for i in range(6):
        rotated[i, 0] = scaled[i, 0] * cos_a - scaled[i, 1] * sin_a
        rotated[i, 1] = scaled[i, 0] * sin_a + scaled[i, 1] * cos_a
    
    # Translate
    vertices = rotated + np.array([x, y])
    
    return vertices

def hexagon_polygon(x, y, angle_deg, side_length=1):
    """Create shapely Polygon representation of hexagon."""
    vertices = hexagon_vertices(x, y, angle_deg, side_length)
    return Polygon(vertices)

def check_containment(hexagon_poly, outer_hex_poly):
    """Check if a hexagon is fully contained within the outer hexagon."""
    return outer_hex_poly.contains(hexagon_poly)

def check_overlap(hex1_poly, hex2_poly):
    """Check if two hexagons overlap."""
    return hex1_poly.intersects(hex2_poly)

def compute_outer_hexagon_radius(inner_positions, inner_angles):
    """Estimate the minimum radius needed for outer hexagon to contain all inner hexagons."""
    max_dist = 0
    for i in range(len(inner_positions)):
        x, y = inner_positions[i]
        # Distance from center to corner of hexagon
        dist = np.sqrt(x*x + y*y) + np.sqrt(3)  # Approximate distance to farthest point
        if dist > max_dist:
            max_dist = dist
    return max_dist

def evaluate_solution(params):
    """Evaluate a solution by computing the inverse of outer hexagon side length."""
    # Extract positions and angles
    positions = params[0::2].reshape(-1, 2)
    angles = params[1::2]
    # Set outer hexagon centered at (0,0) with estimated radius
    outer_radius = compute_outer_hexagon_radius(positions, angles)
    outer_side_length = outer_radius / np.cos(np.pi/6)  # Convert radius to side length
    
    # Create polygons
    inner_polygons = []
    for i in range(len(positions)):
        poly = hexagon_polygon(positions[i][0], positions[i][1], angles[i])
        inner_polygons.append(poly)
    
    outer_poly = hexagon_polygon(0, 0, 0, outer_side_length)
    
    # Check containment and overlaps
    for i in range(len(inner_polygons)):
        if not check_containment(inner_polygons[i], outer_poly):
            return 1e10  # Penalty for violation
        for j in range(i+1, len(inner_polygons)):
            if check_overlap(inner_polygons[i], inner_polygons[j]):
                return 1e10  # Penalty for overlap
    
    # Return inverse of outer side length
    return 1.0 / outer_side_length

def optimize_hexagon_packing():
    """Main optimization function using differential evolution."""
    n_inner = 11
    
    # Initial guess - organized layout
    initial_positions = np.array([
        [0, 0],
        [-2.5, 0],
        [2.5, 0],
        [-1.25, 2.17],
        [1.25, 2.17],
        [-1.25, -2.17],
        [1.25, -2.17],
        [-3.75, 2.17],
        [3.75, 2.17],
        [-3.75, -2.17],
        [3.75, -2.17]
    ])
    
    initial_angles = np.zeros(n_inner)
    
    # Combine into single parameter array
    initial_params = np.hstack([initial_positions.flatten(), initial_angles])
    
    # Define bounds: positions [-10, 10] and angles [0, 360]
    bounds = []
    for _ in range(n_inner):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])
    
    # Optimization settings
    maxiter = 100
    popsize = 50
    mutation = 0.8
    recombination = 0.7
    
    # Run optimization
    result = differential_evolution(
        evaluate_solution,
        bounds,
        maxiter=maxiter,
        popsize=popsize,
        mutation=mutation,
        recombination=recombination,
        seed=42,
        disp=False
    )
    
    # Extract best solution
    best_positions = result.x[0::3].reshape(-1, 2)
    best_angles = result.x[2::3]
    
    # Refine with local search
    refined_params = local_search(best_positions, best_angles)
    final_positions = refined_params[0::3].reshape(-1, 2)
    final_angles = refined_params[2::3]
    
    return final_positions, final_angles

def local_search(initial_positions, initial_angles):
    """Perform simple local search around best found solution."""
    n_inner = len(initial_positions)
    params = np.hstack([initial_positions.flatten(), initial_angles])
    
    def objective(p):
        return evaluate_solution(p)
    
    # Simple gradient descent-like local search
    learning_rate = 0.01
    max_iter = 50
    for _ in range(max_iter):
        current_val = objective(params)
        grad = numerical_gradient(params, objective)
        params -= learning_rate * grad
        
        # Ensure parameters stay within bounds
        for i in range(n_inner):
            params[3*i] = np.clip(params[3*i], -10, 10)
            params[3*i+1] = np.clip(params[3*i+1], -10, 10)
            params[3*i+2] = params[3*i+2] % 360
    
    return params

def numerical_gradient(x, f, eps=1e-5):
    """Numerical gradient computation."""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Get optimized solution
    positions, angles = optimize_hexagon_packing()
    
    # Calculate outer hexagon side length
    inner_coords = np.array([[positions[i][0], positions[i][1]] for i in range(11)])
    inner_angles = angles
    outer_radius = compute_outer_hexagon_radius(inner_coords, inner_angles)
    outer_side_length = outer_radius / np.cos(np.pi/6)
    
    # Format output data
    inner_hex_data = np.column_stack([positions, angles])
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    
    end_time = time.time()
    
    return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END
