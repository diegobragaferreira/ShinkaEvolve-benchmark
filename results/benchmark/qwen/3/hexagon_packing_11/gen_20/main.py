# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
import math
from scipy.optimize import differential_evolution
import time

def create_regular_hexagon(center_x, center_y, side_length=1, rotation_deg=0):
    """Create a regular hexagon as a Shapely polygon"""
    rotation_rad = math.radians(rotation_deg)
    points = []
    for i in range(6):
        angle = rotation_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        points.append((x, y))
    return Polygon(points)

def check_containment_and_overlap(inner_hexagons, outer_hexagon):
    """Check if all inner hexagons are contained in outer hexagon and don't overlap"""
    # Check containment
    for hex_poly in inner_hexagons:
        if not outer_hexagon.contains(hex_poly):
            return False
    
    # Check pairwise overlaps
    for i in range(len(inner_hexagons)):
        for j in range(i+1, len(inner_hexagons)):
            if inner_hexagons[i].intersects(inner_hexagons[j]):
                return False
    
    return True

def compute_outer_hexagon_radius(inner_hexagons, padding=0.01):
    """Compute minimum radius needed to contain all inner hexagons with some padding"""
    # Get all vertices of all hexagons
    all_vertices = []
    for hex_poly in inner_hexagons:
        all_vertices.extend(list(hex_poly.exterior.coords))
    
    # Find center of bounding box
    xs = [p[0] for p in all_vertices]
    ys = [p[1] for p in all_vertices]
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2
    
    # Compute max distance from center to any vertex
    max_dist = 0
    for x, y in all_vertices:
        dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = max(max_dist, dist)
    
    # Add padding and convert to side length
    # For a regular hexagon, radius = side_length
    return max_dist + padding

def evaluate_layout(inner_positions_angles, outer_center=(0, 0), initial_outer_radius=8):
    """Evaluate the layout quality"""
    # Convert to hexagon polygons
    inner_hexagons = []
    for pos_angle in inner_positions_angles:
        x, y, angle = pos_angle
        hex_poly = create_regular_hexagon(x, y, 1, angle)
        inner_hexagons.append(hex_poly)
    
    # Create outer hexagon with current radius
    outer_radius = compute_outer_hexagon_radius(inner_hexagons, 0.01)
    outer_hexagon = create_regular_hexagon(outer_center[0], outer_center[1], outer_radius, 0)
    
    # Validate constraints
    valid = check_containment_and_overlap(inner_hexagons, outer_hexagon)
    
    # Return negative because we want to maximize 1/R (minimize R)
    outer_side_length = outer_radius
    inv_radius = 1.0 / outer_side_length if valid else 0.0
    
    return -inv_radius, outer_side_length

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Initial arrangement: hexagonal lattice pattern based on geometric intuition
    initial_positions = [
        # Central hexagon
        [0, 0, 0],
        # First ring
        [-2.5, 0, 0],
        [2.5, 0, 0],
        [-1.25, 2.17, 0],
        [1.25, 2.17, 0],
        [-1.25, -2.17, 0],
        [1.25, -2.17, 0],
        # Second ring
        [-3.75, 2.17, 0],
        [3.75, 2.17, 0],
        [-3.75, -2.17, 0],
        [3.75, -2.17, 0],
    ]
    
    # Convert initial positions to numpy array
    inner_hex_data = np.array(initial_positions)
    
    # Optimization bounds for each parameter (x, y, angle)
    bounds = []
    for i in range(11):
        # x coordinate bounds
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])  # angle in degrees

    # Define objective function for optimization
    def objective(params):
        # Reshape parameters into positions and angles
        positions_angles = []
        for i in range(11):
            x = params[i*3]
            y = params[i*3 + 1]
            angle = params[i*3 + 2]
            positions_angles.append([x, y, angle])
        
        score, side_length = evaluate_layout(positions_angles)
        return score  # Negative since we minimize -score = maximize score

    # Use differential evolution for global optimization
    start_time = time.time()
    
    # Run optimization with limited time
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=15,
            seed=42,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7
        )
        
        # Extract best solution
        best_params = result.x
        final_positions_angles = []
        for i in range(11):
            x = best_params[i*3]
            y = best_params[i*3 + 1]
            angle = best_params[i*3 + 2]
            final_positions_angles.append([x, y, angle])
            
        # Evaluate final result
        final_score, final_side_length = evaluate_layout(final_positions_angles)
        best_inner_data = np.array(final_positions_angles)
        best_outer_side_length = final_side_length
        
    except Exception as e:
        # Fallback to heuristic improvement if optimization fails
        print(f"Optimization failed: {e}")
        best_inner_data = inner_hex_data.copy()
        best_outer_side_length = 8.0

    # Final refinement using local adjustment
    inner_hexagons = []
    for pos_angle in best_inner_data:
        x, y, angle = pos_angle
        hex_poly = create_regular_hexagon(x, y, 1, angle)
        inner_hexagons.append(hex_poly)
    
    # Compute current outer hexagon size
    outer_radius = compute_outer_hexagon_radius(inner_hexagons)
    outer_hexagon = create_regular_hexagon(0, 0, outer_radius, 0)
    
    # Validate constraints
    if not check_containment_and_overlap(inner_hexagons, outer_hexagon):
        # If invalid, fall back to initial configuration
        best_inner_data = inner_hex_data.copy()
        inner_hexagons = []
        for pos_angle in best_inner_data:
            x, y, angle = pos_angle
            hex_poly = create_regular_hexagon(x, y, 1, angle)
            inner_hexagons.append(hex_poly)
        outer_radius = compute_outer_hexagon_radius(inner_hexagons)
    
    # Ensure we're returning the correct data format
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    
    # Return results
    return best_inner_data, outer_hex_data, outer_radius

# EVOLVE-BLOCK-END
