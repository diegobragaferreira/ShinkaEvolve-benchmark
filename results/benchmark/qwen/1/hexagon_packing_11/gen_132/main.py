# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
from shapely.validation import make_valid
import time

def get_hexagon_vertices(center_x, center_y, side_length=1, rotation_deg=0):
    """Get vertices of a regular hexagon"""
    rotation_rad = np.radians(rotation_deg)
    angles = np.linspace(0, 2*np.pi, 7) + rotation_rad
    x_coords = center_x + side_length * np.cos(angles)
    y_coords = center_y + side_length * np.sin(angles)
    return list(zip(x_coords, y_coords))

def check_hexagon_containment(hexagon_vertices, outer_hex_center, outer_hex_side_length):
    """Check if hexagon is fully contained within outer hexagon"""
    outer_hex_vertices = get_hexagon_vertices(outer_hex_center[0], outer_hex_center[1], outer_hex_side_length, 0)
    outer_polygon = Polygon(outer_hex_vertices)
    
    hex_polygon = Polygon(hexagon_vertices)
    
    # Check if hexagon is fully contained
    try:
        return outer_polygon.contains(hex_polygon)
    except:
        try:
            valid_outer = make_valid(outer_polygon)
            valid_hex = make_valid(hex_polygon)
            return valid_outer.contains(valid_hex)
        except:
            return False

def calculate_outer_hexagon_radius(inner_hex_data, outer_hex_center=(0, 0)):
    """Calculate the minimum radius needed to contain all inner hexagons"""
    max_distance = 0
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        # Get all vertices of this hexagon
        vertices = get_hexagon_vertices(center_x, center_y, 1, angle)
        # Find the maximum distance from center to any vertex
        for vx, vy in vertices:
            dist = np.sqrt((vx - outer_hex_center[0])**2 + (vy - outer_hex_center[1])**2)
            max_distance = max(max_distance, dist)
    
    return max_distance

def is_valid_arrangement(inner_hex_data, outer_hex_center=(0, 0), outer_hex_side_length=None):
    """Check if the arrangement is valid (no overlaps and fully contained)"""
    if outer_hex_side_length is None:
        outer_hex_side_length = calculate_outer_hexagon_radius(inner_hex_data, outer_hex_center)
    
    # Check containment first
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = get_hexagon_vertices(center_x, center_y, 1, angle)
        if not check_hexagon_containment(vertices, outer_hex_center, outer_hex_side_length):
            return False, outer_hex_side_length
    
    # Check for overlaps between hexagons using shapely
    for i in range(len(inner_hex_data)):
        for j in range(i+1, len(inner_hex_data)):
            center_x1, center_y1, angle1 = inner_hex_data[i]
            center_x2, center_y2, angle2 = inner_hex_data[j]
            
            vertices1 = get_hexagon_vertices(center_x1, center_y1, 1, angle1)
            vertices2 = get_hexagon_vertices(center_x2, center_y2, 1, angle2)
            
            try:
                poly1 = Polygon(vertices1)
                poly2 = Polygon(vertices2)
                if poly1.intersects(poly2):
                    return False, outer_hex_side_length
            except:
                try:
                    valid_poly1 = make_valid(Polygon(vertices1))
                    valid_poly2 = make_valid(Polygon(vertices2))
                    if valid_poly1.intersects(valid_poly2):
                        return False, outer_hex_side_length
                except:
                    return False, outer_hex_side_length
    
    return True, outer_hex_side_length

def evaluate_fitness(individual, outer_hex_center=(0, 0)):
    """Evaluate fitness of an individual - higher is better"""
    # Reshape individual to (11, 3) array where each row is (x, y, angle)
    inner_hex_data = individual.reshape(-1, 3)
    
    # Calculate outer hexagon size needed
    outer_hex_side_length = calculate_outer_hexagon_radius(inner_hex_data, outer_hex_center)
    
    # Check validity
    valid, adjusted_radius = is_valid_arrangement(inner_hex_data, outer_hex_center, outer_hex_side_length)
    
    if not valid:
        # Penalize invalid arrangements heavily
        return -1000000
    
    # The fitness is the inverse of the outer hexagon side length (we want to minimize it)
    return 1.0 / adjusted_radius

def create_initial_geometric_arrangement():
    """Create a smart initial geometric arrangement"""
    # Start with a known good configuration based on hexagonal packing principles
    # Place central hexagon, then surrounding hexagons in rings
    positions = [
        (0, 0),          # center
        (-2.5, 0),       # left
        (2.5, 0),        # right
        (0, 2.5),        # top
        (0, -2.5),       # bottom
        (-1.25, 2.17),   # top-left
        (1.25, 2.17),    # top-right
        (-1.25, -2.17),  # bottom-left
        (1.25, -2.17),   # bottom-right
        (-2.5, 2.17),    # far top
        (2.5, 2.17),     # far top
    ]
    
    # Create initial individual with slight perturbations
    individual = []
    for i, (x, y) in enumerate(positions):
        # Add small random perturbations
        x += np.random.normal(0, 0.1)
        y += np.random.normal(0, 0.1)
        # Set rotation to 0 for all initially
        angle = 0
        individual.append([x, y, angle])
    
    return np.array(individual).flatten()

def optimize_with_local_search(initial_individual):
    """Use local optimization to fine-tune the arrangement"""
    # Define bounds for each parameter
    bounds = []
    for i in range(33):  # 11 hexagons * 3 parameters each
        if i % 3 == 0:  # x coordinate
            bounds.append((-10, 10))
        elif i % 3 == 1:  # y coordinate
            bounds.append((-10, 10))
        else:  # angle
            bounds.append((0, 360))
    
    # Use L-BFGS-B optimizer for local refinement
    result = minimize(
        lambda x: -evaluate_fitness(x),  # Negative because minimize finds minimum
        initial_individual,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-6}
    )
    
    return result.x

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Create initial geometric arrangement
    initial_individual = create_initial_geometric_arrangement()
    
    # Perform local optimization on the initial arrangement
    optimized_individual = optimize_with_local_search(initial_individual)
    
    # Final evaluation of best solution
    final_fitness = evaluate_fitness(optimized_individual)
    inner_hex_data = optimized_individual.reshape(-1, 3)
    
    # Determine the actual outer hexagon size needed
    outer_hex_side_length = 1.0 / final_fitness if final_fitness > 0 else 1000.0
    
    # Outer hexagon data - centered at origin with no rotation (for consistency)
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
