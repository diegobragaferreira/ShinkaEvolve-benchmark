# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
from scipy.spatial import Voronoi
import time

def hexagon_vertices(center_x, center_y, rotation_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, rotation, and side length."""
    angle_rad = np.radians(rotation_deg)
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

def check_containment(hexagon_vertices, outer_hex_vertices):
    """Check if all vertices of a hexagon are inside the outer hexagon."""
    outer_polygon = Polygon(outer_hex_vertices)
    for vertex in hexagon_vertices:
        point = Point(vertex)
        if not outer_polygon.contains(point):
            return False
    return True

def check_collision(hex1_vertices, hex2_vertices):
    """Check if two hexagons collide using Shapely."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def calculate_voronoi_density(points, outer_side_length):
    """Calculate a measure of how well points fill space based on Voronoi cells."""
    # Create Voronoi diagram for the points
    vor = Voronoi(points)
    
    # Calculate average cell area
    total_area = 0
    num_cells = 0
    
    for region in vor.regions:
        if len(region) > 0 and -1 not in region:
            # Get the vertices of this region
            vertices = [vor.vertices[i] for i in region if i >= 0]
            if len(vertices) >= 3:
                try:
                    poly = Polygon(vertices)
                    total_area += poly.area
                    num_cells += 1
                except:
                    continue
    
    if num_cells > 0:
        avg_area = total_area / num_cells
        # Normalize by expected area for unit hexagon
        expected_area = (3 * np.sqrt(3) / 2) * 1  # Area of unit hexagon
        return avg_area / expected_area
    else:
        return 0

def objective_function(params, outer_side_length):
    """Objective function to minimize (negative inverse of outer hex side length)."""
    n = 11
    # Reshape parameters into positions and rotations
    positions = params[:2*n].reshape(-1, 2)
    rotations = params[2*n:3*n]
    
    # Create outer hexagon vertices
    outer_vertices = hexagon_vertices(0, 0, 0, outer_side_length)
    
    # Check containment and collisions
    total_penalty = 0
    
    # Check containment of all inner hexagons
    for i in range(n):
        x, y = positions[i]
        rot = rotations[i]
        hex_vertices = hexagon_vertices(x, y, rot, 1)
        
        # Check if all vertices are within outer hexagon
        if not check_containment(hex_vertices, outer_vertices):
            total_penalty += 1e6
    
    # Check collisions between all pairs of inner hexagons
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = positions[i]
            rot1 = rotations[i]
            x2, y2 = positions[j]
            rot2 = rotations[j]
            
            hex1_vertices = hexagon_vertices(x1, y1, rot1, 1)
            hex2_vertices = hexagon_vertices(x2, y2, rot2, 1)
            
            if check_collision(hex1_vertices, hex2_vertices):
                total_penalty += 1e6
    
    # Penalize for being too close to boundary
    for i in range(n):
        x, y = positions[i]
        distance_from_center = np.sqrt(x*x + y*y)
        # If too close to outer boundary, penalize
        if distance_from_center > outer_side_length - 1.1:
            total_penalty += 1e4
    
    # Return negative inverse of outer hex side length plus penalties
    return -(1.0 / outer_side_length) + total_penalty

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Start with a good hexagonal arrangement
    # First, create a hexagonal lattice pattern that fits well
    positions = []
    
    # Center hexagon
    positions.append([0, 0])
    
    # First ring
    for i in range(6):
        angle = i * np.pi / 3
        x = 2 * np.cos(angle)
        y = 2 * np.sin(angle)
        positions.append([x, y])
    
    # Second ring
    for i in range(6):
        angle = i * np.pi / 3 + np.pi/6
        x = 3.464 * np.cos(angle)
        y = 3.464 * np.sin(angle)
        positions.append([x, y])
    
    # Take first 11 positions
    positions = positions[:11]
    
    # Initial reasonable guess for outer side length
    outer_side_length = 8.0
    
    # Create initial parameter vector (positions and rotations)
    initial_params = []
    for i, (x, y) in enumerate(positions):
        initial_params.extend([x, y, 0])  # x, y, rotation (initially 0)
    
    # Add a reasonable initial guess for outer side length  
    initial_params.append(outer_side_length)
    initial_params = np.array(initial_params)
    
    # Bounds for optimization
    bounds = []
    
    # Position bounds for all 11 hexagons
    for i in range(11):
        bounds.extend([(-10, 10), (-10, 10)])  # x, y bounds
    
    # Rotation bounds for all 11 hexagons  
    for i in range(11):
        bounds.append((-180, 180))  # rotation bounds
    
    # Outer hexagon side length bound
    bounds.append((1.0, 20.0))
    
    # Optimizer settings
    options = {'maxiter': 2000, 'disp': False, 'ftol': 1e-8, 'gtol': 1e-8}
    
    # Define constraints for the optimization
    def constraint_func(params):
        # Simple constraint for feasibility - this will be handled by penalty in objective
        return 0.0
    
    # Use scipy minimize with L-BFGS-B
    try:
        result = minimize(
            lambda p: objective_function(p, outer_side_length),
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options=options,
            tol=1e-8
        )
        
        # Extract results
        best_params = result.x
        n = 11
        positions = best_params[:2*n].reshape(-1, 2)
        rotations = best_params[2*n:3*n]
        final_outer_side_length = best_params[-1]
        
        # Create inner hex data
        inner_hex_data = np.zeros((n, 3))
        for i in range(n):
            inner_hex_data[i] = [positions[i][0], positions[i][1], rotations[i]]
        
        # Create outer hex data (centered at origin)
        outer_hex_data = np.array([0, 0, 0])
        
        # Validate the solution
        outer_vertices = hexagon_vertices(0, 0, 0, final_outer_side_length)
        
        valid_solution = True
        
        # Check all constraints
        for i in range(n):
            x, y = positions[i]
            rot = rotations[i]
            hex_vertices = hexagon_vertices(x, y, rot, 1)
            
            # Check containment
            if not check_containment(hex_vertices, outer_vertices):
                valid_solution = False
                break

        # If not valid, fallback to a simpler approach
        if not valid_solution or final_outer_side_length < 1.0:
            raise ValueError("Invalid solution")
            
    except Exception as e:
        # Fallback to simple arrangement in case of optimization failure
        inner_hex_data = np.array([
            [0, 0, 0],  # center
            [-2.5, 0, 0],  # left
            [2.5, 0, 0],  # right
            [-1.25, 2.17, 0],  # top-left
            [1.25, 2.17, 0],  # top-right
            [-1.25, -2.17, 0],  # bottom-left
            [1.25, -2.17, 0],  # bottom-right
            [-3.75, 2.17, 0],  # far top-left
            [3.75, 2.17, 0],  # far top-right
            [-3.75, -2.17, 0],  # far bottom-left
            [3.75, -2.17, 0],  # far bottom-right
        ])
        outer_hex_data = np.array([0, 0, 0])
        final_outer_side_length = 8.0

    return inner_hex_data, outer_hex_data, final_outer_side_length

# EVOLVE-BLOCK-END