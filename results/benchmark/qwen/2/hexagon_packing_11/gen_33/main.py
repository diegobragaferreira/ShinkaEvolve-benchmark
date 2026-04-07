# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from scipy.optimize import minimize
import time
from numba import jit

# Constants
NUM_INNER_HEX = 11
UNIT_HEX_RADIUS = 1.0
HEX_VERTICES = 6

@jit(nopython=True)
def get_hexagon_vertices(x, y, angle_deg, radius=1.0):
    """Get vertices of a regular hexagon given position and angle"""
    vertices = np.zeros((6, 2))
    angle_rad = np.radians(angle_deg)
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i] = [x + radius * np.cos(theta), y + radius * np.sin(theta)]
    return vertices

def calculate_outer_hex_side_length(inner_hex_data, outer_hex_center=(0, 0)):
    """Calculate minimum outer hexagon side length needed to contain all inner hexagons"""
    if len(inner_hex_data) == 0:
        return 1000.0
    
    max_distance = 0.0
    center_x, center_y = outer_hex_center
    
    # For each inner hexagon, check all 6 vertices
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        vertices = get_hexagon_vertices(x, y, angle)
        
        # Calculate distance from center to each vertex
        for vertex in vertices:
            distance = np.sqrt((vertex[0] - center_x)**2 + (vertex[1] - center_y)**2)
            max_distance = max(max_distance, distance)
    
    # Account for hexagon radius
    # The outer hexagon needs to be large enough so that any vertex of inner hexagons 
    # lies inside the outer hexagon
    return max_distance * 2.0 / np.sqrt(3)  # Convert circumradius to side length

def check_containment(hex_vertices, outer_center=(0, 0), outer_radius=1000.0):
    """Check if hexagon vertices are within the outer hexagon"""
    outer_center_x, outer_center_y = outer_center
    # Check if all vertices are within the outer hexagon
    # Outer hexagon circumscribed circle has radius = outer_radius * sqrt(3)/2
    outer_circumradius = outer_radius * np.sqrt(3) / 2
    
    for vertex in hex_vertices:
        dist_from_center = np.sqrt((vertex[0] - outer_center_x)**2 + (vertex[1] - outer_center_y)**2)
        if dist_from_center > outer_circumradius:
            return False
    return True

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagon vertices arrays overlap"""
    # Create Shapely polygons
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    
    # Check for intersection
    return poly1.intersects(poly2)

def compute_forces(hex_data, outer_radius):
    """Compute total forces acting on each hexagon"""
    n = len(hex_data)
    forces = np.zeros((n, 2))  # Force vectors (dx, dy) for each hexagon
    
    # Repulsive forces between overlapping hexagons
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, ang1 = hex_data[i]
            x2, y2, ang2 = hex_data[j]
            
            # Get vertices for both hexagons
            verts1 = get_hexagon_vertices(x1, y1, ang1)
            verts2 = get_hexagon_vertices(x2, y2, ang2)
            
            # Check for overlap
            if check_overlap(verts1, verts2):
                # Compute repulsive force along line connecting centers
                dx = x1 - x2
                dy = y1 - y2
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance > 0.01:  # Avoid division by zero
                    # Normalize and scale with inverse distance
                    fx = dx / distance * 1000.0 / (distance * distance + 1.0)
                    fy = dy / distance * 1000.0 / (distance * distance + 1.0)
                    
                    forces[i][0] += fx
                    forces[i][1] += fy
                    forces[j][0] -= fx
                    forces[j][1] -= fy
    
    # Attractive force towards center (gravity)
    for i in range(n):
        x, y, angle = hex_data[i]
        dx = -x
        dy = -y
        
        # Normalize and scale force
        distance = np.sqrt(dx*dx + dy*dy)
        if distance > 0.01:
            fx = dx / distance * 0.1
            fy = dy / distance * 0.1
            
            forces[i][0] += fx
            forces[i][1] += fy
    
    return forces

def simulate_hexagon_system(hex_data, max_iterations=5000):
    """Run physics simulation to optimize hexagon positions"""
    # Start with a relatively large outer radius to avoid immediate containment issues
    outer_radius = 100.0
    
    for iteration in range(max_iterations):
        # Check current state and update outer radius if needed
        current_radius = calculate_outer_hex_side_length(hex_data)
        if current_radius < outer_radius:
            outer_radius = current_radius * 1.2  # Add some margin
        
        # Compute forces
        forces = compute_forces(hex_data, outer_radius)
        
        # Update positions (with damping)
        damping = 0.1
        for i in range(len(hex_data)):
            x, y, angle = hex_data[i]
            # Apply force and damping
            new_x = x + forces[i][0] * damping
            new_y = y + forces[i][1] * damping
            
            # Keep angles within 0-360 range
            new_angle = angle % 360
            
            hex_data[i] = [new_x, new_y, new_angle]
        
        # Occasionally recenter the system to prevent drift
        if iteration % 20 == 0:
            avg_x = np.mean(hex_data[:, 0])
            avg_y = np.mean(hex_data[:, 1])
            for i in range(len(hex_data)):
                hex_data[i][0] -= avg_x
                hex_data[i][1] -= avg_y
    
    return hex_data

def objective_function(hex_data):
    """Objective function to minimize (negative of 1/outer_radius)"""
    outer_radius = calculate_outer_hex_side_length(hex_data)
    return -1.0 / outer_radius

def evaluate_solution(hex_data):
    """Complete solution evaluation including constraints"""
    # Check containment and overlaps
    outer_radius = calculate_outer_hex_side_length(hex_data)
    
    # Check containment constraints
    outer_circumradius = outer_radius * np.sqrt(3) / 2
    penalty = 0.0
    
    # Check each hexagon for containment
    for i in range(NUM_INNER_HEX):
        x, y, angle = hex_data[i]
        vertices = get_hexagon_vertices(x, y, angle)
        if not check_containment(vertices, (0, 0), outer_circumradius):
            penalty += 1000000.0  # Heavy penalty
    
    # Check for overlaps between hexagons
    for i in range(NUM_INNER_HEX):
        for j in range(i+1, NUM_INNER_HEX):
            x1, y1, angle1 = hex_data[i]
            x2, y2, angle2 = hex_data[j]
            
            vertices1 = get_hexagon_vertices(x1, y1, angle1)
            vertices2 = get_hexagon_vertices(x2, y2, angle2)
            
            if check_overlap(vertices1, vertices2):
                penalty += 1000000.0  # Heavy penalty
    
    # Return fitness (negative inverse of side length plus penalties)
    fitness = -1.0 / outer_radius
    if penalty > 0:
        fitness -= penalty
    
    return fitness, outer_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Initial configuration - attempt to place hexagons in a symmetric pattern
    initial_positions = [
        [0, 0, 0],      # center
        [-1.8, 0, 0],   # left
        [1.8, 0, 0],    # right
        [0, 1.8, 0],    # top
        [0, -1.8, 0],   # bottom
        [-1.2, 1.2, 0], # top-left
        [1.2, 1.2, 0],  # top-right
        [-1.2, -1.2, 0], # bottom-left
        [1.2, -1.2, 0], # bottom-right
        [-2.5, 0, 0],   # far left
        [2.5, 0, 0],    # far right
    ]
    
    # Convert to numpy array
    hex_data = np.array(initial_positions)
    
    # Run physics simulation
    start_time = time.time()
    max_time = 170  # seconds
    
    # Simulate until time limit
    while time.time() - start_time < max_time:
        # Run simulation step
        hex_data = simulate_hexagon_system(hex_data.copy(), 1000)
        
        # Check if we've found a better solution
        _, current_outer_radius = evaluate_solution(hex_data)
        
        # Break if time is up
        if time.time() - start_time > max_time - 1:
            break
    
    # Final evaluation
    final_fitness, outer_hex_side_length = evaluate_solution(hex_data)
    
    # Create outer hexagon data
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    
    # If we didn't find anything good, fall back to simple grid
    if outer_hex_side_length > 50:
        hex_data = np.array([
            [0, 0, 0],      # center
            [-2.5, 0, 0],   # left
            [2.5, 0, 0],    # right
            [-1.25, 2.17, 0],   # top-left
            [1.25, 2.17, 0],    # top-right
            [-1.25, -2.17, 0],  # bottom-left
            [1.25, -2.17, 0],   # bottom-right
            [-3.75, 2.17, 0],   # far top-left
            [3.75, 2.17, 0],    # far top-right
            [-3.75, -2.17, 0],  # far bottom-left
            [3.75, -2.17, 0],   # far bottom-right
        ])
        outer_hex_side_length = calculate_outer_hex_side_length(hex_data)
        outer_hex_data = np.array([0, 0, 0])
    
    return hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
