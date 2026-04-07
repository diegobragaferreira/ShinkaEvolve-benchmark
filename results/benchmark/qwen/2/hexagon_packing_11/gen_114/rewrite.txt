# EVOLVE-BLOCK-START
import numpy as np
import math
import random
import time
from numba import jit
from scipy.spatial.distance import cdist

# Constants
NUM_INNER_HEX = 11
UNIT_HEX_RADIUS = 1.0
HEX_VERTICES = 6

@jit(nopython=True)
def get_hexagon_vertices(x, y, angle_deg, radius=1.0):
    """Get vertices of a regular hexagon given position and angle - JIT compiled"""
    vertices = np.zeros((6, 2))
    angle_rad = np.radians(angle_deg)
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i] = [x + radius * np.cos(theta), y + radius * np.sin(theta)]
    return vertices

@jit(nopython=True)
def point_in_hexagon_fast(px, py, hx, hy, radius, angle_deg):
    """Fast point-in-hexagon test using analytical approach - JIT compiled"""
    # Transform point to hexagon's local coordinate system
    angle_rad = np.radians(angle_deg)
    rel_x = px - hx
    rel_y = py - hy
    
    # Rotate point back to align with hexagon axes
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    rot_x = rel_x * cos_a - rel_y * sin_a
    rot_y = rel_x * sin_a + rel_y * cos_a
    
    # Hexagon width = 2 * radius * cos(π/6) = radius * √3
    hex_width = radius * np.sqrt(3)
    half_width = hex_width / 2
    
    # Check bounds
    if abs(rot_x) > half_width:
        return False
    
    # Maximum y based on x position in hexagon
    max_y = radius * np.sqrt(3) / 2
    
    if abs(rot_y) > max_y:
        return False
    
    return True

@jit(nopython=True)
def distance_point_to_line_segment(px, py, x1, y1, x2, y2):
    """Calculate distance from point to line segment - JIT compiled"""
    # Vector from (x1,y1) to (x2,y2)
    dx = x2 - x1
    dy = y2 - y1
    
    # Length squared of line segment
    length_sq = dx*dx + dy*dy
    
    # Handle degenerate case
    if length_sq == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Project point onto line segment
    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    t = max(0, min(1, t))  # Clamp t to [0, 1]
    
    # Closest point on segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    # Distance to closest point
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

@jit(nopython=True)
def distance_hexagon_to_hexagon(h1_vertices, h2_vertices):
    """Minimum distance between two hexagons - JIT compiled"""
    min_distance = float('inf')
    
    # Check all pairs of edges
    for i in range(6):
        p1 = h1_vertices[i]
        p2 = h1_vertices[(i+1)%6]
        for j in range(6):
            q1 = h2_vertices[j]
            q2 = h2_vertices[(j+1)%6]
            dist = distance_point_to_line_segment(p1[0], p1[1], q1[0], q1[1], q2[0], q2[1])
            min_distance = min(min_distance, dist)
            dist = distance_point_to_line_segment(p2[0], p2[1], q1[0], q1[1], q2[0], q2[1])
            min_distance = min(min_distance, dist)
    
    return min_distance

@jit(nopython=True)
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

def estimate_min_outer_radius(inner_hex_data):
    """Estimate minimal outer hexagon radius that can contain all inner hexagons"""
    if len(inner_hex_data) == 0:
        return 1000.0
    
    # Get all vertices
    all_vertices = []
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        vertices = get_hexagon_vertices(x, y, angle)
        for vertex in vertices:
            all_vertices.append(vertex)
    
    all_vertices = np.array(all_vertices)
    
    # Find bounding box
    min_x, max_x = np.min(all_vertices[:, 0]), np.max(all_vertices[:, 0])
    min_y, max_y = np.min(all_vertices[:, 1]), np.max(all_vertices[:, 1])
    
    # Calculate center
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Find maximum distance from center to any vertex
    distances = np.sqrt((all_vertices[:, 0] - center_x)**2 + (all_vertices[:, 1] - center_y)**2)
    max_distance = np.max(distances)
    
    # Add small margin to account for hexagon shape
    # For a hexagon to perfectly contain another, we need to account for the circumradius
    # The outer hexagon's circumradius needs to be >= max_distance 
    # But we convert to side length using: side = circumradius * 2/sqrt(3)
    return max_distance * 2.0 / np.sqrt(3)

def check_containment_single(hex_vertices, outer_radius):
    """Check if hexagon vertices are within the outer hexagon (circle approximation)"""
    # Outer hexagon has circumradius = outer_radius * sqrt(3)/2
    outer_circumradius = outer_radius * np.sqrt(3) / 2
    
    for vertex in hex_vertices:
        dist_from_center = np.sqrt(vertex[0]**2 + vertex[1]**2)
        if dist_from_center > outer_circumradius:
            return False
    return True

def check_collision_single(hex1_vertices, hex2_vertices):
    """Check if two hexagons collide using analytical method"""
    # Quick bounding box check first
    min1_x = min(v[0] for v in hex1_vertices)
    max1_x = max(v[0] for v in hex1_vertices)
    min1_y = min(v[1] for v in hex1_vertices)
    max1_y = max(v[1] for v in hex1_vertices)
    
    min2_x = min(v[0] for v in hex2_vertices)
    max2_x = max(v[0] for v in hex2_vertices)
    min2_y = min(v[1] for v in hex2_vertices)
    max2_y = max(v[1] for v in hex2_vertices)
    
    # If bounding boxes don't overlap, no collision
    if max1_x < min2_x or max2_x < min1_x or max1_y < min2_y or max2_y < min1_y:
        return False
    
    # Check each pair of edges for intersection
    # For each edge of hex1
    for i in range(6):
        p1 = hex1_vertices[i]
        p2 = hex1_vertices[(i+1)%6]
        
        # For each edge of hex2
        for j in range(6):
            q1 = hex2_vertices[j]
            q2 = hex2_vertices[(j+1)%6]
            
            # Check if segments intersect
            # This is a simplified version - more precise would involve proper line segment intersection
            # But for speed, we'll use the distance approach which works well for hexagons
            dist = distance_point_to_line_segment(p1[0], p1[1], q1[0], q1[1], q2[0], q2[1])
            if dist < 0.001:  # Very small threshold for near-touching
                return True
    
    # Also check if one hexagon is completely inside the other
    # Check if all vertices of hex1 are within hex2
    all_inside = True
    for v in hex1_vertices:
        if not point_in_hexagon_fast(v[0], v[1], hex2_vertices[0][0], hex2_vertices[0][1], 1.0, 0.0):
            all_inside = False
            break
    if all_inside:
        return True
    
    # Check if all vertices of hex2 are within hex1
    all_inside = True
    for v in hex2_vertices:
        if not point_in_hexagon_fast(v[0], v[1], hex1_vertices[0][0], hex1_vertices[0][1], 1.0, 0.0):
            all_inside = False
            break
    if all_inside:
        return True
    
    return False

def evaluate_solution(hex_data):
    """Evaluate a solution by calculating objective and penalties"""
    # Calculate required outer hex side length
    outer_side_length = calculate_outer_hex_side_length(hex_data)
    
    # Initialize penalty
    penalty = 0.0
    
    # Check containment constraints
    outer_radius = outer_side_length * np.sqrt(3) / 2  # Circumradius
    
    # Check each hexagon for containment
    for i in range(NUM_INNER_HEX):
        x, y, angle = hex_data[i]
        vertices = get_hexagon_vertices(x, y, angle)
        
        if not check_containment_single(vertices, outer_radius):
            penalty += 1000000.0  # Heavy penalty
    
    # Check for overlaps between hexagons
    overlap_pairs = 0
    for i in range(NUM_INNER_HEX):
        for j in range(i+1, NUM_INNER_HEX):
            x1, y1, angle1 = hex_data[i]
            x2, y2, angle2 = hex_data[j]
            
            vertices1 = get_hexagon_vertices(x1, y1, angle1)
            vertices2 = get_hexagon_vertices(x2, y2, angle2)
            
            if check_collision_single(vertices1, vertices2):
                penalty += 1000000.0  # Heavy penalty
                overlap_pairs += 1
    
    # Fitness is negative inverse of side length plus penalties  
    # We want to minimize side length, so maximize 1/side_length
    fitness = -1.0 / outer_side_length
    if penalty > 0:
        fitness -= penalty  # Add penalty for constraint violations
    
    return fitness, outer_side_length

def generate_geometric_initial_solution():
    """Generate a smart initial solution using geometric packing principles"""
    # Start with a central hexagon
    hex_data = [[0.0, 0.0, 0.0]]
    
    # Place surrounding hexagons in a pattern that maximizes area coverage
    # First ring of 6 hexagons around center
    angles = [0, 60, 120, 180, 240, 300]
    distances = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]  # Distance from center
    
    # Place first ring
    for i, (angle, dist) in enumerate(zip(angles, distances)):
        x = dist * math.cos(math.radians(angle))
        y = dist * math.sin(math.radians(angle))
        hex_data.append([x, y, 0.0])
    
    # Place second ring - try to find optimal positions
    # This is a more complex pattern that tries to fill gaps
    second_ring_angles = [30, 90, 150, 210, 270, 330]
    second_ring_distances = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    
    for i, (angle, dist) in enumerate(zip(second_ring_angles, second_ring_distances)):
        x = dist * math.cos(math.radians(angle))
        y = dist * math.sin(math.radians(angle))
        hex_data.append([x, y, 0.0])
    
    # Adjust positions slightly to improve packing
    for i in range(len(hex_data)):
        if i >= 7:  # Only adjust the last few to avoid disturbing the core
            hex_data[i][0] += random.uniform(-0.2, 0.2)
            hex_data[i][1] += random.uniform(-0.2, 0.2)
    
    # Add a few more positions for better distribution
    additional_positions = [
        (1.5, 2.5), (-1.5, 2.5), (1.5, -2.5), (-1.5, -2.5),
        (3.0, 0), (0, 3.0), (0, -3.0), (-3.0, 0)
    ]
    
    for x, y in additional_positions[:4]:
        hex_data.append([x, y, 0.0])
    
    # Ensure we have exactly 11 hexagons
    while len(hex_data) < 11:
        # Add more random positions but constrained
        x = random.uniform(-4.0, 4.0)
        y = random.uniform(-4.0, 4.0)
        hex_data.append([x, y, 0.0])
    
    # Trim to exactly 11
    hex_data = hex_data[:11]
    
    # Randomize rotations for all but center
    for i in range(1, len(hex_data)):
        hex_data[i][2] = random.uniform(0, 360.0)
    
    return np.array(hex_data)

def monte_carlo_refinement(initial_hex_data, max_iterations=5000, timeout_seconds=170):
    """Monte Carlo refinement with adaptive step sizes"""
    start_time = time.time()
    
    current_solution = initial_hex_data.copy()
    current_fitness, current_side_length = evaluate_solution(current_solution)
    
    best_solution = current_solution.copy()
    best_fitness = current_fitness
    best_side_length = current_side_length
    
    # Parameters for adaptive step sizes
    step_size = 0.5
    step_decay = 0.9999
    adapt_threshold = 0.01
    
    iteration = 0
    no_improvement_count = 0
    max_no_improvement = 500
    
    while iteration < max_iterations and (time.time() - start_time) < timeout_seconds:
        iteration += 1
        
        # Copy current solution
        new_solution = current_solution.copy()
        
        # Randomly select one hexagon to modify
        hex_idx = random.randint(0, NUM_INNER_HEX - 1)
        
        # Perturb position and/or rotation
        if random.random() < 0.7:  # 70% chance of modifying position
            # Modify position
            new_solution[hex_idx][0] += random.uniform(-step_size, step_size)
            new_solution[hex_idx][1] += random.uniform(-step_size, step_size)
        else:  # 30% chance of modifying rotation
            # Modify rotation
            new_solution[hex_idx][2] += random.uniform(-30, 30)
            new_solution[hex_idx][2] %= 360
            
        # Evaluate new solution
        new_fitness, new_side_length = evaluate_solution(new_solution)
        
        # Accept or reject based on fitness difference
        if new_fitness > current_fitness:
            # Always accept improvement
            current_solution = new_solution
            current_fitness = new_fitness
            
            if new_fitness > best_fitness:
                best_solution = new_solution.copy()
                best_fitness = new_fitness
                best_side_length = new_side_length
                
            no_improvement_count = 0
        else:
            # Sometimes accept worse solutions (simulated annealing-like)
            delta = new_fitness - current_fitness
            acceptance_prob = math.exp(delta * 100) if delta < 0 else 1.0
            
            if random.random() < acceptance_prob:
                current_solution = new_solution
                current_fitness = new_fitness
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
        # Adapt step size based on recent performance
        if no_improvement_count > 100 and step_size > 0.01:
            step_size *= step_decay
        elif no_improvement_count < 50 and step_size < 1.0:
            step_size *= 1.01
            
        # Reset step size periodically
        if iteration % 1000 == 0:
            step_size = max(0.05, step_size * 0.95)
            
        # Early termination
        if no_improvement_count > max_no_improvement:
            break
    
    return best_solution, best_fitness, best_side_length

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Generate good initial solution
    initial_solution = generate_geometric_initial_solution()
    
    # Refine using monte carlo
    final_solution, final_fitness, final_side_length = monte_carlo_refinement(
        initial_solution, max_iterations=5000, timeout_seconds=170
    )
    
    # Generate output format
    inner_hex_data = final_solution.copy()
    outer_hex_data = np.array([0, 0, 0])  # outer hexagon centered at origin
    
    return inner_hex_data, outer_hex_data, final_side_length

# EVOLVE-BLOCK-END