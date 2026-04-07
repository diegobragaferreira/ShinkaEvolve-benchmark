# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.spatial.distance import cdist
import time

# Constants
NUM_INNER_HEXAGONS = 11
UNIT_HEXAGON_RADIUS = 1.0
MAX_EVAL_TIME = 180.0

# Precomputed unit hexagon vertices (centered at origin)
def get_unit_hexagon_vertices():
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 angles + close the loop
    vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    return vertices

UNIT_HEXAGON_VERTICES = get_unit_hexagon_vertices()

def rotate_point(point, angle_rad):
    """Rotate a point around origin"""
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([point[0]*cos_a - point[1]*sin_a, point[0]*sin_a + point[1]*cos_a])

def hexagon_vertices(center, angle_rad, scale=1.0):
    """Get vertices of a hexagon at given position and rotation"""
    rotated_vertices = np.array([rotate_point(v, angle_rad) for v in UNIT_HEXAGON_VERTICES])
    return rotated_vertices * scale + np.array(center)

def calculate_outer_hexagon_radius(inner_hex_data, outer_center=[0,0], outer_angle=0):
    """Calculate minimum radius needed for outer hexagon to contain all inner hexagons"""
    max_dist = 0
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = np.radians(inner_hex_data[i][2])

        # Get all vertices of this hexagon
        vertices = hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)

        # Calculate max distance from outer center to any vertex
        for vertex in vertices:
            dist = np.linalg.norm(np.array(vertex) - np.array(outer_center))
            max_dist = max(max_dist, dist)

    return max_dist

def hexagon_collision(hex1_vertices, hex2_vertices):
    """Check if two hexagons collide using Separating Axis Theorem"""
    # Quick bounding box check first
    min1_x = min(v[0] for v in hex1_vertices)
    max1_x = max(v[0] for v in hex1_vertices)
    min1_y = min(v[1] for v in hex1_vertices)
    max1_y = max(v[1] for v in hex1_vertices)

    min2_x = min(v[0] for v in hex2_vertices)
    max2_x = max(v[0] for v in hex2_vertices)
    min2_y = min(v[1] for v in hex2_vertices)
    max2_y = max(v[1] for v in hex2_vertices)

    # If bounding boxes don't overlap, no collision possible
    if max1_x < min2_x or max2_x < min1_x or max1_y < min2_y or max2_y < min1_y:
        return False

    # Get all edges of both hexagons
    edges1 = []
    edges2 = []

    for i in range(6):
        p1 = hex1_vertices[i]
        p2 = hex1_vertices[(i+1)%6]
        edge = (p2[0]-p1[0], p2[1]-p1[1])
        edges1.append(edge)

        p1 = hex2_vertices[i]
        p2 = hex2_vertices[(i+1)%6]
        edge = (p2[0]-p1[0], p2[1]-p1[1])
        edges2.append(edge)

    # Combine all potential separating axes
    all_axes = edges1 + edges2

    # Normalize axes
    for i, axis in enumerate(all_axes):
        length = math.sqrt(axis[0]**2 + axis[1]**2)
        if length > 0:
            all_axes[i] = (axis[0]/length, axis[1]/length)

    # Check projection overlap on each axis
    for axis in all_axes:
        # Project both hexagons onto this axis
        proj1 = []
        proj2 = []

        for v in hex1_vertices:
            dot = v[0]*axis[0] + v[1]*axis[1]
            proj1.append(dot)

        for v in hex2_vertices:
            dot = v[0]*axis[0] + v[1]*axis[1]
            proj2.append(dot)

        min1, max1 = min(proj1), max(proj1)
        min2, max2 = min(proj2), max(proj2)

        # If projections don't overlap, then there's separation
        if max1 < min2 or max2 < min1:
            return False

    return True

def check_containment(hex_vertices, outer_center=[0,0], outer_radius=10.0):
    """Check if hexagon vertices are contained within outer hexagon"""
    for vertex in hex_vertices:
        x, y = vertex
        dx = x - outer_center[0]
        dy = y - outer_center[1]
        distance = math.sqrt(dx*dx + dy*dy)
        if distance >= outer_radius:
            return False
    return True

def calculate_placement_score(hex_data, available_positions):
    """Calculate a heuristic score for each available position"""
    if not available_positions:
        return []
    
    scores = []
    for pos in available_positions:
        x, y = pos
        # Score based on how close to center (better score for central positions)
        center_distance = math.sqrt(x*x + y*y)
        # Score based on proximity to existing hexagons (penalize overlapping)
        proximity_penalty = 0
        for hex_pos in hex_data:
            dist = math.sqrt((x-hex_pos[0])**2 + (y-hex_pos[1])**2)
            if dist < 2.0:  # If too close to existing hexagon
                proximity_penalty += 1000 / (dist + 0.1)  # Higher penalty for closer proximity
        
        # Prefer positions away from corners of the theoretical hexagon
        # This helps with packing efficiency
        angle_to_center = math.atan2(y, x) if x != 0 or y != 0 else 0
        angle_deviation = abs(angle_to_center % (math.pi/3) - math.pi/6)
        angle_score = 100 - angle_deviation * 10  # Better score for less angular deviation
        
        # Overall score: prefer central positions, avoid overlaps, consider angular distribution
        score = 10000 / (center_distance + 1) - proximity_penalty + angle_score
        scores.append(score)
    
    return scores

def greedy_construction():
    """Construct initial solution using greedy approach"""
    # Start with center hexagon
    hex_data = [[0, 0, 0]]
    
    # Define a grid of potential spots to place hexagons
    # We'll try placing in triangular lattice pattern, but with some randomness for variety
    potential_spots = []
    
    # Generate candidate positions in a triangular pattern around center
    base_distance = 2.0  # Distance between centers
    angles = [0, math.pi/3, 2*math.pi/3, math.pi, 4*math.pi/3, 5*math.pi/3]
    
    # Add positions at increasing distances from center
    for layer in range(1, 6):
        for angle in angles:
            x = layer * base_distance * math.cos(angle)
            y = layer * base_distance * math.sin(angle)
            potential_spots.append([x, y])
    
    # Add some additional positions near corners
    corner_positions = [
        [base_distance, 0],
        [-base_distance, 0],
        [0, base_distance],
        [0, -base_distance],
        [base_distance/2, base_distance * math.sqrt(3)/2],
        [base_distance/2, -base_distance * math.sqrt(3)/2],
        [-base_distance/2, base_distance * math.sqrt(3)/2],
        [-base_distance/2, -base_distance * math.sqrt(3)/2],
    ]
    
    potential_spots.extend(corner_positions)
    
    # Shuffle to add some randomness
    np.random.shuffle(potential_spots)
    
    # Greedy insertion
    remaining_spots = potential_spots[:]
    
    # Try to place remaining hexagons greedily
    while len(hex_data) < NUM_INNER_HEXAGONS and remaining_spots:
        # Select best spot among remaining
        best_spot = None
        best_score = -float('inf')
        
        # Test up to 100 spots for best placement
        test_spots = remaining_spots[:min(100, len(remaining_spots))]
        
        for spot in test_spots:
            # Try to place hexagon at this spot
            test_data = hex_data + [[spot[0], spot[1], 0]]
            
            # Verify this doesn't cause overlaps
            valid = True
            if len(test_data) > 1:
                vertices_new = hexagon_vertices(spot, 0, UNIT_HEXAGON_RADIUS)
                for i, hex_pos in enumerate(test_data[:-1]):
                    vertices_existing = hexagon_vertices(hex_pos[:2], 0, UNIT_HEXAGON_RADIUS)
                    if hexagon_collision(vertices_new, vertices_existing):
                        valid = False
                        break
            
            if valid:
                score = calculate_placement_score(test_data, [spot])[0]
                if score > best_score:
                    best_score = score
                    best_spot = spot
        
        if best_spot is not None:
            hex_data.append([best_spot[0], best_spot[1], 0])
            remaining_spots.remove(best_spot)
        else:
            # If we can't find a good spot, just take the first available one
            if remaining_spots:
                spot = remaining_spots.pop(0)
                hex_data.append([spot[0], spot[1], 0])
    
    # Fill with remaining positions if needed (random or at predefined positions)
    while len(hex_data) < NUM_INNER_HEXAGONS:
        hex_data.append([np.random.uniform(-5, 5), np.random.uniform(-5, 5), 0])
    
    return np.array(hex_data)

def local_improvement(hex_data, max_iterations=500):
    """Improve solution using simulated annealing like local search"""
    current_solution = hex_data.copy()
    current_radius = calculate_outer_hexagon_radius(current_solution)
    
    # Simulated annealing parameters
    temperature = 5.0
    cooling_rate = 0.999
    min_temperature = 0.001
    max_no_improvement = 50
    
    best_solution = current_solution.copy()
    best_radius = current_radius
    
    no_improvement_counter = 0
    
    for iteration in range(max_iterations):
        if temperature < min_temperature or no_improvement_counter > max_no_improvement:
            break
            
        # Create neighbor solution
        neighbor_solution = current_solution.copy()
        
        # Choose a random hexagon to modify
        hex_idx = np.random.randint(0, NUM_INNER_HEXAGONS)
        
        # Perturb position slightly
        neighbor_solution[hex_idx][0] += np.random.normal(0, 0.2)
        neighbor_solution[hex_idx][1] += np.random.normal(0, 0.2)
        neighbor_solution[hex_idx][2] += np.random.normal(0, 5.0)
        neighbor_solution[hex_idx][2] %= 360
        
        # Check if neighbor is valid
        valid = True
        vertices_new = hexagon_vertices(neighbor_solution[hex_idx][:2], 
                                       np.radians(neighbor_solution[hex_idx][2]), 
                                       UNIT_HEXAGON_RADIUS)
        
        # Check containment
        outer_radius = current_radius + 1  # Conservative check
        for vertex in vertices_new:
            x, y = vertex
            dx = x - 0
            dy = y - 0
            distance = math.sqrt(dx*dx + dy*dy)
            if distance >= outer_radius:
                valid = False
                break
        
        # Check overlaps with other hexagons
        if valid:
            for i, hex_pos in enumerate(neighbor_solution):
                if i == hex_idx:
                    continue
                vertices_existing = hexagon_vertices(hex_pos[:2], 
                                                   np.radians(hex_pos[2]), 
                                                   UNIT_HEXAGON_RADIUS)
                if hexagon_collision(vertices_new, vertices_existing):
                    valid = False
                    break
        
        if valid:
            new_radius = calculate_outer_hexagon_radius(neighbor_solution)
            if new_radius < current_radius:
                # Accept better solution
                current_solution = neighbor_solution.copy()
                current_radius = new_radius
                if new_radius < best_radius:
                    best_solution = neighbor_solution.copy()
                    best_radius = new_radius
                no_improvement_counter = 0
            else:
                # Accept worse solution with probability based on temperature
                delta = new_radius - current_radius
                acceptance_prob = math.exp(-delta / temperature)
                if np.random.random() < acceptance_prob:
                    current_solution = neighbor_solution.copy()
                    current_radius = new_radius
                else:
                    no_improvement_counter += 1
        else:
            no_improvement_counter += 1
        
        # Cool down
        temperature *= cooling_rate
    
    return best_solution, best_radius

def validate_solution(inner_hex_data, outer_center=[0,0], outer_angle=0):
    """Validate solution: check containment and non-overlap"""
    # Calculate outer radius once
    outer_radius = calculate_outer_hexagon_radius(inner_hex_data, outer_center, outer_angle)
    
    # Check containment and overlaps
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = np.radians(inner_hex_data[i][2])
        vertices = hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)
        
        # Check containment
        if not check_containment(vertices, outer_center, outer_radius):
            return False
            
        # Check overlaps with all others
        for j in range(i+1, len(inner_hex_data)):
            center2 = inner_hex_data[j][:2]
            angle2 = np.radians(inner_hex_data[j][2])
            vertices2 = hexagon_vertices(center2, angle2, UNIT_HEXAGON_RADIUS)
            
            if hexagon_collision(vertices, vertices2):
                return False
    
    return True

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Use GRASP approach: greedy construction + local improvement
    best_config = None
    best_radius = float('inf')
    
    # Run several iterations of greedy construction + local improvement
    num_iterations = 10
    
    for iteration in range(num_iterations):
        if time.time() - start_time > MAX_EVAL_TIME - 2:
            break
            
        # Greedy construction
        construction = greedy_construction()
        
        # Local improvement
        improved_solution, improved_radius = local_improvement(construction, 300)
        
        # Validate the improved solution
        if validate_solution(improved_solution):
            if improved_radius < best_radius:
                best_radius = improved_radius
                best_config = improved_solution.copy()
    
    # If no valid solution was found, fallback to a known good configuration
    if best_config is None:
        best_config = np.array([
            [0, 0, 0],           # center
            [-2.5, 0, 0],        # left
            [2.5, 0, 0],         # right
            [-1.25, 2.17, 0],    # top-left
            [1.25, 2.17, 0],     # top-right
            [-1.25, -2.17, 0],   # bottom-left
            [1.25, -2.17, 0],    # bottom-right
            [-3.75, 2.17, 0],    # far top-left
            [3.75, 2.17, 0],     # far top-right
            [-3.75, -2.17, 0],   # far bottom-left
            [3.75, -2.17, 0],    # far bottom-right
        ])
        best_radius = 8.0
    
    # Return result
    inner_hex_data = best_config
    outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered at origin
    outer_hex_side_length = best_radius
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END