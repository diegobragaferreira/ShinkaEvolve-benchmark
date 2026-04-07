# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
import itertools
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
    return outer_polygon.contains(hex_polygon)

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
    
    # Check for overlaps between hexagons
    for i in range(len(inner_hex_data)):
        for j in range(i+1, len(inner_hex_data)):
            center_x1, center_y1, angle1 = inner_hex_data[i]
            center_x2, center_y2, angle2 = inner_hex_data[j]
            
            vertices1 = get_hexagon_vertices(center_x1, center_y1, 1, angle1)
            vertices2 = get_hexagon_vertices(center_x2, center_y2, 1, angle2)
            
            poly1 = Polygon(vertices1)
            poly2 = Polygon(vertices2)
            
            if poly1.intersects(poly2):
                return False, outer_hex_side_length
    
    return True, outer_hex_side_length

def evaluate_fitness(individual, outer_hex_center=(0, 0)):
    """Evaluate fitness of an individual - higher is better"""
    inner_hex_data = individual.reshape(-1, 3)  # Each row is (x, y, angle)
    
    # Calculate outer hexagon size needed
    outer_hex_side_length = calculate_outer_hexagon_radius(inner_hex_data, outer_hex_center)
    
    # Check validity
    valid, adjusted_radius = is_valid_arrangement(inner_hex_data, outer_hex_center, outer_hex_side_length)
    
    if not valid:
        # Penalize invalid arrangements heavily
        return -1000000
    
    # The fitness is the inverse of the outer hexagon side length (we want to minimize it)
    return 1.0 / adjusted_radius

def create_grid_search_candidates():
    """Create a set of promising candidate arrangements using grid search"""
    candidates = []
    
    # Define search spaces for positions and rotations
    # Central region for main hexagons
    positions = [
        (0, 0),      # center
        (-2.5, 0),   # left
        (2.5, 0),    # right
        (-1.25, 2.17),  # top-left
        (1.25, 2.17),   # top-right
        (-1.25, -2.17), # bottom-left
        (1.25, -2.17),  # bottom-right
    ]
    
    # Additional positions for extended coverage
    additional_positions = [
        (-3.75, 2.17),  # far top-left
        (3.75, 2.17),   # far top-right
        (-3.75, -2.17), # far bottom-left
        (3.75, -2.17),  # far bottom-right
    ]
    
    # Define rotation ranges (in degrees)
    angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    
    # Create combinations around central positions
    for pos_combo in itertools.combinations(positions, 7):
        # Fill remaining positions with additional ones
        remaining_positions = [p for p in additional_positions if p not in pos_combo]
        
        # Use all 11 positions
        final_positions = list(pos_combo) + remaining_positions[:4]
        
        # Generate all combinations of rotations for all 11 positions
        for angle_combo in itertools.product(angles, repeat=11):
            individual = []
            for i, (pos, angle) in enumerate(zip(final_positions, angle_combo)):
                individual.append([pos[0], pos[1], angle])
            
            candidates.append(np.array(individual).flatten())
    
    return candidates

def generate_refined_arrangements():
    """Generate refined arrangements by slightly perturbing good candidates"""
    # Start with the basic arrangement from the original implementation
    base_arrangement = np.array([
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
    ])
    
    candidates = [base_arrangement.flatten()]
    
    # Perturb each position slightly
    for i in range(11):
        for dx in [-0.2, -0.1, 0, 0.1, 0.2]:
            for dy in [-0.2, -0.1, 0, 0.1, 0.2]:
                if (dx == 0 and dy == 0): continue
                perturbed = base_arrangement.copy()
                perturbed[i, 0] += dx
                perturbed[i, 1] += dy
                candidates.append(perturbed.flatten())
                
    return candidates

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    start_time = time.time()
    
    # Generate candidates using multiple strategies
    candidates = []
    
    # Strategy 1: Basic refined arrangements
    candidates.extend(generate_refined_arrangements())
    
    # Strategy 2: Some structured combinations
    # Try a few specific arrangements that might be promising
    specific_arrangements = []
    
    # Hexagonal close packing pattern variation
    hex_pattern = [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0.5, 0.866, 0],
        [-0.5, 0.866, 0],
        [0.5, -0.866, 0],
        [-0.5, -0.866, 0],
        [1.5, 1.732, 0],
        [-1.5, 1.732, 0],
        [1.5, -1.732, 0],
        [-1.5, -1.732, 0]
    ]
    
    # Add variations of this pattern
    for i in range(11):
        for angle in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
            variation = [list(hex_pattern[j]) for j in range(11)]
            variation[i][2] = angle
            specific_arrangements.append(np.array(variation).flatten())
    
    candidates.extend(specific_arrangements)
    
    # Evaluate all candidates and find the best one
    best_fitness = -float('inf')
    best_individual = None
    
    for individual in candidates:
        fitness = evaluate_fitness(individual)
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_individual = individual.copy()
            
        # Early stopping if we've found something very good
        if best_fitness > 0.2544:  # SOTA benchmark
            break
            
        # Time limit check
        if time.time() - start_time > 170:
            break
    
    # If no valid candidates were found, fall back to the original
    if best_individual is None:
        best_individual = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0]
        ]).flatten()
    
    # Final evaluation
    final_fitness = evaluate_fitness(best_individual)
    inner_hex_data = best_individual.reshape(-1, 3)
    
    # Determine the actual outer hexagon size needed
    outer_hex_side_length = 1.0 / final_fitness if final_fitness > 0 else 1000.0
    
    # Outer hexagon data - centered at origin with no rotation (for consistency)
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
