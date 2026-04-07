# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import math
from itertools import combinations

def create_regular_hexagon(center=(0, 0), side_length=1, rotation_deg=0):
    """Create a regular hexagon as a shapely polygon"""
    rotation_rad = math.radians(rotation_deg)
    points = []
    for i in range(6):
        angle = rotation_rad + i * math.pi / 3
        x = center[0] + side_length * math.cos(angle)
        y = center[1] + side_length * math.sin(angle)
        points.append((x, y))
    return Polygon(points)

def calculate_min_outer_hex_side_length(inner_hex_data, outer_center=(0, 0)):
    """Calculate the minimum outer hexagon side length that contains all inner hexagons"""
    max_dist = 0
    for i in range(len(inner_hex_data)):
        pos = (inner_hex_data[i][0], inner_hex_data[i][1])
        rot = inner_hex_data[i][2]
        
        # Create temporary hexagon to get vertices
        temp_hex = create_regular_hexagon(pos, 1, rot)
        vertices = list(temp_hex.exterior.coords)[:-1]  # Exclude duplicate last point
        
        # Find max distance from center to any vertex
        for vertex in vertices:
            dist = math.sqrt((vertex[0] - outer_center[0])**2 + (vertex[1] - outer_center[1])**2)
            max_dist = max(max_dist, dist)
    
    # Add some margin to ensure containment
    return max_dist * 1.05

def check_containment(hexagon, outer_hexagon):
    """Check if hexagon is fully contained within outer_hexagon"""
    return outer_hexagon.contains(hexagon)

def check_overlap(hex1, hex2):
    """Check if two hexagons overlap"""
    return hex1.intersects(hex2)

def generate_voronoi_based_initialization():
    """Generate initial hexagon arrangement using Voronoi-inspired approach"""
    # Start with a central hexagon
    positions = [[0, 0]]
    rotations = [0]
    
    # Generate positions in a hexagonal pattern around the center
    # Using the fact that hexagons can be arranged in a honeycomb pattern
    # We'll use a systematic approach to place them
    hex_pattern = [
        (0, 2),           # top
        (1.732, 1),       # top-right
        (1.732, -1),      # bottom-right
        (0, -2),          # bottom
        (-1.732, -1),     # bottom-left
        (-1.732, 1),      # top-left
        (3.464, 2),       # far top-right
        (-3.464, 2),      # far top-left
        (3.464, -2),      # far bottom-right
        (-3.464, -2),     # far bottom-left
    ]
    
    # Add the pattern points
    for i, (x, y) in enumerate(hex_pattern):
        if i < 10:  # We want exactly 11 hexagons total
            positions.append([x, y])
            rotations.append(0)
    
    return positions, rotations

def compute_hexagon_distance(hex1, hex2):
    """Compute minimum distance between two hexagons"""
    # Get vertices of both hexagons
    h1_vertices = list(hex1.exterior.coords)[:-1]
    h2_vertices = list(hex2.exterior.coords)[:-1]
    
    # Compute minimum distance between any pair of vertices
    min_dist = float('inf')
    for v1 in h1_vertices:
        for v2 in h2_vertices:
            dist = distance.euclidean(v1, v2)
            min_dist = min(min_dist, dist)
    
    return min_dist

def construct_optimal_hexagon_packaging():
    """Construct the optimal hexagon packing using geometric optimization"""
    # Generate initial Voronoi-based configuration
    positions, rotations = generate_voronoi_based_initialization()
    
    # Convert to array format
    inner_hex_data = np.array(list(zip(positions, rotations)))
    
    # Refine using iterative improvement
    best_positions = positions.copy()
    best_rotations = rotations.copy()
    best_side_length = float('inf')
    
    # Try different configurations with rotation adjustments
    for iter_count in range(100):  # Limit iterations to prevent infinite loop
        # Apply small random adjustments to positions
        new_positions = []
        new_rotations = []
        
        for i, (pos, rot) in enumerate(zip(best_positions, best_rotations)):
            if i == 0:  # Keep central hexagon at center
                new_positions.append(pos)
                new_rotations.append(rot)
            else:
                # Slight perturbations
                new_pos = [
                    pos[0] + np.random.normal(0, 0.1),
                    pos[1] + np.random.normal(0, 0.1)
                ]
                new_rot = rot + np.random.normal(0, 5)  # Small rotation change
                new_positions.append(new_pos)
                new_rotations.append(new_rot % 360)
        
        # Evaluate this configuration
        temp_data = np.array(list(zip(new_positions, new_rotations)))
        temp_positions = [list(p) for p in temp_data[:, 0]]
        temp_rotations = temp_data[:, 1].tolist()
        
        # Create hexagons and check constraints
        hexagons = []
        for pos, rot in zip(temp_positions, temp_rotations):
            hexagon = create_regular_hexagon(pos, 1, rot)
            hexagons.append(hexagon)
        
        # Check overlaps
        valid = True
        for i, j in combinations(range(len(hexagons)), 2):
            if check_overlap(hexagons[i], hexagons[j]):
                valid = False
                break
        
        if valid:
            # Calculate outer hexagon side length
            test_data = np.zeros((len(temp_positions), 3))
            for i, (pos, rot) in enumerate(zip(temp_positions, temp_rotations)):
                test_data[i] = [pos[0], pos[1], rot]
            
            side_length = calculate_min_outer_hex_side_length(test_data)
            
            if side_length < best_side_length:
                best_positions = new_positions
                best_rotations = new_rotations
                best_side_length = side_length
    
    # Final validation and return
    final_data = np.zeros((len(best_positions), 3))
    for i, (pos, rot) in enumerate(zip(best_positions, best_rotations)):
        final_data[i] = [pos[0], pos[1], rot]
    
    return final_data

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a Voronoi-based deterministic geometric optimization approach.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Generate and optimize hexagon packing using Voronoi-inspired approach
    inner_hex_data = construct_optimal_hexagon_packaging()
    
    # Calculate final outer hexagon side length
    outer_hex_side_length = calculate_min_outer_hex_side_length(inner_hex_data)
    
    # Ensure the outer hexagon contains all inner hexagons
    outer_hexagon = create_regular_hexagon((0, 0), outer_hex_side_length, 0)
    
    # Perform final validation
    valid = True
    for i in range(len(inner_hex_data)):
        pos = (inner_hex_data[i][0], inner_hex_data[i][1])
        rot = inner_hex_data[i][2]
        hexagon = create_regular_hexagon(pos, 1, rot)
        if not check_containment(hexagon, outer_hexagon):
            valid = False
            break
    
    # Check for overlaps
    if valid:
        hexagons = []
        for i in range(len(inner_hex_data)):
            pos = (inner_hex_data[i][0], inner_hex_data[i][1])
            rot = inner_hex_data[i][2]
            hexagon = create_regular_hexagon(pos, 1, rot)
            hexagons.append(hexagon)
        
        for i, j in combinations(range(len(hexagons)), 2):
            if check_overlap(hexagons[i], hexagons[j]):
                valid = False
                break
    
    # If invalid, fallback to known good configuration
    if not valid:
        # Simple good arrangement that works well
        inner_hex_data = np.array([
            [0, 0, 0],        # center
            [-2.5, 0, 0],     # left
            [2.5, 0, 0],      # right
            [-1.25, 2.17, 0], # top-left
            [1.25, 2.17, 0],  # top-right
            [-1.25, -2.17, 0], # bottom-left
            [1.25, -2.17, 0], # bottom-right
            [-3.75, 2.17, 0], # far top-left
            [3.75, 2.17, 0],  # far top-right
            [-3.75, -2.17, 0], # far bottom-left
            [3.75, -2.17, 0], # far bottom-right
        ])
        outer_hex_side_length = 8
    
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END