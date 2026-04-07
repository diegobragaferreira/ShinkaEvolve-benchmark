# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import time
from numba import jit
import warnings
import math
from collections import defaultdict

warnings.filterwarnings('ignore')

@jit(nopython=True)
def hexagon_vertices(x, y, angle_deg, side_length=1):
    """Calculate vertices of a hexagon given center, angle, and side length"""
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vx = x + side_length * np.cos(theta)
        vy = y + side_length * np.sin(theta)
        vertices.append((vx, vy))
    return np.array(vertices)

@jit(nopython=True)
def distance_point_to_line(point, line_start, line_end):
    """Calculate distance from point to line segment"""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Vector from line_start to line_end
    dx, dy = x2 - x1, y2 - y1
    # Vector from line_start to point
    px_minus_x1, py_minus_y1 = px - x1, py - y1
    
    # Length squared of line segment
    length_sq = dx*dx + dy*dy
    
    if length_sq == 0:
        # Line segment is actually a point
        return np.sqrt(px_minus_x1*px_minus_x1 + py_minus_y1*py_minus_y1)
    
    # Project point onto line
    t = (px_minus_x1*dx + py_minus_y1*dy) / length_sq
    
    # Clamp t to [0, 1] to stay on line segment
    t = max(0, min(1, t))
    
    # Closest point on line segment
    closest_x = x1 + t*dx
    closest_y = y1 + t*dy
    
    # Distance to closest point
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

@jit(nopython=True)
def point_in_hexagon(point, hex_vertices):
    """Check if a point is inside a hexagon using ray casting"""
    x, y = point
    n = len(hex_vertices)
    inside = False
    
    p1x, p1y = hex_vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = hex_vertices[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def get_hexagon_polygon(x, y, angle_deg, side_length=1):
    """Get shapely polygon representation of hexagon"""
    vertices = hexagon_vertices(x, y, angle_deg, side_length)
    return Polygon(vertices)

def calculate_outer_hexagon_radius(inner_positions, inner_angles):
    """Calculate minimum radius needed to contain all inner hexagons"""
    max_dist = 0
    outer_center = (0, 0)
    
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_positions)):
        pos = inner_positions[i]
        angle = inner_angles[i]
        hex_vertices = hexagon_vertices(pos[0], pos[1], angle)
        all_vertices.extend(hex_vertices)
    
    # Find maximum distance from center
    for vertex in all_vertices:
        dist = np.sqrt((vertex[0] - outer_center[0])**2 + (vertex[1] - outer_center[1])**2)
        max_dist = max(max_dist, dist)
    
    # Add buffer for safety and account for hexagon shape
    return max_dist * 1.1  # Safety factor

def compute_outer_hexagon_bounds(inner_hex_data):
    """Compute outer hexagon that bounds all inner hexagons"""
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i][0], inner_hex_data[i][1])
        rotation = inner_hex_data[i][2]
        vertices = hexagon_vertices(center[0], center[1], rotation)
        all_vertices.extend(vertices)
    
    if not all_vertices:
        return [(0, 0)] * 6
    
    # Find bounding box
    min_x = min(v[0] for v in all_vertices)
    max_x = max(v[0] for v in all_vertices)
    min_y = min(v[1] for v in all_vertices)
    max_y = max(v[1] for v in all_vertices)
    
    # Compute approximate hexagon bounds
    avg_x = (min_x + max_x) / 2
    avg_y = (min_y + max_y) / 2
    width = max_x - min_x
    height = max_y - min_y
    
    # Approximate side length based on dimensions
    side_length = max(width, height) / math.sqrt(3) * 2
    
    # Generate final hexagon vertices
    outer_vertices = []
    for i in range(6):
        theta = i * math.pi / 3
        x = avg_x + side_length * math.cos(theta)
        y = avg_y + side_length * math.sin(theta)
        outer_vertices.append((x, y))
    
    return outer_vertices

def check_containment(hex_vertices, outer_vertices):
    """Check if all vertices of inner hexagon are inside outer hexagon"""
    outer_polygon = Polygon(outer_vertices)
    for vertex in hex_vertices:
        point = Point(vertex)
        if not outer_polygon.contains(point):
            return False
    return True

def hexagon_intersects(hex1_vertices, hex2_vertices):
    """Check if two hexagons intersect using Shapely"""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def evaluate_solution_fast(hex_data):
    """Fast evaluation of solution using geometric constraints"""
    # Extract parameters
    positions = hex_data[:, :2]
    rotations = hex_data[:, 2]
    
    # Check if all hexagons fit within reasonable bounds
    max_radius = np.max(np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)) + 2.0
    if max_radius > 20:
        return 1e10
    
    # Precompute all hexagon vertices once
    all_inner_vertices = []
    for i in range(len(hex_data)):
        center = tuple(positions[i])
        rotation = rotations[i]
        vertices = hexagon_vertices(center[0], center[1], rotation)
        all_inner_vertices.append(vertices)
    
    # Compute outer hexagon bounds
    outer_vertices = compute_outer_hexagon_bounds(hex_data)
    outer_polygon = Polygon(outer_vertices)
    
    # Check containment
    for vertices in all_inner_vertices:
        if not check_containment(vertices, outer_vertices):
            return 1e10
    
    # Check overlaps using spatial indexing
    # Simple approach: check all pairs
    for i in range(len(all_inner_vertices)):
        for j in range(i+1, len(all_inner_vertices)):
            if hexagon_intersects(all_inner_vertices[i], all_inner_vertices[j]):
                return 1e10
    
    # Calculate fitness as 1/outer_radius
    outer_radius = calculate_outer_hexagon_radius(positions, rotations)
    return 1.0 / outer_radius

def generate_geometric_pattern(num_patterns=3):
    """Generate geometrically sensible initial patterns"""
    patterns = []
    
    # Pattern 1: Hexagonal lattice arrangement
    pattern1_positions = []
    pattern1_angles = []
    
    # Center hexagon
    pattern1_positions.append([0.0, 0.0])
    pattern1_angles.append(0.0)
    
    # Ring of 6 surrounding hexagons
    for i in range(6):
        angle = i * 60
        radius = 2.0
        x = radius * math.cos(math.radians(angle))
        y = radius * math.sin(math.radians(angle))
        pattern1_positions.append([x, y])
        pattern1_angles.append(0.0)
    
    # Additional positions to make 11 hexagons
    pattern1_positions.extend([
        [-3.0, 1.0], [3.0, 1.0],
        [-3.0, -1.0], [3.0, -1.0],
        [0.0, 3.0], [0.0, -3.0],
        [1.5, 2.6], [-1.5, -2.6],
        [-1.5, 2.6], [1.5, -2.6]
    ])
    
    pattern1_angles.extend([0.0] * 10)
    
    # Trim to exactly 11
    pattern1_positions = pattern1_positions[:11]
    pattern1_angles = pattern1_angles[:11]
    
    # Pattern 2: Spiral arrangement
    pattern2_positions = []
    pattern2_angles = []
    
    # Start with center
    pattern2_positions.append([0.0, 0.0])
    pattern2_angles.append(0.0)
    
    # Spiral outward
    for i in range(1, 11):
        angle = i * 30  # Gradually increasing angle
        radius = i * 1.2
        x = radius * math.cos(math.radians(angle))
        y = radius * math.sin(math.radians(angle))
        pattern2_positions.append([x, y])
        pattern2_angles.append(0.0)
    
    # Pattern 3: Clustered arrangement
    pattern3_positions = []
    pattern3_angles = []
    
    # Central cluster of 4 hexagons
    for i in range(4):
        angle = i * 90
        radius = 1.5
        x = radius * math.cos(math.radians(angle))
        y = radius * math.sin(math.radians(angle))
        pattern3_positions.append([x, y])
        pattern3_angles.append(0.0)
    
    # Surrounding ring of 6
    for i in range(6):
        angle = i * 60
        radius = 3.5
        x = radius * math.cos(math.radians(angle))
        y = radius * math.sin(math.radians(angle))
        pattern3_positions.append([x, y])
        pattern3_angles.append(0.0)
    
    # Trim to exactly 11
    pattern3_positions = pattern3_positions[:11]
    pattern3_angles = pattern3_angles[:11]
    
    patterns = [
        np.column_stack([pattern1_positions, pattern1_angles]),
        np.column_stack([pattern2_positions, pattern2_angles]),
        np.column_stack([pattern3_positions, pattern3_angles])
    ]
    
    return patterns

def constrained_local_search(hex_data, max_iterations=100):
    """Perform local search with geometric constraints"""
    best_hex_data = hex_data.copy()
    best_fitness = evaluate_solution_fast(best_hex_data)
    
    # Define search bounds
    bounds = [
        (-15, 15),  # x positions
        (-15, 15),  # y positions
        (0, 360),   # angles
    ]
    
    # Track improvements
    last_improvement = 0
    patience = 20
    
    for iteration in range(max_iterations):
        improvement_made = False
        current_improvement = 0
        
        # Try systematic perturbations
        for i in range(11):  # Skip center for now (more stable)
            # Try small position adjustments
            for dim in range(2):  # x and y
                old_pos = best_hex_data[i, dim]
                for delta in [-0.1, 0.1]:  # Small steps
                    best_hex_data[i, dim] = old_pos + delta
                    new_fitness = evaluate_solution_fast(best_hex_data)
                    if new_fitness > best_fitness:  # Maximizing fitness
                        best_fitness = new_fitness
                        improvement_made = True
                        current_improvement += 1
                    else:
                        best_hex_data[i, dim] = old_pos
            
            # Try small angle adjustments
            old_angle = best_hex_data[i, 2]
            for delta in [-2, 2]:  # Small angle changes
                best_hex_data[i, 2] = (old_angle + delta) % 360
                new_fitness = evaluate_solution_fast(best_hex_data)
                if new_fitness > best_fitness:  # Maximizing fitness
                    best_fitness = new_fitness
                    improvement_made = True
                    current_improvement += 1
                else:
                    best_hex_data[i, 2] = old_angle
        
        # If no improvements were made recently, stop
        if improvement_made:
            last_improvement = 0
        else:
            last_improvement += 1
            if last_improvement >= patience:
                break
    
    return best_hex_data, best_fitness

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Generate initial geometric patterns
    patterns = generate_geometric_pattern(3)
    
    best_fitness = float('-inf')
    best_solution = None
    
    # Try all initial patterns
    for i, pattern in enumerate(patterns):
        # Local search refinement for each pattern
        refined_pattern, fitness = constrained_local_search(pattern)
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_solution = refined_pattern
    
    # Final post-processing with focused refinement
    final_solution, _ = constrained_local_search(best_solution, max_iterations=50)
    
    # Extract results
    inner_hex_data = final_solution.copy()
    
    # Calculate outer hexagon side length
    outer_radius = calculate_outer_hexagon_radius(inner_hex_data[:, :2], inner_hex_data[:, 2])
    outer_hex_side_length = outer_radius / (np.sqrt(3) / 2)
    
    # Outer hexagon centered at origin
    outer_hex_data = np.array([0, 0, 0])
    
    elapsed_time = time.time() - start_time
    print(f"Optimization completed in {elapsed_time:.2f} seconds")
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END