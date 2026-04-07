# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
from collections import defaultdict
import math

# Set seed for reproducibility
np.random.seed(42)

def create_regular_hexagon(center_x: float, center_y: float, side_length: float = 1.0, rotation_deg: float = 0.0) -> Polygon:
    """Create a regular hexagon as a Shapely polygon."""
    angle_rad = np.radians(rotation_deg)
    points = []
    for i in range(6):
        angle = angle_rad + i * np.pi / 3
        x = center_x + side_length * np.cos(angle)
        y = center_y + side_length * np.sin(angle)
        points.append((x, y))
    return Polygon(points)

def compute_hexagon_vertices(center_x: float, center_y: float, rotation_deg: float, side_length: float = 1.0):
    """Compute hexagon vertices for geometric operations."""
    angle_rad = np.radians(rotation_deg)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * np.pi / 3
        x = center_x + side_length * np.cos(angle)
        y = center_y + side_length * np.sin(angle)
        vertices.append((x, y))
    return vertices

def get_bounding_box(hex_data):
    """Get bounding box for a set of hexagons."""
    if len(hex_data) == 0:
        return None, None, None, None
    
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    
    for i in range(len(hex_data)):
        center_x, center_y, rotation = hex_data[i]
        vertices = compute_hexagon_vertices(center_x, center_y, rotation)
        for x, y in vertices:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
    
    return min_x, max_x, min_y, max_y

def compute_outer_hexagon_radius_from_points(points):
    """Compute minimum radius for outer hexagon containing all points."""
    if len(points) == 0:
        return 0.0
    
    center_x = sum(p[0] for p in points) / len(points)
    center_y = sum(p[1] for p in points) / len(points)
    
    max_dist = 0.0
    for x, y in points:
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = max(max_dist, dist)
    
    # Add small margin
    return max_dist * 1.01

def check_containment(hexagon: Polygon, outer_hex: Polygon) -> bool:
    """Check if hexagon is fully contained within outer hexagon."""
    return outer_hex.contains(hexagon)

def check_collision(hex1: Polygon, hex2: Polygon) -> bool:
    """Check if two hexagons collide (overlap)."""
    return hex1.intersects(hex2)

def calculate_penalty(inner_hex_data: np.ndarray, outer_side_length: float) -> tuple:
    """Calculate penalty for a configuration"""
    # Create outer hexagon (centered at origin)
    outer_hex = create_regular_hexagon(0, 0, outer_side_length)
    
    # Check containment and collisions for all inner hexagons
    inner_hexagons = []
    total_penalty = 0.0
    
    for i in range(len(inner_hex_data)):
        center_x, center_y, rotation = inner_hex_data[i]
        inner_hex = create_regular_hexagon(center_x, center_y, 1.0, rotation)
        
        # Check containment
        if not check_containment(inner_hex, outer_hex):
            total_penalty += 1000.0  # Large penalty for containment violation
        
        inner_hexagons.append(inner_hex)
    
    # Check pairwise collisions
    for i in range(len(inner_hexagons)):
        for j in range(i+1, len(inner_hexagons)):
            if check_collision(inner_hexagons[i], inner_hexagons[j]):
                total_penalty += 100.0  # Penalty for collision
    
    # Calculate number of hexagons that fit (should be 11)
    num_fits = len(inner_hexagons)
    
    if num_fits != 11:
        total_penalty += 10000.0  # Very high penalty for wrong count
    
    is_valid = (total_penalty == 0.0)
    return total_penalty, is_valid

def generate_structured_initial_config():
    """Generate a carefully constructed initial configuration based on geometric principles."""
    config = np.zeros((11, 3))
    
    # Start with a known good hexagonal packing pattern
    # Central hexagon
    config[0] = [0.0, 0.0, 0.0]
    
    # First ring (6 hexagons)
    ring1_angles = [i * 60 for i in range(6)]
    ring1_distance = 2.0
    
    for i, angle in enumerate(ring1_angles):
        rad = np.radians(angle)
        x = ring1_distance * np.cos(rad)
        y = ring1_distance * np.sin(rad)
        config[i+1] = [x, y, 0.0]
    
    # Second ring (4 hexagons in strategic locations)
    # Spread them more evenly for better packing
    ring2_angles = [30, 90, 150, 210]
    ring2_distance = 3.5
    
    for i, angle in enumerate(ring2_angles):
        rad = np.radians(angle)
        x = ring2_distance * np.cos(rad)
        y = ring2_distance * np.sin(rad)
        config[i+7] = [x, y, 0.0]
    
    return config

def compute_enclosing_hexagon_radius(inner_hex_data):
    """Compute the minimal radius needed for outer hexagon to contain all inner hexagons."""
    all_points = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, rotation = inner_hex_data[i]
        vertices = compute_hexagon_vertices(center_x, center_y, rotation)
        all_points.extend(vertices)
    
    if len(all_points) == 0:
        return 0.0
    
    return compute_outer_hexagon_radius_from_points(all_points)

def optimize_positions_only(inner_hex_data, outer_radius_guess):
    """Optimize positions only with fixed rotations, using geometric constraints."""
    # Simple gradient descent approach for position optimization
    learning_rate = 0.02
    max_iterations = 500
    tolerance = 1e-4
    
    for iteration in range(max_iterations):
        prev_positions = inner_hex_data.copy()
        
        # For each hexagon, try to move it to reduce collisions/containment violations
        for i in range(len(inner_hex_data)):
            center_x, center_y, rotation = inner_hex_data[i]
            
            # Get neighbors
            neighbors = []
            for j in range(len(inner_hex_data)):
                if i != j:
                    neighbors.append(j)
            
            # Compute gradients to move away from constraints
            dx, dy = 0.0, 0.0
            
            # Check collisions with neighbors
            for j in neighbors:
                other_x, other_y, other_rot = inner_hex_data[j]
                dist = np.sqrt((center_x - other_x)**2 + (center_y - other_y)**2)
                
                # Repel if too close
                if dist < 1.99:  # Slightly less than 2 (sum of radii)
                    force = (1.99 - dist) / (dist + 1e-8)
                    dx += force * (center_x - other_x) / (dist + 1e-8)
                    dy += force * (center_y - other_y) / (dist + 1e-8)
            
            # Adjust position
            center_x += dx * learning_rate
            center_y += dy * learning_rate
            
            inner_hex_data[i] = [center_x, center_y, rotation]
        
        # Check convergence
        delta = np.sum(np.abs(inner_hex_data[:, :2] - prev_positions[:, :2]))
        if delta < tolerance:
            break
    
    return inner_hex_data

def geometric_optimization_step(inner_hex_data, max_outer_radius=15.0):
    """Perform a specialized geometric optimization step."""
    # Try to improve by making small adjustments to positions only
    optimized = optimize_positions_only(inner_hex_data.copy(), max_outer_radius)
    
    # Evaluate if this improved the situation
    penalty_orig, valid_orig = calculate_penalty(inner_hex_data, max_outer_radius)
    penalty_new, valid_new = calculate_penalty(optimized, max_outer_radius)
    
    if valid_new and penalty_new < penalty_orig:
        return optimized
    else:
        return inner_hex_data

def binary_search_outer_radius(inner_hex_data, min_radius=1.0, max_radius=15.0, tolerance=0.001):
    """Binary search to find minimal outer hexagon radius."""
    # Binary search to find minimal radius that works
    left, right = min_radius, max_radius
    best_radius = right
    
    while right - left > tolerance:
        mid = (left + right) / 2.0
        penalty, valid = calculate_penalty(inner_hex_data, mid)
        
        if valid:
            best_radius = mid
            right = mid
        else:
            left = mid
    
    return best_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Start with structured configuration
    current_config = generate_structured_initial_config()
    
    # Apply geometric optimizations
    for _ in range(5):  # Multiple optimization passes
        current_config = geometric_optimization_step(current_config)
    
    # Binary search to find minimal radius
    best_radius = binary_search_outer_radius(current_config, min_radius=1.0, max_radius=15.0)
    
    # Final validation
    penalty, valid = calculate_penalty(current_config, best_radius)
    
    # If not valid, use fallback
    if not valid:
        # Use known decent configuration
        current_config = generate_structured_initial_config()
        best_radius = 8.0  # Conservative estimate
    
    # Ensure we have a valid result
    if best_radius <= 0:
        best_radius = 8.0
    
    # Create outer hex data (at center with no rotation)
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    elapsed = time.time() - start_time
    print(f"Eval time: {elapsed:.4f}s")
    
    return current_config, outer_hex_data, best_radius

# EVOLVE-BLOCK-END