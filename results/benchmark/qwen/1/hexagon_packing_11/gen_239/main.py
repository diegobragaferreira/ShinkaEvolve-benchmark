# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
import time
import random

def generate_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, rotation, and side length."""
    angle_rad = np.radians(angle_deg)
    # Vertices of a regular hexagon with side_length=1 centered at origin
    base_vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = np.cos(theta)
        y = np.sin(theta)
        base_vertices.append((x, y))

    # Scale and translate
    vertices = [(center_x + side_length * vx, center_y + side_length * vy) for vx, vy in base_vertices]
    return vertices

def check_containment(hexagon_vertices, outer_hexagon_vertices):
    """Check if all vertices of inner hexagon are inside outer hexagon."""
    inner_poly = Polygon(hexagon_vertices)
    outer_poly = Polygon(outer_hexagon_vertices)
    return outer_poly.contains(inner_poly)

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def calculate_outer_hexagon_radius(inner_positions, inner_angles):
    """Calculate minimum radius needed to contain all inner hexagons"""
    max_dist = 0
    outer_center = (0, 0)
    
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_positions)):
        pos = inner_positions[i]
        angle = inner_angles[i]
        hex_vertices = generate_hexagon_vertices(pos[0], pos[1], angle)
        all_vertices.extend(hex_vertices)
    
    # Find maximum distance from center
    for vertex in all_vertices:
        dist = np.sqrt((vertex[0] - outer_center[0])**2 + (vertex[1] - outer_center[1])**2)
        max_dist = max(max_dist, dist)
    
    # Add buffer for safety and account for hexagon shape
    return max_dist * 1.1  # Safety factor

def get_hexagon_polygon(x, y, angle_deg, side_length=1):
    """Get shapely polygon representation of hexagon"""
    vertices = generate_hexagon_vertices(x, y, angle_deg, side_length)
    return Polygon(vertices)

def is_valid_config(inner_positions, inner_angles, outer_side_length):
    """Check if the current configuration is valid"""
    # Create outer hexagon
    outer_vertices = generate_hexagon_vertices(0, 0, 0, outer_side_length)
    outer_poly = Polygon(outer_vertices)
    
    # Check containment
    for i in range(len(inner_positions)):
        pos = inner_positions[i]
        angle = inner_angles[i]
        vertices = generate_hexagon_vertices(pos[0], pos[1], angle)
        inner_poly = Polygon(vertices)
        if not outer_poly.contains(inner_poly):
            return False, outer_side_length
    
    # Check overlaps
    for i in range(len(inner_positions)):
        for j in range(i+1, len(inner_positions)):
            pos1 = inner_positions[i]
            angle1 = inner_angles[i]
            pos2 = inner_positions[j]
            angle2 = inner_angles[j]
            
            vertices1 = generate_hexagon_vertices(pos1[0], pos1[1], angle1)
            vertices2 = generate_hexagon_vertices(pos2[0], pos2[1], angle2)
            
            if check_overlap(vertices1, vertices2):
                return False, outer_side_length
    
    return True, outer_side_length

def greedy_hexagon_placement():
    """Construct an initial good configuration using greedy placement"""
    # Start with a symmetric arrangement
    # Center hexagon
    positions = [[0.0, 0.0]]
    angles = [0.0]
    
    # Surrounding hexagons in a hexagonal pattern
    for i in range(6):
        angle = i * 60
        radius = 2.0
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        positions.append([x, y])
        angles.append(0.0)
    
    # Additional positions to fill the pattern
    additional_positions = [
        (-3.0, 1.0), (3.0, 1.0),
        (-3.0, -1.0), (3.0, -1.0),
        (0.0, 3.0), (0.0, -3.0),
        (1.5, 2.6), (-1.5, -2.6),
        (-1.5, 2.6), (1.5, -2.6)
    ]
    
    for pos in additional_positions:
        if len(positions) < 11:
            positions.append(list(pos))
            angles.append(0.0)
    
    # Ensure exactly 11 positions
    while len(positions) < 11:
        positions.append([0.0, 0.0])
        angles.append(0.0)
    
    return np.array(positions), np.array(angles)

def compute_penalty(inner_positions, inner_angles, outer_side_length):
    """Compute penalty for current configuration"""
    penalty = 0
    
    # Create outer hexagon
    outer_vertices = generate_hexagon_vertices(0, 0, 0, outer_side_length)
    outer_poly = Polygon(outer_vertices)
    
    # Check containment penalty
    for i in range(len(inner_positions)):
        pos = inner_positions[i]
        angle = inner_angles[i]
        vertices = generate_hexagon_vertices(pos[0], pos[1], angle)
        inner_poly = Polygon(vertices)
        if not outer_poly.contains(inner_poly):
            penalty += 1000000
    
    # Check overlap penalty
    for i in range(len(inner_positions)):
        for j in range(i+1, len(inner_positions)):
            pos1 = inner_positions[i]
            angle1 = inner_angles[i]
            pos2 = inner_positions[j]
            angle2 = inner_angles[j]
            
            vertices1 = generate_hexagon_vertices(pos1[0], pos1[1], angle1)
            vertices2 = generate_hexagon_vertices(pos2[0], pos2[1], angle2)
            
            if check_overlap(vertices1, vertices2):
                penalty += 1000000
    
    # Add inverse of outer hexagon size as objective
    if outer_side_length > 0:
        penalty += -1.0 / outer_side_length
    else:
        penalty += 1000000
    
    return penalty

def local_improve_positions(inner_positions, inner_angles, outer_side_length, max_iter=50):
    """Locally improve positions using gradient-based optimization"""
    # Flatten parameters for optimization
    flat_params = np.concatenate([inner_positions.flatten(), inner_angles])
    
    def objective(params):
        # Reshape parameters
        new_positions = params[:22].reshape(-1, 2)
        new_angles = params[22:]
        
        return compute_penalty(new_positions, new_angles, outer_side_length)
    
    # Optimize using L-BFGS method for better convergence
    try:
        result = minimize(
            objective,
            flat_params,
            method='L-BFGS-B',
            bounds=[(-10, 10)] * 22 + [(0, 360)] * 11,
            options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-8},
            tol=1e-8
        )
        
        if result.success:
            new_positions = result.x[:22].reshape(-1, 2)
            new_angles = result.x[22:]
            
            # Validate the result
            valid, _ = is_valid_config(new_positions, new_angles, outer_side_length)
            if valid:
                return new_positions, new_angles
    except:
        pass
    
    return inner_positions, inner_angles

def construct_packing():
    """Main construction algorithm using greedy approach"""
    # Start with a good initial configuration
    positions, angles = greedy_hexagon_placement()
    
    # Get initial outer side length
    outer_radius = calculate_outer_hexagon_radius(positions, angles)
    outer_side_length = outer_radius / (np.sqrt(3) / 2)
    
    # Iteratively improve
    improvement_count = 0
    max_improvements = 20
    
    for iteration in range(100):
        # Store previous configuration
        prev_positions = positions.copy()
        prev_angles = angles.copy()
        prev_side_length = outer_side_length
        
        # Try to improve using local optimization
        positions, angles = local_improve_positions(positions, angles, outer_side_length)
        
        # Recalculate outer side length
        outer_radius = calculate_outer_hexagon_radius(positions, angles)
        outer_side_length = outer_radius / (np.sqrt(3) / 2)
        
        # Check if we made improvement
        if abs(prev_side_length - outer_side_length) > 1e-6 and improvement_count < max_improvements:
            improvement_count += 1
        else:
            improvement_count = 0
            
        # Early stopping if no significant improvements
        if improvement_count >= max_improvements:
            break
    
    return positions, angles, outer_side_length

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    try:
        # Construct the packing
        positions, angles, outer_side_length = construct_packing()
        
        # Create inner hex data
        inner_hex_data = np.column_stack([positions, angles])
        
        # Create outer hex data (centered)
        outer_hex_data = np.array([0, 0, 0])
        
        elapsed_time = time.time() - start_time
        print(f"Construction completed in {elapsed_time:.2f} seconds")
        
        return inner_hex_data, outer_hex_data, outer_side_length
        
    except Exception as e:
        print(f"Construction failed: {e}")
        # Fallback to simple solution
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
        outer_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END