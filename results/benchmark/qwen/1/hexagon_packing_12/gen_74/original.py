# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
import time

def generate_hexagon_vertices(center_x, center_y, angle_degrees, side_length=1):
    """Generate vertices of a regular hexagon."""
    angle_rad = np.radians(angle_degrees)
    angles = np.linspace(0, 2*np.pi, 7) + angle_rad  # 6 sides + closing vertex
    vertices = []
    for angle in angles:
        x = center_x + side_length * np.cos(angle)
        y = center_y + side_length * np.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

def check_containment(hexagon_vertices, outer_hexagon_vertices):
    """Check if all vertices of inner hexagon are within outer hexagon."""
    inner_polygon = Polygon(hexagon_vertices)
    outer_polygon = Polygon(outer_hexagon_vertices)
    return outer_polygon.contains(inner_polygon)

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap."""
    polygon1 = Polygon(hex1_vertices)
    polygon2 = Polygon(hex2_vertices)
    return polygon1.intersects(polygon2)

def evaluate_configuration_symmetric(params):
    """
    Optimized evaluation function using symmetric hexagonal arrangement.
    Takes 12 parameters: 3 positions * 4 layers (center, 3 rings)
    """
    # Extract core parameters: radius, angle offset, layer properties
    # params[0]: center hexagon radius (fixed to 0)
    # params[1]: middle ring radius  
    # params[2]: outer ring radius
    # params[3]: middle ring angle offset
    # params[4]: outer ring angle offset
    # params[5]: middle ring hexagon spacing factor (0-1) 
    # params[6]: outer ring hexagon spacing factor (0-1)
    # params[7]: outer hexagon angle (rotation)
    # params[8]: outer hexagon center x
    # params[9]: outer hexagon center y
    # params[10]: center hexagon rotation
    # params[11]: middle ring rotation
    
    # Get layer distances and angles
    middle_radius = params[1]
    outer_radius = params[2] 
    middle_angle_offset = params[3]
    outer_angle_offset = params[4]
    
    # Layer 1: Center (1 hexagon)
    layer1_pos = [(0.0, 0.0)]
    layer1_angles = [params[10]]
    
    # Layer 2: Middle ring (6 hexagons)
    layer2_pos = []
    layer2_angles = []
    for i in range(6):
        angle = (i * 60 + middle_angle_offset) % 360
        rad = middle_radius
        x = rad * np.cos(np.radians(angle))
        y = rad * np.sin(np.radians(angle))
        layer2_pos.append((x, y))
        layer2_angles.append(params[11])
    
    # Layer 3: Outer ring (5 hexagons) - we'll place them carefully
    layer3_pos = []
    layer3_angles = []
    # Place 5 hexagons around the circle
    for i in range(5):
        angle = (i * 72 + outer_angle_offset) % 360
        rad = outer_radius
        x = rad * np.cos(np.radians(angle))
        y = rad * np.sin(np.radians(angle))
        layer3_pos.append((x, y))
        layer3_angles.append(0.0)  # No rotation for now
        
    # Combine all positions and angles
    all_positions = layer1_pos + layer2_pos + layer3_pos
    all_angles = layer1_angles + layer2_angles + layer3_angles
    
    # Create inner hexagons
    inner_hexagons = []
    for i, (pos, angle) in enumerate(zip(all_positions, all_angles)):
        x, y = pos
        vertices = generate_hexagon_vertices(x, y, angle)
        inner_hexagons.append(vertices)
        
    # Create outer hexagon with optimized size based on the furthest point
    max_dist = 0
    for hex_vertices in inner_hexagons:
        for vertex in hex_vertices:
            dist = np.sqrt(vertex[0]**2 + vertex[1]**2)
            max_dist = max(max_dist, dist)
            
    # Add buffer and create outer hexagon
    outer_radius_final = max_dist * 1.05  # 5% extra space
    outer_center_x, outer_center_y, outer_angle = params[8:11]
    outer_vertices = generate_hexagon_vertices(outer_center_x, outer_center_y, outer_angle, outer_radius_final)
    
    # Check constraints
    total_penalty = 0
    
    # Check containment
    for hex_vertices in inner_hexagons:
        if not check_containment(hex_vertices, outer_vertices):
            total_penalty += 1000
            
    # Check overlaps (only critical ones)
    num_pairs_to_check = 0
    overlap_pairs = []
    
    # Check center with ring 1 and ring 2
    for i in range(1, 7):  # center with ring 1 hexes
        if check_overlap(inner_hexagons[0], inner_hexagons[i]):
            total_penalty += 1000
            
    for i in range(7, 12):  # center with ring 2 hexes  
        if check_overlap(inner_hexagons[0], inner_hexagons[i]):
            total_penalty += 1000
            
    # Check ring 1 vs ring 2
    for i in range(1, 7):  # ring 1 vs ring 2
        for j in range(7, 12):
            if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                total_penalty += 1000
                
    # Check ring 1 self-intersection (shouldn't happen with good spacing)
    for i in range(1, 7):
        for j in range(i+1, 7):
            if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                total_penalty += 1000
                
    # Check ring 2 self-intersection (shouldn't happen)
    for i in range(7, 12):  
        for j in range(i+1, 12):
            if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                total_penalty += 1000
    
    # Return negative inverse of outer radius plus penalties
    return -(1.0 / (outer_radius_final + total_penalty))

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses symmetric optimization approach for better performance.
    """
    # Initialize with a smart symmetric configuration
    initial_params = np.array([
        0.0,      # center hexagon radius (fixed)
        2.2,      # middle ring radius  
        3.5,      # outer ring radius
        30.0,     # middle ring angle offset
        0.0,      # outer ring angle offset  
        0.9,      # middle ring spacing
        0.9,      # outer ring spacing
        0.0,      # outer hexagon angle
        0.0,      # outer hexagon center x
        0.0,      # outer hexagon center y
        0.0,      # center hexagon rotation
        0.0       # middle ring rotation
    ])
    
    # Define bounds for our reduced parameter space
    bounds = [
        (0.0, 0.0),     # center radius - fixed
        (1.0, 4.0),     # middle ring radius
        (2.0, 6.0),     # outer ring radius
        (-180, 180),    # middle ring angle offset
        (-180, 180),    # outer ring angle offset
        (0.5, 1.0),     # middle spacing
        (0.5, 1.0),     # outer spacing
        (-180, 180),    # outer hex angle
        (-5.0, 5.0),    # outer center x
        (-5.0, 5.0),    # outer center y
        (-180, 180),    # center rotation
        (-180, 180)     # middle rotation
    ]
    
    # Run differential evolution on the symmetric model
    try:
        result = differential_evolution(
            evaluate_configuration_symmetric, 
            bounds, 
            maxiter=30, 
            popsize=10, 
            seed=42, 
            disp=False,
            atol=1e-6,
            ftol=1e-6
        )
        
        # Extract optimized parameters
        optimized_params = result.x
        
        # Recreate final configuration
        middle_radius = optimized_params[1]
        outer_radius = optimized_params[2] 
        middle_angle_offset = optimized_params[3]
        outer_angle_offset = optimized_params[4]
        
        # Reconstruct layout
        layer1_pos = [(0.0, 0.0)]
        layer1_angles = [optimized_params[10]]
        
        layer2_pos = []
        layer2_angles = []
        for i in range(6):
            angle = (i * 60 + middle_angle_offset) % 360
            rad = middle_radius
            x = rad * np.cos(np.radians(angle))
            y = rad * np.sin(np.radians(angle))
            layer2_pos.append((x, y))
            layer2_angles.append(optimized_params[11])
            
        layer3_pos = []
        layer3_angles = []
        for i in range(5):
            angle = (i * 72 + outer_angle_offset) % 360
            rad = outer_radius
            x = rad * np.cos(np.radians(angle))
            y = rad * np.sin(np.radians(angle))
            layer3_pos.append((x, y))
            layer3_angles.append(0.0)
            
        all_positions = layer1_pos + layer2_pos + layer3_pos
        all_angles = layer1_angles + layer2_angles + layer3_angles
        
        # Create inner hexagons
        inner_hexagons = []
        for i, (pos, angle) in enumerate(zip(all_positions, all_angles)):
            x, y = pos
            vertices = generate_hexagon_vertices(x, y, angle)
            inner_hexagons.append(vertices)
            
        # Calculate exact outer radius needed
        max_dist = 0
        for hex_vertices in inner_hexagons:
            for vertex in hex_vertices:
                dist = np.sqrt(vertex[0]**2 + vertex[1]**2)
                max_dist = max(max_dist, dist)
                
        outer_radius_final = max_dist * 1.05  # Add buffer
        outer_center_x, outer_center_y, outer_angle = optimized_params[8:11]
        outer_vertices = generate_hexagon_vertices(outer_center_x, outer_center_y, outer_angle, outer_radius_final)
        
        # Final validation
        valid = True
        for hex_vertices in inner_hexagons:
            if not check_containment(hex_vertices, outer_vertices):
                valid = False
                break
                
        # Overlap checking
        for i in range(len(inner_hexagons)):
            for j in range(i+1, len(inner_hexagons)):
                if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                    valid = False
                    break
            if not valid:
                break
                
        if not valid:
            # Fallback to previous working configuration
            inner_hex_data = np.array([
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
                [3.75, -2.17, 0],  
                [0, -4, 0],  
            ])
            outer_hex_data = np.array([0, 0, 0])
            outer_hex_side_length = 8
            return inner_hex_data, outer_hex_data, outer_hex_side_length

        # Format output with optimized positions
        inner_hex_data = np.zeros((12, 3))
        for i, (pos, angle) in enumerate(zip(all_positions, all_angles)):
            inner_hex_data[i] = [pos[0], pos[1], angle]
            
        outer_hex_data = np.array([outer_center_x, outer_center_y, outer_angle])
        outer_hex_side_length = outer_radius_final
        
        return inner_hex_data, outer_hex_data, outer_hex_side_length
        
    except Exception as e:
        # Fallback if optimization fails
        inner_hex_data = np.array([
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
            [3.75, -2.17, 0],  
            [0, -4, 0],  
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
