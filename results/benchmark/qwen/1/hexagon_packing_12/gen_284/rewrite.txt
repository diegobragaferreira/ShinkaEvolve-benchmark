# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
from scipy.spatial import cKDTree
import time
from numba import jit

@jit(nopython=True)
def hexagon_vertices(x, y, angle_deg, side_length=1):
    """Compute vertices of a hexagon given center, rotation, and side length."""
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    # Vertices of regular hexagon with side length 1 centered at origin
    base_verts = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    # Rotate and translate
    rotated_verts = np.empty_like(base_verts)
    for i in range(6):
        x_orig, y_orig = base_verts[i]
        rotated_verts[i] = [
            x + side_length * (x_orig * cos_a - y_orig * sin_a),
            y + side_length * (x_orig * sin_a + y_orig * cos_a)
        ]

    return rotated_verts

@jit(nopython=True)
def compute_min_distance_hexagon_hexagon(h1_x, h1_y, h1_angle, h2_x, h2_y, h2_angle):
    """Compute minimum distance between two hexagons using analytical approach."""
    v1 = hexagon_vertices(h1_x, h1_y, h1_angle)
    v2 = hexagon_vertices(h2_x, h2_y, h2_angle)
    
    min_dist = np.inf
    # Check vertex-to-vertex distances
    for i in range(6):
        for j in range(6):
            dist = np.sqrt((v1[i,0]-v2[j,0])**2 + (v1[i,1]-v2[j,1])**2)
            if dist < min_dist:
                min_dist = dist
    
    # Check vertex-to-edge distances
    for i in range(6):
        for j in range(6):
            # Distance from vertex v1[i] to edge v2[j]-v2[(j+1)%6]
            px, py = v1[i,0], v1[i,1]
            x1, y1 = v2[j,0], v2[j,1]
            x2, y2 = v2[(j+1)%6,0], v2[(j+1)%6,1]
            
            # Vector from (x1,y1) to (x2,y2)
            dx = x2 - x1
            dy = y2 - y1

            # Length squared of line segment
            length_sq = dx*dx + dy*dy

            if length_sq == 0:
                # Line segment is a point
                dist = np.sqrt((px - x1)**2 + (py - y1)**2)
            else:
                # Project point onto line
                t = ((px - x1) * dx + (py - y1) * dy) / length_sq
                t = max(0, min(1, t))  # Clamp projection to line segment

                # Find closest point on line segment
                closest_x = x1 + t * dx
                closest_y = y1 + t * dy

                # Distance to closest point
                dist = np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
                
            if dist < min_dist:
                min_dist = dist
            
            # Distance from vertex v2[j] to edge v1[i]-v1[(i+1)%6]
            px, py = v2[j,0], v2[j,1]
            x1, y1 = v1[i,0], v1[i,1]
            x2, y2 = v1[(i+1)%6,0], v1[(i+1)%6,1]
            
            # Vector from (x1,y1) to (x2,y2)
            dx = x2 - x1
            dy = y2 - y1

            # Length squared of line segment
            length_sq = dx*dx + dy*dy

            if length_sq == 0:
                # Line segment is a point
                dist = np.sqrt((px - x1)**2 + (py - y1)**2)
            else:
                # Project point onto line
                t = ((px - x1) * dx + (py - y1) * dy) / length_sq
                t = max(0, min(1, t))  # Clamp projection to line segment

                # Find closest point on line segment
                closest_x = x1 + t * dx
                closest_y = y1 + t * dy

                # Distance to closest point
                dist = np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
                
            if dist < min_dist:
                min_dist = dist

    return min_dist

def compute_hexagon_polygon(x, y, angle_deg, side_length=1):
    """Convert hexagon parameters to shapely polygon."""
    vertices = hexagon_vertices(x, y, angle_deg, side_length)
    return Polygon(vertices)

def evaluate_configuration_symmetric(params):
    """
    Optimized evaluation function using symmetric hexagonal arrangement.
    """
    # Extract core parameters: 9 parameters for reduced search space
    # params[0]: middle ring radius  
    # params[1]: outer ring radius
    # params[2]: middle ring angle offset
    # params[3]: outer ring angle offset  
    # params[4]: outer hexagon angle (rotation)
    # params[5]: outer hexagon center x
    # params[6]: outer hexagon center y
    # params[7]: center hexagon rotation
    # params[8]: middle ring rotation
    
    middle_radius = params[0]
    outer_radius = params[1] 
    middle_angle_offset = params[2]
    outer_angle_offset = params[3]
    
    # Layer 1: Center (1 hexagon)
    layer1_pos = [(0.0, 0.0)]
    layer1_angles = [params[7]]
    
    # Layer 2: Middle ring (6 hexagons)
    layer2_pos = []
    layer2_angles = []
    for i in range(6):
        angle = (i * 60 + middle_angle_offset) % 360
        rad = middle_radius
        x = rad * np.cos(np.radians(angle))
        y = rad * np.sin(np.radians(angle))
        layer2_pos.append((x, y))
        layer2_angles.append(params[8])
    
    # Layer 3: Outer ring (5 hexagons)
    layer3_pos = []
    layer3_angles = []
    for i in range(5):
        angle = (i * 72 + outer_angle_offset) % 360
        rad = outer_radius
        x = rad * np.cos(np.radians(angle))
        y = rad * np.sin(np.radians(angle))
        layer3_pos.append((x, y))
        layer3_angles.append(0.0)
        
    # Combine all positions and angles
    all_positions = layer1_pos + layer2_pos + layer3_pos
    all_angles = layer1_angles + layer2_angles + layer3_angles
    
    # Create inner hexagons
    inner_hexagons = []
    for i, (pos, angle) in enumerate(zip(all_positions, all_angles)):
        x, y = pos
        vertices = hexagon_vertices(x, y, angle)
        inner_hexagons.append(vertices)
        
    # Create outer hexagon with optimized size based on the furthest point
    max_dist = 0
    for hex_vertices in inner_hexagons:
        for vertex in hex_vertices:
            dist = np.sqrt(vertex[0]**2 + vertex[1]**2)
            max_dist = max(max_dist, dist)
            
    # Add buffer and create outer hexagon
    outer_radius_final = max_dist * 1.02  # 2% extra space for numerical stability
    outer_center_x, outer_center_y, outer_angle = params[5:8]
    outer_vertices = hexagon_vertices(outer_center_x, outer_center_y, outer_angle, outer_radius_final)
    
    # Check constraints efficiently
    total_penalty = 0
    
    # Check containment - only need to check center point distance
    for hex_vertices in inner_hexagons:
        # Simple containment check using distance from origin
        for vertex in hex_vertices:
            dist = np.sqrt(vertex[0]**2 + vertex[1]**2)
            if dist > outer_radius_final - 1.0:  # 1.0 is hexagon radius
                total_penalty += 10000
                
    # Optimized overlap checking using spatial acceleration
    if len(inner_hexagons) < 2:
        return -(1.0 / (outer_radius_final + total_penalty + 1e-8))
    
    # Use spatial indexing for efficient overlap detection
    # Build spatial index for efficient neighbor querying
    hex_centers = np.array([[pos[0], pos[1]] for pos in all_positions])
    tree = cKDTree(hex_centers)
    
    # Find neighbors within a reasonable distance (2x hexagon diameter)
    pairs_to_check = tree.query_pairs(r=3.0, p=np.inf)
    
    # Check overlaps using the optimized distance function
    for i, j in pairs_to_check:
        # Skip center with itself (index 0)
        if i == 0 and j == 0:
            continue
            
        # Check overlap using fast distance calculation
        min_dist = compute_min_distance_hexagon_hexagon(
            all_positions[i][0], all_positions[i][1], all_angles[i],
            all_positions[j][0], all_positions[j][1], all_angles[j]
        )
        
        if min_dist < 0.01:  # Overlapping
            total_penalty += 10000
            
    # Additional specific overlap checks for critical pairs
    # Center with all others
    for i in range(1, 12):  # center with all other hexagons
        min_dist = compute_min_distance_hexagon_hexagon(
            all_positions[0][0], all_positions[0][1], all_angles[0],
            all_positions[i][0], all_positions[i][1], all_angles[i]
        )
        if min_dist < 0.01:
            total_penalty += 10000
    
    # Middle ring vs outer ring
    for i in range(1, 7):  # middle ring
        for j in range(7, 12):  # outer ring
            min_dist = compute_min_distance_hexagon_hexagon(
                all_positions[i][0], all_positions[i][1], all_angles[i],
                all_positions[j][0], all_positions[j][1], all_angles[j]
            )
            if min_dist < 0.01:
                total_penalty += 10000
                
    # Middle ring self-intersection
    for i in range(1, 7):
        for j in range(i+1, 7):
            min_dist = compute_min_distance_hexagon_hexagon(
                all_positions[i][0], all_positions[i][1], all_angles[i],
                all_positions[j][0], all_positions[j][1], all_angles[j]
            )
            if min_dist < 0.01:
                total_penalty += 10000
                
    # Outer ring self-intersection  
    for i in range(7, 12):
        for j in range(i+1, 12):
            min_dist = compute_min_distance_hexagon_hexagon(
                all_positions[i][0], all_positions[i][1], all_angles[i],
                all_positions[j][0], all_positions[j][1], all_angles[j]
            )
            if min_dist < 0.01:
                total_penalty += 10000
                
    # Return negative inverse of outer radius plus penalties
    return -(1.0 / (outer_radius_final + total_penalty + 1e-8))

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses symmetric optimization approach for better performance.
    """
    # Initialize with a smart symmetric configuration based on known good patterns
    initial_params = np.array([
        2.15,     # middle ring radius  
        3.6,      # outer ring radius
        30.0,     # middle ring angle offset
        0.0,      # outer ring angle offset  
        0.0,      # outer hexagon angle
        0.0,      # outer hexagon center x
        0.0,      # outer hexagon center y
        0.0,      # center hexagon rotation
        0.0       # middle ring rotation
    ])
    
    # Define tighter bounds for our reduced parameter space
    bounds = [
        (1.0, 4.0),     # middle ring radius
        (2.0, 6.0),     # outer ring radius
        (-180, 180),    # middle ring angle offset
        (-180, 180),    # outer ring angle offset
        (-180, 180),    # outer hex angle
        (-5.0, 5.0),    # outer center x
        (-5.0, 5.0),    # outer center y
        (-180, 180),    # center rotation
        (-180, 180)     # middle rotation
    ]
    
    # Run differential evolution with optimized settings
    try:
        result = differential_evolution(
            evaluate_configuration_symmetric, 
            bounds, 
            maxiter=25, 
            popsize=15, 
            seed=42, 
            disp=False,
            atol=1e-6,
            ftol=1e-6,
            workers=1
        )
        
        # Extract optimized parameters
        optimized_params = result.x
        
        # Recreate final configuration with optimized parameters
        middle_radius = optimized_params[0]
        outer_radius = optimized_params[1] 
        middle_angle_offset = optimized_params[2]
        outer_angle_offset = optimized_params[3]
        
        # Reconstruct layout
        layer1_pos = [(0.0, 0.0)]
        layer1_angles = [optimized_params[7]]
        
        layer2_pos = []
        layer2_angles = []
        for i in range(6):
            angle = (i * 60 + middle_angle_offset) % 360
            rad = middle_radius
            x = rad * np.cos(np.radians(angle))
            y = rad * np.sin(np.radians(angle))
            layer2_pos.append((x, y))
            layer2_angles.append(optimized_params[8])
            
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
            vertices = hexagon_vertices(x, y, angle)
            inner_hexagons.append(vertices)
            
        # Calculate exact outer radius needed
        max_dist = 0
        for hex_vertices in inner_hexagons:
            for vertex in hex_vertices:
                dist = np.sqrt(vertex[0]**2 + vertex[1]**2)
                max_dist = max(max_dist, dist)
                
        outer_radius_final = max_dist * 1.02  # Add 2% buffer for numerical stability
        outer_center_x, outer_center_y, outer_angle = optimized_params[5:8]
        outer_vertices = hexagon_vertices(outer_center_x, outer_center_y, outer_angle, outer_radius_final)
        
        # Final validation
        valid = True
        for hex_vertices in inner_hexagons:
            # Simple containment check
            for vertex in hex_vertices:
                dist = np.sqrt(vertex[0]**2 + vertex[1]**2)
                if dist > outer_radius_final - 1.0:
                    valid = False
                    break
            if not valid:
                break
                
        # More thorough overlap checking for robustness
        if valid:
            for i in range(len(inner_hexagons)):
                for j in range(i+1, len(inner_hexagons)):
                    min_dist = compute_min_distance_hexagon_hexagon(
                        all_positions[i][0], all_positions[i][1], all_angles[i],
                        all_positions[j][0], all_positions[j][1], all_angles[j]
                    )
                    if min_dist < 0.01:
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