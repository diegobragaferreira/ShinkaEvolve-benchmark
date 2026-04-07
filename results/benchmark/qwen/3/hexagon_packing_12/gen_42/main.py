# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
import time

def hexagon_vertices(center_x, center_y, size=1, angle_deg=0):
    """Generate vertices of a regular hexagon given center, size, and rotation."""
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * np.pi / 3
        x = center_x + size * np.cos(angle)
        y = center_y + size * np.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def compute_outer_hex_radius(inner_hex_data, outer_center_x, outer_center_y):
    """Compute minimum outer hexagon radius that contains all inner hexagons."""
    max_distance = 0
    for i in range(len(inner_hex_data)):
        cx, cy, _ = inner_hex_data[i]
        distance = np.sqrt((cx - outer_center_x)**2 + (cy - outer_center_y)**2)
        max_distance = max(max_distance, distance + 1)  # Add radius of unit hexagon
    return max_distance

def evaluate_configuration(inner_hex_data, outer_center_x, outer_center_y):
    """Evaluate current configuration: returns (validity, inv_radius)."""
    # Check for overlaps
    for i in range(len(inner_hex_data)):
        hex1_vertices = hexagon_vertices(inner_hex_data[i][0], inner_hex_data[i][1], 1, inner_hex_data[i][2])
        for j in range(i+1, len(inner_hex_data)):
            hex2_vertices = hexagon_vertices(inner_hex_data[j][0], inner_hex_data[j][1], 1, inner_hex_data[j][2])
            if check_overlap(hex1_vertices, hex2_vertices):
                return False, 0
    
    # Check containment
    outer_radius = compute_outer_hex_radius(inner_hex_data, outer_center_x, outer_center_y)
    outer_vertices = hexagon_vertices(outer_center_x, outer_center_y, outer_radius, 0)
    outer_polygon = Polygon(outer_vertices)
    
    for i in range(len(inner_hex_data)):
        hex_vertices = hexagon_vertices(inner_hex_data[i][0], inner_hex_data[i][1], 1, inner_hex_data[i][2])
        for vertex in hex_vertices:
            point = Point(vertex[0], vertex[1])
            if not outer_polygon.contains(point):
                return False, 0
    
    # Return inverse of outer radius
    return True, 1.0 / outer_radius

def generate_symmetric_config():
    """Generate a highly symmetric initial configuration for 12 hexagons."""
    # This uses the principle that optimal packings often have high symmetry
    # We'll place hexagons in rings with rotational symmetry
    
    config = []
    
    # Central hexagon
    config.append([0, 0, 0])
    
    # First ring - 6 hexagons, placed at 60-degree intervals around a circle
    ring1_radius = 2.0  # This value is chosen to allow good packing efficiency
    for i in range(6):
        angle = i * 60
        x = ring1_radius * np.cos(np.radians(angle))
        y = ring1_radius * np.sin(np.radians(angle))
        config.append([x, y, 0])
    
    # Second ring - 5 hexagons, placed with some symmetry
    # Using a smaller radius to fit within the central cluster
    ring2_radius = 3.5  # Adjusted for better packing
    angles = [0, 72, 144, 216, 288]  # 5 evenly spaced angles for 5 hexagons
    for angle in angles:
        x = ring2_radius * np.cos(np.radians(angle))
        y = ring2_radius * np.sin(np.radians(angle))
        config.append([x, y, 0])
    
    # Add one more hexagon to reach 12 (place at negative y-axis)
    config.append([0, -ring2_radius - 1.0, 0])
    
    return np.array(config)

def refine_with_optimization(initial_config, outer_center_x, outer_center_y):
    """Refine the symmetric configuration using optimization while maintaining symmetry."""
    
    # Define constraint that maintains rotational symmetry for certain groups
    def objective(params):
        # Reconstruct configuration from flattened parameters
        config = initial_config.copy()
        
        # Update positions only (leave angles as they are)
        # We'll optimize the radial distances and angular positions of symmetric groups
        idx = 0
        
        # Central hexagon stays at origin
        # First ring: 6 hexagons, we'll treat them as rotating group with 60-degree spacing
        # For simplicity, let's update first ring hexagon positions
        for i in range(1, 7):  # First ring hexagons
            config[i][0] = params[idx]
            config[i][1] = params[idx + 1]
            idx += 2
        
        # Second ring: 5 hexagons
        for i in range(7, 12):  # Second ring hexagons
            config[i][0] = params[idx]
            config[i][1] = params[idx + 1]
            idx += 2
        
        validity, inv_radius = evaluate_configuration(config, outer_center_x, outer_center_y)
        if not validity:
            return 1e10  # Large penalty for invalid configurations
        return -inv_radius  # Negative because we want to maximize
    
    # Flatten initial configuration for optimization
    initial_params = []
    
    # First ring hexagon positions (6 hexagons)
    for i in range(1, 7):
        initial_params.extend([initial_config[i][0], initial_config[i][1]])
    
    # Second ring hexagon positions (5 hexagons)
    for i in range(7, 12):
        initial_params.extend([initial_config[i][0], initial_config[i][1]])
    
    # Perform optimization with bounds
    bounds = [(-10, 10), (-10, 10)] * 11  # Only optimize 11 of the 12 positions
    # Keep center hexagon fixed (index 0)
    
    result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds, 
                     options={'maxiter': 1000})
    
    # Reconstruct optimized configuration
    optimized_config = initial_config.copy()
    idx = 0
    
    # First ring hexagon positions (6 hexagons)
    for i in range(1, 7):
        optimized_config[i][0] = result.x[idx]
        optimized_config[i][1] = result.x[idx + 1]
        idx += 2
    
    # Second ring hexagon positions (5 hexagons)
    for i in range(7, 12):
        optimized_config[i][0] = result.x[idx]
        optimized_config[i][1] = result.x[idx + 1]
        idx += 2
        
    return optimized_config

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Generate symmetric initial configuration
    initial_config = generate_symmetric_config()
    
    # Set outer hexagon at center
    outer_center_x, outer_center_y = 0.0, 0.0
    
    # Refine using optimization while maintaining symmetry structure
    optimized_config = refine_with_optimization(initial_config, outer_center_x, outer_center_y)
    
    # Final verification and refinement loop
    max_attempts = 5
    for attempt in range(max_attempts):
        validity, inv_radius = evaluate_configuration(optimized_config, outer_center_x, outer_center_y)
        if validity:
            break
            
        # If not valid, try small random adjustments to positions
        for i in range(len(optimized_config)):
            optimized_config[i][0] += np.random.normal(0, 0.02)
            optimized_config[i][1] += np.random.normal(0, 0.02)
    
    # Compute final outer hexagon radius
    outer_radius = 1.0 / inv_radius if inv_radius > 0 else 10.0
    
    # Ensure we have exactly 12 hexagons and return final result
    inner_hex_data = np.array(optimized_config)
    if len(inner_hex_data) != 12:
        # Fallback to simple configuration if needed (shouldn't happen)
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
            [0, -4, 0]
        ])
        outer_radius = 8.0
    
    outer_hex_data = np.array([outer_center_x, outer_center_y, 0])
    outer_hex_side_length = outer_radius * 2  # approximate
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
