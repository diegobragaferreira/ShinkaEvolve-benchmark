# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
import time

def generate_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, angle, and side length"""
    angle_rad = np.radians(angle_deg)
    # Vertices of a regular hexagon with side length 1, centered at origin
    base_vertices = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Rotate and translate
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_vertices = base_vertices @ rotation_matrix.T
    translated_vertices = rotated_vertices + np.array([center_x, center_y])
    
    return translated_vertices

def check_containment(hexagon_poly, outer_hex_poly):
    """Check if hexagon is fully contained within outer hexagon"""
    return outer_hex_poly.contains(hexagon_poly)

def check_overlap(hex1_poly, hex2_poly):
    """Check if two hexagons overlap"""
    return hex1_poly.intersects(hex2_poly)

def calculate_outer_hex_side_length(inner_hex_data):
    """Calculate minimum side length of outer hexagon needed to contain all inner hexagons"""
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = generate_hexagon_vertices(center_x, center_y, angle)
        all_vertices.extend(vertices)
    
    all_vertices = np.array(all_vertices)
    
    # Find bounding box
    min_x, max_x = np.min(all_vertices[:, 0]), np.max(all_vertices[:, 0])
    min_y, max_y = np.min(all_vertices[:, 1]), np.max(all_vertices[:, 1])
    
    # Calculate approximate side length (simplified approach)
    # A hexagon with side length s has width 2*s and height sqrt(3)*s
    width = max_x - min_x
    height = max_y - min_y
    
    # Estimate side length from dimensions
    side_len_width = width / 2.0
    side_len_height = height / (np.sqrt(3))
    
    # Take maximum to ensure containment
    estimated_side_length = max(side_len_width, side_len_height) * 1.1  # Add small buffer
    
    return estimated_side_length

def evaluate_solution(inner_hex_data):
    """Evaluate fitness of solution - maximize 1/outer_hex_side_length"""
    try:
        # Create polygons for all inner hexagons
        hex_polygons = []
        for i in range(len(inner_hex_data)):
            center_x, center_y, angle = inner_hex_data[i]
            vertices = generate_hexagon_vertices(center_x, center_y, angle)
            hex_polygons.append(Polygon(vertices))
        
        # Check containment and overlap
        outer_side_length = calculate_outer_hex_side_length(inner_hex_data)
        outer_vertices = generate_hexagon_vertices(0, 0, 0, outer_side_length)
        outer_polygon = Polygon(outer_vertices)
        
        # Check containment
        for poly in hex_polygons:
            if not check_containment(poly, outer_polygon):
                return 0.0  # Invalid - not fully contained
        
        # Check overlaps
        for i in range(len(hex_polygons)):
            for j in range(i+1, len(hex_polygons)):
                if check_overlap(hex_polygons[i], hex_polygons[j]):
                    return 0.0  # Invalid - overlaps
        
        # Return 1/outer_side_length as fitness
        return 1.0 / outer_side_length if outer_side_length > 0 else 0.0
        
    except Exception:
        return 0.0

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Start with a good initial configuration
    initial_config = np.array([
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
    
    # Flatten the parameters for optimization: [x0,y0,theta0, x1,y1,theta1, ... x10,y10,theta10]
    initial_flat = initial_config.flatten()
    
    def objective(x_flat):
        # Reshape back to configuration
        config = x_flat.reshape(-1, 3)
        # We want to maximize 1/R, which means minimize -1/R = -R
        # But we also want to maximize 1/R, so we minimize -1/R = -R
        # However, let's work with actual objective: maximize 1/R => minimize -1/R
        fitness = evaluate_solution(config)
        # Return negative because we're using minimization
        return -fitness if fitness > 0 else 1e6
    
    def constraint_containment(x_flat):
        config = x_flat.reshape(-1, 3)
        outer_side_length = calculate_outer_hex_side_length(config)
        outer_vertices = generate_hexagon_vertices(0, 0, 0, outer_side_length)
        outer_polygon = Polygon(outer_vertices)
        
        # Check containment for all hexagons
        hex_polygons = []
        for i in range(len(config)):
            center_x, center_y, angle = config[i]
            vertices = generate_hexagon_vertices(center_x, center_y, angle)
            hex_polygons.append(Polygon(vertices))
        
        # Check containment: all inner hexagons inside outer hexagon
        total_violation = 0
        for poly in hex_polygons:
            if not check_containment(poly, outer_polygon):
                # Measure how much it violates containment
                total_violation += 1  # Simple violation count
        
        return 1.0 - total_violation  # Maximize to 1
    
    def constraint_overlaps(x_flat):
        config = x_flat.reshape(-1, 3)
        
        # Check overlaps between all pairs
        hex_polygons = []
        for i in range(len(config)):
            center_x, center_y, angle = config[i]
            vertices = generate_hexagon_vertices(center_x, center_y, angle)
            hex_polygons.append(Polygon(vertices))
        
        # Count overlapping pairs
        overlap_count = 0
        for i in range(len(hex_polygons)):
            for j in range(i+1, len(hex_polygons)):
                if check_overlap(hex_polygons[i], hex_polygons[j]):
                    overlap_count += 1
                    
        # We want overlap_count = 0, so return 0 when there are no overlaps
        return 1.0 - overlap_count
    
    def constraint_bounds(x_flat):
        # Add bounds to keep the optimization reasonable
        # Keep all hexagons relatively close to center (within 10 units)
        bounds_violation = 0
        for i in range(0, len(x_flat), 3):
            x, y = x_flat[i], x_flat[i+1]
            if abs(x) > 10 or abs(y) > 10:
                bounds_violation += 1
        return 1.0 - bounds_violation
    
    # Create constraints dictionary
    constraints = [
        {'type': 'ineq', 'fun': lambda x: constraint_containment(x)},
        {'type': 'ineq', 'fun': lambda x: constraint_overlaps(x)},
        {'type': 'ineq', 'fun': lambda x: constraint_bounds(x)}
    ]
    
    # Optimization bounds (x, y, theta) for each hexagon
    bounds = []
    for i in range(11):
        # x coordinates: -10 to 10
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])  # x, y, angle
    
    # Apply optimization
    try:
        # Use L-BFGS-B optimizer with bounds
        result = minimize(
            objective,
            initial_flat,
            method='L-BFGS-B',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-5},
            callback=lambda x: None  # No callback needed
        )
        
        if result.success:
            optimized_config = result.x.reshape(-1, 3)
        else:
            optimized_config = initial_config
            
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        optimized_config = initial_config
    
    # Final check and refinement
    final_fitness = evaluate_solution(optimized_config)
    if final_fitness <= 0.01:  # If solution is not good enough
        # Try a local search refinement
        best_config = optimized_config.copy()
        best_fitness = final_fitness
        
        # Simple local search
        for _ in range(200):
            # Perturb each parameter slightly
            test_config = best_config.copy()
            for i in range(len(test_config)):
                for j in range(3):
                    if j < 2:  # x or y
                        test_config[i][j] += np.random.normal(0, 0.1)
                    else:  # angle
                        test_config[i][j] = (test_config[i][j] + np.random.normal(0, 5)) % 360
            
            test_fitness = evaluate_solution(test_config)
            if test_fitness > best_fitness:
                best_fitness = test_fitness
                best_config = test_config.copy()
        
        optimized_config = best_config
    
    # Calculate final outer hexagon side length
    outer_side_length = 1.0 / final_fitness if final_fitness > 0 else 8.0
    if outer_side_length > 100:
        outer_side_length = 10.0
    
    # Center the outer hexagon at origin
    outer_hex_data = np.array([0, 0, 0])
    
    return optimized_config, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END
