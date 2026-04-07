# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import time

def generate_hexagon_vertices(center_x, center_y, side_length, angle_degrees):
    """Generate vertices of a regular hexagon given center, side length, and rotation."""
    angle_rad = np.radians(angle_degrees)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * np.pi / 3
        x = center_x + side_length * np.cos(angle)
        y = center_y + side_length * np.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

def check_hexagon_containment(hex_vertices, outer_hex_center, outer_hex_side_length):
    """Check if all vertices of a hexagon are within the outer hexagon."""
    outer_hex_vertices = generate_hexagon_vertices(outer_hex_center[0], outer_hex_center[1], outer_hex_side_length, 0)
    outer_polygon = Polygon(outer_hex_vertices)
    
    for vertex in hex_vertices:
        point = Point(vertex)
        if not outer_polygon.contains(point):
            return False
    return True

def check_hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def calculate_outer_hex_radius(inner_hex_data, margin_factor=1.1):
    """Calculate the minimal radius needed for the outer hexagon."""
    # Get all vertices of inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        hex_vertices = generate_hexagon_vertices(center_x, center_y, 1, angle)
        all_vertices.extend(hex_vertices)
    
    # Find bounding box
    all_vertices = np.array(all_vertices)
    min_x, max_x = np.min(all_vertices[:, 0]), np.max(all_vertices[:, 0])
    min_y, max_y = np.min(all_vertices[:, 1]), np.max(all_vertices[:, 1])
    
    # Calculate radius needed to contain all vertices with margin
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    max_dist = 0
    for vertex in all_vertices:
        dist = np.sqrt((vertex[0] - center_x)**2 + (vertex[1] - center_y)**2)
        max_dist = max(max_dist, dist)
    
    # Add margin and convert to side length
    margin = max_dist * margin_factor
    # Convert distance from center to side length of hexagon
    # For a regular hexagon, distance from center to corner = side length
    return margin

def evaluate_configuration(hex_data, outer_center=(0, 0)):
    """Evaluate a configuration and return penalty if constraints violated."""
    num_hexagons = len(hex_data)
    
    # Check containment and overlap
    penalty = 0
    
    # Calculate outer hexagon radius
    outer_radius = calculate_outer_hex_radius(hex_data)
    
    # Check containment
    for i in range(num_hexagons):
        center_x, center_y, angle = hex_data[i]
        hex_vertices = generate_hexagon_vertices(center_x, center_y, 1, angle)
        
        if not check_hexagon_containment(hex_vertices, outer_center, outer_radius):
            penalty += 1000  # Severe penalty for containment violations
    
    # Check overlaps
    for i in range(num_hexagons):
        for j in range(i + 1, num_hexagons):
            hex1_vertices = generate_hexagon_vertices(hex_data[i][0], hex_data[i][1], 1, hex_data[i][2])
            hex2_vertices = generate_hexagon_vertices(hex_data[j][0], hex_data[j][1], 1, hex_data[j][2])
            
            if check_hexagon_overlap(hex1_vertices, hex2_vertices):
                penalty += 1000  # Severe penalty for overlaps
    
    # Return negative inverse radius (since we want to maximize 1/R)
    return penalty - 1.0 / outer_radius if penalty == 0 else penalty + 1000

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Define bounds for optimization: [x, y, angle] for each hexagon 
    bounds = []
    # Bounds for center coordinates (-10, 10) to allow for wide search space
    for i in range(11):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])
    
    # Multiple initialization strategies
    initial_configs = []
    
    # Strategy 1: Spiral arrangement
    spiral_config = np.zeros((11, 3))
    spiral_config[0] = [0, 0, 0]  # center
    for i in range(1, 11):
        angle = (i - 1) * 360 / 10  # even distribution around circle
        radius = 1.5 + (i - 1) * 0.5  # increasing radius
        spiral_config[i] = [radius * np.cos(np.radians(angle)), 
                           radius * np.sin(np.radians(angle)), 0]
    initial_configs.append(spiral_config.flatten())
    
    # Strategy 2: Grid arrangement (from original)
    grid_config = np.array([
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
    initial_configs.append(grid_config.flatten())
    
    # Strategy 3: Concentric arrangement
    concentric_config = np.zeros((11, 3))
    concentric_config[0] = [0, 0, 0]  # center
    # First ring: 6 hexagons around center
    for i in range(1, 7):
        angle = (i - 1) * 60
        concentric_config[i] = [2 * np.cos(np.radians(angle)), 
                               2 * np.sin(np.radians(angle)), 0]
    # Second ring: 4 hexagons further out
    for i in range(7, 11):
        angle = (i - 7) * 90
        concentric_config[i] = [3 * np.cos(np.radians(angle)), 
                               3 * np.sin(np.radians(angle)), 0]
    initial_configs.append(concentric_config.flatten())
    
    best_result = None
    best_score = float('-inf')
    
    # Run optimization for each initial configuration
    for init_config in initial_configs:
        def objective(x):
            # Reshape x back into hexagon data
            hex_data = x.reshape(11, 3)
            return evaluate_configuration(hex_data)
        
        # Use differential evolution with bounds
        try:
            result = differential_evolution(objective, bounds, maxiter=500, popsize=15, 
                                          tol=1e-6, mutation=(0.5, 1), recombination=0.7,
                                          seed=42, disp=False, polish=True)
            
            if result.success:
                final_config = result.x.reshape(11, 3)
                score = evaluate_configuration(final_config)
                
                if score > best_score:
                    best_score = score
                    best_result = final_config.copy()
                    
        except Exception as e:
            continue
    
    # Final refinement with local search if needed
    if best_result is not None:
        # Refine using a local search approach
        def local_refinement(x):
            hex_data = x.reshape(11, 3)
            # Evaluate current state
            current_score = evaluate_configuration(hex_data)
            
            # Try small perturbations
            best_x = x.copy()
            best_score = current_score
            
            for i in range(11):
                for j in range(3):  # try perturbing x, y, and angle
                    old_value = hex_data[i, j]
                    # Try small positive and negative perturbations
                    for delta in [-0.1, 0.1]:
                        hex_data[i, j] = old_value + delta
                        new_score = evaluate_configuration(hex_data)
                        if new_score > best_score:
                            best_score = new_score
                            best_x = hex_data.copy().flatten()
                    # Revert
                    hex_data[i, j] = old_value
            
            return -best_score  # Minimize negative score
        
        # Apply local refinement
        try:
            refined_result = local_refinement(best_result.flatten())
            final_best = best_result.copy()
        except:
            final_best = best_result.copy()
    else:
        # Fall back to the original configuration if nothing works
        final_best = grid_config
    
    # Calculate final outer hexagon dimensions
    outer_radius = calculate_outer_hex_radius(final_best)
    
    # Create the return values
    inner_hex_data = final_best
    outer_hex_data = np.array([0, 0, 0])  # Centered at origin
    outer_hex_side_length = outer_radius
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
