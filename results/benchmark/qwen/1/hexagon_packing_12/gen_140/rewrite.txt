# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon, Point
import time

def create_hexagon_vertices(center, side_length, rotation_degrees):
    """Create vertices of a regular hexagon."""
    angle_rad = np.radians(rotation_degrees)
    angle_step = 2 * np.pi / 6
    vertices = []
    for i in range(6):
        angle = angle_step * i + angle_rad
        x = center[0] + side_length * np.cos(angle)
        y = center[1] + side_length * np.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

def check_containment_all_vertices(hex_vertices, outer_hex_center, outer_hex_side_length):
    """Check if all vertices of a hexagon are inside the outer hexagon."""
    outer_vertices = create_hexagon_vertices(outer_hex_center, outer_hex_side_length, 0)
    outer_polygon = Polygon(outer_vertices)
    
    for vertex in hex_vertices:
        point = Point(vertex)
        if not outer_polygon.contains(point):
            return False
    return True

def check_overlap_pair(hex1_vertices, hex2_vertices):
    """Fast overlap check using Shapely."""
    hex1_polygon = Polygon(hex1_vertices)
    hex2_polygon = Polygon(hex2_vertices)
    return hex1_polygon.intersects(hex2_polygon)

def compute_outer_hex_side_from_config(inner_hex_data, center=(0,0)):
    """Compute the minimum required outer hexagon side length from current configuration."""
    if len(inner_hex_data) == 0:
        return 100
    
    max_dist = 0
    for i in range(len(inner_hex_data)):
        cx, cy, _ = inner_hex_data[i]
        dist = np.sqrt((cx - center[0])**2 + (cy - center[1])**2)
        # Add the circumradius of inner hexagon (1 for unit hexagon)
        dist_to_edge = dist + 1.0
        max_dist = max(max_dist, dist_to_edge)
    
    return max_dist * 2.0

def evaluate_configuration(inner_hex_data, outer_hex_center=(0,0)):
    """Evaluate a configuration for validity and return inverse side length."""
    if len(inner_hex_data) != 12:
        return 1e-10
        
    # Create all hexagon polygons
    hex_polygons = []
    for i in range(len(inner_hex_data)):
        cx, cy, angle = inner_hex_data[i]
        vertices = create_hexagon_vertices((cx, cy), 1.0, angle)
        hex_polygons.append(Polygon(vertices))
    
    # Check containment: all hexagon vertices must be within outer hexagon
    outer_side_length = compute_outer_hex_side_from_config(inner_hex_data, outer_hex_center)
    outer_vertices = create_hexagon_vertices(outer_hex_center, outer_side_length, 0)
    outer_polygon = Polygon(outer_vertices)
    
    # Check containment for all vertices - optimized check
    for i in range(len(inner_hex_data)):
        vertices = hex_polygons[i].exterior.coords[:-1]  # Exclude last point (duplicate of first)
        for vertex in vertices:
            point = Point(vertex)
            if not outer_polygon.contains(point):
                return 1e-10
    
    # Check overlaps between all pairs of hexagons with early termination
    for i in range(len(inner_hex_data)):
        for j in range(i+1, len(inner_hex_data)):
            if hex_polygons[i].intersects(hex_polygons[j]):
                return 1e-10
    
    # If we reach here, the configuration is valid
    return 1.0 / outer_side_length

def generate_initial_placement():
    """Generate an initial placement with enhanced mathematical insight."""
    # Start with a more sophisticated arrangement inspired by lattice packings
    positions = []
    
    # Central hexagon
    positions.append([0, 0, 0])
    
    # First ring - 6 hexagons around center in a hexagonal pattern
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 directions
    radius = 2.0
    
    for angle in angles:
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        positions.append([x, y, 0])
    
    # Second ring - 6 hexagons arranged in triangular pattern
    angles2 = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 directions
    radius2 = 3.5
    
    for angle in angles2:
        x = radius2 * np.cos(angle)
        y = radius2 * np.sin(angle)
        positions.append([x, y, 0])
    
    # Adjust to make sure we have exactly 12
    positions = positions[:12]
    
    # Convert to array format
    config = np.array(positions)
    
    # Add controlled randomness for better exploration
    np.random.seed(42)
    config[:, 0] += np.random.normal(0, 0.15, 12)
    config[:, 1] += np.random.normal(0, 0.15, 12)
    
    return config

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Start with a sophisticated initial configuration
    initial_guess = generate_initial_placement()
    
    # Define bounds for optimization:
    # [x1, y1, angle1, x2, y2, angle2, ..., x12, y12, angle12]
    bounds = []
    # Positions: -10 to 10 for both x and y
    for _ in range(12):
        bounds.extend([(-10, 10), (-10, 10)])
    # Angles: 0 to 360 degrees
    for _ in range(12):
        bounds.append((0, 360))
    
    def objective(x):
        # Reshape the flat vector back to 12 hexagons
        hex_data = x.reshape(-1, 3)
        # Try to optimize for the best configuration
        score = evaluate_configuration(hex_data)
        return -score  # Negative because we want to maximize
    
    # Stage 1: Global optimization with differential evolution
    try:
        start_time = time.time()
        
        # Run differential evolution for robust global search
        result_de = differential_evolution(
            objective, 
            bounds, 
            maxiter=150, 
            popsize=20,
            seed=42,
            strategy='best1bin'
        )
        
        # Extract intermediate result
        intermediate_result = result_de.x.reshape(-1, 3)
        intermediate_score = evaluate_configuration(intermediate_result)
        
        # Stage 2: Local refinement with L-BFGS if DE found a good solution
        if intermediate_score > 0.2 and result_de.success:
            # Flatten for optimization
            x0_refine = intermediate_result.flatten()
            
            # Local optimization bounds
            refine_bounds = []
            for i in range(24):
                refine_bounds.append((-10, 10))
            for i in range(12):
                refine_bounds.append((0, 360))
            
            # Local refinement with L-BFGS
            result_lbfgs = minimize(
                objective, 
                x0_refine, 
                method='L-BFGS-B', 
                bounds=refine_bounds, 
                options={'maxiter': 500}
            )
            
            # Extract final optimized values
            optimized_hex_data = result_lbfgs.x.reshape(-1, 3)
            final_score = evaluate_configuration(optimized_hex_data)
            
            if result_lbfgs.success and final_score > 1e-5:
                # Compute the outer hexagon parameters
                outer_side_length = 1.0 / final_score  
                outer_hex_center = (0, 0)
                
                # Create outer hexagon data (centered at origin, no rotation)
                outer_hex_data = np.array([0, 0, 0])
                
                return optimized_hex_data, outer_hex_data, outer_side_length
        
        # If L-BFGS didn't work, use DE result directly
        if intermediate_score > 1e-5:
            outer_side_length = 1.0 / intermediate_score  
            outer_hex_center = (0, 0)
            outer_hex_data = np.array([0, 0, 0])
            return intermediate_result, outer_hex_data, outer_side_length
        
    except Exception as e:
        pass
    
    # Fallback to the more symmetric configuration from the minimization approach
    inner_hex_data = np.array([
        [0, 0, 0],      # center
        [0, 2, 0],      # top
        [0, -2, 0],     # bottom
        [1.732, 1, 0],  # top right
        [-1.732, 1, 0], # top left
        [1.732, -1, 0], # bottom right
        [-1.732, -1, 0],# bottom left
        [3.464, 0, 0],  # far right
        [-3.464, 0, 0], # far left
        [1.732, 3, 0],  # top far right
        [-1.732, 3, 0], # top far left
        [1.732, -3, 0], # bottom far right
    ])
    
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 6.928  # approximated value
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END