# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
from shapely.geometry import Polygon
from shapely.ops import unary_union
from scipy.spatial import distance

# Constants
UNIT_HEXAGON_RADIUS = 1.0  # Circumradius of unit hexagon
UNIT_HEXAGON_APOGEE = np.sqrt(3)/2  # Apothem of unit hexagon
UNIT_HEXAGON_VERTEX_ANGLE = np.pi/3  # Angle between adjacent vertices
PI_3 = np.pi/3
SQRT_3 = np.sqrt(3)

def create_unit_hexagon_vertices(center=(0,0), rotation=0):
    """Create vertices of a unit regular hexagon centered at center with given rotation."""
    vertices = []
    for i in range(6):
        angle = rotation + i * UNIT_HEXAGON_VERTEX_ANGLE
        x = center[0] + UNIT_HEXAGON_RADIUS * np.cos(angle)
        y = center[1] + UNIT_HEXAGON_RADIUS * np.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

def check_hexagon_containment(inner_hex_vertices, outer_hex_vertices):
    """Check if all vertices of inner hexagon are within outer hexagon."""
    inner_polygon = Polygon(inner_hex_vertices)
    outer_polygon = Polygon(outer_hex_vertices)

    # Check if inner polygon is completely contained within outer polygon
    return outer_polygon.contains(inner_polygon)

def check_hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)

    # Return True if they overlap (intersection area > 0)
    return poly1.intersects(poly2)

def compute_outer_hexagon_vertices(center=(0,0), side_length=1.0, rotation=0):
    """Create vertices of the outer hexagon."""
    vertices = []
    for i in range(6):
        angle = rotation + i * UNIT_HEXAGON_VERTEX_ANGLE
        x = center[0] + side_length * np.cos(angle)
        y = center[1] + side_length * np.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

def compute_inner_hex_positions(config, outer_side_length):
    """Compute actual hexagon positions from configuration."""
    # Config should be flattened array of 12*(x,y,angle) = 36 values
    positions = config.reshape(-1, 3)

    # Ensure hexagons don't exceed outer boundary
    hex_vertices_list = []
    for i, (x, y, angle) in enumerate(positions):
        # Create hexagon vertices with given position and rotation
        hex_v = create_unit_hexagon_vertices((x, y), np.radians(angle))
        hex_vertices_list.append(hex_v)

    return hex_vertices_list

def evaluate_configuration(config, outer_side_length):
    """Evaluate a configuration of hexagon positions."""
    # Config should be flattened array of 12*(x,y,angle) = 36 values
    hex_vertices_list = compute_inner_hex_positions(config, outer_side_length)

    # Test containment
    outer_hex_vertices = compute_outer_hexagon_vertices((0,0), outer_side_length)

    # Check if all inner hexagons are contained
    for hex_v in hex_vertices_list:
        if not check_hexagon_containment(hex_v, outer_hex_vertices):
            return False

    # Check for overlaps
    n = len(hex_vertices_list)
    for i in range(n):
        for j in range(i+1, n):
            if check_hexagon_overlap(hex_vertices_list[i], hex_vertices_list[j]):
                return False

    return True

def objective_function(config):
    """Objective function to minimize (negative inverse of outer hexagon side length)."""
    # Extract outer side length (last value in config)
    outer_side_length = config[-1]

    # If outer side length is too small, penalize heavily
    if outer_side_length < 1.0:
        return 1e10

    # Check validity of configuration
    if not evaluate_configuration(config[:-1], outer_side_length):
        return 1e10

    # Return negative inverse (since we want to maximize 1/R)
    return -1.0 / outer_side_length

def generate_initial_symmetric_config():
    """Generate a symmetric initial configuration for 12 hexagons."""
    # More refined symmetric pattern
    config = []
    
    # Center hexagon
    config.extend([0.0, 0.0, 0.0])
    
    # First ring of 6 hexagons around center
    for i in range(6):
        angle = i * PI_3
        x = 1.0 * np.cos(angle)
        y = 1.0 * np.sin(angle)
        config.extend([x, y, 0.0])
        
    # Second ring of 5 hexagons (not perfectly symmetrical to allow optimization flexibility)
    for i in range(5):
        angle = i * PI_3
        x = 2.0 * np.cos(angle)
        y = 2.0 * np.sin(angle)
        config.extend([x, y, 0.0])
    
    # One more at the bottom
    config.extend([0.0, -2.0, 0.0])
    
    # Add outer side length parameter (this will be optimized)
    config.append(5.0)  # Initial guess for outer side length
    
    return np.array(config)

def optimize_hexagon_packing():
    """Optimize the 12 hexagon packing configuration."""
    # Start with symmetric configuration
    initial_config = generate_initial_symmetric_config()

    # Define bounds for optimization
    # Positions can vary widely but reasonable constraints 
    # (x,y) can range from -8 to 8, angles from 0 to 360
    bounds = []
    for _ in range(12):
        bounds.extend([(-8.0, 8.0), (-8.0, 8.0), (0.0, 360.0)])
    bounds.append((1.0, 10.0))  # Outer side length should be positive

    # Optimization options
    options = {'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-10}

    def callback_func(x):
        # Optional: print progress
        pass

    # Run optimization
    try:
        result = minimize(
            objective_function,
            initial_config,
            method='L-BFGSB',
            bounds=bounds,
            options=options,
            callback=callback_func
        )

        if result.success:
            opt_config = result.x
            # Extract the final configuration
            final_positions = opt_config[:-1].reshape(-1, 3)
            final_side_length = opt_config[-1]

            # Return in the required format
            inner_hex_data = final_positions.copy()
            outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered

            return inner_hex_data, outer_hex_data, final_side_length
        else:
            raise Exception("Optimization failed")
    except Exception as e:
        # Fallback to the original configuration if optimization fails
        print(f"Optimization error: {e}")
        return generate_fallback_config()

def generate_fallback_config():
    """Generate a fallback configuration when optimization fails."""
    # Use a more refined initial configuration than before
    inner_hex_data = np.array([
        [0, 0, 0],          # center
        [-1.5, 0, 0],       # left
        [1.5, 0, 0],        # right
        [-0.75, 1.3, 0],    # top-left
        [0.75, 1.3, 0],     # top-right
        [-0.75, -1.3, 0],   # bottom-left
        [0.75, -1.3, 0],    # bottom-right
        [-2.25, 1.3, 0],    # far top-left
        [2.25, 1.3, 0],     # far top-right
        [-2.25, -1.3, 0],   # far bottom-left
        [2.25, -1.3, 0],    # far bottom-right
        [0, -2.6, 0],       # far bottom-center
    ])

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 6.0  # Reasonable starting point

    return inner_hex_data, outer_hex_data, outer_hex_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Time the optimization
    start_time = time.time()

    try:
        inner_hex_data, outer_hex_data, outer_hex_side_length = optimize_hexagon_packing()
    except Exception:
        # Fallback to old method if anything goes wrong
        inner_hex_data, outer_hex_data, outer_hex_side_length = generate_fallback_config()

    end_time = time.time()

    # Calculate performance metrics
    inv_outer_hex_side_length = 1.0 / outer_hex_side_length if outer_hex_side_length > 0 else 0.0
    benchmark_ratio = inv_outer_hex_side_length / 0.2537

    print(f"Optimized result: inverse_side_length={inv_outer_hex_side_length:.6f}, "
          f"benchmark_ratio={benchmark_ratio:.6f}, eval_time={(end_time-start_time):.3f}s")

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END