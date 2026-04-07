# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
from shapely.geometry import Polygon
import math

# Constants
UNIT_HEXAGON_RADIUS = 1.0
UNIT_HEXAGON_APOGEE = np.sqrt(3)/2
UNIT_HEXAGON_VERTEX_ANGLE = np.pi/3
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

def compute_outer_hexagon_vertices(center=(0,0), side_length=1.0, rotation=0):
    """Create vertices of the outer hexagon."""
    vertices = []
    for i in range(6):
        angle = rotation + i * UNIT_HEXAGON_VERTEX_ANGLE
        x = center[0] + side_length * np.cos(angle)
        y = center[1] + side_length * np.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

def distance_point_to_hexagon(point, hex_vertices):
    """Calculate minimum distance from point to hexagon boundary."""
    p = Point(point)
    hex_poly = Polygon(hex_vertices)
    return hex_poly.distance(p)

def point_in_hexagon(point, hex_vertices):
    """Check if point is inside hexagon."""
    p = Point(point)
    hex_poly = Polygon(hex_vertices)
    return hex_poly.contains(p)

def hexagon_overlap_distance(hex1_vertices, hex2_vertices):
    """Calculate minimum distance between hexagons (negative if overlapping)."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    
    # Use buffer for robustness against floating point errors
    buf1 = poly1.buffer(1e-10)
    buf2 = poly2.buffer(1e-10)
    
    dist = buf1.distance(buf2)
    return -dist  # Negative if overlapping

def compute_outer_radius(inner_hex_data, outer_hex_side_length):
    """Compute maximum distance from center to any vertex of any inner hexagon."""
    max_dist = 0.0
    
    # Create outer hexagon vertices
    outer_vertices = compute_outer_hexagon_vertices((0,0), outer_hex_side_length)
    outer_polygon = Polygon(outer_vertices)
    
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        hex_vertices = create_unit_hexagon_vertices((x, y), np.radians(angle))
        
        # Check if this hexagon is contained within outer hexagon
        hex_polygon = Polygon(hex_vertices)
        if not outer_polygon.contains(hex_polygon):
            return float('inf')  # Invalid configuration
            
        # Find maximum vertex distance from center
        for vertex in hex_vertices:
            dist = np.sqrt(vertex[0]**2 + vertex[1]**2)
            max_dist = max(max_dist, dist)
            
    return max_dist

def evaluate_hexagon_config(config):
    """Evaluate a configuration and return penalty value."""
    # Reshape config to 12 hexagons with (x, y, angle)
    hex_data = config.reshape(-1, 3)
    
    # Extract outer side length (last parameter)
    outer_side_length = config[-1]
    
    # Check containment first - compute outer radius
    outer_radius = compute_outer_radius(hex_data, outer_side_length)
    
    # If outer radius exceeds outer hexagon size, penalize heavily
    if outer_radius > outer_side_length:
        return 1e10
    
    # Check pairwise overlaps
    penalty = 0.0
    n_hex = len(hex_data)
    
    # Precompute all hexagon vertices for efficiency
    hex_vertices_list = []
    for i in range(n_hex):
        x, y, angle = hex_data[i]
        hex_vertices = create_unit_hexagon_vertices((x, y), np.radians(angle))
        hex_vertices_list.append(hex_vertices)
    
    # Check overlaps
    for i in range(n_hex):
        for j in range(i+1, n_hex):
            dist = hexagon_overlap_distance(hex_vertices_list[i], hex_vertices_list[j])
            # Allow minimal overlap tolerance
            if dist < -1e-8:  # Overlapping
                penalty += abs(dist) * 1e6
            elif dist < 0.1:  # Very close but not overlapping
                penalty += abs(dist) * 1000
    
    return penalty

def construct_lattice_pattern(lattice_params):
    """Construct 12 hexagon positions from lattice parameters."""
    # lattice_params: [center_x, center_y, radius, angle_offset, ring_spacing]
    center_x, center_y, radius, angle_offset, ring_spacing = lattice_params
    
    # Generate positions using a radial lattice approach
    positions = []
    
    # Center hexagon
    positions.append([center_x, center_y, 0])
    
    # First ring - 6 hexagons
    for i in range(6):
        angle = angle_offset + i * PI_3
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        positions.append([x, y, 0])
    
    # Second ring - 5 hexagons
    for i in range(5):
        angle = angle_offset + i * PI_3
        x = center_x + ring_spacing * radius * np.cos(angle)
        y = center_y + ring_spacing * radius * np.sin(angle)
        positions.append([x, y, 0])
    
    # One more at bottom
    positions.append([center_x, center_y - ring_spacing * radius, 0])
    
    return np.array(positions).flatten()

def objective_function_lattice(config):
    """Objective function for lattice-based optimization."""
    # config is [center_x, center_y, radius, angle_offset, ring_spacing, outer_side_length]
    
    # Extract parameters
    center_x, center_y, radius, angle_offset, ring_spacing, outer_side_length = config
    
    # Ensure valid values
    if outer_side_length < 1.0:
        return 1e10
    
    # Construct positions
    hex_positions = construct_lattice_pattern(config[:-1])
    
    # Concatenate with outer side length
    full_config = np.concatenate([hex_positions, [outer_side_length]])
    
    # Evaluate
    penalty = evaluate_hexagon_config(full_config)
    
    if penalty > 1e9:
        return penalty
    
    # Return negative inverse of outer side length (to minimize)
    return -1.0 / outer_side_length

def optimize_lattice_hexagon_packing():
    """Optimize using lattice-based approach."""
    
    # Initial parameter guess for lattice
    # [center_x, center_y, radius, angle_offset, ring_spacing, outer_side_length]
    initial_guess = [0.0, 0.0, 1.0, 0.0, 2.0, 4.0]
    
    # Bounds for parameters
    bounds = [
        (-5.0, 5.0),    # center_x
        (-5.0, 5.0),    # center_y
        (0.5, 3.0),     # radius
        (0.0, 2*np.pi), # angle_offset
        (1.0, 3.0),     # ring_spacing
        (1.0, 10.0)     # outer_side_length
    ]
    
    # Use differential evolution for global optimization
    try:
        result = differential_evolution(
            objective_function_lattice,
            bounds,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42
        )
        
        if result.success:
            final_params = result.x
            # Reconstruct full configuration
            hex_positions = construct_lattice_pattern(final_params[:-1])
            full_config = np.concatenate([hex_positions, [final_params[-1]]])
            
            # Extract inner hex data
            inner_hex_data = full_config[:-1].reshape(-1, 3)
            outer_hex_data = np.array([0.0, 0.0, 0.0])
            outer_hex_side_length = final_params[-1]
            
            return inner_hex_data, outer_hex_data, outer_hex_side_length
            
    except Exception as e:
        pass
    
    # Fallback to standard optimization
    try:
        result = minimize(
            objective_function_lattice,
            initial_guess,
            method='L-BFGSB',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        if result.success:
            final_params = result.x
            hex_positions = construct_lattice_pattern(final_params[:-1])
            full_config = np.concatenate([hex_positions, [final_params[-1]]])
            
            inner_hex_data = full_config[:-1].reshape(-1, 3)
            outer_hex_data = np.array([0.0, 0.0, 0.0])
            outer_hex_side_length = final_params[-1]
            
            return inner_hex_data, outer_hex_data, outer_hex_side_length
    except Exception as e:
        pass
        
    # Final fallback
    return generate_fallback_config()

def generate_fallback_config():
    """Generate a fallback configuration when optimization fails."""
    inner_hex_data = np.array([
        [0, 0, 0],          # center
        [-2.5, 0, 0],       # left
        [2.5, 0, 0],        # right
        [-1.25, 2.17, 0],   # top-left
        [1.25, 2.17, 0],    # top-right
        [-1.25, -2.17, 0],  # bottom-left
        [1.25, -2.17, 0],   # bottom-right
        [-3.75, 2.17, 0],   # far top-left
        [3.75, 2.17, 0],    # far top-right
        [-3.75, -2.17, 0],  # far bottom-left
        [3.75, -2.17, 0],   # far bottom-right
        [0, -4, 0],         # far bottom-center
    ])

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 8  # large enough to contain all inner hexagons

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
        inner_hex_data, outer_hex_data, outer_hex_side_length = optimize_lattice_hexagon_packing()
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