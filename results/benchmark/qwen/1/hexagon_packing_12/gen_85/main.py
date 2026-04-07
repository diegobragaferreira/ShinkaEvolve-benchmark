# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon, Point
from joblib import Parallel, delayed
import time


def create_unit_hexagon(center=(0,0), rotation=0):
    """Create a unit regular hexagon with given center and rotation."""
    angle = rotation * np.pi / 180
    # Vertices of a unit hexagon centered at origin
    hex_vertices = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = np.cos(theta)
        y = np.sin(theta)
        hex_vertices.append((x + center[0], y + center[1]))
    return Polygon(hex_vertices)

def check_containment(inner_hex, outer_hex):
    """Check if inner hexagon is fully contained within outer hexagon."""
    # Check if all vertices of inner hex are inside outer hex
    for point in list(inner_hex.exterior.coords):
        if not outer_hex.contains(Point(point)):
            return False
    return True

def check_overlap(hex1, hex2):
    """Check if two hexagons overlap."""
    return hex1.intersects(hex2)

def parallel_overlap_check(hexagons, start_idx, end_idx):
    """Parallel overlap checking for a subset of hexagon pairs."""
    overlaps = []
    for i in range(start_idx, end_idx):
        for j in range(i + 1, len(hexagons)):
            if check_overlap(hexagons[i], hexagons[j]):
                overlaps.append((i, j))
    return overlaps

def evaluate_configuration_parallel(config):
    """
    Evaluate a configuration with parallel constraint checking.
    config: array of shape (37,) - [x1,y1,theta1,...,x12,y12,theta12,R]
    Returns negative inverse side length (to maximize inverse side length)
    """
    # Extract parameters
    positions_angles = config[:-1].reshape(-1, 3)
    outer_radius = config[-1]
    
    # Create outer hexagon
    outer_hex = create_unit_hexagon((0, 0), 0)
    # Scale the outer hexagon to have side length = outer_radius
    scaled_outer_vertices = []
    for i in range(6):
        theta = i * np.pi / 3
        x = outer_radius * np.cos(theta)
        y = outer_radius * np.sin(theta)
        scaled_outer_vertices.append((x, y))
    outer_hex = Polygon(scaled_outer_vertices)

    # Create inner hexagons
    inner_hexagons = []
    for i in range(12):
        x, y, angle = positions_angles[i]
        inner_hex = create_unit_hexagon((x, y), angle)
        inner_hexagons.append(inner_hex)

        # Check containment early
        if not check_containment(inner_hex, outer_hex):
            return 1e10  # Penalty for violation

    # Check pairwise overlaps in parallel
    # Split work into chunks for parallel processing
    num_pairs = 12 * 11 // 2  # Number of unique pairs
    chunk_size = max(1, num_pairs // 4)  # Process 4 chunks
    
    # Use joblib for parallel overlap checking
    overlap_results = Parallel(n_jobs=-1)(
        delayed(parallel_overlap_check)(inner_hexagons, i*chunk_size, min((i+1)*chunk_size, len(inner_hexagons)))
        for i in range(4)
    )
    
    # Check if any overlaps were found
    for result in overlap_results:
        if result:
            return 1e10  # Penalty for overlap

    # Return negative inverse side length (we want to maximize 1/R)
    return -1.0 / outer_radius

def get_initial_guess_better():
    """Get a better initial guess based on known hexagon packing patterns"""
    # Start with a known dense configuration
    # Arrange in a hexagonal pattern with strategic positioning
    positions_angles = []
    
    # Central hexagon
    positions_angles.append([0.0, 0.0, 0.0])
    
    # First ring (6 hexagons)
    for i in range(6):
        angle = i * np.pi/3
        x = 2.0 * np.cos(angle)
        y = 2.0 * np.sin(angle)
        positions_angles.append([x, y, 0.0])
    
    # Second ring (6 hexagons) 
    for i in range(6):
        angle = i * np.pi/3 + np.pi/6
        x = 3.0 * np.cos(angle)
        y = 3.0 * np.sin(angle)
        positions_angles.append([x, y, 0.0])
        
    # Add reasonable starting outer radius
    initial_radius = 5.5

    # Flatten for optimization
    flat_config = np.array(positions_angles).flatten()
    flat_config = np.append(flat_config, initial_radius)

    return flat_config

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses an enhanced optimization approach.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Define bounds for optimization
    # Positions: x,y in [-10, 10], angles in [0, 360]
    # Outer radius should be reasonable
    bounds = []
    for _ in range(12):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])  # x, y, angle
    bounds.append((2.0, 15.0))  # outer_radius

    # Get initial configuration
    initial_guess = get_initial_guess_better()

    # Phase 1: Coarse global optimization with larger population
    start_time = time.time()

    # Use differential evolution for global optimization with increased population
    result = differential_evolution(
        evaluate_configuration_parallel,
        bounds,
        maxiter=150,
        popsize=25,  # Larger population for better exploration
        seed=42,
        disp=False,
        mutation=(0.5, 1.0),
        recombination=0.7,
        tol=1e-6
    )

    # Phase 2: Local refinement with L-BFGS-B if needed
    if result.fun < -0.25:  # If we haven't reached target yet, do local refinement
        # Refine using L-BFGS-B
        refined_result = minimize(
            evaluate_configuration_parallel,
            result.x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-9}
        )
        if refined_result.fun < result.fun:
            result = refined_result

    end_time = time.time()

    # Extract results
    final_config = result.x
    positions_angles = final_config[:-1].reshape(-1, 3)
    outer_hex_side_length = final_config[-1]

    # Convert back to required format
    # The inner hex data is positions_angles
    inner_hex_data = positions_angles.copy()

    # Outer hex is centered at origin
    outer_hex_data = np.array([0, 0, 0])

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
