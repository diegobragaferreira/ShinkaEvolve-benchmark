# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon, Point
import time
from scipy.spatial.distance import cdist

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

def compute_outer_hex_side_from_config(config):
    """Compute the minimum outer hexagon side length needed to contain all inner hexagons."""
    # Extract positions and rotations
    positions_angles = config.reshape(-1, 3)
    
    # Create inner hexagons to find bounding box
    inner_hexagons = []
    for i in range(12):
        x, y, angle = positions_angles[i]
        inner_hex = create_unit_hexagon((x, y), angle)
        inner_hexagons.append(inner_hex)
    
    # Find extreme points
    all_points = []
    for hexagon in inner_hexagons:
        for point in list(hexagon.exterior.coords):
            all_points.append(point)
    
    all_points = np.array(all_points)
    
    # Compute distances from origin to all extreme points
    distances = np.sqrt(all_points[:, 0]**2 + all_points[:, 1]**2)
    
    # The outer hexagon needs to be large enough to contain the furthest vertex
    # For a regular hexagon, the distance from center to vertex is the side length
    return np.max(distances)

def evaluate_symmetry_configuration(config):
    """
    Evaluate a configuration using symmetry-aware representation.
    config: array of shape (14,) - [r1,theta1,r2,theta2,R] where r1,theta1 describe the first hexagon
                          and r2,theta2 describe the second hexagon in the first ring
    Returns negative inverse side length (to maximize inverse side length)
    """
    # Extract core parameters
    r1, theta1, r2, theta2, outer_radius = config
    
    # Generate symmetric configuration for 12 hexagons
    positions_angles = []
    
    # Central hexagon (always at origin)
    positions_angles.append([0.0, 0.0, 0.0])
    
    # First ring around center (6 hexagons)
    for i in range(6):
        angle = i * np.pi/3
        x = r1 * np.cos(angle + theta1 * np.pi/180)
        y = r1 * np.sin(angle + theta1 * np.pi/180)
        positions_angles.append([x, y, 0.0])
    
    # Second ring (6 hexagons) - rotated
    for i in range(6):
        angle = i * np.pi/3 + np.pi/6
        x = r2 * np.cos(angle + theta2 * np.pi/180)
        y = r2 * np.sin(angle + theta2 * np.pi/180)
        positions_angles.append([x, y, 0.0])
    
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

    # Check pairwise overlaps
    for i in range(12):
        for j in range(i+1, 12):
            if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                return 1e10  # Penalty for overlap

    # Return negative inverse side length (we want to maximize 1/R)
    return -1.0 / outer_radius

def evaluate_full_configuration(config):
    """
    Full evaluation using all 36 parameters - used for final refinement
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

    # Check pairwise overlaps
    for i in range(12):
        for j in range(i+1, 12):
            if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                return 1e10  # Penalty for overlap

    # Return negative inverse side length (we want to maximize 1/R)
    return -1.0 / outer_radius

def get_initial_symmetric_guess():
    """Get initial guess using symmetry approach that should be better than naive grid"""
    # Start with a known good symmetric configuration approach
    # This mimics the best-known packing patterns for 12 hexagons
    
    # Parameters derived from known good packing (approximate values)
    r1 = 1.9   # Distance from center for first ring hexagons
    theta1 = 0.0  # Rotation for first ring (0 degrees)
    r2 = 2.8   # Distance from center for second ring hexagons  
    theta2 = 30.0  # Rotation for second ring (30 degrees)
    initial_radius = 4.5  # Initial outer radius estimate
    
    return np.array([r1, theta1, r2, theta2, initial_radius])

def get_refined_initial_guess():
    """Get even better initial guess for final refinement"""
    # Better educated initial guess based on literature and heuristics
    # Try to place hexagons in pattern that avoids conflicts
    
    # Use a pattern that has worked well in previous attempts
    positions_angles = []
    
    # Central hexagon
    positions_angles.append([0.0, 0.0, 0.0])
    
    # First ring (6 hexagons)
    for i in range(6):
        angle = i * np.pi/3
        # Place them at distance slightly less than 2 (to allow some overlap in positioning)
        x = 1.9 * np.cos(angle)
        y = 1.9 * np.sin(angle)
        positions_angles.append([x, y, 0.0])
    
    # Second ring (6 hexagons) - offset by 30 degrees
    for i in range(6):
        angle = i * np.pi/3 + np.pi/6
        # Place them at distance slightly larger than 2.5
        x = 2.7 * np.cos(angle)
        y = 2.7 * np.sin(angle)
        positions_angles.append([x, y, 0.0])
    
    # Add initial outer radius
    initial_radius = 4.2  # Estimate
    
    # Flatten for optimization
    flat_config = np.array(positions_angles).flatten()
    flat_config = np.append(flat_config, initial_radius)
    
    return flat_config

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses symmetry-aware optimization for better efficiency and solution quality.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Phase 1: Symmetry-aware optimization (coarse-grained)
    # Use reduced parameter space for faster convergence
    bounds = [
        (1.0, 3.0),     # r1 - distance for first ring
        (-180, 180),    # theta1 - angle for first ring
        (2.0, 4.0),     # r2 - distance for second ring  
        (-180, 180),    # theta2 - angle for second ring
        (3.0, 8.0)      # outer_radius - min outer radius
    ]
    
    # Get initial symmetric guess
    initial_guess = get_initial_symmetric_guess()
    
    # Optimize using differential evolution on symmetry-reduced space
    try:
        result = differential_evolution(
            evaluate_symmetry_configuration,
            bounds,
            maxiter=50,
            popsize=10,
            seed=42,
            disp=False
        )
        
        # Extract the best symmetric configuration
        best_symmetric_config = result.x
        r1, theta1, r2, theta2, outer_radius = best_symmetric_config
        
        # Reconstruct full configuration from symmetry
        positions_angles = []
        
        # Central hexagon
        positions_angles.append([0.0, 0.0, 0.0])
        
        # First ring
        for i in range(6):
            angle = i * np.pi/3
            x = r1 * np.cos(angle + theta1 * np.pi/180)
            y = r1 * np.sin(angle + theta1 * np.pi/180)
            positions_angles.append([x, y, 0.0])
        
        # Second ring
        for i in range(6):
            angle = i * np.pi/3 + np.pi/6
            x = r2 * np.cos(angle + theta2 * np.pi/180)
            y = r2 * np.sin(angle + theta2 * np.pi/180)
            positions_angles.append([x, y, 0.0])
            
        # Convert to full configuration vector
        full_config = np.array(positions_angles).flatten()
        full_config = np.append(full_config, outer_radius)
        
    except Exception as e:
        print("Symmetric optimization failed:", str(e))
        # Fall back to full optimization
        bounds_full = []
        for _ in range(12):
            bounds_full.extend([(-10, 10), (-10, 10), (0, 360)])  # x, y, angle
        bounds_full.append((1.0, 20.0))  # outer_radius
        
        initial_guess = get_refined_initial_guess()
        full_config = initial_guess.copy()
    
    # Phase 2: Final refinement with full parameter space 
    # Use local optimization around the best symmetric solution
    try:
        # Run L-BFGS-B optimization on full configuration
        refined_result = minimize(
            evaluate_full_configuration,
            full_config,
            method='L-BFGS-B',
            bounds=[(-10, 10), (-10, 10), (0, 360)] * 12 + [(1.0, 20.0)],
            options={'maxiter': 50}
        )
        
        final_config = refined_result.x
    except Exception as e:
        print("Refinement failed:", str(e))
        final_config = full_config

    # Extract final results
    positions_angles = final_config[:-1].reshape(-1, 3)
    outer_hex_side_length = final_config[-1]

    # Convert back to required format
    inner_hex_data = positions_angles.copy()
    outer_hex_data = np.array([0, 0, 0])

    end_time = time.time()
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
