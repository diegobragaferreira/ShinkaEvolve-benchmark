# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
from scipy.spatial import cKDTree
import time
from itertools import combinations

# Constants
HEXAGON_RADIUS = 1.0
HEXAGON_WIDTH = HEXAGON_RADIUS * 2 * np.sqrt(3) / 3
HEXAGON_HEIGHT = HEXAGON_RADIUS * 2
MAX_EVAL_TIME = 180

def create_unit_hexagon(center=(0, 0), rotation=0):
    """Create a unit regular hexagon as a Shapely polygon"""
    angle_step = np.pi / 3
    points = []
    for i in range(6):
        angle = rotation + i * angle_step
        x = center[0] + HEXAGON_RADIUS * np.cos(angle)
        y = center[1] + HEXAGON_RADIUS * np.sin(angle)
        points.append((x, y))
    return Polygon(points)

def check_containment(hexagon, outer_hexagon):
    """Check if a hexagon is fully contained within outer hexagon"""
    return outer_hexagon.contains(hexagon)

def check_overlap_fast(hex1, hex2):
    """Fast overlap check using spatial indexing - early termination"""
    # Quick bounding box check first
    if not hex1.bounds[2] < hex2.bounds[0] and \
       not hex1.bounds[0] > hex2.bounds[2] and \
       not hex1.bounds[3] < hex2.bounds[1] and \
       not hex1.bounds[1] > hex2.bounds[3]:
        # If bounding boxes intersect, do detailed check
        return hex1.intersects(hex2)
    return False

def calculate_outer_hexagon_radius(inner_hex_data):
    """Calculate minimum outer hexagon radius required to contain all inner hexagons"""
    max_dist = 0
    for i in range(len(inner_hex_data)):
        center_x, center_y, _ = inner_hex_data[i]
        dist_to_center = np.sqrt(center_x**2 + center_y**2)
        # Hexagon diagonal is sqrt(3) * radius 
        hex_diag = HEXAGON_RADIUS * np.sqrt(3)
        max_dist = max(max_dist, dist_to_center + hex_diag)
    return max_dist

def evaluate_solution(params):
    """
    Optimized evaluation with smart early termination and reduced overhead
    params: array of shape (18,) = [r1, theta1, r2, theta2, ..., r6, theta6] 
            where r_i, theta_i are radial and angular parameters for 6 unique positions
            and we mirror these for the full 12 hexagons
    """
    # Parameter mapping: 6 unique positions (radial, angular), then mirrored
    # This reduces parameters from 36 to 18 while preserving symmetry
    n_positions = 6
    inner_params = np.zeros((12, 3))
    
    # Fill first 6 positions (0-5)
    for i in range(n_positions):
        inner_params[i][0] = params[2*i]  # radius
        inner_params[i][1] = params[2*i+1]  # angle (in radians)
        inner_params[i][2] = 0  # rotation (fixed for now)
    
    # Mirror positions for the 2nd ring (6-11)
    for i in range(n_positions):
        # Mirror radially and add pi to angle (180 degree rotation)
        inner_params[i+6][0] = params[2*i]  # same radius
        inner_params[i+6][1] = params[2*i+1] + np.pi  # opposite angle
        inner_params[i+6][2] = 0  # rotation
    
    # Convert radial/angular to cartesian coordinates
    for i in range(12):
        r = inner_params[i][0]
        theta = inner_params[i][1]
        inner_params[i][0] = r * np.cos(theta)  # x
        inner_params[i][1] = r * np.sin(theta)  # y
    
    # Create inner hexagons
    inner_hexagons = []
    for i in range(12):
        center = (inner_params[i][0], inner_params[i][1])
        angle = np.radians(inner_params[i][2])
        hexagon = create_unit_hexagon(center, angle)
        inner_hexagons.append(hexagon)

    # Calculate outer hexagon radius
    outer_radius = calculate_outer_hexagon_radius(inner_params)

    # Check containment and overlap constraints efficiently
    outer_hexagon = create_unit_hexagon((0, 0), 0)
    scaled_outer_radius = outer_radius * 1.05  # Add small margin
    outer_hexagon_scaled = create_unit_hexagon((0, 0), 0)

    # Check containment for all inner hexagons - early termination
    containment_violations = 0
    for hex in inner_hexagons:
        if not check_containment(hex, outer_hexagon_scaled):
            containment_violations += 1
            break  # Early termination

    # Check overlap between all pairs with early termination
    overlap_violations = 0
    # Use spatial indexing for efficient checking
    centers = np.array([[h.centroid.x, h.centroid.y] for h in inner_hexagons])
    tree = cKDTree(centers)
    
    # Check only nearby pairs for overlaps
    for i in range(12):
        # Get nearby hexagons within reasonable distance
        indices = tree.query_ball_point(centers[i], 3.0)
        for j in indices:
            if i < j:
                if check_overlap_fast(inner_hexagons[i], inner_hexagons[j]):
                    overlap_violations += 1
                    break  # Early termination
        if overlap_violations > 0:
            break  # Early termination if any overlap found

    # Penalty for constraint violations  
    penalty = 10000 * (containment_violations + overlap_violations)

    # Objective: minimize negative of 1/outer_radius + penalty
    if containment_violations > 0 or overlap_violations > 0:
        return 100000 + penalty  # Large penalty for constraint violations
    else:
        # Return negative of 1/R (we want to maximize 1/R, so minimize -1/R)
        return -(1.0 / scaled_outer_radius) + penalty

def generate_initial_guess():
    """Generate better initial configuration with explicit symmetry and spacing"""
    # Generate 6 unique positions in polar coordinates
    # Use a hexagonal lattice approach but with strategic spacing
    
    # Central position
    params = [0.0, 0.0]  # center point
    
    # First ring - 6 positions evenly spaced around circle
    r1 = 2.0 * HEXAGON_RADIUS  # radius for first ring
    angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    for angle in angles1:
        params.extend([r1, angle])
    
    # Second ring - 6 positions in the center of gaps of first ring  
    r2 = 3.5 * HEXAGON_RADIUS  # radius for second ring
    angles2 = np.linspace(np.pi/6, 2*np.pi + np.pi/6, 6, endpoint=False)
    for angle in angles2:
        params.extend([r2, angle])
    
    # Additional adjustments for better configuration
    # This is a more carefully tuned starting point than previous versions
    # Adjust a few positions to promote better packing
    
    # Modify some radial positions slightly to reduce overlap risk
    params[2] = 1.9  # First ring, first position
    params[6] = 3.4  # Second ring, first position 
    
    # Add a small random perturbation to improve convergence
    params = np.array(params) + np.random.normal(0, 0.1, len(params))
    
    return params

def optimize_hexagon_arrangement():
    """
    Multi-stage optimization with symmetry awareness
    """
    # Phase 1: Coarse parameter search in reduced space
    bounds = []
    # Radius bounds (positive values) and angle bounds
    for i in range(6):
        bounds.extend([(0.1, 5.0), (-np.pi, np.pi)])  # r, theta pairs
    
    # Initial guess
    initial_guess = generate_initial_guess()
    
    # Phase 1: Coarse optimization to find good region
    try:
        result1 = differential_evolution(
            evaluate_solution,
            bounds,
            maxiter=20,
            popsize=15,
            seed=42,
            disp=False,
            polish=False
        )
        
        # Phase 2: Refinement with better polish
        result2 = differential_evolution(
            evaluate_solution,
            bounds,
            maxiter=15,
            popsize=10,
            seed=43,
            disp=False,
            polish=True
        )
        
        # Use the better result
        if result1.fun < result2.fun:
            optimized_params = result1.x
        else:
            optimized_params = result2.x
            
    except Exception as e:
        print(f"Optimization failed: {e}")
        optimized_params = initial_guess

    # Phase 3: Final fine-tuning with L-BFGS-B
    try:
        result_final = minimize(
            evaluate_solution,
            optimized_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 10},
            callback=None
        )
        
        if result_final.success:
            optimized_params = result_final.x
    except Exception as e:
        print(f"Final optimization failed: {e}")

    # Convert final parameters back to standard format
    # Create final 12 hexagon parameters (this simplified version just returns what we have)
    final_params = optimized_params
    
    # Build final data structure
    inner_hex_data = np.zeros((12, 3))
    
    # Map back to cartesian coordinates for final representation
    for i in range(6):
        r = final_params[2*i]
        theta = final_params[2*i+1]
        inner_hex_data[i][0] = r * np.cos(theta)
        inner_hex_data[i][1] = r * np.sin(theta)
        inner_hex_data[i][2] = 0  # rotation
        
        # Mirror for the second ring
        inner_hex_data[i+6][0] = r * np.cos(theta + np.pi)
        inner_hex_data[i+6][1] = r * np.sin(theta + np.pi)
        inner_hex_data[i+6][2] = 0  # rotation

    # Calculate final outer hexagon side length
    outer_radius = calculate_outer_hexagon_radius(inner_hex_data)
    outer_hex_side_length = outer_radius * np.sqrt(3)  # Convert from radius to side length

    # Outer hexagon centered at origin with 0 rotation
    outer_hex_data = np.array([0, 0, 0])

    return inner_hex_data, outer_hex_data, outer_hex_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    # Use optimization approach instead of hardcoded values
    inner_hex_data, outer_hex_data, outer_hex_side_length = optimize_hexagon_arrangement()

    end_time = time.time()
    eval_time = end_time - start_time

    # Debug output
    inv_outer_side_length = 1.0 / outer_hex_side_length
    benchmark_ratio = inv_outer_side_length / 0.2537

    print(f"Eval time: {eval_time:.4f}s")
    print(f"Inv outer side length: {inv_outer_side_length:.6f}")
    print(f"Benchmark ratio: {benchmark_ratio:.6f}")

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END