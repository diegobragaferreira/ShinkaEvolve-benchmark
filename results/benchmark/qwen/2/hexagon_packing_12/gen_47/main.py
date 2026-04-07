# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Point, Polygon
from numba import njit
import time
import math

# Numba-compiled functions for performance
@njit
def generate_hexagon_vertices(x, y, angle_degrees, side_length=1):
    """Generate vertices of a regular hexagon given center, rotation, and side length"""
    angle_rad = np.radians(angle_degrees)
    vertices = np.empty((6, 2))
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i, 0] = x + side_length * np.cos(theta)
        vertices[i, 1] = y + side_length * np.sin(theta)
    return vertices

@njit
def distance_point_to_hexagon(px, py, hx, hy, angle, side_length=1):
    """Calculate distance from point to hexagon boundary"""
    # Find closest point on hexagon boundary
    hex_vertices = generate_hexagon_vertices(hx, hy, angle, side_length)
    
    # Simple approach: return distance to center minus radius
    dx = px - hx
    dy = py - hy
    return np.sqrt(dx*dx + dy*dy) - side_length

@njit
def check_containment_simple(x, y, angle, outer_x, outer_y, outer_angle, outer_side_length):
    """Simple containment check using distance from center"""
    # Distance from center of inner hexagon to center of outer hexagon
    dx = x - outer_x
    dy = y - outer_y
    center_distance = np.sqrt(dx*dx + dy*dy)
    
    # Maximum distance from outer hexagon center to its boundary
    # For regular hexagon with side length S, max distance is S
    max_inner_dist = 1.0  # unit hexagon
    max_outer_dist = outer_side_length
    
    # Inner hexagon is contained if its center is within outer hexagon with margin
    return (center_distance + max_inner_dist) <= max_outer_dist

@njit
def check_overlap_simple(x1, y1, angle1, x2, y2, angle2):
    """Simple overlap check using distance between centers"""
    dx = x1 - x2
    dy = y1 - y2
    center_dist = np.sqrt(dx*dx + dy*dy)
    # Unit hexagons overlap if centers are closer than 2 units
    return center_dist < 2.0

@njit  
def compute_outer_hexagon_radius(inner_positions, inner_angles, outer_side_length):
    """Compute actual radius needed for outer hexagon to contain all inner hexagons"""
    max_dist_from_center = 0.0
    
    for i in range(len(inner_positions)):
        x, y = inner_positions[i]
        # Distance from origin to hexagon center
        dist = np.sqrt(x*x + y*y)
        # Add maximum distance from center to vertex (1 for unit hexagon)
        total_dist = dist + 1.0
        max_dist_from_center = max(max_dist_from_center, total_dist)
    
    return max_dist_from_center

def evaluate_symmetric_config(params):
    """Evaluate configuration using symmetric arrangement logic"""
    # params: [center_x, center_y, cluster_radius, cluster_angle_offset, 
    #          outer_side_length, cluster_count=12]
    
    center_x, center_y, cluster_radius, cluster_angle_offset, outer_side_length = params[:5]
    
    # Generate positions based on symmetry
    inner_positions = []
    inner_angles = []
    
    # Central hexagon
    inner_positions.append([center_x, center_y])
    inner_angles.append(0.0)
    
    # First ring (6 hexagons)
    ring_radius = cluster_radius
    for i in range(6):
        angle = cluster_angle_offset + i * 60.0
        rad = np.radians(angle)
        x = center_x + ring_radius * np.cos(rad)
        y = center_y + ring_radius * np.sin(rad)
        inner_positions.append([x, y])
        inner_angles.append(0.0)
    
    # Second ring (5 hexagons) 
    ring_radius = 2.0 * cluster_radius
    for i in range(5):
        angle = cluster_angle_offset + i * 72.0
        rad = np.radians(angle)
        x = center_x + ring_radius * np.cos(rad)
        y = center_y + ring_radius * np.sin(rad)
        inner_positions.append([x, y])
        inner_angles.append(0.0)
    
    # Bottom center hexagon
    inner_positions.append([center_x, center_y - 3.0 * cluster_radius])
    inner_angles.append(0.0)
    
    penalty = 0.0
    n = len(inner_positions)
    
    # Check containment
    for i in range(n):
        x, y = inner_positions[i]
        if not check_containment_simple(x, y, inner_angles[i], 0, 0, 0, outer_side_length):
            penalty += 1e8
    
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = inner_positions[i]
            x2, y2 = inner_positions[j]
            if check_overlap_simple(x1, y1, inner_angles[i], x2, y2, inner_angles[j]):
                penalty += 1e8
    
    # Calculate inverse side length (negative for minimization)
    objective = -1.0 / outer_side_length + penalty
    
    return objective

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Initial guess for symmetric parameters
    initial_guess = [0.0, 0.0, 1.0, 0.0, 4.0]  # [center_x, center_y, cluster_radius, cluster_angle_offset, outer_side_length]
    
    # Bounds: center (0,0), cluster radius (0.5, 3.0), angle offset (0, 360), outer side length (2.0, 10.0)
    bounds = [(-0.5, 0.5), (-0.5, 0.5), (0.5, 3.0), (0.0, 360.0), (2.0, 10.0)]
    
    # Global optimization with multiple restarts
    best_result = None
    best_score = float('inf')
    
    # Try multiple starting points for better exploration
    for _ in range(5):
        try:
            # Randomize initial values slightly
            rand_params = [
                np.random.uniform(bounds[0][0], bounds[0][1]),
                np.random.uniform(bounds[1][0], bounds[1][1]),
                np.random.uniform(bounds[2][0], bounds[2][1]),
                np.random.uniform(bounds[3][0], bounds[3][1]),
                np.random.uniform(bounds[4][0], bounds[4][1])
            ]
            
            # Global optimization
            result = differential_evolution(
                evaluate_symmetric_config,
                bounds,
                seed=None,
                maxiter=50,
                popsize=10,
                mutation=(0.5, 1),
                recombination=0.7,
                tol=1e-5,
                workers=1,
                init=[rand_params]
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result
                
        except Exception as e:
            continue
    
    # Final refinement using local optimization
    if best_result is not None:
        try:
            # Refinement using L-BFGS-B for fine adjustment
            result = minimize(
                evaluate_symmetric_config,
                best_result.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 30}
            )
            
            optimized_params = result.x
        except:
            optimized_params = best_result.x
    else:
        # Fallback to default configuration
        optimized_params = initial_guess
    
    # Generate final hexagon positions
    center_x, center_y, cluster_radius, cluster_angle_offset, outer_side_length = optimized_params
    
    inner_positions = []
    inner_angles = []
    
    # Central hexagon
    inner_positions.append([center_x, center_y])
    inner_angles.append(0.0)
    
    # First ring (6 hexagons)
    ring_radius = cluster_radius
    for i in range(6):
        angle = cluster_angle_offset + i * 60.0
        rad = np.radians(angle)
        x = center_x + ring_radius * np.cos(rad)
        y = center_y + ring_radius * np.sin(rad)
        inner_positions.append([x, y])
        inner_angles.append(0.0)
    
    # Second ring (5 hexagons) 
    ring_radius = 2.0 * cluster_radius
    for i in range(5):
        angle = cluster_angle_offset + i * 72.0
        rad = np.radians(angle)
        x = center_x + ring_radius * np.cos(rad)
        y = center_y + ring_radius * np.sin(rad)
        inner_positions.append([x, y])
        inner_angles.append(0.0)
    
    # Bottom center hexagon
    inner_positions.append([center_x, center_y - 3.0 * cluster_radius])
    inner_angles.append(0.0)
    
    # Convert to required format
    inner_hex_data = []
    for i in range(12):
        inner_hex_data.append([inner_positions[i][0], inner_positions[i][1], inner_angles[i]])
    
    inner_hex_data = np.array(inner_hex_data)
    outer_hex_data = np.array([0, 0, 0])
    
    # Ensure we don't exceed time limits
    elapsed_time = time.time() - start_time
    if elapsed_time > 175:
        print("Warning: Approaching time limit")
    
    return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END
