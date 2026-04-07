# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
import math
import time
from copy import deepcopy

def hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, angle, and side length"""
    angle_rad = np.radians(angle_deg)
    angles = np.linspace(0, 2*np.pi, 7) + angle_rad
    vertices = np.array([
        [center_x + side_length * np.cos(a), center_y + side_length * np.sin(a)]
        for a in angles
    ])
    return vertices

def check_containment(hex_vertices, outer_hex_vertices):
    """Check if hexagon vertices are contained within outer hexagon using Shapely"""
    inner_polygon = Polygon(hex_vertices)
    outer_polygon = Polygon(outer_hex_vertices)
    return outer_polygon.contains(inner_polygon)

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely"""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def compute_outer_hexagon_radius(inner_positions, inner_angles, initial_radius_estimate=5.0):
    """Compute minimum outer hexagon radius that contains all inner hexagons"""
    # Direct computation of bounding circle approach
    all_points = []
    for i, (pos, angle) in enumerate(zip(inner_positions, inner_angles)):
        hex_verts = hexagon_vertices(pos[0], pos[1], angle)
        all_points.extend(hex_verts.tolist())
    
    if not all_points:
        return initial_radius_estimate
    
    # Find centroid of all vertices
    all_array = np.array(all_points)
    centroid = np.mean(all_array, axis=0)
    
    # Compute maximum distance from centroid to any vertex
    max_dist = 0
    for point in all_points:
        dist = np.sqrt((point[0] - centroid[0])**2 + (point[1] - centroid[1])**2)
        max_dist = max(max_dist, dist)
    
    return max_dist + 0.1  # Add small padding

def evaluate_geometric_constraints(inner_positions, inner_angles, outer_radius):
    """Evaluate geometric constraints more efficiently"""
    outer_vertices = hexagon_vertices(0, 0, 0, outer_radius)
    outer_polygon = Polygon(outer_vertices)
    
    # Check containment for all inner hexagons - vectorized
    containment_violations = 0
    overlap_violations = 0
    
    for i, (pos, angle) in enumerate(zip(inner_positions, inner_angles)):
        hex_vertices = hexagon_vertices(pos[0], pos[1], angle)
        hex_polygon = Polygon(hex_vertices)
        
        if not outer_polygon.contains(hex_polygon):
            containment_violations += 1
            
    # Check overlaps - pairwise comparison  
    for i in range(len(inner_positions)):
        for j in range(i+1, len(inner_positions)):
            hex1_vertices = hexagon_vertices(inner_positions[i][0], inner_positions[i][1], inner_angles[i])
            hex2_vertices = hexagon_vertices(inner_positions[j][0], inner_positions[j][1], inner_angles[j])
            
            if check_overlap(hex1_vertices, hex2_vertices):
                overlap_violations += 1
                
    return containment_violations, overlap_violations

def geometric_objective_with_constraints(params, fixed_positions=None, fixed_angles=None):
    """
    Geometric objective function with penalty for constraints
    params: flattened array of [x1, y1, angle1, x2, y2, angle2, ...] for 11 hexagons
    """
    # Reshape parameters
    positions_angles = params.reshape(-1, 3)
    
    # If fixed positions/angles provided, use them
    if fixed_positions is not None and fixed_angles is not None:
        positions_angles = positions_angles.copy()
        for i in range(len(fixed_positions)):
            positions_angles[i][:2] = fixed_positions[i]
            positions_angles[i][2] = fixed_angles[i]
    
    # Extract positions and angles
    inner_positions = positions_angles[:, :2]
    inner_angles = positions_angles[:, 2]
    
    # Compute outer radius
    outer_radius = compute_outer_hexagon_radius(inner_positions, inner_angles)
    
    # Evaluate constraints
    containment_violations, overlap_violations = evaluate_geometric_constraints(
        inner_positions, inner_angles, outer_radius)
    
    # Penalty terms
    penalty = 0
    if containment_violations > 0:
        penalty += containment_violations * 10000
    if overlap_violations > 0:
        penalty += overlap_violations * 10000
    
    # Objective: maximize 1/outer_radius (minimize outer_radius)
    # Add penalty to objective
    objective_value = -(1.0 / outer_radius) - penalty
    
    return objective_value

def create_initial_geometric_layout():
    """Create a carefully designed initial geometric configuration"""
    # Use a honeycomb-like arrangement that is known to work well
    # Place in a pattern that forms a compact cluster
    layout = [
        # Central hexagon
        [0.0, 0.0, 0.0],
        # First ring - 6 hexagons around center
        [-1.732, 0.0, 0.0],      # left
        [1.732, 0.0, 0.0],       # right
        [-0.866, 1.5, 0.0],      # top-left
        [0.866, 1.5, 0.0],       # top-right
        [-0.866, -1.5, 0.0],     # bottom-left
        [0.866, -1.5, 0.0],      # bottom-right
        # Second ring - 6 hexagons
        [-2.598, 1.5, 0.0],      # far top-left
        [2.598, 1.5, 0.0],       # far top-right
        [-2.598, -1.5, 0.0],     # far bottom-left
        [2.598, -1.5, 0.0],      # far bottom-right
        [0.0, 3.0, 0.0],         # top-center
    ]
    
    # Add small random perturbations to break symmetry and find better local optima
    for i in range(len(layout)):
        layout[i][0] += np.random.uniform(-0.1, 0.1)
        layout[i][1] += np.random.uniform(-0.1, 0.1)
        layout[i][2] += np.random.uniform(-5, 5)
    
    return np.array(layout)

def geometric_programming_optimization():
    """Use geometric programming approach with mathematical optimization"""
    
    # Start with good initial configuration
    initial_layout = create_initial_geometric_layout()
    initial_params = initial_layout.flatten()
    
    # Bounds for each parameter (x, y, angle for each of 11 hexagons)
    bounds = []
    for i in range(11):
        # x coordinates: roughly within [-10, 10] for stability
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])
    
    # Optimization parameters
    max_iterations = 200
    ftol = 1e-8
    gtol = 1e-8
    
    # Run optimization with various algorithms
    best_result = None
    best_value = float('-inf')
    
    # Try multiple optimization strategies
    strategies = [
        {'method': 'L-BFGS-B', 'options': {'maxiter': 100, 'ftol': ftol, 'gtol': gtol}},
        {'method': 'TNC', 'options': {'maxiter': 100, 'ftol': ftol, 'gtol': gtol}},
    ]
    
    # Global optimization with geometric constraints
    for strategy in strategies:
        try:
            # Run optimization
            result = minimize(
                geometric_objective_with_constraints,
                initial_params,
                method=strategy['method'],
                bounds=bounds,
                options=strategy['options'],
                callback=lambda x: None  # No callback to save time
            )
            
            # Evaluate final result
            final_value = geometric_objective_with_constraints(result.x)
            
            if final_value > best_value:
                best_value = final_value
                best_result = result
                
        except Exception as e:
            continue
    
    # If no good result found, return original layout
    if best_result is None:
        final_positions_angles = initial_layout
    else:
        final_positions_angles = best_result.x.reshape(-1, 3)
    
    # Final validation
    positions = final_positions_angles[:, :2]
    angles = final_positions_angles[:, 2]
    outer_radius = compute_outer_hexagon_radius(positions, angles)
    
    # Refine once more with local search if needed
    if outer_radius > 5.0:  # Only refine if solution isn't already very good
        # Do one final optimization with tighter constraints
        bounds_tight = [(pos[0]-1, pos[0]+1) for pos in positions] + \
                      [(pos[1]-1, pos[1]+1) for pos in positions] + \
                      [(a-10, a+10) for a in angles]
        
        for i in range(3):
            try:
                result = minimize(
                    geometric_objective_with_constraints,
                    final_positions_angles.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-9, 'gtol': 1e-9}
                )
                final_positions_angles = result.x.reshape(-1, 3)
                positions = final_positions_angles[:, :2]
                angles = final_positions_angles[:, 2]
                outer_radius = compute_outer_hexagon_radius(positions, angles)
                break
            except:
                pass
    
    return final_positions_angles, outer_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Run geometric programming optimization
    inner_hex_data, outer_hex_side_length = geometric_programming_optimization()

    # Format output as required
    outer_hex_data = np.array([0, 0, 0])  # centered at origin

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END