# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
from scipy.spatial import cKDTree
import time
from numba import jit
from collections import deque

@jit(nopython=True)
def hexagon_vertices(center_x, center_y, angle_rad, side_length=1.0):
    """Fast computation of hexagon vertices using Numba"""
    vertices = np.empty((6, 2))
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i, 0] = center_x + side_length * np.cos(theta)
        vertices[i, 1] = center_y + side_length * np.sin(theta)
    return vertices

def create_unit_hexagon(center=(0, 0), angle_deg=0):
    """Create a unit regular hexagon centered at center with rotation angle_deg."""
    angle_rad = np.deg2rad(angle_deg)
    # Vertices of unit hexagon centered at origin
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = np.cos(theta)
        y = np.sin(theta)
        vertices.append((x + center[0], y + center[1]))
    return Polygon(vertices)

def estimate_outer_hexagon_radius(positions, angles):
    """Better estimation of outer hexagon radius from positions"""
    # Get all vertices of all hexagons
    all_vertices = []
    for pos, angle in zip(positions, angles):
        vertices = hexagon_vertices(pos[0], pos[1], np.deg2rad(angle))
        all_vertices.extend(vertices)

    if len(all_vertices) == 0:
        return 10.0

    all_coords = np.array(all_vertices)
    min_x, max_x = all_coords[:, 0].min(), all_coords[:, 0].max()
    min_y, max_y = all_coords[:, 1].min(), all_coords[:, 1].max()

    # Calculate distance from center to bounding box corners
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Find maximum distance to any corner
    max_dist = 0
    for vx, vy in all_vertices:
        dist = np.sqrt((vx - center_x)**2 + (vy - center_y)**2)
        max_dist = max(max_dist, dist)

    # Add safety margin
    return max_dist * 1.05

def get_hexagon_centers(positions, angles):
    """Get centers of all hexagons for spatial indexing"""
    return np.array(positions)

def fast_check_overlap_pair_fast(hex1_vertices, hex2_vertices):
    """Fast overlap check using bounding circles for early rejection"""
    # Calculate centroids
    center1 = np.mean(hex1_vertices, axis=0)
    center2 = np.mean(hex2_vertices, axis=0)

    # Distance between centers
    dist = np.sqrt(np.sum((center1 - center2)**2))

    # If distance is greater than sum of radii, no overlap
    # For unit hexagon, approximate circumradius is 1
    if dist > 2.0:
        return False

    # Fall back to actual polygon intersection test
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def build_bvh_for_hexagons(positions, angles):
    """Build a simple bounding volume hierarchy for hexagons"""
    # For each hexagon, compute its bounding box
    boxes = []
    for i, (pos, angle) in enumerate(zip(positions, angles)):
        vertices = hexagon_vertices(pos[0], pos[1], np.deg2rad(angle))
        min_x = vertices[:, 0].min()
        max_x = vertices[:, 0].max()
        min_y = vertices[:, 1].min()
        max_y = vertices[:, 1].max()
        boxes.append(((min_x, min_y), (max_x, max_y), i))

    return boxes

def bvh_intersect_boxes(box1, box2):
    """Check if two bounding boxes intersect"""
    (min_x1, min_y1), (max_x1, max_y1) = box1
    (min_x2, min_y2), (max_x2, max_y2) = box2
    return not (max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1)

def bvh_get_candidates_better(positions, angles):
    """Get candidate pairs using BVH approach for overlap detection"""
    # Build bounding boxes for all hexagons
    boxes = build_bvh_for_hexagons(positions, angles)

    # Use a simpler spatial approach for now - check if bounding boxes overlap
    candidates = []
    n_hexagons = len(positions)

    # For each pair, check if their bounding boxes intersect
    for i in range(n_hexagons):
        for j in range(i+1, n_hexagons):
            if bvh_intersect_boxes(boxes[i], boxes[j]):
                candidates.append((i, j))

    return candidates

def fast_check_overlaps_bvh(positions, angles):
    """Fast overlap checking using BVH approach"""
    # Get candidate pairs from bounding boxes
    candidates = bvh_get_candidates_better(positions, angles)

    # Check actual overlaps for candidate pairs
    for i, j in candidates:
        # Get vertices for both hexagons
        hex1_vertices = hexagon_vertices(positions[i][0], positions[i][1], np.deg2rad(angles[i]))
        hex2_vertices = hexagon_vertices(positions[j][0], positions[j][1], np.deg2rad(angles[j]))

        if fast_check_overlap_pair_fast(hex1_vertices, hex2_vertices):
            return True  # Found overlap

    return False  # No overlaps found

def calculate_objective(params, outer_radius_guess=None, penalty_scale=1000000, violation_history=[]):
    """Calculate objective function with adaptive penalty scaling"""
    # Extract positions and angles
    positions_angles = params.reshape(-1, 3)
    positions = positions_angles[:, :2]
    angles = positions_angles[:, 2]

    # Estimate outer hexagon radius
    estimated_radius = estimate_outer_hexagon_radius(positions, angles)
    if outer_radius_guess is not None:
        outer_radius = outer_radius_guess
    else:
        outer_radius = estimated_radius

    # Create outer hexagon (centered at origin)
    outer_hex = create_unit_hexagon((0, 0), 0)

    # Check constraints
    total_penalty = 0
    violation_count = 0

    # Check containment
    for i, (pos, angle) in enumerate(zip(positions, angles)):
        hexagon = create_unit_hexagon(pos, angle)
        if not check_containment(hexagon, outer_hex):
            total_penalty += penalty_scale
            violation_count += 1

    # Check overlaps using BVH approach (more efficient than spatial tree for our case)
    overlap_found = False
    try:
        # Use BVH for fast overlap checking
        overlap_found = fast_check_overlaps_bvh(positions, angles)
    except Exception as e:
        # Fallback to brute force if BVH fails
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                hex1 = create_unit_hexagon(positions[i], angles[i])
                hex2 = create_unit_hexagon(positions[j], angles[j])
                if check_overlap(hex1, hex2):
                    overlap_found = True
                    break
            if overlap_found:
                break

    if overlap_found:
        total_penalty += penalty_scale
        violation_count += 1

    # Adaptive penalty scaling based on violation history
    if len(violation_history) > 0:
        avg_violations = np.mean(violation_history[-10:])  # Average over last 10 evaluations
        if avg_violations > 0.5:  # If many violations recently
            # Increase penalty to be more strict
            penalty_scale *= 1.2
        elif avg_violations < 0.2:  # If few violations recently
            # Decrease penalty slightly to allow more exploration
            penalty_scale *= 0.95

    # Update violation history
    violation_history.append(violation_count)

    # Return negative 1/outer_radius plus penalties
    if outer_radius > 0:
        obj_val = -1.0 / outer_radius + total_penalty
    else:
        obj_val = np.inf

    return obj_val

def check_overlap(hex1, hex2):
    """Check if two hexagons overlap."""
    return hex1.intersects(hex2)

def check_containment(hexagon, outer_hexagon):
    """Check if hexagon is fully contained within outer_hexagon."""
    return outer_hexagon.contains(hexagon)

def create_better_initial_config():
    """Create a much better initial configuration using hexagonal packing principles"""
    # Start with more sophisticated symmetric arrangement
    positions_angles = np.zeros((12, 3))

    # Central hexagon
    positions_angles[0] = [0, 0, 0]

    # Arrange in concentric rings - optimized layout
    # First ring: 6 hexagons around center (at radius of 2.0)
    ring1_radius = 2.0
    for i in range(1, 7):
        angle = 2 * np.pi * (i-1) / 6
        positions_angles[i] = [ring1_radius * np.cos(angle), ring1_radius * np.sin(angle), 0]

    # Second ring: 5 hexagons in a pattern that leaves room for optimization
    # These should be placed to maximize space efficiency
    ring2_radius = 3.5
    for i in range(7, 12):
        # Adjusted angles to avoid some overlap issues
        angle = 2 * np.pi * (i-7) / 5 + np.pi/12  # Small offset for better distribution
        positions_angles[i] = [ring2_radius * np.cos(angle), ring2_radius * np.sin(angle), 0]

    return positions_angles

def optimize_hexagon_packing_multistage():
    """Multi-stage optimization for better results"""
    n_hexagons = 12

    # Stage 1: Coarse optimization with relaxed constraints
    initial_positions_angles = create_better_initial_config()
    x0 = initial_positions_angles.flatten()

    # Bounds for optimization
    bounds = []
    # Positions: allow wider movement
    for i in range(n_hexagons * 2):
        bounds.append((-15, 15))
    # Angles: 0 to 360 degrees
    for i in range(n_hexagons):
        bounds.append((0, 360))

    # Stage 1: Coarse optimization
    try:
        result1 = differential_evolution(
            calculate_objective,
            bounds,
            args=(None, 100000),  # Reduced penalty for coarse stage
            maxiter=150,
            popsize=25,
            seed=42,
            disp=False
        )

        # Stage 2: Refine with tighter constraints
        refined_params = result1.x
        positions_angles = refined_params.reshape(-1, 3)

        # More precise optimization with full penalties
        result2 = differential_evolution(
            calculate_objective,
            bounds,
            args=(None, 1000000),  # Full penalty for fine stage
            maxiter=300,
            popsize=30,
            seed=42,
            disp=False
        )

        optimized_params = result2.x
        positions_angles = optimized_params.reshape(-1, 3)
        positions = positions_angles[:, :2]
        angles = positions_angles[:, 2]

        # Compute final outer radius based on optimized positions
        outer_radius = estimate_outer_hexagon_radius(positions, angles)

    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback to good initial configuration
        positions_angles = initial_positions_angles
        outer_radius = estimate_outer_hexagon_radius(initial_positions_angles[:, :2], initial_positions_angles[:, 2])

    return positions_angles, np.array([0, 0, 0]), outer_radius

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    # Optimize the hexagon packing using multi-stage approach
    inner_hex_data, outer_hex_data, outer_hex_side_length = optimize_hexagon_packing_multistage()

    # Validate the solution
    try:
        # Create hexagon objects for validation
        all_hexagons = []
        for i, (pos, angle) in enumerate(zip(inner_hex_data[:, :2], inner_hex_data[:, 2])):
            h = create_unit_hexagon(pos, angle)
            all_hexagons.append(h)

        # Check all pairwise overlaps
        for i in range(len(all_hexagons)):
            for j in range(i+1, len(all_hexagons)):
                if all_hexagons[i].intersects(all_hexagons[j]):
                    raise ValueError("Overlapping hexagons detected")

        # Check containment in outer hexagon (approximate)
        outer_hex = create_unit_hexagon((0, 0), 0)
        for hexagon in all_hexagons:
            if not outer_hex.contains(hexagon):
                raise ValueError("Some hexagons outside outer hexagon")

    except ValueError as e:
        print(f"Validation error: {e}")
        # Fallback to a reasonable configuration if validation fails
        inner_hex_data = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
            [0, -4, 0],
        ])
        outer_hex_side_length = 8.0

    end_time = time.time()

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END