# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
from numba import jit, njit
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
import math

class BVHNode:
    """Bounding Volume Hierarchy node for spatial acceleration."""
    def __init__(self, indices, bounds, left=None, right=None):
        self.indices = indices  # indices of items in this node
        self.bounds = bounds    # bounding box [min_x, min_y, max_x, max_y]
        self.left = left        # left child
        self.right = right      # right child
        self.is_leaf = left is None and right is None

    def get_bounds(self):
        return self.bounds

    def get_indices(self):
        return self.indices

    def is_leaf_node(self):
        return self.is_leaf

def build_bvh_tree(items, max_items_per_node=4):
    """Build a BVH tree from a list of items with bounding boxes."""
    # Calculate bounding boxes for all items
    bounds_list = []
    for i, item in enumerate(items):
        # Assuming item is a tuple (index, bounding_box)
        bounds_list.append((i, item[1]))

    # Build BVH recursively
    root = _build_bvh_recursive(bounds_list, max_items_per_node)
    return root

def _build_bvh_recursive(bounds_list, max_items_per_node):
    """Recursively build BVH tree."""
    if len(bounds_list) <= max_items_per_node:
        # Create leaf node
        indices = [item[0] for item in bounds_list]
        if not bounds_list:
            return BVHNode(indices, [0, 0, 0, 0])

        # Calculate combined bounding box
        min_x = min(item[1][0] for item in bounds_list)
        min_y = min(item[1][1] for item in bounds_list)
        max_x = max(item[1][2] for item in bounds_list)
        max_y = max(item[1[3] for item in bounds_list)

        return BVHNode(indices, [min_x, min_y, max_x, max_y])

    # Split items along longest axis
    # Calculate overall bounds
    min_x = min(item[1][0] for item in bounds_list)
    min_y = min(item[1][1] for item in bounds_list)
    max_x = max(item[1][2] for item in bounds_list)
    max_y = max(item[1][3] for item in bounds_list)

    # Determine split axis (longer dimension)
    width = max_x - min_x
    height = max_y - min_y

    if width > height:
        # Split along x-axis
        mid_x = (min_x + max_x) / 2
        left_items = [item for item in bounds_list if item[1][2] <= mid_x]
        right_items = [item for item in bounds_list if item[1][0] > mid_x]
    else:
        # Split along y-axis
        mid_y = (min_y + max_y) / 2
        left_items = [item for item in bounds_list if item[1][3] <= mid_y]
        right_items = [item for item in bounds_list if item[1][1] > mid_y]

    # Create internal node
    left_child = _build_bvh_recursive(left_items, max_items_per_node)
    right_child = _build_bvh_recursive(right_items, max_items_per_node)

    # Calculate combined bounds
    min_x = min(left_child.bounds[0], right_child.bounds[0])
    min_y = min(left_child.bounds[1], right_child.bounds[1])
    max_x = max(left_child.bounds[2], right_child.bounds[2])
    max_y = max(left_child.bounds[3], right_child.bounds[3])

    return BVHNode([], [min_x, min_y, max_x, max_y], left_child, right_child)

def query_bvh(bvh_root, query_bounds):
    """Query BVH for items that potentially intersect with query bounds."""
    potential_hits = []
    _query_recursive(bvh_root, query_bounds, potential_hits)
    return potential_hits

def _query_recursive(node, query_bounds, hits):
    """Recursive query implementation."""
    if not node:
        return

    # Check if query bounds intersect with node bounds
    if not _bounds_intersect(node.bounds, query_bounds):
        return

    if node.is_leaf_node():
        # Add all indices in this leaf
        hits.extend(node.get_indices())
    else:
        # Recurse into children
        _query_recursive(node.left, query_bounds, hits)
        _query_recursive(node.right, query_bounds, hits)

def _bounds_intersect(bounds1, bounds2):
    """Check if two bounding boxes intersect."""
    return not (bounds1[2] < bounds2[0] or bounds2[2] < bounds1[0] or
               bounds1[3] < bounds2[1] or bounds2[3] < bounds1[1])

@njit
def hexagon_vertices(x, y, angle_deg, side_length=1):
    """Compute vertices of a hexagon given center, rotation, and side length."""
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    # Vertices of regular hexagon with side length 1 centered at origin
    base_verts = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    # Rotate and translate
    rotated_verts = np.empty_like(base_verts)
    for i in range(6):
        x_orig, y_orig = base_verts[i]
        rotated_verts[i] = [
            x + side_length * (x_orig * cos_a - y_orig * sin_a),
            y + side_length * (x_orig * sin_a + y_orig * cos_a)
        ]

    return rotated_verts

@njit
def compute_bounding_circle_center_radius(x, y, angle_deg):
    """Compute center and radius of bounding circle for a hexagon."""
    # For a regular hexagon with side length 1, the circumradius is 1
    # The bounding circle has same center as hexagon and radius 1
    return x, y, 1.0

@njit
def distance_between_centers(x1, y1, x2, y2):
    """Fast Euclidean distance between two points."""
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx * dx + dy * dy)

@njit
def estimate_hexagon_overlap_fast(x1, y1, angle1, x2, y2, angle2):
    """Fast heuristic to estimate if two hexagons might overlap."""
    # Use bounding circles for fast rejection
    _, _, r1 = compute_bounding_circle_center_radius(x1, y1, angle1)
    _, _, r2 = compute_bounding_circle_center_radius(x2, y2, angle2)

    # If centers are further apart than sum of radii, they can't overlap
    dist_centers = distance_between_centers(x1, y1, x2, y2)
    if dist_centers > (r1 + r2):
        return False

    # If centers are closer than difference of radii, one is inside the other
    if dist_centers < abs(r1 - r2):
        return True

    return True  # Could overlap, need detailed check

@njit
def point_in_polygon_fast(px, py, vertices):
    """Fast point-in-polygon test using ray casting."""
    n = len(vertices)
    inside = False
    p1x, p1y = vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = vertices[i % n]
        if py > min(p1y, p2y):
            if py <= max(p1y, p2y):
                if px <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or px <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

@njit
def distance_point_to_segment(px, py, x1, y1, x2, y2):
    """Fast distance from point to line segment."""
    # Vector from (x1,y1) to (x2,y2)
    dx = x2 - x1
    dy = y2 - y1

    # Length squared of line segment
    length_sq = dx*dx + dy*dy

    if length_sq == 0:
        # Line segment is a point
        return np.sqrt((px - x1)**2 + (py - y1)**2)

    # Project point onto line
    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    t = max(0, min(1, t))  # Clamp projection to line segment

    # Find closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    # Distance to closest point
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

@njit
def hexagon_distance_fast(v1, v2):
    """Fast minimum distance between two hexagons."""
    min_dist = np.inf
    for i in range(6):
        for j in range(6):
            # Distance between vertices
            dist = distance_point_to_segment(v1[i,0], v1[i,1], v2[j,0], v2[j,1], v2[(j+1)%6,0], v2[(j+1)%6,1])
            if dist < min_dist:
                min_dist = dist
            dist = distance_point_to_segment(v2[j,0], v2[j,1], v1[i,0], v1[i,1], v1[(i+1)%6,0], v1[(i+1)%6,1])
            if dist < min_dist:
                min_dist = dist
    return min_dist

def build_spatial_index(hex_data):
    """Build a spatial index for efficient neighbor lookups."""
    centers = np.array([[hex_data[i, 0], hex_data[i, 1]] for i in range(len(hex_data))])
    return cKDTree(centers)

def compute_hexagon_polygon(x, y, angle_deg, side_length=1):
    """Convert hexagon parameters to shapely polygon."""
    vertices = hexagon_vertices(x, y, angle_deg, side_length)
    return Polygon(vertices)

def check_containment(hex_poly, outer_hex_poly):
    """Check if hexagon is fully contained within outer hexagon."""
    return outer_hex_poly.contains(hex_poly) or outer_hex_poly.covers(hex_poly)

def check_overlap(hex1_poly, hex2_poly):
    """Check if two hexagons overlap."""
    return hex1_poly.intersects(hex2_poly) and not hex1_poly.touches(hex2_poly)

def fast_overlap_detection_fast(inner_hex_data, outer_hex_side_length):
    """Fast overlap detection using bounding circle pre-filtering."""
    # Precompute all hexagon centers and bounds
    centers = np.array([[inner_hex_data[i, 0], inner_hex_data[i, 1]] for i in range(len(inner_hex_data))])

    # Build spatial index for neighbor search
    tree = cKDTree(centers)

    # Get pairs within reasonable distance (2 units for unit hexagons)
    pairs = tree.query_pairs(r=2.5, p=2)

    # Check overlaps for candidate pairs
    for i, j in pairs:
        if i < j:  # Avoid double checking
            x1, y1, angle1 = inner_hex_data[i]
            x2, y2, angle2 = inner_hex_data[j]

            # Fast bounding circle check first
            if estimate_hexagon_overlap_fast(x1, y1, angle1, x2, y2, angle2):
                # Check if actually overlapping with full polygon operations
                # For performance, we'll use our numba-based distance computation
                v1 = hexagon_vertices(x1, y1, angle1)
                v2 = hexagon_vertices(x2, y2, angle2)

                # If distance is less than 2 (sum of radii) then they probably overlap
                min_dist = hexagon_distance_fast(v1, v2)
                if min_dist < 0.001:  # Very small distance means overlap
                    return True

    return False

def fast_containment_check_fast(inner_hex_data, outer_hex_side_length):
    """Fast containment check using bounding circle."""
    outer_center = (0.0, 0.0)
    outer_radius = outer_hex_side_length

    # Check each hexagon's center against outer hexagon
    for i in range(len(inner_hex_data)):
        x, y, _ = inner_hex_data[i]
        dist = distance_between_centers(x, y, outer_center[0], outer_center[1])
        # Need to account for hexagon size (circumradius = 1)
        if dist + 1.0 > outer_radius:
            return False

    return True

def evaluate_configuration_fast(params):
    """Fast evaluation with optimized geometric checks."""
    # Extract inner hexagon data and outer radius
    inner_hex_data = params[:-1].reshape(12, 3)
    outer_hex_side_length = params[-1]

    # Fast containment check using bounding circle
    if not fast_containment_check_fast(inner_hex_data, outer_hex_side_length):
        return 1e6  # Large penalty for containment failure

    # Fast overlap detection
    if fast_overlap_detection_fast(inner_hex_data, outer_hex_side_length):
        return 1e6  # Large penalty for overlap

    # If valid, return inverse of outer hexagon side length (negative because we minimize)
    return -1.0 / outer_hex_side_length

def generate_improved_initial_guess():
    """Generate an improved initial configuration based on mathematical insight."""
    # Use a more sophisticated configuration that places hexagons in a pattern designed to
    # minimize the outer hexagon size while maintaining non-overlap and containment

    # Pattern: central hexagon with surrounding rings
    positions = []

    # Central hexagon
    positions.append([0, 0, 0])

    # First ring - six hexagons at distance of ~sqrt(3) from center
    for i in range(6):
        angle = i * 60
        # Distance chosen to allow efficient packing
        dist = 1.732  # sqrt(3)
        x = dist * np.cos(np.radians(angle))
        y = dist * np.sin(np.radians(angle))
        positions.append([x, y, 0])

    # Second ring - six hexagons at distance of ~2*sqrt(3) from center
    for i in range(6):
        angle = i * 60 + 30  # offset to create dense packing
        dist = 3.464  # 2*sqrt(3)
        x = dist * np.cos(np.radians(angle))
        y = dist * np.sin(np.radians(angle))
        positions.append([x, y, 0])

    # Additional strategic positioning for better packing
    positions.append([0, -4.5, 0])

    return np.array(positions[:12])

def optimize_packing():
    """Main optimization function using hybrid approach."""
    # Generate initial guess
    initial_guess_inner = generate_improved_initial_guess()

    # Initial estimate for outer radius based on configuration
    max_dist = 0
    for i in range(12):
        x, y, _ = initial_guess_inner[i]
        dist = np.sqrt(x*x + y*y)
        max_dist = max(max_dist, dist)

    # Add margin for hexagon size (hexagon has width approximately 2)
    initial_outer_radius = max_dist + 2.0

    # Combine into single parameter vector: [12*3 positions + 1 outer radius]
    initial_params = np.concatenate([initial_guess_inner.flatten(), [initial_outer_radius]])

    # Define bounds for optimization
    # Positions: x, y bounded to reasonable range, angle 0-360
    bounds = []
    for _ in range(12):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])
    # Outer radius bound (should be positive)
    bounds.append((0.1, 20.0))

    # First, use global optimization to explore the space broadly
    def objective_global(params):
        # Use smaller penalty scaling for global optimization
        return evaluate_configuration_fast(params)

    # Run differential evolution with fewer iterations for speed
    de_result = differential_evolution(
        objective_global,
        bounds,
        maxiter=30,
        popsize=10,
        seed=42,
        disp=False
    )

    # Use the best solution from global search as starting point for local refinement
    best_params = de_result.x

    # Local refinement using L-BFGS-B
    # We need to make sure we're still within bounds for the local optimizer
    # Let's define tighter bounds for local optimization
    refined_bounds = [(b[0], b[1]) for b in bounds]

    # Perform local optimization
    try:
        local_result = minimize(
            evaluate_configuration_fast,
            best_params,
            method='L-BFGS-B',
            bounds=refined_bounds,
            options={'maxiter': 50, 'disp': False}
        )
        if local_result.success:
            best_params = local_result.x
    except:
        # Fall back to previous solution if local optimization fails
        pass

    # Extract results
    inner_hex_data = best_params[:-1].reshape(12, 3)
    outer_hex_side_length = best_params[-1]

    # Ensure the outer hexagon is actually large enough
    # Recalculate to make sure it contains all hexagons
    min_outer_radius = 0
    for i in range(12):
        x, y, _ = inner_hex_data[i]
        dist = np.sqrt(x*x + y*y) + 1.0  # +1 for hexagon radius
        min_outer_radius = max(min_outer_radius, dist)

    # If we computed a smaller radius than needed, adjust it up
    if outer_hex_side_length < min_outer_radius:
        outer_hex_side_length = min_outer_radius * 1.05  # Add small margin

    return inner_hex_data, outer_hex_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Track execution time
    start_time = time.time()

    try:
        # Get optimized configuration
        inner_hex_data, outer_hex_side_length = optimize_packing()

        # Create outer hexagon data (centered at origin, no rotation)
        outer_hex_data = np.array([0, 0, 0])

        # Ensure we don't exceed time limits
        end_time = time.time()
        eval_time = end_time - start_time

        # Calculate benchmark ratio
        benchmark_ratio = (1.0 / outer_hex_side_length) / 0.2537

        return inner_hex_data, outer_hex_data, outer_hex_side_length

    except Exception as e:
        # Fallback to improved grid configuration if optimization fails
        inner_hex_data = np.array([
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
            [3.75, -2.17, 0],  # far bottom-right,
            [0, -4, 0],  # far bottom-center
        ])

        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8  # Large enough to contain all inner hexagons

        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END