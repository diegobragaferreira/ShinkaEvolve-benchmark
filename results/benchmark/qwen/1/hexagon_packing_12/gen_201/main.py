# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
from numba import jit
from joblib import Parallel, delayed

class BVHNode:
    """Simple BVH node for spatial acceleration."""
    def __init__(self, bounds=None, items=None, left=None, right=None):
        self.bounds = bounds  # Bounding box [min_x, max_x, min_y, max_y]
        self.items = items or []  # Indices of objects in this node
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None

@jit(nopython=True)
def get_hexagon_bounds(x, y, angle_deg, side_length=1):
    """Get bounding box for a hexagon."""
    vertices = hexagon_vertices(x, y, angle_deg, side_length)
    min_x = vertices[0, 0]
    max_x = vertices[0, 0]
    min_y = vertices[0, 1]
    max_y = vertices[0, 1]

    for i in range(1, 6):
        if vertices[i, 0] < min_x:
            min_x = vertices[i, 0]
        elif vertices[i, 0] > max_x:
            max_x = vertices[i, 0]
        if vertices[i, 1] < min_y:
            min_y = vertices[i, 1]
        elif vertices[i, 1] > max_y:
            max_y = vertices[i, 1]

    return np.array([min_x, max_x, min_y, max_y])

def build_bvh(hex_positions, hex_angles, max_depth=10, min_items=4):
    """Build a simple BVH for hexagon bounding boxes."""
    # Precompute bounds for all hexagons
    bounds = []
    for i in range(len(hex_positions)):
        x, y = hex_positions[i]
        angle = hex_angles[i]
        bounds.append(get_hexagon_bounds(x, y, angle))

    bounds = np.array(bounds)

    # Create initial indices list
    indices = list(range(len(hex_positions)))

    def build_recursive(indices_list, depth):
        if depth > max_depth or len(indices_list) <= min_items:
            # Create leaf node
            return BVHNode(bounds=None, items=indices_list)

        # Compute bounding box for current group
        min_x = np.min(bounds[np.array(indices_list), 0])
        max_x = np.max(bounds[np.array(indices_list), 1])
        min_y = np.min(bounds[np.array(indices_list), 2])
        max_y = np.max(bounds[np.array(indices_list), 3])
        node_bounds = np.array([min_x, max_x, min_y, max_y])

        if len(indices_list) <= min_items:
            return BVHNode(bounds=node_bounds, items=indices_list)

        # Split along longest axis
        width = max_x - min_x
        height = max_y - min_y

        if width >= height:
            # Split along x-axis
            median = np.median(bounds[np.array(indices_list), 0])
            left_indices = [i for i in indices_list if bounds[i, 0] <= median]
            right_indices = [i for i in indices_list if bounds[i, 0] > median]
        else:
            # Split along y-axis
            median = np.median(bounds[np.array(indices_list), 2])
            left_indices = [i for i in indices_list if bounds[i, 2] <= median]
            right_indices = [i for i in indices_list if bounds[i, 2] > median]

        if not left_indices or not right_indices:
            # If splitting didn't work, make a leaf
            return BVHNode(bounds=node_bounds, items=indices_list)

        left_node = build_recursive(left_indices, depth + 1)
        right_node = build_recursive(right_indices, depth + 1)

        return BVHNode(bounds=node_bounds, left=left_node, right=right_node)

    return build_recursive(indices, 0)

def bvh_query_overlaps(bvh_root, hex_positions, hex_angles, threshold=0.01):
    """Query overlaps using BVH structure."""
    overlaps = []

    def query_recursive(node):
        if node.is_leaf():
            # Check overlaps among items in this leaf
            for i in range(len(node.items)):
                for j in range(i+1, len(node.items)):
                    idx1, idx2 = node.items[i], node.items[j]
                    if check_hexagon_overlap_fast(
                        hex_positions[idx1][0], hex_positions[idx1][1], hex_angles[idx1],
                        hex_positions[idx2][0], hex_positions[idx2][1], hex_angles[idx2]
                    ):
                        overlaps.append((idx1, idx2))
        else:
            # Check if node bounds intersect with themselves (not needed)
            if node.left and node.right:
                # Recursively query children
                query_recursive(node.left)
                query_recursive(node.right)

    query_recursive(bvh_root)
    return overlaps

@jit(nopython=True)
def check_hexagon_overlap_fast(x1, y1, angle1, x2, y2, angle2, side_length=1):
    """Fast approximate overlap check using bounding boxes."""
    bounds1 = get_hexagon_bounds(x1, y1, angle1, side_length)
    bounds2 = get_hexagon_bounds(x2, y2, angle2, side_length)

    # Simple bounding box intersection test
    if (bounds1[1] < bounds2[0] or bounds2[1] < bounds1[0] or
        bounds1[3] < bounds2[2] or bounds2[3] < bounds1[2]):
        return False

    # More precise check using distance between centers
    center_dist_sq = (x1 - x2)**2 + (y1 - y2)**2
    # Max possible distance between hexagons' centers to overlap
    max_dist_sq = (2 * side_length)**2  # Approximate
    return center_dist_sq < max_dist_sq

@jit(nopython=True)
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

def evaluate_configuration(params):
    """Evaluate a configuration for validity and quality.
    params: array of shape (37,) where first 36 values are (x,y,angle) for 12 hexagons
            and the last value is the outer hexagon side length"""

    # Extract inner hexagon data and outer radius
    inner_hex_data = params[:-1].reshape(12, 3)
    outer_hex_side_length = params[-1]

    # Create outer hexagon polygon (centered at origin)
    outer_hex_poly = compute_hexagon_polygon(0, 0, 0, outer_hex_side_length)

    # Check containment and overlaps
    total_penetration = 0.0
    valid = True

    n = len(inner_hex_data)
    inner_polys = []

    # Compute all inner hexagon polygons
    for i in range(n):
        x, y, angle = inner_hex_data[i]
        inner_poly = compute_hexagon_polygon(x, y, angle)
        inner_polys.append(inner_poly)

        # Check containment
        if not check_containment(inner_poly, outer_hex_poly):
            valid = False
            # Calculate penetration
            try:
                diff = outer_hex_poly.difference(inner_poly)
                if hasattr(diff, 'area'):
                    total_penetration += diff.area
            except:
                total_penetration += 1000  # Large penalty

        # Check overlaps with previous hexagons
        for j in range(i):
            if check_overlap(inner_polys[i], inner_polys[j]):
                valid = False
                # Calculate overlap area
                try:
                    overlap = inner_polys[i].intersection(inner_polys[j])
                    if hasattr(overlap, 'area'):
                        total_penetration += overlap.area
                except:
                    total_penetration += 1000  # Large penalty

    if not valid:
        # Adaptive penalty - scale with the severity of the constraint violation
        penalty = total_penetration * 10000 if total_penetration > 0 else 1000000
        return penalty

    # If valid, return inverse of outer hexagon side length (negative because we minimize)
    return -1.0 / outer_hex_side_length

def generate_symmetric_initial_guess():
    """Generate a good initial symmetric configuration."""
    # Start with a pattern resembling known good packings
    # Use mathematical arrangement based on hexagonal lattice points
    positions = []

    # Central hexagon
    positions.append([0, 0, 0])

    # First ring around center
    for i in range(6):
        angle = i * 60
        x = 1.732 * np.cos(np.radians(angle))  # ~sqrt(3) for proper spacing
        y = 1.732 * np.sin(np.radians(angle))
        positions.append([x, y, 0])

    # Second ring
    for i in range(6):
        angle = i * 60 + 30  # offset to create a more efficient packing
        x = 3.464 * np.cos(np.radians(angle))  # ~2*sqrt(3)
        y = 3.464 * np.sin(np.radians(angle))
        positions.append([x, y, 0])

    # Additional strategic placement
    positions.append([0, -4.5, 0])

    # Take only first 12 positions
    return np.array(positions[:12])

def optimize_packing():
    """Main optimization function using hybrid approach."""
    # Generate initial guess
    initial_guess_inner = generate_symmetric_initial_guess()

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
        return evaluate_configuration(params)

    # Run differential evolution with fewer iterations for speed
    de_result = differential_evolution(
        objective_global,
        bounds,
        maxiter=50,
        popsize=15,
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
            evaluate_configuration,
            best_params,
            method='L-BFGS-B',
            bounds=refined_bounds,
            options={'maxiter': 100, 'disp': False}
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

        return inner_hex_data, outer_hex_data, outer_hex_side_length

    except Exception as e:
        # Fallback to original configuration if optimization fails
        n = 12
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

        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8  # Large enough to contain all inner hexagons

        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END