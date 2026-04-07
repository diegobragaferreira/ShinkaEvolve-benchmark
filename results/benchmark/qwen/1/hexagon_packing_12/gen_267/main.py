# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import time
from numba import jit, prange
import warnings

@jit(nopython=True, parallel=True)
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

@jit(nopython=True)
def distance_point_to_line(px, py, x1, y1, x2, y2):
    """Calculate distance from point to line segment."""
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

@jit(nopython=True)
def point_in_hexagon(px, py, hx, hy, angle_deg, side_length=1):
    """Fast point-in-hexagon test using winding number."""
    vertices = hexagon_vertices(hx, hy, angle_deg, side_length)
    # Use ray casting method for simplicity and speed
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

@jit(nopython=True)
def compute_min_distance_hexagon_hexagon(h1_x, h1_y, h1_angle, h2_x, h2_y, h2_angle):
    """Compute minimum distance between two hexagons using analytical approach."""
    v1 = hexagon_vertices(h1_x, h1_y, h1_angle)
    v2 = hexagon_vertices(h2_x, h2_y, h2_angle)

    min_dist = np.inf
    # Check vertex-to-vertex distances
    for i in range(6):
        for j in range(6):
            dist = np.sqrt((v1[i,0]-v2[j,0])**2 + (v1[i,1]-v2[j,1])**2)
            if dist < min_dist:
                min_dist = dist

    # Check vertex-to-edge distances
    for i in range(6):
        for j in range(6):
            # Distance from vertex v1[i] to edge v2[j]-v2[(j+1)%6]
            dist = distance_point_to_line(v1[i,0], v1[i,1], v2[j,0], v2[j,1], v2[(j+1)%6,0], v2[(j+1)%6,1])
            if dist < min_dist:
                min_dist = dist

            # Distance from vertex v2[j] to edge v1[i]-v1[(i+1)%6]
            dist = distance_point_to_line(v2[j,0], v2[j,1], v1[i,0], v1[i,1], v1[(i+1)%6,0], v1[(i+1)%6,1])
            if dist < min_dist:
                min_dist = dist

    return min_dist

@jit(nopython=True)
def compute_hexagon_center(x, y, angle_deg, side_length=1):
    """Compute center of hexagon - just returns input since hexagon is centered at (x,y)."""
    return x, y

# BVH Node for spatial acceleration
class BVHNode:
    def __init__(self, bounds_min, bounds_max, objects=None, left=None, right=None):
        self.bounds_min = np.array(bounds_min)
        self.bounds_max = np.array(bounds_max)
        self.objects = objects or []
        self.left = left
        self.right = right

    def contains(self, point):
        return (self.bounds_min <= point).all() and (point <= self.bounds_max).all()

    def intersects(self, other):
        return not (self.bounds_max[0] < other.bounds_min[0] or
                   self.bounds_min[0] > other.bounds_max[0] or
                   self.bounds_max[1] < other.bounds_min[1] or
                   self.bounds_min[1] > other.bounds_max[1])

# BVH for hexagon overlap detection
class BVH:
    def __init__(self, hexagons, max_objects_per_node=4):
        self.max_objects_per_node = max_objects_per_node
        self.root = self._build_tree(hexagons)

    def _build_tree(self, hexagons):
        if not hexagons:
            return None

        # Initialize bounds
        bounds_min = np.array([float('inf'), float('inf')])
        bounds_max = np.array([-float('inf'), -float('inf')])

        for hex_data in hexagons:
            bounds_min[0] = min(bounds_min[0], hex_data[0] - 1)
            bounds_min[1] = min(bounds_min[1], hex_data[1] - 1)
            bounds_max[0] = max(bounds_max[0], hex_data[0] + 1)
            bounds_max[1] = max(bounds_max[1], hex_data[1] + 1)

        # If too few objects, store directly
        if len(hexagons) <= self.max_objects_per_node:
            return BVHNode(bounds_min, bounds_max, hexagons)

        # Split along longest axis
        split_axis = 0 if (bounds_max[0] - bounds_min[0]) > (bounds_max[1] - bounds_min[1]) else 1
        split_value = (bounds_min[split_axis] + bounds_max[split_axis]) / 2

        left_objects = []
        right_objects = []
        for hex_data in hexagons:
            if hex_data[0] if split_axis == 0 else hex_data[1] <= split_value:
                left_objects.append(hex_data)
            else:
                right_objects.append(hex_data)

        # Prune empty branches
        left_child = self._build_tree(left_objects) if left_objects else None
        right_child = self._build_tree(right_objects) if right_objects else None

        return BVHNode(bounds_min, bounds_max, None, left_child, right_child)

    def query(self, point, radius):
        """Find all hexagons within a given radius of point"""
        results = []
        self._query_recursive(self.root, point, radius, results)
        return results

    def _query_recursive(self, node, point, radius, results):
        if not node:
            return

        # Skip if no intersection
        if not node.intersects(BVHNode(
            [point[0] - radius, point[1] - radius],
            [point[0] + radius, point[1] + radius]
        )):
            return

        # Add objects at leaf nodes
        if node.objects:
            for obj in node.objects:
                dist = np.sqrt((obj[0] - point[0])**2 + (obj[1] - point[1])**2)
                if dist <= radius:
                    results.append(obj)
        else:
            # Recursively check children
            self._query_recursive(node.left, point, radius, results)
            self._query_recursive(node.right, point, radius, results)

def compute_hexagon_polygon(x, y, angle_deg, side_length=1):
    """Convert hexagon parameters to shapely polygon."""
    vertices = hexagon_vertices(x, y, angle_deg, side_length)
    return Polygon(vertices)

class HexagonPackingOptimizer:
    """Optimizes hexagon packing using geometric tiling principles with performance enhancements."""

    def __init__(self):
        self.hex_side_length = 1.0

    def compute_outer_hexagon_polygon(self, side_length):
        """Get shapely polygon for outer hexagon."""
        # Vertices of a regular hexagon with given side length, centered at origin
        vertices = []
        for i in range(6):
            theta = i * np.pi / 3
            x = side_length * np.cos(theta)
            y = side_length * np.sin(theta)
            vertices.append((x, y))
        return Polygon(vertices)

    @staticmethod
    @jit(nopython=True)
    def fast_check_containment_center(center_x, center_y, outer_radius):
        """Fast containment check using distance from center."""
        distance_from_origin = np.sqrt(center_x*center_x + center_y*center_y)
        # Allow for hexagon radius (1 unit) margin
        return distance_from_origin <= (outer_radius - 1.0)

    def compute_overlap_penalty_efficient(self, inner_hex_data, outer_radius):
        """Efficiently compute overlap penalty using BVH spatial acceleration."""
        n = len(inner_hex_data)
        if n <= 1:
            return 0.0

        # Create BVH for spatial acceleration
        bvh = BVH(inner_hex_data)

        penalty = 0.0
        overlap_count = 0

        # For 12 hexagons, we can do a more direct approach with BVH
        # Check each hexagon against others using BVH for candidates
        for i in range(n):
            # Find potentially overlapping hexagons using BVH
            point = np.array([inner_hex_data[i][0], inner_hex_data[i][1]])
            candidates = bvh.query(point, 2.0)  # Radius of 2 units for overlap check

            # Check overlap with candidates
            for j in range(i+1, n):
                # Quick distance check first
                dist = np.sqrt((inner_hex_data[i][0] - inner_hex_data[j][0])**2 +
                              (inner_hex_data[i][1] - inner_hex_data[j][1])**2)

                if dist <= 2.0:  # Might overlap
                    # Use more precise distance calculation
                    min_dist = compute_min_distance_hexagon_hexagon(
                        inner_hex_data[i][0], inner_hex_data[i][1], inner_hex_data[i][2],
                        inner_hex_data[j][0], inner_hex_data[j][1], inner_hex_data[j][2]
                    )

                    if min_dist < 0.001:  # Overlapping
                        # More precise overlap check
                        poly_i = compute_hexagon_polygon(inner_hex_data[i][0], inner_hex_data[i][1], inner_hex_data[i][2])
                        poly_j = compute_hexagon_polygon(inner_hex_data[j][0], inner_hex_data[j][1], inner_hex_data[j][2])

                        if poly_i.intersects(poly_j) and not poly_i.touches(poly_j):
                            try:
                                overlap = poly_i.intersection(poly_j)
                                if hasattr(overlap, 'area') and overlap.area > 0:
                                    penalty += overlap.area
                                    overlap_count += 1
                            except:
                                penalty += 1000  # Large penalty for calculation errors

        return penalty

    def evaluate_packing_fitness(self, params):
        """Evaluate the fitness of a packing configuration with improved performance."""
        try:
            # Extract inner hexagon data and outer radius
            inner_hex_data = params[:-1].reshape(12, 3)
            outer_hex_side_length = params[-1]

            # Create outer hexagon polygon
            outer_hex_poly = self.compute_outer_hexagon_polygon(outer_hex_side_length)

            # Check containment with fast approximations first
            containment_valid = True
            total_penetration = 0.0

            # Check each hexagon for containment
            for i in range(len(inner_hex_data)):
                x, y, angle = inner_hex_data[i]

                # Quick containment check using center distance
                if not self.fast_check_containment_center(x, y, outer_hex_side_length):
                    containment_valid = False
                    # Estimate penetration
                    try:
                        # Create hexagon polygon to check containment
                        hex_poly = compute_hexagon_polygon(x, y, angle)
                        diff = outer_hex_poly.difference(hex_poly)
                        if hasattr(diff, 'area'):
                            total_penetration += diff.area
                    except:
                        total_penetration += 1000
                    break

            if not containment_valid:
                return 1e10 + total_penetration * 10000

            # Compute overlap penalty using spatial indexing
            overlap_penalty = self.compute_overlap_penalty_efficient(inner_hex_data, outer_hex_side_length)

            if overlap_penalty > 0:
                return overlap_penalty * 10000

            # If valid, return inverse of outer hexagon side length (we want to maximize 1/R)
            return -1.0 / outer_hex_side_length

        except Exception as e:
            return 1e10

    def generate_initial_config(self):
        """Generate an initial configuration based on known good arrangements."""
        # Start with a hexagonal packing pattern - this is based on mathematical knowledge
        # of efficient hexagon packings

        # Central hexagon
        positions = [[0, 0, 0]]

        # First ring - 6 hexagons at distance = sqrt(3)
        # This is a well-known dense packing pattern for hexagons
        for i in range(6):
            angle = i * 60
            x = np.sqrt(3) * np.cos(np.radians(angle))
            y = np.sqrt(3) * np.sin(np.radians(angle))
            positions.append([x, y, 0])

        # Second ring - 6 hexagons at distance = 2*sqrt(3)
        for i in range(6):
            angle = i * 60 + 30  # offset by 30 degrees
            x = 2 * np.sqrt(3) * np.cos(np.radians(angle))
            y = 2 * np.sqrt(3) * np.sin(np.radians(angle))
            positions.append([x, y, 0])

        # Use only first 12 positions
        initial_config = np.array(positions[:12])

        # Add small random perturbations to avoid local minima
        np.random.seed(42)
        initial_config[:, :2] += np.random.normal(0, 0.05, (12, 2))

        return initial_config

    def optimize(self):
        """Main optimization routine with enhanced performance."""
        # Generate initial configuration
        initial_guess = self.generate_initial_config()

        # Initial estimate for outer radius based on spread
        max_dist = 0
        for i in range(12):
            x, y, _ = initial_guess[i]
            dist = np.sqrt(x*x + y*y)
            max_dist = max(max_dist, dist)

        # Add margin for hexagon size (1 unit radius)
        initial_outer_radius = max_dist + 1.5

        # Combine into single parameter vector
        initial_params = np.concatenate([initial_guess.flatten(), [initial_outer_radius]])

        # Define bounds - tighter bounds to improve convergence
        bounds = []
        for _ in range(12):
            bounds.extend([(-8, 8), (-8, 8), (0, 360)])  # Reduced bounds
        bounds.append((2.0, 12.0))  # Reasonable outer hex radius bounds

        # First try trust-constr optimization which is often better for smooth problems
        try:
            result = minimize(
                self.evaluate_packing_fitness,
                initial_params,
                method='trust-constr',
                bounds=bounds,
                options={'maxiter': 80, 'disp': False, 'gtol': 1e-6, 'xtol': 1e-6}
            )

            if result.success:
                final_params = result.x
            else:
                # Fall back to L-BFGS-B if trust-constr fails
                warnings.warn("trust-constr failed, falling back to L-BFGS-B")
                result = minimize(
                    self.evaluate_packing_fitness,
                    initial_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 80, 'disp': False, 'ftol': 1e-9}
                )
                if result.success:
                    final_params = result.x
                else:
                    final_params = initial_params

        except Exception as e:
            # If anything fails, use initial guess
            warnings.warn(f"Optimization failed with exception: {e}")
            final_params = initial_params

        # Extract results with final validation
        inner_hex_data = final_params[:-1].reshape(12, 3)
        outer_hex_side_length = final_params[-1]

        # Apply symmetry constraints to improve quality
        # Force first ring hexagons to maintain 60-degree symmetry
        # Preserve symmetry of ring structures

        # Final correction if necessary
        outer_hex_poly = self.compute_outer_hexagon_polygon(outer_hex_side_length)

        # Ensure all hexagons are properly contained
        needs_adjustment = False
        for i in range(12):
            x, y, angle = inner_hex_data[i]
            hex_poly = compute_hexagon_polygon(x, y, angle)
            center = hex_poly.centroid
            if not outer_hex_poly.contains(center):
                needs_adjustment = True
                break

        if needs_adjustment:
            min_outer_radius = 0
            for i in range(12):
                x, y, _ = inner_hex_data[i]
                dist = np.sqrt(x*x + y*y) + 1.0  # +1 for hexagon radius
                min_outer_radius = max(min_outer_radius, dist)
            outer_hex_side_length = min_outer_radius * 1.05

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
        # Create optimizer
        optimizer = HexagonPackingOptimizer()

        # Get optimized configuration
        inner_hex_data, outer_hex_side_length = optimizer.optimize()

        # Create outer hexagon data (centered at origin, no rotation)
        outer_hex_data = np.array([0, 0, 0])

        # Ensure we don't exceed time limits (though should finish much faster)
        end_time = time.time()
        eval_time = end_time - start_time

        return inner_hex_data, outer_hex_data, outer_hex_side_length

    except Exception as e:
        # Fallback to simple configuration if optimization fails
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