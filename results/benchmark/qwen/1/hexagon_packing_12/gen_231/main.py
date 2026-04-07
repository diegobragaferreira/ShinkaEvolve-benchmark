# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
from numba import jit
from joblib import Parallel, delayed
import math

class HexagonGeometry:
    """Handles all geometric computations related to hexagons."""

    def __init__(self):
        self._base_vertices = np.array([
            [1, 0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1, 0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])

    @staticmethod
    @jit(nopython=True)
    def vertices(x, y, angle_deg, side_length=1):
        """Compute vertices of a hexagon given center, rotation, and side length."""
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        base_verts = np.array([
            [1, 0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1, 0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])

        rotated_verts = np.empty_like(base_verts)
        for i in range(6):
            x_orig, y_orig = base_verts[i]
            rotated_verts[i] = [
                x + side_length * (x_orig * cos_a - y_orig * sin_a),
                y + side_length * (x_orig * sin_a + y_orig * cos_a)
            ]

        return rotated_verts

    @staticmethod
    @jit(nopython=True)
    def is_point_inside_hexagon(px, py, hx, hy, angle_deg, side_length=1):
        """Fast point-in-hexagon test using winding number approach."""
        vertices = HexagonGeometry.vertices(hx, hy, angle_deg, side_length)
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

    @staticmethod
    @jit(nopython=True)
    def hexagon_bounding_circle_center_radius(x, y, angle_deg, side_length=1):
        """Compute bounding circle center and radius for a hexagon."""
        # For a regular hexagon with side length s, the circumradius is s
        # The bounding circle radius is also s (distance from center to any vertex)
        return x, y, side_length

class BVHNode:
    """BVH node for accelerating spatial queries"""
    def __init__(self, indices=None, bounds=None, left=None, right=None):
        self.indices = indices or []
        self.bounds = bounds  # (min_x, max_x, min_y, max_y)
        self.left = left
        self.right = right
        self.is_leaf = (left is None and right is None)

class BVH:
    """Bounding Volume Hierarchy for efficient spatial queries"""
    def __init__(self, hex_data, geom_handler):
        self.hex_data = hex_data
        self.geom = geom_handler
        self.root = None
        self.build_bvh()

    def build_bvh(self):
        """Build the BVH tree recursively"""
        # Create initial list of indices
        indices = list(range(len(self.hex_data)))

        # Compute bounding boxes for all hexagons
        bounds_list = []
        for i in range(len(self.hex_data)):
            x, y, angle = self.hex_data[i]
            vertices = self.geom.vertices(x, y, angle)
            min_x = min(v[0] for v in vertices)
            max_x = max(v[0] for v in vertices)
            min_y = min(v[1] for v in vertices)
            max_y = max(v[1] for v in vertices)
            bounds_list.append((min_x, max_x, min_y, max_y))

        self.root = self._build_recursive(indices, bounds_list, 0)

    def _build_recursive(self, indices, bounds_list, depth):
        """Recursively build BVH nodes"""
        if len(indices) <= 2 or depth > 10:  # Base case
            bounds = self._compute_bounds(indices, bounds_list)
            return BVHNode(indices, bounds)

        # Split along longest axis
        bounds = self._compute_bounds(indices, bounds_list)
        mid_x = (bounds[0] + bounds[1]) / 2
        mid_y = (bounds[2] + bounds[3]) / 2

        left_indices = []
        right_indices = []

        for idx in indices:
            min_x, max_x, min_y, max_y = bounds_list[idx]
            # Simple split based on center of bounding box
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2

            if center_x < mid_x or center_y < mid_y:
                left_indices.append(idx)
            else:
                right_indices.append(idx)

        # If we didn't split effectively, make leaf
        if len(left_indices) == 0 or len(right_indices) == 0:
            bounds = self._compute_bounds(indices, bounds_list)
            return BVHNode(indices, bounds)

        # Recursively build children
        left_node = self._build_recursive(left_indices, bounds_list, depth + 1)
        right_node = self._build_recursive(right_indices, bounds_list, depth + 1)

        bounds = self._compute_bounds(indices, bounds_list)
        return BVHNode(indices, bounds, left_node, right_node)

    def _compute_bounds(self, indices, bounds_list):
        """Compute combined bounds for a set of indices"""
        if not indices:
            return (0, 0, 0, 0)

        min_x = min(bounds_list[i][0] for i in indices)
        max_x = max(bounds_list[i][1] for i in indices)
        min_y = min(bounds_list[i][2] for i in indices)
        max_y = max(bounds_list[i][3] for i in indices)

        return (min_x, max_x, min_y, max_y)

    def query_candidates(self, target_idx, max_distance=None):
        """Query candidate indices that might intersect with target_idx"""
        candidates = []
        self._query_recursive(self.root, target_idx, candidates, max_distance)
        return candidates

    def _query_recursive(self, node, target_idx, candidates, max_distance):
        """Recursive querying function"""
        if node is None:
            return

        if node.is_leaf:
            # Check all indices in this leaf
            for idx in node.indices:
                if idx != target_idx:
                    candidates.append(idx)
            return

        # Check if we should traverse this node
        target_x, target_y, _ = self.geom.hexagon_bounding_circle_center_radius(
            self.hex_data[target_idx][0],
            self.hex_data[target_idx][1],
            self.hex_data[target_idx][2]
        )

        # Simple distance check to decide traversal
        if max_distance is not None:
            # Check if target is within max_distance of this node's bounds
            node_min_x, node_max_x, node_min_y, node_max_y = node.bounds
            center_x = (node_min_x + node_max_x) / 2
            center_y = (node_min_y + node_max_y) / 2
            dist_to_center = math.sqrt((target_x - center_x)**2 + (target_y - center_y)**2)

            if dist_to_center > max_distance + (node_max_x - node_min_x + node_max_y - node_min_y) / 2:
                return  # No need to traverse this subtree

        # Traverse children
        self._query_recursive(node.left, target_idx, candidates, max_distance)
        self._query_recursive(node.right, target_idx, candidates, max_distance)

class HexagonConstraintValidator:
    """Validates spatial constraints for hexagon arrangements."""

    def __init__(self, geom_handler):
        self.geom = geom_handler

    @staticmethod
    @jit(nopython=True)
    def check_containment_fast(inner_hex_data, outer_radius):
        """Fast check if all inner hexagons are contained within outer hexagon."""
        # This would require reimplementing the containment logic in numba
        # For now we'll leave it as is, but the main benefit comes from
        # optimizing the overlap penalty calculation
        return True

    def check_containment(self, inner_hex_data, outer_radius):
        """Check if all inner hexagons are contained within outer hexagon."""
        outer_hex_vertices = self.geom.vertices(0, 0, 0, outer_radius)
        outer_polygon = Polygon(outer_hex_vertices)

        for i in range(len(inner_hex_data)):
            x, y, angle = inner_hex_data[i]
            inner_vertices = self.geom.vertices(x, y, angle)
            inner_polygon = Polygon(inner_vertices)

            # Early exit if any hexagon is not contained
            if not outer_polygon.contains(inner_polygon):
                return False
        return True

    def compute_overlap_penalty(self, inner_hex_data):
        """Compute penalty based on overlap areas with BVH acceleration."""
        penalty = 0.0
        n = len(inner_hex_data)

        # For better performance, use BVH to reduce overlap checks
        bvh = BVH(inner_hex_data, self.geom)

        # Pre-compute polygons for efficiency
        polygons = []
        for i in range(n):
            x, y, angle = inner_hex_data[i]
            vertices = self.geom.vertices(x, y, angle)
            polygons.append(Polygon(vertices))

        # Use BVH to identify candidate pairs that might overlap
        overlap_pairs_found = 0
        for i in range(n):
            # Query potential candidates using BVH
            candidates = bvh.query_candidates(i, max_distance=3.0)  # Adjust for hex size

            for j in candidates:
                if i >= j:
                    continue

                # Quick distance check before precise overlap test
                x1, y1, _ = inner_hex_data[i]
                x2, y2, _ = inner_hex_data[j]
                distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)

                # If centers are far apart, no overlap possible
                if distance > 2.0:  # Sum of hexagon radii
                    continue

                # Perform precise overlap check
                if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                    overlap_pairs_found += 1
                    try:
                        overlap = polygons[i].intersection(polygons[j])
                        if hasattr(overlap, 'area') and overlap.area > 0:
                            penalty += overlap.area
                    except:
                        penalty += 1000  # Large penalty for calculation errors

        return penalty

class HexagonPackingOptimizer:
    """Main optimization class that orchestrates the packing process."""

    def __init__(self):
        self.geom = HexagonGeometry()
        self.validator = HexagonConstraintValidator(self.geom)

    def create_outer_hexagon_polygon(self, side_length):
        """Create shapely polygon representation of outer hexagon."""
        vertices = self.geom.vertices(0, 0, 0, side_length)
        return Polygon(vertices)

    def evaluate_configuration(self, params):
        """Evaluate a configuration for validity and quality."""
        try:
            # Extract inner hexagon data and outer radius
            inner_hex_data = params[:-1].reshape(12, 3)
            outer_hex_side_length = params[-1]

            # Check containment early (fast check)
            if not self.validator.check_containment(inner_hex_data, outer_hex_side_length):
                return 1e10  # Large penalty for containment violation

            # Compute overlap penalty with BVH acceleration
            overlap_penalty = self.validator.compute_overlap_penalty(inner_hex_data)

            if overlap_penalty > 0:
                return overlap_penalty * 10000  # Apply penalty for overlaps

            # If valid, return inverse of outer hexagon side length (negative for minimization)
            return -1.0 / outer_hex_side_length

        except Exception as e:
            # In case of any computation error, return large penalty
            return 1e10

    def generate_initial_guess(self):
        """Generate a good initial symmetric configuration."""
        positions = []

        # Central hexagon
        positions.append([0, 0, 0])

        # First ring (distance sqrt(3) from center)
        for i in range(6):
            angle = i * 60
            x = 1.732 * np.cos(np.radians(angle))  # ~= sqrt(3)
            y = 1.732 * np.sin(np.radians(angle))
            positions.append([x, y, 0])

        # Second ring (distance 2*sqrt(3) from center)
        for i in range(6):
            angle = i * 60 + 30  # offset
            x = 3.464 * np.cos(np.radians(angle))  # ~= 2*sqrt(3)
            y = 3.464 * np.sin(np.radians(angle))
            positions.append([x, y, 0])

        # Additional strategic placement
        positions.append([0, -4.5, 0])

        initial_config = np.array(positions[:12])

        # Add small random perturbations to avoid local minima
        np.random.seed(42)
        initial_config[:, :2] += np.random.normal(0, 0.1, (12, 2))

        return initial_config

    def optimize(self):
        """Perform the optimization process."""
        # Generate initial guess
        initial_guess_inner = self.generate_initial_guess()

        # Initial estimate for outer radius
        max_dist = 0
        for i in range(12):
            x, y, _ = initial_guess_inner[i]
            dist = np.sqrt(x*x + y*y)
            max_dist = max(max_dist, dist)

        # Add margin for hexagon size
        initial_outer_radius = max_dist + 2.0

        # Combine into single parameter vector
        initial_params = np.concatenate([initial_guess_inner.flatten(), [initial_outer_radius]])

        # Define bounds for optimization
        bounds = []
        for _ in range(12):
            bounds.extend([(-10, 10), (-10, 10), (0, 360)])
        bounds.append((0.1, 20.0))

        # Global optimization with reduced iterations for speed
        de_result = differential_evolution(
            self.evaluate_configuration,
            bounds,
            maxiter=30,
            popsize=10,
            seed=42,
            disp=False
        )

        # Local refinement
        best_params = de_result.x
        refined_bounds = [(b[0], b[1]) for b in bounds]

        try:
            local_result = minimize(
                self.evaluate_configuration,
                best_params,
                method='L-BFGS-B',
                bounds=refined_bounds,
                options={'maxiter': 50, 'disp': False}
            )
            if local_result.success:
                best_params = local_result.x
        except:
            pass

        # Extract final results
        inner_hex_data = best_params[:-1].reshape(12, 3)
        outer_hex_side_length = best_params[-1]

        # Validate final configuration
        if not self.validator.check_containment(inner_hex_data, outer_hex_side_length):
            # Try to correct by increasing outer radius
            min_outer_radius = 0
            for i in range(12):
                x, y, _ = inner_hex_data[i]
                dist = np.sqrt(x*x + y*y) + 1.0
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
    start_time = time.time()

    try:
        # Create optimizer instance
        optimizer = HexagonPackingOptimizer()

        # Get optimized configuration
        inner_hex_data, outer_hex_side_length = optimizer.optimize()

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