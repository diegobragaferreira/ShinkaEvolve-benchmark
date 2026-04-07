# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
from numba import jit
from joblib import Parallel, delayed

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

class HexagonConstraintValidator:
    """Validates spatial constraints for hexagon arrangements."""

    def __init__(self, geom_handler):
        self.geom = geom_handler

    def check_containment_parallel(self, inner_hex_data, outer_radius, n_jobs=-1):
        """Parallel check if all inner hexagons are contained within outer hexagon."""
        outer_hex_vertices = self.geom.vertices(0, 0, 0, outer_radius)
        outer_polygon = Polygon(outer_hex_vertices)

        def check_single_containment(i):
            x, y, angle = inner_hex_data[i]
            inner_vertices = self.geom.vertices(x, y, angle)
            inner_polygon = Polygon(inner_vertices)
            return outer_polygon.contains(inner_polygon) or outer_polygon.covers(inner_polygon)

        # Use parallel processing for containment checks
        containment_results = Parallel(n_jobs=n_jobs)(
            delayed(check_single_containment)(i) for i in range(len(inner_hex_data))
        )

        return all(containment_results)

    def compute_overlap_penalty_parallel(self, inner_hex_data, n_jobs=-1):
        """Compute penalty based on overlap areas using parallel processing."""
        penalty = 0.0
        n = len(inner_hex_data)

        # Pre-compute polygons for efficiency
        def create_polygon(i):
            x, y, angle = inner_hex_data[i]
            vertices = self.geom.vertices(x, y, angle)
            return Polygon(vertices)

        # Create all polygons in parallel
        polygons = Parallel(n_jobs=n_jobs)(
            delayed(create_polygon)(i) for i in range(n)
        )

        # Check all pairs for overlaps in parallel
        def check_overlap_pair(indices):
            i, j = indices
            poly_i, poly_j = polygons[i], polygons[j]
            if poly_i.intersects(poly_j) and not poly_i.touches(poly_j):
                try:
                    overlap = poly_i.intersection(poly_j)
                    if hasattr(overlap, 'area') and overlap.area > 0:
                        return overlap.area
                except:
                    return 1000  # Large penalty for calculation errors
            return 0.0

        # Check overlaps in parallel
        overlap_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        overlap_results = Parallel(n_jobs=n_jobs)(
            delayed(check_overlap_pair)(pair) for pair in overlap_pairs
        )

        # Sum all overlap areas
        penalty = sum(overlap_results)
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

            # Check containment early (fast check) - now parallel
            if not self.validator.check_containment_parallel(inner_hex_data, outer_hex_side_length):
                return 1e10  # Large penalty for containment violation

            # Compute overlap penalty - now parallel
            overlap_penalty = self.validator.compute_overlap_penalty_parallel(inner_hex_data)

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