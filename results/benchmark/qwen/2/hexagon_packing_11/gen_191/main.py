# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid
import warnings
import time
from joblib import Parallel, delayed
import functools
import copy

# Constants
UNIT_HEX_RADIUS = 1.0
UNIT_HEX_APOGEE = np.sqrt(3)/2

class HexagonGeometry:
    """Handles geometric operations for hexagons with caching for performance"""
    
    @staticmethod
    @functools.lru_cache(maxsize=1000)
    def create_unit_hexagon(center=(0,0), rotation=0):
        """Create a unit regular hexagon as a Shapely Polygon with caching"""
        angle_offset = np.deg2rad(rotation)
        points = []
        for i in range(6):
            angle = angle_offset + i * np.pi/3
            x = center[0] + UNIT_HEX_RADIUS * np.cos(angle)
            y = center[1] + UNIT_HEX_RADIUS * np.sin(angle)
            points.append((x, y))
        return Polygon(points)

    @staticmethod
    def validate_polygon(polygon):
        """Ensure polygon is valid for geometric operations"""
        if not polygon.is_valid:
            return make_valid(polygon)
        return polygon

    @staticmethod
    def hexagon_vertices(center=(0,0), rotation=0):
        """Get vertices of a unit hexagon for SAT calculations"""
        angle_offset = np.deg2rad(rotation)
        vertices = []
        for i in range(6):
            angle = angle_offset + i * np.pi/3
            x = center[0] + UNIT_HEX_RADIUS * np.cos(angle)
            y = center[1] + UNIT_HEX_RADIUS * np.sin(angle)
            vertices.append((x, y))
        return vertices

    @staticmethod
    def get_hexagon_edges(vertices):
        """Get edges from vertices for SAT calculation"""
        edges = []
        for i in range(len(vertices)):
            p1 = np.array(vertices[i])
            p2 = np.array(vertices[(i+1) % len(vertices)])
            edge = p2 - p1
            edges.append(edge)
        return edges

class ConstraintValidator:
    """Handles constraint validation for hexagon packing with optimized algorithms"""
    
    @staticmethod
    def check_containment(inner_hexagon, outer_hexagon):
        """Check if inner hexagon is fully contained within outer hexagon with buffer"""
        # Use a small buffer to avoid floating point precision issues
        buffered_inner = inner_hexagon.buffer(-1e-10)
        return outer_hexagon.contains(buffered_inner)

    @staticmethod
    def check_overlap_sat(hex1_vertices, hex2_vertices):
        """Check overlap using Separating Axis Theorem for better precision"""
        # Early rejection using bounding boxes
        min1 = np.min(hex1_vertices, axis=0)
        max1 = np.max(hex1_vertices, axis=0)
        min2 = np.min(hex2_vertices, axis=0)
        max2 = np.max(hex2_vertices, axis=0)

        if max1[0] < min2[0] or max2[0] < min1[0] or max1[1] < min2[1] or max2[1] < min1[1]:
            return False

        # Get edges of both hexagons
        edges1 = HexagonGeometry.get_hexagon_edges(hex1_vertices)
        edges2 = HexagonGeometry.get_hexagon_edges(hex2_vertices)

        # Combine all axes
        all_axes = edges1 + edges2

        # Normalize axes
        for i, axis in enumerate(all_axes):
            norm = np.linalg.norm(axis)
            if norm > 1e-10:
                all_axes[i] = axis / norm

        # Project both polygons onto each axis
        for axis in all_axes:
            # Project hex1
            projections1 = [np.dot(vertex, axis) for vertex in hex1_vertices]
            min1_proj, max1_proj = min(projections1), max(projections1)

            # Project hex2
            projections2 = [np.dot(vertex, axis) for vertex in hex2_vertices]
            min2_proj, max2_proj = min(projections2), max(projections2)

            # Check for separation
            if max1_proj < min2_proj or max2_proj < min1_proj:
                return False  # Separating axis found, no overlap

        return True  # No separating axis found, overlap exists

    @staticmethod
    def check_containment_precise(inner_vertices, outer_vertices):
        """Precise containment check using vertex-by-vertex test with buffer"""
        outer_polygon = Polygon(outer_vertices)
        # Use a small buffer to avoid floating point precision issues
        buffered_outer = outer_polygon.buffer(1e-8)
        for vertex in inner_vertices:
            point = Point(vertex)
            if not buffered_outer.contains(point):
                return False
        return True

class PackingEvaluator:
    """Evaluates packing configurations and constraints with performance optimizations"""
    
    def __init__(self):
        self.geom_utils = HexagonGeometry()
        self.validator = ConstraintValidator()

    def calculate_tight_outer_radius(self, inner_params, n_hexagons=11):
        """Calculate tightest possible outer hexagon radius using actual vertex positions"""
        # Get all hexagon vertices and find bounding circle
        all_vertices = []

        for i in range(n_hexagons):
            x, y, angle = inner_params[3*i:3*i+3]
            hexagon = self.geom_utils.create_unit_hexagon((x, y), angle)
            # Get all vertices of this hexagon
            for point in hexagon.exterior.coords[:-1]:  # exclude closing point
                all_vertices.append(point)

        if not all_vertices:
            return 1.0

        # Convert to numpy array for easier computation
        vertices_array = np.array(all_vertices)

        # Find centroid of all vertices
        centroid = np.mean(vertices_array, axis=0)

        # Calculate distances from centroid to all vertices
        distances = np.sqrt(np.sum((vertices_array - centroid)**2, axis=1))

        # Outer radius is the maximum distance plus a small margin for numerical stability
        outer_radius = np.max(distances) + 1e-6

        return outer_radius

    def calculate_better_tight_radius(self, inner_params, n_hexagons=11):
        """Calculate a better tight radius with less conservative margins"""
        # Get all hexagon vertices and compute actual bounding box
        all_vertices = []

        for i in range(n_hexagons):
            x, y, angle = inner_params[3*i:3*i+3]
            hexagon = self.geom_utils.create_unit_hexagon((x, y), angle)
            # Get all vertices of this hexagon
            for point in hexagon.exterior.coords[:-1]:  # exclude closing point
                all_vertices.append(point)

        if not all_vertices:
            return 1.0

        # Convert to numpy array for easier computation
        vertices_array = np.array(all_vertices)

        # Calculate bounding box directly
        min_x, max_x = np.min(vertices_array[:, 0]), np.max(vertices_array[:, 0])
        min_y, max_y = np.min(vertices_array[:, 1]), np.max(vertices_array[:, 1])

        # Calculate center of bounding box
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        # Calculate distance from center to corners of bounding box
        # This provides a much tighter bound than using centroids
        distances = []
        for vertex in vertices_array:
            dist = np.sqrt((vertex[0] - center_x)**2 + (vertex[1] - center_y)**2)
            distances.append(dist)

        # Use the maximum distance from center to any vertex
        outer_radius = np.max(distances) * 1.05  # 5% margin for safety

        return outer_radius

    def evaluate_constraints(self, inner_params, outer_radius, n_hexagons=11):
        """Comprehensive constraint evaluation with early termination"""
        # Pre-compute all inner hexagons for reuse
        inner_hexagons = []
        inner_vertices_list = []

        # Create inner hexagons (cache-friendly) and collect vertices
        for i in range(n_hexagons):
            x, y, angle = inner_params[3*i:3*i+3]
            hexagon = self.geom_utils.create_unit_hexagon((x, y), angle)
            inner_hexagons.append(hexagon)
            
            # Also collect vertices for SAT checks
            vertices = self.geom_utils.hexagon_vertices((x, y), angle)
            inner_vertices_list.append(vertices)

        # Create outer hexagon (single instance)
        outer_hexagon = self.geom_utils.create_unit_hexagon((0, 0), 0)
        outer_coords = list(outer_hexagon.exterior.coords)
        scaled_coords = [(x*outer_radius, y*outer_radius) for x, y in outer_coords]
        outer_hexagon_scaled = Polygon(scaled_coords)

        # Check containment (early termination) - using precise checking
        for i, hexagon in enumerate(inner_hexagons):
            if not self.validator.check_containment(hexagon, outer_hexagon_scaled):
                return False, False, 0.0  # containment violated

        # Check overlaps (early termination) - using SAT for better precision
        for i in range(n_hexagons):
            for j in range(i+1, n_hexagons):
                if self.validator.check_overlap_sat(inner_vertices_list[i], inner_vertices_list[j]):
                    return False, False, 0.0  # overlap violated

        # Calculate actual tight radius for better objective function
        actual_tight_radius = self.calculate_better_tight_radius(inner_params, n_hexagons)
        return True, True, 1.0 / actual_tight_radius  # valid solution

class Optimizer:
    """Handles the optimization process with improved strategy"""
    
    def __init__(self, n_inner=11):
        self.n_inner = n_inner
        self.evaluator = PackingEvaluator()

    def objective_function(self, params):
        """Objective function to minimize: negative of 1/outer_radius (i.e., maximize 1/outer_radius)"""
        # params: [x1,y1,a1, x2,y2,a2, ..., x11,y11,a11, outer_radius]
        outer_radius = params[-1]

        # Extract inner hexagon parameters
        inner_params = params[:-1]

        # Check constraints
        containment_ok, overlap_ok, inv_radius = self.evaluator.evaluate_constraints(
            inner_params, outer_radius, self.n_inner
        )

        # If any constraint violated, return large penalty
        if not (containment_ok and overlap_ok):
            return 10000.0 + abs(outer_radius)  # penalty for constraint violations

        # Return negative of inverse radius to minimize (maximize 1/outer_radius)
        return -inv_radius

    def get_initial_positions(self):
        """Generate diverse initial positions based on known good patterns"""
        # Base arrangement patterns - more optimized for better packing
        patterns = [
            # Pattern 1: Optimized compact hexagonal packing
            [
                (0.0, 0.0),       # center
                (-1.732, 0.0),      # left (sqrt(3) ≈ 1.732)
                (1.732, 0.0),       # right
                (0.0, 1.732),       # top
                (0.0, -1.732),      # bottom
                (-0.866, 0.866),      # top-left (sqrt(3)/2 ≈ 0.866)
                (0.866, 0.866),       # top-right
                (-0.866, -0.866),     # bottom-left
                (0.866, -0.866),      # bottom-right
                (-2.598, 0.0),      # further left (3*sqrt(3)/2 ≈ 2.598)
                (2.598, 0.0),       # further right
            ],
            # Pattern 2: Optimized spiral-like arrangement
            [
                (0.0, 0.0),       # center
                (-1.8, 0.0),      # left
                (1.8, 0.0),       # right
                (0.0, 1.8),       # top
                (0.0, -1.8),      # bottom
                (-1.3, 1.3),      # top-left
                (1.3, 1.3),       # top-right
                (-1.3, -1.3),     # bottom-left
                (1.3, -1.3),      # bottom-right
                (-2.2, 0.0),      # further left
                (2.2, 0.0),       # further right
            ],
            # Pattern 3: High-density ring arrangement
            [
                (0.0, 0.0),       # center
                (-1.9, 0.0),      # left
                (1.9, 0.0),       # right
                (0.0, 1.9),       # top
                (0.0, -1.9),      # bottom
                (-1.4, 1.4),      # top-left
                (1.4, 1.4),       # top-right
                (-1.4, -1.4),     # bottom-left
                (1.4, -1.4),      # bottom-right
                (-2.6, 0.0),      # further left
                (2.6, 0.0),       # further right
            ],
            # Pattern 4: Optimized 6-ring pattern
            [
                (0.0, 0.0),       # center
                (-1.78, 0.0),      # left
                (1.78, 0.0),       # right
                (0.0, 1.78),       # top
                (0.0, -1.78),      # bottom
                (-1.25, 1.25),     # top-left
                (1.25, 1.25),      # top-right
                (-1.25, -1.25),    # bottom-left
                (1.25, -1.25),     # bottom-right
                (-2.5, 0.0),       # further left
                (2.5, 0.0),       # further right
            ]
        ]

        # Generate multiple diverse configurations
        configs = []
        for pattern in patterns:
            for _ in range(10):  # 10 variations per pattern for better exploration
                config = []
                for i, (cx, cy) in enumerate(pattern):
                    # Add small random variation with careful control
                    jitter_x = np.random.uniform(-0.15, 0.15)
                    jitter_y = np.random.uniform(-0.15, 0.15)
                    config.extend([cx + jitter_x, cy + jitter_y, np.random.uniform(0, 360)])
                # Use a more accurate initial radius estimate
                config.append(3.8 + np.random.uniform(0.1, 0.5))
                configs.append(config)

        return configs

    def optimize_with_parallel_evaluation(self):
        """Optimize using parallel evaluation of multiple initial configurations"""
        # Generate bounds for optimization
        bounds = []
        # Bounds for inner hexagon positions and rotations
        for _ in range(self.n_inner):
            bounds.extend([(-6.0, 6.0), (-6.0, 6.0), (0, 360)])  # x, y, angle
        # Bound for outer radius
        bounds.append((3.0, 8.0))  # Reasonable range for outer radius

        # Generate multiple initial configurations
        initial_configs = self.get_initial_positions()

        # Create a wrapper function for parallel execution
        def evaluate_config(config):
            try:
                # Use DE with fewer iterations for faster screening
                result = differential_evolution(
                    self.objective_function,
                    bounds,
                    seed=42,
                    maxiter=25,  # Reduced iterations for faster screening
                    popsize=15,
                    tol=1e-5,
                    mutation=(0.5, 1.0),
                    recombination=0.7,
                    disp=False
                )

                if result.success:
                    return result.x
            except Exception:
                return None
            return None

        # Evaluate all configurations in parallel (limit to 4 jobs for resource management)
        results = Parallel(n_jobs=min(4, len(initial_configs)))(
            delayed(evaluate_config)(config) for config in initial_configs[:12]  # Limit number of configs
        )

        # Filter out None results and find the best
        valid_results = [r for r in results if r is not None]
        if not valid_results:
            return None

        # Find the best result based on objective function value
        best_params = None
        best_value = float('inf')

        for params in valid_results:
            try:
                obj_value = self.objective_function(params)
                if obj_value < best_value:
                    best_value = obj_value
                    best_params = params
            except Exception:
                continue

        return best_params

    def refine_with_local_search(self, initial_params):
        """Refine solution using local optimization after global search"""
        bounds = []
        # Bounds for inner hexagon positions and rotations
        for _ in range(self.n_inner):
            bounds.extend([(-6.0, 6.0), (-6.0, 6.0), (0, 360)])  # x, y, angle
        # Bound for outer radius
        bounds.append((3.0, 8.0))  # Reasonable range for outer radius

        options = {'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}

        try:
            # Use L-BFGS-B for fine-tuning with stricter tolerances
            result = minimize(
                self.objective_function,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options=options
            )

            if result.success:
                return result.x

        except Exception as e:
            warnings.warn(f"Local search failed: {str(e)}")

        return initial_params

    def optimize_solution(self):
        """Main optimization routine with improved strategy"""
        # Try parallel evaluation first
        best_params = self.optimize_with_parallel_evaluation()

        # If found, refine with local search
        if best_params is not None:
            refined_params = self.refine_with_local_search(best_params)
            return refined_params

        # Fallback: try single optimization with default parameters
        try:
            bounds = []
            for _ in range(self.n_inner):
                bounds.extend([(-6.0, 6.0), (-6.0, 6.0), (0, 360)])
            bounds.append((3.0, 8.0))

            result = differential_evolution(
                self.objective_function,
                bounds,
                seed=42,
                maxiter=60,
                popsize=20,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                disp=False
            )

            if result.success:
                refined_params = self.refine_with_local_search(result.x)
                return refined_params

        except Exception as e:
            warnings.warn(f"Optimization failed: {str(e)}")

        # Return default initial guess if everything fails
        initial_guess = []
        centers = [
            (0.0, 0.0),       # center
            (-1.8, 0.0),      # left
            (1.8, 0.0),       # right
            (0.0, 1.8),       # top
            (0.0, -1.8),      # bottom
            (-1.3, 1.3),      # top-left
            (1.3, 1.3),       # top-right
            (-1.3, -1.3),     # bottom-left
            (1.3, -1.3),      # bottom-right
            (-2.2, 0.0),      # further left
            (2.2, 0.0),       # further right
        ]

        for i, (cx, cy) in enumerate(centers):
            initial_guess.extend([cx, cy, np.random.uniform(0, 360)])

        initial_guess.append(4.0)  # Initial outer radius estimate
        return initial_guess

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses evolutionary optimization to find the best arrangement.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    optimizer = Optimizer(n_inner=11)

    try:
        # Run optimization
        final_params = optimizer.optimize_solution()

        # Extract results
        n = 11
        inner_params = final_params[:-1]
        outer_radius = final_params[-1]

        # Validate solution (re-evaluate constraints)
        evaluator = PackingEvaluator()
        containment_ok, overlap_ok, inv_radius = evaluator.evaluate_constraints(
            inner_params, outer_radius, n
        )

        if containment_ok and overlap_ok:
            # Format output
            inner_hex_data = np.zeros((n, 3))
            for i in range(n):
                inner_hex_data[i] = inner_params[3*i:3*i+3]

            outer_hex_data = np.array([0, 0, 0])

            return inner_hex_data, outer_hex_data, outer_radius

    except Exception as e:
        warnings.warn(f"Error in optimization: {str(e)}")
        pass

    # Fallback to original method if optimization fails
    inner_hex_data = np.array([
        [0, 0, 0],        # center
        [-2.5, 0, 0],     # left
        [2.5, 0, 0],      # right
        [-1.25, 2.17, 0], # top-left
        [1.25, 2.17, 0],  # top-right
        [-1.25, -2.17, 0], # bottom-left
        [1.25, -2.17, 0], # bottom-right
        [-3.75, 2.17, 0], # far top-left
        [3.75, 2.17, 0],  # far top-right
        [-3.75, -2.17, 0], # far bottom-left
        [3.75, -2.17, 0], # far bottom-right
    ])

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 8  # large enough to contain all inner hexagons

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END