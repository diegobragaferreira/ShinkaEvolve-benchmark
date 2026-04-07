# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
from shapely.validation import make_valid
import warnings
import time
from joblib import Parallel, delayed
import functools

# Constants
UNIT_HEX_RADIUS = 1.0
UNIT_HEX_APOGEE = np.sqrt(3)/2

class HexagonGeometry:
    """Handles geometric operations for hexagons with caching"""
    
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

class ConstraintValidator:
    """Handles constraint validation for hexagon packing"""
    
    @staticmethod
    def check_containment(inner_hexagon, outer_hexagon):
        """Check if inner hexagon is fully contained within outer hexagon with buffer"""
        # Use a small buffer to avoid floating point precision issues
        buffered_inner = inner_hexagon.buffer(-1e-10)
        return outer_hexagon.contains(buffered_inner)

    @staticmethod
    def check_overlap(hex1, hex2):
        """Check if two hexagons overlap with buffer"""
        # Use a small buffer to avoid floating point precision issues
        buffered_hex1 = hex1.buffer(1e-10)
        buffered_hex2 = hex2.buffer(1e-10)
        return buffered_hex1.intersects(buffered_hex2)

class PackingEvaluator:
    """Evaluates packing configurations and constraints"""
    
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

    def evaluate_constraints(self, inner_params, outer_radius, n_hexagons=11):
        """Comprehensive constraint evaluation with early termination"""
        inner_hexagons = []
        
        # Create inner hexagons (cache-friendly)
        for i in range(n_hexagons):
            x, y, angle = inner_params[3*i:3*i+3]
            hexagon = self.geom_utils.create_unit_hexagon((x, y), angle)
            inner_hexagons.append(hexagon)

        # Create outer hexagon
        outer_hexagon = self.geom_utils.create_unit_hexagon((0, 0), 0)
        outer_coords = list(outer_hexagon.exterior.coords)
        scaled_coords = [(x*outer_radius, y*outer_radius) for x, y in outer_coords]
        outer_hexagon_scaled = Polygon(scaled_coords)

        # Check containment (early termination)
        for hexagon in inner_hexagons:
            if not self.validator.check_containment(hexagon, outer_hexagon_scaled):
                return False, False, 0.0  # containment violated

        # Check overlaps (early termination)
        for i in range(n_hexagons):
            for j in range(i+1, n_hexagons):
                if self.validator.check_overlap(inner_hexagons[i], inner_hexagons[j]):
                    return False, False, 0.0  # overlap violated

        # Calculate actual tight radius for better objective function
        actual_tight_radius = self.calculate_tight_outer_radius(inner_params, n_hexagons)
        return True, True, 1.0 / actual_tight_radius  # valid solution

class Optimizer:
    """Handles the optimization process with enhanced strategies"""
    
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
        # Base arrangement patterns - refined from multiple sources
        patterns = [
            # Pattern 1: Compact grid arrangement
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
            # Pattern 2: More spread out arrangement
            [
                (0.0, 0.0),       # center
                (-1.7, 0.0),      # left
                (1.7, 0.0),       # right
                (0.0, 1.7),       # top
                (0.0, -1.7),      # bottom
                (-1.2, 1.2),      # top-left
                (1.2, 1.2),       # top-right
                (-1.2, -1.2),     # bottom-left
                (1.2, -1.2),      # bottom-right
                (-2.0, 0.0),      # further left
                (2.0, 0.0),       # further right
            ],
            # Pattern 3: Ring arrangement with wider spacing
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
                (-2.3, 0.0),      # further left
                (2.3, 0.0),       # further right
            ],
            # Pattern 4: Hexagonal ring pattern optimized for 11 hexagons
            [
                (0.0, 0.0),       # center
                (-1.5, 0.0),      # left
                (1.5, 0.0),       # right
                (0.0, 1.5),       # top
                (0.0, -1.5),      # bottom
                (-1.0, 1.0),      # top-left
                (1.0, 1.0),       # top-right
                (-1.0, -1.0),     # bottom-left
                (1.0, -1.0),      # bottom-right
                (-2.5, 0.0),      # further left
                (2.5, 0.0),       # further right
            ]
        ]
        
        # Generate multiple diverse configurations
        configs = []
        for pattern in patterns:
            for _ in range(15):  # 15 variations per pattern
                config = []
                for i, (cx, cy) in enumerate(pattern):
                    # Add small random variation to avoid symmetric solutions
                    jitter_x = np.random.uniform(-0.25, 0.25)
                    jitter_y = np.random.uniform(-0.25, 0.25)
                    config.extend([cx + jitter_x, cy + jitter_y, np.random.uniform(0, 360)])
                config.append(4.0 + np.random.uniform(0.2, 0.8))  # outer radius estimate
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
                    maxiter=40,  # Reduced iterations for screening
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

        # Evaluate all configurations in parallel with better resource management
        results = Parallel(n_jobs=min(4, len(initial_configs)))(
            delayed(evaluate_config)(config) for config in initial_configs[:20]
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

        options = {'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}

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
        """Main optimization routine with enhanced strategy"""
        # Try parallel evaluation first with multiple configurations
        best_params = self.optimize_with_parallel_evaluation()
        
        # If found, refine with local search
        if best_params is not None:
            refined_params = self.refine_with_local_search(best_params)
            return refined_params
        
        # Fallback: try single optimization with default parameters but with improved settings
        try:
            bounds = []
            for _ in range(self.n_inner):
                bounds.extend([(-6.0, 6.0), (-6.0, 6.0), (0, 360)])
            bounds.append((3.0, 8.0))

            # Try with more iterations and better parameters
            result = differential_evolution(
                self.objective_function,
                bounds,
                seed=42,
                maxiter=120,
                popsize=25,
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
