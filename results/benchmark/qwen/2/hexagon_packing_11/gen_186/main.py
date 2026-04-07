# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid
import warnings
import time
import multiprocessing as mp
from functools import partial

# Constants
UNIT_HEX_RADIUS = 1.0  # Side length of unit hexagon
UNIT_HEX_APOGEE = np.sqrt(3)/2  # Distance from center to corner of unit hexagon

class HexagonGeometry:
    """Handles all geometric operations for hexagon creation and manipulation."""
    
    @staticmethod
    def create_unit_hexagon(center=(0,0), rotation=0):
        """Create a unit regular hexagon as a Shapely Polygon."""
        angle_offset = np.deg2rad(rotation)
        points = []
        for i in range(6):
            angle = angle_offset + i * np.pi/3
            x = center[0] + UNIT_HEX_RADIUS * np.cos(angle)
            y = center[1] + UNIT_HEX_RADIUS * np.sin(angle)
            points.append((x, y))
        return Polygon(points)

    @staticmethod
    def get_all_vertices(hexagon):
        """Extract all vertices from a hexagon polygon."""
        return list(hexagon.exterior.coords)[:-1]  # Exclude closing point

class ConstraintValidator:
    """Handles all constraint checking operations for hexagon packing."""
    
    @staticmethod
    def check_containment(inner_hexagon, outer_hexagon):
        """Check if inner hexagon is fully contained within outer hexagon."""
        # Use a tighter buffer to avoid floating point precision issues
        buffered_inner = inner_hexagon.buffer(-1e-12)
        return outer_hexagon.contains(buffered_inner)

    @staticmethod
    def check_overlap(hex1, hex2):
        """Check if two hexagons overlap with buffer for precision."""
        # Use a tighter buffer to detect real overlaps
        buffered_hex1 = hex1.buffer(1e-12)
        buffered_hex2 = hex2.buffer(1e-12)
        return buffered_hex1.intersects(buffered_hex2)

    @classmethod
    def evaluate_constraints(cls, inner_params, outer_radius, n_inner=11):
        """Comprehensive constraint evaluation with early termination."""
        # Create inner hexagons
        inner_hexagons = []
        for i in range(n_inner):
            x, y, angle = inner_params[3*i:3*i+3]
            hexagon = HexagonGeometry.create_unit_hexagon((x, y), angle)
            inner_hexagons.append(hexagon)

        # Create outer hexagon
        outer_hexagon = HexagonGeometry.create_unit_hexagon((0, 0), 0)
        outer_coords = list(outer_hexagon.exterior.coords)
        scaled_coords = [(x*outer_radius, y*outer_radius) for x, y in outer_coords]
        outer_hexagon_scaled = Polygon(scaled_coords)

        # Check containment (early termination) - verify all vertices are contained
        for hexagon in inner_hexagons:
            # Check each vertex individually for better precision
            vertices = HexagonGeometry.get_all_vertices(hexagon)
            for vertex in vertices:
                point = Point(vertex)
                if not outer_hexagon_scaled.contains(point):
                    return False, False, 0.0  # containment violated

        # Check overlaps (early termination) - more robust overlap detection
        for i in range(n_inner):
            for j in range(i+1, n_inner):
                if cls.check_overlap(inner_hexagons[i], inner_hexagons[j]):
                    return False, False, 0.0  # overlap violated

        # Calculate tight radius for better objective function
        return True, True, 1.0 / outer_radius  # valid solution

class OuterRadiusCalculator:
    """Handles outer radius calculation optimizations."""
    
    @staticmethod
    def calculate_tight_outer_radius(inner_params, n_inner=11):
        """Calculate tightest possible outer hexagon radius using actual vertex positions."""
        # Get all hexagon vertices and find bounding circle
        all_vertices = []

        for i in range(n_inner):
            x, y, angle = inner_params[3*i:3*i+3]
            hexagon = HexagonGeometry.create_unit_hexagon((x, y), angle)
            # Get all vertices of this hexagon
            for point in HexagonGeometry.get_all_vertices(hexagon):
                all_vertices.append(point)

        if not all_vertices:
            return 1.0

        # Convert to numpy array for easier computation
        vertices_array = np.array(all_vertices)

        # Find centroid of all vertices
        centroid = np.mean(vertices_array, axis=0)

        # Calculate distances from centroid to all vertices
        distances = np.sqrt(np.sum((vertices_array - centroid)**2, axis=1))

        # Use adaptive margin based on the average distance and number of hexagons
        avg_distance = np.mean(distances)
        max_distance = np.max(distances)

        # Adaptive margin: smaller for tightly packed arrangements, larger for scattered ones
        # Scale margin based on how much the distribution deviates from uniform
        deviation_ratio = max_distance / (avg_distance + 1e-10)  # Avoid division by zero
        adaptive_margin = 1e-8 + (0.005 * deviation_ratio)  # Dynamic margin

        # Outer radius is the maximum distance plus adaptive margin for numerical stability
        outer_radius = max_distance + adaptive_margin

        return outer_radius

class InitialConfigGenerator:
    """Generates diverse initial configurations for optimization."""
    
    @classmethod
    def generate_strategic_configs(cls):
        """Generate optimized initial positions based on known good packing patterns."""
        configs = []

        # Pattern 1: Spiral arrangement
        def generate_spiral_pattern():
            positions = []
            # Center
            positions.append((0.0, 0.0))
            # First ring (6 hexagons around center)
            for i in range(6):
                angle = i * np.pi/3
                radius = 1.85  # slightly more than the hexagon diameter
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                positions.append((x, y))
            # Second ring (additional hexagons)
            for i in range(4):
                angle = i * np.pi/2
                radius = 3.1
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                positions.append((x, y))
            return positions[:11]  # Keep only 11 positions

        # Pattern 2: Grid-based with careful spacing
        def generate_grid_pattern():
            positions = [
                (0.0, 0.0),       # center
                (-2.0, 0.0),      # left
                (2.0, 0.0),       # right
                (0.0, 2.0),       # top
                (0.0, -2.0),      # bottom
                (-1.5, 1.5),      # top-left
                (1.5, 1.5),       # top-right
                (-1.5, -1.5),     # bottom-left
                (1.5, -1.5),      # bottom-right
                (-2.5, 0.0),      # further left
                (2.5, 0.0),       # further right
            ]
            return positions

        # Pattern 3: Optimized cluster arrangement
        def generate_cluster_pattern():
            positions = [
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
            return positions

        # Generate multiple variations of each pattern
        patterns = [generate_spiral_pattern(), generate_grid_pattern(), generate_cluster_pattern()]
        for pattern_func in patterns:
            pattern = pattern_func()
            for _ in range(15):  # 15 configs per pattern
                config = []
                for i, (cx, cy) in enumerate(pattern):
                    # Add random perturbations but keep within reasonable bounds
                    jitter_x = np.random.uniform(-0.25, 0.25)
                    jitter_y = np.random.uniform(-0.25, 0.25)
                    config.extend([cx + jitter_x, cy + jitter_y, np.random.uniform(0, 360)])
                config.append(4.0 + np.random.uniform(0.1, 0.5))  # outer radius estimate
                configs.append(config)

        return configs

class Optimizer:
    """Main optimization class orchestrating the entire process."""
    
    def __init__(self):
        self.n_inner = 11
        self.bounds = self._generate_bounds()
        self.initial_configs = InitialConfigGenerator.generate_strategic_configs()

    def _generate_bounds(self):
        """Generate optimization bounds."""
        bounds = []
        # Bounds for inner hexagon positions and rotations
        for _ in range(self.n_inner):
            bounds.extend([(-7.0, 7.0), (-7.0, 7.0), (0, 360)])  # x, y, angle
        # Bound for outer radius
        bounds.append((3.2, 7.5))  # Reasonable range for outer radius
        return bounds

    def objective_function(self, params):
        """Objective function to minimize: negative of 1/outer_radius (i.e., maximize 1/outer_radius)."""
        outer_radius = params[-1]
        inner_params = params[:-1]

        # Check constraints
        containment_ok, overlap_ok, inv_radius = ConstraintValidator.evaluate_constraints(
            inner_params, outer_radius, self.n_inner
        )

        # If any constraint violated, return large penalty
        if not (containment_ok and overlap_ok):
            return 10000.0 + abs(outer_radius)  # penalty for constraint violations

        # Return negative of inverse radius to minimize (maximize 1/outer_radius)
        return -inv_radius

    def local_refinement(self, initial_params):
        """Refine solution using local optimization after global search."""
        # Use tighter bounds for local search
        local_bounds = []
        for _ in range(self.n_inner):
            local_bounds.extend([(-6.0, 6.0), (-6.0, 6.0), (0, 360)])  # x, y, angle
        local_bounds.append((3.0, 8.0))  # Outer radius bounds

        options = {'maxiter': 1500, 'ftol': 1e-10, 'gtol': 1e-10}

        try:
            result = minimize(
                self.objective_function,
                initial_params,
                method='L-BFGS-B',
                bounds=local_bounds,
                options=options,
                callback=lambda x: None  # Empty callback
            )

            if result.success:
                return result.x

        except Exception as e:
            warnings.warn(f"Local search failed: {str(e)}")

        return initial_params

    def parallel_de_optimization(self):
        """Run multiple differential evolution instances in parallel."""
        def run_de_instance(seed_val):
            try:
                result = differential_evolution(
                    self.objective_function,
                    self.bounds,
                    seed=seed_val,
                    maxiter=80,  # Reduced iterations for faster screening
                    popsize=15,   # Smaller population for faster screening
                    tol=1e-6,     # Tighter tolerance
                    mutation=(0.6, 1.0),  # Moderate mutation rate
                    recombination=0.8,    # Moderate recombination
                    disp=False
                )
                if result.success:
                    return result.x
            except Exception as e:
                warnings.warn(f"DE instance with seed {seed_val} failed: {str(e)}")
            return None

        # Run multiple instances in parallel
        seeds = [42, 123, 456, 789, 101]  # Different random seeds
        with mp.Pool(processes=min(len(seeds), mp.cpu_count())) as pool:
            results = pool.map(run_de_instance, seeds)

        # Filter out None results
        valid_results = [r for r in results if r is not None]

        if not valid_results:
            return None

        # Find the best result based on objective function value
        best_result = min(valid_results, key=lambda x: self.objective_function(x))
        return best_result

    def optimize_solution(self):
        """Main optimization routine."""
        # Try parallel differential evolution first
        best_params = self.parallel_de_optimization()

        # If parallel approach failed, fall back to single optimization with better settings
        if best_params is None:
            try:
                result = differential_evolution(
                    self.objective_function,
                    self.bounds,
                    seed=42,
                    maxiter=120,   # Increased iterations for more thorough search
                    popsize=25,    # Larger population
                    tol=1e-7,      # Tighter tolerance
                    mutation=(0.8, 1.0),  # Higher mutation rate
                    recombination=0.9,    # High recombination
                    disp=False
                )

                if result.success:
                    best_params = result.x
            except Exception as e:
                warnings.warn(f"Single DE optimization failed: {str(e)}")
                pass

        # Refine with local search if we have a good candidate
        if best_params is not None:
            refined_params = self.local_refinement(best_params)
            return refined_params

        # Return default initial guess if optimization fails
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

        initial_guess.append(4.1)  # Initial outer radius estimate
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
    try:
        optimizer = Optimizer()
        final_params = optimizer.optimize_solution()

        # Extract results
        n = 11
        inner_params = final_params[:-1]
        outer_radius = final_params[-1]

        # Validate solution
        containment_ok, overlap_ok, inv_radius = ConstraintValidator.evaluate_constraints(
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