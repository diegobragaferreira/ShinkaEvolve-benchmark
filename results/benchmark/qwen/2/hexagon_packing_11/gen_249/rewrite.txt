# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid
from shapely.ops import unary_union
import warnings
import time
import multiprocessing as mp
from functools import partial
import random

# Constants
UNIT_HEX_RADIUS = 1.0
UNIT_HEX_APOGEE = np.sqrt(3)/2

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
        # Use a small buffer to avoid floating point precision issues
        buffered_inner = inner_hexagon.buffer(-1e-10)
        return outer_hexagon.contains(buffered_inner)

    @staticmethod
    def check_overlap(hex1, hex2):
        """Check if two hexagons overlap with buffer for precision."""
        # Use a small buffer to avoid floating point precision issues
        buffered_hex1 = hex1.buffer(1e-10)
        buffered_hex2 = hex2.buffer(1e-10)
        return buffered_hex1.intersects(buffered_hex2)

    @classmethod
    def validate_solution(cls, inner_params, outer_radius, n_inner=11):
        """Complete constraint validation with early termination."""
        inner_hexagons = []
        
        # Create inner hexagons
        for i in range(n_inner):
            x, y, angle = inner_params[3*i:3*i+3]
            hexagon = HexagonGeometry.create_unit_hexagon((x, y), angle)
            inner_hexagons.append(hexagon)

        # Create outer hexagon
        outer_hexagon = HexagonGeometry.create_unit_hexagon((0, 0), 0)
        outer_coords = list(outer_hexagon.exterior.coords)
        scaled_coords = [(x*outer_radius, y*outer_radius) for x, y in outer_coords]
        outer_hexagon_scaled = Polygon(scaled_coords)

        # Check containment (early termination)
        for hexagon in inner_hexagons:
            if not cls.check_containment(hexagon, outer_hexagon_scaled):
                return False, 0.0  # containment violated

        # Check overlaps (early termination)
        for i in range(n_inner):
            for j in range(i+1, n_inner):
                if cls.check_overlap(inner_hexagons[i], inner_hexagons[j]):
                    return False, 0.0  # overlap violated

        return True, 1.0 / outer_radius  # valid solution

class Initializer:
    """Generates diverse initial configurations for optimization."""
    
    @classmethod
    def generate_diverse_configs(cls, count=30):
        """Generate diverse initial positions based on multiple heuristic patterns."""
        configs = []
        
        # Pattern 1: Clustered arrangement - inspired by successful geometric layouts
        cluster_patterns = [
            # Center + ring pattern
            [(0.0, 0.0), (-1.8, 0.0), (1.8, 0.0), (0.0, 1.8), (0.0, -1.8),
             (-1.3, 1.3), (1.3, 1.3), (-1.3, -1.3), (1.3, -1.3),
             (-2.2, 0.0), (2.2, 0.0)],
            # Grid-like pattern  
            [(0.0, 0.0), (-2.0, 0.0), (2.0, 0.0), (0.0, 2.0), (0.0, -2.0),
             (-1.5, 1.5), (1.5, 1.5), (-1.5, -1.5), (1.5, -1.5),
             (-2.5, 0.0), (2.5, 0.0)]
        ]
        
        for pattern_idx, pattern in enumerate(cluster_patterns):
            for _ in range(count // len(cluster_patterns)):  # Split configs between patterns
                config = []
                for i, (cx, cy) in enumerate(pattern):
                    # Add small random variation
                    jitter_x = np.random.uniform(-0.2, 0.2)
                    jitter_y = np.random.uniform(-0.2, 0.2)
                    config.extend([cx + jitter_x, cy + jitter_y, np.random.uniform(0, 360)])
                # Add outer radius estimate
                config.append(4.0 + np.random.uniform(0.2, 0.8))
                configs.append(config)
        
        # Pattern 2: Spiral arrangement
        def generate_spiral_config():
            config = []
            # Center
            config.extend([0.0, 0.0, np.random.uniform(0, 360)])
            # Ring 1 (6 hexagons)
            for i in range(6):
                angle = i * np.pi/3
                radius = 1.9
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                config.extend([x, y, np.random.uniform(0, 360)])
            # Ring 2 (4 hexagons)
            for i in range(4):
                angle = i * np.pi/2
                radius = 3.2
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                config.extend([x, y, np.random.uniform(0, 360)])
            config.append(5.0 + np.random.uniform(0.1, 0.5))
            return config
            
        for _ in range(count // 3):
            configs.append(generate_spiral_config())
            
        return configs

class OptimizerStrategy:
    """Encapsulates optimization strategies for different phases."""
    
    def __init__(self, n_inner=11):
        self.n_inner = n_inner
        self.bounds = self._generate_bounds()
        
    def _generate_bounds(self):
        """Generate optimization bounds."""
        bounds = []
        # Bounds for inner hexagon positions and rotations
        for _ in range(self.n_inner):
            bounds.extend([(-6.0, 6.0), (-6.0, 6.0), (0, 360)])
        # Bound for outer radius
        bounds.append((3.0, 8.0))
        return bounds

    def objective_function(self, params):
        """Objective function to minimize: negative of 1/outer_radius."""
        outer_radius = params[-1]
        inner_params = params[:-1]

        # Check constraints
        valid, inv_radius = ConstraintValidator.validate_solution(
            inner_params, outer_radius, self.n_inner
        )

        # If constraint violated, return large penalty
        if not valid:
            return 10000.0 + abs(outer_radius)

        # Return negative of inverse radius to minimize (maximize 1/outer_radius)
        return -inv_radius

    def parallel_de_search(self, seeds=None):
        """Run multiple differential evolution instances in parallel."""
        if seeds is None:
            seeds = [42, 123, 456, 789]
            
        def run_de_instance(seed_val):
            try:
                result = differential_evolution(
                    self.objective_function,
                    self.bounds,
                    seed=seed_val,
                    maxiter=100,
                    popsize=20,
                    tol=1e-6,
                    mutation=(0.8, 1.0),
                    recombination=0.9,
                    disp=False
                )
                if result.success:
                    return result.x
            except Exception as e:
                warnings.warn(f"DE instance with seed {seed_val} failed: {str(e)}")
            return None

        with mp.Pool(processes=min(len(seeds), mp.cpu_count())) as pool:
            results = pool.map(run_de_instance, seeds)

        # Filter out None results
        valid_results = [r for r in results if r is not None]

        if not valid_results:
            return None

        # Find the best result based on objective function value
        best_result = min(valid_results, key=lambda x: self.objective_function(x))
        return best_result

    def local_refinement(self, initial_params):
        """Refine solution using local optimization after global search."""
        options = {'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}

        try:
            result = minimize(
                self.objective_function,
                initial_params,
                method='L-BFGS-B',
                bounds=self.bounds,
                options=options,
                callback=lambda x: None
            )

            if result.success:
                return result.x

        except Exception as e:
            warnings.warn(f"Local search failed: {str(e)}")

        return initial_params

class HexagonPackingOptimizer:
    """Main optimization coordinator implementing phase-based approach."""
    
    def __init__(self, n_inner=11):
        self.n_inner = n_inner
        self.strategy = OptimizerStrategy(n_inner)
        self.initializer = Initializer()
        
    def phase_one_initialization(self):
        """Generate diverse initial configurations."""
        return self.initializer.generate_diverse_configs(30)
        
    def phase_two_global_search(self, initial_configs):
        """Perform global optimization using parallel DE."""
        # First try parallel DE with various seeds
        best_params = self.strategy.parallel_de_search([42, 123, 456, 789])
        
        if best_params is None:
            # Fall back to single DE with enhanced parameters
            try:
                result = differential_evolution(
                    self.strategy.objective_function,
                    self.strategy.bounds,
                    seed=42,
                    maxiter=150,
                    popsize=25,
                    tol=1e-7,
                    mutation=(0.9, 1.0),
                    recombination=0.95,
                    disp=False
                )
                if result.success:
                    best_params = result.x
            except Exception as e:
                warnings.warn(f"Single DE optimization failed: {str(e)}")
                
        return best_params
        
    def phase_three_local_refinement(self, global_best):
        """Apply local refinement to global best solution."""
        if global_best is not None:
            return self.strategy.local_refinement(global_best)
        return None
        
    def phase_four_validation_and_selection(self, refined_params):
        """Validate final solution and return structured results."""
        if refined_params is None:
            return None
            
        inner_params = refined_params[:-1]
        outer_radius = refined_params[-1]
        
        # Validate solution
        valid, inv_radius = ConstraintValidator.validate_solution(
            inner_params, outer_radius, self.n_inner
        )
        
        if valid:
            # Format output
            inner_hex_data = np.zeros((self.n_inner, 3))
            for i in range(self.n_inner):
                inner_hex_data[i] = inner_params[3*i:3*i+3]
                
            outer_hex_data = np.array([0, 0, 0])
            
            return inner_hex_data, outer_hex_data, outer_radius
            
        return None

    def optimize(self):
        """Execute the complete optimization pipeline."""
        # Phase 1: Initialize diverse configurations
        initial_configs = self.phase_one_initialization()
        
        # Phase 2: Global search
        global_best = self.phase_two_global_search(initial_configs)
        
        # Phase 3: Local refinement  
        refined_best = self.phase_three_local_refinement(global_best)
        
        # Phase 4: Validation and output formatting
        result = self.phase_four_validation_and_selection(refined_best)
        
        return result

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
        # Set seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        optimizer = HexagonPackingOptimizer(11)
        result = optimizer.optimize()
        
        if result is not None:
            return result
        else:
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
            
    except Exception as e:
        warnings.warn(f"Error in optimization: {str(e)}")
        # Final fallback
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