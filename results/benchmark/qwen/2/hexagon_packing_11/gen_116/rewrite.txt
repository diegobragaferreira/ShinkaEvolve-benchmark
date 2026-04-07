# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid
import warnings
import time
from typing import Tuple, List, Optional, Any
import itertools

# Constants
UNIT_HEX_RADIUS = 1.0  # Side length of unit hexagon
UNIT_HEX_APOGEE = np.sqrt(3)/2  # Distance from center to corner of unit hexagon

class HexagonUtils:
    """Utility class for hexagon-related geometric operations"""
    
    @staticmethod
    def create_unit_hexagon(center: Tuple[float, float] = (0,0), rotation: float = 0) -> Polygon:
        """Create a unit regular hexagon as a Shapely Polygon"""
        angle_offset = np.deg2rad(rotation)
        points = []
        for i in range(6):
            angle = angle_offset + i * np.pi/3
            x = center[0] + UNIT_HEX_RADIUS * np.cos(angle)
            y = center[1] + UNIT_HEX_RADIUS * np.sin(angle)
            points.append((x, y))
        return Polygon(points)

    @staticmethod
    def validate_polygon(polygon) -> Polygon:
        """Ensure polygon is valid for geometric operations"""
        if not polygon.is_valid:
            return make_valid(polygon)
        return polygon

    @staticmethod
    def check_containment(inner_hexagon: Polygon, outer_hexagon: Polygon) -> bool:
        """Check if inner hexagon is fully contained within outer hexagon with buffer for precision"""
        # Use a small buffer to avoid floating point precision issues
        buffered_inner = inner_hexagon.buffer(-1e-10)
        return outer_hexagon.contains(buffered_inner)

    @staticmethod
    def check_overlap(hex1: Polygon, hex2: Polygon) -> bool:
        """Check if two hexagons overlap with buffer for precision"""
        # Use a small buffer to avoid floating point precision issues
        buffered_hex1 = hex1.buffer(1e-10)
        buffered_hex2 = hex2.buffer(1e-10)
        return buffered_hex1.intersects(buffered_hex2)

    @staticmethod
    def calculate_tight_outer_radius(inner_params: np.ndarray) -> float:
        """Calculate tightest possible outer hexagon radius using actual vertex positions"""
        # Get all hexagon vertices and find bounding circle
        all_vertices = []

        for i in range(11):  # 11 inner hexagons
            x, y, angle = inner_params[3*i:3*i+3]
            hexagon = HexagonUtils.create_unit_hexagon((x, y), angle)
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

class HexagonInitializer:
    """Handles generation of initial configurations for optimization"""
    
    @staticmethod
    def generate_spiral_pattern() -> List[Tuple[float, float]]:
        """Generate a spiral-based initial pattern"""
        positions = []
        # Center
        positions.append((0.0, 0.0))
        # First ring (6 hexagons around center)
        for i in range(6):
            angle = i * np.pi/3
            radius = 1.9  # slightly more than the hexagon diameter
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions.append((x, y))
        # Second ring (additional hexagons)
        for i in range(4):
            angle = i * np.pi/2
            radius = 3.2
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions.append((x, y))
        return positions[:11]  # Keep only 11 positions

    @staticmethod
    def generate_grid_pattern() -> List[Tuple[float, float]]:
        """Generate a grid-based initial pattern"""
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

    @staticmethod
    def generate_cluster_pattern() -> List[Tuple[float, float]]:
        """Generate a cluster-based initial pattern"""
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

    @classmethod
    def generate_initial_solutions(cls) -> List[np.ndarray]:
        """Generate multiple initial configurations"""
        configs = []
        patterns = [cls.generate_spiral_pattern(), cls.generate_grid_pattern(), cls.generate_cluster_pattern()]
        
        for pattern_func in patterns:
            pattern = pattern_func()
            for _ in range(10):  # 10 configs per pattern
                config = []
                for i, (cx, cy) in enumerate(pattern):
                    # Add random perturbations but keep within reasonable bounds
                    jitter_x = np.random.uniform(-0.3, 0.3)
                    jitter_y = np.random.uniform(-0.3, 0.3)
                    config.extend([cx + jitter_x, cy + jitter_y, np.random.uniform(0, 360)])
                config.append(4.0 + np.random.uniform(0.1, 0.6))  # outer radius estimate
                configs.append(np.array(config))
        
        return configs

class OptimizerPipeline:
    """Manages the complete optimization pipeline"""
    
    def __init__(self):
        self.bounds = self._setup_bounds()
        self.n_inner = 11
        
    def _setup_bounds(self) -> List[Tuple[float, float]]:
        """Setup optimization bounds"""
        bounds = []
        # Bounds for inner hexagon positions and rotations
        for _ in range(self.n_inner):
            bounds.extend([(-8.0, 8.0), (-8.0, 8.0), (0, 360)])  # x, y, angle
        # Bound for outer radius
        bounds.append((3.5, 7.0))  # Reasonable range for outer radius
        return bounds
    
    def objective_function(self, params: np.ndarray) -> float:
        """Objective function to minimize: negative of 1/outer_radius (i.e., maximize 1/outer_radius)"""
        # params: [x1,y1,a1, x2,y2,a2, ..., x11,y11,a11, outer_radius]
        outer_radius = params[-1]
        inner_params = params[:-1]

        # Early constraint validation
        if not self._validate_parameters(inner_params, outer_radius):
            return 10000.0 + abs(outer_radius)  # penalty for constraint violations

        # Calculate actual tight radius for better objective function  
        actual_tight_radius = HexagonUtils.calculate_tight_outer_radius(inner_params)
        # Return negative of inverse radius to minimize (maximize 1/outer_radius)
        return -1.0 / actual_tight_radius

    def _validate_parameters(self, inner_params: np.ndarray, outer_radius: float) -> bool:
        """Validate parameters before full constraint checking"""
        # Quick bounds check
        if outer_radius < 3.5 or outer_radius > 7.0:
            return False
            
        # Quick parameter count check
        if len(inner_params) != 33:  # 11 hexagons * 3 params each
            return False
            
        return True

    def evaluate_constraints(self, inner_params: np.ndarray, outer_radius: float) -> Tuple[bool, bool, float]:
        """Comprehensive constraint evaluation with early termination"""
        inner_hexagons = []
        
        # Create inner hexagons
        for i in range(self.n_inner):
            x, y, angle = inner_params[3*i:3*i+3]
            hexagon = HexagonUtils.create_unit_hexagon((x, y), angle)
            inner_hexagons.append(hexagon)

        # Create outer hexagon
        outer_hexagon = HexagonUtils.create_unit_hexagon((0, 0), 0)
        outer_coords = list(outer_hexagon.exterior.coords)
        scaled_coords = [(x*outer_radius, y*outer_radius) for x, y in outer_coords]
        outer_hexagon_scaled = Polygon(scaled_coords)

        # Check containment (early termination)
        for hexagon in inner_hexagons:
            if not HexagonUtils.check_containment(hexagon, outer_hexagon_scaled):
                return False, False, 0.0  # containment violated

        # Check overlaps (early termination)
        for i in range(self.n_inner):
            for j in range(i+1, self.n_inner):
                if HexagonUtils.check_overlap(inner_hexagons[i], inner_hexagons[j]):
                    return False, False, 0.0  # overlap violated

        # Calculate actual tight radius for better objective function
        actual_tight_radius = HexagonUtils.calculate_tight_outer_radius(inner_params)
        return True, True, 1.0 / actual_tight_radius  # valid solution

    def optimize_with_local_search(self, initial_params: np.ndarray) -> np.ndarray:
        """Refine solution using local optimization after global search"""
        # Redefine bounds for local search with tighter constraints
        local_bounds = []
        for _ in range(self.n_inner):
            local_bounds.extend([(-6.0, 6.0), (-6.0, 6.0), (0, 360)])  # x, y, angle
        local_bounds.append((3.0, 8.0))  # Outer radius bounds

        options = {'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}

        try:
            # Use L-BFGS-B for fine-tuning with stricter tolerances
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

    def run_optimization(self, initial_configs: List[np.ndarray]) -> Optional[np.ndarray]:
        """Run the complete optimization process"""
        best_value = float('inf')
        best_params = None
        
        # Try several initial configurations
        for config in initial_configs[:15]:  # Test first 15 configs more thoroughly
            try:
                # Use DE with more iterations for better exploration
                result = differential_evolution(
                    self.objective_function,
                    self.bounds,
                    seed=42,
                    maxiter=100,  # More iterations for better exploration
                    popsize=20,   # Larger population
                    tol=1e-6,     # Tighter tolerance
                    mutation=(0.7, 1.0),  # Higher mutation rate for more exploration
                    recombination=0.8,    # Higher recombination for more mixing
                    disp=False
                )

                if result.success:
                    # Evaluate the result
                    temp_params = result.x
                    # Check if this is better than what we have
                    current_value = self.objective_function(temp_params)
                    if current_value < best_value:
                        best_value = current_value
                        best_params = temp_params
                        
            except Exception as e:
                continue

        # If we found a good initial solution, refine it with more thorough local search
        if best_params is not None:
            refined_params = self.optimize_with_local_search(best_params)
            return refined_params

        # Fallback to a more thorough single run with better parameters
        try:
            # Use differential evolution for global optimization with more iterations and better parameters
            result = differential_evolution(
                self.objective_function,
                self.bounds,
                seed=42,
                maxiter=200,   # Increased iterations
                popsize=30,    # Larger population
                tol=1e-7,      # Tighter tolerance
                mutation=(0.8, 1.0),  # Even higher mutation rate
                recombination=0.9,    # High recombination
                disp=False
            )

            if result.success:
                # Refine with local search
                refined_params = self.optimize_with_local_search(result.x)
                return refined_params

        except Exception as e:
            warnings.warn(f"Optimization failed: {str(e)}")

        return None

    def get_fallback_initial_guess(self) -> np.ndarray:
        """Generate a fallback initial guess"""
        initial_guess = []
        # Better initial pattern based on theoretical optimal arrangements
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

        initial_guess.append(4.2)  # Slightly better initial estimate
        return np.array(initial_guess)

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
        # Initialize components
        initializer = HexagonInitializer()
        optimizer = OptimizerPipeline()
        
        # Generate initial configurations
        initial_configs = initializer.generate_initial_solutions()
        
        # Run optimization
        final_params = optimizer.run_optimization(initial_configs)
        
        # If optimization succeeded
        if final_params is not None:
            # Extract results
            inner_params = final_params[:-1]
            outer_radius = final_params[-1]

            # Validate solution
            containment_ok, overlap_ok, inv_radius = optimizer.evaluate_constraints(inner_params, outer_radius)

            if containment_ok and overlap_ok:
                # Format output
                inner_hex_data = np.zeros((11, 3))
                for i in range(11):
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