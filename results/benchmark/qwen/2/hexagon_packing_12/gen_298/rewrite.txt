# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
from scipy.optimize import differential_evolution
import time
import math
from numba import jit
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

class HexagonConfigManager:
    """Manages hexagon configurations and transformations."""
    
    def __init__(self):
        self.hex_radius = 1.0
        self.hex_apothem = np.sqrt(3) / 2
        self.hex_height = 2 * self.hex_apothem
        self.hex_width = 2 * self.hex_radius
    
    def generate_hexagon_vertices(self, center_x: float, center_y: float, angle_deg: float, side_length: float = 1.0) -> np.ndarray:
        """Generate vertices of a regular hexagon given center, angle, and side length."""
        angle_rad = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            angle = angle_rad + i * math.pi / 3
            x = center_x + side_length * math.cos(angle)
            y = center_y + side_length * math.sin(angle)
            vertices.append((x, y))
        return np.array(vertices)
    
    def create_hexagon_polygon(self, center_x: float, center_y: float, angle_deg: float, side_length: float = 1.0) -> Polygon:
        """Create Shapely polygon representation of a hexagon."""
        vertices = self.generate_hexagon_vertices(center_x, center_y, angle_deg, side_length)
        return Polygon(vertices)
    
    def outer_hexagon_vertices(self, side_length: float) -> np.ndarray:
        """Generate vertices of outer hexagon centered at origin."""
        return self.generate_hexagon_vertices(0, 0, 0, side_length)
    
    def get_config_from_array(self, config_array: np.ndarray) -> np.ndarray:
        """Convert flat configuration array to structured hexagon data."""
        return config_array.reshape(12, 3)
    
    def get_array_from_config(self, hex_config: np.ndarray) -> np.ndarray:
        """Convert structured hexagon data to flat configuration array."""
        return hex_config.flatten()

class ConstraintValidator:
    """Validates hexagon configurations against constraints."""
    
    def __init__(self, geo_manager: HexagonConfigManager):
        self.geo = geo_manager
    
    @staticmethod
    @jit(nopython=True)
    def fast_containment_check(vertices_list, outer_side_length):
        """Fast containment check using numba."""
        outer_apothem_sq = (outer_side_length * math.sqrt(3) / 2) ** 2
        for vertices in vertices_list:
            for i in range(6):
                x = vertices[i, 0]
                y = vertices[i, 1]
                distance_sq = x*x + y*y
                if distance_sq > outer_apothem_sq:
                    return False
        return True
    
    @staticmethod
    @jit(nopython=True)
    def fast_overlap_check(vertices_list):
        """Fast overlap check using numba."""
        num_hexagons = len(vertices_list)
        for i in range(num_hexagons):
            for j in range(i+1, num_hexagons):
                # Quick center-based check
                cx1, cy1 = vertices_list[i][0]
                cx2, cy2 = vertices_list[j][0]
                dx = cx1 - cx2
                dy = cy1 - cy2
                distance_sq = dx * dx + dy * dy
                if distance_sq < 4.0:  # Unit hexagons can't be closer than 2 units apart
                    return False
        return True
    
    def validate_containment(self, hexagon_vertices_list, outer_side_length):
        """Validate containment with fast approximation followed by precise check."""
        # Fast check first
        if not self.fast_containment_check(hexagon_vertices_list, outer_side_length):
            return False, 1500
        
        # Precise check
        outer_polygon = self.geo.create_hexagon_polygon(0, 0, 0, outer_side_length)
        for vertices in hexagon_vertices_list:
            hex_polygon = self.geo.create_hexagon_polygon(
                vertices[0, 0], vertices[0, 1], 0, 1
            )
            if not outer_polygon.contains(hex_polygon):
                return False, 1500
        
        return True, 0
    
    def validate_overlap(self, hexagon_vertices_list):
        """Validate overlap with fast approximation followed by precise check."""
        # Fast check first
        if not self.fast_overlap_check(hexagon_vertices_list):
            # Fall back to precise check
            try:
                polygons = [self.geo.create_hexagon_polygon(
                    vertices[0, 0], vertices[0, 1], 0, 1
                ) for vertices in hexagon_vertices_list]
                
                # Check overlaps
                for i in range(len(polygons)):
                    for j in range(i+1, len(polygons)):
                        if polygons[i].intersects(polygons[j]):
                            return False, 1000
                return True, 0
            except:
                return False, 1000
        
        return True, 0
    
    def validate_configuration(self, config_array: np.ndarray, outer_side_length: float):
        """Complete validation of a configuration."""
        hexagons = self.geo.get_config_from_array(config_array)
        hexagon_vertices_list = [self.geo.generate_hexagon_vertices(
            hexagons[i][0], hexagons[i][1], hexagons[i][2]
        ) for i in range(12)]
        
        # Validate containment
        valid_containment, penalty_containment = self.validate_containment(
            hexagon_vertices_list, outer_side_length
        )
        if not valid_containment:
            return False, penalty_containment
        
        # Validate overlap
        valid_overlap, penalty_overlap = self.validate_overlap(hexagon_vertices_list)
        if not valid_overlap:
            return False, penalty_overlap
        
        return True, 0

class EvolutionaryOptimizer:
    """Handles evolutionary optimization of hexagon configurations."""
    
    def __init__(self, geo_manager: HexagonConfigManager, validator: ConstraintValidator):
        self.geo = geo_manager
        self.validator = validator
        self.max_evaluations = 5000
        self.max_time = 180.0
    
    def generate_initial_population(self, n_individuals: int) -> list:
        """Generate diverse initial population with strategic placements."""
        population = []
        
        # Base symmetric configuration
        base_config = np.array([
            [0.0, 0.0, 0.0],      # center
            [-1.732, 0.0, 0.0],   # left
            [1.732, 0.0, 0.0],    # right
            [0.0, 1.732, 0.0],    # top
            [0.0, -1.732, 0.0],   # bottom
            [-0.866, 0.866, 0.0], # top-left
            [0.866, 0.866, 0.0],  # top-right
            [-0.866, -0.866, 0.0], # bottom-left
            [0.866, -0.866, 0.0], # bottom-right
            [-2.598, 0.0, 0.0],   # far left
            [2.598, 0.0, 0.0],    # far right
            [0.0, 2.598, 0.0],    # far top
        ])
        
        for i in range(n_individuals):
            if i == 0:
                # First individual is base configuration
                individual = base_config.copy()
            else:
                # Perturb the base configuration
                individual = base_config + np.random.normal(0, 0.1, base_config.shape)
            
            population.append(individual.flatten())
        
        return population
    
    def evaluate_individual(self, individual: np.ndarray, outer_side_length: float) -> float:
        """Evaluate single individual with adaptive penalty."""
        valid, penalty = self.validator.validate_configuration(individual, outer_side_length)
        if valid:
            return -1.0 / outer_side_length  # Maximize 1/outer_side_length
        else:
            return penalty  # Return penalty for invalid configurations
    
    def parallel_evaluate_population(self, population: list, outer_side_length: float) -> list:
        """Evaluate entire population in parallel."""
        results = Parallel(n_jobs=-1, backend='threading')(
            delayed(self.evaluate_individual)(individual, outer_side_length)
            for individual in population
        )
        return results
    
    def optimize_with_de(self, initial_config: np.ndarray, outer_side_length: float) -> tuple:
        """Optimize using differential evolution with adaptive bounds."""
        bounds = []
        for _ in range(12):
            bounds.extend([(-5.0, 5.0), (-5.0, 5.0), (0.0, 360.0)])
        
        def objective(x):
            return self.evaluate_individual(x, outer_side_length)
        
        try:
            result = differential_evolution(
                objective, bounds, seed=42, maxiter=100, popsize=15, disp=False
            )
            return result.x, result.success
        except Exception:
            return initial_config, False
    
    def adaptive_sampling_optimization(self, target_side_length: float) -> tuple:
        """Adaptive sampling approach for finding optimal configuration."""
        # Generate diverse initial population
        population = self.generate_initial_population(10)
        
        best_config = None
        best_score = float('inf')
        best_side_length = target_side_length
        
        # Evaluate initial population
        fitness_scores = self.parallel_evaluate_population(population, target_side_length)
        
        # Find best individual from initial population
        for i, (individual, score) in enumerate(zip(population, fitness_scores)):
            if score < best_score and score >= 0:  # Valid configuration
                best_score = score
                best_config = individual
        
        # If we have a good starting point, refine with DE
        if best_config is not None:
            refined_config, success = self.optimize_with_de(best_config, target_side_length)
            if success:
                # Check if we can fit with smaller outer hexagon
                valid, _ = self.validator.validate_configuration(refined_config, target_side_length)
                if valid:
                    for test_side in np.linspace(3.8, target_side_length, 20)[::-1]:
                        valid_test, _ = self.validator.validate_configuration(refined_config, test_side)
                        if valid_test:
                            if test_side < best_side_length:
                                best_side_length = test_side
                                best_config = refined_config.copy()
        
        return best_config, best_side_length

class HexagonPackingSolver:
    """Main solver class that orchestrates the hexagon packing process."""
    
    def __init__(self):
        self.geo_manager = HexagonConfigManager()
        self.validator = ConstraintValidator(self.geo_manager)
        self.optimizer = EvolutionaryOptimizer(self.geo_manager, self.validator)
    
    def solve(self, target_side_length: float = 3.9419123) -> tuple:
        """Main solving method."""
        # Adaptive sampling optimization
        best_config, best_side_length = self.optimizer.adaptive_sampling_optimization(target_side_length)
        
        # Final validation
        if best_config is not None:
            valid, _ = self.validator.validate_configuration(best_config, best_side_length)
            if valid:
                return (
                    self.geo_manager.get_config_from_array(best_config),
                    np.array([0, 0, 0]),
                    best_side_length
                )
        
        # Fallback to known good configuration
        fallback_config = np.array([
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
        
        return fallback_config, np.array([0, 0, 0]), 8.0

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Create solver instance
    solver = HexagonPackingSolver()
    
    # Solve
    inner_hex_data, outer_hex_data, outer_hex_side_length = solver.solve()
    
    # Calculate metrics
    inv_side_length = 1.0 / outer_hex_side_length
    eval_time = time.time() - start_time
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END