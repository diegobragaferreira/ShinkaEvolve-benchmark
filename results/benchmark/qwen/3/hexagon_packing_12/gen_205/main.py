# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
from scipy.optimize import minimize
import math
import random
import time
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class HexagonConfig:
    """Data class to represent hexagon configuration"""
    positions: np.ndarray  # Shape (12, 2) - x, y coordinates
    rotations: np.ndarray  # Shape (12,) - rotation angles in degrees

class HexagonUtils:
    """Utility class for hexagon geometric operations"""
    
    # Pre-computed vertices for unit hexagon at origin (cached for performance)
    UNIT_HEX_VERTICES = np.array([
        [1.0, 0.0],
        [0.5, 0.8660254037844386],
        [-0.5, 0.8660254037844386],
        [-1.0, 0.0],
        [-0.5, -0.8660254037844386],
        [0.5, -0.8660254037844386]
    ])
    
    @staticmethod
    def create_hexagon_vertices(center: Tuple[float, float], 
                              rotation: float, 
                              scale: float = 1.0) -> np.ndarray:
        """Efficiently compute hexagon vertices using pre-computed unit vertices"""
        # Rotate and translate unit vertices
        angle_rad = math.radians(rotation)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        # Apply rotation matrix to unit vertices
        rotated_vertices = np.dot(HexagonUtils.UNIT_HEX_VERTICES, 
                                np.array([[cos_a, sin_a], [-sin_a, cos_a]]).T) * scale
        
        # Translate to center
        return rotated_vertices + np.array(center)
    
    @staticmethod
    def hexagon_to_polygon(center: Tuple[float, float], 
                          rotation: float) -> Polygon:
        """Convert hexagon parameters to shapely polygon"""
        vertices = HexagonUtils.create_hexagon_vertices(center, rotation)
        return Polygon(vertices)
    
    @staticmethod
    def get_all_hexagon_vertices(hex_config: HexagonConfig) -> List[np.ndarray]:
        """Get vertices for all hexagons efficiently"""
        vertices_list = []
        for i in range(12):
            center = (hex_config.positions[i][0], hex_config.positions[i][1])
            vertices = HexagonUtils.create_hexagon_vertices(center, hex_config.rotations[i])
            vertices_list.append(vertices)
        return vertices_list

class ConstraintValidator:
    """Validates constraints for hexagon configurations"""
    
    @staticmethod
    def fast_overlap_check(hex1_vertices: np.ndarray, 
                          hex2_vertices: np.ndarray) -> bool:
        """Fast bounding box overlap check"""
        # Get bounding boxes
        min1, max1 = np.min(hex1_vertices, axis=0), np.max(hex1_vertices, axis=0)
        min2, max2 = np.min(hex2_vertices, axis=0), np.max(hex2_vertices, axis=0)
        
        # Quick rejection test
        if (max1[0] < min2[0] or max2[0] < min1[0] or 
            max1[1] < min2[1] or max2[1] < min1[1]):
            return False
            
        # Use shapely for precise intersection check
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2) and not poly1.touches(poly2)
    
    @staticmethod
    def validate_overlap_free(hex_config: HexagonConfig) -> bool:
        """Check that no hexagons overlap"""
        vertices_list = HexagonUtils.get_all_hexagon_vertices(hex_config)
        n = len(vertices_list)
        
        for i in range(n):
            for j in range(i+1, n):
                if ConstraintValidator.fast_overlap_check(vertices_list[i], vertices_list[j]):
                    return False
        return True
    
    @staticmethod
    def validate_containment(hex_config: HexagonConfig, 
                           outer_radius: float) -> bool:
        """Check that all hexagons are contained in outer hexagon"""
        # Create outer hexagon vertices
        outer_vertices = HexagonUtils.create_hexagon_vertices((0,0), 0, outer_radius)
        outer_polygon = Polygon(outer_vertices)
        
        # Check each inner hexagon
        vertices_list = HexagonUtils.get_all_hexagon_vertices(hex_config)
        for vertices in vertices_list:
            # Create inner polygon and check containment
            inner_polygon = Polygon(vertices)
            if not outer_polygon.contains(inner_polygon):
                return False
        return True

class ObjectiveEvaluator:
    """Evaluates configurations for optimization"""
    
    @staticmethod
    def compute_outer_radius(hex_config: HexagonConfig) -> float:
        """Compute minimum outer radius that contains all hexagons"""
        vertices_list = HexagonUtils.get_all_hexagon_vertices(hex_config)
        
        # Collect all vertices
        all_vertices = np.vstack(vertices_list)
        
        # Compute distances from origin
        distances = np.sqrt(np.sum(all_vertices**2, axis=1))
        
        # Return maximum distance + buffer
        return np.max(distances) + 1.0
    
    @staticmethod
    def evaluate_configuration(hex_config: HexagonConfig) -> Tuple[bool, float]:
        """Evaluate configuration: return (valid, objective_value)"""
        try:
            # Check overlaps first (fast)
            if not ConstraintValidator.validate_overlap_free(hex_config):
                return False, -1e10
            
            # Compute outer radius
            outer_radius = ObjectiveEvaluator.compute_outer_radius(hex_config)
            
            # Compute containment
            if not ConstraintValidator.validate_containment(hex_config, outer_radius):
                return False, -1e10
            
            # Return inverse of outer radius (higher is better)
            return True, 1.0 / outer_radius
            
        except Exception:
            return False, -1e10

class ConfigurationGenerator:
    """Generates initial configurations with various patterns"""
    
    @staticmethod
    def generate_symmetric_config() -> HexagonConfig:
        """Generate a symmetric initial configuration"""
        # Central hexagon
        positions = [[0.0, 0.0]]
        
        # First ring - 6 hexagons
        for i in range(6):
            angle = i * 60  # degrees
            rad = math.radians(angle)
            x = 2.0 * math.cos(rad)
            y = 2.0 * math.sin(rad)
            positions.append([x, y])
        
        # Second ring - 5 hexagons (make it compact)
        angles = [0, 72, 144, 216, 288]  # 5 evenly spaced
        for angle in angles:
            rad = math.radians(angle)
            x = 3.464 * math.cos(rad)  # sqrt(12)
            y = 3.464 * math.sin(rad)
            positions.append([x, y])
        
        # Add one more to reach 12
        positions.append([0.0, -3.464])
        
        # Trim to exactly 12
        positions = positions[:12]
        
        # Create config
        positions_array = np.array(positions)
        rotations_array = np.zeros(12)
        
        return HexagonConfig(positions_array, rotations_array)
    
    @staticmethod
    def generate_optimized_config() -> HexagonConfig:
        """Generate an optimized configuration based on literature"""
        # Based on known high-density packings
        positions = np.array([
            [0, 0],           # center
            [0, 2.0],         # top
            [0, -2.0],        # bottom
            [1.732, 1.0],     # top-right
            [-1.732, 1.0],    # top-left
            [1.732, -1.0],    # bottom-right
            [-1.732, -1.0],   # bottom-left
            [3.464, 0],       # far right
            [-3.464, 0],      # far left
            [1.732, 3.0],     # upper right corner
            [-1.732, 3.0],    # upper left corner
            [1.732, -3.0],    # lower right corner
        ])
        
        rotations = np.zeros(12)
        
        return HexagonConfig(positions, rotations)
    
    @staticmethod
    def generate_fallback_config() -> HexagonConfig:
        """Generate fallback configuration for edge cases"""
        positions = np.array([
            [0, 0],              # center
            [-2.5, 0],           # left
            [2.5, 0],            # right
            [-1.25, 2.17],       # top-left
            [1.25, 2.17],        # top-right
            [-1.25, -2.17],      # bottom-left
            [1.25, -2.17],       # bottom-right
            [-3.75, 2.17],       # far top-left
            [3.75, 2.17],        # far top-right
            [-3.75, -2.17],      # far bottom-left
            [3.75, -2.17],       # far bottom-right
            [0, -4],             # far bottom-center
        ])
        
        rotations = np.zeros(12)
        
        return HexagonConfig(positions, rotations)

class MutationStrategy:
    """Handles different mutation strategies for evolutionary optimization"""
    
    @staticmethod
    def mutate_config(config: HexagonConfig, 
                     mutation_strength: float = 0.1,
                     stage: int = 1) -> HexagonConfig:
        """Apply mutation to configuration"""
        mutated = config
        
        # Adjust mutation strength based on optimization stage
        if stage == 1:
            strength = mutation_strength * 2.0  # Coarse
        elif stage == 2:
            strength = mutation_strength * 1.0  # Fine
        else:
            strength = mutation_strength * 0.5  # Final
            
        # Mutate positions
        mutated.positions += np.random.normal(0, strength, mutated.positions.shape)
        
        # Mutate rotations (small changes)
        mutated.rotations += np.random.normal(0, 5, mutated.rotations.shape)
        
        return mutated

class OptimizerStage:
    """Represents a single optimization stage"""
    
    def __init__(self, stage_num: int, max_generations: int, 
                 population_size: int = 20):
        self.stage_num = stage_num
        self.max_generations = max_generations
        self.population_size = population_size
        self.best_score = -1e10
        self.best_config = None

    def run(self, initial_config: HexagonConfig) -> Tuple[HexagonConfig, float]:
        """Run this optimization stage"""
        # Initialize population
        population = [initial_config]
        
        # Add variations
        for _ in range(self.population_size - 1):
            variant = MutationStrategy.mutate_config(initial_config, stage=self.stage_num)
            population.append(variant)
        
        # Evolution loop
        for generation in range(self.max_generations):
            # Evaluate fitness
            fitness_scores = []
            for config in population:
                valid, score = ObjectiveEvaluator.evaluate_configuration(config)
                fitness_scores.append(score if valid else -1e10)
            
            # Sort by fitness descending
            sorted_indices = np.argsort(fitness_scores)[::-1]
            
            # Elitism: keep top 1/3
            elite_count = max(1, self.population_size // 3)
            elite = [population[i] for i in sorted_indices[:elite_count]]
            
            # Create new population
            new_population = elite.copy()
            
            # Fill rest with mutations of elite members
            while len(new_population) < self.population_size:
                parent = random.choice(elite)
                child = MutationStrategy.mutate_config(parent, stage=self.stage_num)
                new_population.append(child)
            
            population = new_population
            
            # Track best
            for config in population:
                valid, score = ObjectiveEvaluator.evaluate_configuration(config)
                if valid and score > self.best_score:
                    self.best_score = score
                    self.best_config = config
        
        return self.best_config, self.best_score

class AdvancedOptimizer:
    """Main orchestrator for hexagon packing optimization"""
    
    def __init__(self):
        self.start_time = time.time()
        self.timeout = 180.0
        self.stages = [
            OptimizerStage(1, 30, 25),  # Coarse stage
            OptimizerStage(2, 30, 20),  # Fine stage  
            OptimizerStage(3, 20, 15),  # Final stage
        ]
    
    def run_scipy_refinement(self, config: HexagonConfig) -> HexagonConfig:
        """Apply scipy-based local optimization"""
        try:
            # Flatten positions for scipy
            initial_positions = config.positions.flatten()
            
            def objective(x):
                # Reshape positions
                positions = x.reshape(-1, 2)
                temp_config = HexagonConfig(positions, config.rotations)
                outer_radius = ObjectiveEvaluator.compute_outer_radius(temp_config)
                return outer_radius
            
            # Simple bounds for positions
            bounds = [(-10, 10), (-10, 10)] * 12
            
            result = minimize(objective, initial_positions, 
                            method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 50})
            
            if result.success:
                final_positions = result.x.reshape(-1, 2)
                config.positions = final_positions
                
        except Exception:
            # If refinement fails, return original config
            pass
            
        return config
    
    def optimize(self) -> Tuple[HexagonConfig, float]:
        """Run complete optimization pipeline"""
        # Generate initial configurations
        configs = [
            ConfigurationGenerator.generate_symmetric_config(),
            ConfigurationGenerator.generate_optimized_config()
        ]
        
        # Find best initial configuration
        best_initial = None
        best_initial_score = -1e10
        
        for config in configs:
            valid, score = ObjectiveEvaluator.evaluate_configuration(config)
            if valid and score > best_initial_score:
                best_initial_score = score
                best_initial = config
        
        if best_initial is None:
            # Fallback to basic config
            best_initial = ConfigurationGenerator.generate_fallback_config()
        
        # Run multi-stage optimization
        current_config = best_initial
        
        for stage in self.stages:
            if time.time() - self.start_time > self.timeout * 0.9:
                break
                
            current_config, _ = stage.run(current_config)
        
        # Final scipy refinement
        final_config = self.run_scipy_refinement(current_config)
        
        # Final evaluation
        valid, final_score = ObjectiveEvaluator.evaluate_configuration(final_config)
        if not valid:
            # Fallback to initial config
            final_config = best_initial
            valid, final_score = ObjectiveEvaluator.evaluate_configuration(final_config)
        
        return final_config, final_score

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    try:
        optimizer = AdvancedOptimizer()
        final_config, score = optimizer.optimize()
        
        # Convert to required output format
        outer_radius = ObjectiveEvaluator.compute_outer_radius(final_config)
        outer_hex_side_length = outer_radius + 0.5  # Add margin
        
        # Format into (12,3) array with rotations
        inner_hex_data = np.column_stack([
            final_config.positions,
            final_config.rotations
        ])
        
        outer_hex_data = np.array([0, 0, 0])  # centered at origin
        
        return inner_hex_data, outer_hex_data, outer_hex_side_length
        
    except Exception as e:
        # Return fallback on error
        print(f"Fallback due to error: {e}")
        fallback_config = ConfigurationGenerator.generate_fallback_config()
        outer_radius = ObjectiveEvaluator.compute_outer_radius(fallback_config)
        outer_hex_side_length = outer_radius + 0.5
        
        inner_hex_data = np.column_stack([
            fallback_config.positions,
            fallback_config.rotations
        ])
        
        outer_hex_data = np.array([0, 0, 0])
        
        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END