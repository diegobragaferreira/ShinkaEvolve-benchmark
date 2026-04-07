# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import time
import random
from typing import Tuple, List, Optional
import warnings

class HexagonGeometry:
    """Handles all geometric operations for hexagon computations."""
    
    @staticmethod
    def create_hexagon(center: Tuple[float, float], side_length: float, rotation_degrees: float) -> Polygon:
        """Create a regular hexagon as a shapely polygon."""
        angle_rad = np.radians(rotation_degrees)
        vertices = []
        for i in range(6):
            angle = angle_rad + i * np.pi / 3
            x = center[0] + side_length * np.cos(angle)
            y = center[1] + side_length * np.sin(angle)
            vertices.append((x, y))
        return Polygon(vertices)
    
    @staticmethod
    def compute_hexagon_vertices(center: Tuple[float, float], side_length: float, rotation_degrees: float) -> List[Tuple[float, float]]:
        """Compute vertices of a regular hexagon."""
        angle_rad = np.radians(rotation_degrees)
        vertices = []
        for i in range(6):
            angle = angle_rad + i * np.pi / 3
            x = center[0] + side_length * np.cos(angle)
            y = center[1] + side_length * np.sin(angle)
            vertices.append((x, y))
        return vertices
    
    @staticmethod
    def check_containment(inner_hex: Polygon, outer_hex: Polygon) -> bool:
        """Check if inner hexagon is fully contained within outer hexagon."""
        return outer_hex.contains(inner_hex)
    
    @staticmethod
    def check_overlap(hex1: Polygon, hex2: Polygon) -> bool:
        """Check if two hexagons overlap."""
        return hex1.intersects(hex2)
    
    @staticmethod
    def calculate_outer_hex_side_length(hexagons: List[Polygon]) -> float:
        """Calculate minimum outer hexagon side length that contains all inner hexagons."""
        if not hexagons:
            return 10.0
        
        # Collect all vertices from all hexagons
        all_points = []
        for hexagon in hexagons:
            all_points.extend(list(hexagon.exterior.coords))
        
        if not all_points:
            return 10.0
        
        # Calculate distances from center (0,0) to all points
        distances = []
        for point in all_points:
            dist = np.sqrt(point[0]**2 + point[1]**2)
            distances.append(dist)
        
        if not distances:
            return 10.0
            
        # Maximum distance determines the required outer radius
        max_dist = max(distances)
        
        # For regular hexagon, side length = max_dist / sqrt(3)
        return max_dist / np.sqrt(3) * 2

class ConstraintValidator:
    """Validates geometric constraints for hexagon packing."""
    
    @classmethod
    def validate_packing(cls, hexagon_data: np.ndarray, outer_center: Tuple[float, float] = (0, 0)) -> Tuple[bool, float]:
        """
        Validate if all constraints are satisfied for the packing.
        
        Returns:
            tuple: (is_valid, inverse_outer_side_length)
        """
        try:
            # Create individual hexagon polygons
            hexagons = []
            for i in range(12):
                center = (hexagon_data[i][0], hexagon_data[i][1])
                angle = hexagon_data[i][2]
                hexagon = HexagonGeometry.create_hexagon(center, 1.0, angle)
                hexagons.append(hexagon)
            
            # Check for overlaps between any pair of hexagons
            for i in range(12):
                for j in range(i+1, 12):
                    if HexagonGeometry.check_overlap(hexagons[i], hexagons[j]):
                        return False, 0.0
            
            # Create outer hexagon
            outer_side_length = HexagonGeometry.calculate_outer_hex_side_length(hexagons)
            outer_hexagon = HexagonGeometry.create_hexagon(outer_center, outer_side_length, 0.0)
            
            # Check containment of all hexagon vertices
            for hexagon in hexagons:
                for vertex in hexagon.exterior.coords:
                    point = Point(vertex[0], vertex[1])
                    if not outer_hexagon.contains(point):
                        return False, 0.0
            
            # Return inverse of outer side length as objective
            return True, 1.0 / outer_side_length
            
        except Exception as e:
            warnings.warn(f"Constraint validation error: {e}")
            return False, 0.0

class ConfigurationGenerator:
    """Generates various initial configurations for optimization."""
    
    @staticmethod
    def get_symmetric_configurations() -> List[np.ndarray]:
        """Generate multiple symmetric configurations."""
        configs = []
        
        # Configuration 1: Traditional 2-ring arrangement
        config1 = np.array([
            [0.0, 0.0, 0],           # Center
            [0.0, 2.0, 0],           # Top
            [1.732050808, 1.0, 0],   # Top right
            [1.732050808, -1.0, 0],  # Bottom right
            [0.0, -2.0, 0],          # Bottom
            [-1.732050808, -1.0, 0], # Bottom left
            [-1.732050808, 1.0, 0],  # Top left
            [3.464101616, 2.0, 0],   # Far top right
            [3.464101616, -2.0, 0],  # Far bottom right
            [-3.464101616, -2.0, 0], # Far bottom left
            [-3.464101616, 2.0, 0],  # Far top left
            [0.0, -4.0, 0],          # Far bottom
        ], dtype=float)
        configs.append(config1)
        
        # Configuration 2: Honeycomb-like arrangement
        config2 = np.array([
            [0.0, 0.0, 0],
            [2.0, 0.0, 0],
            [1.0, 1.732050808, 0],
            [-1.0, 1.732050808, 0],
            [-2.0, 0.0, 0],
            [-1.0, -1.732050808, 0],
            [1.0, -1.732050808, 0],
            [3.0, 1.732050808, 0],
            [3.0, -1.732050808, 0],
            [-3.0, -1.732050808, 0],
            [-3.0, 1.732050808, 0],
            [0.0, -3.464101616, 0],
        ], dtype=float)
        configs.append(config2)
        
        # Configuration 3: Optimized known pattern
        config3 = np.array([
            [0.0, 0.0, 0],
            [0.0, 2.0, 0],
            [1.732050808, 1.0, 0],
            [1.732050808, -1.0, 0],
            [0.0, -2.0, 0],
            [-1.732050808, -1.0, 0],
            [-1.732050808, 1.0, 0],
            [3.464101616, 2.0, 0],
            [3.464101616, -2.0, 0],
            [-3.464101616, -2.0, 0],
            [-3.464101616, 2.0, 0],
            [0.0, -4.0, 0],
        ], dtype=float)
        configs.append(config3)
        
        return configs
    
    @staticmethod
    def generate_random_perturbed_config(base_config: np.ndarray, perturbation_magnitude: float = 0.1) -> np.ndarray:
        """Generate a random perturbed version of a base configuration."""
        perturbed = base_config.copy()
        for i in range(len(perturbed)):
            # Perturb positions slightly
            perturbed[i][0] += random.uniform(-perturbation_magnitude, perturbation_magnitude)
            perturbed[i][1] += random.uniform(-perturbation_magnitude, perturbation_magnitude)
        return perturbed

class OptimizationEngine:
    """Handles the optimization process with multi-stage approach."""
    
    def __init__(self):
        self.best_score = 0.0
        self.best_config = None
        self.best_outer_side_length = float('inf')
    
    def objective_function(self, params: np.ndarray) -> float:
        """Objective function to minimize (negative of 1/outer_radius)"""
        # Reshape params into 12 hexagons with (x,y,angle) each
        hex_params = params.reshape(-1, 3)
        
        # Validate the packing
        is_valid, objective_value = ConstraintValidator.validate_packing(hex_params)
        
        # If invalid configuration, penalize heavily
        if not is_valid:
            return 1e6  # Large penalty
        
        # Return negative because we want to maximize 1/outer_radius, which means minimizing -1/outer_radius
        return -objective_value
    
    def optimize_single_configuration(self, initial_config: np.ndarray, max_iter: int = 1000) -> Tuple[np.ndarray, float]:
        """Perform optimization on a single configuration."""
        # Flatten the data for optimization
        initial_flat = initial_config.flatten()
        
        # Define bounds for optimization
        bounds = []
        # Bounds for x,y positions
        pos_bounds = [(-10, 10), (-10, 10)] * 12
        # Bounds for angles (0-360 deg)
        angle_bounds = [(-180, 180)] * 12
        
        for b in pos_bounds + angle_bounds:
            bounds.append(b)
        
        # Optimize using L-BFGS-B which handles bound constraints well
        try:
            result = minimize(
                self.objective_function,
                initial_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'disp': False}
            )
            
            if result.success:
                optimized_params = result.x
                optimized_hex_params = optimized_params.reshape(-1, 3)
                _, score = ConstraintValidator.validate_packing(optimized_hex_params)
                return optimized_hex_params, score
        except Exception as e:
            warnings.warn(f"Optimization failed: {e}")
        
        return initial_config, 0.0
    
    def multi_stage_optimization(self, initial_configs: List[np.ndarray]) -> Tuple[np.ndarray, float]:
        """Perform multi-stage optimization on multiple configurations."""
        best_config = None
        best_score = 0.0
        
        for i, config in enumerate(initial_configs):
            # Stage 1: Quick coarse optimization
            stage1_result, stage1_score = self.optimize_single_configuration(config, max_iter=300)
            
            # Stage 2: Refinement with tighter bounds
            stage2_result, stage2_score = self.optimize_single_configuration(stage1_result, max_iter=500)
            
            # Stage 3: Final fine-tuning
            stage3_result, stage3_score = self.optimize_single_configuration(stage2_result, max_iter=700)
            
            # Select best among stages
            current_best_score = max(stage1_score, stage2_score, stage3_score)
            current_best_config = stage3_result if stage3_score > stage2_score and stage3_score > stage1_score else \
                                 stage2_result if stage2_score > stage1_score else stage1_result
            
            if current_best_score > best_score:
                best_score = current_best_score
                best_config = current_best_config
        
        return best_config, best_score

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Initialize optimization engine
    optimizer = OptimizationEngine()
    
    # Generate multiple initial configurations
    generator = ConfigurationGenerator()
    initial_configs = generator.get_symmetric_configurations()
    
    # Add some random variants to ensure exploration
    for config in initial_configs[:2]:  # Only perturb first two
        perturbed = generator.generate_random_perturbed_config(config, 0.05)
        initial_configs.append(perturbed)
    
    # Perform multi-stage optimization
    try:
        best_config, best_score = optimizer.multi_stage_optimization(initial_configs)
        
        # Validate again after optimization
        is_valid, final_score = ConstraintValidator.validate_packing(best_config)
        
        if is_valid and final_score > 0:
            # Compute actual outer side length
            hexagons = []
            for i in range(12):
                center = (best_config[i][0], best_config[i][1])
                angle = best_config[i][2]
                hexagon = HexagonGeometry.create_hexagon(center, 1.0, angle)
                hexagons.append(hexagon)
            
            outer_side_length = HexagonGeometry.calculate_outer_hex_side_length(hexagons)
            
            # Outer hexagon centered at origin with no rotation
            outer_hex_data = np.array([0, 0, 0])
            return best_config, outer_hex_data, outer_side_length
            
    except Exception as e:
        warnings.warn(f"Main optimization failed: {e}")
    
    # Fallback to a reliable configuration
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
        [0, -4, 0]
    ])
    
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = 8.0
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END