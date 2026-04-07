# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import time
import random
from typing import Tuple, List, Optional
import warnings
from dataclasses import dataclass
from enum import Enum

class OptimizationStage(Enum):
    COARSE = 1
    MEDIUM = 2
    FINE = 3

@dataclass
class HexagonConfig:
    """Data class for hexagon configuration storage."""
    centers: np.ndarray  # Shape (12, 2) - x, y coordinates
    rotations: np.ndarray  # Shape (12,) - rotation angles in degrees

class GeometryUtils:
    """Static utility class for geometric operations."""
    
    @staticmethod
    def create_hexagon_vertices(center_x: float, center_y: float, side_length: float, rotation_deg: float) -> np.ndarray:
        """Create hexagon vertices efficiently using vectorized operations."""
        angle_rad = np.radians(rotation_deg)
        angles = np.linspace(0, 2*np.pi, 7)[:-1] + angle_rad  # 6 angles, omit last to avoid duplication
        vertices_x = center_x + side_length * np.cos(angles)
        vertices_y = center_y + side_length * np.sin(angles)
        return np.column_stack([vertices_x, vertices_y])
    
    @staticmethod
    def check_overlap_fast(hex1_vertices: np.ndarray, hex2_vertices: np.ndarray) -> bool:
        """Fast overlap checking using Shapely polygons."""
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)
    
    @staticmethod
    def calculate_outer_hex_side_length_fast(hexagon_vertices_list: List[np.ndarray]) -> float:
        """Efficiently calculate outer hexagon side length."""
        if not hexagon_vertices_list:
            return 10.0
        
        # Stack all vertices
        all_vertices = np.vstack(hexagon_vertices_list)
        
        if len(all_vertices) == 0:
            return 10.0
        
        # Calculate distances from origin to all vertices
        distances = np.sqrt(np.sum(all_vertices**2, axis=1))
        max_dist = np.max(distances)
        
        # For regular hexagon, side length = max_dist / sqrt(3) * 2
        return max_dist / np.sqrt(3) * 2

class ConstraintChecker:
    """Validates hexagon packing constraints with optimized performance."""
    
    def __init__(self):
        self._cached_polygons = {}
        self._cache_valid = False
    
    def validate_and_score(self, hex_config: HexagonConfig, 
                          outer_center: Tuple[float, float] = (0, 0)) -> Tuple[bool, float]:
        """Validate constraints and compute objective score."""
        try:
            # Cache validation - avoid recomputing if possible
            self._validate_cache(hex_config)
            
            # Create hexagon polygons
            hexagons = []
            vertices_list = []
            
            for i in range(12):
                center = (hex_config.centers[i][0], hex_config.centers[i][1])
                angle = hex_config.rotations[i]
                
                # Use cached if available
                key = (center[0], center[1], angle)
                if key in self._cached_polygons:
                    hexagon = self._cached_polygons[key]
                else:
                    vertices = GeometryUtils.create_hexagon_vertices(center[0], center[1], 1.0, angle)
                    vertices_list.append(vertices)
                    hexagon = Polygon(vertices)
                    self._cached_polygons[key] = hexagon
                
                hexagons.append(hexagon)
            
            # Check overlaps
            for i in range(12):
                for j in range(i+1, 12):
                    if GeometryUtils.check_overlap_fast(hexagons[i], hexagons[j]):
                        return False, 0.0
            
            # Calculate outer radius efficiently
            outer_side_length = GeometryUtils.calculate_outer_hex_side_length_fast(vertices_list)
            
            # Create outer hexagon
            outer_vertices = GeometryUtils.create_hexagon_vertices(
                outer_center[0], outer_center[1], outer_side_length, 0.0
            )
            outer_hexagon = Polygon(outer_vertices)
            
            # Check containment
            for hexagon in hexagons:
                for vertex in hexagon.exterior.coords:
                    point = Point(vertex[0], vertex[1])
                    if not outer_hexagon.contains(point):
                        return False, 0.0
            
            # Return inverse of outer side length
            return True, 1.0 / outer_side_length
            
        except Exception as e:
            warnings.warn(f"Validation error: {e}")
            return False, 0.0
    
    def _validate_cache(self, hex_config: HexagonConfig):
        """Clear cache when configuration changes."""
        # Simple approach: Clear cache for now; could implement smarter caching
        self._cached_polygons.clear()
        self._cache_valid = True

class ConfigurationBuilder:
    """Creates various initial configurations for optimization."""
    
    @staticmethod
    def get_optimized_configurations() -> List[HexagonConfig]:
        """Generate multiple optimized configurations."""
        configs = []
        
        # Configuration 1: Optimized 2-ring arrangement
        centers1 = np.array([
            [0.0, 0.0],           # Center
            [0.0, 2.0],           # Top
            [1.732050808, 1.0],   # Top right
            [1.732050808, -1.0],  # Bottom right
            [0.0, -2.0],          # Bottom
            [-1.732050808, -1.0], # Bottom left
            [-1.732050808, 1.0],  # Top left
            [3.464101616, 2.0],   # Far top right
            [3.464101616, -2.0],  # Far bottom right
            [-3.464101616, -2.0], # Far bottom left
            [-3.464101616, 2.0],  # Far top left
            [0.0, -4.0],          # Far bottom
        ], dtype=float)
        
        rotations1 = np.zeros(12)
        configs.append(HexagonConfig(centers1, rotations1))
        
        # Configuration 2: Honeycomb-like arrangement
        centers2 = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, 1.732050808],
            [-1.0, 1.732050808],
            [-2.0, 0.0],
            [-1.0, -1.732050808],
            [1.0, -1.732050808],
            [3.0, 1.732050808],
            [3.0, -1.732050808],
            [-3.0, -1.732050808],
            [-3.0, 1.732050808],
            [0.0, -3.464101616],
        ], dtype=float)
        
        rotations2 = np.zeros(12)
        configs.append(HexagonConfig(centers2, rotations2))
        
        # Configuration 3: Golden ratio inspired placement
        centers3 = np.array([
            [0.0, 0.0],
            [0.0, 2.0],
            [1.732050808, 1.0],
            [1.732050808, -1.0],
            [0.0, -2.0],
            [-1.732050808, -1.0],
            [-1.732050808, 1.0],
            [3.464101616, 2.0],
            [3.464101616, -2.0],
            [-3.464101616, -2.0],
            [-3.464101616, 2.0],
            [0.0, -4.0],
        ], dtype=float)
        
        rotations3 = np.zeros(12)
        configs.append(HexagonConfig(centers3, rotations3))
        
        return configs
    
    @staticmethod
    def generate_random_perturbed_config(base_config: HexagonConfig, 
                                       perturbation_magnitude: float = 0.1) -> HexagonConfig:
        """Generate a random perturbed version of a base configuration."""
        perturbed_centers = base_config.centers.copy()
        perturbed_rotations = base_config.rotations.copy()
        
        for i in range(len(perturbed_centers)):
            # Perturb positions slightly
            perturbed_centers[i][0] += random.uniform(-perturbation_magnitude, perturbation_magnitude)
            perturbed_centers[i][1] += random.uniform(-perturbation_magnitude, perturbation_magnitude)
            
            # Random rotation
            perturbed_rotations[i] = random.uniform(-180, 180)
        
        return HexagonConfig(perturbed_centers, perturbed_rotations)

class Optimizer:
    """Handles the optimization process with staged approach."""
    
    def __init__(self):
        self.constraint_checker = ConstraintChecker()
        self.best_result = None
        self.best_score = 0.0
    
    def _flatten_config(self, hex_config: HexagonConfig) -> np.ndarray:
        """Convert configuration to flat array for optimization."""
        # Combine centers and rotations
        flat_array = np.empty(12 * 3)
        flat_array[::3] = hex_config.centers[:, 0]  # x positions
        flat_array[1::3] = hex_config.centers[:, 1]  # y positions  
        flat_array[2::3] = hex_config.rotations  # rotations
        return flat_array
    
    def _unflatten_config(self, flat_array: np.ndarray) -> HexagonConfig:
        """Convert flat array back to configuration."""
        centers = np.empty((12, 2))
        centers[:, 0] = flat_array[::3]  # x positions
        centers[:, 1] = flat_array[1::3]  # y positions
        
        rotations = flat_array[2::3]  # rotations
        
        return HexagonConfig(centers, rotations)
    
    def _objective_function(self, flat_params: np.ndarray) -> float:
        """Objective function for scipy optimization."""
        hex_config = self._unflatten_config(flat_params)
        
        # Validate the packing
        is_valid, objective_value = self.constraint_checker.validate_and_score(hex_config)
        
        # If invalid configuration, penalize heavily
        if not is_valid:
            return 1e6  # Large penalty
        
        # Return negative because we want to maximize 1/outer_radius
        return -objective_value
    
    def _optimize_single_stage(self, initial_config: HexagonConfig, 
                              stage: OptimizationStage, max_iter: int) -> Tuple[HexagonConfig, float]:
        """Perform optimization at specific stage."""
        # Flatten for optimization
        initial_flat = self._flatten_config(initial_config)
        
        # Define bounds based on stage
        bounds = []
        # Positions bounds (more relaxed in coarse stages)
        if stage == OptimizationStage.COARSE:
            pos_bounds = [(-8, 8), (-8, 8)] * 12
        elif stage == OptimizationStage.MEDIUM:
            pos_bounds = [(-6, 6), (-6, 6)] * 12
        else:  # FINE
            pos_bounds = [(-5, 5), (-5, 5)] * 12
        
        # Rotation bounds (all angles allowed)
        rot_bounds = [(-180, 180)] * 12
        
        for b in pos_bounds + rot_bounds:
            bounds.append(b)
        
        try:
            # Optimize using L-BFGS-B
            result = minimize(
                self._objective_function,
                initial_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'disp': False}
            )
            
            if result.success:
                optimized_config = self._unflatten_config(result.x)
                _, score = self.constraint_checker.validate_and_score(optimized_config)
                return optimized_config, score
                
        except Exception as e:
            warnings.warn(f"Optimization stage {stage.name} failed: {e}")
        
        return initial_config, 0.0
    
    def optimize(self, initial_configs: List[HexagonConfig]) -> Tuple[HexagonConfig, float]:
        """Multi-stage optimization."""
        best_config = None
        best_score = 0.0
        
        for i, config in enumerate(initial_configs):
            # Stage 1: Coarse optimization
            stage1_result, stage1_score = self._optimize_single_stage(config, 
                                                                    OptimizationStage.COARSE, 
                                                                    200)
            
            # Stage 2: Medium refinement
            stage2_result, stage2_score = self._optimize_single_stage(stage1_result, 
                                                                    OptimizationStage.MEDIUM, 
                                                                    300)
            
            # Stage 3: Fine tuning
            stage3_result, stage3_score = self._optimize_single_stage(stage2_result, 
                                                                    OptimizationStage.FINE, 
                                                                    400)
            
            # Select best among stages
            current_best_score = max(stage1_score, stage2_score, stage3_score)
            
            if current_best_score > best_score:
                best_score = current_best_score
                if stage3_score >= stage2_score and stage3_score >= stage1_score:
                    best_config = stage3_result
                elif stage2_score >= stage1_score:
                    best_config = stage2_result
                else:
                    best_config = stage1_result
        
        return best_config, best_score

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Initialize components
    optimizer = Optimizer()
    builder = ConfigurationBuilder()
    
    # Generate multiple initial configurations
    initial_configs = builder.get_optimized_configurations()
    
    # Add some random variants to ensure exploration
    for config in initial_configs[:2]:  # Only perturb first two
        perturbed = builder.generate_random_perturbed_config(config, 0.05)
        initial_configs.append(perturbed)
    
    # Perform optimization
    try:
        best_config, best_score = optimizer.optimize(initial_configs)
        
        # Validate again after optimization
        is_valid, final_score = optimizer.constraint_checker.validate_and_score(best_config)
        
        if is_valid and final_score > 0:
            # Compute actual outer side length
            vertices_list = []
            for i in range(12):
                center = (best_config.centers[i][0], best_config.centers[i][1])
                angle = best_config.rotations[i]
                vertices = GeometryUtils.create_hexagon_vertices(center[0], center[1], 1.0, angle)
                vertices_list.append(vertices)
            
            outer_side_length = GeometryUtils.calculate_outer_hex_side_length_fast(vertices_list)
            
            # Convert to requested format
            inner_hex_data = np.column_stack([
                best_config.centers[:, 0],  # x positions
                best_config.centers[:, 1],  # y positions
                best_config.rotations      # rotations
            ])
            
            # Outer hexagon centered at origin with no rotation
            outer_hex_data = np.array([0, 0, 0])
            return inner_hex_data, outer_hex_data, outer_side_length
            
    except Exception as e:
        warnings.warn(f"Main optimization failed: {e}")
    
    # Fallback to simple configuration
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