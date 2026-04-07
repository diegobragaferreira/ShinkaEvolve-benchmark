# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
from scipy.spatial.distance import cdist
import time
import math
import random
from typing import Tuple, List, Optional, Any
from dataclasses import dataclass
from numba import jit

@dataclass
class HexagonConfig:
    """Data class for hexagon configuration storage."""
    centers: np.ndarray  # Shape (12, 2) - x, y coordinates
    rotations: np.ndarray  # Shape (12,) - rotation angles in degrees

@jit(nopython=True)
def get_hexagon_vertices_numba(x, y, angle_deg, radius=1.0):
    """Get vertices of a hexagon given center, angle, and radius - numba compiled"""
    vertices = np.zeros((6, 2))
    angle_rad = np.radians(angle_deg)
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i] = [x + radius * np.cos(theta), y + radius * np.sin(theta)]
    return vertices

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
    def create_hexagon_vertices_numba(center_x: float, center_y: float, side_length: float, rotation_deg: float) -> np.ndarray:
        """Numba-compiled hexagon vertex creation."""
        return get_hexagon_vertices_numba(center_x, center_y, rotation_deg, side_length)
    
    @staticmethod
    def check_overlap_fast(hex1_vertices: np.ndarray, hex2_vertices: np.ndarray) -> bool:
        """Fast overlap checking using bounding boxes before precise check."""
        # Quick bounding box check first
        bbox1 = [np.min(hex1_vertices[:, 0]), np.min(hex1_vertices[:, 1]), 
                 np.max(hex1_vertices[:, 0]), np.max(hex1_vertices[:, 1])]
        bbox2 = [np.min(hex2_vertices[:, 0]), np.min(hex2_vertices[:, 1]), 
                 np.max(hex2_vertices[:, 0]), np.max(hex2_vertices[:, 1])]
        
        if (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or 
            bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1]):
            return False
            
        # Precise check
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
            # Create hexagon polygons efficiently
            hexagons = []
            vertices_list = []
            
            for i in range(12):
                center = (hex_config.centers[i][0], hex_config.centers[i][1])
                angle = hex_config.rotations[i]
                
                vertices = GeometryUtils.create_hexagon_vertices(center[0], center[1], 1.0, angle)
                vertices_list.append(vertices)
                hexagon = Polygon(vertices)
                hexagons.append(hexagon)
            
            # Check overlaps with fast bounding box pre-check
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
            return False, 0.0
    
    def _validate_cache(self, hex_config: HexagonConfig):
        """Clear cache when configuration changes."""
        self._cached_polygons.clear()
        self._cache_valid = True

class ConfigurationBuilder:
    """Creates various initial configurations for optimization."""
    
    @staticmethod
    def get_optimized_configurations() -> List[HexagonConfig]:
        """Generate multiple optimized configurations."""
        configs = []
        
        # Configuration 1: Optimized 2-ring arrangement with careful spacing
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
        
        # Configuration 2: Honeycomb-like arrangement with slight perturbations
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
                              max_iter: int, bounds: List[Tuple[float, float]]) -> Tuple[HexagonConfig, float]:
        """Perform optimization at specific stage."""
        # Flatten for optimization
        initial_flat = self._flatten_config(initial_config)
        
        try:
            # Optimize using L-BFGS-B
            result = minimize(
                self._objective_function,
                initial_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.success:
                optimized_config = self._unflatten_config(result.x)
                _, score = self.constraint_checker.validate_and_score(optimized_config)
                return optimized_config, score
                
        except Exception as e:
            pass
        
        return initial_config, 0.0
    
    def optimize(self, initial_configs: List[HexagonConfig]) -> Tuple[HexagonConfig, float]:
        """Multi-stage optimization with progressive refinement."""
        best_config = None
        best_score = 0.0
        
        for i, config in enumerate(initial_configs):
            # Stage 1: Coarse optimization with broader bounds
            bounds_coarse = [(-8, 8), (-8, 8)] * 12 + [(-180, 180)] * 12
            stage1_result, stage1_score = self._optimize_single_stage(config, 
                                                                    200, bounds_coarse)
            
            # Stage 2: Medium refinement with tighter bounds
            bounds_medium = [(-6, 6), (-6, 6)] * 12 + [(-180, 180)] * 12
            stage2_result, stage2_score = self._optimize_single_stage(stage1_result, 
                                                                    300, bounds_medium)
            
            # Stage 3: Fine tuning with very tight bounds
            bounds_fine = [(-5, 5), (-5, 5)] * 12 + [(-180, 180)] * 12
            stage3_result, stage3_score = self._optimize_single_stage(stage2_result, 
                                                                    400, bounds_fine)
            
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

def generate_deterministic_initial_solution():
    """Generate highly optimized deterministic starting configuration"""
    # Based on proven mathematical approach for hexagon packing
    # Using sqrt(3) based distances for optimal packing efficiency
    positions = [
        # Central hexagon
        [0.0, 0.0, 0.0],
        # First shell - 6 hexagons arranged in a hexagon pattern
        [0.0, 2.0, 0.0],      # Top
        [1.732, 1.0, 0.0],    # Top right
        [1.732, -1.0, 0.0],   # Bottom right
        [0.0, -2.0, 0.0],     # Bottom
        [-1.732, -1.0, 0.0],  # Bottom left
        [-1.732, 1.0, 0.0],   # Top left
        # Second shell - 6 hexagons in larger hexagon pattern
        [3.464, 0.0, 0.0],    # Far right
        [3.464, 2.0, 0.0],    # Far top right
        [3.464, -2.0, 0.0],   # Far bottom right
        [-3.464, 0.0, 0.0],   # Far left
        [-3.464, 2.0, 0.0],   # Far top left
        [-3.464, -2.0, 0.0],  # Far bottom left
    ]
    
    # Return first 12 positions, ensuring exact count
    return np.array(positions[:12])

def solve_constraint_equilibrium(hex_data):
    """Solve constraint equilibrium using hybrid optimization approach"""
    # Convert to flat representation for optimization
    flat_params = hex_data.flatten()
    
    # Define the objective function - we want to minimize outer radius
    def objective(params):
        # Reshape back to hex_data format
        new_hex_data = params.reshape(-1, 3)
        # Simple fitness evaluation
        valid, fitness = validate_solution_basic(new_hex_data)
        if not valid:
            return 1e6
        return -fitness  # Negative because we minimize
    
    # Bounds for positions (reasonable constraints)
    bounds = [(-10.0, 10.0)] * 36  # 12 hexagons * 3 params each
    
    # Apply optimization using L-BFGS-B with bounds
    try:
        result = minimize(
            objective,
            flat_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-10}
        )
        
        if result.success:
            flat_params = result.x
    
    except Exception as e:
        # If optimization fails, continue with current configuration
        pass
    
    # Convert back to hex_data format
    new_hex_data = flat_params.reshape(-1, 3)
    
    return new_hex_data

def validate_solution_basic(inner_hex_data):
    """Basic validation without expensive containment checks"""
    if len(inner_hex_data) != 12:
        return False, "Wrong number of hexagons"
    
    # Check for overlaps between any pair of hexagons
    for i in range(len(inner_hex_data)):
        x1, y1, angle1 = inner_hex_data[i]
        vertices1 = get_hexagon_vertices_numba(x1, y1, angle1, 1.0)
        poly1 = Polygon(vertices1)
        
        for j in range(i+1, len(inner_hex_data)):
            x2, y2, angle2 = inner_hex_data[j]
            vertices2 = get_hexagon_vertices_numba(x2, y2, angle2, 1.0)
            poly2 = Polygon(vertices2)
            
            if poly1.intersects(poly2):
                return False, f"Overlapping hexagons {i} and {j}"
    
    # Fitness = 1/outer_radius (higher is better)
    outer_radius = compute_outer_hexagon_radius(inner_hex_data)
    if outer_radius <= 0:
        return False, "Invalid outer radius"
    
    return True, 1.0 / outer_radius

def compute_outer_hexagon_radius(inner_hex_data):
    """Compute minimum outer hexagon radius that contains all inner hexagons"""
    if len(inner_hex_data) == 0:
        return 0.0
    
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        vertices = get_hexagon_vertices_numba(x, y, angle, 1.0)
        all_vertices.extend(vertices)
    
    if len(all_vertices) == 0:
        return 0.0
    
    # Compute centroid
    centroid_x = np.mean([v[0] for v in all_vertices])
    centroid_y = np.mean([v[1] for v in all_vertices])
    
    # Find maximum distance from centroid to any vertex
    max_distance = 0.0
    for x, y in all_vertices:
        distance = math.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
        max_distance = max(max_distance, distance)
    
    # Add buffer for hexagon radius calculation
    return max_distance + 1.0

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    try:
        # Run the optimized deterministic approach
        initial_config = generate_deterministic_initial_solution()
        refined_config = solve_constraint_equilibrium(initial_config)
        final_config = solve_constraint_equilibrium(refined_config)
        
        # Validate final solution
        valid, fitness = validate_solution_basic(final_config)
        if not valid:
            raise ValueError("Final validation failed")
        
        # Calculate outer hexagon side length
        outer_hex_side_length = compute_outer_hexagon_radius(final_config)
        outer_hex_data = np.array([0, 0, 0])
        
        # Convert to expected return format
        inner_hex_data = final_config
        
        return inner_hex_data, outer_hex_data, outer_hex_side_length
        
    except Exception as e:
        # Fallback to simple solution
        print(f"Fallback due to error: {e}")
        inner_hex_data = np.array([
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
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END