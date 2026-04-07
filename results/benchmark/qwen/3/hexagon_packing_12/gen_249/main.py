# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
from scipy.spatial.distance import cdist
import time
import math
from typing import Tuple, List, Optional
import warnings

class HexagonGeometry:
    """Efficient geometric operations for hexagons with vectorized computations"""
    
    @staticmethod
    def compute_hexagon_vertices_batch(positions: np.ndarray, rotations: np.ndarray) -> np.ndarray:
        """Vectorized computation of hexagon vertices for batch processing"""
        # positions: (n, 2) array of [x, y] coordinates
        # rotations: (n,) array of rotation angles in degrees
        n = len(positions)
        
        # Precompute constants
        angles = np.arange(6) * np.pi / 3  # 0, π/3, 2π/3, π, 4π/3, 5π/3
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)
        
        # Expand dimensions for broadcasting
        cos_angles = cos_angles.reshape(1, -1)  # (1, 6)
        sin_angles = sin_angles.reshape(1, -1)  # (1, 6)
        positions = positions.reshape(-1, 1, 2)  # (n, 1, 2)
        
        # Compute rotation matrix elements
        rot_rad = np.radians(rotations)  # (n,)
        rot_cos = np.cos(rot_rad).reshape(-1, 1)  # (n, 1)
        rot_sin = np.sin(rot_rad).reshape(-1, 1)  # (n, 1)
        
        # Apply rotation and translation to all vertices at once
        x_offsets = cos_angles * rot_cos - sin_angles * rot_sin  # (n, 6)
        y_offsets = sin_angles * rot_cos + cos_angles * rot_sin  # (n, 6)
        
        vertices = np.zeros((n, 6, 2))
        vertices[:, :, 0] = positions[:, :, 0] + x_offsets  # x-coordinates
        vertices[:, :, 1] = positions[:, :, 1] + y_offsets  # y-coordinates
        
        return vertices
    
    @staticmethod
    def compute_hexagon_vertices_single(center_x: float, center_y: float, 
                                      rotation_deg: float, radius: float = 1.0) -> np.ndarray:
        """Compute vertices for a single hexagon"""
        angle_rad = np.radians(rotation_deg)
        vertices = np.zeros((6, 2))
        for i in range(6):
            angle = angle_rad + i * np.pi / 3
            vertices[i] = [center_x + radius * np.cos(angle), 
                          center_y + radius * np.sin(angle)]
        return vertices

class EfficientConstraintChecker:
    """High-performance constraint checking with early termination and bounding box optimizations"""
    
    @staticmethod
    def fast_overlap_check_batch(hex_vertices: np.ndarray) -> bool:
        """Quick overlap detection using bounding boxes before precise checks"""
        n = len(hex_vertices)
        
        # Compute bounding boxes for all hexagons
        min_coords = np.min(hex_vertices, axis=1)  # (n, 2)
        max_coords = np.max(hex_vertices, axis=1)  # (n, 2)
        
        # Check overlapping bounding boxes
        for i in range(n):
            for j in range(i+1, n):
                # Early termination if bounding boxes don't overlap
                if (max_coords[i, 0] < min_coords[j, 0] or 
                    max_coords[j, 0] < min_coords[i, 0] or
                    max_coords[i, 1] < min_coords[j, 1] or
                    max_coords[j, 1] < min_coords[i, 1]):
                    continue  # No overlap in bounding boxes
                    
                # Precise overlap check if bounding boxes overlap
                poly1 = Polygon(hex_vertices[i])
                poly2 = Polygon(hex_vertices[j])
                if poly1.intersects(poly2):
                    return True  # Found overlap
                    
        return False

    @staticmethod
    def check_containment_batch(inner_vertices: np.ndarray, outer_radius: float) -> bool:
        """Check containment of all hexagons in outer hexagon"""
        # Create outer hexagon vertices
        outer_vertices = HexagonGeometry.compute_hexagon_vertices_single(0, 0, 0, outer_radius)
        outer_polygon = Polygon(outer_vertices)
        
        # Check each inner hexagon
        for vertices in inner_vertices:
            # Check containment of all vertices
            for vertex in vertices:
                point = Point(vertex[0], vertex[1])
                if not outer_polygon.contains(point):
                    return False
        return True

class HexagonPackingEvaluator:
    """Specialized evaluator for hexagon packing configurations"""
    
    @staticmethod
    def calculate_outer_radius_batch(hex_data: np.ndarray) -> float:
        """Vectorized calculation of outer radius"""
        # Extract positions
        positions = hex_data[:, :2]  # (12, 2)
        
        # Compute all distances from origin to positions
        distances = np.sqrt(np.sum(positions**2, axis=1))
        
        # Add hexagon radius (1.0) to each distance and take maximum
        max_distance = np.max(distances) + 1.0
        return max_distance

    @staticmethod
    def evaluate_batch(hex_data: np.ndarray, outer_radius: float) -> Tuple[bool, float]:
        """Batch evaluation of configuration quality"""
        # Extract positions and rotations
        positions = hex_data[:, :2]  # (12, 2)
        rotations = hex_data[:, 2]   # (12,)
        
        # Compute all vertices at once
        vertices = HexagonGeometry.compute_hexagon_vertices_batch(positions, rotations)
        
        # Check overlaps with early termination
        if EfficientConstraintChecker.fast_overlap_check_batch(vertices):
            return False, 0.0
        
        # Check containment
        if not EfficientConstraintChecker.check_containment_batch(vertices, outer_radius):
            return False, 0.0
        
        # Return inverse of outer radius as fitness value
        return True, 1.0 / outer_radius

class SymmetryAwareInitializer:
    """Generates high-quality initial configurations based on mathematical symmetries"""
    
    @staticmethod
    def generate_hexagonal_arrangement() -> np.ndarray:
        """Generate a mathematically optimized hexagonal arrangement"""
        config = np.array([
            # Central
            [0.0, 0.0, 0.0],
            # First ring (6 hexagons)
            [0.0, 2.0, 0.0],      # Top
            [1.732, 1.0, 0.0],    # Top-right
            [1.732, -1.0, 0.0],   # Bottom-right
            [0.0, -2.0, 0.0],     # Bottom
            [-1.732, -1.0, 0.0],  # Bottom-left
            [-1.732, 1.0, 0.0],   # Top-left
            # Second ring (6 hexagons)
            [3.464, 0.0, 0.0],    # Far right
            [3.464, 2.0, 0.0],    # Far top-right
            [3.464, -2.0, 0.0],   # Far bottom-right
            [-3.464, 0.0, 0.0],   # Far left
            [-3.464, 2.0, 0.0],   # Far top-left
            [-3.464, -2.0, 0.0],  # Far bottom-left
        ])
        return config[:12]  # Ensure exactly 12 hexagons

    @staticmethod
    def generate_rotated_arrangement() -> np.ndarray:
        """Generate rotated version of hexagonal arrangement"""
        config = SymmetryAwareInitializer.generate_hexagonal_arrangement()
        # Rotate by 30 degrees
        config[:, 2] = 30.0
        return config

    @staticmethod
    def generate_alternative_arrangement() -> np.ndarray:
        """Generate alternative arrangement with compact packing"""
        config = np.array([
            # Central
            [0.0, 0.0, 0.0],
            # First ring (6 hexagons) - closer spacing
            [0.0, 1.8, 0.0],      # Top
            [1.55, 0.9, 0.0],     # Top-right
            [1.55, -0.9, 0.0],    # Bottom-right
            [0.0, -1.8, 0.0],     # Bottom
            [-1.55, -0.9, 0.0],   # Bottom-left
            [-1.55, 0.9, 0.0],    # Top-left
            # Second ring (6 hexagons)
            [3.1, 0.0, 0.0],      # Far right
            [3.1, 1.8, 0.0],      # Far top-right
            [3.1, -1.8, 0.0],     # Far bottom-right
            [-3.1, 0.0, 0.0],     # Far left
            [-3.1, 1.8, 0.0],     # Far top-left
            [-3.1, -1.8, 0.0],    # Far bottom-left
        ])
        return config[:12]

def adaptive_optimization_wrapper(initial_config: np.ndarray, max_time: float = 170.0) -> np.ndarray:
    """Adaptive optimization with dynamic parameters based on early results"""
    start_time = time.time()
    
    # Try different optimization strategies based on initial performance
    packer = HexagonPackingEvaluator()
    
    # Strategy 1: Differential Evolution - usually most effective for this problem
    def objective_func(params):
        # Reshape parameters
        hex_data = initial_config.copy()
        for i in range(len(hex_data)):
            hex_data[i][0] = params[i*3]
            hex_data[i][1] = params[i*3 + 1]
            hex_data[i][2] = params[i*3 + 2]
        
        # Calculate outer radius 
        outer_radius = packer.calculate_outer_radius_batch(hex_data)
        
        # Evaluate
        validity, inv_radius = packer.evaluate_batch(hex_data, outer_radius)
        
        if not validity:
            return 1e10  # Penalty for invalid solutions
        return -inv_radius  # Minimize negative inverse radius (maximize inverse radius)
    
    # Set bounds for optimization
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (0.0, 360.0)] * 12
    
    # Run differential evolution with early stopping
    try:
        # Use fewer iterations initially to see quick convergence
        result = differential_evolution(
            objective_func,
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            callback=lambda x, f=0: time.time() - start_time > max_time - 5
        )
        
        # If we have a valid result, reconstruct configuration
        if hasattr(result, 'success') and result.success:
            optimized_config = initial_config.copy()
            for i in range(len(optimized_config)):
                optimized_config[i][0] = result.x[i*3]
                optimized_config[i][1] = result.x[i*3 + 1]
                optimized_config[i][2] = result.x[i*3 + 2]
            return optimized_config
    
    except Exception as e:
        warnings.warn(f"Differential evolution failed: {str(e)}")
    
    # Fallback to simpler local optimization approach if needed
    return initial_config

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Generate multiple high-quality initial configurations
    configs = [
        SymmetryAwareInitializer.generate_hexagonal_arrangement(),
        SymmetryAwareInitializer.generate_rotated_arrangement(), 
        SymmetryAwareInitializer.generate_alternative_arrangement()
    ]
    
    best_config = None
    best_inv_radius = 0.0
    best_outer_radius = float('inf')
    
    # Try each initial configuration
    for initial_config in configs:
        # Skip if we're running low on time
        if time.time() - start_time > 170:
            break
            
        # Optimize this configuration
        optimized_config = adaptive_optimization_wrapper(initial_config, 170 - (time.time() - start_time))
        
        # Evaluate the optimized configuration 
        packer = HexagonPackingEvaluator()
        outer_radius = packer.calculate_outer_radius_batch(optimized_config)
        validity, inv_radius = packer.evaluate_batch(optimized_config, outer_radius)
        
        if validity and inv_radius > best_inv_radius:
            best_inv_radius = inv_radius
            best_config = optimized_config.copy()
            best_outer_radius = outer_radius
            
        # Early exit if we've found a good enough solution
        if inv_radius > 0.2536:  # Close to target
            break
    
    # Fallback to known good configuration if nothing worked well
    if best_config is None:
        # Simple grid-like arrangement that is guaranteed to work
        best_config = np.array([
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
        best_inv_radius = 1.0 / 8.0
        best_outer_radius = 8.0
    
    # Prepare return values
    inner_hex_data = np.array(best_config)
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = best_outer_radius * 2
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END