# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import random
import time
from numba import jit, prange
import math
from scipy.optimize import minimize
from typing import Tuple, List, Optional, Any
import warnings

# Constants
UNIT_HEX_RADIUS = 1.0
MAX_EVAL_TIME = 180.0  # seconds
TARGET_RATIO = 0.2537

@jit(nopython=True)
def get_hexagon_vertices(x, y, angle_deg, radius=1.0):
    """Get vertices of a hexagon given center, angle, and radius"""
    vertices = np.zeros((6, 2))
    angle_rad = np.radians(angle_deg)
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i] = [x + radius * np.cos(theta), y + radius * np.sin(theta)]
    return vertices

class Hexagon:
    """Represents a hexagon with position and rotation."""
    
    def __init__(self, x: float, y: float, angle_deg: float, radius: float = 1.0):
        self.x = x
        self.y = y
        self.angle_deg = angle_deg
        self.radius = radius
    
    def get_vertices(self) -> np.ndarray:
        """Get vertices of the hexagon."""
        return get_hexagon_vertices(self.x, self.y, self.angle_deg, self.radius)
    
    def to_polygon(self) -> Polygon:
        """Convert hexagon to shapely polygon."""
        vertices = self.get_vertices()
        return Polygon(vertices)
    
    def copy(self) -> 'Hexagon':
        """Create a copy of this hexagon."""
        return Hexagon(self.x, self.y, self.angle_deg, self.radius)

class PackingValidator:
    """Validates hexagon packing configurations."""
    
    @staticmethod
    def check_overlap_fast(hex1_poly: Polygon, hex2_poly: Polygon) -> bool:
        """Fast overlap check using bounding boxes."""
        # Quick bounding box check first
        bbox1 = hex1_poly.bounds
        bbox2 = hex2_poly.bounds
        if (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or 
            bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1]):
            return False
        return hex1_poly.intersects(hex2_poly) and not hex1_poly.touches(hex2_poly)
    
    @staticmethod
    def validate_solution_basic(hexagons: List[Hexagon]) -> Tuple[bool, str]:
        """Basic validation without expensive containment checks."""
        if len(hexagons) != 12:
            return False, "Wrong number of hexagons"
        
        # Check for overlaps between any pair of hexagons
        # Use efficient pairwise overlap checking with early exit
        for i in range(len(hexagons)):
            hex1_poly = hexagons[i].to_polygon()
            
            for j in range(i+1, len(hexagons)):
                hex2_poly = hexagons[j].to_polygon()
                
                if PackingValidator.check_overlap_fast(hex1_poly, hex2_poly):
                    return False, f"Overlapping hexagons {i} and {j}"
        
        return True, "Valid solution"
    
    @staticmethod
    def validate_solution_complete(hexagons: List[Hexagon], outer_center: Tuple[float, float] = (0, 0)) -> Tuple[bool, str]:
        """Complete validation including containment."""
        if len(hexagons) != 12:
            return False, "Wrong number of hexagons"
        
        # Compute outer hexagon radius
        outer_radius = PackingValidator.compute_outer_hexagon_radius(hexagons)
        outer_hex = Hexagon(outer_center[0], outer_center[1], 0.0, outer_radius).to_polygon()
        
        # Check each inner hexagon
        for i, hexagon in enumerate(hexagons):
            inner_hex = hexagon.to_polygon()
            
            # Check containment
            if not outer_hex.contains(inner_hex):
                return False, f"Inner hexagon {i} not contained"
            
            # Check overlaps with others
            for j in range(i+1, len(hexagons)):
                hex2_poly = hexagons[j].to_polygon()
                
                if PackingValidator.check_overlap_fast(inner_hex, hex2_poly):
                    return False, f"Overlapping hexagons {i} and {j}"
        
        return True, "Valid solution"
    
    @staticmethod
    def compute_outer_hexagon_radius(hexagons: List[Hexagon]) -> float:
        """Compute minimum outer hexagon radius that contains all inner hexagons."""
        if len(hexagons) == 0:
            return 0.0
        
        # Get all vertices of all inner hexagons
        all_vertices = []
        for hexagon in hexagons:
            vertices = hexagon.get_vertices()
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
        return max_distance + UNIT_HEX_RADIUS

class ConfigurationGenerator:
    """Generates initial hexagon configurations."""
    
    @staticmethod
    def generate_deterministic_initial_solution() -> List[Hexagon]:
        """Generate highly optimized deterministic starting configuration."""
        # Create a highly symmetric arrangement that's known to be close to optimal
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
        
        return [Hexagon(pos[0], pos[1], pos[2]) for pos in positions[:12]]

class Optimizer:
    """Handles the optimization of hexagon configurations."""
    
    @staticmethod
    def evaluate_fitness_simple(hexagons: List[Hexagon]) -> float:
        """Simple fitness evaluation - used for preliminary checks."""
        # Check overlap constraints
        valid, msg = PackingValidator.validate_solution_basic(hexagons)
        if not valid:
            return -1e10  # Penalize invalid solutions heavily
        
        # Fitness = 1/outer_radius (higher is better)
        outer_radius = PackingValidator.compute_outer_hexagon_radius(hexagons)
        if outer_radius <= 0:
            return -1e10
        
        return 1.0 / outer_radius
    
    @staticmethod
    def solve_constraint_equilibrium(hexagons: List[Hexagon], max_iterations: int = 20) -> List[Hexagon]:
        """Iteratively solve for constraint equilibrium using a gradient-like approach."""
        # Convert to flat representation for optimization
        flat_params = []
        for hexagon in hexagons:
            flat_params.extend([hexagon.x, hexagon.y, hexagon.angle_deg])
        
        # Define the objective function - we want to minimize outer radius
        def objective(params):
            # Reshape back to hexagons format
            new_hexagons = []
            for i in range(0, len(params), 3):
                new_hexagons.append(Hexagon(params[i], params[i+1], params[i+2]))
            
            return -Optimizer.evaluate_fitness_simple(new_hexagons)  # Negative because we minimize
        
        # Bounds for positions (reasonable constraints)
        bounds = [(-10.0, 10.0)] * 36  # 12 hexagons * 3 params each
        
        try:
            # First optimize positions only (no rotations) for initial improvement
            # Fix rotations for faster convergence initially  
            fixed_rotation_params = flat_params.copy()
            for i in range(12):
                fixed_rotation_params[i*3 + 2] = 0.0  # Set all rotations to 0
                
            # Use L-BFGS-B with bounds for fast local optimization
            result = minimize(objective, fixed_rotation_params, 
                             method='L-BFGS-B', bounds=bounds, 
                             options={'maxiter': 50, 'ftol': 1e-8})
            
            if result.success:
                # Refine with rotation optimization
                refined_params = result.x.copy()
                # Allow rotation optimization for final refinement
                result_final = minimize(objective, refined_params, 
                                      method='L-BFGS-B', bounds=bounds,
                                      options={'maxiter': 30, 'ftol': 1e-10})
                if result_final.success:
                    flat_params = result_final.x
                    
        except Exception as e:
            # If optimization fails, continue with current configuration
            pass
        
        # Convert back to hexagons format
        new_hexagons = []
        for i in range(0, len(flat_params), 3):
            new_hexagons.append(Hexagon(flat_params[i], flat_params[i+1], flat_params[i+2]))
        
        # Validate and refine
        valid, _ = PackingValidator.validate_solution_basic(new_hexagons)
        if not valid:
            # Try a more conservative approach - basic constraint solving
            new_hexagons = [h.copy() for h in hexagons]
        
        return new_hexagons

def hexagon_packing_optimized():
    """Main optimized hexagon packing function using modular approach."""
    start_time = time.time()
    
    # Step 1: Generate a highly optimized initial configuration
    initial_config = ConfigurationGenerator.generate_deterministic_initial_solution()
    
    # Step 2: Apply constraint solving to improve the configuration
    refined_config = Optimizer.solve_constraint_equilibrium(initial_config)
    
    # Step 3: Apply additional optimization if time allows
    if time.time() - start_time < MAX_EVAL_TIME - 10:
        # Try a more thorough refinement
        final_config = Optimizer.solve_constraint_equilibrium(refined_config)
    else:
        final_config = refined_config
    
    # Final validation
    valid, msg = PackingValidator.validate_solution_complete(final_config, (0, 0))
    
    # If still invalid, fallback to a known good configuration
    if not valid:
        fallback_config = ConfigurationGenerator.generate_deterministic_initial_solution()
        valid, _ = PackingValidator.validate_solution_complete(fallback_config, (0, 0))
        if valid:
            final_config = fallback_config
    
    # Final computation of outer hexagon side length
    outer_hex_side_length = PackingValidator.compute_outer_hexagon_radius(final_config)
    outer_hex_data = np.array([0, 0, 0])
    
    return final_config, outer_hex_data, outer_hex_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    try:
        # Run the optimized modular approach
        hexagons, outer_hex_data, outer_hex_side_length = hexagon_packing_optimized()
        
        # Convert to required format
        inner_hex_data = np.array([[h.x, h.y, h.angle_deg] for h in hexagons])
        
    except Exception as e:
        # Fallback to simple solution
        warnings.warn(f"Fallback due to error: {e}")
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
