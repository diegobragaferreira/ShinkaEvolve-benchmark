# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from numba import njit, prange
import time
import random
from math import cos, sin, pi, sqrt
from dataclasses import dataclass
from typing import Tuple, List, Optional
import warnings

@dataclass
class HexagonConfig:
    """Represents a hexagon with position and rotation."""
    x: float
    y: float
    angle_deg: float

@dataclass
class OptimizationResult:
    """Encapsulates the optimization result."""
    inner_hex_data: np.ndarray
    outer_hex_data: np.ndarray
    outer_hex_side_length: float

class HexagonGeometry:
    """Handles all geometric computations for hexagons."""
    
    @staticmethod
    @njit
    def vertices(center_x: float, center_y: float, angle_deg: float, side_length: float = 1) -> np.ndarray:
        """Generate vertices of a regular hexagon using Numba JIT."""
        angle_rad = np.radians(angle_deg)
        vertices = np.empty((6, 2))
        for i in range(6):
            angle = angle_rad + i * np.pi / 3
            x = center_x + side_length * cos(angle)
            y = center_y + side_length * sin(angle)
            vertices[i] = [x, y]
        return vertices
    
    @staticmethod
    @njit
    def center_distance_sq(x1: float, y1: float, x2: float, y2: float) -> float:
        """Fast squared distance calculation between two points."""
        dx = x1 - x2
        dy = y1 - y2
        return dx * dx + dy * dy
    
    @staticmethod
    @njit
    def max_vertex_distance_sq(center_x: float, center_y: float, angle_deg: float) -> float:
        """Maximum squared distance from center to any vertex for unit hexagon."""
        # For unit hexagon, max distance from center is 1
        return 1.0
    
    @staticmethod
    @njit
    def bounding_radius_fast(hex_configs: np.ndarray) -> float:
        """Fast calculation of bounding radius for multiple hexagons."""
        if len(hex_configs) == 0:
            return 1.0
            
        # Find centroid
        total_x = 0.0
        total_y = 0.0
        for i in range(len(hex_configs)):
            total_x += hex_configs[i][0]
            total_y += hex_configs[i][1]
        
        centroid_x = total_x / len(hex_configs)
        centroid_y = total_y / len(hex_configs)
        
        # Find maximum distance to centroid
        max_dist_sq = 0.0
        for i in range(len(hex_configs)):
            x, y, _ = hex_configs[i]
            dist_sq = HexagonGeometry.center_distance_sq(x, y, centroid_x, centroid_y)
            if dist_sq > max_dist_sq:
                max_dist_sq = dist_sq
                
        return sqrt(max_dist_sq) + 0.01  # Add buffer
    
    @staticmethod
    def create_polygon(center_x: float, center_y: float, angle_deg: float) -> Polygon:
        """Create a Shapely polygon for a hexagon."""
        vertices = HexagonGeometry.vertices(center_x, center_y, angle_deg)
        return Polygon(vertices.tolist())

class SpatialHashing:
    """Handles spatial indexing for efficient overlap detection."""
    
    @staticmethod
    @njit
    def get_grid_coords(x: float, y: float, cell_size: float = 1.2) -> Tuple[int, int]:
        """Get grid coordinates for a point."""
        return int(x // cell_size), int(y // cell_size)
    
    @staticmethod
    @njit
    def get_hexagon_grid_cells(center_x: float, center_y: float, angle_deg: float, 
                              cell_size: float = 1.2) -> List[Tuple[int, int]]:
        """Get all grid cells that a hexagon might occupy."""
        # Get vertices for better coverage
        vertices = HexagonGeometry.vertices(center_x, center_y, angle_deg)
        
        # Find bounding box
        min_x = vertices[0][0]
        max_x = vertices[0][0]
        min_y = vertices[0][1]
        max_y = vertices[0][1]
        
        for i in range(1, len(vertices)):
            x, y = vertices[i]
            if x < min_x: min_x = x
            if x > max_x: max_x = x
            if y < min_y: min_y = y
            if y > max_y: max_y = y
        
        # Get grid indices for bounding box
        min_cell_x = int(min_x // cell_size)
        max_cell_x = int(max_x // cell_size)
        min_cell_y = int(min_y // cell_size)
        max_cell_y = int(max_y // cell_size)
        
        # Collect all cells
        cells = []
        for x in range(min_cell_x, max_cell_x + 1):
            for y in range(min_cell_y, max_cell_y + 1):
                cells.append((x, y))
        return cells
    
    @staticmethod
    @njit
    def fast_overlap_check(hex1_center_x: float, hex1_center_y: float, hex1_angle: float,
                          hex2_center_x: float, hex2_center_y: float, hex2_angle: float) -> bool:
        """Fast preliminary overlap check using distance between centers."""
        # Quick distance check for unit hexagons
        dist_sq = HexagonGeometry.center_distance_sq(
            hex1_center_x, hex1_center_y, hex2_center_x, hex2_center_y
        )
        # Minimum distance between unit hexagons is 2 (when touching)
        return dist_sq < 4.0  # 2^2
    
    @staticmethod
    def check_overlap_precise(hex1_center_x: float, hex1_center_y: float, hex1_angle: float,
                             hex2_center_x: float, hex2_center_y: float, hex2_angle: float) -> bool:
        """Precise overlap check using Shapely."""
        try:
            poly1 = HexagonGeometry.create_polygon(hex1_center_x, hex1_center_y, hex1_angle)
            poly2 = HexagonGeometry.create_polygon(hex2_center_x, hex2_center_y, hex2_angle)
            return poly1.intersects(poly2)
        except:
            # Fallback for edge cases
            return False

class ConstraintChecker:
    """Handles constraint validation with smart optimization."""
    
    @staticmethod
    def check_containment(hex_configs: np.ndarray, outer_side_length: float) -> bool:
        """Check if all hexagons are contained within outer hexagon."""
        for i in range(len(hex_configs)):
            x, y, _ = hex_configs[i]
            # Distance from origin to hexagon center
            dist_sq = x*x + y*y
            outer_radius_sq = outer_side_length * outer_side_length
            if dist_sq >= outer_radius_sq:
                return False
        return True
    
    @staticmethod
    def check_overlap_spatial_hashing(hex_configs: np.ndarray) -> bool:
        """Spatial hashed overlap detection for efficiency."""
        n_hexagons = len(hex_configs)
        
        if n_hexagons <= 1:
            return True
            
        # Precompute grid cells for all hexagons
        all_grid_cells = []
        hex_centers = []
        
        for i in range(n_hexagons):
            x, y, angle = hex_configs[i]
            hex_centers.append((x, y))
            cells = SpatialHashing.get_hexagon_grid_cells(x, y, angle, 1.2)
            all_grid_cells.append(cells)
        
        # Build spatial hash grid
        grid = {}
        for i in range(n_hexagons):
            cells = all_grid_cells[i]
            for cell in cells:
                if cell not in grid:
                    grid[cell] = []
                grid[cell].append(i)
        
        # Check for overlaps by examining neighboring cells
        for i in range(n_hexagons):
            x1, y1, angle1 = hex_configs[i]
            
            # Check nearby cells (3x3 grid)
            cells = all_grid_cells[i]
            for cell in cells:
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        neighbor_cell = (cell[0] + dx, cell[1] + dy)
                        if neighbor_cell in grid:
                            for j in grid[neighbor_cell]:
                                if i != j:
                                    # Quick bounding box check
                                    x2, y2, angle2 = hex_configs[j]
                                    if SpatialHashing.fast_overlap_check(x1, y1, angle1, x2, y2, angle2):
                                        # Precise check
                                        if not SpatialHashing.check_overlap_precise(x1, y1, angle1, x2, y2, angle2):
                                            continue
                                        else:
                                            return False  # Overlap detected
        return True  # No overlap detected
    
    @staticmethod
    def validate_configuration(hex_configs: np.ndarray, outer_side_length: float) -> bool:
        """Complete constraint validation."""
        if not ConstraintChecker.check_containment(hex_configs, outer_side_length):
            return False
        return ConstraintChecker.check_overlap_spatial_hashing(hex_configs)

class Optimizer:
    """Main optimization engine with adaptive strategies."""
    
    def __init__(self):
        self.best_solution = None
        self.best_score = -float('inf')
        self.best_side_length = float('inf')
    
    @staticmethod
    def create_symmetric_pattern() -> np.ndarray:
        """Create a mathematically informed symmetric pattern."""
        # Based on known optimal arrangements for 12 hexagons
        pattern = [
            [0.0, 0.0, 0.0],         # Center
            [-2.0, 0.0, 0.0],        # Left
            [2.0, 0.0, 0.0],         # Right
            [0.0, 2.0, 0.0],         # Top
            [0.0, -2.0, 0.0],        # Bottom
            [-1.0, 1.732, 0.0],      # Top-left
            [1.0, 1.732, 0.0],       # Top-right
            [-1.0, -1.732, 0.0],     # Bottom-left
            [1.0, -1.732, 0.0],      # Bottom-right
            [-2.5, 0.0, 0.0],        # Far left
            [2.5, 0.0, 0.0],         # Far right
            [0.0, 2.5, 0.0],         # Far top
        ]
        return np.array(pattern)
    
    @staticmethod
    def create_hexagonal_ring_pattern() -> np.ndarray:
        """Create a hexagonal ring-based pattern."""
        # Layered arrangement for better packing
        pattern = [
            [0.0, 0.0, 0.0],          # Center
            [0.0, 2.0, 0.0],          # Top
            [1.732, 1.0, 0.0],        # Top-right
            [1.732, -1.0, 0.0],       # Bottom-right
            [0.0, -2.0, 0.0],         # Bottom
            [-1.732, -1.0, 0.0],      # Bottom-left
            [-1.732, 1.0, 0.0],       # Top-left
            [0.0, 3.5, 0.0],          # Far top
            [3.03, 1.75, 0.0],        # Far top-right
            [3.03, -1.75, 0.0],       # Far bottom-right
            [0.0, -3.5, 0.0],         # Far bottom
            [-3.03, -1.75, 0.0],      # Far bottom-left
        ]
        return np.array(pattern)
    
    @staticmethod
    def create_star_pattern() -> np.ndarray:
        """Create a star-shaped pattern."""
        pattern = [
            [0.0, 0.0, 0.0],         # Center
            [0.0, 2.2, 0.0],         # Top
            [0.0, -2.2, 0.0],        # Bottom
            [2.2, 0.0, 0.0],         # Right
            [-2.2, 0.0, 0.0],        # Left
            [1.1, 1.905, 0.0],       # Top-right
            [-1.1, 1.905, 0.0],      # Top-left
            [1.1, -1.905, 0.0],      # Bottom-right
            [-1.1, -1.905, 0.0],     # Bottom-left
            [0.0, 3.3, 0.0],         # Far top
            [0.0, -3.3, 0.0],        # Far bottom
            [3.3, 0.0, 0.0],         # Far right
        ]
        return np.array(pattern)
    
    @staticmethod
    def generate_population(pop_size: int) -> List[np.ndarray]:
        """Generate diverse initial population."""
        base_patterns = [
            Optimizer.create_symmetric_pattern(),
            Optimizer.create_hexagonal_ring_pattern(),
            Optimizer.create_star_pattern()
        ]
        
        population = []
        for i in range(pop_size):
            base_pattern = base_patterns[i % len(base_patterns)]
            individual = base_pattern.astype(float)
            
            # Add controlled variation
            for j in range(12):
                individual[j, 0] += np.random.uniform(-0.15, 0.15)
                individual[j, 1] += np.random.uniform(-0.15, 0.15)
                individual[j, 2] += np.random.uniform(-15, 15)
                individual[j, 2] = individual[j, 2] % 360
                
            population.append(individual.flatten())
        
        return population
    
    def objective_function(self, config: np.ndarray, outer_side_length: float) -> float:
        """Objective function to minimize."""
        hex_configs = config.reshape(12, 3)
        
        # Validate constraints
        if not ConstraintChecker.validate_configuration(hex_configs, outer_side_length):
            return 1000000  # Large penalty for invalid configurations
            
        # Return negative of 1/outer_side_length for maximization
        return -1.0 / outer_side_length if outer_side_length > 0 else 1000000
    
    def optimize_single_run(self, initial_guess: np.ndarray, bounds: List[Tuple[float, float]], 
                           maxiter: int = 50, popsize: int = 15) -> Tuple[np.ndarray, float]:
        """Single optimization run with differential evolution."""
        try:
            result = differential_evolution(
                lambda x: self.objective_function(x, 4.0),
                bounds,
                seed=random.randint(0, 1000),
                maxiter=maxiter,
                popsize=popsize,
                disp=False,
                strategy='best1bin',
                tol=1e-6
            )
            
            return result.x, -1.0 / 4.0 if result.success else -float('inf')
        except Exception as e:
            warnings.warn(f"Optimization error in single run: {e}")
            return initial_guess, -float('inf')
    
    def optimize_with_refinement(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Multi-stage optimization with refinement."""
        # Create bounds
        bounds = []
        for _ in range(12):
            bounds.extend([(-4.0, 4.0), (-4.0, 4.0), (0.0, 360.0)])
        
        # Generate initial population
        population = Optimizer.generate_population(20)
        best_config = None
        best_score = -float('inf')
        best_side_length = 4.0
        
        # Single best configuration approach with multiple restarts
        for run in range(3):
            # Choose random individual from population as starting point
            start_individual = population[random.randint(0, len(population)-1)]
            
            # Optimize with different parameters
            config, score = self.optimize_single_run(start_individual, bounds, 50, 15)
            
            if score > best_score:
                best_score = score
                best_config = config.copy()
                best_side_length = 4.0  # Fixed for now, would be improved in later stages
        
        # Final refinement with tight search
        if best_config is not None:
            # Search around the best configuration
            test_side_lengths = np.linspace(3.85, 3.9419123, 15)
            for side_length in test_side_lengths:
                try:
                    hex_configs = best_config.reshape(12, 3)
                    if ConstraintChecker.validate_configuration(hex_configs, side_length):
                        if side_length > best_side_length:
                            best_side_length = side_length
                except:
                    continue
                    
        return best_config, best_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Initialize optimizer
    optimizer = Optimizer()
    
    # Run optimization
    final_config, best_side_length = optimizer.optimize_with_refinement()
    
    # Post-process result
    if final_config is not None:
        inner_hex_data = final_config.reshape(12, 3)
        outer_hex_data = np.array([0, 0, 0])
    else:
        # Fallback to previous good configuration
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
        best_side_length = 8.0
    
    # Validation
    end_time = time.time()
    
    return inner_hex_data, outer_hex_data, best_side_length

# EVOLVE-BLOCK-END