# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
import time
import random
from numba import jit
import math

# Constants
UNIT_HEX_RADIUS = 1.0
MAX_EVAL_TIME = 180.0
TARGET_RATIO = 0.2537

@jit(nopython=True)
def get_hexagon_vertices_fast(x, y, angle_deg, radius=1.0):
    """Fast hexagon vertex computation using numba."""
    vertices = np.empty((6, 2))
    angle_rad = np.radians(angle_deg)
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i, 0] = x + radius * np.cos(theta)
        vertices[i, 1] = y + radius * np.sin(theta)
    return vertices

@jit(nopython=True) 
def point_in_hexagon_fast(px, py, hx, hy, angle_deg, radius=1.0):
    """Fast point-in-hexagon test."""
    angle_rad = np.radians(angle_deg)
    # Transform point to hexagon coordinate system
    dx = px - hx
    dy = py - hy
    # Rotate point by negative angle
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    rx = dx * cos_a - dy * sin_a
    ry = dx * sin_a + dy * cos_a

    # Check against hexagon boundaries
    abs_rx = np.abs(rx)
    abs_ry = np.abs(ry)
    
    # Simple boundary checks for unit hexagon
    return abs_rx <= radius and abs_ry <= radius * np.sqrt(3)/2 and abs_rx + abs_ry <= radius * (1 + np.sqrt(3)/2)

@jit(nopython=True)
def distance_squared_point_to_hexagon_fast(px, py, hx, hy, angle_deg, radius=1.0):
    """Fast distance squared from point to hexagon boundary."""
    angle_rad = np.radians(angle_deg)
    # Transform point to hexagon coordinate system
    dx = px - hx
    dy = py - hy
    # Rotate point by negative angle
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    rx = dx * cos_a - dy * sin_a
    ry = dx * sin_a + dy * cos_a

    # Distance to center minus radius (simplified)
    dist_center_sq = rx * rx + ry * ry
    return dist_center_sq - radius * radius

@jit(nopython=True)
def hexagon_overlap_fast(h1_center_x, h1_center_y, h1_angle, h2_center_x, h2_center_y, h2_angle, radius=1.0):
    """Fast overlap detection between two hexagons."""
    # Simple distance-based check first
    dx = h1_center_x - h2_center_x
    dy = h1_center_y - h2_center_y
    dist_sq = dx*dx + dy*dy

    # If centers too far apart, no overlap
    if dist_sq > 4 * radius * radius:
        return False

    # If centers too close, definitely overlap 
    if dist_sq < 0.1:
        return True

    # Conservative overlap check for unit hexagons
    return dist_sq < 4.0 * radius * radius

class HexagonPackingEvolutionary:
    """High-performance hexagon packing optimizer using evolutionary principles"""
    
    def __init__(self):
        self.start_time = time.time()
        self.best_score = 0.0
        self.best_config = None
        self.target_ratio = TARGET_RATIO
        
    def get_elapsed_time(self):
        return time.time() - self.start_time
        
    @staticmethod
    def create_lattice_configurations():
        """Generate multiple lattice-based configurations using mathematical principles"""
        configs = []
        
        # Kagome lattice-inspired arrangement
        config1 = np.array([
            [0, 0, 0],        # center
            [0, 2.0, 0],      # top
            [0, -2.0, 0],     # bottom
            [1.732, 1.0, 0],  # top-right
            [-1.732, 1.0, 0], # top-left
            [1.732, -1.0, 0], # bottom-right
            [-1.732, -1.0, 0],# bottom-left
            [3.464, 0, 0],    # far right
            [-3.464, 0, 0],   # far left
            [1.732, 3.0, 0],  # upper right corner
            [-1.732, 3.0, 0], # upper left corner
            [1.732, -3.0, 0], # lower right corner
            [-1.732, -3.0, 0] # lower left corner
        ])
        configs.append(config1[:12])
        
        # Hexagonal close-packed (HCP) inspired arrangement
        config2 = np.array([
            [0, 0, 0],           # center
            [0, 1.9, 0],         # top
            [0, -1.9, 0],        # bottom
            [1.65, 0.95, 0],     # top-right
            [-1.65, 0.95, 0],    # top-left
            [1.65, -0.95, 0],    # bottom-right
            [-1.65, -0.95, 0],   # bottom-left
            [3.3, 0, 0],         # far right
            [-3.3, 0, 0],        # far left
            [1.65, 2.85, 0],     # upper right corner
            [-1.65, 2.85, 0],    # upper left corner
            [1.65, -2.85, 0],    # lower right corner
            [-1.65, -2.85, 0]    # lower left corner
        ])
        configs.append(config2[:12])
        
        # Compact ring arrangement
        config3 = np.array([
            [0, 0, 0],           # center
            [0, 2.1, 0],         # top
            [1.8, 1.0, 0],       # top-right
            [1.8, -1.0, 0],      # bottom-right
            [0, -2.1, 0],        # bottom
            [-1.8, -1.0, 0],     # bottom-left
            [-1.8, 1.0, 0],      # top-left
            [3.6, 0, 0],         # far right
            [0, 3.6, 0],         # far top
            [-3.6, 0, 0],        # far left
            [0, -3.6, 0],        # far bottom
            [1.8, 2.1, 0],       # upper right corner
            [-1.8, 2.1, 0],      # upper left corner
            [1.8, -2.1, 0],      # lower right corner
            [-1.8, -2.1, 0]      # lower left corner
        ])
        configs.append(config3[:12])
        
        # Optimized geometric arrangement
        config4 = np.array([
            [0, 0, 0],           # center
            [0, 1.8, 0],         # top
            [0, -1.8, 0],        # bottom
            [1.55, 0.9, 0],      # top-right
            [-1.55, 0.9, 0],     # top-left
            [1.55, -0.9, 0],     # bottom-right
            [-1.55, -0.9, 0],    # bottom-left
            [3.1, 0, 0],         # far right
            [-3.1, 0, 0],        # far left
            [1.55, 2.7, 0],      # upper right corner
            [-1.55, 2.7, 0],     # upper left corner
            [1.55, -2.7, 0],     # lower right corner
            [-1.55, -2.7, 0]     # lower left corner
        ])
        configs.append(config4[:12])
        
        return configs
        
    @staticmethod
    def compute_outer_hex_radius_fast(hex_data):
        """Fast computation of outer hexagon radius."""
        if len(hex_data) == 0:
            return 0.0
            
        max_distance = 0.0
        for i in range(len(hex_data)):
            x, y, _ = hex_data[i]
            distance = np.sqrt(x*x + y*y)
            max_distance = max(max_distance, distance + UNIT_HEX_RADIUS)
        return max_distance + 0.1

    @staticmethod
    def evaluate_feasibility_fast(hex_data):
        """Fast feasibility evaluation with early termination."""
        # Early time check
        if time.time() - self.start_time > MAX_EVAL_TIME * 0.95:
            return 0.0
            
        # Check overlaps using fast method
        n = len(hex_data)
        for i in range(n):
            if time.time() - self.start_time > MAX_EVAL_TIME * 0.95:
                return 0.0
                
            for j in range(i+1, n):
                if time.time() - self.start_time > MAX_EVAL_TIME * 0.95:
                    return 0.0
                    
                # Fast overlap check
                if hexagon_overlap_fast(
                    hex_data[i][0], hex_data[i][1], hex_data[i][2],
                    hex_data[j][0], hex_data[j][1], hex_data[j][2]
                ):
                    # Verify with precise check
                    try:
                        v1 = get_hexagon_vertices_fast(hex_data[i][0], hex_data[i][1], hex_data[i][2])
                        v2 = get_hexagon_vertices_fast(hex_data[j][0], hex_data[j][1], hex_data[j][2])
                        p1 = Polygon(v1)
                        p2 = Polygon(v2)
                        if p1.intersects(p2):
                            return 0.0  # Invalid due to overlap
                    except:
                        return 0.0
                        
        # Check containment using fast method
        outer_radius = HexagonPackingEvolutionary.compute_outer_hex_radius_fast(hex_data)
        center_x, center_y = 0.0, 0.0
        
        for i in range(n):
            if time.time() - self.start_time > MAX_EVAL_TIME * 0.95:
                return 0.0
                
            # Fast containment check
            x, y, _ = hex_data[i]
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if distance + UNIT_HEX_RADIUS > outer_radius:
                return 0.0  # Not contained
                
        # Return inverse of outer radius when valid
        return 1.0 / outer_radius if outer_radius > 0 else 0.0

    def optimize_layered(self, initial_config):
        """Multi-layered optimization approach"""
        # Layer 1: Global structure optimization
        print("Optimizing global structure...")
        layer1_config = self.global_structure_optimize(initial_config)
        
        # Layer 2: Position refinement  
        print("Refining positions...")
        layer2_config = self.position_refinement(layer1_config)
        
        # Layer 3: Rotation optimization
        print("Optimizing rotations...")
        layer3_config = self.rotation_optimization(layer2_config)
        
        return layer3_config

    def global_structure_optimize(self, config):
        """Optimize global hexagon arrangement structure"""
        # Simple L-BFGS-B optimization on positions only
        def objective(params):
            # Reconstruct positions
            positions = params.reshape(-1, 2)
            temp_config = config.copy()
            temp_config[:, 0] = positions[:, 0]
            temp_config[:, 1] = positions[:, 1]
            
            # Check validity and return negative for minimization
            score = self.evaluate_feasibility_fast(temp_config)
            return -score if score > 0 else 1e10

        # Flatten positions for optimization
        flat_positions = config[:, :2].flatten()
        
        # Bounds for positions
        bounds = [(-10, 10), (-10, 10)] * 12
        
        try:
            result = minimize(
                objective, 
                flat_positions,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if result.success:
                final_positions = result.x.reshape(-1, 2)
                optimized_config = config.copy()
                optimized_config[:, 0] = final_positions[:, 0]
                optimized_config[:, 1] = final_positions[:, 1]
                return optimized_config
        except:
            pass
            
        return config

    def position_refinement(self, config):
        """Fine tune positions with tighter constraints"""
        # More aggressive optimization with tighter tolerances
        def objective(params):
            positions = params.reshape(-1, 2)
            temp_config = config.copy()
            temp_config[:, 0] = positions[:, 0]
            temp_config[:, 1] = positions[:, 1]
            
            score = self.evaluate_feasibility_fast(temp_config)
            return -score if score > 0 else 1e10
            
        flat_positions = config[:, :2].flatten()
        bounds = [(-8, 8), (-8, 8)] * 12
        
        try:
            result = minimize(
                objective,
                flat_positions,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10}
            )
            
            if result.success:
                final_positions = result.x.reshape(-1, 2)
                optimized_config = config.copy()
                optimized_config[:, 0] = final_positions[:, 0]
                optimized_config[:, 1] = final_positions[:, 1]
                return optimized_config
        except:
            pass
            
        return config

    def rotation_optimization(self, config):
        """Optimize rotations for better packing"""
        # Optimize rotations with positional constraints fixed
        def objective(angles):
            temp_config = config.copy()
            temp_config[:, 2] = angles
            
            score = self.evaluate_feasibility_fast(temp_config)
            return -score if score > 0 else 1e10
            
        initial_angles = config[:, 2].tolist()
        bounds = [(-180, 180)] * 12
        
        try:
            result = minimize(
                objective,
                initial_angles,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            
            if result.success:
                final_angles = result.x
                optimized_config = config.copy()
                optimized_config[:, 2] = final_angles
                return optimized_config
        except:
            pass
            
        return config

    def run_optimization(self):
        """Main optimization routine with multiple strategies"""
        # Generate multiple high-quality initial configurations
        configs = self.create_lattice_configurations()
        
        best_score = 0.0
        best_config = None
        
        # Try different starting configurations
        for i, initial_config in enumerate(configs):
            # Add some randomization to escape local minima
            random_config = initial_config.copy()
            for j in range(12):
                random_config[j, 0] += random.uniform(-0.2, 0.2)
                random_config[j, 1] += random.uniform(-0.2, 0.2)
                random_config[j, 2] += random.uniform(-5, 5)
            
            # Layered optimization approach
            optimized_config = self.optimize_layered(random_config)
            
            # Evaluate final result
            score = self.evaluate_feasibility_fast(optimized_config)
            
            if score > best_score:
                best_score = score
                best_config = optimized_config.copy()
                
            # Early termination check
            if self.get_elapsed_time() > MAX_EVAL_TIME * 0.95:
                break
                
        # If no valid solution found, return the best we have
        if best_config is None:
            best_config = configs[0]
            
        return best_config, best_score

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    optimizer = HexagonPackingEvolutionary()
    best_config, best_score = optimizer.run_optimization()
    
    # Compute final outer hexagon radius
    outer_radius = 1.0 / best_score if best_score > 0 else 10.0
    
    # Ensure proper shape
    if len(best_config) != 12:
        # Fallback to a simple configuration
        best_config = np.array([
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
        outer_radius = 8.0

    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = outer_radius * 2  # Approximate, since we're measuring from center

    return best_config, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END