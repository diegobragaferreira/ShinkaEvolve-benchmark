# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
from scipy.optimize import minimize
import math
import random
from itertools import combinations
from typing import Tuple, List, Optional
import time
import itertools

class HexagonGeometry:
    """Handles all geometric operations for hexagons"""
    
    @staticmethod
    def create_unit_hexagon(center: Tuple[float, float] = (0, 0), rotation: float = 0) -> Polygon:
        """Create a unit regular hexagon with given center and rotation"""
        angle_offset = math.radians(rotation)
        radius = 1
        vertices = []
        for i in range(6):
            angle = angle_offset + i * math.pi / 3
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            vertices.append((x, y))
        return Polygon(vertices)

    @staticmethod
    def get_all_vertices(hex_data: np.ndarray) -> List[Tuple[float, float]]:
        """Extract all vertices from all hexagons"""
        all_vertices = []
        for i in range(len(hex_data)):
            center = (hex_data[i][0], hex_data[i][1])
            rotation = hex_data[i][2]
            hexagon = HexagonGeometry.create_unit_hexagon(center, rotation)
            all_vertices.extend(list(hexagon.exterior.coords))
        return all_vertices

class HexagonConstraintChecker:
    """Handles constraint checking for hexagon arrangements"""
    
    @staticmethod
    def check_overlap(hex1: Polygon, hex2: Polygon) -> bool:
        """Check if two hexagons overlap with numerical stability"""
        # Add small buffer to handle floating point precision issues
        buffered_hex1 = hex1.buffer(1e-10)
        buffered_hex2 = hex2.buffer(1e-10)
        return buffered_hex1.intersects(buffered_hex2)

    @staticmethod
    def check_containment(inner_hex: Polygon, outer_hex: Polygon) -> bool:
        """Check if inner hexagon is fully contained within outer hexagon"""
        return outer_hex.contains(inner_hex)

    @staticmethod
    def compute_overlap_penalty(hexagons: List[Polygon]) -> float:
        """Compute penalty for overlaps between hexagons"""
        penalty = 0
        n = len(hexagons)
        for i in range(n):
            for j in range(i+1, n):
                if HexagonConstraintChecker.check_overlap(hexagons[i], hexagons[j]):
                    penalty += 1000
        return penalty

class HexagonPackingEvaluator:
    """Evaluates hexagon packing configurations"""
    
    @staticmethod
    def calculate_outer_hex_radius(hex_data: np.ndarray) -> float:
        """Calculate minimum outer hexagon radius needed to contain all inner hexagons"""
        all_vertices = HexagonGeometry.get_all_vertices(hex_data)
        max_distance = 0
        for vertex in all_vertices:
            distance = math.sqrt(vertex[0]**2 + vertex[1]**2)
            max_distance = max(max_distance, distance)
        return max_distance + 0.1

    @staticmethod
    def evaluate_configuration(hex_data: np.ndarray) -> float:
        """Evaluate a configuration and return the inverse radius"""
        outer_radius = HexagonPackingEvaluator.calculate_outer_hex_radius(hex_data)

        # Create hexagon polygons
        hexagons = []
        for i in range(len(hex_data)):
            center = (hex_data[i][0], hex_data[i][1])
            rotation = hex_data[i][2]
            hexagon = HexagonGeometry.create_unit_hexagon(center, rotation)
            hexagons.append(hexagon)

        # Compute penalties
        overlap_penalty = HexagonConstraintChecker.compute_overlap_penalty(hexagons)
        total_penalty = overlap_penalty

        # If valid configuration, return inverse of outer radius; otherwise return a very small value
        if total_penalty == 0:
            return 1.0 / outer_radius
        else:
            # Invalid configuration gets penalized heavily
            return 1e-10

class GridSearchOptimizer:
    """Optimizes using grid search and direct sampling approaches"""
    
    def __init__(self):
        self.best_score = 0
        self.best_config = None
        self.start_time = time.time()
        self.timeout = 180  # seconds

    def generate_initial_grid_configurations(self) -> List[np.ndarray]:
        """Generate high-quality initial configurations using grid-based patterns"""
        configs = []
        
        # Configuration 1: Optimized hexagonal lattice pattern (based on mathematical research)
        config1 = np.array([
            [0, 0, 0],           # center
            [0, 2.0, 0],         # top
            [0, -2.0, 0],        # bottom
            [1.732, 1.0, 0],     # top-right
            [-1.732, 1.0, 0],    # top-left
            [1.732, -1.0, 0],    # bottom-right
            [-1.732, -1.0, 0],   # bottom-left
            [3.464, 0, 0],       # far right
            [-3.464, 0, 0],      # far left
            [1.732, 3.0, 0],     # upper right corner
            [-1.732, 3.0, 0],    # upper left corner
            [1.732, -3.0, 0],    # lower right corner
            [-1.732, -3.0, 0],   # lower left corner
        ])
        configs.append(config1[:12])
        
        # Configuration 2: Compact hexagonal arrangement optimized for minimal radius
        config2 = np.array([
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
            [-1.55, -2.7, 0],    # lower left corner
        ])
        configs.append(config2[:12])
        
        # Configuration 3: Ring pattern with strategic positioning
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
            [-1.8, -2.1, 0],     # lower left corner
        ])
        configs.append(config3[:12])
        
        # Configuration 4: Mathematical optimum based on literature (approximate)
        config4 = np.array([
            [0, 0, 0],
            [0, 2.0, 0],
            [1.732, 1.0, 0], 
            [1.732, -1.0, 0],
            [0, -2.0, 0],
            [-1.732, -1.0, 0],
            [-1.732, 1.0, 0],
            [3.464, 0, 0],
            [0, 3.464, 0],
            [0, -3.464, 0],
            [-3.464, 0, 0],
            [1.732, 3.0, 0],
            [-1.732, 3.0, 0],
            [1.732, -3.0, 0],
            [-1.732, -3.0, 0]
        ])
        configs.append(config4[:12])
        
        return configs

    def create_grid_sample_points(self, center_range: tuple, spacing: float) -> List[Tuple[float, float]]:
        """Create a grid of sampling points within specified range"""
        x_min, x_max = center_range
        y_min, y_max = center_range
        
        points = []
        x = x_min
        while x <= x_max:
            y = y_min
            while y <= y_max:
                points.append((x, y))
                y += spacing
            x += spacing
        return points

    def grid_refine_initial_config(self, base_config: np.ndarray, grid_spacing: float = 0.2) -> np.ndarray:
        """Refine a configuration using grid search around the base configuration"""
        best_config = base_config.copy()
        best_score = HexagonPackingEvaluator.evaluate_configuration(best_config)
        
        # Create a more focused grid around the base configuration
        # Only sample around the outermost points
        base_centers = [(base_config[i][0], base_config[i][1]) for i in range(len(base_config))]
        max_dist = max(math.sqrt(x*x + y*y) for x, y in base_centers)
        
        # Sample points around the area where hexagons are positioned
        search_range = (-max_dist * 2, max_dist * 2)
        grid_points = self.create_grid_sample_points(search_range, grid_spacing)
        
        # Try different permutations of positions
        for i in range(len(best_config)):
            if i == 0:  # center - keep fixed or sample nearby
                continue
                
            # For each hexagon position, try several nearby locations
            original_x, original_y = best_config[i][0], best_config[i][1]
            best_local_score = best_score
            best_local_config = best_config.copy()
            
            # Test nearby points (but not too far out)
            nearby_points = [(original_x + dx, original_y + dy) 
                           for dx in [-0.5, -0.25, 0, 0.25, 0.5] 
                           for dy in [-0.5, -0.25, 0, 0.25, 0.5]]
            
            for px, py in nearby_points:
                if abs(px) > 5 or abs(py) > 5:  # Keep within reasonable bounds
                    continue
                    
                test_config = best_config.copy()
                test_config[i][0] = px
                test_config[i][1] = py
                
                score = HexagonPackingEvaluator.evaluate_configuration(test_config)
                if score > best_local_score:
                    best_local_score = score
                    best_local_config = test_config.copy()
            
            if best_local_score > best_score:
                best_score = best_local_score
                best_config = best_local_config.copy()
        
        return best_config

    def direct_optimization_refinement(self, config: np.ndarray) -> np.ndarray:
        """Use direct optimization techniques to refine the configuration"""
        # Create a more efficient optimization setup
        def objective_func(positions_flat):
            # Reshape flat array into 12x2 positions
            positions = positions_flat.reshape(-1, 2)
            
            # Create temporary config with updated positions
            temp_config = config.copy()
            temp_config[:, 0] = positions[:, 0]
            temp_config[:, 1] = positions[:, 1]
            
            # Calculate outer radius
            outer_radius = HexagonPackingEvaluator.calculate_outer_hex_radius(temp_config)
            return outer_radius
            
        def constraint_func(positions_flat):
            # Reshape flat array into 12x2 positions
            positions = positions_flat.reshape(-1, 2)
            
            # Create hexagon polygons
            hexagons = []
            for i in range(12):
                center = (positions[i][0], positions[i][1])
                rotation = config[i][2]
                hexagon = HexagonGeometry.create_unit_hexagon(center, rotation)
                hexagons.append(hexagon)
            
            penalty = HexagonConstraintChecker.compute_overlap_penalty(hexagons)
            outer_radius = HexagonPackingEvaluator.calculate_outer_hex_radius(
                np.column_stack((positions[:, 0], positions[:, 1], np.zeros(12)))
            )
            
            return penalty

        # Flatten initial positions
        initial_positions = np.column_stack((config[:, 0], config[:, 1])).flatten()
        
        # Use L-BFGS-B for optimization with bounds
        try:
            result = minimize(
                objective_func, 
                initial_positions, 
                method='L-BFGS-B',
                bounds=[(-5, 5) for _ in range(24)],
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if result.success:
                final_positions = result.x.reshape(-1, 2)
                config[:, 0] = final_positions[:, 0]
                config[:, 1] = final_positions[:, 1]
        except Exception as e:
            # If optimization fails, just return the input config
            pass
        
        return config

    def run_full_optimization(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run the complete grid-based optimization pipeline"""
        # Get multiple initial configurations
        configs = self.generate_initial_grid_configurations()
        
        # Try all configurations and select the best one as starting point
        best_initial_score = 0
        best_initial_config = None
        
        for config in configs:
            score = HexagonPackingEvaluator.evaluate_configuration(config)
            if score > best_initial_score:
                best_initial_score = score
                best_initial_config = config.copy()
        
        # Store the best configuration found so far
        self.best_score = best_initial_score
        self.best_config = best_initial_config.copy()
        
        # Stage 1: Grid refinement around the best initial configuration
        print("Stage 1: Grid refinement...")
        grid_refined = self.grid_refine_initial_config(best_initial_config, grid_spacing=0.15)
        grid_score = HexagonPackingEvaluator.evaluate_configuration(grid_refined)
        
        if grid_score > self.best_score:
            self.best_score = grid_score
            self.best_config = grid_refined.copy()
        
        # Stage 2: Direct optimization refinement
        print("Stage 2: Direct optimization...")
        optimized_config = self.direct_optimization_refinement(self.best_config)
        optimized_score = HexagonPackingEvaluator.evaluate_configuration(optimized_config)
        
        if optimized_score > self.best_score:
            self.best_score = optimized_score
            self.best_config = optimized_config.copy()
            
        # Stage 3: Additional refinement with different approach
        print("Stage 3: Additional refinement...")
        # Try a second grid refinement
        double_refined = self.grid_refine_initial_config(self.best_config, grid_spacing=0.05)
        double_score = HexagonPackingEvaluator.evaluate_configuration(double_refined)
        
        if double_score > self.best_score:
            self.best_score = double_score
            self.best_config = double_refined.copy()

        # Final evaluation
        final_outer_radius = HexagonPackingEvaluator.calculate_outer_hex_radius(self.best_config)
        outer_hex_side_length = final_outer_radius + 0.2  # Add margin

        # Return result
        outer_hex_data = np.array([0, 0, 0])  # centered at origin

        return self.best_config, outer_hex_data, outer_hex_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    optimizer = GridSearchOptimizer()
    return optimizer.run_full_optimization()

# EVOLVE-BLOCK-END