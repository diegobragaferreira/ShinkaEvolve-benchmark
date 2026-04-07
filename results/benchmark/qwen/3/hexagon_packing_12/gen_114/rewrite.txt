# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
from scipy.optimize import minimize
import math
import random
from itertools import combinations
from typing import Tuple, List, Optional
import time
from scipy.spatial.distance import cdist

class DirectHexagonPackingOptimizer:
    """Direct geometric optimization approach for hexagon packing"""
    
    def __init__(self):
        self.best_score = 0
        self.best_config = None
        self.start_time = time.time()
        self.timeout = 180  # seconds
        
    @staticmethod
    def create_unit_hexagon(center: Tuple[float, float], rotation: float = 0) -> Polygon:
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
            hexagon = DirectHexagonPackingOptimizer.create_unit_hexagon(center, rotation)
            all_vertices.extend(list(hexagon.exterior.coords))
        return all_vertices
    
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
    def calculate_outer_hex_radius(hex_data: np.ndarray) -> float:
        """Calculate minimum outer hexagon radius needed to contain all inner hexagons"""
        all_vertices = DirectHexagonPackingOptimizer.get_all_vertices(hex_data)
        max_distance = 0
        for vertex in all_vertices:
            distance = math.sqrt(vertex[0]**2 + vertex[1]**2)
            max_distance = max(max_distance, distance)
        return max_distance + 0.1
    
    @staticmethod
    def compute_feasibility_score(hex_data: np.ndarray) -> float:
        """Compute a feasibility score based on geometric constraints"""
        # Create hexagon polygons
        hexagons = []
        for i in range(len(hex_data)):
            center = (hex_data[i][0], hex_data[i][1])
            rotation = hex_data[i][2]
            hexagon = DirectHexagonPackingOptimizer.create_unit_hexagon(center, rotation)
            hexagons.append(hexagon)
        
        # Check overlap penalty
        overlap_penalty = 0
        n = len(hexagons)
        for i in range(n):
            for j in range(i+1, n):
                if DirectHexagonPackingOptimizer.check_overlap(hexagons[i], hexagons[j]):
                    overlap_penalty += 1000
                    
        # Check containment penalty using bounding circle
        containment_penalty = 0
        outer_radius = DirectHexagonPackingOptimizer.calculate_outer_hex_radius(hex_data)
        for hexagon in hexagons:
            for point in list(hexagon.exterior.coords):
                distance = math.sqrt(point[0]**2 + point[1]**2)
                if distance > outer_radius - 0.05:  # Small buffer
                    containment_penalty += 1000
        
        total_penalty = overlap_penalty + containment_penalty
        
        # If valid configuration, return inverse of outer radius; otherwise return a very small value
        if total_penalty == 0:
            return 1.0 / outer_radius
        else:
            return 1e-10
    
    @staticmethod
    def generate_initial_configurations() -> List[np.ndarray]:
        """Generate high-quality initial configurations using geometric intuition"""
        configs = []
        
        # Configuration 1: Optimized hexagonal arrangement
        # Based on mathematical packing theory and known optimal arrangements
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
        
        # Configuration 2: More compact arrangement
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
        
        # Configuration 3: Ring pattern with strategic spacing
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
        
        # Configuration 4: Optimized radial arrangement (based on packing literature)
        config4 = np.array([
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
            [-1.65, -2.85, 0],   # lower left corner
        ])
        configs.append(config4[:12])
        
        return configs
    
    @staticmethod
    def build_geometry_constraints(hex_data: np.ndarray) -> tuple:
        """Build geometric constraint matrices for efficient computation"""
        # Extract positions (ignoring rotation for constraint building)
        positions = hex_data[:, :2]
        
        # Compute pairwise distances between centers
        dist_matrix = cdist(positions, positions)
        
        # Define minimum distance constraint (centers should be at least 2 units apart for non-overlapping)
        min_dist = 2.0
        # Define maximum distance constraint (all hexagons should fit in reasonable region)
        max_dist = 10.0
        
        return dist_matrix, min_dist, max_dist
    
    @staticmethod
    def optimize_positions_directly(initial_config: np.ndarray, max_iterations: int = 100) -> np.ndarray:
        """Direct geometric optimization using constrained optimization"""
        # Convert to flattened format for scipy
        initial_positions = np.column_stack((initial_config[:, 0], initial_config[:, 1])).flatten()
        
        def objective_func(params):
            # Reconstruct positions from flattened array
            positions = params.reshape(-1, 2)
            temp_data = initial_config.copy()
            temp_data[:, 0] = positions[:, 0]
            temp_data[:, 1] = positions[:, 1]
            
            # Calculate outer radius
            outer_radius = DirectHexagonPackingOptimizer.calculate_outer_hex_radius(temp_data)
            return outer_radius
        
        def constraint_func(params):
            # Reconstruct positions from flattened array
            positions = params.reshape(-1, 2)
            temp_data = initial_config.copy()
            temp_data[:, 0] = positions[:, 0]
            temp_data[:, 1] = positions[:, 1]
            
            # Create hexagon polygons
            hexagons = []
            for i in range(12):
                center = (positions[i][0], positions[i][1])
                rotation = initial_config[i][2]
                hexagon = DirectHexagonPackingOptimizer.create_unit_hexagon(center, rotation)
                hexagons.append(hexagon)
            
            # Compute overlap penalty
            penalty = 0
            n = len(hexagons)
            for i in range(n):
                for j in range(i+1, n):
                    if DirectHexagonPackingOptimizer.check_overlap(hexagons[i], hexagons[j]):
                        penalty += 1000
                        
            return penalty
        
        # Optimization with bounds and constraints
        bounds = [(-5, 5) for _ in range(24)]
        
        try:
            result = minimize(objective_func, initial_positions, method='L-BFGS-B',
                             bounds=bounds,
                             constraints={'type': 'ineq', 'fun': constraint_func},
                             options={'maxiter': max_iterations, 'ftol': 1e-6})
            
            if result.success:
                final_positions = result.x.reshape(-1, 2)
                optimized_config = initial_config.copy()
                optimized_config[:, 0] = final_positions[:, 0]
                optimized_config[:, 1] = final_positions[:, 1]
                return optimized_config
        except Exception as e:
            # If optimization fails, return original
            print(f"Optimization error: {e}")
            pass
            
        return initial_config
    
    def run_full_optimization(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run the direct geometric optimization pipeline"""
        # Get multiple symmetric configurations
        configs = self.generate_initial_configurations()
        
        # Try multiple configurations and find the best starting point
        best_initial_score = 0
        best_initial_config = None
        
        for config in configs:
            score = self.compute_feasibility_score(config)
            if score > best_initial_score:
                best_initial_score = score
                best_initial_config = config.copy()
        
        # Store the best configuration found so far
        self.best_score = best_initial_score
        self.best_config = best_initial_config.copy()
        
        # Direct geometric optimization refinement
        print("Performing direct geometric optimization...")
        refined_config = self.optimize_positions_directly(best_initial_config, max_iterations=100)
        
        # Additional fine-tuning by adjusting rotation angles
        # Keep positions fixed, optimize rotations to minimize outer radius
        def rotate_objective(angles):
            temp_config = refined_config.copy()
            for i, angle in enumerate(angles):
                temp_config[i][2] = angle
            return self.calculate_outer_hex_radius(temp_config)
        
        # Try optimizing rotations for better packing
        initial_angles = refined_config[:, 2].tolist()
        try:
            # Simple rotation optimization using bounded minimization
            rotation_result = minimize(rotate_objective, initial_angles, 
                                     bounds=[(-180, 180) for _ in range(12)],
                                     method='L-BFGS-B', 
                                     options={'maxiter': 50})
            
            if rotation_result.success:
                final_angles = rotation_result.x
                refined_config[:, 2] = final_angles
        except:
            pass  # Keep original rotations if optimization fails
        
        # Final comprehensive optimization
        final_config = self.optimize_positions_directly(refined_config, max_iterations=150)
        
        # Final evaluation
        final_score = self.compute_feasibility_score(final_config)
        final_outer_radius = self.calculate_outer_hex_radius(final_config)
        outer_hex_side_length = final_outer_radius + 0.2  # Add margin
        
        # Return result
        outer_hex_data = np.array([0, 0, 0])  # centered at origin
        
        return final_config, outer_hex_data, outer_hex_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    optimizer = DirectHexagonPackingOptimizer()
    return optimizer.run_full_optimization()

# EVOLVE-BLOCK-END