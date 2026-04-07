# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
import math
from joblib import Parallel, delayed
import multiprocessing
import time

class HexagonGeometry:
    """Handles all geometric operations related to hexagons"""
    
    @staticmethod
    def create_vertices(center_x, center_y, angle_deg, side_length=1):
        """Create vertices of a regular hexagon given center, rotation, and side length"""
        angle_rad = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            angle = angle_rad + i * math.pi / 3
            x = center_x + side_length * math.cos(angle)
            y = center_y + side_length * math.sin(angle)
            vertices.append((x, y))
        return vertices
    
    @staticmethod
    def calculate_bounding_box(vertices):
        """Calculate tight bounding box for a set of vertices"""
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        return min(xs), max(xs), min(ys), max(ys)
    
    @staticmethod
    def calculate_circumradius(vertices):
        """Calculate maximum distance from center to any vertex"""
        center_x = sum(v[0] for v in vertices) / len(vertices)
        center_y = sum(v[1] for v in vertices) / len(vertices)
        max_dist = 0
        for x, y in vertices:
            dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = max(max_dist, dist)
        return max_dist

class ConstraintValidator:
    """Validates all constraints for hexagon arrangements"""
    
    @staticmethod
    def check_containment(hex_vertices, outer_hex_vertices, buffer=1e-6):
        """Check if all vertices of inner hexagon are within outer hexagon"""
        outer_polygon = Polygon(outer_hex_vertices)
        buffered_outer = outer_polygon.buffer(buffer)
        for vertex in hex_vertices:
            point = Point(vertex[0], vertex[1])
            if not buffered_outer.contains(point):
                return False
        return True
    
    @staticmethod
    def check_overlap(hex1_vertices, hex2_vertices, buffer=1e-6):
        """Check if two hexagons overlap using Shapely with buffer"""
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        buffered_poly1 = poly1.buffer(buffer)
        buffered_poly2 = poly2.buffer(buffer)
        return buffered_poly1.intersects(buffered_poly2)

class ObjectiveCalculator:
    """Calculates objective values and outer hexagon size"""
    
    @staticmethod
    def calculate_outer_hex_side_length(inner_hex_data):
        """Calculate minimum outer hexagon side length that contains all inner hexagons"""
        if len(inner_hex_data) == 0:
            return 1000

        # Get all vertices of inner hexagons
        all_vertices = []
        for center_x, center_y, angle in inner_hex_data:
            vertices = HexagonGeometry.create_vertices(center_x, center_y, angle)
            all_vertices.extend(vertices)

        if not all_vertices:
            return 1000

        # Calculate tight bounding box
        min_x, max_x, min_y, max_y = HexagonGeometry.calculate_bounding_box(all_vertices)
        
        # Calculate diagonal of bounding box
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        diagonal = math.sqrt(bbox_width**2 + bbox_height**2)
        
        # Calculate side length based on hexagon geometry
        side_length = diagonal / math.sqrt(3)
        side_length *= 1.1  # Add margin
        
        return side_length

class Initializer:
    """Generates initial configurations for optimization"""
    
    @staticmethod
    def generate_grid_configurations():
        """Generate multiple grid-based configurations"""
        configs = []
        
        # Simple grid configuration
        config1 = np.array([
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
            [3.75, -2.17, 0],  # far bottom-right
        ])
        configs.append(config1)
        
        # Spiral pattern configuration
        config2 = np.array([
            [0, 0, 0],  # center
            [-1.5, 0, 0],  # left
            [1.5, 0, 0],  # right
            [-0.75, 1.30, 0],  # top-left
            [0.75, 1.30, 0],  # top-right
            [-0.75, -1.30, 0],  # bottom-left
            [0.75, -1.30, 0],  # bottom-right
            [-2.25, 1.30, 0],  # far top-left
            [2.25, 1.30, 0],  # far top-right
            [-2.25, -1.30, 0],  # far bottom-left
            [2.25, -1.30, 0],  # far bottom-right
        ])
        configs.append(config2)
        
        return configs
    
    @staticmethod
    def generate_spiral_configuration():
        """Generate a spiral-based initial configuration"""
        centers = [
            [0, 0, 0],      # center
            [-2.0, 0, 0],   # left
            [2.0, 0, 0],    # right
            [0, 2.0, 0],    # top
            [0, -2.0, 0],   # bottom
            [1.73, 1.0, 0], # top-right
            [-1.73, 1.0, 0], # top-left
            [1.73, -1.0, 0], # bottom-right
            [-1.73, -1.0, 0], # bottom-left
            [3.46, 0, 0],   # far right
            [0, 3.46, 0],   # far top
        ]
        return np.array(centers[:11])

class HexagonPackingOptimizer:
    """Main optimizer orchestrating the hexagon packing process"""
    
    def __init__(self):
        self.geometry = HexagonGeometry()
        self.validator = ConstraintValidator()
        self.objective = ObjectiveCalculator()
        self.initializer = Initializer()
        
    def evaluate_individual(self, individual):
        """Evaluate fitness of a solution - maximize 1/outer_hex_side_length"""
        # Convert individual to hexagon data
        hex_data = np.array(individual).reshape(-1, 3)
        
        # Create outer hexagon vertices (assuming centered at origin)
        outer_side_length = self.objective.calculate_outer_hex_side_length(hex_data)
        
        # Check constraints
        try:
            # Check containment for all inner hexagons
            outer_hex_vertices = self.geometry.create_vertices(0, 0, 0, outer_side_length)
            
            # Check if all hexagons are contained
            total_penalty = 0
            
            # Check containment
            for center_x, center_y, angle in hex_data:
                inner_hex_vertices = self.geometry.create_vertices(center_x, center_y, angle)
                if not self.validator.check_containment(inner_hex_vertices, outer_hex_vertices):
                    total_penalty += 10000  # Large penalty for containment violation
            
            # Check overlaps
            for i in range(len(hex_data)):
                for j in range(i+1, len(hex_data)):
                    center_x1, center_y1, angle1 = hex_data[i]
                    center_x2, center_y2, angle2 = hex_data[j]
                    
                    hex1_vertices = self.geometry.create_vertices(center_x1, center_y1, angle1)
                    hex2_vertices = self.geometry.create_vertices(center_x2, center_y2, angle2)
                    
                    if self.validator.check_overlap(hex1_vertices, hex2_vertices):
                        total_penalty += 10000  # Large penalty for overlap
            
            # Return fitness (inverse of outer hex side length + penalties)
            if total_penalty > 0:
                return 1.0 / outer_side_length - total_penalty
                
            return 1.0 / outer_side_length
            
        except Exception:
            return -10000  # Very poor fitness for invalid solutions
    
    def optimize_single_configuration(self, initial_guess):
        """Optimize a single configuration using differential evolution"""
        def objective(params):
            return -self.evaluate_individual(params)  # Negative because we want to maximize
        
        # Set bounds for each parameter (x, y, angle for 11 hexagons)
        bounds = []
        for _ in range(11):
            bounds.extend([(-10, 10), (-10, 10), (0, 360)])
        
        try:
            result = differential_evolution(objective, bounds, seed=42, maxiter=50, popsize=10, disp=False)
            return result.x, -result.fun
        except:
            # Fallback to initial guess if optimization fails
            return initial_guess, self.evaluate_individual(initial_guess)
    
    def run_multiple_attempts(self):
        """Run multiple optimization attempts with different initial configurations"""
        # Generate multiple initial configurations
        initial_configs = self.initializer.generate_grid_configurations()
        spiral_config = self.initializer.generate_spiral_configuration()
        initial_configs.append(spiral_config)
        
        # Add random configurations
        random.seed(42)
        for _ in range(4):
            random_config = []
            for _ in range(11):
                x = random.uniform(-5, 5)
                y = random.uniform(-5, 5)
                angle = random.uniform(0, 360)
                random_config.extend([x, y, angle])
            initial_configs.append(np.array(random_config))
        
        best_fitness = -float('inf')
        best_config = None
        best_side_length = float('inf')
        
        # Try multiple configurations with local optimization
        for i, config in enumerate(initial_configs):
            try:
                # Perform optimization on this configuration
                optimized_params, fitness = self.optimize_single_configuration(config)
                
                # Evaluate final result
                final_fitness = self.evaluate_individual(optimized_params)
                
                if final_fitness > best_fitness:
                    best_fitness = final_fitness
                    best_config = optimized_params
                    best_side_length = self.objective.calculate_outer_hex_side_length(
                        np.array(optimized_params).reshape(-1, 3)
                    )
                    
            except Exception:
                continue
        
        return best_config, best_side_length

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Initialize optimizer
    optimizer = HexagonPackingOptimizer()
    
    # Run optimization
    best_config, best_side_length = optimizer.run_multiple_attempts()
    
    if best_config is not None:
        # Convert best result to required format
        best_hex_data = np.array(best_config).reshape(-1, 3)
        outer_hex_data = np.array([0, 0, 0])  # centered at origin
        return best_hex_data, outer_hex_data, best_side_length
    else:
        # Fallback to simple grid if optimization fails
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
            [3.75, -2.17, 0],  # far bottom-right
        ])
        outer_hex_data = np.array([0, 0, 0])  # centered at origin
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
