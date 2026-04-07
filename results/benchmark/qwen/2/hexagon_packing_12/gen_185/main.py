# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
from numba import njit
import time
import random
from collections import defaultdict

class HexagonGeometry:
    """Handles all geometric operations for hexagon calculations"""
    
    @staticmethod
    @njit
    def generate_hexagon_vertices(x, y, angle_degrees, side_length=1):
        """Generate vertices of a regular hexagon given center, rotation, and side length"""
        angle_rad = np.radians(angle_degrees)
        vertices = np.empty((6, 2))
        for i in range(6):
            theta = angle_rad + i * np.pi / 3
            vertices[i, 0] = x + side_length * np.cos(theta)
            vertices[i, 1] = y + side_length * np.sin(theta)
        return vertices

    @staticmethod
    @njit
    def check_containment_inner_to_outer(inner_x, inner_y, inner_angle, outer_x, outer_y, outer_angle, outer_side_length):
        """Check if inner hexagon is fully contained within outer hexagon"""
        inner_vertices = HexagonGeometry.generate_hexagon_vertices(inner_x, inner_y, inner_angle, 1.0)
        
        # Create outer hexagon vertices
        outer_vertices = HexagonGeometry.generate_hexagon_vertices(outer_x, outer_y, outer_angle, outer_side_length)
        outer_polygon = Polygon(outer_vertices)
        
        # Check if all vertices of inner hexagon are inside outer hexagon
        for i in range(6):
            if not outer_polygon.contains(Point(inner_vertices[i, 0], inner_vertices[i, 1])):
                return False
        
        return True

    @staticmethod
    @njit
    def check_overlap_hexagons(x1, y1, angle1, x2, y2, angle2):
        """Check if two hexagons overlap using vertex-based collision detection"""
        vertices1 = HexagonGeometry.generate_hexagon_vertices(x1, y1, angle1, 1.0)
        vertices2 = HexagonGeometry.generate_hexagon_vertices(x2, y2, angle2, 1.0)
        
        # Simple bounding box check first
        min1 = np.min(vertices1, axis=0)
        max1 = np.max(vertices1, axis=0)
        min2 = np.min(vertices2, axis=0)
        max2 = np.max(vertices2, axis=0)
        
        if max1[0] < min2[0] or max2[0] < min1[0] or max1[1] < min2[1] or max2[1] < min1[1]:
            return False
        
        # Create polygons and check intersection
        poly1 = Polygon(vertices1)
        poly2 = Polygon(vertices2)
        
        # If intersection exists, they overlap
        return poly1.intersects(poly2)

class HexagonConfiguration:
    """Represents and manages hexagon configurations"""
    
    def __init__(self, hexagons_data, outer_side_length):
        self.hexagons = hexagons_data  # List of (x, y, angle) tuples
        self.outer_side_length = outer_side_length
    
    @classmethod
    def from_flat_params(cls, params):
        """Create configuration from flat parameter array"""
        hexagons_data = []
        idx = 0
        for i in range(12):
            hexagons_data.append([params[idx], params[idx+1], params[idx+2]])
            idx += 3
        outer_side_length = params[-1]
        return cls(hexagons_data, outer_side_length)
    
    def to_flat_params(self):
        """Convert configuration to flat parameter array"""
        params = []
        for hex_data in self.hexagons:
            params.extend(hex_data)
        params.append(self.outer_side_length)
        return params

class PenaltyCalculator:
    """Calculates penalties for constraint violations"""
    
    @staticmethod
    def calculate_penalty(hexagons_data, outer_side_length, penalty_weights=(1e7, 1e6)):
        """Calculate penalty based on constraint violations with adaptive weights"""
        penalty = 0.0
        n = len(hexagons_data)
        
        # Check containment penalties (higher priority)
        containment_penalty_weight = penalty_weights[0]
        overlap_penalty_weight = penalty_weights[1]
        
        # Check containment for all hexagons
        for i in range(n):
            if not HexagonGeometry.check_containment_inner_to_outer(
                hexagons_data[i][0], hexagons_data[i][1], hexagons_data[i][2],
                0, 0, 0, outer_side_length
            ):
                penalty += containment_penalty_weight
        
        # Check overlap penalties (lower priority)
        for i in range(n):
            for j in range(i+1, n):
                if HexagonGeometry.check_overlap_hexagons(
                    hexagons_data[i][0], hexagons_data[i][1], hexagons_data[i][2],
                    hexagons_data[j][0], hexagons_data[j][1], hexagons_data[j][2]
                ):
                    penalty += overlap_penalty_weight
                    
        return penalty

class OptimizationEngine:
    """Handles the optimization process"""
    
    @staticmethod
    def evaluate_configuration(params):
        """Evaluate the fitness of a given configuration"""
        config = HexagonConfiguration.from_flat_params(params)
        
        # Calculate penalty
        penalty = PenaltyCalculator.calculate_penalty(
            config.hexagons, 
            config.outer_side_length
        )
        
        # Inverse side length (negative because we minimize)
        # We want to maximize inverse side length, so minimize negative value
        objective_value = -1.0 / config.outer_side_length + penalty
        
        return objective_value
    
    @staticmethod
    def generate_symmetric_initial_population(pop_size=30):
        """Generate highly symmetric configurations guided by mathematical insights"""
        population = []
        
        # Base pattern based on hexagonal lattice geometry with mathematical considerations
        base_pattern = [
            [0.0, 0.0, 0.0],          # Center (primary)
            [0.0, 3.0, 0.0],          # Top (secondary)
            [0.0, -3.0, 0.0],         # Bottom (secondary)
            [2.6, 1.5, 0.0],          # Top Right (tertiary)
            [-2.6, 1.5, 0.0],         # Top Left (tertiary)
            [2.6, -1.5, 0.0],         # Bottom Right (tertiary)
            [-2.6, -1.5, 0.0],        # Bottom Left (tertiary)
            [3.5, 0.0, 0.0],          # Far Right (quaternary)
            [-3.5, 0.0, 0.0],         # Far Left (quaternary)
            [1.75, 3.03, 0.0],        # Upper Middle Right (quinary)
            [-1.75, 3.03, 0.0],       # Upper Middle Left (quinary)
            [1.75, -3.03, 0.0],       # Lower Middle Right (quinary)
            [-1.75, -3.03, 0.0],      # Lower Middle Left (quinary)
        ]
        
        # Generate variations of the symmetric pattern
        for _ in range(pop_size):
            config = []
            # Add perturbation to base pattern with controlled variation
            for i, (x, y, angle) in enumerate(base_pattern[:-1]):  # Skip last element (outer side)
                # Add small random perturbations with reduced variance
                pert_x = x + random.uniform(-0.3, 0.3)
                pert_y = y + random.uniform(-0.3, 0.3)
                pert_angle = angle + random.uniform(-10, 10)  # Random rotation
                config.extend([pert_x, pert_y, pert_angle])
            
            # Add outer side length with reasonable range
            config.append(6.5 + random.uniform(0, 2.5))
            population.append(config)
        
        return population
    
    @staticmethod
    def local_search(best_config):
        """Apply advanced local search with multiple refinement techniques"""
        best_value = OptimizationEngine.evaluate_configuration(best_config)
        improved = True
        iterations = 0
        
        # Momentum-based search with adaptive step size
        momentum = [0.0] * len(best_config)
        step_sizes = [0.1, 0.05, 0.025]  # Multiple step sizes for adaptive refinement
        
        while improved and iterations < 150:  # Increased iteration limit
            improved = False
            iterations += 1
            
            # Try different step sizes for parameter tuning
            for step_idx, step_size in enumerate(step_sizes):
                for i in range(len(best_config)):
                    original_val = best_config[i]
                    
                    # Try multiple step directions for each parameter
                    for delta in [-step_size, step_size]:
                        # Apply momentum for smoother movement
                        if i < len(best_config) - 1:  # Not outer side length
                            new_val = original_val + delta + momentum[i] * 0.1
                            if new_val < -10 or new_val > 10:
                                continue
                        else:  # outer side length - constrain positive
                            new_val = original_val + delta + momentum[i] * 0.1
                            if new_val <= 0.1:
                                continue
                        
                        best_config[i] = new_val
                        new_value = OptimizationEngine.evaluate_configuration(best_config)
                        
                        if new_value < best_value:
                            best_value = new_value
                            improved = True
                            momentum[i] = delta  # Update momentum
                        else:
                            best_config[i] = original_val  # Revert if no improvement
                            
            # Occasionally perform a more thorough search
            if iterations % 20 == 0:
                # Perturb all parameters slightly to escape local minima
                for i in range(len(best_config)):
                    if i < len(best_config) - 1:  # Not outer side length
                        best_config[i] += random.uniform(-0.05, 0.05)
                    else:  # outer side length
                        best_config[i] += random.uniform(-0.05, 0.05)
                        if best_config[i] <= 0.1:
                            best_config[i] = 0.15
                        
        return best_config

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Define bounds for optimization
    # Each hexagon has 3 parameters: x, y, angle; plus outer side length
    # Bounds: x, y from -10 to 10, angle from 0 to 360, outer side length from 1 to 20
    bounds = []
    for _ in range(12):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])
    bounds.append((1, 20))  # Outer side length bound
    
    # Generate enhanced initial population
    initial_pop = OptimizationEngine.generate_symmetric_initial_population(25)
    
    # Set up differential evolution with optimized parameters for better convergence
    def objective_func(params):
        return OptimizationEngine.evaluate_configuration(params)
    
    # Run optimization with optimized settings
    try:
        result = differential_evolution(
            objective_func,
            bounds,
            seed=42,
            maxiter=150,           # More iterations for better convergence
            popsize=25,            # Larger population for better exploration
            mutation=(0.8, 1),     # Higher mutation rate for better exploration
            recombination=0.9,     # Even higher recombination rate
            tol=1e-6,
            workers=1,
            init=initial_pop
        )
        
        # Best result
        best_params = result.x
        best_score = result.fun
        
        # Local search refinement with enhanced parameters
        best_params = OptimizationEngine.local_search(best_params)
        refined_score = OptimizationEngine.evaluate_configuration(best_params)
        
        # Extract configuration
        config = HexagonConfiguration.from_flat_params(best_params)
        inner_hex_data = np.array(config.hexagons)
        outer_hex_data = np.array([0, 0, 0])
        outer_side_length = config.outer_side_length
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback to previous solution
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
            [0, -4, 0],
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_side_length = 8.0
    
    # Ensure all computations completed within time limit
    elapsed_time = time.time() - start_time
    if elapsed_time > 175:  # Leave buffer
        print("Warning: Time limit approaching")
    
    return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END