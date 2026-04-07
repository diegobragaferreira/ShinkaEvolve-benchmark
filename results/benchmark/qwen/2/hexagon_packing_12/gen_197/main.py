# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
import time
import math
from scipy.spatial.distance import cdist

class SymmetricHexPackOptimizer:
    def __init__(self):
        self.hex_radius = 1.0
        self.hex_apothem = self.hex_radius * math.sqrt(3) / 2
        self.outer_hex_apothem_ratio = math.sqrt(3) / 2
        
    def generate_symmetric_hexagon_pattern(self, center_params, rotation_params):
        """
        Generate 12 hexagons using D6 symmetry group with 4 parameters:
        - center_x, center_y: offset from origin
        - radius: distance from center for radial symmetry
        - angle_offset: rotational offset
        """
        # Base pattern using D6 symmetry
        # We place 6 hexagons in a ring and their reflections
        base_positions = []
        
        # Ring 1: 6 hexagons arranged around center
        for i in range(6):
            angle = i * math.pi / 3 + rotation_params['angle_offset']
            x = rotation_params['radius'] * math.cos(angle) + center_params['center_x']
            y = rotation_params['radius'] * math.sin(angle) + center_params['center_y']
            base_positions.append((x, y, 0))
        
        # Ring 2: 6 hexagons in mirror arrangement
        for i in range(6):
            angle = i * math.pi / 3 + rotation_params['angle_offset'] + math.pi/6
            x = rotation_params['radius'] * math.cos(angle) + center_params['center_x']
            y = rotation_params['radius'] * math.sin(angle) + center_params['center_y']
            base_positions.append((x, y, 0))
        
        # Ensure we have exactly 12 positions
        if len(base_positions) > 12:
            base_positions = base_positions[:12]
        elif len(base_positions) < 12:
            # Add additional positions to reach 12
            while len(base_positions) < 12:
                base_positions.append((0, 0, 0))
        
        return np.array(base_positions)
    
    def hexagon_vertices(self, center_x, center_y, angle_deg, side_length=1):
        """Generate vertices of a regular hexagon."""
        angle_rad = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            angle = angle_rad + i * math.pi / 3
            x = center_x + side_length * math.cos(angle)
            y = center_y + side_length * math.sin(angle)
            vertices.append((x, y))
        return vertices
    
    def check_containment(self, hexagon_vertices_list, outer_side_length):
        """Check containment using apothem relationship."""
        outer_apothem = outer_side_length * self.outer_hex_apothem_ratio
        
        for vertices in hexagon_vertices_list:
            # Check distance from center of each hexagon to origin
            for vertex in vertices:
                x, y = vertex
                distance = math.sqrt(x*x + y*y)
                if distance > outer_apothem:
                    return False
        return True
    
    def check_overlap(self, hexagon_vertices_list):
        """Check overlap using Shapely."""
        try:
            polygons = [Polygon(vertices) for vertices in hexagon_vertices_list]
            # Check pair-wise intersections
            for i in range(len(polygons)):
                for j in range(i+1, len(polygons)):
                    if polygons[i].intersects(polygons[j]):
                        return False
            return True
        except:
            return False
    
    def calculate_outer_side_length(self, hexagon_vertices_list):
        """Calculate minimal outer hexagon side length."""
        # Find all vertices
        all_vertices = []
        for vertices in hexagon_vertices_list:
            all_vertices.extend(vertices)
        
        if not all_vertices:
            return 1e6
            
        # Compute bounding circle
        all_vertices = np.array(all_vertices)
        center = np.mean(all_vertices, axis=0)
        distances = np.sqrt(np.sum((all_vertices - center)**2, axis=1))
        max_distance = np.max(distances)
        
        # Convert to hexagon side length
        # For a regular hexagon, side length = max_distance * 2 / sqrt(3)
        return max_distance * 2 / math.sqrt(3)
    
    def evaluate_fitness(self, params, outer_side_length):
        """
        Evaluate fitness with 4 symmetry parameters:
        params[0]: center_x
        params[1]: center_y  
        params[2]: radius
        params[3]: angle_offset
        """
        # Convert flat parameter array to named parameters
        center_params = {'center_x': params[0], 'center_y': params[1]}
        rotation_params = {'radius': params[2], 'angle_offset': params[3]}
        
        # Generate hexagon pattern
        hexagon_positions = self.generate_symmetric_hexagon_pattern(center_params, rotation_params)
        
        # Get vertices for all hexagons  
        hexagon_vertices_list = []
        for i in range(len(hexagon_positions)):
            x, y, angle = hexagon_positions[i]
            vertices = self.hexagon_vertices(x, y, angle)
            hexagon_vertices_list.append(vertices)
        
        # Perform constraint checking
        if not self.check_containment(hexagon_vertices_list, outer_side_length):
            # Heavy penalty for containment violations
            penalty = 1e6
        else:
            penalty = 0
            
        if not self.check_overlap(hexagon_vertices_list):
            # Heavy penalty for overlap violations  
            penalty += 1e5
            
        # Calculate actual side length for this configuration
        actual_side_length = self.calculate_outer_side_length(hexagon_vertices_list)
        
        # Objective: maximize 1/actual_side_length (minimize actual_side_length)
        # But we minimize negative log of actual_side_length plus penalties
        if actual_side_length <= 0:
            return 1e12
            
        fitness = -math.log(actual_side_length) + penalty
        
        return fitness, actual_side_length
    
    def optimize_symmetric_pattern(self):
        """Optimize the symmetric pattern parameters."""
        # Define bounds for the 4 symmetry parameters
        bounds = [(-5.0, 5.0),   # center_x
                  (-5.0, 5.0),   # center_y
                  (1.0, 4.0),    # radius
                  (0.0, 2*math.pi)] # angle_offset
        
        # Multi-stage optimization to avoid local minima
        best_result = None
        best_side_length = float('inf')
        
        # Stage 1: Coarse global optimization
        try:
            # Run multiple times with different seeds for robustness
            for seed in [42, 123, 456, 789]:
                result = differential_evolution(
                    lambda x: self.evaluate_fitness(x, 10.0)[0],  # Use large initial side length
                    bounds,
                    seed=seed,
                    maxiter=50,
                    popsize=10,
                    disp=False
                )
                
                if result.success:
                    # Evaluate final result with proper objective
                    fitness, side_length = self.evaluate_fitness(result.x, 10.0)
                    if side_length < best_side_length:
                        best_side_length = side_length
                        best_result = result
                    
        except Exception as e:
            pass
            
        if best_result is None:
            # Fallback to simple initialization
            best_result = np.array([0.0, 0.0, 2.0, 0.0])
            
        # Stage 2: Local refinement
        try:
            # Use the best result from global optimization as starting point
            refined_result = minimize(
                lambda x: self.evaluate_fitness(x, 10.0)[0],
                best_result.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 30, 'ftol': 1e-8}
            )
            
            if refined_result.success:
                best_result = refined_result
                
        except Exception as e:
            pass
            
        return best_result, best_side_length
    
    def adaptive_optimization(self):
        """Adaptive optimization that progressively tightens the outer hexagon size."""
        # Start with reasonably large outer hexagon
        outer_side_length = 5.0
        
        # Optimize using symmetric pattern
        best_result, best_side_length = self.optimize_symmetric_pattern()
        
        # Now refine by progressively tightening the outer hexagon
        target_sota = 3.9419123
        current_side_length = best_side_length
        
        # Try to find better solution with tighter bounds
        for attempt in range(5):
            # Tighten the outer hexagon size
            test_sizes = np.linspace(max(3.0, current_side_length - 0.2), 
                                   min(target_sota, current_side_length + 0.2), 10)
            
            best_in_attempt = current_side_length
            best_params_in_attempt = best_result.x.copy()
            
            for size in test_sizes:
                try:
                    fitness, side_length = self.evaluate_fitness(best_result.x, size)
                    if side_length < best_in_attempt and side_length > 0:
                        best_in_attempt = side_length
                        best_params_in_attempt = best_result.x.copy()
                except:
                    continue
                    
            current_side_length = best_in_attempt
            best_result.x = best_params_in_attempt
            
            # Early stopping if we're close to target
            if abs(current_side_length - target_sota) < 0.001:
                break
            
        return best_result, current_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    optimizer = SymmetricHexPackOptimizer()
    
    # Run adaptive optimization
    best_result, outer_hex_side_length = optimizer.adaptive_optimization()
    
    # Convert final symmetric result back to full 12-hexagon specification
    center_params = {'center_x': best_result.x[0], 'center_y': best_result.x[1]}
    rotation_params = {'radius': best_result.x[2], 'angle_offset': best_result.x[3]}
    
    # Generate the final pattern
    hexagon_positions = optimizer.generate_symmetric_hexagon_pattern(center_params, rotation_params)
    
    # Ensure we have exactly 12 hexagons
    if len(hexagon_positions) < 12:
        # Fill with default values
        while len(hexagon_positions) < 12:
            hexagon_positions = np.vstack([hexagon_positions, [0, 0, 0]])
    elif len(hexagon_positions) > 12:
        hexagon_positions = hexagon_positions[:12]
    
    # Format output
    inner_hex_data = hexagon_positions.astype(float)
    outer_hex_data = np.array([0, 0, 0])  # Centered at origin
    
    end_time = time.time()
    eval_time = end_time - start_time
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END