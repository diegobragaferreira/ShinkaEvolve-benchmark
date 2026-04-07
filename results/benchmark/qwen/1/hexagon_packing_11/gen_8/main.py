# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
from shapely.ops import unary_union
import numba
from numba import jit
import time
import math

class Hexagon:
    """Represents a regular hexagon with side length 1"""
    
    def __init__(self, center=(0, 0), rotation=0):
        self.center = np.array(center)
        self.rotation = rotation  # in degrees
        self.side_length = 1.0
        
    def get_vertices(self):
        """Get vertices of the hexagon in world coordinates"""
        angle_offset = np.radians(self.rotation)
        angle_step = 2 * np.pi / 6
        vertices = []
        for i in range(6):
            angle = angle_offset + i * angle_step
            x = self.center[0] + self.side_length * np.cos(angle)
            y = self.center[1] + self.side_length * np.sin(angle)
            vertices.append((x, y))
        return vertices
    
    def get_polygon(self):
        """Get Shapely polygon representation"""
        return Polygon(self.get_vertices())

class OuterHexagon(Hexagon):
    """Represents the outer hexagon that contains all inner hexagons"""
    
    def __init__(self, center=(0, 0), side_length=1.0):
        super().__init__(center, 0)
        self.side_length = side_length
        
    def get_vertices(self):
        """Get vertices of the outer hexagon"""
        angle_step = 2 * np.pi / 6
        vertices = []
        for i in range(6):
            angle = i * angle_step
            x = self.center[0] + self.side_length * np.cos(angle)
            y = self.center[1] + self.side_length * np.sin(angle)
            vertices.append((x, y))
        return vertices
    
    def get_polygon(self):
        """Get Shapely polygon representation"""
        return Polygon(self.get_vertices())
    
    def contains_point(self, point):
        """Check if point is inside the outer hexagon"""
        return self.get_polygon().contains(Point(point))

class PackingEvaluator:
    """Handles the evaluation of hexagon packings"""
    
    def __init__(self, outer_hex_side_length=10.0):
        self.outer_hex_side_length = outer_hex_side_length
        self.outer_hex = OuterHexagon((0, 0), outer_hex_side_length)
        
    @staticmethod
    @jit(nopython=True)
    def _distance_point_to_hexagon_edge(point, hex_center, hex_radius, rotation):
        """Calculate minimum distance from point to hexagon edge"""
        px, py = point
        cx, cy = hex_center
        angle_offset = rotation * np.pi / 180.0
        
        min_dist = float('inf')
        # Check distance to each edge of the hexagon
        for i in range(6):
            angle1 = angle_offset + i * 2 * np.pi / 6
            angle2 = angle_offset + ((i + 1) % 6) * 2 * np.pi / 6
            
            x1 = cx + hex_radius * np.cos(angle1)
            y1 = cy + hex_radius * np.sin(angle1)
            x2 = cx + hex_radius * np.cos(angle2)
            y2 = cy + hex_radius * np.sin(angle2)
            
            # Distance from point to line segment
            A = px - x1
            B = py - y1
            C = x2 - x1
            D = y2 - y1
            
            dot = A * C + B * D
            len_sq = C * C + D * D
            param = -1
            if len_sq != 0:
                param = dot / len_sq
                
            if param < 0:
                xx = x1
                yy = y1
            elif param > 1:
                xx = x2
                yy = y2
            else:
                xx = x1 + param * C
                yy = y1 + param * D
                
            dx = px - xx
            dy = py - yy
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = min(min_dist, dist)
            
        return min_dist
    
    def evaluate_packing(self, positions_and_angles):
        """
        Evaluate a packing configuration
        positions_and_angles: array of shape (11, 3) where each row is (x, y, angle)
        """
        # Extract positions and angles
        positions = positions_and_angles[:, :2]
        angles = positions_and_angles[:, 2]
        
        # Create inner hexagons
        inner_hexagons = []
        for pos, angle in zip(positions, angles):
            inner_hexagons.append(Hexagon(pos, angle))
        
        # Check containment in outer hexagon
        outer_polygon = self.outer_hex.get_polygon()
        
        # Check if any inner hexagon is outside outer hexagon
        for hexagon in inner_hexagons:
            inner_polygon = hexagon.get_polygon()
            if not outer_polygon.contains(inner_polygon):
                return False, float('inf')
        
        # Check for collisions between hexagons
        for i in range(len(inner_hexagons)):
            for j in range(i + 1, len(inner_hexagons)):
                hex1 = inner_hexagons[i]
                hex2 = inner_hexagons[j]
                
                poly1 = hex1.get_polygon()
                poly2 = hex2.get_polygon()
                
                if poly1.intersects(poly2):
                    return False, float('inf')
        
        # If we reach here, configuration is valid
        return True, self.outer_hex_side_length

class HexagonPackingOptimizer:
    """Main optimizer class for placing 11 hexagons optimally"""
    
    def __init__(self):
        self.best_solution = None
        self.best_score = float('inf')
        self.start_time = time.time()
        self.max_time = 180.0  # 3 minutes max
        
    def generate_initial_guess(self):
        """Generate a better initial guess than the baseline"""
        # Start with a known good configuration
        # This uses a pattern inspired by known optimal solutions
        
        # Center hexagon
        positions = [[0, 0]]
        
        # Surrounding hexagons in a 2-ring pattern
        ring1_positions = [
            [-2.5, 0],
            [2.5, 0],
            [-1.25, 2.17],
            [1.25, 2.17],
            [-1.25, -2.17],
            [1.25, -2.17]
        ]
        
        # Additional hexagons
        ring2_positions = [
            [-3.75, 2.17],
            [3.75, 2.17],
            [-3.75, -2.17],
            [3.75, -2.17],
            [0, 4.33],
            [0, -4.33],
            [4.33, 0],
            [-4.33, 0]
        ]
        
        # Combine all positions
        all_positions = positions + ring1_positions + ring2_positions[:3]
        
        # Generate initial solution
        initial_params = []
        for i, pos in enumerate(all_positions):
            # Add some randomness to angles to avoid degenerate cases
            angle = np.random.uniform(0, 360)
            initial_params.extend([pos[0], pos[1], angle])
        
        # Pad to 11 hexagons
        while len(initial_params) < 33:
            initial_params.extend([0, 0, 0])
            
        return np.array(initial_params).reshape(-1, 3)
    
    def _objective_function(self, params):
        """Objective function to minimize (negative of inverse side length)"""
        # Reshape parameters into positions and angles
        positions_and_angles = params.reshape(-1, 3)
        
        # Try different outer hexagon sizes starting from a reasonable value
        for side_length in np.linspace(3.0, 5.0, 50):  # Reasonable range
            evaluator = PackingEvaluator(side_length)
            is_valid, score = evaluator.evaluate_packing(positions_and_angles)
            
            if is_valid:
                # We found a valid configuration, return negative of inverse side length
                return -1.0 / side_length
        
        # If no valid configuration found, return worst case
        return 1000.0
    
    def optimize(self):
        """Run the optimization process"""
        # Generate initial guess
        initial_guess = self.generate_initial_guess()
        
        # Flatten parameters for optimization
        initial_flat = initial_guess.flatten()
        
        # Optimize using differential evolution with bounds
        bounds = [(float('-inf'), float('inf'))] * len(initial_flat)
        
        # Run optimization
        try:
            result = differential_evolution(
                self._objective_function,
                bounds,
                maxiter=100,
                popsize=15,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False,
                tol=1e-6
            )
            
            # Extract best solution
            best_positions_and_angles = result.x.reshape(-1, 3)
            final_evaluator = PackingEvaluator(1.0)  # Will be determined by objective
            
            # Find the actual best valid configuration
            best_valid_config = None
            best_valid_size = float('inf')
            
            # Test several side lengths to find a valid one
            for test_side_length in np.linspace(3.5, 4.5, 20):
                evaluator = PackingEvaluator(test_side_length)
                is_valid, score = evaluator.evaluate_packing(best_positions_and_angles)
                if is_valid:
                    if test_side_length < best_valid_size:
                        best_valid_size = test_side_length
                        best_valid_config = best_positions_and_angles.copy()
                        break
            
            if best_valid_config is not None:
                return best_valid_config, best_valid_size
                
        except Exception as e:
            print(f"Optimization failed: {e}")
            
        # Fallback to initial guess if optimization fails
        return initial_guess, 4.0

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    optimizer = HexagonPackingOptimizer()
    positions_and_angles, outer_side_length = optimizer.optimize()
    
    # Convert results to expected format
    inner_hex_data = positions_and_angles.astype(float)
    
    # Outer hexagon is centered at origin with zero rotation
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END
