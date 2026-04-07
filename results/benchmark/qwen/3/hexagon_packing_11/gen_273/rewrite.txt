# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Constants
UNIT_HEX_SIDE = 1.0
UNIT_HEX_HEIGHT = math.sqrt(3) / 2.0 * UNIT_HEX_SIDE
UNIT_HEX_APOGEE = UNIT_HEX_SIDE  # Distance from center to vertex for unit hexagon

class HexagonSAT:
    """
    Custom SAT-based implementation for fast hexagon collision detection.
    This avoids expensive Shapely operations while maintaining numerical stability.
    """
    
    @staticmethod
    def get_hexagon_axes(center_x, center_y, side_length, rotation_degrees):
        """Get axes for SAT test of a regular hexagon."""
        angle_rad = math.radians(rotation_degrees)
        axes = []
        for i in range(6):
            angle = angle_rad + i * math.pi / 3
            # Normal vectors to edges
            norm_x = math.cos(angle + math.pi / 2)
            norm_y = math.sin(angle + math.pi / 2)
            axes.append((norm_x, norm_y))
        return axes
    
    @staticmethod
    def project_hexagon_onto_axis(center_x, center_y, side_length, rotation_degrees, axis_x, axis_y):
        """Project a hexagon onto an axis and return min/max projection."""
        # Get vertices
        vertices = []
        angle_offset = math.radians(rotation_degrees)
        for i in range(6):
            angle = angle_offset + i * math.pi / 3
            x = center_x + side_length * math.cos(angle)
            y = center_y + side_length * math.sin(angle)
            vertices.append((x, y))
        
        # Project all vertices onto axis
        projections = []
        for x, y in vertices:
            proj = x * axis_x + y * axis_y
            projections.append(proj)
        
        return min(projections), max(projections)
    
    @staticmethod
    def sat_collision_detect(hex1_center_x, hex1_center_y, hex1_side, hex1_rot,
                           hex2_center_x, hex2_center_y, hex2_side, hex2_rot):
        """Fast SAT-based collision detection between two hexagons."""
        # Get axes for both hexagons
        axes1 = HexagonSAT.get_hexagon_axes(hex1_center_x, hex1_center_y, hex1_side, hex1_rot)
        axes2 = HexagonSAT.get_hexagon_axes(hex2_center_x, hex2_center_y, hex2_side, hex2_rot)
        
        # Test all axes
        all_axes = axes1 + axes2
        
        for axis_x, axis_y in all_axes:
            min1, max1 = HexagonSAT.project_hexagon_onto_axis(hex1_center_x, hex1_center_y, hex1_side, hex1_rot, axis_x, axis_y)
            min2, max2 = HexagonSAT.project_hexagon_onto_axis(hex2_center_x, hex2_center_y, hex2_side, hex2_rot, axis_x, axis_y)
            
            # Check for separation
            if max1 < min2 or max2 < min1:
                return False  # Separating axis found, no collision
        
        return True  # No separating axis found, collision detected

class HexagonPackingOptimizer:
    """
    Multi-stage optimizer for hexagon packing problem.
    Combines global search with local refinement and geometric insights.
    """
    
    def __init__(self):
        self.best_fitness = 0.0
        self.best_solution = None
        self.best_radius = float('inf')
    
    def compute_outer_hexagon_radius(self, hex_data) -> float:
        """Compute minimum outer hexagon radius that contains all inner hexagons."""
        # Find maximum distance from origin to any vertex
        max_dist = 0.0
        
        for hex_params in hex_data:
            center_x, center_y, rotation = hex_params
            # Get vertices of this hexagon
            angle_offset = math.radians(rotation)
            for i in range(6):
                angle = angle_offset + i * math.pi / 3
                x = center_x + UNIT_HEX_APOGEE * math.cos(angle)
                y = center_y + UNIT_HEX_APOGEE * math.sin(angle)
                dist = math.sqrt(x*x + y*y)
                max_dist = max(max_dist, dist)
        
        # Add small safety margin
        return max_dist + 0.01
    
    def check_containment(self, hex_data, outer_radius) -> bool:
        """Check if all hexagons are contained within outer hexagon."""
        # Outer hexagon vertices (centered at origin)
        outer_vertices = []
        for i in range(6):
            angle = i * math.pi / 3
            x = outer_radius * math.cos(angle)
            y = outer_radius * math.sin(angle)
            outer_vertices.append((x, y))
        
        # Check if each inner hexagon vertex is inside outer hexagon
        for hex_params in hex_data:
            center_x, center_y, rotation = hex_params
            
            # Get all hexagon vertices
            angle_offset = math.radians(rotation)
            for i in range(6):
                angle = angle_offset + i * math.pi / 3
                x = center_x + UNIT_HEX_APOGEE * math.cos(angle)
                y = center_y + UNIT_HEX_APOGEE * math.sin(angle)
                
                # Point-in-polygon test using ray casting
                if not self.point_in_hexagon(x, y, outer_vertices):
                    return False
        
        return True
    
    def point_in_hexagon(self, px, py, hex_vertices) -> bool:
        """Check if point is inside hexagon using ray casting."""
        n = len(hex_vertices)
        inside = False
        
        p1x, p1y = hex_vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = hex_vertices[i % n]
            if py > min(p1y, p2y):
                if py <= max(p1y, p2y):
                    if px <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or px <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside
    
    def check_all_overlaps(self, hex_data) -> bool:
        """Check all pairs for overlaps using SAT."""
        n_hex = len(hex_data)
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                hex1 = hex_data[i]
                hex2 = hex_data[j]
                
                if HexagonSAT.sat_collision_detect(
                    hex1[0], hex1[1], UNIT_HEX_SIDE, hex1[2],
                    hex2[0], hex2[1], UNIT_HEX_SIDE, hex2[2]
                ):
                    return False
        return True
    
    def evaluate_fitness(self, hex_data) -> float:
        """Evaluate fitness of a hexagon configuration."""
        try:
            # Check if all hexagons are properly contained
            outer_radius = self.compute_outer_hexagon_radius(hex_data)
            
            # Check containment first
            if not self.check_containment(hex_data, outer_radius):
                return 0.0  # Invalid - not contained
            
            # Check overlaps
            if not self.check_all_overlaps(hex_data):
                return 0.0  # Invalid - overlaps exist
            
            # Fitness is inverse of radius
            return 1.0 / outer_radius
            
        except Exception as e:
            return 0.0  # Invalid solution
    
    def generate_initial_lattice_arrangement(self) -> np.ndarray:
        """Generate initial arrangement based on hexagonal lattice for 11 hexagons."""
        # Create arrangement in layers - similar to honeycomb structure
        arrangement = []
        
        # Center hexagon
        arrangement.append([0.0, 0.0, 0.0])
        
        # First ring: 6 hexagons at distance sqrt(3) from center
        ring1_distance = 2 * UNIT_HEX_HEIGHT  # Distance between hexagon centers
        for i in range(6):
            angle = i * math.pi / 3
            x = ring1_distance * math.cos(angle)
            y = ring1_distance * math.sin(angle)
            arrangement.append([x, y, 0.0])
        
        # Second ring: 4 hexagons (not at perfect angles to avoid symmetry)
        ring2_distance = 3 * UNIT_HEX_HEIGHT
        ring2_angles = [math.pi/6, 3*math.pi/6, 5*math.pi/6, 7*math.pi/6]  # Skewed positions
        for i, angle in enumerate(ring2_angles):
            x = ring2_distance * math.cos(angle)
            y = ring2_distance * math.sin(angle)
            arrangement.append([x, y, 0.0])
        
        # Trim to exactly 11
        arrangement = arrangement[:11]
        
        # Add some randomness to break symmetry
        for i in range(len(arrangement)):
            arrangement[i][0] += np.random.normal(0, 0.1)
            arrangement[i][1] += np.random.normal(0, 0.1)
            arrangement[i][2] += np.random.normal(0, 5)  # Small rotation variation
            
        return np.array(arrangement)
    
    def hexagon_gradient_descent(self, initial_solution, max_iter=50) -> np.ndarray:
        """Custom gradient descent for hexagon positioning."""
        solution = initial_solution.copy()
        n_hex = len(solution)
        
        # Simplified gradient descent with adaptive step sizes
        for iteration in range(max_iter):
            # Store current fitness
            current_fitness = self.evaluate_fitness(solution)
            
            # Perturb each hexagon slightly and see if fitness improves
            for i in range(n_hex):
                # Save current state
                old_pos = solution[i, 0:2].copy()
                old_rot = solution[i, 2]
                
                # Try small moves in all directions
                best_move = None
                best_fitness = current_fitness
                
                # Try small movements
                for dx in [-0.05, -0.02, 0, 0.02, 0.05]:
                    for dy in [-0.05, -0.02, 0, 0.02, 0.05]:
                        for drot in [-5, -2, 0, 2, 5]:
                            # Try new position
                            temp_solution = solution.copy()
                            temp_solution[i, 0] += dx
                            temp_solution[i, 1] += dy
                            temp_solution[i, 2] += drot
                            temp_solution[i, 2] = temp_solution[i, 2] % 360
                            
                            new_fitness = self.evaluate_fitness(temp_solution)
                            if new_fitness > best_fitness:
                                best_fitness = new_fitness
                                best_move = (dx, dy, drot)
                
                # Apply best move if we found one
                if best_move is not None:
                    solution[i, 0] += best_move[0]
                    solution[i, 1] += best_move[1]
                    solution[i, 2] += best_move[2]
                    solution[i, 2] = solution[i, 2] % 360
            
            # Early stopping if we're not improving much
            if abs(best_fitness - current_fitness) < 1e-8:
                break
                
        return solution
    
    def simulated_annealing(self, initial_solution, max_iter=1000) -> np.ndarray:
        """Simulated Annealing for global exploration."""
        current_solution = initial_solution.copy()
        current_fitness = self.evaluate_fitness(current_solution)
        
        # Initial temperature and cooling rate
        temp = 1.0
        cooling_rate = 0.999
        min_temp = 1e-6
        
        for iteration in range(max_iter):
            # Generate neighbor solution
            neighbor_solution = current_solution.copy()
            
            # Choose which hexagon to perturb
            hex_idx = np.random.randint(0, len(neighbor_solution))
            
            # Perturb position and rotation
            neighbor_solution[hex_idx, 0] += np.random.normal(0, 0.2)
            neighbor_solution[hex_idx, 1] += np.random.normal(0, 0.2)
            neighbor_solution[hex_idx, 2] += np.random.normal(0, 10)
            neighbor_solution[hex_idx, 2] = neighbor_solution[hex_idx, 2] % 360
            
            # Evaluate neighbor
            neighbor_fitness = self.evaluate_fitness(neighbor_solution)
            
            # Accept or reject
            if neighbor_fitness > current_fitness:
                current_solution = neighbor_solution
                current_fitness = neighbor_fitness
            else:
                # Accept with probability based on temperature
                delta = neighbor_fitness - current_fitness
                if delta > 0 and np.random.random() < math.exp(delta / temp):
                    current_solution = neighbor_solution
                    current_fitness = neighbor_fitness
            
            # Cool down
            temp *= cooling_rate
            
            # Early stopping
            if temp < min_temp or current_fitness > 0.25:
                break
        
        return current_solution
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """Main optimization routine."""
        # Step 1: Generate initial solution
        initial_solution = self.generate_initial_lattice_arrangement()
        
        # Step 2: Simulated Annealing for global search
        sa_solution = self.simulated_annealing(initial_solution, max_iter=1000)
        sa_fitness = self.evaluate_fitness(sa_solution)
        
        # Step 3: Local refinement with gradient descent
        refined_solution = self.hexagon_gradient_descent(sa_solution, max_iter=100)
        refined_fitness = self.evaluate_fitness(refined_solution)
        
        # Step 4: Final optimization with a more thorough local search
        final_solution = self.hexagon_gradient_descent(refined_solution, max_iter=200)
        final_fitness = self.evaluate_fitness(final_solution)
        
        # Return best solution found
        if final_fitness > refined_fitness:
            return final_solution, final_fitness
        else:
            return refined_solution, refined_fitness

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    try:
        # Initialize optimizer
        optimizer = HexagonPackingOptimizer()
        
        # Run optimization
        inner_hex_data, fitness = optimizer.optimize()
        
        # Compute final outer radius
        outer_radius = optimizer.compute_outer_hexagon_radius(inner_hex_data)
        
        # Create outer hexagon data (centered at origin)
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        
        # Calculate benchmark ratio
        benchmark_ratio = fitness / 0.2544
        
    except Exception as e:
        # Fallback to baseline approach
        print(f"Optimization failed: {e}")
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
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_radius = 8.0
        fitness = 1.0 / outer_radius
        benchmark_ratio = fitness / 0.2544
    
    elapsed = time.time() - start_time
    print(f"Eval time: {elapsed:.4f}s")
    
    return inner_hex_data, outer_hex_data, outer_radius

# EVOLVE-BLOCK-END