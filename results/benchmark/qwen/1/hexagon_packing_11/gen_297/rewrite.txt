# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
from numba import jit
import warnings
import random
from collections import defaultdict
import math

warnings.filterwarnings('ignore')

@jit(nopython=True)
def hexagon_vertices(x, y, angle_deg, side_length=1):
    """Calculate vertices of a hexagon given center, angle, and side length"""
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vx = x + side_length * np.cos(theta)
        vy = y + side_length * np.sin(theta)
        vertices.append((vx, vy))
    return np.array(vertices)

class GeometricConstraintSolver:
    """Solves geometric constraints efficiently using analytical methods"""
    
    @staticmethod
    def distance_point_to_hexagon(point, hex_center, hex_angle, side_length=1):
        """Fast distance calculation from point to hexagon boundary"""
        # Simplified distance calculation for hexagon
        x, y = point
        cx, cy = hex_center
        dx = x - cx
        dy = y - cy
        distance = np.sqrt(dx*dx + dy*dy)
        # For unit hexagon, the circumradius is 1, inradius is sqrt(3)/2
        return max(0, distance - 1.0)
    
    @staticmethod
    def hexagon_overlap_fast(h1_center, h1_angle, h2_center, h2_angle, side_length=1):
        """Quick overlap detection without full polygon intersection"""
        # Fast circle-circle overlap test (circumradius = 1)
        dx = h1_center[0] - h2_center[0]
        dy = h1_center[1] - h2_center[1]
        dist_sq = dx*dx + dy*dy
        return dist_sq < 4.0  # 2*2 = 4 (sum of circumradii)
    
    @staticmethod
    def compute_min_outer_radius(hex_centers, hex_angles, side_length=1):
        """Compute minimum outer radius analytically"""
        max_dist = 0
        center = (0.0, 0.0)
        for i in range(len(hex_centers)):
            x, y = hex_centers[i]
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            max_dist = max(max_dist, dist + 1.0)  # Add 1 for circumradius
        return max_dist * 1.1  # Safety margin

class HexagonPacker:
    """Manages hexagon packing with efficient constraint checking"""
    
    def __init__(self, num_hexagons=11, side_length=1.0):
        self.num_hexagons = num_hexagons
        self.side_length = side_length
        self.constraint_solver = GeometricConstraintSolver()
    
    def validate_solution(self, positions, angles):
        """Fast validation using geometric constraints"""
        # Check containment quickly
        outer_radius = self.constraint_solver.compute_min_outer_radius(positions, angles)
        
        # Quick overlap check using distance bounds
        for i in range(self.num_hexagons):
            for j in range(i+1, self.num_hexagons):
                if not self.constraint_solver.hexagon_overlap_fast(
                    positions[i], angles[i],
                    positions[j], angles[j]
                ):
                    continue
                    
                # Only do detailed check if necessary
                if self._check_overlap_detail(positions[i], angles[i], positions[j], angles[j]):
                    return False
                    
        return True
    
    def _check_overlap_detail(self, pos1, angle1, pos2, angle2):
        """Detailed overlap checking using shapely"""
        vertices1 = hexagon_vertices(pos1[0], pos1[1], angle1)
        vertices2 = hexagon_vertices(pos2[0], pos2[1], angle2)
        poly1 = Polygon(vertices1)
        poly2 = Polygon(vertices2)
        return poly1.intersects(poly2)
    
    def check_containment(self, positions, angles, outer_radius):
        """Check if all hexagons fit within outer hexagon"""
        outer_vertices = hexagon_vertices(0, 0, 0, outer_radius)
        outer_poly = Polygon(outer_vertices)
        
        for i in range(self.num_hexagons):
            vertices = hexagon_vertices(positions[i][0], positions[i][1], angles[i])
            hex_poly = Polygon(vertices)
            if not outer_poly.contains(hex_poly):
                return False
        return True

class HybridOptimizationEngine:
    """Hybrid optimization combining global and local search strategies"""
    
    def __init__(self, num_hexagons=11, side_length=1.0):
        self.num_hexagons = num_hexagons
        self.side_length = side_length
        self.packer = HexagonPacker(num_hexagons, side_length)
        self.iteration_count = 0
    
    def objective_function(self, solution):
        """Objective function to maximize 1/outer_radius"""
        positions = solution[:22].reshape(-1, 2)
        angles = solution[22:]
        
        outer_radius = self.packer.constraint_solver.compute_min_outer_radius(positions, angles)
        
        # If invalid, penalize heavily
        if not self.packer.validate_solution(positions, angles):
            return 1e10
            
        # Check containment
        if not self.packer.check_containment(positions, angles, outer_radius):
            return 1e10
            
        # Return negative of 1/outer_radius (we want to maximize 1/outer_radius)
        return -1.0 / outer_radius
    
    def generate_initial_population(self):
        """Generate diverse initial solutions using geometric insight"""
        # Start with a known good configuration inspired by hexagonal packing theory
        base_patterns = [
            # Pattern 1: Honeycomb with one extra at corner
            {
                'positions': [
                    [0.0, 0.0],      # center
                    [-2.0, 0.0],     # left
                    [2.0, 0.0],      # right
                    [-1.0, 1.732],   # top-left
                    [1.0, 1.732],    # top-right
                    [-1.0, -1.732],  # bottom-left
                    [1.0, -1.732],   # bottom-right
                    [-3.0, 1.732],   # far top-left
                    [3.0, 1.732],    # far top-right
                    [-3.0, -1.732],  # far bottom-left
                    [3.0, -1.732],   # far bottom-right
                ],
                'angles': [0.0] * 11
            },
            # Pattern 2: Concentric rings with central core
            {
                'positions': [
                    [0.0, 0.0],      # center
                    [0.0, 2.0],      # top
                    [0.0, -2.0],     # bottom
                    [1.732, 0.0],    # right
                    [-1.732, 0.0],   # left
                    [0.866, 1.5],    # top-right
                    [-0.866, 1.5],   # top-left
                    [0.866, -1.5],   # bottom-right
                    [-0.866, -1.5],  # bottom-left
                    [1.732, 1.732],  # far top-right
                    [-1.732, -1.732], # far bottom-left
                ],
                'angles': [0.0] * 11
            },
            # Pattern 3: Zig-zag arrangement
            {
                'positions': [
                    [0.0, 0.0],      # center
                    [0.0, 2.0],      # top
                    [0.0, -2.0],     # bottom
                    [1.732, 1.0],    # top-right
                    [-1.732, 1.0],   # top-left
                    [1.732, -1.0],   # bottom-right
                    [-1.732, -1.0],  # bottom-left
                    [3.464, 0.0],    # far right
                    [-3.464, 0.0],   # far left
                    [0.0, 3.464],    # very top
                    [0.0, -3.464],   # very bottom
                ],
                'angles': [0.0] * 11
            }
        ]
        
        initial_solutions = []
        for pattern in base_patterns:
            # Add random perturbations to each pattern
            for _ in range(2):  # Generate 2 variants per pattern
                solution = []
                for pos in pattern['positions']:
                    # Add small random perturbations
                    new_pos = [
                        pos[0] + random.uniform(-0.5, 0.5),
                        pos[1] + random.uniform(-0.5, 0.5)
                    ]
                    solution.extend(new_pos)
                solution.extend(pattern['angles'])
                
                # Add Gaussian noise to angles
                for i in range(len(pattern['angles'])):
                    solution[22+i] += random.gauss(0, 10)
                
                initial_solutions.append(np.array(solution))
        
        return initial_solutions
    
    def local_search_refinement(self, solution):
        """Refine solution using local search with multiple strategies"""
        # Local gradient-based optimization
        def local_objective(x):
            return self.objective_function(x)
        
        # Multiple local search attempts with different starting points
        best_solution = solution.copy()
        best_value = local_objective(solution)
        
        # Try gradient-based refinement
        try:
            # Use L-BFGS-B for local refining
            result = minimize(
                local_objective,
                solution,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            if result.success and local_objective(result.x) < best_value:
                best_solution = result.x
                best_value = local_objective(result.x)
        except:
            pass
        
        # Try neighborhood search
        for _ in range(20):
            neighbor = best_solution.copy()
            # Randomly modify positions and angles
            for i in range(22):
                if i < 22:  # Positions
                    neighbor[i] += random.uniform(-0.1, 0.1)
            for i in range(11):
                neighbor[22+i] += random.uniform(-2, 2)
            
            new_value = local_objective(neighbor)
            if new_value < best_value:
                best_solution = neighbor
                best_value = new_value
        
        return best_solution
    
    def hybrid_optimization(self):
        """Main hybrid optimization routine"""
        # Generate diverse starting points
        initial_solutions = self.generate_initial_population()
        
        best_solution = None
        best_score = float('inf')
        best_iter = 0
        
        # Simulated Annealing with geometric improvements
        current_solution = initial_solutions[0] if initial_solutions else np.zeros(33)
        current_score = self.objective_function(current_solution)
        
        if current_score < best_score:
            best_solution = current_solution
            best_score = current_score
            best_iter = 0
        
        # Simulated Annealing parameters
        temperature = 1000.0
        cooling_rate = 0.95
        min_temperature = 1e-6
        max_iterations = 1000
        
        # Iterative optimization with hybrid strategies
        for iteration in range(max_iterations):
            # Create neighbor solution
            neighbor = current_solution.copy()
            
            # Random perturbation
            for i in range(33):
                if i < 22:  # Position coordinates
                    neighbor[i] += random.gauss(0, 0.5)
                else:  # Angles
                    neighbor[i] += random.gauss(0, 5)
            
            # Accept or reject neighbor
            neighbor_score = self.objective_function(neighbor)
            delta = neighbor_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / max(temperature, 1e-10)):
                current_solution = neighbor
                current_score = neighbor_score
                
                if current_score < best_score:
                    best_solution = current_solution
                    best_score = current_score
                    best_iter = iteration
            
            # Cool down temperature
            temperature *= cooling_rate
            
            # Early stopping if no improvement in long time
            if iteration - best_iter > 200:
                break
        
        # Final local refinement
        if best_solution is not None:
            best_solution = self.local_search_refinement(best_solution)
        
        return best_solution if best_solution is not None else initial_solutions[0] if initial_solutions else np.zeros(33)

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    try:
        # Initialize optimization engine
        optimizer = HybridOptimizationEngine(num_hexagons=11, side_length=1.0)
        
        # Run hybrid optimization
        solution = optimizer.hybrid_optimization()
        
        # Extract results
        positions = solution[:22].reshape(-1, 2)
        angles = solution[22:]
        
        # Create inner hex data
        inner_hex_data = np.column_stack([positions, angles])
        
        # Create outer hex data (centered)
        outer_hex_data = np.array([0, 0, 0])
        
        # Calculate outer hex side length
        max_dist = 0
        outer_center = (0, 0)
        
        # Get all vertices of all inner hexagons
        all_vertices = []
        for i in range(len(positions)):
            x, y = positions[i]
            angle = angles[i]
            hex_vertices = hexagon_vertices(x, y, angle)
            all_vertices.extend(hex_vertices)
        
        # Find maximum distance from center
        for vertex in all_vertices:
            dist = np.sqrt((vertex[0] - outer_center[0])**2 + (vertex[1] - outer_center[1])**2)
            max_dist = max(max_dist, dist)
        
        # Convert to side length for regular hexagon
        outer_hex_side_length = max_dist / (np.sqrt(3) / 2) * 1.1  # Adding safety factor
        
        elapsed_time = time.time() - start_time
        print(f"Optimization completed in {elapsed_time:.2f} seconds")
        
        return inner_hex_data, outer_hex_data, outer_hex_side_length
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback to initial solution
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
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END