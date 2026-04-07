# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import math
from collections import defaultdict
import time

class Hexagon:
    """Represents a unit regular hexagon with position and rotation."""
    
    def __init__(self, center_x, center_y, angle_degrees):
        self.center_x = center_x
        self.center_y = center_y
        self.angle_degrees = angle_degrees
        self.radius = 1.0
    
    def get_vertices(self):
        """Get vertices of the hexagon."""
        angle_rad = math.radians(self.angle_degrees)
        base_vertices = []
        for i in range(6):
            theta = angle_rad + i * math.pi/3
            x = self.radius * math.cos(theta)
            y = self.radius * math.sin(theta)
            base_vertices.append((x, y))
        
        vertices = [(x + self.center_x, y + self.center_y) for x, y in base_vertices]
        return np.array(vertices)
    
    def contains_vertex(self, x, y, outer_radius):
        """Check if a point is within the outer hexagon boundary."""
        dx = x - 0
        dy = y - 0
        distance = math.sqrt(dx*dx + dy*dy)
        return distance < outer_radius - 1e-10
    
    def distance_to_center(self, x, y):
        """Calculate distance from center to point."""
        dx = x - self.center_x
        dy = y - self.center_y
        return math.sqrt(dx*dx + dy*dy)

class CollisionDetector:
    """Handles collision detection with spatial indexing acceleration."""
    
    def __init__(self, grid_cell_size=2.8):
        self.grid_cell_size = grid_cell_size
    
    def get_hexagon_bounding_box(self, hexagon):
        """Get bounding box of a hexagon."""
        vertices = hexagon.get_vertices()
        min_x = min(v[0] for v in vertices)
        max_x = max(v[0] for v in vertices)
        min_y = min(v[1] for v in vertices)
        max_y = max(v[1] for v in vertices)
        return min_x, max_x, min_y, max_y
    
    def build_spatial_grid(self, hexagons):
        """Build spatial grid for efficient collision detection."""
        grid = defaultdict(list)
        cell_size = self.grid_cell_size
        
        for i, hexagon in enumerate(hexagons):
            min_x, max_x, min_y, max_y = self.get_hexagon_bounding_box(hexagon)
            
            min_col = int((min_x - 10) // cell_size)
            max_col = int((max_x + 10) // cell_size)
            min_row = int((min_y - 10) // cell_size)
            max_row = int((max_y + 10) // cell_size)

            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    grid[(row, col)].append(i)
        
        return grid
    
    def get_potential_collisions(self, grid, hexagons, hex_index):
        """Get potential collision partners from spatial grid."""
        hexagon = hexagons[hex_index]
        min_x, max_x, min_y, max_y = self.get_hexagon_bounding_box(hexagon)
        
        cell_size = self.grid_cell_size
        min_col = int((min_x - 10) // cell_size)
        max_col = int((max_x + 10) // cell_size)
        min_row = int((min_y - 10) // cell_size)
        max_row = int((max_y + 10) // cell_size)

        candidates = set()
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                if (row, col) in grid:
                    candidates.update(grid[(row, col)])
        
        return list(candidates)
    
    def hexagon_collision(self, hex1_vertices, hex2_vertices):
        """Check if two hexagons collide using Separating Axis Theorem."""
        # Quick bounding box check first
        min1_x = min(v[0] for v in hex1_vertices)
        max1_x = max(v[0] for v in hex1_vertices)
        min1_y = min(v[1] for v in hex1_vertices)
        max1_y = max(v[1] for v in hex1_vertices)

        min2_x = min(v[0] for v in hex2_vertices)
        max2_x = max(v[0] for v in hex2_vertices)
        min2_y = min(v[1] for v in hex2_vertices)
        max2_y = max(v[1] for v in hex2_vertices)

        # If bounding boxes don't overlap, no collision possible
        if max1_x < min2_x or max2_x < min1_x or max1_y < min2_y or max2_y < min1_y:
            return False

        # Get all edges of both hexagons
        edges1 = []
        edges2 = []

        for i in range(6):
            p1 = hex1_vertices[i]
            p2 = hex1_vertices[(i+1)%6]
            edge = (p2[0]-p1[0], p2[1]-p1[1])
            edges1.append(edge)

            p1 = hex2_vertices[i]
            p2 = hex2_vertices[(i+1)%6]
            edge = (p2[0]-p1[0], p2[1]-p1[1])
            edges2.append(edge)

        # Combine all potential separating axes
        all_axes = edges1 + edges2

        # Normalize axes
        for i, axis in enumerate(all_axes):
            length = math.sqrt(axis[0]**2 + axis[1]**2)
            if length > 0:
                all_axes[i] = (axis[0]/length, axis[1]/length)

        # Check projection overlap on each axis
        for axis in all_axes:
            # Project both hexagons onto this axis
            proj1 = []
            proj2 = []

            for v in hex1_vertices:
                dot = v[0]*axis[0] + v[1]*axis[1]
                proj1.append(dot)

            for v in hex2_vertices:
                dot = v[0]*axis[0] + v[1]*axis[1]
                proj2.append(dot)

            min1, max1 = min(proj1), max(proj1)
            min2, max2 = min(proj2), max(proj2)

            # If projections don't overlap, then there's separation
            if max1 < min2 or max2 < min1:
                return False

        return True

class PackingProblem:
    """Main class managing the hexagon packing optimization problem."""
    
    def __init__(self):
        self.hex_radius = 1.0
        self.collision_detector = CollisionDetector()
    
    def calculate_outer_hex_radius(self, hexagons, outer_center=(0,0)):
        """Calculate minimum radius needed for outer hexagon to contain all inner hexagons."""
        max_distance = 0
        
        for hexagon in hexagons:
            vertices = hexagon.get_vertices()
            for x, y in vertices:
                distance = math.sqrt((x - outer_center[0])**2 + (y - outer_center[1])**2)
                max_distance = max(max_distance, distance)
        
        # Add buffer to ensure complete containment
        return max_distance + self.hex_radius * 1.001
    
    def check_containment(self, hexagons, outer_radius):
        """Check if all hexagons are contained within outer hexagon."""
        for hexagon in hexagons:
            vertices = hexagon.get_vertices()
            for x, y in vertices:
                if not hexagon.contains_vertex(x, y, outer_radius):
                    return False
        return True
    
    def evaluate_fitness(self, hexagons):
        """
        Evaluate the fitness of a solution configuration.
        Returns negative value because we want to maximize 1/R (minimize R)
        """
        # Calculate outer radius
        outer_radius = self.calculate_outer_hex_radius(hexagons)
        
        # Check collisions and containment
        num_collisions = 0
        num_out_of_bounds = 0
        
        # Build spatial grid for collision detection
        grid = self.collision_detector.build_spatial_grid(hexagons)
        
        # Check all hexagon pairs for collision
        for i in range(len(hexagons)):
            hex1 = hexagons[i]
            vertices_i = hex1.get_vertices()
            
            # Check containment
            if not self.check_containment([hex1], outer_radius):
                num_out_of_bounds += 1
            
            # Efficiently get potential collision partners using spatial indexing
            potential_collisions = self.collision_detector.get_potential_collisions(grid, hexagons, i)
            for j in potential_collisions:
                if i >= j:
                    continue
                
                hex2 = hexagons[j]
                vertices_j = hex2.get_vertices()
                
                if self.collision_detector.hexagon_collision(vertices_i, vertices_j):
                    num_collisions += 1
        
        # Penalty for collisions or out of bounds
        penalty = 100000 * (num_collisions + num_out_of_bounds)
        
        # If invalid configuration, return poor fitness
        if num_collisions > 0 or num_out_of_bounds > 0:
            return 10000000 + penalty  # Large penalty for invalid solutions
        
        # Return inverse of outer radius (we want to maximize 1/R)
        return 1.0 / outer_radius

class Optimizer:
    """Handles the optimization process with multiple strategies."""
    
    def __init__(self, problem):
        self.problem = problem
        self.max_time = 175
        self.start_time = time.time()
    
    def generate_initial_configs(self):
        """Generate multiple initial configurations."""
        configs = []
        
        # Config 1: Honeycomb-like arrangement
        config1 = [
            [0, 0, 0],           # center
            [-2.1, 0, 0],        # left
            [2.1, 0, 0],         # right
            [-1.05, 1.8, 0],     # top-left
            [1.05, 1.8, 0],      # top-right
            [-1.05, -1.8, 0],    # bottom-left
            [1.05, -1.8, 0],     # bottom-right
            [-3.15, 1.8, 0],     # far top-left
            [3.15, 1.8, 0],      # far top-right
            [-3.15, -1.8, 0],    # far bottom-left
            [3.15, -1.8, 0],     # far bottom-right
        ]
        configs.append(config1)
        
        # Config 2: Alternative honeycomb pattern
        config2 = [
            [0, 0, 0],           # center
            [-2.0, 0, 30],       # left rotated
            [2.0, 0, 30],        # right rotated
            [-1.0, 1.73, 0],     # top-left
            [1.0, 1.73, 0],      # top-right
            [-1.0, -1.73, 0],    # bottom-left
            [1.0, -1.73, 0],     # bottom-right
            [-3.0, 1.73, 30],    # far top-left rotated
            [3.0, 1.73, 30],     # far top-right rotated
            [-3.0, -1.73, 30],   # far bottom-left rotated
            [3.0, -1.73, 30],    # far bottom-right rotated
        ]
        configs.append(config2)
        
        # Config 3: Spiral arrangement
        config3 = [
            [0, 0, 0],           # center
            [1.2, 0, 0],         # right
            [0.6, 1.039, 0],     # upper right
            [-0.6, 1.039, 0],    # upper left
            [-1.2, 0, 0],        # left
            [-0.6, -1.039, 0],   # lower left
            [0.6, -1.039, 0],    # lower right
            [1.8, 1.559, 0],     # far upper right
            [-1.8, 1.559, 0],    # far upper left
            [-1.8, -1.559, 0],   # far lower left
            [1.8, -1.559, 0],    # far lower right
        ]
        configs.append(config3)
        
        # Config 4: Radial pattern
        config4 = [
            [0, 0, 0],           # center
            [-1.9, 0, 0],        # left
            [1.9, 0, 0],         # right
            [-0.95, 1.645, 0],   # top-left
            [0.95, 1.645, 0],    # top-right
            [-0.95, -1.645, 0],  # bottom-left
            [0.95, -1.645, 0],   # bottom-right
            [-2.85, 1.645, 0],   # far top-left
            [2.85, 1.645, 0],    # far top-right
            [-2.85, -1.645, 0],  # far bottom-left
            [2.85, -1.645, 0],   # far bottom-right
        ]
        configs.append(config4)
        
        return configs
    
    def create_hexagons_from_array(self, hex_array):
        """Convert flat array to list of Hexagon objects."""
        hexagons = []
        for i in range(0, len(hex_array), 3):
            center_x, center_y, angle = hex_array[i:i+3]
            hexagons.append(Hexagon(center_x, center_y, angle))
        return hexagons
    
    def evaluate_individual(self, individual):
        """Evaluate a single individual solution."""
        hexagons = self.create_hexagons_from_array(individual)
        return self.problem.evaluate_fitness(hexagons)
    
    def simulated_annealing_refinement(self, individual, max_iter=200):
        """
        Apply simulated annealing refinement to improve a solution
        """
        current_individual = individual.copy()
        current_fitness = self.evaluate_individual(current_individual)

        best_individual = current_individual.copy()
        best_fitness = current_fitness

        temp = 1.0
        cooling_rate = 0.95

        for iteration in range(max_iter):
            # Create neighbor solution by perturbing a random hexagon
            neighbor = current_individual.copy()

            # Choose a random hexagon to perturb
            hex_index = np.random.randint(0, 11)

            # Perturb position and rotation
            pos_change = np.random.normal(0, 0.02, 2)  # Small random movement
            rot_change = np.random.normal(0, 2.0)      # Small rotation change

            # Apply changes
            neighbor[hex_index*3] += pos_change[0]      # x position
            neighbor[hex_index*3 + 1] += pos_change[1]  # y position
            neighbor[hex_index*3 + 2] += rot_change     # rotation

            # Keep rotation within [0, 360]
            neighbor[hex_index*3 + 2] = neighbor[hex_index*3 + 2] % 360

            # Evaluate neighbor
            neighbor_fitness = self.evaluate_individual(neighbor)

            # Accept or reject based on SA criteria
            if neighbor_fitness > current_fitness:
                current_individual = neighbor
                current_fitness = neighbor_fitness
            else:
                # Accept with probability based on temperature
                acceptance_prob = math.exp((neighbor_fitness - current_fitness) / temp)
                if np.random.random() < acceptance_prob:
                    current_individual = neighbor
                    current_fitness = neighbor_fitness

            # Update best solution
            if current_fitness > best_fitness:
                best_individual = current_individual.copy()
                best_fitness = current_fitness

            # Cool down temperature
            temp *= cooling_rate

            # Early stopping if temperature gets too low
            if temp < 1e-8:
                break

        return best_individual, best_fitness
    
    def optimize_with_de(self, initial_guess, bounds):
        """Run differential evolution optimization."""
        def adaptive_evaluate(indiv):
            return self.evaluate_individual(indiv)
        
        try:
            result = differential_evolution(
                func=adaptive_evaluate,
                bounds=bounds,
                maxiter=120,
                popsize=20,
                seed=42,
                disp=False,
                tol=1e-8,
                mutation=(0.8, 1.0),
                recombination=0.9
            )
            return result
        except Exception:
            return None
    
    def solve(self):
        """Main solving method."""
        # Generate initial configurations
        initial_configs = self.generate_initial_configs()
        best_result = None
        best_score = -float('inf')
        
        # Multi-start approach
        for i, config in enumerate(initial_configs):
            if time.time() - self.start_time > self.max_time:
                break
                
            # Convert config to individual array
            individual = np.array(config).flatten()
            
            # Initial evaluation
            initial_fitness = self.evaluate_individual(individual)
            
            if initial_fitness > best_score:
                best_score = initial_fitness
                best_individual = individual.copy()
            
            # Run DE optimization
            bounds = []
            for _ in range(11):
                bounds.extend([(-10, 10), (-10, 10), (0, 360)])
            
            de_result = self.optimize_with_de(individual, bounds)
            
            if de_result and de_result.success:
                de_individual = de_result.x
                
                # Evaluate DE result
                de_fitness = self.evaluate_individual(de_individual)
                
                # Local refinement
                refined_individual, refined_fitness = self.simulated_annealing_refinement(de_individual, max_iter=200)
                
                # Choose the better of DE or refined solution
                final_fitness = max(de_fitness, refined_fitness)
                final_individual = refined_individual if refined_fitness > de_fitness else de_individual
                
                if final_fitness > best_score:
                    best_score = final_fitness
                    best_individual = final_individual.copy()
        
        # Convert back to required format
        best_hexagons = self.create_hexagons_from_array(best_individual)
        outer_radius = self.problem.calculate_outer_hex_radius(best_hexagons)
        
        inner_hex_data = np.array([[h.center_x, h.center_y, h.angle_degrees] for h in best_hexagons])
        outer_hex_data = np.array([0, 0, 0])
        
        return inner_hex_data, outer_hex_data, outer_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    np.random.seed(42)  # For reproducibility
    
    # Create problem instance
    problem = PackingProblem()
    optimizer = Optimizer(problem)
    
    # Solve the problem
    inner_hex_data, outer_hex_data, outer_hex_side_length = optimizer.solve()
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END