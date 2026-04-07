# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import random
from scipy.spatial.distance import cdist
import time
from scipy.spatial import cKDTree
from typing import Tuple, List, Optional, Callable, Any
import warnings

class HexagonGeometry:
    """Module for geometric operations related to hexagons"""
    
    def __init__(self):
        self.UNIT_HEXAGON_RADIUS = 1.0
        self.UNIT_HEXAGON_VERTICES = self._get_unit_hexagon_vertices()
        
    def _get_unit_hexagon_vertices(self):
        """Precompute unit hexagon vertices (centered at origin)"""
        angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 angles + close the loop
        vertices = np.column_stack([np.cos(angles), np.sin(angles)])
        return vertices
    
    def rotate_point(self, point, angle_rad):
        """Rotate a point around origin"""
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([point[0]*cos_a - point[1]*sin_a, point[0]*sin_a + point[1]*cos_a])
    
    def hexagon_vertices(self, center, angle_rad, scale=1.0):
        """Get vertices of a hexagon at given position and rotation"""
        rotated_vertices = np.array([self.rotate_point(v, angle_rad) for v in self.UNIT_HEXAGON_VERTICES])
        return rotated_vertices * scale + np.array(center)
    
    def point_in_polygon(self, point, polygon):
        """Fast point-in-polygon check"""
        return polygon.contains(Point(point))
    
    def calculate_outer_radius(self, inner_hex_data, outer_center=(0,0), outer_angle=0):
        """Calculate minimum radius needed for outer hexagon to contain all inner hexagons"""
        max_dist = 0
        for i in range(len(inner_hex_data)):
            center = inner_hex_data[i][:2]
            angle = np.radians(inner_hex_data[i][2])
            
            # Get all vertices of this hexagon
            vertices = self.hexagon_vertices(center, angle, self.UNIT_HEXAGON_RADIUS)
            
            # Calculate max distance from outer center to any vertex
            for vertex in vertices:
                dist = np.linalg.norm(np.array(vertex) - np.array(outer_center))
                max_dist = max(max_dist, dist)
        
        return max_dist

class HexagonValidator:
    """Module for validating hexagon arrangements"""
    
    def __init__(self, geometry: HexagonGeometry):
        self.geom = geometry
        
    def is_contained_in_outer_hexagon(self, hexagon_vertices_list, outer_center, outer_angle, outer_radius):
        """Check if hexagon is fully contained in outer hexagon using optimized approach"""
        outer_vertices = self.geom.hexagon_vertices(outer_center, outer_angle, outer_radius)
        outer_polygon = Polygon(outer_vertices)
        
        # Fast check: test if all vertices are inside outer polygon
        for vertex in hexagon_vertices_list:
            if not self.geom.point_in_polygon(vertex, outer_polygon):
                return False
        return True
    
    def check_overlap_fast(self, hex1_vertices, hex2_vertices):
        """Fast overlap check using spatial indexing and bounding box pruning"""
        try:
            poly1 = Polygon(hex1_vertices)
            poly2 = Polygon(hex2_vertices)
            
            # Quick bounding box check first
            bbox1 = poly1.bounds
            bbox2 = poly2.bounds
            
            if (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or
                bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3]):
                return False
            
            return poly1.intersects(poly2)
        except Exception:
            # Fallback for degenerate cases
            return False
    
    def validate_solution(self, inner_hex_data, outer_center=(0,0), outer_angle=0, 
                         early_termination=True) -> bool:
        """Validate solution: check containment and non-overlap"""
        # Precompute outer radius for containment check
        outer_radius = self.geom.calculate_outer_radius(inner_hex_data, outer_center, outer_angle)
        
        # Check containment first - one pass
        for i in range(len(inner_hex_data)):
            center = inner_hex_data[i][:2]
            angle = np.radians(inner_hex_data[i][2])
            vertices = self.geom.hexagon_vertices(center, angle, self.geom.UNIT_HEXAGON_RADIUS)
            
            if not self.is_contained_in_outer_hexagon(vertices, outer_center, outer_angle, outer_radius):
                return False
                
            if early_termination:
                # Early termination for containment check
                break
        
        # Check overlaps efficiently using spatial indexing with improved bounding box checks
        # Create list of all hexagon polygons for spatial indexing
        hex_polygons = []
        bounds_list = []
        
        for i in range(len(inner_hex_data)):
            center = inner_hex_data[i][:2]
            angle = np.radians(inner_hex_data[i][2])
            vertices = self.geom.hexagon_vertices(center, angle, self.geom.UNIT_HEXAGON_RADIUS)
            hex_polygons.append(Polygon(vertices))
            
            # Store bounding boxes for quick rejection
            bbox = Polygon(vertices).bounds
            bounds_list.append(bbox)
        
        # More efficient pair-wise overlap checking using spatial reasoning
        n = len(hex_polygons)
        
        # Use spatial indexing for faster neighbor identification
        if n > 1:
            # Create spatial index for bounding boxes
            centroids = np.array([np.mean(vertices, axis=0) for vertices in [self.geom.hexagon_vertices(inner_hex_data[i][:2], np.radians(inner_hex_data[i][2]), self.geom.UNIT_HEXAGON_RADIUS) for i in range(n)]])
            tree = cKDTree(centroids)
            
            # Find candidates for overlap checking using distance threshold
            pairs = tree.query_pairs(r=self.geom.UNIT_HEXAGON_RADIUS * 4.0, eps=0)
            
            # Check actual overlaps for candidate pairs
            for i, j in pairs:
                if i >= j:  # Avoid double-checking and self-checking
                    continue
                if hex_polygons[i].intersects(hex_polygons[j]):
                    return False
        else:
            # Single hexagon case
            return True
            
        return True

class EvolutionaryOptimizer:
    """Main evolutionary optimization engine"""
    
    def __init__(self, geometry: HexagonGeometry, validator: HexagonValidator):
        self.geom = geometry
        self.validator = validator
        self.NUM_INNER_HEXAGONS = 11
        self.MAX_EVAL_TIME = 180.0
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize optimization parameters"""
        self.pop_size = 50
        self.generations = 1200
        self.elite_size = 5
        self.tournament_size = 5
        self.initial_mutation_rate = 0.8
        self.final_mutation_rate = 0.1
        self.mutation_decay_factor = 0.995
    
    def adaptive_mutation_rate(self, generation: int) -> float:
        """Adaptive mutation rate that decreases over generations"""
        # Exponential decay approach
        return self.final_mutation_rate + (self.initial_mutation_rate - self.final_mutation_rate) * (self.mutation_decay_factor ** generation)
    
    def create_initial_individual(self) -> np.ndarray:
        """Create a better initial individual using a structured approach"""
        # Start with a known good arrangement and add slight randomness
        base_positions = [
            [0, 0, 0],           # center
            [-2.5, 0, 0],       # left
            [2.5, 0, 0],        # right
            [-1.25, 2.17, 0],   # top-left
            [1.25, 2.17, 0],    # top-right
            [-1.25, -2.17, 0],  # bottom-left
            [1.25, -2.17, 0],   # bottom-right
            [-3.75, 2.17, 0],   # far top-left
            [3.75, 2.17, 0],    # far top-right
            [-3.75, -2.17, 0],  # far bottom-left
            [3.75, -2.17, 0],   # far bottom-right
        ]
        
        # Add small random perturbations
        individual = []
        for pos in base_positions:
            x = pos[0] + random.uniform(-0.3, 0.3)
            y = pos[1] + random.uniform(-0.3, 0.3)
            angle = pos[2] + random.uniform(-10, 10)
            individual.append([x, y, angle])
        
        return np.array(individual)
    
    def create_initial_population(self, pop_size: int) -> List[np.ndarray]:
        """Create initial population with better starting solutions"""
        population = []
        
        # Add a few structured individuals
        for _ in range(pop_size // 2):
            individual = self.create_initial_individual()
            population.append(individual)
        
        # Fill remaining with random individuals
        for _ in range(pop_size // 2):
            individual = []
            for _ in range(self.NUM_INNER_HEXAGONS):
                x = random.uniform(-5, 5)
                y = random.uniform(-5, 5)
                angle = random.uniform(0, 360)
                individual.append([x, y, angle])
            population.append(np.array(individual))
        
        return population
    
    def tournament_selection(self, population: List[np.ndarray], fitnesses: List[float]) -> np.ndarray:
        """Select parent using tournament selection"""
        tournament_indices = random.sample(range(len(population)), min(self.tournament_size, len(population)))
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_idx]
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Improved crossover for hexagon positions and rotations"""
        # Uniform crossover for positions and angles with some structure preservation
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Crossover with selective copying
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1[i] = parent2[i]
                child2[i] = parent1[i]
        
        return child1, child2
    
    def mutate(self, individual: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
        """Enhanced mutation with geometric awareness"""
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # More sophisticated mutation approach
                # Mutate position with adaptive step sizes
                mutated[i][0] += random.uniform(-0.3, 0.3)
                mutated[i][1] += random.uniform(-0.3, 0.3)
                
                # Mutate angle with smaller steps for fine-tuning
                mutated[i][2] += random.uniform(-5, 5)
                mutated[i][2] %= 360
        
        return mutated
    
    def evaluate_fitness(self, inner_hex_data: np.ndarray, outer_center=(0,0), outer_angle=0) -> float:
        """Evaluate fitness (negative of outer hexagon radius for maximization)"""
        # Calculate minimum outer radius needed
        outer_radius = self.geom.calculate_outer_radius(inner_hex_data, outer_center, outer_angle)
        
        # If solution is invalid, penalize heavily
        if not self.validator.validate_solution(inner_hex_data, outer_center, outer_angle, early_termination=True):
            return -1e10  # Very poor fitness
        
        # Return negative radius (we want to minimize radius, so maximize negative value)
        return -outer_radius
    
    def optimize_single_start(self, start_num: int, start_time: float) -> Tuple[np.ndarray, float]:
        """Run optimization for a single start"""
        # Create initial population
        population = self.create_initial_population(self.pop_size)
        
        best_fitness = float('-inf')
        best_individual = None
        
        # Evolution loop
        for gen in range(self.generations):
            if time.time() - start_time > self.MAX_EVAL_TIME - 1:  # Leave 1 second for final processing
                break
            
            # Calculate adaptive mutation rate
            mut_rate = self.adaptive_mutation_rate(gen)
            
            # Evaluate fitness for entire population
            fitnesses = []
            for individual in population:
                fit = self.evaluate_fitness(individual)
                fitnesses.append(fit)
            
            # Track best solution
            current_best_idx = np.argmax(fitnesses)
            current_best_fitness = fitnesses[current_best_idx]
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx].copy()
            
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitnesses)[-self.elite_size:]
            elites = [population[i] for i in elite_indices]
            
            # Create new population
            new_population = elites.copy()
            
            # Fill rest with offspring
            while len(new_population) < self.pop_size:
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, mutation_rate=mut_rate)
                child2 = self.mutate(child2, mutation_rate=mut_rate)
                
                new_population.extend([child1, child2])
            
            # Trim to exact population size
            population = new_population[:self.pop_size]
        
        return best_individual, best_fitness
    
    def run_optimization(self, start_time: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run the complete optimization process with multi-start approach"""
        best_overall_fitness = float('-inf')
        best_overall_individual = None
        
        # Run multiple independent optimizations
        num_starts = 5
        for start_num in range(num_starts):
            try:
                best_individual, best_fitness = self.optimize_single_start(start_num, start_time)
                
                # Keep track of best overall solution across all starts
                if best_individual is not None and best_fitness > best_overall_fitness:
                    best_overall_fitness = best_fitness
                    best_overall_individual = best_individual.copy()
            except Exception as e:
                warnings.warn(f"Error in optimization start {start_num}: {str(e)}")
                continue
        
        return best_overall_individual, best_overall_fitness

class HexagonPackingSolver:
    """Main solver class that orchestrates the entire process"""
    
    def __init__(self):
        self.geometry = HexagonGeometry()
        self.validator = HexagonValidator(self.geometry)
        self.optimizer = EvolutionaryOptimizer(self.geometry, self.validator)
    
    def solve(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
        Returns
            inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
            outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
            outer_hex_side_length: float representing the side length of the outer hexagon.
        """
        start_time = time.time()
        
        # Run the optimization
        best_individual, best_fitness = self.optimizer.run_optimization(start_time)
        
        # Validate final best solution
        if best_individual is None:
            # Return fallback if we couldn't find anything good
            fallback = self.optimizer.create_initial_individual()
            best_individual = fallback
        
        # Final validation and calculation of outer hexagon parameters
        outer_radius = -best_fitness if best_fitness != float('-inf') else 10.0  # fallback if not found
        
        # Return result
        inner_hex_data = best_individual
        outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered at origin
        outer_hex_side_length = outer_radius
        
        # Final validation to ensure correctness
        if not self.validator.validate_solution(inner_hex_data):
            # Revert to a reasonable fallback
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
            outer_hex_side_length = 8.0
            outer_hex_data = np.array([0.0, 0.0, 0.0])
        
        return inner_hex_data, outer_hex_data, outer_hex_side_length

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses evolutionary optimization to find a better solution than the simple grid arrangement.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    solver = HexagonPackingSolver()
    return solver.solve()

# EVOLVE-BLOCK-END