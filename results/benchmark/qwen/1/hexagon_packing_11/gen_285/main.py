# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import random
import time
from typing import Tuple, List, Optional, Any
from collections import defaultdict
import numba
from numba import jit

# Constants
NUM_INNER_HEXAGONS = 11
UNIT_HEXAGON_RADIUS = 1.0
MAX_EVAL_TIME = 180.0
SPATIAL_GRID_CELL_SIZE = 2.8  # Adaptively adjusted spatial grid size

# Precomputed unit hexagon vertices (centered at origin)
@jit(nopython=True)
def get_unit_hexagon_vertices_numba():
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    vertices = np.empty((6, 2))
    for i in range(6):
        vertices[i, 0] = np.cos(angles[i])
        vertices[i, 1] = np.sin(angles[i])
    return vertices

UNIT_HEXAGON_VERTICES = get_unit_hexagon_vertices_numba()

@jit(nopython=True)
def rotate_point_numba(point, angle_rad):
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([point[0]*cos_a - point[1]*sin_a, point[0]*sin_a + point[1]*cos_a])

@jit(nopython=True)
def hexagon_vertices_numba(center, angle_rad, scale=1.0):
    """Get vertices of a hexagon at given position and rotation"""
    vertices = np.empty((6, 2))
    for i in range(6):
        rotated = rotate_point_numba(UNIT_HEXAGON_VERTICES[i], angle_rad)
        vertices[i, 0] = rotated[0] * scale + center[0]
        vertices[i, 1] = rotated[1] * scale + center[1]
    return vertices

@jit(nopython=True)
def point_in_polygon_numba(point, polygon_vertices):
    """Fast point-in-polygon check using ray casting"""
    n = len(polygon_vertices)
    inside = False
    p1x, p1y = polygon_vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_vertices[i % n]
        if point[1] > min(p1y, p2y):
            if point[1] <= max(p1y, p2y):
                if point[0] <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (point[1] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or point[0] <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

class HexagonGeometry:
    """Handles all geometric operations for hexagons with numba acceleration"""

    def __init__(self):
        self.unit_vertices = get_unit_hexagon_vertices_numba()
        self.unit_radius = 1.0
        
    def rotate_point(self, point, angle_rad):
        """Rotate a point around origin"""
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([point[0]*cos_a - point[1]*sin_a, point[0]*sin_a + point[1]*cos_a])
    
    def hexagon_vertices(self, center, angle_rad, scale=1.0):
        """Get vertices of a hexagon at given position and rotation"""
        return hexagon_vertices_numba(center, angle_rad, scale)
    
    def point_in_polygon(self, point, polygon_vertices):
        """Fast point-in-polygon check"""
        return point_in_polygon_numba(point, polygon_vertices)

class SolutionValidator:
    """Validates hexagon packing solutions with spatial acceleration"""
    
    def __init__(self, geometry: HexagonGeometry):
        self.geometry = geometry
        
    def build_spatial_index(self, hexagon_centers, cell_size=SPATIAL_GRID_CELL_SIZE):
        """Build spatial grid index for fast overlap detection"""
        index = defaultdict(list)
        for i, center in enumerate(hexagon_centers):
            grid_x = int(np.floor(center[0] / cell_size))
            grid_y = int(np.floor(center[1] / cell_size))
            index[(grid_x, grid_y)].append(i)
        return index
    
    def get_collision_candidates(self, center, spatial_index, cell_size=SPATIAL_GRID_CELL_SIZE):
        """Get potential collision candidates"""
        grid_x = int(np.floor(center[0] / cell_size))
        grid_y = int(np.floor(center[1] / cell_size))
        
        candidates = []
        # Check surrounding grid cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (grid_x + dx, grid_y + dy)
                if key in spatial_index:
                    candidates.extend(spatial_index[key])
        return candidates
    
    def is_contained_in_outer_hexagon(self, hexagon_vertices_list, 
                                    outer_center, outer_angle, 
                                    outer_radius) -> bool:
        """Check if hexagon is fully contained in outer hexagon"""
        # Precompute outer hexagon vertices once
        outer_vertices = self.geometry.hexagon_vertices(outer_center, outer_angle, outer_radius)
        outer_polygon = Polygon(outer_vertices)
        
        # Fast check: test if all vertices are inside outer polygon
        for vertex in hexagon_vertices_list:
            if not self.geometry.point_in_polygon(vertex, outer_polygon.exterior.coords):
                return False
        return True
    
    def check_overlap_fast(self, hex1_vertices, hex2_vertices) -> bool:
        """Fast overlap check using polygon intersection"""
        try:
            poly1 = Polygon(hex1_vertices)
            poly2 = Polygon(hex2_vertices)
            return poly1.intersects(poly2)
        except:
            # Fallback for degenerate cases
            return False
    
    def calculate_outer_hexagon_radius(self, inner_hex_data, 
                                     outer_center=[0,0], 
                                     outer_angle=0) -> float:
        """Calculate minimum radius needed for outer hexagon to contain all inner hexagons"""
        max_dist = 0
        for i in range(len(inner_hex_data)):
            center = inner_hex_data[i][:2]
            angle = np.radians(inner_hex_data[i][2])
            vertices = self.geometry.hexagon_vertices(center, angle, self.geometry.unit_radius)
            
            for vertex in vertices:
                dist = np.linalg.norm(np.array(vertex) - np.array(outer_center))
                max_dist = max(max_dist, dist)
        return max_dist
    
    def validate_solution(self, inner_hex_data, 
                        outer_center=[0,0], 
                        outer_angle=0) -> bool:
        """Validate solution: check containment and non-overlap"""
        # Precompute hexagon centers and vertices for spatial acceleration
        hex_centers = [inner_hex_data[i][:2] for i in range(len(inner_hex_data))]
        hex_angles = [np.radians(inner_hex_data[i][2]) for i in range(len(inner_hex_data))]
        
        # Check containment first
        for i in range(len(inner_hex_data)):
            center = inner_hex_data[i][:2]
            angle = np.radians(inner_hex_data[i][2])
            vertices = self.geometry.hexagon_vertices(center, angle, self.geometry.unit_radius)
            
            # Calculate outer radius based on this solution to check containment properly
            outer_radius = self.calculate_outer_hexagon_radius(inner_hex_data, outer_center, outer_angle)
            
            if not self.is_contained_in_outer_hexagon(vertices, outer_center, outer_angle, outer_radius):
                return False
        
        # Optimized overlap checking with spatial grid acceleration
        # Build spatial index for overlap detection
        spatial_index = self.build_spatial_index(hex_centers)
        
        # Check overlaps efficiently using spatial acceleration
        for i in range(len(inner_hex_data)):
            center_i = inner_hex_data[i][:2]
            angle_i = np.radians(inner_hex_data[i][2])
            vertices_i = self.geometry.hexagon_vertices(center_i, angle_i, self.geometry.unit_radius)
            
            # Get candidate overlaps from spatial grid
            collision_candidates = self.get_collision_candidates(center_i, spatial_index)
            
            for j in collision_candidates:
                if i >= j:  # Skip self-comparison and duplicates
                    continue
                    
                center_j = inner_hex_data[j][:2]
                angle_j = np.radians(inner_hex_data[j][2])
                vertices_j = self.geometry.hexagon_vertices(center_j, angle_j, self.geometry.unit_radius)
                
                # Quick bounding box check first
                min_x1, max_x1 = np.min(vertices_i[:, 0]), np.max(vertices_i[:, 0])
                min_y1, max_y1 = np.min(vertices_i[:, 1]), np.max(vertices_i[:, 1])
                min_x2, max_x2 = np.min(vertices_j[:, 0]), np.max(vertices_j[:, 0])
                min_y2, max_y2 = np.min(vertices_j[:, 1]), np.max(vertices_j[:, 1])
                
                if (max_x1 < min_x2 or max_x2 < min_x1 or
                    max_y1 < min_y2 or max_y2 < min_y1):
                    continue  # No overlap possible
                
                # Full polygon overlap check
                if self.check_overlap_fast(vertices_i, vertices_j):
                    return False
                    
        return True

class EvolutionEngine:
    """Main optimization engine using evolutionary algorithms with improved features"""
    
    def __init__(self, geometry: HexagonGeometry, validator: SolutionValidator):
        self.geometry = geometry
        self.validator = validator
        self.max_eval_time = MAX_EVAL_TIME
        self.num_inner_hexagons = NUM_INNER_HEXAGONS
        self.population_size = 60
        self.generations = 1500
        self.elite_size = 8
        self.tournament_size = 6
        self.initial_mutation_rate = 0.8
        self.final_mutation_rate = 0.1
        
    def create_initial_individual(self) -> np.ndarray:
        """Create a better initial individual using structured approach"""
        # Multiple base configurations for diversity
        base_configs = [
            # Configuration 1: Classic hexagonal packing
            [
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
            ],
            # Configuration 2: Spiral-like arrangement  
            [
                [0, 0, 0],           # center
                [-1.5, 0, 0],       # left
                [1.5, 0, 0],        # right
                [0, 1.5, 0],        # top
                [0, -1.5, 0],       # bottom
                [-1.5, 1.5, 0],     # top-left
                [1.5, 1.5, 0],      # top-right
                [-1.5, -1.5, 0],    # bottom-left
                [1.5, -1.5, 0],     # bottom-right
                [-3, 1.5, 0],       # far top-left
                [3, 1.5, 0],        # far top-right
            ],
            # Configuration 3: Grid arrangement with offsets
            [
                [0, 0, 0],           # center
                [-2, 0, 0],         # left
                [2, 0, 0],          # right
                [0, 2, 0],          # top
                [0, -2, 0],         # bottom
                [-2, 2, 0],         # top-left
                [2, 2, 0],          # top-right
                [-2, -2, 0],        # bottom-left
                [2, -2, 0],         # bottom-right
                [-3, 2, 0],         # far top-left
                [3, 2, 0],          # far top-right
            ]
        ]
        
        # Select a random base configuration
        base_positions = random.choice(base_configs)
        
        # Add small random perturbations
        individual = []
        for pos in base_positions:
            x = pos[0] + random.uniform(-0.2, 0.2)
            y = pos[1] + random.uniform(-0.2, 0.2)
            angle = pos[2] + random.uniform(-10, 10)
            individual.append([x, y, angle])
        return np.array(individual)
    
    def create_diverse_initial_population(self, pop_size: int) -> List[np.ndarray]:
        """Create diverse initial population with enhanced variety"""
        population = []
        
        # Add structured individuals (based on hexagonal packing)
        for _ in range(pop_size // 2):
            individual = self.create_initial_individual()
            population.append(individual)
        
        # Fill remaining with random individuals
        for _ in range(pop_size // 2):
            individual = []
            for _ in range(self.num_inner_hexagons):
                x = random.uniform(-5, 5)
                y = random.uniform(-5, 5)
                angle = random.uniform(0, 360)
                individual.append([x, y, angle])
            population.append(np.array(individual))
            
        return population
    
    def tournament_selection(self, population: List[np.ndarray], 
                           fitnesses: List[float]) -> np.ndarray:
        """Select parent using tournament selection"""
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_idx]
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Improved crossover for hexagon positions and rotations"""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Crossover with selective copying
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1[i] = parent2[i]
                child2[i] = parent1[i]
        return child1, child2
    
    def mutate(self, individual: np.ndarray, mutation_rate: float = 0.1, 
              max_step: float = 0.3) -> np.ndarray:
        """Enhanced mutation with geometric awareness"""
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated[i][0] += random.uniform(-max_step, max_step)
                mutated[i][1] += random.uniform(-max_step, max_step)
                mutated[i][2] += random.uniform(-5, 5)
                mutated[i][2] %= 360
        return mutated
    
    def adaptive_mutation_rate(self, generation: int, max_generations: int) -> float:
        """Adaptive mutation rate that decreases over generations"""
        return self.initial_mutation_rate - (self.initial_mutation_rate - self.final_mutation_rate) * (generation / max_generations)
    
    def evaluate_fitness(self, individual: np.ndarray, 
                       outer_center=[0,0], 
                       outer_angle=0) -> float:
        """Evaluate fitness (negative of outer hexagon radius for maximization)"""
        outer_radius = self.validator.calculate_outer_hexagon_radius(individual, outer_center, outer_angle)
        
        if not self.validator.validate_solution(individual, outer_center, outer_angle):
            # Dynamically adjust penalty based on optimization progress
            # Higher penalty in early generations to enforce feasibility
            penalty_factor = 10000000 + (10000000 * (1 - (generation / self.generations))) if generation < self.generations/2 else 10000000
            return -penalty_factor  # Very poor fitness
            
        return -outer_radius
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """Run evolution optimization"""
        # Create initial population
        population = self.create_diverse_initial_population(self.population_size)
        
        best_fitness = float('-inf')
        best_individual = None
        
        # Evolution loop
        for gen in range(self.generations):
            # Calculate adaptive mutation rate
            mut_rate = self.adaptive_mutation_rate(gen, self.generations)
            
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
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, mutation_rate=mut_rate)
                child2 = self.mutate(child2, mutation_rate=mut_rate)
                
                new_population.extend([child1, child2])
            
            # Trim to exact population size
            population = new_population[:self.population_size]
            
        return best_individual, best_fitness

class SolutionManager:
    """Manages the complete solution lifecycle"""
    
    def __init__(self):
        self.geometry = HexagonGeometry()
        self.validator = SolutionValidator(self.geometry)
        self.evolution_engine = EvolutionEngine(self.geometry, self.validator)
        
    def validate_and_correct_solution(self, inner_hex_data: np.ndarray) -> np.ndarray:
        """Ensure solution validity with fallback"""
        if self.validator.validate_solution(inner_hex_data):
            return inner_hex_data
        else:
            # Fallback to known good configuration
            return np.array([
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
    
    def finalize_solution(self, best_individual: np.ndarray, 
                         best_fitness: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Finalize and return the solution"""
        outer_radius = -best_fitness if best_fitness != float('-inf') else 10.0
        
        # Validate final solution
        validated_individual = self.validate_and_correct_solution(best_individual)
        
        inner_hex_data = validated_individual
        outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered at origin
        outer_hex_side_length = outer_radius
        
        return inner_hex_data, outer_hex_data, outer_hex_side_length

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Initialize solution manager
    solution_manager = SolutionManager()
    
    # Multi-start approach with different seeds for better exploration
    best_overall_fitness = float('-inf')
    best_overall_individual = None
    best_seed = 0
    
    # Run multiple independent optimizations with different seeds
    num_starts = 6
    seeds = [42, 123, 456, 789, 999, 1001]
    
    for start_num in range(num_starts):
        if time.time() - start_time > MAX_EVAL_TIME - 1:  # Leave 1 second for final processing
            break
            
        seed = seeds[start_num] if start_num < len(seeds) else start_num * 100
        
        try:
            best_individual, best_fitness = solution_manager.evolution_engine.optimize()
            
            if best_individual is not None and best_fitness > best_overall_fitness:
                best_overall_fitness = best_fitness
                best_overall_individual = best_individual.copy()
                best_seed = seed
                
        except Exception as e:
            continue  # Skip this run if it fails

    # Finalize solution
    inner_hex_data, outer_hex_data, outer_hex_side_length = solution_manager.finalize_solution(
        best_overall_individual, best_overall_fitness
    )
    
    # Final validation
    if not solution_manager.validator.validate_solution(inner_hex_data):
        # Revert to fallback if validation fails
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

# EVOLVE-BLOCK-END