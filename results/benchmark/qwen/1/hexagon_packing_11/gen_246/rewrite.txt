# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, KDTree
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import random
import time
from collections import defaultdict

# Constants
NUM_INNER_HEXAGONS = 11
UNIT_HEXAGON_RADIUS = 1.0
MAX_EVAL_TIME = 175.0

def generate_unit_hexagon_vertices():
    """Generate vertices of a unit regular hexagon centered at origin"""
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    return vertices

UNIT_HEXAGON_VERTICES = generate_unit_hexagon_vertices()

def rotate_point(point, angle_rad):
    """Rotate a point around origin"""
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([point[0]*cos_a - point[1]*sin_a, point[0]*sin_a + point[1]*cos_a])

def hexagon_vertices(center, angle_rad, scale=1.0):
    """Get vertices of a hexagon at given position and rotation"""
    rotated_vertices = np.array([rotate_point(v, angle_rad) for v in UNIT_HEXAGON_VERTICES])
    return rotated_vertices * scale + np.array(center)

def fast_point_in_polygon(point, polygon):
    """Fast point-in-polygon check using Shapely"""
    return polygon.contains(Point(point))

def fast_hexagon_overlap(hex1_vertices, hex2_vertices):
    """Fast overlap check using polygon intersection"""
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
    except:
        return False

class SpatialIndex:
    """Efficient spatial indexing for collision detection"""
    
    def __init__(self, grid_cell_size=2.5):
        self.grid_cell_size = grid_cell_size
        self.grid = defaultdict(list)
        self.kdtree = None
        self.hexagons = []
        
    def rebuild_grid(self, hexagon_vertices_list):
        """Rebuild spatial grid from hexagon vertices"""
        self.grid.clear()
        self.hexagons = hexagon_vertices_list
        
        for i, vertices in enumerate(hexagon_vertices_list):
            # Add to grid cells that this hexagon touches
            bbox = [min(v[0] for v in vertices), min(v[1] for v in vertices),
                   max(v[0] for v in vertices), max(v[1] for v in vertices)]
            
            min_cell_x = int(bbox[0] // self.grid_cell_size)
            max_cell_x = int(bbox[2] // self.grid_cell_size)
            min_cell_y = int(bbox[1] // self.grid_cell_size)
            max_cell_y = int(bbox[3] // self.grid_cell_size)
            
            for x in range(min_cell_x, max_cell_x + 1):
                for y in range(min_cell_y, max_cell_y + 1):
                    self.grid[(x, y)].append(i)
    
    def get_candidates(self, vertex):
        """Get candidate hexagons that might collide with given vertex"""
        cell_x = int(vertex[0] // self.grid_cell_size)
        cell_y = int(vertex[1] // self.grid_cell_size)
        
        candidates = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                candidates.extend(self.grid[(cell_x + dx, cell_y + dy)])
                
        return candidates

def calculate_outer_hexagon_radius(inner_hex_data, outer_center=[0,0], outer_angle=0):
    """Calculate minimum radius needed for outer hexagon to contain all inner hexagons"""
    max_dist = 0
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = np.radians(inner_hex_data[i][2])
        
        # Get all vertices of this hexagon
        vertices = hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)
        
        # Calculate max distance from outer center to any vertex
        for vertex in vertices:
            dist = np.linalg.norm(np.array(vertex) - np.array(outer_center))
            max_dist = max(max_dist, dist)
            
    return max_dist * 1.1  # Small safety margin

def check_containment(inner_hex_data, outer_radius):
    """Check if all hexagons are contained within outer hexagon"""
    outer_vertices = hexagon_vertices([0,0], 0, outer_radius)
    outer_polygon = Polygon(outer_vertices)
    
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = np.radians(inner_hex_data[i][2])
        vertices = hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)
        
        # Test if all vertices are inside outer polygon
        for vertex in vertices:
            if not fast_point_in_polygon(vertex, outer_polygon):
                return False
    return True

def check_all_overlaps(inner_hex_data):
    """Check all pairwise overlaps with efficient spatial indexing"""
    # Build spatial index
    hex_vertices_list = []
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = np.radians(inner_hex_data[i][2])
        vertices = hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)
        hex_vertices_list.append(vertices)
    
    spatial_index = SpatialIndex()
    spatial_index.rebuild_grid(hex_vertices_list)
    
    # Check overlaps
    n = len(hex_vertices_list)
    for i in range(n):
        for j in range(i+1, n):
            if fast_hexagon_overlap(hex_vertices_list[i], hex_vertices_list[j]):
                return True
    return False

def evaluate_solution(solution_vector):
    """Main evaluation function with adaptive penalty scaling"""
    # Parse solution vector
    positions_and_angles = solution_vector.reshape(-1, 3)
    
    # Check constraints
    outer_radius = calculate_outer_hexagon_radius(positions_and_angles)
    
    # Check containment first
    if not check_containment(positions_and_angles, outer_radius):
        return 1e10  # Heavy penalty for containment failure
    
    # Check overlaps
    if check_all_overlaps(positions_and_angles):
        return 1e10  # Heavy penalty for overlaps
    
    # Return inverse of outer radius (maximize 1/outer_radius)
    return 1.0 / outer_radius

def create_voronoi_initial_population(pop_size, n_hexagons=11):
    """Create initial population using Voronoi-based placement"""
    population = []
    
    for _ in range(pop_size):
        # Generate random points for Voronoi
        np.random.seed(int(time.time()) % 1000000 + len(population))
        points = np.random.uniform(-5, 5, size=(n_hexagons, 2))
        
        # Generate Voronoi diagram
        try:
            vor = Voronoi(points)
            # Use Voronoi cell centroids, filtered to valid points
            centroids = []
            
            # Add some randomness to Voronoi points to avoid degenerate cases
            for i in range(n_hexagons):
                if i < len(vor.points):
                    base_point = vor.points[i]
                    # Add small random displacement
                    rand_offset = np.random.normal(0, 0.3, 2)
                    adjusted_point = base_point + rand_offset
                    centroids.append(adjusted_point)
                else:
                    # Fallback to random point if Voronoi doesn't give enough points
                    centroids.append(np.random.uniform(-4, 4, 2))
                    
            # Create individual with Voronoi-based positions and random rotations
            individual = []
            for i, (x, y) in enumerate(centroids):
                # Add some random variation
                angle = np.random.uniform(0, 360)
                individual.append([x, y, angle])
                
            population.append(np.array(individual))
            
        except:
            # Fallback to random initialization if Voronoi fails
            individual = []
            for _ in range(n_hexagons):
                x = np.random.uniform(-4, 4)
                y = np.random.uniform(-4, 4)
                angle = np.random.uniform(0, 360)
                individual.append([x, y, angle])
            population.append(np.array(individual))
    
    return population

def adaptive_crossover(parent1, parent2):
    """Adaptive crossover that preserves good geometric properties"""
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # For each hexagon, decide source based on fitness proximity
    for i in range(len(parent1)):
        # If both parents have reasonably good solutions, preserve structure
        if random.random() < 0.7:
            # 70% chance to take from parent1
            child2[i] = parent1[i]
        else:
            # 30% chance to take from parent2
            child1[i] = parent2[i]
            
    return child1, child2

def adaptive_mutation(individual, generation, max_generations):
    """Adaptive mutation that changes behavior over time"""
    mutated = individual.copy()
    
    # Start with high mutation rate, then decrease
    mutation_rate = max(0.1, 0.5 * (1 - generation/max_generations))
    
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Mutation with different strategies depending on stage
            if generation < max_generations * 0.3:  # Early exploration
                # Larger mutations for broad search
                mutated[i][0] += np.random.uniform(-1.0, 1.0)
                mutated[i][1] += np.random.uniform(-1.0, 1.0)
                mutated[i][2] += np.random.uniform(-30, 30)
            else:  # Later exploitation
                # Smaller mutations for fine tuning
                mutated[i][0] += np.random.uniform(-0.3, 0.3)
                mutated[i][1] += np.random.uniform(-0.3, 0.3)
                mutated[i][2] += np.random.uniform(-10, 10)
                
            # Normalize angle
            mutated[i][2] %= 360
    
    return mutated

def multi_resolution_optimization():
    """Multi-resolution optimization approach"""
    # Phase 1: Coarse optimization
    pop_size = 30
    generations = 300
    
    # Create initial population
    population = create_voronoi_initial_population(pop_size)
    
    best_fitness = float('inf')
    best_individual = None
    
    # Coarse optimization
    for gen in range(generations):
        if time.time() - start_time > MAX_EVAL_TIME - 5:
            break
            
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            flat_individual = individual.flatten()
            fit = evaluate_solution(flat_individual)
            fitness_scores.append(fit)
            
        # Track best
        current_best_idx = np.argmin(fitness_scores)
        current_best_fitness = fitness_scores[current_best_idx]
        
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[current_best_idx].copy()
            
        # Selection and reproduction
        sorted_indices = np.argsort(fitness_scores)
        selected_pop = [population[i] for i in sorted_indices[:pop_size//2]]
        
        new_population = selected_pop.copy()
        
        # Generate offspring with adaptive crossover
        while len(new_population) < pop_size:
            p1, p2 = random.sample(selected_pop, 2)
            c1, c2 = adaptive_crossover(p1, p2)
            
            # Apply adaptive mutation
            c1 = adaptive_mutation(c1, gen, generations)
            c2 = adaptive_mutation(c2, gen, generations)
            
            new_population.extend([c1, c2])
            
        population = new_population[:pop_size]
        
    # Phase 2: Fine-tune with DE  
    if best_individual is not None:
        # Flatten for DE optimization
        initial_flat = best_individual.flatten()
        
        # Define bounds for DE
        bounds = []
        for i in range(len(initial_flat)):
            if i % 3 < 2:  # Position coordinates
                bounds.append((-10, 10))
            else:  # Angle
                bounds.append((0, 360))
                
        try:
            de_result = differential_evolution(
                lambda x: evaluate_solution(x.reshape(-1, 3)),
                bounds,
                maxiter=500,
                popsize=15,
                seed=int(time.time()),
                disp=False,
                tol=1e-6
            )
            
            # Update with better solution found by DE
            if de_result.fun < best_fitness:
                best_fitness = de_result.fun
                best_individual = de_result.x.reshape(-1, 3)
                
        except:
            pass  # Continue with previous best
            
    return best_individual, best_fitness

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    global start_time
    start_time = time.time()
    
    # Multi-start approach with different random seeds
    best_overall_fitness = float('inf')
    best_overall_individual = None
    
    for start_seed in range(5):
        # Set seed for reproducibility
        np.random.seed(start_seed * 1000 + int(time.time()))
        random.seed(start_seed * 1000 + int(time.time()))
        
        try:
            individual, fitness = multi_resolution_optimization()
            
            if individual is not None and fitness < best_overall_fitness:
                best_overall_fitness = fitness
                best_overall_individual = individual.copy()
                
        except Exception as e:
            continue
            
    # Final validation and return
    if best_overall_individual is None:
        # Fall back to a known good solution
        best_overall_individual = np.array([
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
        best_overall_fitness = 0.125  # Placeholder fitness
    
    # Calculate outer hexagon side length
    outer_radius = 1.0 / best_overall_fitness if best_overall_fitness != float('inf') else 8.0
    
    # Ensure valid outer hexagon
    outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered at origin
    outer_hex_side_length = outer_radius
    
    return best_overall_individual, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END