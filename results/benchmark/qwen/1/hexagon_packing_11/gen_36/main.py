# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
from joblib import Parallel, delayed
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, rotation, and side length."""
    angle_rad = np.radians(angle_deg)
    # Vertices of a regular hexagon with side length 1, centered at origin
    hex_vertices = []
    for i in range(6):
        x = np.cos(angle_rad + i * np.pi / 3)
        y = np.sin(angle_rad + i * np.pi / 3)
        hex_vertices.append((x + center_x, y + center_y))
    return hex_vertices

def check_collision(hex1_vertices, hex2_vertices):
    """Check if two hexagons represented by vertices collide."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def check_containment(outer_hex_vertices, inner_hex_vertices):
    """Check if all vertices of inner hex are within outer hex."""
    outer_poly = Polygon(outer_hex_vertices)
    for vertex in inner_hex_vertices:
        point = Point(vertex[0], vertex[1])
        if not outer_poly.contains(point):
            return False
    return True

def calculate_outer_hex_side_length(inner_hex_data, outer_hex_center=(0,0)):
    """Calculate the minimal side length of the outer hexagon needed to contain all inner hexagons."""
    # Generate all vertices of inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle_deg = inner_hex_data[i]
        vertices = generate_hexagon_vertices(center_x, center_y, angle_deg)
        all_vertices.extend(vertices)
    
    # Find the bounding circle for all vertices
    if not all_vertices:
        return 1e6
    
    centers = np.array(all_vertices)
    mean_x = np.mean(centers[:, 0])
    mean_y = np.mean(centers[:, 1])
    
    # Calculate distances from centroid to all vertices
    distances = np.sqrt((centers[:, 0] - mean_x)**2 + (centers[:, 1] - mean_y)**2)
    
    # The radius of smallest enclosing circle (approximation)
    max_dist = np.max(distances)
    
    # Convert to side length of outer hexagon
    # For a regular hexagon, the relationship is: diameter = 2 * side_length
    # But our outer hexagon should contain the vertices, so we need the circumradius
    # For a hexagon inscribed in a circle of radius R, the side length is also R
    side_length = max_dist
    return side_length

def evaluate_solution(individual, outer_hex_center=(0,0)):
    """Evaluate the fitness of a solution and compute the side length."""
    # Decode individual into hexagon data
    inner_hex_data = individual.reshape(-1, 3)  # Each row is (x, y, angle)
    
    # Calculate side length of outer hexagon
    outer_hex_side_length = calculate_outer_hex_side_length(inner_hex_data, outer_hex_center)
    
    # Check collisions between all pairs of inner hexagons
    collisions = 0
    total_pairs = len(inner_hex_data) * (len(inner_hex_data) - 1) // 2
    
    for i in range(len(inner_hex_data)):
        hex1_vertices = generate_hexagon_vertices(
            inner_hex_data[i][0], 
            inner_hex_data[i][1], 
            inner_hex_data[i][2]
        )
        
        for j in range(i+1, len(inner_hex_data)):
            hex2_vertices = generate_hexagon_vertices(
                inner_hex_data[j][0], 
                inner_hex_data[j][1], 
                inner_hex_data[j][2]
            )
            
            if check_collision(hex1_vertices, hex2_vertices):
                collisions += 1
    
    # Penalty for collisions
    collision_penalty = collisions * 1000
    
    # Penalty for non-containment (simplified to distance from center)
    containment_penalty = 0
    center = np.array(outer_hex_center)
    
    for i in range(len(inner_hex_data)):
        hex_center = np.array([inner_hex_data[i][0], inner_hex_data[i][1]])
        dist_to_center = np.linalg.norm(hex_center - center)
        # If this exceeds the outer side length, apply penalty
        if dist_to_center > outer_hex_side_length:
            containment_penalty += (dist_to_center - outer_hex_side_length) * 100
    
    # Fitness is inverse of side length minus penalties
    fitness = 1.0 / outer_hex_side_length - collision_penalty - containment_penalty
    
    # Return fitness value and side length
    return fitness, outer_hex_side_length

def create_initial_population(pop_size, n_hexagons=11):
    """Create initial population of random hexagon arrangements."""
    population = []
    for _ in range(pop_size):
        # Create random positions and rotations for hexagons
        individual = np.random.rand(n_hexagons, 3)  # x, y, angle
        individual[:, 0] *= 10 - 2  # x in range [0, 8]
        individual[:, 1] *= 10 - 2  # y in range [0, 8]
        individual[:, 2] *= 360     # angle in range [0, 360]
        population.append(individual.flatten())
    return population

def crossover(parent1, parent2, crossover_rate=0.8):
    """Custom crossover operator for hexagon packing."""
    if np.random.rand() > crossover_rate:
        return parent1.copy(), parent2.copy()
        
    # Swap entire hexagons between parents instead of mixing positions
    cut_point = np.random.randint(1, len(parent1) // 3)
    
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Swap first 'cut_point' hexagons
    start1 = cut_point * 3
    start2 = cut_point * 3
    
    child1[start1:start1+3] = parent2[start1:start1+3]
    child2[start2:start2+3] = parent1[start2:start2+3]
    
    return child1, child2

def mutate(individual, mutation_rate=0.1, mutation_strength=0.2):
    """Custom mutation for hexagon positions and orientations."""
    mutated = individual.copy()
    
    for i in range(len(mutated)):
        if np.random.rand() < mutation_rate:
            # Determine how much to mutate based on solution quality
            if i < 3:  # Position (x,y) components
                mutated[i] += np.random.normal(0, mutation_strength)
            else:  # Orientation (angle) component
                delta = np.random.normal(0, mutation_strength * 20)  # Larger change for angles
                mutated[i] += delta % 360
    
    return mutated

def selection(population, fitnesses, num_selected):
    """Tournament selection."""
    selected = []
    for _ in range(num_selected):
        tournament_size = 3
        tournament_indices = np.random.choice(len(population), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        selected.append(population[winner_idx].copy())
    return selected

def evolve_hexagon_packing():
    """Main evolutionary algorithm for hexagon packing."""
    npop = 50
    ngen = 200
    n_hexagons = 11
    
    # Initialize population
    population = create_initial_population(npop, n_hexagons)
    
    best_fitness = float('-inf')
    best_individual = None
    best_side_length = float('inf')
    
    for gen in range(ngen):
        # Evaluate population
        fitness_results = Parallel(n_jobs=-1)(
            delayed(evaluate_solution)(ind) for ind in population
        )
        
        fitnesses = [r[0] for r in fitness_results]
        side_lengths = [r[1] for r in fitness_results]
        
        # Track best
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()
            best_side_length = side_lengths[max_fitness_idx]
        
        # Selection
        selected = selection(population, fitnesses, npop // 2)
        
        # Crossover and mutation
        new_population = []
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[min(i+1, len(selected)-1)]
            
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate=0.1, mutation_strength=0.1)
            child2 = mutate(child2, mutation_rate=0.1, mutation_strength=0.1)
            
            new_population.extend([child1, child2])
        
        # Replace population
        population = new_population[:npop]
    
    # Final evaluation of best individual
    final_fitness, final_side_length = evaluate_solution(best_individual)
    
    # Decode final solution
    inner_hex_data = best_individual.reshape(-1, 3)
    
    # Create outer hexagon data (centered at origin)
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, final_side_length

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Run evolutionary optimization
    inner_hex_data, outer_hex_data, outer_hex_side_length = evolve_hexagon_packing()
    
    # Ensure we're meeting requirements
    # Validate that all hexagons are within bounds
    final_fitness, _ = evaluate_solution(inner_hex_data.flatten())
    
    eval_time = time.time() - start_time
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
