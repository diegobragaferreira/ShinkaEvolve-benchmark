# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
import random
import time
from copy import deepcopy

def generate_hexagon_vertices(center_x, center_y, side_length, angle_degrees):
    """Generate vertices of a regular hexagon given center, side length, and rotation."""
    angle_rad = np.radians(angle_degrees)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * np.pi / 3
        x = center_x + side_length * np.cos(angle)
        y = center_y + side_length * np.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

def create_hexagon_polygon(center_x, center_y, side_length, angle_degrees):
    """Create a shapely polygon representation of a hexagon."""
    vertices = generate_hexagon_vertices(center_x, center_y, side_length, angle_degrees)
    return Polygon(vertices)

def check_overlap(hex1, hex2):
    """Check if two hexagons overlap using shapely."""
    poly1 = create_hexagon_polygon(hex1[0], hex1[1], 1, hex1[2])
    poly2 = create_hexagon_polygon(hex2[0], hex2[1], 1, hex2[2])
    return poly1.intersects(poly2)

def check_containment(hexagon, outer_hexagon_center, outer_hexagon_side_length):
    """Check if a hexagon is fully contained within the outer hexagon."""
    hex_poly = create_hexagon_polygon(hexagon[0], hexagon[1], 1, hexagon[2])
    outer_poly = create_hexagon_polygon(outer_hexagon_center[0], outer_hexagon_center[1], outer_hexagon_side_length, outer_hexagon_center[2])
    return outer_poly.contains(hex_poly)

def compute_outer_hexagon_radius(inner_hex_data, outer_hexagon_center, tolerance=1e-6):
    """Compute minimum outer hexagon radius that contains all inner hexagons."""
    # Start with a conservative estimate
    min_radius = 0.0
    
    # Find maximum distance from center to any hexagon vertex
    max_dist = 0.0
    for i in range(len(inner_hex_data)):
        hex_center_x, hex_center_y, _ = inner_hex_data[i]
        # Get all vertices of the hexagon
        vertices = generate_hexagon_vertices(hex_center_x, hex_center_y, 1, 0)
        # Calculate distances from center to all vertices
        for vx, vy in vertices:
            dist = np.sqrt((vx - outer_hexagon_center[0])**2 + (vy - outer_hexagon_center[1])**2)
            max_dist = max(max_dist, dist)
    
    # Add a safety margin
    max_dist += 1.0
    
    # Binary search for precise boundary
    max_radius = max_dist
    prev_radius = 0.0
    
    while abs(max_radius - min_radius) > tolerance:
        mid_radius = (max_radius + min_radius) / 2.0
        # Check if all hexagons fit within this radius
        fits = True
        for i in range(len(inner_hex_data)):
            if not check_containment(inner_hex_data[i], [outer_hexagon_center[0], outer_hexagon_center[1], outer_hexagon_center[2]], mid_radius):
                fits = False
                break
        
        if fits:
            max_radius = mid_radius
        else:
            min_radius = mid_radius
            
        # Prevent infinite loops
        if abs(max_radius - min_radius) < tolerance:
            break
            
    return (max_radius + min_radius) / 2.0

def evaluate_fitness(individual, outer_hex_center=[0, 0, 0]):
    """Evaluate the fitness of an individual (lower is better)."""
    # Convert individual to hexagon data format
    hex_data = individual.reshape(-1, 3)
    
    # Check for overlaps
    num_overlaps = 0
    for i in range(len(hex_data)):
        for j in range(i+1, len(hex_data)):
            if check_overlap(hex_data[i], hex_data[j]):
                num_overlaps += 1
    
    if num_overlaps > 0:
        # Penalize overlapping configurations heavily
        return 1e10 + num_overlaps * 1e6
    
    # Compute the minimal outer radius that contains all hexagons
    try:
        outer_radius = compute_outer_hexagon_radius(hex_data, outer_hex_center)
        # Return inverse of radius (we want to maximize 1/radius, i.e., minimize radius)
        return 1.0 / outer_radius if outer_radius > 0 else 1e10
    except Exception:
        return 1e10

def create_random_individual(n_hexagons=11):
    """Create a random individual with proper hexagon arrangements."""
    # Start with a diversified initial arrangement
    individual = []
    
    # Center hexagon
    individual.append([0.0, 0.0, 0.0])
    
    # Place others in a ring pattern around the center
    angles = np.linspace(0, 2*np.pi, 10, endpoint=False)
    distances = [2.5, 3.5, 4.5]  # Multiple rings
    
    for i in range(1, min(11, len(distances) * 3 + 1)):
        if i >= len(distances) * 3 + 1:
            break
        # Distribute evenly among rings
        ring_idx = (i - 1) // 3
        pos_in_ring = (i - 1) % 3
        if ring_idx < len(distances):
            angle = angles[pos_in_ring] + np.random.uniform(-0.2, 0.2)
            dist = distances[ring_idx] + np.random.uniform(-0.5, 0.5)
            x = dist * np.cos(angle)
            y = dist * np.sin(angle)
            individual.append([x, y, np.random.uniform(0, 360)])
    
    # Fill up to 11 hexagons with random positions
    while len(individual) < 11:
        individual.append([
            np.random.uniform(-6, 6),
            np.random.uniform(-6, 6),
            np.random.uniform(0, 360)
        ])
        
    return np.array(individual).flatten()

def local_optimization(individual, outer_hex_center=[0, 0, 0], max_iter=50):
    """Apply local optimization to improve individual."""
    def objective(x_flat):
        # Reshape back to individual format
        individual_copy = x_flat.reshape(-1, 3)
        return evaluate_fitness(individual_copy, outer_hex_center)
    
    # Optimize using L-BFGS-B
    result = minimize(objective, individual, method='L-BFGS-B', options={'maxiter': max_iter})
    return result.x

def crossover(parent1, parent2):
    """Perform simulated binary crossover."""
    # SBX crossover with eta=15
    eta = 15.0
    alpha = 0.5
    
    # Create offspring
    child1 = np.zeros_like(parent1)
    child2 = np.zeros_like(parent2)
    
    for i in range(len(parent1)):
        if np.random.rand() < 0.5:
            # No crossover for this gene
            child1[i] = parent1[i]
            child2[i] = parent2[i]
        else:
            u = np.random.rand()
            if u <= 0.5:
                beta = (2*u)**(1/(eta+1))
            else:
                beta = (1/(2*(1-u)))**(1/(eta+1))
                
            child1[i] = 0.5*((parent1[i] + parent2[i]) - beta*(parent1[i] - parent2[i]))
            child2[i] = 0.5*((parent1[i] + parent2[i]) + beta*(parent1[i] - parent2[i]))
    
    return child1, child2

def mutate(individual, mutation_rate=0.1, gen_num=None, total_gens=None):
    """Apply mutation to an individual."""
    mutated = individual.copy()
    
    # Adaptive mutation rate
    if gen_num is not None and total_gens is not None:
        # Decrease mutation rate over time
        decay_factor = 1.0 - (gen_num / total_gens)
        actual_mutation_rate = mutation_rate * decay_factor
    else:
        actual_mutation_rate = mutation_rate
    
    for i in range(len(mutated)):
        if np.random.rand() < actual_mutation_rate:
            if i % 3 == 0:  # x coordinate
                mutated[i] += np.random.normal(0, 0.5)
            elif i % 3 == 1:  # y coordinate
                mutated[i] += np.random.normal(0, 0.5)
            else:  # angle
                mutated[i] += np.random.normal(0, 10)
                mutated[i] = mutated[i] % 360
    
    return mutated

def tournament_selection(population, fitnesses, tournament_size=3):
    """Select an individual using tournament selection."""
    selected_indices = np.random.choice(len(population), size=tournament_size, replace=False)
    best_idx = selected_indices[np.argmin([fitnesses[i] for i in selected_indices])]
    return population[best_idx].copy()

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Parameters
    population_size = 50
    num_generations = 100
    mutation_rate = 0.1
    elite_size = 5
    
    # Initialize population
    population = [create_random_individual() for _ in range(population_size)]
    
    # Track the best individual
    best_fitness = float('inf')
    best_individual = None
    
    # Evolutionary algorithm
    for gen in range(num_generations):
        # Evaluate fitness of all individuals
        fitnesses = []
        for ind in population:
            fitness = evaluate_fitness(ind.reshape(-1, 3))
            fitnesses.append(fitness)
            
            if fitness < best_fitness:
                best_fitness = fitness
                best_individual = ind.copy()
                
        # Apply local optimization to best individuals
        for i in range(elite_size):
            idx = np.argsort(fitnesses)[i]
            optimized = local_optimization(population[idx])
            population[idx] = optimized
            
        # Selection, crossover, and mutation
        new_population = []
        
        # Elitism: keep best individuals
        elite_indices = np.argsort(fitnesses)[:elite_size]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
            
        # Generate rest of population
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutation
            child1 = mutate(child1, mutation_rate, gen, num_generations)
            child2 = mutate(child2, mutation_rate, gen, num_generations)
            
            new_population.extend([child1, child2])
            
        population = new_population[:population_size]
        
        # Print progress
        if gen % 20 == 0:
            print(f"Generation {gen}: Best fitness = {best_fitness}")
    
    # Final optimization of best individual
    if best_individual is not None:
        final_individual = local_optimization(best_individual)
        best_individual = final_individual
    
    # Convert best individual to correct format
    inner_hex_data = best_individual.reshape(-1, 3)
    
    # Compute final outer hexagon dimensions
    outer_hex_center = [0, 0, 0]
    outer_hex_side_length = compute_outer_hexagon_radius(inner_hex_data, outer_hex_center)
    
    # Ensure outer hexagon is properly centered
    outer_hex_data = np.array(outer_hex_center)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
