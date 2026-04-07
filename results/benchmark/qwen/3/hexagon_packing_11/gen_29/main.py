# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
import time
from deap import base, creator, tools, algorithms
import random
from copy import deepcopy

def hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, angle, and side length"""
    angle_rad = np.radians(angle_deg)
    angles = np.linspace(0, 2*np.pi, 7) + angle_rad
    vertices = np.array([
        [center_x + side_length * np.cos(a), center_y + side_length * np.sin(a)]
        for a in angles
    ])
    return vertices

def check_containment(hex_vertices, outer_hex_vertices):
    """Check if hexagon vertices are contained within outer hexagon using Shapely"""
    inner_polygon = Polygon(hex_vertices)
    outer_polygon = Polygon(outer_hex_vertices)
    return outer_polygon.contains(inner_polygon)

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely"""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def evaluate_individual(individual, num_hexes=11):
    """Evaluate fitness of an individual solution"""
    # Extract positions and angles
    positions = np.array(individual[:num_hexes*2]).reshape(-1, 2)
    angles = np.array(individual[num_hexes*2:num_hexes*3])
    outer_radius = individual[-1]
    
    # Create outer hexagon vertices
    outer_vertices = hexagon_vertices(0, 0, 0, outer_radius)
    
    # Check containment and overlap penalties
    penalty = 0
    
    # Check containment for all inner hexagons
    for i, (pos, angle) in enumerate(zip(positions, angles)):
        hex_vertices = hexagon_vertices(pos[0], pos[1], angle)
        if not check_containment(hex_vertices, outer_vertices):
            penalty += 10000  # Large penalty for containment violation
    
    # Check overlaps between all pairs of inner hexagons
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            hex1_vertices = hexagon_vertices(positions[i][0], positions[i][1], angles[i])
            hex2_vertices = hexagon_vertices(positions[j][0], positions[j][1], angles[j])
            if check_overlap(hex1_vertices, hex2_vertices):
                penalty += 10000  # Large penalty for overlap violation
    
    # Fitness is inverse of radius (to maximize 1/r)
    fitness = 1.0 / outer_radius if outer_radius > 0 else 0.0
    
    # Add penalty to fitness (lower fitness = worse solution)
    fitness -= penalty * 0.001
    
    return fitness

def create_individual(num_hexes=11):
    """Create a random individual with positions, angles, and outer radius"""
    # Random positions within a reasonable range
    positions = np.random.uniform(-5, 5, (num_hexes, 2))
    
    # Random angles between 0 and 360
    angles = np.random.uniform(0, 360, num_hexes)
    
    # Random outer radius
    outer_radius = np.random.uniform(3, 15)
    
    individual = np.concatenate([positions.flatten(), angles, [outer_radius]])
    return individual

def mutate_individual(individual, indpb=0.1):
    """Mutate an individual"""
    # Mutate positions
    for i in range(len(individual)-1):  # Exclude outer_radius
        if random.random() < indpb:
            if i < 22:  # Positions
                individual[i] += np.random.normal(0, 0.5)
            elif i < 33:  # Angles
                individual[i] = (individual[i] + np.random.normal(0, 15)) % 360
            else:  # Outer radius
                individual[i] = max(1, individual[i] + np.random.normal(0, 0.5))
    
    return individual,

def initialize_population(pop_size=50, num_hexes=11):
    """Initialize population with diverse starting points"""
    population = []
    
    # Strategy 1: Hexagonal arrangement
    positions = []
    angles = []
    
    # Center hexagon
    positions.append([0, 0])
    angles.append(0)
    
    # Ring 1
    for i in range(6):
        angle = i * 60
        x = 2.5 * np.cos(np.radians(angle))
        y = 2.5 * np.sin(np.radians(angle))
        positions.append([x, y])
        angles.append(0)
    
    # Ring 2
    for i in range(6):
        angle = i * 60 + 30
        x = 4.33 * np.cos(np.radians(angle))
        y = 4.33 * np.sin(np.radians(angle))
        positions.append([x, y])
        angles.append(0)
    
    # Fill remaining positions if needed
    while len(positions) < num_hexes:
        positions.append([np.random.uniform(-5, 5), np.random.uniform(-5, 5)])
        angles.append(np.random.uniform(0, 360))
    
    # Create first individual from hexagonal arrangement
    first_individual = np.concatenate([np.array(positions)[:num_hexes].flatten(), 
                                       np.array(angles)[:num_hexes], 
                                       [8.0]])
    population.append(first_individual)
    
    # Add random individuals
    for _ in range(pop_size - 1):
        population.append(create_individual(num_hexes))
    
    return population

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Algorithm parameters
    POP_SIZE = 50  # Increased population size
    NGEN = 200     # More generations
    MUTPB = 0.2    # Mutation probability
    CXPB = 0.5     # Crossover probability
    TOURNAMENT_SIZE = 3
    
    # Initialize DEAP framework
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", initialize_population)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", mutate_individual, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
    
    # Create initial population
    pop = toolbox.population()
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = (fit,)
    
    # Main evolutionary loop
    best_fitness = float('-inf')
    no_improve_count = 0
    max_no_improve = 20  # Early stopping condition
    
    for gen in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)
        
        # Replace the old population with the new generation
        pop[:] = offspring
        
        # Track best solution
        best_ind = tools.selBest(pop, 1)[0]
        current_best_fitness = best_ind.fitness.values[0]
        
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Early stopping if no improvement
        if no_improve_count >= max_no_improve:
            break
    
    # Get the best individual
    best_individual = tools.selBest(pop, 1)[0]
    
    # Extract results
    positions = np.array(best_individual[:22]).reshape(-1, 2)
    angles = np.array(best_individual[22:33])
    outer_radius = best_individual[-1]
    
    # Create final data structures
    inner_hex_data = np.column_stack([
        positions,
        angles
    ])
    
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = outer_radius
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
