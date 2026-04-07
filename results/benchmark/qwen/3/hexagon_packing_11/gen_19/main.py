# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
import time
from itertools import combinations
import random

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
    try:
        inner_polygon = Polygon(hex_vertices)
        outer_polygon = Polygon(outer_hex_vertices)
        return outer_polygon.contains(inner_polygon)
    except:
        return False

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely"""
    try:
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)
    except:
        return True

def evaluate_fitness(individual, outer_radius):
    """Evaluate fitness of individual solution"""
    # Decode individual into positions and rotations
    positions = individual[:22].reshape(-1, 2)
    rotations = individual[22:]
    
    # Create outer hexagon vertices
    outer_vertices = hexagon_vertices(0, 0, 0, outer_radius)
    
    # Initialize penalty
    penalty = 0.0
    
    # Check containment for all inner hexagons
    for i, (pos, rot) in enumerate(zip(positions, rotations)):
        hex_vertices = hexagon_vertices(pos[0], pos[1], rot)
        if not check_containment(hex_vertices, outer_vertices):
            penalty += 10000.0
    
    # Check overlaps between all pairs of inner hexagons
    for i, j in combinations(range(11), 2):
        hex1_vertices = hexagon_vertices(positions[i][0], positions[i][1], rotations[i])
        hex2_vertices = hexagon_vertices(positions[j][0], positions[j][1], rotations[j])
        if check_overlap(hex1_vertices, hex2_vertices):
            penalty += 10000.0
    
    # If any constraint violated, heavily penalize
    if penalty > 0:
        return penalty
    
    # Otherwise, reward denser packing (smaller outer radius)
    # Normalize by a reasonable scale factor to avoid numerical issues
    return 1.0 / max(outer_radius, 1e-6)

def create_individual():
    """Create a random individual for the population"""
    # Positions: 11 hexagons, 2 coordinates each
    positions = np.random.uniform(-5, 5, (11, 2))
    # Rotations: 11 hexagons, each with angle between 0 and 360
    rotations = np.random.uniform(0, 360, 11)
    # Combine into one array
    return np.concatenate([positions.flatten(), rotations])

def crossover(parent1, parent2):
    """Perform crossover between two parents"""
    # Uniform crossover for positions and rotations separately
    mask = np.random.rand(22) > 0.5  # 50% chance of inheriting from parent2
    child1_pos = np.where(mask, parent1[:22], parent2[:22])
    child2_pos = np.where(~mask, parent1[:22], parent2[:22])
    
    # For rotations, just take average with some noise
    child1_rot = (parent1[22:] + parent2[22:]) / 2
    child2_rot = (parent1[22:] + parent2[22:]) / 2
    
    # Add some mutation to rotations
    child1_rot += np.random.normal(0, 10, 11)
    child2_rot += np.random.normal(0, 10, 11)
    
    return np.concatenate([child1_pos, child1_rot]), np.concatenate([child2_pos, child2_rot])

def mutate(individual, mutation_rate=0.1):
    """Mutate an individual"""
    mutated = individual.copy()
    
    # Mutate positions
    for i in range(22):
        if np.random.random() < mutation_rate:
            mutated[i] += np.random.normal(0, 0.5)
    
    # Mutate rotations
    for i in range(22, 33):
        if np.random.random() < mutation_rate:
            mutated[i] += np.random.normal(0, 20)
            # Keep within 0-360 range
            mutated[i] = mutated[i] % 360
    
    return mutated

def create_seed_layout():
    """Create a good initial seed layout based on hexagonal packing principles"""
    positions = []
    rotations = []
    
    # Place hexagons in a structured pattern
    # Center hexagon
    positions.append([0, 0])
    rotations.append(0)
    
    # First ring around center
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    for angle in angles:
        x = 2.0 * np.cos(angle)
        y = 2.0 * np.sin(angle)
        positions.append([x, y])
        rotations.append(0)
    
    # Second ring
    # Alternate rows for better packing
    row1_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    row2_angles = np.linspace(np.pi/6, 2*np.pi + np.pi/6, 6, endpoint=False)
    
    # Row 1 (top/bottom)
    for angle in row1_angles:
        x = 3.0 * np.cos(angle)
        y = 3.0 * np.sin(angle)
        positions.append([x, y])
        rotations.append(0)
    
    # Row 2 (middle)
    for angle in row2_angles:
        x = 3.0 * np.cos(angle)
        y = 3.0 * np.sin(angle)
        positions.append([x, y])
        rotations.append(0)
    
    # Fill remaining positions with careful spacing
    # Place one more in top-left area
    positions.append([-2.5, 2.5])
    rotations.append(0)
    
    # Place one more in bottom-right area
    positions.append([2.5, -2.5])
    rotations.append(0)
    
    # Ensure we have exactly 11 positions
    positions = positions[:11]
    rotations = rotations[:11]
    
    return np.array(positions), np.array(rotations)

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
    
    # Initialize parameters for evolution
    population_size = 50
    generations = 100
    elite_size = 5
    mutation_rate = 0.1
    
    # Create initial seed layout
    seed_positions, seed_rotations = create_seed_layout()
    initial_individual = np.concatenate([seed_positions.flatten(), seed_rotations])
    
    # Initialize population with seed and random individuals
    population = [initial_individual]
    for _ in range(population_size - 1):
        population.append(create_individual())
    
    best_fitness = float('-inf')
    best_individual = None
    best_radius = float('inf')
    
    # Evolutionary loop
    for generation in range(generations):
        # Evaluate fitness for entire population
        fitness_scores = []
        for individual in population:
            # Try different radii for this individual
            test_radii = [4.0, 5.0, 6.0, 7.0, 8.0]
            best_local_fitness = float('-inf')
            best_local_radius = 8.0
            
            for radius in test_radii:
                try:
                    fitness_score = evaluate_fitness(individual, radius)
                    if fitness_score > best_local_fitness:
                        best_local_fitness = fitness_score
                        best_local_radius = radius
                except:
                    continue
            
            fitness_scores.append(best_local_fitness)
            
            # Track global best
            if best_local_fitness > best_fitness:
                best_fitness = best_local_fitness
                best_individual = individual.copy()
                best_radius = best_local_radius
        
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        
        # Keep elite
        elite = population[:elite_size]
        
        # Generate new population through selection, crossover, and mutation
        new_population = elite[:]
        
        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(len(elite), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            
            # Select two parents from elite
            parent1 = elite[winner_index]
            parent2 = elite[np.random.choice(len(elite))]
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutation
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            
            new_population.extend([child1, child2])
        
        population = new_population[:population_size]
        
        # Early stopping if no significant improvement
        if generation > 10 and abs(best_fitness - float('-inf')) < 1e-6:
            break
    
    # Final evaluation with best individual
    if best_individual is None:
        best_individual = initial_individual
    
    final_positions = best_individual[:22].reshape(-1, 2)
    final_rotations = best_individual[22:]
    
    # Correct final radius to be more accurate
    final_radius = best_radius
    
    # Create output arrays
    inner_hex_data = np.column_stack([
        final_positions,
        final_rotations
    ])
    
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = final_radius
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
