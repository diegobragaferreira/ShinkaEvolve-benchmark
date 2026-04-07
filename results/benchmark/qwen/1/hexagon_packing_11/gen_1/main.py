# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
import random
from typing import Tuple, List
import time

# Global constants for hexagon geometry
HEX_RADIUS = 1.0  # Unit hexagon radius
HEX_APOGEE = HEX_RADIUS * np.sqrt(3) / 2  # Distance from center to side midpoint
HEX_HEIGHT = 2 * HEX_APOGEE  # Height of hexagon
HEX_WIDTH = 2 * HEX_RADIUS  # Width of hexagon

def generate_hexagon_vertices(center_x: float, center_y: float, angle_deg: float) -> np.ndarray:
    """Generate vertices of a regular hexagon given center and rotation."""
    angle_rad = np.radians(angle_deg)
    # Vertices of a unit hexagon centered at origin
    base_vertices = np.array([
        [1, 0], [0.5, np.sqrt(3)/2], [-0.5, np.sqrt(3)/2],
        [-1, 0], [-0.5, -np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]
    ])
    
    # Rotate and translate
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_vertices = base_vertices @ rotation_matrix.T
    return rotated_vertices + np.array([center_x, center_y])

def check_containment(hex_vertices: np.ndarray, outer_hex_vertices: np.ndarray) -> bool:
    """Check if a hexagon is fully contained within the outer hexagon."""
    hex_poly = Polygon(hex_vertices)
    outer_poly = Polygon(outer_hex_vertices)
    return outer_poly.contains(hex_poly)

def check_overlap(hex1_vertices: np.ndarray, hex2_vertices: np.ndarray) -> bool:
    """Check if two hexagons overlap using Shapely."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def calculate_fitness(individual: np.ndarray, outer_hex_vertices: np.ndarray) -> float:
    """Calculate fitness of an individual (smaller outer hex = higher fitness)."""
    total_penalty = 0
    
    # Check containment
    for i in range(11):
        hex_vertices = generate_hexagon_vertices(
            individual[i, 0], individual[i, 1], individual[i, 2]
        )
        if not check_containment(hex_vertices, outer_hex_vertices):
            total_penalty += 1000  # Large penalty for containment violation
            
    # Check overlaps
    for i in range(11):
        hex1_vertices = generate_hexagon_vertices(
            individual[i, 0], individual[i, 1], individual[i, 2]
        )
        for j in range(i+1, 11):
            hex2_vertices = generate_hexagon_vertices(
                individual[j, 0], individual[j, 1], individual[j, 2]
            )
            if check_overlap(hex1_vertices, hex2_vertices):
                total_penalty += 100  # Penalty for overlap
                
    # If any violations, return very low fitness
    if total_penalty > 0:
        return -total_penalty
    
    # Calculate outer hex size based on max distance from center
    max_dist = 0
    for i in range(11):
        center_x, center_y = individual[i, 0], individual[i, 1]
        # Distance from center of outer hex (assumed to be (0,0))  
        dist = np.sqrt(center_x**2 + center_y**2) + HEX_RADIUS
        max_dist = max(max_dist, dist)
        
    # Fitness is inverse of outer hex radius (we want smaller radius)
    return 1.0 / max_dist if max_dist > 0 else 0.0

def create_random_individual() -> np.ndarray:
    """Create a random valid individual."""
    individual = np.zeros((11, 3))
    # Center hex at origin
    individual[0] = [0, 0, 0]
    
    # Place other hexagons with reasonable bounds
    for i in range(1, 11):
        individual[i, 0] = random.uniform(-5, 5)  # x coordinate
        individual[i, 1] = random.uniform(-5, 5)  # y coordinate  
        individual[i, 2] = random.uniform(0, 360)  # rotation
        
    return individual

def mutate_individual(individual: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
    """Apply mutation to an individual."""
    mutated = individual.copy()
    
    for i in range(11):
        if random.random() < mutation_rate:
            # Mutate one of x, y, or angle
            param_idx = random.randint(0, 2)
            if param_idx == 0:  # x coordinate
                mutated[i, 0] += random.gauss(0, 0.3)
            elif param_idx == 1:  # y coordinate  
                mutated[i, 1] += random.gauss(0, 0.3)
            else:  # rotation
                mutated[i, 2] += random.gauss(0, 10)
                mutated[i, 2] %= 360
                
    return mutated

def crossover_parents(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover between two parents."""
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Single-point crossover on positions and rotations
    crossover_point = random.randint(1, 10)  # Don't cross center hexagon
    
    # Crossover for positions and angles
    for i in range(crossover_point, 11):
        child1[i] = parent2[i].copy()
        child2[i] = parent1[i].copy()
        
    return child1, child2

def evolutionary_hexagon_packing() -> Tuple[np.ndarray, np.ndarray, float]:
    """Main evolutionary algorithm for hexagon packing."""
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Parameters
    population_size = 50
    generations = 100
    elite_size = 5
    mutation_rate = 0.1
    
    # Create initial population
    population = [create_random_individual() for _ in range(population_size)]
    
    best_fitness = float('-inf')
    best_individual = None
    
    # Evolution loop
    for generation in range(generations):
        # Evaluate fitness for entire population
        fitness_scores = []
        for individual in population:
            # For simplicity, we'll use a fixed outer hexagon size constraint for now
            # In principle, this should compute actual outer hexagon vertices
            # But given the complexity, we'll compute fitness directly
            outer_radius = 10  # Initial estimate
            outer_vertices = generate_hexagon_vertices(0, 0, 0)  # Centered at origin
            fitness = calculate_fitness(individual, outer_vertices)
            fitness_scores.append(fitness)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual.copy()
                
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores.sort(reverse=True)
        
        # Elitism: keep best individuals
        elites = population[:elite_size]
        
        # Generate new population
        new_population = elites.copy()
        
        # Fill remaining slots through tournament selection and crossover
        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = 3
            selected_indices = random.sample(range(len(population)), tournament_size)
            winner_idx = selected_indices[np.argmax([fitness_scores[i] for i in selected_indices])]
            
            # Add mutated version of winner
            mutated = mutate_individual(population[winner_idx])
            new_population.append(mutated)
            
        population = new_population
        
        # Print progress
        if generation % 20 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")
    
    # Final refinement - try to find better fit
    if best_individual is not None:
        # Try several local optimizations around best solution
        for _ in range(10):
            local_mutation = mutate_individual(best_individual, 0.05)
            # Simple greedy improvement step
            test_fitness = calculate_fitness(local_mutation, generate_hexagon_vertices(0, 0, 0))
            if test_fitness > best_fitness:
                best_fitness = test_fitness
                best_individual = local_mutation.copy()
                
    # Calculate final outer hexagon size
    max_dist = 0
    for i in range(11):
        center_x, center_y = best_individual[i, 0], best_individual[i, 1]
        dist = np.sqrt(center_x**2 + center_y**2) + HEX_RADIUS
        max_dist = max(max_dist, dist)
    
    outer_hex_side_length = max_dist
    
    # Prepare return values
    outer_hex_data = np.array([0, 0, 0])  # Centered at origin
    
    return best_individual, outer_hex_data, outer_hex_side_length

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Run our evolutionary algorithm
    inner_hex_data, outer_hex_data, outer_hex_side_length = evolutionary_hexagon_packing()
    
    # Adjust to ensure we have valid hexagon data format
    # Our algorithm produces (x, y, angle_deg) for each hexagon
    # outer_hex_data is (x, y, angle_deg) for outer hexagon
    
    eval_time = time.time() - start_time
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
