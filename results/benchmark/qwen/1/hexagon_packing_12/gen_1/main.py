# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
import random
import math
from numba import jit

@jit(nopython=True)
def hexagon_vertices(x, y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, angle, and side length"""
    vertices = np.zeros((6, 2))
    angle_rad = np.radians(angle_deg)
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i, 0] = x + side_length * np.cos(theta)
        vertices[i, 1] = y + side_length * np.sin(theta)
    return vertices

@jit(nopython=True)
def distance_point_to_polygon(point, polygon_vertices):
    """Calculate minimum distance from point to polygon edges"""
    min_dist = float('inf')
    px, py = point
    
    for i in range(len(polygon_vertices)):
        x1, y1 = polygon_vertices[i]
        x2, y2 = polygon_vertices[(i + 1) % len(polygon_vertices)]
        
        # Distance from point to line segment
        A = px - x1
        B = py - y1
        C = x2 - x1
        D = y2 - y1
        
        dot = A * C + B * D
        len_sq = C * C + D * D
        param = -1
        if len_sq != 0:
            param = dot / len_sq
            
        if param < 0:
            xx, yy = x1, y1
        elif param > 1:
            xx, yy = x2, y2
        else:
            xx = x1 + param * C
            yy = y1 + param * D
            
        dx = px - xx
        dy = py - yy
        dist = np.sqrt(dx * dx + dy * dy)
        if dist < min_dist:
            min_dist = dist
            
    return min_dist

def get_outer_hexagon_vertices(side_length):
    """Get vertices of outer hexagon centered at origin"""
    vertices = []
    for i in range(6):
        angle = i * np.pi / 3
        vertices.append([side_length * np.cos(angle), side_length * np.sin(angle)])
    return np.array(vertices)

def evaluate_individual(individual, outer_side_length):
    """Evaluate fitness of an individual configuration"""
    # Convert individual to hexagon data
    hexagons = []
    for i in range(12):
        x, y, angle = individual[i*3:(i+1)*3]
        hexagons.append(hexagon_vertices(x, y, angle))
    
    # Check containment constraints
    outer_vertices = get_outer_hexagon_vertices(outer_side_length)
    total_penalty = 0
    
    for hexagon in hexagons:
        # Check if any vertex is outside outer hexagon
        for vertex in hexagon:
            dist = distance_point_to_polygon(vertex, outer_vertices)
            if dist < 0:
                total_penalty += abs(dist) * 1000
                
    # Check overlap penalties
    for i in range(12):
        for j in range(i+1, 12):
            # Simple distance-based penalty (simplified overlap check)
            center_i = np.mean(hexagons[i], axis=0)
            center_j = np.mean(hexagons[j], axis=0)
            dist = np.linalg.norm(center_i - center_j)
            if dist < 1.9:  # If centers too close, likely overlapping
                overlap_penalty = (1.9 - dist) * 100
                total_penalty += overlap_penalty
                
    # Fitness is inverse outer side length minus penalties
    fitness = 1.0 / outer_side_length - total_penalty * 1e-6
    return max(0, fitness)

def generate_initial_population(pop_size, outer_side_length):
    """Generate initial population with symmetric arrangements"""
    population = []
    for _ in range(pop_size):
        individual = []
        # Generate hexagon positions with some symmetry
        for i in range(12):
            # Create radial symmetry pattern
            angle = 30 * i  # Every 30 degrees
            radius = 0.5 + random.random() * 2.5  # Random distance from center
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            angle_deg = random.random() * 360
            
            individual.extend([x, y, angle_deg])
            
        population.append(individual)
    return population

def mutate_individual(individual, mutation_rate=0.1):
    """Mutate an individual with careful handling of symmetry constraints"""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            if i % 3 == 0:  # x coordinate
                mutated[i] += random.uniform(-0.5, 0.5)
            elif i % 3 == 1:  # y coordinate  
                mutated[i] += random.uniform(-0.5, 0.5)
            else:  # angle
                mutated[i] += random.uniform(-30, 30)
                mutated[i] = mutated[i] % 360
    return mutated

def crossover(parent1, parent2):
    """Perform crossover between two parents"""
    child = []
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    return child

def optimize_hexagon_packing():
    """Main optimization routine"""
    pop_size = 50
    generations = 100
    mutation_rate = 0.1
    
    # Start with good symmetric baseline
    best_individual = []
    for i in range(12):
        angle = 30 * i
        radius = 1.5
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        angle_deg = 0
        best_individual.extend([x, y, angle_deg])
    
    best_fitness = 0
    best_outer_side = 100  # Large initial value
    
    # Evolution loop
    for gen in range(generations):
        # Evaluate population
        fitness_scores = []
        for ind in [best_individual]:
            fitness = evaluate_individual(ind, best_outer_side)
            fitness_scores.append(fitness)
        
        # Simple hill climbing step
        # Try to reduce outer hexagon size if possible
        test_side = best_outer_side - 0.05
        if test_side > 0:
            test_fitness = evaluate_individual(best_individual, test_side)
            if test_fitness > best_fitness:
                best_outer_side = test_side
                best_fitness = test_fitness
        
        # Adaptive mutation rate
        if gen > 20 and gen % 10 == 0:
            mutation_rate = max(0.01, mutation_rate * 0.9)
            
        # Occasionally add diversity
        if random.random() < 0.1:
            best_individual = mutate_individual(best_individual, mutation_rate)
    
    return best_individual, best_outer_side

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Run optimization
    individual, outer_side_length = optimize_hexagon_packing()
    
    # Convert individual to required format
    inner_hex_data = np.array(individual).reshape(12, 3)
    
    # Outer hexagon at center (0,0) with no rotation
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END
