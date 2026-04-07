# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
from deap import base, creator, tools, algorithms
import random
import math

# Constants
UNIT_HEX_RADIUS = 1.0
MAX_EVAL_TIME = 180.0
POPULATION_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.8

class Hexagon:
    def __init__(self, center_x, center_y, rotation_deg):
        self.center_x = center_x
        self.center_y = center_y
        self.rotation_deg = rotation_deg
        self.vertices = self._compute_vertices()
        
    def _compute_vertices(self):
        # Compute vertices of regular hexagon with radius 1
        rad = math.radians(self.rotation_deg)
        vertices = []
        for i in range(6):
            angle = rad + i * math.pi / 3
            x = self.center_x + UNIT_HEX_RADIUS * math.cos(angle)
            y = self.center_y + UNIT_HEX_RADIUS * math.sin(angle)
            vertices.append((x, y))
        return vertices
    
    def get_polygon(self):
        return Polygon(self.vertices)

class HexagonPacker:
    def __init__(self, num_inner=11):
        self.num_inner = num_inner
        self.outer_hex_radius = None
        
    def compute_outer_radius(self, inner_hexagons):
        """Compute minimum radius of outer hexagon that contains all inner hexagons"""
        if not inner_hexagons:
            return 0
        all_vertices = []
        for hex_obj in inner_hexagons:
            all_vertices.extend(hex_obj.vertices)
        
        # Find bounding circle for all vertices
        if not all_vertices:
            return 0
            
        # Center at origin for simplicity
        center_x = sum(v[0] for v in all_vertices) / len(all_vertices)
        center_y = sum(v[1] for v in all_vertices) / len(all_vertices)
        
        max_dist_sq = 0
        for x, y in all_vertices:
            dist_sq = (x - center_x)**2 + (y - center_y)**2
            max_dist_sq = max(max_dist_sq, dist_sq)
            
        return math.sqrt(max_dist_sq) + UNIT_HEX_RADIUS
    
    def check_collision(self, hex1, hex2):
        """Check if two hexagons collide using Shapely"""
        poly1 = hex1.get_polygon()
        poly2 = hex2.get_polygon()
        return poly1.intersects(poly2)
    
    def check_containment(self, hexagon, outer_radius):
        """Check if hexagon is contained within outer hexagon"""
        # Create outer hexagon centered at origin
        outer_vertices = []
        for i in range(6):
            angle = i * math.pi / 3
            x = outer_radius * math.cos(angle)
            y = outer_radius * math.sin(angle)
            outer_vertices.append((x, y))
        
        outer_poly = Polygon(outer_vertices)
        inner_poly = hexagon.get_polygon()
        
        # Check if inner polygon is completely contained in outer
        return outer_poly.contains(inner_poly)
    
    def evaluate(self, individual):
        """Evaluate fitness of a given configuration"""
        try:
            # Decode individual into hexagon positions and rotations
            hexagons = []
            for i in range(self.num_inner):
                idx = i * 3
                x = individual[idx]
                y = individual[idx + 1]
                angle = individual[idx + 2]
                hexagons.append(Hexagon(x, y, angle))
            
            # Check collisions between all pairs
            collisions = 0
            for i in range(len(hexagons)):
                for j in range(i + 1, len(hexagons)):
                    if self.check_collision(hexagons[i], hexagons[j]):
                        collisions += 1
            
            if collisions > 0:
                return (float('inf'),)  # Invalid solution
            
            # Compute outer hexagon radius
            outer_radius = self.compute_outer_radius(hexagons)
            
            # Validate containment
            valid = True
            for hex_obj in hexagons:
                if not self.check_containment(hex_obj, outer_radius):
                    valid = False
                    break
                    
            if not valid:
                return (float('inf'),)  # Invalid solution
                
            # Return inverse of outer radius (maximize 1/outer_radius)
            return (1.0 / outer_radius,)
        except Exception as e:
            return (float('inf'),)  # Invalid solution due to error

def generate_individual():
    """Generate a random individual (positions and orientations for 11 hexagons)"""
    individual = []
    # Generate positions and rotations for 11 hexagons
    for _ in range(11):
        # Position within reasonable bounds (-10 to 10)
        individual.append(random.uniform(-10, 10))
        individual.append(random.uniform(-10, 10))
        # Rotation in degrees (0 to 360)
        individual.append(random.uniform(0, 360))
    return individual

def mutate_individual(individual):
    """Mutate an individual by slightly perturbing position/orientation"""
    for i in range(len(individual)):
        if random.random() < 0.1:  # 10% chance to mutate each gene
            if i % 3 == 0 or i % 3 == 1:  # position coordinates
                individual[i] += random.gauss(0, 0.5)
            else:  # rotation
                individual[i] += random.gauss(0, 10)
                individual[i] = individual[i] % 360
    return individual,

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Set up DEAP evolutionary algorithm
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", HexagonPacker().evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Initialize population
    pop = toolbox.population(n=POPULATION_SIZE)
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    # Evolutionary process
    for gen in range(GENERATIONS):
        if time.time() - start_time > MAX_EVAL_TIME * 0.9:
            break
            
        # Select next generation
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_RATE:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < MUTATION_RATE:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        # Replace population
        pop[:] = offspring
    
    # Get best individual
    best_ind = tools.selBest(pop, 1)[0]
    
    # Create hexagon objects from best solution
    packer = HexagonPacker(11)
    hexagons = []
    for i in range(11):
        idx = i * 3
        x = best_ind[idx]
        y = best_ind[idx + 1]
        angle = best_ind[idx + 2]
        hexagons.append(Hexagon(x, y, angle))
    
    # Calculate final outer radius
    outer_radius = packer.compute_outer_radius(hexagons)
    
    # Convert to required format
    inner_hex_data = np.array([
        [hexagons[i].center_x, hexagons[i].center_y, hexagons[i].rotation_deg]
        for i in range(11)
    ])
    
    outer_hex_data = np.array([0, 0, 0])  # Outer hexagon centered at origin
    outer_hex_side_length = outer_radius
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
