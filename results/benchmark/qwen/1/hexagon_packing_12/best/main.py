# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
import math
from scipy.spatial.distance import cdist
import random

class HexagonPackingGA:
    def __init__(self, population_size=50, generations=200, mutation_rate=0.1, 
                 crossover_rate=0.8, elite_size=5):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        
    def generate_hexagon_vertices(self, center_x, center_y, rotation_deg, side_length=1):
        """Generate vertices of a regular hexagon with given center, rotation, and side length."""
        vertices = []
        rotation_rad = math.radians(rotation_deg)
        for i in range(6):
            angle = rotation_rad + i * math.pi / 3
            x = center_x + side_length * math.cos(angle)
            y = center_y + side_length * math.sin(angle)
            vertices.append((x, y))
        return vertices
    
    def check_hexagon_containment(self, hexagon_vertices, outer_hex_center, outer_hex_rotation, outer_hex_side_length):
        """Check if all vertices of a hexagon are contained within the outer hexagon."""
        outer_vertices = self.generate_hexagon_vertices(outer_hex_center[0], outer_hex_center[1], outer_hex_rotation, outer_hex_side_length)
        outer_polygon = Polygon(outer_vertices)

        for vertex in hexagon_vertices:
            point = Point(vertex[0], vertex[1])
            if not outer_polygon.contains(point):
                return False
        return True
    
    def check_hexagon_overlap(self, hex1_vertices, hex2_vertices):
        """Check if two hexagons overlap."""
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)
    
    def calculate_fitness(self, individual):
        """Calculate fitness for an individual (higher is better)"""
        # Extract parameters
        params = individual[:-1]  # Last element is the outer radius
        outer_radius = individual[-1]
        
        # Create list of inner hexagon data
        inner_hex_data = []
        for i in range(12):
            idx = i * 3
            x, y, theta = params[idx], params[idx+1], params[idx+2]
            inner_hex_data.append([x, y, theta])

        # Generate all inner hexagon vertices
        inner_hex_vertices = []
        for x, y, theta in inner_hex_data:
            vertices = self.generate_hexagon_vertices(x, y, theta)
            inner_hex_vertices.append(vertices)

        # Check containment and overlap constraints
        outer_hex_center = [0, 0]  # Centered at origin
        outer_hex_rotation = 0

        # Check containment
        for vertices in inner_hex_vertices:
            if not self.check_hexagon_containment(vertices, outer_hex_center, outer_hex_rotation, outer_radius):
                return -1e6  # Penalty for containment violation

        # Check overlaps between all pairs
        for i in range(12):
            for j in range(i+1, 12):
                if self.check_hexagon_overlap(inner_hex_vertices[i], inner_hex_vertices[j]):
                    return -1e6  # Penalty for overlap violation

        # Return inverse of outer radius (we want to maximize 1/R)
        return 1.0 / outer_radius
    
    def create_individual(self):
        """Create a random individual"""
        individual = []
        # Generate 12 hexagons with random positions and rotations
        for _ in range(12):
            # x, y coordinates within reasonable bounds
            x = random.uniform(-5, 5)
            y = random.uniform(-5, 5)
            # rotation angle
            theta = random.uniform(-180, 180)
            individual.extend([x, y, theta])
        
        # Outer radius (should be reasonable)
        outer_radius = random.uniform(3.0, 12.0)
        individual.append(outer_radius)
        
        return np.array(individual)
    
    def initialize_population(self):
        """Initialize the population"""
        return [self.create_individual() for _ in range(self.population_size)]
    
    def selection(self, population, fitnesses):
        """Tournament selection"""
        selected = []
        tournament_size = 3
        
        for _ in range(self.population_size):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
            selected.append(population[winner_index].copy())
            
        return selected
    
    def crossover(self, parent1, parent2):
        """Custom crossover for hexagon packing - preserves hexagon relationships"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Create offspring using uniform crossover but maintain geometric structure
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Crossover for hexagon parameters (x, y, theta) with special handling for symmetry
        for i in range(12):
            # For every third parameter (position/rotation), swap with some probability
            if random.random() < 0.5:
                # Swap x values
                child1[i*3], child2[i*3] = child2[i*3], child1[i*3]
                # Swap y values  
                child1[i*3+1], child2[i*3+1] = child2[i*3+1], child1[i*3+1]
                # Swap rotation values
                child1[i*3+2], child2[i*3+2] = child2[i*3+2], child1[i*3+2]
                
        # Crossover for outer radius (simple average)
        child1[-1] = (child1[-1] + child2[-1]) / 2.0
        child2[-1] = (child1[-1] + child2[-1]) / 2.0
        
        return child1, child2
    
    def mutate(self, individual):
        """Mutation with adaptive rate"""
        mutated = individual.copy()
        
        # Apply mutation to each parameter
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                if i < 36:  # hexagon parameters (x, y, theta)
                    param_idx = i % 3
                    if param_idx == 0:  # x position
                        mutated[i] += random.uniform(-0.5, 0.5)
                    elif param_idx == 1:  # y position
                        mutated[i] += random.uniform(-0.5, 0.5)
                    else:  # rotation angle
                        mutated[i] += random.uniform(-30, 30)
                else:  # outer radius
                    mutated[i] += random.uniform(-0.5, 0.5)
        
        return mutated
    
    def optimize(self):
        """Main optimization loop"""
        # Initialize population
        population = self.initialize_population()
        
        best_fitness_history = []
        
        for generation in range(self.generations):
            # Calculate fitness for entire population
            fitnesses = [self.calculate_fitness(ind) for ind in population]
            
            # Track best fitness
            best_fitness = max(fitnesses)
            best_fitness_history.append(best_fitness)
            
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitnesses)[-self.elite_size:]
            elite = [population[i] for i in elite_indices]
            
            # Selection
            selected = self.selection(population, fitnesses)
            
            # Create new population through crossover and mutation
            new_population = elite.copy()
            
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(selected, 2)
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Trim to exact population size
            population = new_population[:self.population_size]
            
            # Adaptive mutation rate based on convergence
            if generation > 10 and len(set([round(f, 6) for f in best_fitness_history[-10:]])) == 1:
                self.mutation_rate = min(0.3, self.mutation_rate * 1.1)
            else:
                self.mutation_rate = max(0.01, self.mutation_rate * 0.99)
        
        # Return best individual
        final_fitnesses = [self.calculate_fitness(ind) for ind in population]
        best_index = np.argmax(final_fitnesses)
        return population[best_index]

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a hybrid evolutionary algorithm.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Initialize and run the genetic algorithm
    ga = HexagonPackingGA(
        population_size=60,
        generations=150,
        mutation_rate=0.15,
        crossover_rate=0.8,
        elite_size=8
    )
    
    # Run optimization
    best_individual = ga.optimize()
    
    # Extract inner hexagon data
    inner_hex_data = []
    for i in range(12):
        idx = i * 3
        x, y, theta = best_individual[idx], best_individual[idx+1], best_individual[idx+2]
        inner_hex_data.append([x, y, theta])
    
    inner_hex_data = np.array(inner_hex_data)
    
    # Extract outer hexagon data
    outer_radius = best_individual[-1]
    outer_hex_data = np.array([0, 0, 0])  # Centered at origin, no rotation

    return inner_hex_data, outer_hex_data, outer_radius

# EVOLVE-BLOCK-END