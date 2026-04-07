# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
import random
import time
from copy import deepcopy
from functools import lru_cache
from joblib import Parallel, delayed

class HexagonGeometry:
    """Handles all geometric computations related to hexagons"""
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
        """Generate vertices of a regular hexagon given center, angle, and side length"""
        angle_rad = np.radians(angle_deg)
        angles = np.linspace(0, 2*np.pi, 7) + angle_rad
        vertices = np.array([
            [center_x + side_length * np.cos(a), center_y + side_length * np.sin(a)]
            for a in angles
        ])
        return tuple(map(tuple, vertices))  # Convert to hashable format for caching

    @staticmethod
    def check_containment(hex_vertices, outer_hex_vertices):
        """Check if hexagon vertices are contained within outer hexagon using Shapely"""
        inner_polygon = Polygon(hex_vertices)
        outer_polygon = Polygon(outer_hex_vertices)
        return outer_polygon.contains(inner_polygon)

    @staticmethod
    def check_overlap(hex1_vertices, hex2_vertices):
        """Check if two hexagons overlap using Shapely"""
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)

class HexagonEvaluator:
    """Handles fitness evaluation and constraint checking"""
    
    def __init__(self):
        self.geometry = HexagonGeometry()
    
    def compute_outer_hexagon_radius(self, inner_positions, inner_angles, initial_radius_estimate=5.0):
        """Compute minimum outer hexagon radius that contains all inner hexagons with adaptive precision"""
        # Binary search for tightest fit with adaptive precision
        left = initial_radius_estimate
        right = 20.0
        best_radius = right

        # Use adaptive precision thresholds based on search space
        precision_thresholds = [1e-3, 1e-4, 1e-5, 1e-6]
        current_precision = precision_thresholds[0]
        
        # Early termination condition
        max_iterations = 100
        iteration = 0

        while right - left > current_precision and iteration < max_iterations:
            mid = (left + right) / 2.0
            outer_vertices = self.geometry.hexagon_vertices(0, 0, 0, mid)
            valid = True

            # Check all inner hexagons efficiently
            for i, (pos, angle) in enumerate(zip(inner_positions, inner_angles)):
                hex_vertices = self.geometry.hexagon_vertices(pos[0], pos[1], angle)
                if not self.geometry.check_containment(hex_vertices, outer_vertices):
                    valid = False
                    break

            if valid:
                best_radius = mid
                right = mid
            else:
                left = mid

            iteration += 1

            # Adaptively adjust precision based on remaining search space
            if iteration > 30 and (right - left) < 0.05:
                current_precision = precision_thresholds[min(len(precision_thresholds)-1, 1)]
            elif iteration > 60 and (right - left) < 0.005:
                current_precision = precision_thresholds[min(len(precision_thresholds)-1, 2)]
            elif iteration > 80 and (right - left) < 0.0005:
                current_precision = precision_thresholds[min(len(precision_thresholds)-1, 3)]

        return best_radius

    def evaluate_fitness_single(self, individual):
        """Evaluate fitness for a single individual - separated for parallelization"""
        inner_positions = individual[:, :2]
        inner_angles = individual[:, 2]

        # Create outer hexagon vertices
        outer_radius = self.compute_outer_hexagon_radius(tuple(map(tuple, inner_positions)), tuple(inner_angles))

        # Check all constraints
        total_penalty = 0

        # Check containment for all inner hexagons
        outer_vertices = self.geometry.hexagon_vertices(0, 0, 0, outer_radius)
        for i, (pos, angle) in enumerate(zip(inner_positions, inner_angles)):
            hex_vertices = self.geometry.hexagon_vertices(pos[0], pos[1], angle)
            if not self.geometry.check_containment(hex_vertices, outer_vertices):
                total_penalty += 10000  # Large penalty for containment violation

        # Check overlaps between all pairs of inner hexagons
        for i in range(len(inner_positions)):
            for j in range(i+1, len(inner_positions)):
                hex1_vertices = self.geometry.hexagon_vertices(inner_positions[i][0], inner_positions[i][1], inner_angles[i])
                hex2_vertices = self.geometry.hexagon_vertices(inner_positions[j][0], inner_positions[j][1], inner_angles[j])
                if self.geometry.check_overlap(hex1_vertices, hex2_vertices):
                    total_penalty += 10000  # Large penalty for overlap violation

        # Fitness is negative of the radius plus penalties
        # We want to minimize radius, so fitness = -radius
        fitness = -outer_radius - total_penalty

        return fitness, outer_radius

    def evaluate_fitness(self, inner_positions, inner_angles, max_radius=20.0):
        """Evaluate fitness: higher is better, maximize 1/radius"""
        # Create outer hexagon vertices
        outer_radius = self.compute_outer_hexagon_radius(tuple(map(tuple, inner_positions)), tuple(inner_angles))

        # Check all constraints
        total_penalty = 0

        # Check containment for all inner hexagons
        outer_vertices = self.geometry.hexagon_vertices(0, 0, 0, outer_radius)
        for i, (pos, angle) in enumerate(zip(inner_positions, inner_angles)):
            hex_vertices = self.geometry.hexagon_vertices(pos[0], pos[1], angle)
            if not self.geometry.check_containment(hex_vertices, outer_vertices):
                total_penalty += 10000  # Large penalty for containment violation

        # Check overlaps between all pairs of inner hexagons
        for i in range(len(inner_positions)):
            for j in range(i+1, len(inner_positions)):
                hex1_vertices = self.geometry.hexagon_vertices(inner_positions[i][0], inner_positions[i][1], inner_angles[i])
                hex2_vertices = self.geometry.hexagon_vertices(inner_positions[j][0], inner_positions[j][1], inner_angles[j])
                if self.geometry.check_overlap(hex1_vertices, hex2_vertices):
                    total_penalty += 10000  # Large penalty for overlap violation

        # Fitness is negative of the radius plus penalties
        # We want to minimize radius, so fitness = -radius
        fitness = -outer_radius - total_penalty

        return fitness, outer_radius

class EvolutionaryOptimizer:
    """Manages the evolutionary algorithm components"""
    
    def __init__(self):
        self.evaluator = HexagonEvaluator()
        self.geometry = HexagonGeometry()
        
    def mutate_individual(self, individual, mutation_rate=0.1, max_displacement=0.5):
        """Mutate individual with position and rotation changes"""
        mutated = deepcopy(individual)
        n = len(mutated)

        for i in range(n):
            # Mutate position
            if random.random() < mutation_rate:
                mutated[i][0] += random.uniform(-max_displacement, max_displacement)
                mutated[i][1] += random.uniform(-max_displacement, max_displacement)
            
            # Mutate rotation 
            if random.random() < mutation_rate:
                mutated[i][2] += random.uniform(-30, 30)
                mutated[i][2] = mutated[i][2] % 360
        
        return mutated

    def crossover(self, parent1, parent2, crossover_rate=0.8):
        """Single-point crossover for hexagon packing"""
        if random.random() > crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)
        
        # Create offspring by combining parent genes
        n = len(parent1)
        crossover_point = random.randint(1, n-1)
        
        child1 = np.vstack([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.vstack([parent2[:crossover_point], parent1[crossover_point:]])
        
        return child1, child2

    def create_random_individual(self):
        """Create a random valid individual with optimized initialization"""
        # Start with a well-structured initial configuration
        individual = np.array([
            [0, 0, 0],  # center
            [-2.5, 0, 0],  # left
            [2.5, 0, 0],  # right
            [-1.25, 2.17, 0],  # top-left
            [1.25, 2.17, 0],  # top-right
            [-1.25, -2.17, 0],  # bottom-left
            [1.25, -2.17, 0],  # bottom-right
            [-3.75, 2.17, 0],  # far top-left
            [3.75, 2.17, 0],  # far top-right
            [-3.75, -2.17, 0],  # far bottom-left
            [3.75, -2.17, 0],  # far bottom-right
        ])
        
        # Add more precise randomness to positions and rotations
        for i in range(len(individual)):
            individual[i][0] += random.uniform(-0.4, 0.4)
            individual[i][1] += random.uniform(-0.4, 0.4)
            individual[i][2] += random.uniform(-20, 20)
            individual[i][2] = individual[i][2] % 360
        
        return individual

    def initialize_population(self, population_size):
        """Initialize population with diverse individuals"""
        population = []
        for i in range(population_size):
            individual = self.create_random_individual()
            population.append(individual)
        return population

    def select_parents(self, population, fitnesses, tournament_size=3):
        """Tournament selection with diversity preservation"""
        selected = []
        for _ in range(len(population)):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            # Use probabilistic selection to maintain diversity
            winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
            selected.append(deepcopy(population[winner_idx]))
        return selected

    def local_optimization_step(self, individual, max_iter=100):
        """Enhanced local optimization with multi-stage refinement"""
        best_individual = deepcopy(individual)
        best_fitness, _ = self.evaluator.evaluate_fitness(best_individual[:, :2], best_individual[:, 2])
        
        # Stage 1: Coarse position and rotation adjustments
        for _ in range(max_iter // 3):
            mutated = deepcopy(best_individual)
            idx = random.randint(0, len(mutated)-1)
            mutated[idx][0] += random.uniform(-0.1, 0.1)
            mutated[idx][1] += random.uniform(-0.1, 0.1)
            mutated[idx][2] += random.uniform(-5, 5)
            mutated[idx][2] = mutated[idx][2] % 360
            
            mutated_fitness, _ = self.evaluator.evaluate_fitness(mutated[:, :2], mutated[:, 2])
            if mutated_fitness > best_fitness:
                best_individual = mutated
                best_fitness = mutated_fitness
        
        # Stage 2: Fine-tuning
        for _ in range(max_iter // 3):
            mutated = deepcopy(best_individual)
            idx = random.randint(0, len(mutated)-1)
            mutated[idx][0] += random.uniform(-0.05, 0.05)
            mutated[idx][1] += random.uniform(-0.05, 0.05)
            mutated[idx][2] += random.uniform(-2, 2)
            mutated[idx][2] = mutated[idx][2] % 360
            
            mutated_fitness, _ = self.evaluator.evaluate_fitness(mutated[:, :2], mutated[:, 2])
            if mutated_fitness > best_fitness:
                best_individual = mutated
                best_fitness = mutated_fitness
        
        # Stage 3: Very fine adjustments
        for _ in range(max_iter // 3):
            mutated = deepcopy(best_individual)
            idx = random.randint(0, len(mutated)-1)
            mutated[idx][0] += random.uniform(-0.01, 0.01)
            mutated[idx][1] += random.uniform(-0.01, 0.01)
            mutated[idx][2] += random.uniform(-0.5, 0.5)
            mutated[idx][2] = mutated[idx][2] % 360
            
            mutated_fitness, _ = self.evaluator.evaluate_fitness(mutated[:, :2], mutated[:, 2])
            if mutated_fitness > best_fitness:
                best_individual = mutated
                best_fitness = mutated_fitness

        return best_individual

    def run_evolution(self, population_size=60, generations=120, max_time_seconds=170):
        """Main evolutionary optimization loop"""
        start_time = time.time()
        
        # Initialize population
        population = self.initialize_population(population_size)

        best_fitness_history = []
        last_improvement_gen = 0

        for gen in range(generations):
            if time.time() - start_time > max_time_seconds:
                break
                
            # Adaptive mutation rate that decreases over generations
            mutation_rate = max(0.05, 0.2 * (1 - gen / generations))

            # Evaluate fitness for all individuals
            fitnesses = []
            for individual in population:
                fitness, _ = self.evaluator.evaluate_fitness(individual[:, :2], individual[:, 2])
                fitnesses.append(fitness)

            # Track best
            best_idx = np.argmax(fitnesses)
            best_fitness = fitnesses[best_idx]
            best_fitness_history.append(best_fitness)
            
            # Early stopping if no improvement in recent generations
            if len(best_fitness_history) > 10:
                recent_improvement = best_fitness - best_fitness_history[-10]
                if recent_improvement < 1e-6:
                    last_improvement_gen += 1
                else:
                    last_improvement_gen = 0
                    
            if last_improvement_gen > 15:  # Stop if no improvement for 15 generations
                break

            # Local optimization on best individual periodically
            if gen % 5 == 0:
                population[best_idx] = self.local_optimization_step(population[best_idx])

            # Elitism - keep best individuals
            elite_count = int(0.15 * population_size)
            elite_indices = np.argsort(fitnesses)[-elite_count:]
            elites = [deepcopy(population[i]) for i in elite_indices]

            # Selection
            parents = self.select_parents(population, fitnesses)

            # Crossover and mutation
            new_population = elites.copy()

            while len(new_population) < population_size:
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)

                child1, child2 = self.crossover(parent1, parent2)

                child1 = self.mutate_individual(child1, mutation_rate)
                child2 = self.mutate_individual(child2, mutation_rate)

                new_population.extend([child1, child2])

            # Trim to exact population size
            population = new_population[:population_size]

        # Final evaluation
        final_fitnesses = []
        for individual in population:
            fitness, _ = self.evaluator.evaluate_fitness(individual[:, :2], individual[:, 2])
            final_fitnesses.append(fitness)

        best_idx = np.argmax(final_fitnesses)
        best_individual = population[best_idx]

        # Final optimization with more intensive search
        best_individual = self.local_optimization_step(best_individual, max_iter=150)

        # Get final results
        final_fitness, outer_radius = self.evaluator.evaluate_fitness(best_individual[:, :2], best_individual[:, 2])

        return best_individual, outer_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Create optimizer instance
    optimizer = EvolutionaryOptimizer()
    
    # Run evolutionary optimization
    inner_hex_data, outer_hex_side_length = optimizer.run_evolution()

    # Format output as required
    outer_hex_data = np.array([0, 0, 0])  # centered at origin

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END