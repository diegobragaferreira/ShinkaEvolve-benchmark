# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, cKDTree
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import math
from collections import defaultdict

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

class CircleInitializer:
    """Handles all circle initialization strategies"""
    
    @staticmethod
    def voronoi_seed_generation(n_points: int, boundary_margin: float = 0.1) -> np.ndarray:
        """Generate well-distributed seed points using Voronoi diagram."""
        points = np.random.rand(n_points, 2) * (1 - 2*boundary_margin) + boundary_margin
        return points

    @staticmethod
    def voronoi_radius_computation(points: np.ndarray, boundary_margin: float = 0.1) -> np.ndarray:
        """Compute radii based on Voronoi cell areas."""
        try:
            vor = Voronoi(points)
        except:
            # Fallback if Voronoi fails
            return np.array([0.05] * len(points))
            
        radii = []
        for i, point in enumerate(points):
            try:
                distances = cdist([point], np.delete(points, i, axis=0))[0]
                min_distance = np.min(distances)
                max_radius = min(point[0], 1-point[0], point[1], 1-point[1])
                estimated_radius = min(min_distance/2.0, max_radius)
                radii.append(max(estimated_radius, 0.001))
            except:
                radii.append(0.05)  # Fallback radius
        return np.array(radii)

    @staticmethod
    def structured_placement(n_circles: int) -> np.ndarray:
        """Initialize circles with structured grid placement."""
        individual = np.zeros((n_circles, 3))
        rows = int(np.ceil(np.sqrt(n_circles)))
        cols = int(np.ceil(n_circles / rows))
        
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        for i in range(n_circles):
            row = i // cols
            col = i % cols
            
            base_x = (col + 1) * spacing_x
            base_y = (row + 1) * spacing_y
            
            individual[i, 0] = np.clip(base_x + np.random.uniform(-spacing_x/4, spacing_x/4), 0.01, 0.99)
            individual[i, 1] = np.clip(base_y + np.random.uniform(-spacing_y/4, spacing_y/4), 0.01, 0.99)
            
            max_radius = min(0.5 - individual[i, 0], 0.5 - individual[i, 1],
                           individual[i, 0], individual[i, 1])
            individual[i, 2] = np.random.uniform(0.001, max_radius * 0.8)
        
        return individual

    @staticmethod
    def mixed_initialization(n_circles: int) -> np.ndarray:
        """Create one population member with mixed initialization strategy."""
        # Alternate between Voronoi-based and structured initialization for diversity
        if random.random() < 0.5:
            # Voronoi-based initialization
            seed_points = CircleInitializer.voronoi_seed_generation(n_circles)
            radii = CircleInitializer.voronoi_radius_computation(seed_points)
            individual = np.zeros((n_circles, 3))
            for i in range(n_circles):
                individual[i] = [seed_points[i][0], seed_points[i][1], radii[i]]
        else:
            # Structured initialization
            individual = CircleInitializer.structured_placement(n_circles)
        
        return individual

class CircleValidator:
    """Handles all constraint validation and penalty calculation"""
    
    @staticmethod
    def check_containment_constraints(individual: np.ndarray) -> float:
        """Check containment constraints and return penalty"""
        penalty = 0.0
        for i in range(len(individual)):
            x, y, r = individual[i]
            # Calculate boundary violations
            left_violation = max(0, r - x)
            right_violation = max(0, x + r - 1)
            bottom_violation = max(0, r - y)
            top_violation = max(0, y + r - 1)

            if left_violation > 0 or right_violation > 0 or bottom_violation > 0 or top_violation > 0:
                penalty += 1000 * (left_violation + right_violation + bottom_violation + top_violation)
        return penalty

    @staticmethod
    def check_overlap_constraints(individual: np.ndarray) -> float:
        """Check overlap constraints efficiently using optimized spatial queries"""
        penalty = 0.0
        n = len(individual)
        
        if n <= 1:
            return penalty
        
        # Use cKDTree for efficient neighbor queries
        tree = cKDTree(individual[:, :2])
        
        # Process each circle only once for efficient pairwise comparison
        processed_indices = set()
        
        for i in range(n):
            if i in processed_indices:
                continue
                
            x1, y1, r1 = individual[i]
            
            # Find all neighbors within double the combined radius
            max_radius = np.max(individual[:, 2])
            neighbors = tree.query_ball_point([x1, y1], 2 * (r1 + max_radius), p=2)
            
            # Only check pairs that haven't been processed yet
            for j in neighbors:
                if i >= j or j in processed_indices:
                    continue
                    
                x2, y2, r2 = individual[j]
                
                # Fast distance check using squared distance
                dx = x2 - x1
                dy = y2 - y1
                distance_squared = dx*dx + dy*dy
                
                # Check if circles are close enough to potentially overlap
                combined_radius = r1 + r2
                if distance_squared < combined_radius * combined_radius:
                    actual_distance = math.sqrt(distance_squared)
                    if actual_distance < combined_radius:
                        overlap = combined_radius - actual_distance
                        penalty += overlap * 1000 * (1.0 + overlap * 0.01)
            
            processed_indices.add(i)
        
        return penalty

class CircleEvaluator:
    """Handles fitness evaluation"""
    
    @staticmethod
    def evaluate_fitness(individual: np.ndarray) -> float:
        """Evaluate fitness of an individual (sum of radii) with penalties for violations"""
        total_radius = np.sum(individual[:, 2])
        
        # Calculate penalties
        containment_penalty = CircleValidator.check_containment_constraints(individual)
        overlap_penalty = CircleValidator.check_overlap_constraints(individual)
        
        return total_radius - containment_penalty - overlap_penalty

class TournamentSelector:
    """Handles tournament selection operations"""
    
    @staticmethod
    def select(population: List[np.ndarray], fitness_scores: List[float], 
               tournament_size: int = 3) -> np.ndarray:
        """Select an individual using tournament selection"""
        tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_index]

class CircleCrossover:
    """Handles crossover operations"""
    
    @staticmethod
    def uniform_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform crossover between two parents"""
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Single point crossover on positions and radii
        crossover_point = random.randint(1, len(parent1) - 1)

        # Swap positions and radii for half the circles
        child1[crossover_point:, :2] = parent2[crossover_point:, :2]
        child1[crossover_point:, 2] = parent2[crossover_point:, 2]

        child2[crossover_point:, :2] = parent1[crossover_point:, :2]
        child2[crossover_point:, 2] = parent1[crossover_point:, 2]

        return child1, child2

class CircleMutator:
    """Handles mutation operations"""
    
    @staticmethod
    def mutate(individual: np.ndarray, mutation_rate: float) -> None:
        """Mutate an individual in-place"""
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                # Mutate position slightly
                individual[i, 0] += np.random.normal(0, 0.01)
                individual[i, 1] += np.random.normal(0, 0.01)

                # Keep within bounds
                individual[i, 0] = np.clip(individual[i, 0], 0.01, 0.99)
                individual[i, 1] = np.clip(individual[i, 1], 0.01, 0.99)

                # Mutate radius
                individual[i, 2] += np.random.normal(0, 0.005)
                individual[i, 2] = max(0.001, individual[i, 2])

class OverlapResolver:
    """Handles overlap resolution operations"""
    
    @staticmethod
    def resolve_overlaps(individual: np.ndarray, max_iterations: int = 100) -> None:
        """Improve overlap resolution with better geometric handling"""
        for iteration in range(max_iterations):
            any_changes = False
            # Check each pair of circles
            for i in range(len(individual)):
                for j in range(i+1, len(individual)):
                    x1, y1, r1 = individual[i]
                    x2, y2, r2 = individual[j]

                    dx = x2 - x1
                    dy = y2 - y1
                    distance = math.sqrt(dx*dx + dy*dy)

                    # If circles overlap
                    if distance < r1 + r2:
                        overlap = (r1 + r2) - distance
                        if distance > 0:
                            # Push them apart along the line connecting centers
                            push_x = dx / distance * overlap * 0.5
                            push_y = dy / distance * overlap * 0.5

                            individual[i, 0] -= push_x
                            individual[i, 1] -= push_y
                            individual[j, 0] += push_x
                            individual[j, 1] += push_y
                        else:
                            # If they're at the same position, push them apart randomly
                            angle = np.random.uniform(0, 2*np.pi)
                            push_dist = overlap * 0.5
                            individual[i, 0] -= push_dist * np.cos(angle)
                            individual[i, 1] -= push_dist * np.sin(angle)
                            individual[j, 0] += push_dist * np.cos(angle)
                            individual[j, 1] += push_dist * np.sin(angle)

                        # Keep within bounds
                        individual[i, 0] = np.clip(individual[i, 0], r1, 1-r1)
                        individual[i, 1] = np.clip(individual[i, 1], r1, 1-r1)
                        individual[j, 0] = np.clip(individual[j, 0], r2, 1-r2)
                        individual[j, 1] = np.clip(individual[j, 1], r2, 1-r2)
                        any_changes = True

            if not any_changes:
                break

class EvolutionEngine:
    """Main evolutionary optimization engine"""
    
    def __init__(self, n_circles: int = 26):
        self.n_circles = n_circles
        self.population_size = 200
        self.generations = 500
        self.mutation_rate = 0.1
        self.elite_size = 20

    def run_evolution(self) -> np.ndarray:
        """Run the evolutionary optimization process"""
        # Initialize population
        population = []
        for _ in range(self.population_size):
            individual = CircleInitializer.mixed_initialization(self.n_circles)
            OverlapResolver.resolve_overlaps(individual)
            population.append(individual)

        # Evolutionary loop
        best_fitness_history = []
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness = CircleEvaluator.evaluate_fitness(individual)
                fitness_scores.append(fitness)

            # Track best fitness
            best_fitness = max(fitness_scores)
            best_fitness_history.append(best_fitness)

            # Select top individuals (elitism)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            elite = [population[i] for i in sorted_indices[:self.elite_size]]

            # Generate new population
            new_population = elite.copy()

            # Adaptive mutation rate: decrease over time
            adaptive_mutation_rate = self.mutation_rate * (1 - generation / self.generations)
            if adaptive_mutation_rate < 0.01:
                adaptive_mutation_rate = 0.01

            # Fill rest of population through crossover and mutation
            while len(new_population) < self.population_size:
                parent1 = TournamentSelector.select(population, fitness_scores)
                parent2 = TournamentSelector.select(population, fitness_scores)

                if random.random() < 0.8:  # Crossover probability
                    child1, child2 = CircleCrossover.uniform_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Apply mutation with adaptive rate
                if random.random() < adaptive_mutation_rate:
                    CircleMutator.mutate(child1, adaptive_mutation_rate)
                if random.random() < adaptive_mutation_rate:
                    CircleMutator.mutate(child2, adaptive_mutation_rate)

                new_population.extend([child1, child2])

            # Trim to exact population size
            population = new_population[:self.population_size]

            # Print progress
            if generation % 50 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")

        # Return best solution
        final_fitness_scores = [CircleEvaluator.evaluate_fitness(ind) for ind in population]
        best_index = np.argmax(final_fitness_scores)
        return population[best_index]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    engine = EvolutionEngine(n_circles=26)
    return engine.run_evolution()

# EVOLVE-BLOCK-END