# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List, Optional
import time
from collections import defaultdict
import warnings
from dataclasses import dataclass
from enum import Enum

class OptimizationState(Enum):
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"

@dataclass
class Individual:
    """Data class representing a circle packing individual"""
    positions: np.ndarray  # Shape: (n_circles, 2) - x, y coordinates
    radii: np.ndarray      # Shape: (n_circles,) - radii
    fitness: float = 0.0
    total_radius: float = 0.0
    penalty: float = 0.0

class ConstraintChecker:
    """Handles all constraint checking operations efficiently"""
    
    @staticmethod
    def is_valid_position(x: float, y: float, r: float) -> bool:
        """Check if a circle position is valid (within bounds)"""
        return (r <= x <= 1 - r and r <= y <= 1 - r)
    
    @staticmethod
    def check_all_constraints(individual: Individual) -> Tuple[bool, bool]:
        """Check all constraints and return (containment_valid, overlap_valid)"""
        # Check containment
        containment_valid = all(
            ConstraintChecker.is_valid_position(x, y, r) 
            for x, y, r in zip(individual.positions[:, 0], individual.positions[:, 1], individual.radii)
        )
        
        if not containment_valid:
            return False, False
            
        # Check overlaps using spatial indexing
        if len(individual.positions) <= 1:
            return True, True
            
        # Use distance matrix for overlap detection
        distances = cdist(individual.positions, individual.positions)
        n = len(individual.positions)
        
        for i in range(n):
            for j in range(i + 1, n):
                if distances[i, j] < (individual.radii[i] + individual.radii[j]):
                    return True, False
                    
        return True, True

class CirclePackingOptimizer:
    """Main optimizer class managing the evolutionary process"""
    
    def __init__(self):
        self.population_manager = PopulationManager()
        self.constraint_checker = ConstraintChecker()
        self.state = OptimizationState.EXPLORATION
        self.best_individual = None
        self.best_fitness = -float('inf')
        self.generations_without_improvement = 0
        self.max_generations_without_improvement = 50
        
    def initialize_population(self, pop_size: int, n_circles: int) -> List[Individual]:
        """Initialize population with improved Voronoi-based distribution"""
        return self.population_manager.create_initial_population(pop_size, n_circles)
    
    def evaluate_individual(self, individual: Individual) -> Individual:
        """Evaluate fitness of an individual"""
        # Calculate total radius
        total_radius = np.sum(individual.radii)
        
        # Check constraints
        containment_valid, overlap_valid = self.constraint_checker.check_all_constraints(individual)
        
        # Calculate penalty
        penalty = 0.0
        if not containment_valid:
            # Add penalty for boundary violations
            for x, y, r in zip(individual.positions[:, 0], individual.positions[:, 1], individual.radii):
                if not ConstraintChecker.is_valid_position(x, y, r):
                    # Calculate boundary violations
                    violations = 0
                    if r > x:
                        violations += (r - x) ** 2
                    if r > (1 - x):
                        violations += (r - (1 - x)) ** 2
                    if r > y:
                        violations += (r - y) ** 2
                    if r > (1 - y):
                        violations += (r - (1 - y)) ** 2
                    penalty += violations * 1000.0
        
        if not overlap_valid:
            penalty += 100000.0  # Large penalty for overlaps
            
        # Fitness is total radius minus penalty
        fitness = total_radius - penalty
        
        individual.fitness = fitness
        individual.total_radius = total_radius
        individual.penalty = penalty
        
        return individual
    
    def evolve_generation(self, population: List[Individual]) -> List[Individual]:
        """Evolve population for one generation"""
        # Evaluate fitness for all individuals
        evaluated_population = [self.evaluate_individual(ind) for ind in population]
        
        # Track best individual
        best_in_generation = max(evaluated_population, key=lambda x: x.fitness)
        if best_in_generation.fitness > self.best_fitness:
            self.best_fitness = best_in_generation.fitness
            self.best_individual = best_in_generation
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1
            
        # Create new population
        new_population = self.population_manager.elite_selection(evaluated_population)
        
        # Generate offspring until population is full
        while len(new_population) < len(population):
            # Selection
            parent1 = self.population_manager.tournament_selection(evaluated_population)
            parent2 = self.population_manager.tournament_selection(evaluated_population)
            
            # Crossover
            child = self.population_manager.crossover(parent1, parent2)
            
            # Mutation with adaptive rate
            mut_rate = self.population_manager.adaptive_mutation_rate(len(new_population))
            child = self.population_manager.mutate(child, mut_rate)
            
            # Refinement
            child = self.population_manager.refine_individual(child)
            
            new_population.append(child)
        
        return new_population[:len(population)]
    
    def should_terminate(self, generation: int) -> bool:
        """Check if optimization should terminate"""
        # Early termination if no improvement for many generations
        if self.generations_without_improvement > self.max_generations_without_improvement:
            return True
            
        # Termination based on fitness threshold
        if self.best_fitness > 2.6:
            return True
            
        return False

class PopulationManager:
    """Manages population operations and evolutionary operators"""
    
    def __init__(self):
        self.pop_size = 150
        self.tournament_size = 7
        self.initial_mutation_rate = 0.25
        self.final_mutation_rate = 0.05
        self.crossover_rate = 0.8
        self.elite_count = 3
        
    def create_initial_population(self, pop_size: int, n_circles: int) -> List[Individual]:
        """Create initial population with improved distribution"""
        population = []
        
        # Generate systematic points for better distribution
        points = self._generate_systematic_points(n_circles)
        
        for _ in range(pop_size):
            # Initialize positions
            positions = np.zeros((n_circles, 2))
            radii = np.zeros(n_circles)
            
            # Assign positions and radii
            for i in range(n_circles):
                if i < len(points):
                    positions[i] = points[i]
                else:
                    positions[i] = [random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)]
                
                # Add perturbation
                positions[i] += np.array([random.uniform(-0.03, 0.03), random.uniform(-0.03, 0.03)])
                
                # Clip to valid range
                positions[i] = np.clip(positions[i], [0.05, 0.05], [0.95, 0.95])
                
                # Calculate radius based on distance to boundaries
                x, y = positions[i]
                margin = min(x, y, 1 - x, 1 - y)
                base_radius = min(0.15, margin / 2.0)
                radii[i] = max(0.005, base_radius * random.uniform(0.5, 1.5))
            
            individual = Individual(positions, radii)
            individual = self.refine_individual(individual)
            population.append(individual)
            
        return population
    
    def _generate_systematic_points(self, n_points: int) -> np.ndarray:
        """Generate systematically distributed points"""
        points = []
        grid_size = max(6, int(np.ceil(np.sqrt(n_points))))
        
        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) < n_points:
                    x = 0.05 + (i / (grid_size - 1)) * 0.90
                    y = 0.05 + (j / (grid_size - 1)) * 0.90
                    points.append([x, y])
        
        # Add random points
        while len(points) < n_points:
            points.append([random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)])
            
        return np.array(points[:n_points])
    
    def refine_individual(self, individual: Individual) -> Individual:
        """Refine individual to satisfy constraints"""
        # Apply positional constraints
        for i in range(len(individual.positions)):
            x, y = individual.positions[i]
            r = individual.radii[i]
            
            # Ensure containment
            individual.positions[i, 0] = np.clip(x, r, 1 - r)
            individual.positions[i, 1] = np.clip(y, r, 1 - r)
        
        # Resolve overlaps iteratively
        max_iter = 50
        for _ in range(max_iter):
            changed = False
            for i in range(len(individual.positions)):
                x, y = individual.positions[i]
                r = individual.radii[i]
                
                # Check overlap with others
                for j in range(len(individual.positions)):
                    if i != j:
                        ox, oy = individual.positions[j]
                        oradius = individual.radii[j]
                        distance = np.sqrt((x - ox)**2 + (y - oy)**2)
                        
                        if distance < (r + oradius):
                            # Move apart
                            if distance > 0.0001:
                                dx = (x - ox) / distance
                                dy = (y - oy) / distance
                                separation = (r + oradius - distance) * 0.5
                                individual.positions[i, 0] += dx * separation
                                individual.positions[i, 1] += dy * separation
                            else:
                                # Small random perturbation for identical positions
                                individual.positions[i, 0] += random.uniform(-0.001, 0.001)
                                individual.positions[i, 1] += random.uniform(-0.001, 0.001)
                            
                            # Ensure containment after adjustment
                            individual.positions[i, 0] = np.clip(
                                individual.positions[i, 0], 
                                individual.radii[i], 
                                1 - individual.radii[i]
                            )
                            individual.positions[i, 1] = np.clip(
                                individual.positions[i, 1], 
                                individual.radii[i], 
                                1 - individual.radii[i]
                            )
                            changed = True
            
            if not changed:
                break
                
        return individual
    
    def tournament_selection(self, population: List[Individual]) -> Individual:
        """Select individual using tournament selection"""
        selected_indices = random.sample(range(len(population)), self.tournament_size)
        selected_fitness = [population[i].fitness for i in selected_indices]
        
        winner_idx = selected_indices[np.argmax(selected_fitness)]
        return population[winner_idx].copy()
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Perform crossover between two parents"""
        if random.random() > self.crossover_rate:
            return parent1.copy()
        
        n = len(parent1.positions)
        child_positions = np.zeros_like(parent1.positions)
        child_radii = np.zeros(n)
        
        # Multi-point crossover
        crossover_points = sorted(random.sample(range(1, n), min(3, n-1)))
        
        last_point = 0
        use_parent1 = True
        for point in crossover_points:
            if use_parent1:
                child_positions[last_point:point] = parent1.positions[last_point:point]
                child_radii[last_point:point] = parent1.radii[last_point:point]
            else:
                child_positions[last_point:point] = parent2.positions[last_point:point]
                child_radii[last_point:point] = parent2.radii[last_point:point]
            last_point = point
            use_parent1 = not use_parent1
        
        # Handle final segment
        if use_parent1:
            child_positions[last_point:] = parent1.positions[last_point:]
            child_radii[last_point:] = parent1.radii[last_point:]
        else:
            child_positions[last_point:] = parent2.positions[last_point:]
            child_radii[last_point:] = parent2.radii[last_point:]
            
        return Individual(child_positions, child_radii)
    
    def mutate(self, individual: Individual, mutation_rate: float) -> Individual:
        """Mutate an individual"""
        mutated = individual.copy()
        
        for i in range(len(mutated.positions)):
            if random.random() < mutation_rate:
                # Choose mutation type
                mutation_type = random.choices(
                    [0, 1, 2, 3], 
                    weights=[0.5, 0.5, 0.2, 0.3]  # Position more likely
                )[0]
                
                if mutation_type == 0:  # Mutate x position
                    mutated.positions[i, 0] = np.clip(
                        mutated.positions[i, 0] + random.gauss(0, 0.05), 
                        0.05, 0.95
                    )
                elif mutation_type == 1:  # Mutate y position
                    mutated.positions[i, 1] = np.clip(
                        mutated.positions[i, 1] + random.gauss(0, 0.05), 
                        0.05, 0.95
                    )
                elif mutation_type == 2:  # Mutate radius
                    mutated.radii[i] = np.clip(
                        mutated.radii[i] + random.gauss(0, 0.01), 
                        0.001, 0.2
                    )
                else:  # Mutate both position and radius
                    mutated.positions[i, 0] = np.clip(
                        mutated.positions[i, 0] + random.gauss(0, 0.02), 
                        0.05, 0.95
                    )
                    mutated.positions[i, 1] = np.clip(
                        mutated.positions[i, 1] + random.gauss(0, 0.02), 
                        0.05, 0.95
                    )
                    mutated.radii[i] = np.clip(
                        mutated.radii[i] + random.gauss(0, 0.005), 
                        0.001, 0.2
                    )
        
        return self.refine_individual(mutated)
    
    def adaptive_mutation_rate(self, generation: int) -> float:
        """Adaptive mutation rate that decreases over time"""
        return self.initial_mutation_rate - (self.initial_mutation_rate - self.final_mutation_rate) * (generation / 1000)
    
    def elite_selection(self, population: List[Individual]) -> List[Individual]:
        """Select elite individuals"""
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        return sorted_pop[:self.elite_count]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    # Initialize optimizer
    optimizer = CirclePackingOptimizer()
    n = 26
    
    # Initialize population
    population = optimizer.initialize_population(150, n)
    
    # Evolution loop
    start_time = time.time()
    best_total_radius = 0.0
    generations = 800
    
    for generation in range(generations):
        population = optimizer.evolve_generation(population)
        
        # Log progress
        if generation % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Generation {generation}: Best radius sum = {optimizer.best_fitness:.6f} Time: {elapsed:.2f}s")
            
        # Check for early termination
        if optimizer.should_terminate(generation):
            print(f"Early termination at generation {generation}")
            break
    
    elapsed = time.time() - start_time
    benchmark_ratio = optimizer.best_fitness / 2.6358627564136983 if optimizer.best_fitness else 0.0
    print(f"Final result: Best radius sum = {optimizer.best_fitness:.6f} (penalty={optimizer.best_individual.penalty:.2f}) Time: {elapsed:.2f}s")
    print(f"Benchmark ratio: {benchmark_ratio:.6f}")
    
    # Return best solution
    if optimizer.best_individual is not None:
        return np.column_stack([
            optimizer.best_individual.positions,
            optimizer.best_individual.radii
        ])
    else:
        # Fallback - return first individual from final population
        return np.column_stack([
            population[0].positions,
            population[0].radii
        ])

# EVOLVE-BLOCK-END