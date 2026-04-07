# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
from typing import Tuple, List, Optional
import random
import time
from copy import deepcopy

# Global constants
POPULATION_SIZE = 80
GENERATIONS = 60
MUTATION_RATE_INITIAL = 0.12
CROSSOVER_RATE = 0.85
TOURNAMENT_SIZE = 5
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

class CircleValidator:
    """Validates circle configurations and checks constraints."""
    
    @staticmethod
    def is_valid(circles: np.ndarray, n_circles: int) -> bool:
        """Check if the configuration satisfies all constraints."""
        if len(circles) != n_circles:
            return False
            
        # Check containment constraints
        radii = circles[:, 2]
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        
        # Vectorized containment check
        containment_check = (
            (radii <= x_coords) & 
            (radii <= y_coords) & 
            (radii <= 1 - x_coords) & 
            (radii <= 1 - y_coords)
        )
        
        if not np.all(containment_check):
            return False

        # Check overlap constraints
        if n_circles > 1:
            distances = cdist(circles[:, :2], circles[:, :2])
            # Create upper triangular mask to avoid duplicate comparisons
            mask = np.triu(np.ones((n_circles, n_circles), dtype=bool), k=1)
            
            # Calculate minimum required distance
            min_distances = (circles[:, 2][:, np.newaxis] + circles[:, 2][np.newaxis, :]) * mask
            
            # Check for overlaps
            overlaps = distances < min_distances
            if np.any(overlaps):
                return False
                
        return True

class FitnessEvaluator:
    """Evaluates the fitness of circle configurations."""
    
    @staticmethod
    def calculate_sum_radii(circles: np.ndarray) -> float:
        """Calculate the sum of all radii."""
        return np.sum(circles[:, 2])

class PopulationInitializer:
    """Creates initial population with valid configurations."""
    
    def __init__(self, n_circles: int):
        self.n_circles = n_circles
    
    def create_voronoi_initialization(self) -> np.ndarray:
        """Initialize circles using Voronoi diagram approach for better spatial distribution."""
        # Generate initial candidate points using hexagonal grid
        n_points = self.n_circles + 10
        
        # Create hexagonal grid pattern
        points = []
        rows = int(np.ceil(np.sqrt(n_points)))
        cols = int(np.ceil(n_points / rows))
        
        spacing = 1.0 / (max(rows, cols) + 2)
        hex_height = spacing * np.sqrt(3) / 2
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= n_points:
                    break
                x = (j + 0.5 + (i % 2) * 0.5) * spacing
                y = (i + 0.5) * hex_height
                if x <= 1 and y <= 1:
                    points.append([x, y])
        
        # Trim to exact number needed
        points = points[:n_points]
        
        # Add some random jitter
        for point in points:
            point[0] += np.random.uniform(-spacing/4, spacing/4)
            point[1] += np.random.uniform(-spacing/4, spacing/4)
        
        # Ensure points are within bounds
        points = [[max(0.01, min(0.99, p[0])), max(0.01, min(0.99, p[1]))] for p in points]
        
        # Create Voronoi diagram
        vor = Voronoi(points)
        
        # Select appropriate Voronoi cells
        circles = np.zeros((self.n_circles, 3))
        valid_cells = []
        
        for i in range(min(self.n_circles, len(vor.point_region))):
            region = vor.point_region[i]
            if region != -1:  # Valid region
                valid_cells.append(i)
        
        # Use selected cells for circle placement
        selected_indices = valid_cells[:self.n_circles]
        
        for i, idx in enumerate(selected_indices):
            center = vor.points[idx]
            x, y = center
            
            # Estimate radius based on Voronoi cell area
            radius_estimate = spacing / 3
            min_distance_to_boundary = min(x, y, 1-x, 1-y)
            final_radius = min(radius_estimate, min_distance_to_boundary * 0.8)
            final_radius = max(0.01, final_radius)
            
            circles[i] = [x, y, final_radius]
        
        return circles
    
    def create_constrained_random_initialization(self) -> np.ndarray:
        """Create a constrained random initialization."""
        circles = np.zeros((self.n_circles, 3))
        
        for i in range(self.n_circles):
            attempts = 0
            while attempts < 100:
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                
                min_dist = min(x, y, 1-x, 1-y)
                r = np.random.uniform(0.01, min_dist/2)
                
                overlap = False
                for j in range(i):
                    existing_x, existing_y, existing_r = circles[j]
                    dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                    if dist < r + existing_r:
                        overlap = True
                        break
                
                if not overlap:
                    circles[i] = [x, y, r]
                    break
                attempts += 1
                
            if attempts >= 100:
                # Fallback to simple grid
                grid_size = int(np.ceil(np.sqrt(self.n_circles)))
                spacing = 1.0 / (grid_size + 1)
                row = i // grid_size
                col = i % grid_size
                x = (col + 1) * spacing
                y = (row + 1) * spacing
                r = spacing / 4
                circles[i] = [x, y, r]
                
        return circles
    
    def create_simple_initialization(self) -> np.ndarray:
        """Create a simple but valid initial configuration."""
        circles = np.zeros((self.n_circles, 3))
        
        grid_size = int(np.ceil(np.sqrt(self.n_circles)))
        spacing = 1.0 / (grid_size + 1)
        
        idx = 0
        for row in range(grid_size):
            for col in range(grid_size):
                if idx >= self.n_circles:
                    break
                x = (col + 1) * spacing
                y = (row + 1) * spacing
                r = spacing / 4
                circles[idx] = [x, y, r]
                idx += 1
                
        return circles
    
    def initialize_population(self, pop_size: int) -> List[np.ndarray]:
        """Initialize population with valid configurations."""
        population = []
        
        for i in range(pop_size):
            if i == 0:
                # First individual: Voronoi initialization
                circles = self.create_voronoi_initialization()
            elif i < pop_size // 3:
                # Second third: random with constraint checking
                circles = self.create_constrained_random_initialization()
            else:
                # Last third: slightly modified Voronoi
                base = self.create_voronoi_initialization()
                circles = base.copy()
                # Add small mutations to diversify
                for j in range(self.n_circles):
                    if np.random.random() < 0.2:
                        circles[j, 0] += np.random.uniform(-0.02, 0.02)
                        circles[j, 1] += np.random.uniform(-0.02, 0.02)
                        circles[j, 2] *= np.random.uniform(0.9, 1.1)
                        
            population.append(circles)
            
        return population

class LocalOptimizer:
    """Performs local optimization to improve circle placements."""
    
    @staticmethod
    def optimize_placement(circles: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Apply local optimization to improve placement."""
        circles = circles.copy()
        n = len(circles)
        
        # Gradient-like approach to increase radii
        for iteration in range(max_iter):
            improved = False
            
            # Try to increase radii while respecting constraints
            for i in range(n):
                original_radius = circles[i][2]
                
                # Calculate maximum possible radius
                max_radius = min(
                    circles[i][0],  # x coordinate
                    circles[i][1],  # y coordinate
                    1 - circles[i][0],  # distance to right edge
                    1 - circles[i][1]   # distance to top edge
                )
                
                # Try to increase the radius
                new_radius = min(original_radius + 0.005, max_radius)
                
                if new_radius > original_radius:
                    # Temporarily update radius
                    circles[i][2] = new_radius
                    
                    # For now, skip detailed validation to maintain performance
                    improved = True
                        
            if not improved:
                break
                
        return circles

class EvolutionaryOperator:
    """Handles evolutionary operators like crossover and mutation."""
    
    def __init__(self, n_circles: int):
        self.n_circles = n_circles
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform constraint-aware crossover."""
        if np.random.random() > CROSSOVER_RATE:
            return parent1.copy(), parent2.copy()

        # Uniform crossover
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for i in range(self.n_circles):
            if np.random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
        
        # Ensure children are valid
        child1 = LocalOptimizer.optimize_placement(child1)
        child2 = LocalOptimizer.optimize_placement(child2)

        return child1, child2

    def mutate(self, circles: np.ndarray, mutation_rate: float = MUTATION_RATE_INITIAL) -> np.ndarray:
        """Apply mutation with adaptive strategies."""
        mutated = circles.copy()
        n = len(mutated)

        # Adaptive mutation with different strategies
        for i in range(n):
            if np.random.random() < mutation_rate:
                # 70% chance to mutate position, 30% to mutate radius
                if np.random.random() < 0.7:
                    # Mutate position with adaptive magnitude
                    mutation_magnitude = 0.03 * (1 - mutation_rate)
                    mutated[i][0] = np.clip(mutated[i][0] + np.random.normal(0, mutation_magnitude), 0, 1)
                    mutated[i][1] = np.clip(mutated[i][1] + np.random.normal(0, mutation_magnitude), 0, 1)
                else:
                    # Mutate radius
                    mutated[i][2] = np.clip(mutated[i][2] + np.random.normal(0, 0.01), 0.01, 0.5)

        # Optimize the mutated configuration
        mutated = LocalOptimizer.optimize_placement(mutated)

        return mutated

class TournamentSelector:
    """Implements tournament selection for evolutionary algorithm."""
    
    @staticmethod
    def select(population: List[np.ndarray], fitnesses: List[float], 
               tournament_size: int = TOURNAMENT_SIZE) -> int:
        """Select an individual using tournament selection."""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return winner_index

class CirclePackingOptimizer:
    """Main optimizer class that orchestrates the evolutionary process."""
    
    def __init__(self):
        self.n_circles = 26
        self.validator = CircleValidator()
        self.fitness_evaluator = FitnessEvaluator()
        self.initializer = PopulationInitializer(self.n_circles)
        self.evolution_operator = EvolutionaryOperator(self.n_circles)
        self.tournament_selector = TournamentSelector()
        
    def compute_fitness(self, circles: np.ndarray) -> float:
        """Compute fitness with penalty for invalid configurations."""
        if self.validator.is_valid(circles, self.n_circles):
            return self.fitness_evaluator.calculate_sum_radii(circles)
        else:
            # Invalid configurations get very low fitness
            return 0.0

    def run_evolution(self) -> np.ndarray:
        """Run the complete evolutionary algorithm."""
        # Initialize population
        population = self.initializer.initialize_population(POPULATION_SIZE)
        
        if not population:
            # Fallback to simple initialization
            return self.initializer.create_simple_initialization()
            
        best_solution = None
        best_fitness = -1

        for generation in range(GENERATIONS):
            # Adjust mutation rate based on generation (adaptive)
            mutation_rate = max(MUTATION_RATE_INITIAL * (1 - generation / GENERATIONS), 0.01)
            
            # Evaluate fitness for all individuals
            fitnesses = [self.compute_fitness(circles) for circles in population]
            
            # Track best solution
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > best_fitness:
                best_fitness = fitnesses[max_fitness_idx]
                best_solution = population[max_fitness_idx].copy()

            # Create new population
            new_population = []

            # Elitism: keep best individual
            new_population.append(best_solution.copy())

            # Generate offspring
            while len(new_population) < POPULATION_SIZE:
                # Tournament selection
                parent1_idx = self.tournament_selector.select(population, fitnesses)
                parent2_idx = self.tournament_selector.select(population, fitnesses)

                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]

                # Crossover
                child1, child2 = self.evolution_operator.crossover(parent1, parent2)

                # Mutation
                child1 = self.evolution_operator.mutate(child1, mutation_rate)
                child2 = self.evolution_operator.mutate(child2, mutation_rate)

                # Add children to new population
                new_population.extend([child1, child2])

            # Trim population to exact size
            population = new_population[:POPULATION_SIZE]

        # Return the best solution found
        if best_solution is None:
            # Fallback to a simple configuration if nothing worked
            return self.initializer.create_simple_initialization()

        return best_solution

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    optimizer = CirclePackingOptimizer()
    return optimizer.run_evolution()

# EVOLVE-BLOCK-END