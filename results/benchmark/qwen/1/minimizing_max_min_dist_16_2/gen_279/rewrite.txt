# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import time
import random

class PointConfiguration:
    """Represents a configuration of points and provides utility methods."""

    def __init__(self, points):
        self.points = np.array(points)
        self.n_points = len(points)

    def compute_min_max_ratio(self):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if self.n_points < 2:
            return 0

        # Compute pairwise distances with enhanced numerical stability
        distance_matrix = squareform(pdist(self.points))

        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(distance_matrix, np.inf)

        # Get all finite distances (excluding NaN and inf values)
        finite_distances = distance_matrix[np.isfinite(distance_matrix)]

        if len(finite_distances) == 0:
            return 0

        # Get min and max distances
        dmin = np.min(finite_distances)
        dmax = np.max(finite_distances)

        # Avoid division by zero
        if dmax == 0:
            return 0

        return dmin / dmax

    def compute_distance_matrix(self):
        """Compute full pairwise distance matrix."""
        return squareform(pdist(self.points))

    def get_clipped_points(self, lower=0, upper=1):
        """Get points clipped to specified bounds."""
        return np.clip(self.points, lower, upper)

    def copy(self):
        """Create a copy of this configuration."""
        return PointConfiguration(self.points.copy())

class PointDispersionGA:
    """Genetic Algorithm for point dispersion optimization with specialized operators."""

    def __init__(self, population_size=50, generations=100, elite_size=5):
        self.population_size = population_size
        self.generations = generations
        self.elite_size = elite_size
        self.best_individual = None
        self.best_fitness = 0
        self.history = []
        self.stagnation_counter = 0
        self.max_stagnation = 20

    def initialize_population(self):
        """Initialize population with diverse strategies."""
        population = []
        
        # Strategy 1: Hexagonal grid
        hex_points = self._create_hexagonal_grid()
        
        # Strategy 2: Golden spiral
        spiral_points = self._create_golden_spiral()
        
        # Strategy 3: Random with better spread
        random_points = np.random.rand(16, 2)
        
        # Strategy 4: Grid with jitter
        grid_points = np.array([[i/3, j/3] for i in range(4) for j in range(4) if i*4+j < 16])
        jittered_points = grid_points + np.random.normal(0, 0.05, (16, 2))
        jittered_points = np.clip(jittered_points, 0, 1)
        
        # Strategy 5: Perturbed hexagonal grid
        perturbed_hex = hex_points + np.random.normal(0, 0.03, hex_points.shape)
        perturbed_hex = np.clip(perturbed_hex, 0, 1)
        
        # Add these diverse initial configurations
        initial_configs = [hex_points, spiral_points, random_points, jittered_points, perturbed_hex]
        
        for i, config in enumerate(initial_configs):
            if len(population) < self.population_size:
                population.append(config.copy())
        
        # Fill remaining spots with random configurations
        while len(population) < self.population_size:
            individual = np.random.rand(16, 2)
            population.append(individual)
            
        return population

    def _create_hexagonal_grid(self):
        """Create a hexagonal grid arrangement."""
        points = []
        rows = 4
        cols = 4

        # Hexagonal packing parameters
        spacing_x = 1.0 / (cols - 1)
        spacing_y = np.sqrt(3) / 2 / (rows - 1)  # Height of equilateral triangle

        for i in range(rows):
            for j in range(cols):
                x = j * spacing_x + (i % 2) * spacing_x / 2
                y = i * spacing_y
                points.append([x, y])

        return np.array(points)

    def _create_golden_spiral(self):
        """Create a golden spiral arrangement."""
        indices = np.arange(16)
        golden_angle = 2.399963229728653
        angles = golden_angle * indices
        radii = np.log(indices + 1) / np.log(16)
        golden_spiral = np.column_stack([
            0.5 + 0.45 * radii * np.cos(angles),
            0.5 + 0.45 * radii * np.sin(angles)
        ])
        return np.clip(golden_spiral, 0, 1)

    def fitness_function(self, points):
        """Evaluate fitness based on min/max distance ratio."""
        config = PointConfiguration(points)
        return config.compute_min_max_ratio()

    def selection(self, population, fitnesses):
        """Tournament selection with elitism."""
        # Elitism - keep best individuals
        elite_indices = np.argsort(fitnesses)[-self.elite_size:]
        selected = [population[i] for i in elite_indices]
        
        # Tournament selection for rest
        while len(selected) < self.population_size:
            tournament_size = 3
            tournament_indices = np.random.choice(len(population), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
            selected.append(population[winner_index].copy())
            
        return selected

    def crossover(self, parent1, parent2):
        """Specialized crossover that favors good geometric properties."""
        # Blend crossover (BLX-α) with geometric bias
        alpha = 0.5
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)
        
        for i in range(16):
            # For each point, apply blend crossover
            for j in range(2):  # x and y coordinates
                min_val = min(parent1[i,j], parent2[i,j])
                max_val = max(parent1[i,j], parent2[i,j])
                range_val = max_val - min_val
                
                # Expand the range by alpha
                expanded_min = min_val - alpha * range_val
                expanded_max = max_val + alpha * range_val
                
                # Sample from expanded range
                child1[i,j] = np.random.uniform(expanded_min, expanded_max)
                child2[i,j] = np.random.uniform(expanded_min, expanded_max)
                
        # Clip to bounds
        child1 = np.clip(child1, 0, 1)
        child2 = np.clip(child2, 0, 1)
        
        # Apply geometric bias - favor configurations that maintain good spread
        if self.fitness_function(child1) > self.fitness_function(parent1) and \
           self.fitness_function(child1) > self.fitness_function(parent2):
            return child1
        elif self.fitness_function(child2) > self.fitness_function(parent1) and \
             self.fitness_function(child2) > self.fitness_function(parent2):
            return child2
        else:
            # If neither child is better, return one of parents at random
            return parent1 if np.random.random() < 0.5 else parent2

    def mutate(self, individual, generation):
        """Adaptive mutation with decreasing rate."""
        # Adaptive mutation rate that decreases with generation
        mutation_rate = max(0.1, 0.5 * (1 - generation / self.generations))
        
        # Apply Gaussian mutation with adaptive standard deviation
        std_dev = 0.05 * (1 - generation / self.generations) + 0.005
        
        mutated = individual.copy()
        
        for i in range(16):
            if np.random.random() < mutation_rate:
                # Mutate both coordinates
                mutated[i, 0] += np.random.normal(0, std_dev)
                mutated[i, 1] += np.random.normal(0, std_dev)
                
        # Clip to bounds
        mutated = np.clip(mutated, 0, 1)
        return mutated

    def optimize(self):
        """Main optimization loop."""
        # Initialize population
        population = self.initialize_population()
        
        # Track best solution
        best_fitness = -np.inf
        best_individual = None
        
        # Evolution loop
        for gen in range(self.generations):
            # Evaluate fitness
            fitnesses = [self.fitness_function(individual) for individual in population]
            
            # Track best individual
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > best_fitness:
                best_fitness = fitnesses[max_fitness_idx]
                best_individual = population[max_fitness_idx].copy()
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
                
            # Check for stagnation
            if self.stagnation_counter > self.max_stagnation:
                # Inject diversity by replacing worst individuals with new random ones
                worst_indices = np.argsort(fitnesses)[:self.elite_size]
                for idx in worst_indices:
                    population[idx] = np.random.rand(16, 2)
                self.stagnation_counter = 0
            
            # Selection
            selected = self.selection(population, fitnesses)
            
            # Crossover and mutation
            new_population = selected[:self.elite_size]  # Keep elites
            
            # Create offspring through crossover
            while len(new_population) < self.population_size:
                parent1_idx = np.random.randint(0, len(selected))
                parent2_idx = np.random.randint(0, len(selected))
                
                child = self.crossover(selected[parent1_idx], selected[parent2_idx])
                child = self.mutate(child, gen)
                
                new_population.append(child)
                
            population = new_population[:self.population_size]
            
            # Store history for debugging
            self.history.append(best_fitness)
            
            # Early termination if we've found a very good solution
            if best_fitness > 0.3:  # Early exit threshold
                break
                
        self.best_individual = best_individual
        self.best_fitness = best_fitness
        
        return best_individual

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    # Initialize the genetic algorithm with tuned parameters
    ga = PointDispersionGA(population_size=60, generations=150, elite_size=8)
    
    # Run optimization
    best_points = ga.optimize()
    
    # Final local refinement using scipy to fine-tune
    def objective(x_flat):
        points = x_flat.reshape(-1, 2)
        points = np.clip(points, 0, 1)
        config = PointConfiguration(points)
        return -config.compute_min_max_ratio()  # Negative because we want to maximize
    
    # Prepare initial guess for local optimization
    bounds = [(0, 1) for _ in range(32)]
    
    # Try multiple local optimizations with different starting points
    best_local_points = best_points.copy()
    best_local_ratio = ga.best_fitness
    
    # Multiple local searches
    for i in range(3):
        np.random.seed(1000 + i)
        # Add small perturbation to current best
        perturbed = best_points + np.random.normal(0, 0.01, best_points.shape)
        perturbed = np.clip(perturbed, 0, 1)
        
        # Local optimization
        try:
            result = minimize(objective, perturbed.flatten(), method='L-BFGS-B', 
                            bounds=bounds, options={'ftol': 1e-12, 'gtol': 1e-12})
            if result.success:
                local_points = result.x.reshape(-1, 2)
                local_points = np.clip(local_points, 0, 1)
                local_ratio = PointConfiguration(local_points).compute_min_max_ratio()
                
                if local_ratio > best_local_ratio:
                    best_local_ratio = local_ratio
                    best_local_points = local_points.copy()
        except:
            pass
    
    return best_local_points

# EVOLVE-BLOCK-END