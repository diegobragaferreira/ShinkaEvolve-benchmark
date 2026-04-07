# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import KDTree
import random
from typing import List, Tuple, Optional
import time

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

class CircleValidator:
    """Handles all circle validation operations efficiently."""
    
    def __init__(self):
        self.tree_cache = {}
        
    def is_valid_placement(self, circles: np.ndarray, idx: int) -> bool:
        """Check if circle at index idx is valid with cached spatial queries."""
        x, y, r = circles[idx]
        
        # Check containment constraints
        if x < r or x > 1 - r or y < r or y > 1 - r:
            return False

        # Check overlap constraints with existing circles using KDTree
        return self._check_overlap_with_existing(circles, idx)
    
    def _check_overlap_with_existing(self, circles: np.ndarray, idx: int) -> bool:
        """Check overlap with existing circles using spatial indexing."""
        # Only build tree if we have enough circles
        if len(circles) > 20:
            # Use cached KDTree for performance
            if id(circles) not in self.tree_cache:
                self.tree_cache[id(circles)] = KDTree(circles[:, :2])
            
            tree = self.tree_cache[id(circles)]
            x, y, r = circles[idx]
            
            # Query nearby circles (within 2*r distance)
            nearby_indices = tree.query_ball_point([x, y], 2 * r)
            
            for j in nearby_indices:
                if i != j:
                    x_j, y_j, r_j = circles[j]
                    distance = np.sqrt((x - x_j)**2 + (y - y_j)**2)
                    if distance < r + r_j:
                        return False
        else:
            # For small populations, do direct checking
            x, y, r = circles[idx]
            for i in range(len(circles)):
                if i == idx:
                    continue
                x_i, y_i, r_i = circles[i]
                distance = np.sqrt((x - x_i)**2 + (y - y_i)**2)
                if distance < r + r_i:
                    return False
        
        return True
    
    def validate_all_circles(self, circles: np.ndarray) -> bool:
        """Validate entire circle configuration."""
        for i in range(len(circles)):
            if not self.is_valid_placement(circles, i):
                return False
        return True

class CirclePack:
    """Represents a single circle packing configuration."""
    
    def __init__(self, circles: np.ndarray):
        self.circles = circles.copy()
        self.fitness = 0.0
        self.is_valid = False
    
    def calculate_fitness(self, validator: CircleValidator) -> float:
        """Calculate fitness and update validation status."""
        if validator.validate_all_circles(self.circles):
            self.fitness = np.sum(self.circles[:, 2])
            self.is_valid = True
        else:
            self.fitness = -1000000.0  # Penalty for invalid configurations
            self.is_valid = False
        return self.fitness

class PopulationManager:
    """Manages the population of circle packings."""
    
    def __init__(self, pop_size: int, n_circles: int):
        self.pop_size = pop_size
        self.n_circles = n_circles
        self.population: List[CirclePack] = []
        self.validator = CircleValidator()
        
    def create_initial_population(self) -> None:
        """Create diverse initial population with improved initialization."""
        self.population = []
        
        # Multi-scale grid initialization
        for i in range(self.pop_size):
            circles = self._generate_grid_based_config()
            pack = CirclePack(circles)
            pack.calculate_fitness(self.validator)
            self.population.append(pack)
    
    def _generate_grid_based_config(self) -> np.ndarray:
        """Generate grid-based initial configuration with better spatial distribution."""
        circles = np.zeros((self.n_circles, 3))
        
        # Multi-scale grid approach
        grid_sizes = [int(np.ceil(np.sqrt(self.n_circles))), 
                     max(2, int(np.ceil(np.sqrt(self.n_circles * 1.5))) + 1)]
        
        # Try different grid approaches
        grid_size = grid_sizes[random.randint(0, len(grid_sizes)-1)]
        
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        
        # Place circles in grid pattern with perturbations
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= self.n_circles:
                    break
                x = (i + 1) * spacing_x
                y = (j + 1) * spacing_y
                
                # Add systematic perturbation based on position
                perturbation_factor = 0.1
                x += np.random.uniform(-spacing_x * perturbation_factor, 
                                     spacing_x * perturbation_factor)
                y += np.random.uniform(-spacing_y * perturbation_factor, 
                                     spacing_y * perturbation_factor)
                
                # Ensure within bounds
                x = np.clip(x, spacing_x * 0.1, 1 - spacing_x * 0.1)
                y = np.clip(y, spacing_y * 0.1, 1 - spacing_y * 0.1)
                
                # Radius based on available space
                max_radius = min(x, 1-x, y, 1-y)
                r = min(max_radius * 0.2, 0.15)
                r *= np.random.uniform(0.7, 1.3)  # Randomize slightly
                
                circles[count] = [x, y, r]
                count += 1
            if count >= self.n_circles:
                break
        
        # Fill remaining circles with random valid placements
        for i in range(count, self.n_circles):
            max_attempts = 1000
            attempts = 0
            placed = False
            
            while not placed and attempts < max_attempts:
                # Use triangular distribution for better placement
                x = np.random.triangular(0.05, 0.5, 0.95)
                y = np.random.triangular(0.05, 0.5, 0.95)
                r = np.random.loguniform(0.005, 0.15)  # Log uniform for radius
                
                # Check validity
                if (x - r >= 0 and x + r <= 1 and y - r >= 0 and y + r <= 1):
                    # Check overlap with existing circles
                    valid_placement = True
                    for j in range(i):
                        existing_x, existing_y, existing_r = circles[j]
                        distance = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                        if distance < r + existing_r:
                            valid_placement = False
                            break
                    
                    if valid_placement:
                        circles[i] = [x, y, r]
                        placed = True
                attempts += 1
                
            # Fallback if still not placed
            if not placed:
                x = np.random.uniform(0.1, 0.9)
                y = np.random.uniform(0.1, 0.9)
                r = min(0.05, min(x, 1-x, y, 1-y) * 0.3)
                circles[i] = [x, y, r]
        
        return circles
    
    def get_best_individual(self) -> Optional[CirclePack]:
        """Return the best individual in the population."""
        if not self.population:
            return None
        return max(self.population, key=lambda p: p.fitness)

class EvolutionaryEngine:
    """Handles the evolutionary operations."""
    
    def __init__(self, pop_size: int, n_circles: int, generations: int):
        self.pop_size = pop_size
        self.n_circles = n_circles
        self.generations = generations
        self.validator = CircleValidator()
        self.population_manager = PopulationManager(pop_size, n_circles)
        
    def tournament_selection(self, population: List[CirclePack], 
                           tournament_size: int = 3) -> CirclePack:
        """Tournament selection for parent selection."""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_packs = [population[i] for i in tournament_indices]
        return max(tournament_packs, key=lambda p: p.fitness)
    
    def crossover(self, parent1: CirclePack, parent2: CirclePack) -> CirclePack:
        """Uniform crossover between two parents."""
        child_circles = parent1.circles.copy()
        
        # Uniform crossover for each circle
        for i in range(self.n_circles):
            if np.random.random() < 0.5:
                child_circles[i] = parent2.circles[i].copy()
        
        child = CirclePack(child_circles)
        return child
    
    def mutate(self, individual: CirclePack, mutation_rate: float = 0.1) -> CirclePack:
        """Apply mutation to an individual."""
        mutated_circles = individual.circles.copy()
        
        for i in range(self.n_circles):
            if np.random.random() < mutation_rate:
                # Apply different mutation strategies
                if np.random.random() < 0.7:  # 70% chance to mutate position
                    # Mutate position with larger perturbations for exploration
                    mutated_circles[i, 0] += np.random.normal(0, 0.03)
                    mutated_circles[i, 1] += np.random.normal(0, 0.03)
                    
                    # Boundary correction
                    mutated_circles[i, 0] = np.clip(mutated_circles[i, 0], 
                                                   mutated_circles[i, 2], 
                                                   1 - mutated_circles[i, 2])
                    mutated_circles[i, 1] = np.clip(mutated_circles[i, 1], 
                                                   mutated_circles[i, 2], 
                                                   1 - mutated_circles[i, 2])
                else:  # 30% chance to mutate radius
                    # Log-normal mutation for radius to maintain positivity
                    old_r = mutated_circles[i, 2]
                    new_r = np.exp(np.log(old_r) + np.random.normal(0, 0.2))
                    mutated_circles[i, 2] = np.clip(new_r, 0.001, 0.2)
        
        mutated_pack = CirclePack(mutated_circles)
        return mutated_pack
    
    def evolve(self) -> CirclePack:
        """Main evolution loop."""
        # Initialize population
        self.population_manager.create_initial_population()
        
        best_found = None
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitnesses = []
            for pack in self.population_manager.population:
                fitness = pack.calculate_fitness(self.validator)
                fitnesses.append(fitness)
            
            # Track best individual
            best_in_gen = self.population_manager.get_best_individual()
            if best_in_gen and (best_found is None or best_in_gen.fitness > best_found.fitness):
                best_found = best_in_gen
            
            # Print progress
            if generation % 50 == 0:
                current_best = self.population_manager.get_best_individual()
                if current_best:
                    print(f"Generation {generation}: Best fitness = {current_best.fitness:.6f}")
            
            # Create new population
            new_population = []
            
            # Elitism: keep top individuals
            sorted_packs = sorted(self.population_manager.population, 
                                key=lambda p: p.fitness, reverse=True)
            elite_count = max(1, self.pop_size // 10)
            new_population.extend(sorted_packs[:elite_count])
            
            # Generate offspring
            while len(new_population) < self.pop_size:
                # Selection
                parent1 = self.tournament_selection(self.population_manager.population)
                parent2 = self.tournament_selection(self.population_manager.population)
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutation with adaptive rate
                adaptive_rate = 0.1 * (1.0 - generation / self.generations)
                child = self.mutate(child, adaptive_rate)
                
                new_population.append(child)
            
            # Trim to exact population size
            self.population_manager.population = new_population[:self.pop_size]
        
        return best_found if best_found else self.population_manager.get_best_individual()

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    start_time = time.time()
    
    try:
        # Create evolutionary engine
        engine = EvolutionaryEngine(pop_size=50, n_circles=26, generations=500)
        
        # Evolve solution
        best_solution = engine.evolve()
        
        # Return result
        if best_solution and best_solution.is_valid:
            circles = best_solution.circles
        else:
            # Fallback to simple heuristics if needed
            circles = np.zeros((26, 3))
            for i in range(26):
                circles[i] = [0.5, 0.5, 0.01]
        
        end_time = time.time()
        eval_time = end_time - start_time
        print(f"Evolution completed in {eval_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error during evolution: {e}")
        # Fallback to simple initialization
        circles = np.zeros((26, 3))
        print("Using fallback solution due to error")
    
    return circles

# EVOLVE-BLOCK-END