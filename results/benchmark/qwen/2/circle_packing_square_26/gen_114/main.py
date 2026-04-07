# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import List, Tuple
import time

# Global constants
POPULATION_SIZE = 100
GENERATIONS = 200
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.1
ELITISM_COUNT = 5
MAX_ATTEMPTS = 1000

class CircleValidator:
    """Handles all circle placement validation logic."""
    
    @staticmethod
    def is_valid_placement(circles: np.ndarray) -> bool:
        """Check if all circles are within bounds and don't overlap."""
        n = len(circles)
        
        # Vectorized containment check
        radii = circles[:, 2]
        positions = circles[:, :2]
        
        if np.any(radii <= 0) or np.any(positions[:, 0] < radii) or np.any(positions[:, 0] > 1 - radii) or \
           np.any(positions[:, 1] < radii) or np.any(positions[:, 1] > 1 - radii):
            return False

        # Check overlap constraints using KDTree for efficiency
        tree = cKDTree(positions)
        
        # Vectorized overlap checking - process in batches to reduce overhead
        for i in range(n):
            x, y, r = circles[i]
            # Find nearby circles (within 2*r distance)
            indices = tree.query_ball_point([x, y], 2*r)
            for j in indices:
                if i != j:
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if distance < r + r2:
                        return False

        return True

    @staticmethod
    def validate_and_score(circles: np.ndarray) -> Tuple[bool, float]:
        """Validate placement and return fitness score."""
        if CircleValidator.is_valid_placement(circles):
            return True, np.sum(circles[:, 2])
        else:
            return False, -1000000.0

class CircleInitializer:
    """Handles circle initialization strategies."""
    
    @staticmethod
    def create_grid_initialization(num_circles: int, rows: int, cols: int) -> np.ndarray:
        """Create a grid-based initialization for circles."""
        circles = np.zeros((num_circles, 3))
        
        # Create a grid of positions
        grid_positions = []
        for i in range(rows):
            for j in range(cols):
                if len(grid_positions) >= num_circles:
                    break
                x = (j + 0.5) / cols
                y = (i + 0.5) / rows
                grid_positions.append((x, y))
        
        # Fill circles with grid positions
        for i in range(num_circles):
            if i < len(grid_positions):
                x, y = grid_positions[i]
                # Small random offset
                x += (random.random() - 0.5) * 0.05
                y += (random.random() - 0.5) * 0.05
                # Small random radius
                r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
                circles[i] = [x, y, r]
            else:
                # Random placement for extra circles
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
                r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
                circles[i] = [x, y, r]
                
        return circles
    
    @classmethod
    def create_initial_population(cls, pop_size: int, num_circles: int) -> List[np.ndarray]:
        """Create initial population with improved initialization."""
        population = []
        
        # Try multiple initialization strategies
        strategies = [
            lambda: cls.create_grid_initialization(num_circles, 
                                                   int(np.ceil(np.sqrt(num_circles))), 
                                                   int(np.ceil(num_circles / np.ceil(np.sqrt(num_circles))))),
            lambda: cls.create_grid_initialization(num_circles, 5, 6),
            lambda: cls.create_grid_initialization(num_circles, 6, 5),
            lambda: cls.create_grid_initialization(num_circles, 4, 7)
        ]
        
        for _ in range(pop_size):
            # Try different initialization strategies
            success = False
            attempts = 0
            
            while not success and attempts < MAX_ATTEMPTS:
                # Select random initialization strategy
                init_func = random.choice(strategies)
                circles = init_func()
                
                # Try to improve the configuration with local optimization
                improved = CircleOptimizer.optimize_local(circles)
                
                is_valid, _ = CircleValidator.validate_and_score(improved)
                if is_valid:
                    circles = improved
                    success = True
                else:
                    attempts += 1
                    
            if not success:
                # Fallback to random initialization
                circles = np.zeros((num_circles, 3))
                for i in range(num_circles):
                    x = random.uniform(0.05, 0.95)
                    y = random.uniform(0.05, 0.95)
                    r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
                    circles[i] = [x, y, r]
                    
            population.append(circles)
            
        return population

class CircleOptimizer:
    """Handles local optimization of circle configurations."""
    
    @staticmethod
    def optimize_local(circles: np.ndarray) -> np.ndarray:
        """Apply enhanced local improvement to maximize sum of radii."""
        n = len(circles)
        circles_copy = circles.copy()
        
        # More sophisticated optimization with early termination
        improvement_threshold = 1e-6
        max_iterations = 150
        improvement_count = 0
        
        # Pre-compute distances for efficiency
        positions = circles_copy[:, :2]
        radii = circles_copy[:, 2]
        
        for iteration in range(max_iterations):
            improved = False
            
            # Process circles in random order for better exploration
            circle_order = list(range(n))
            random.shuffle(circle_order)
            
            for i in circle_order:
                x, y, r = circles_copy[i]
                
                # Calculate maximum possible radius at current position
                max_r = min(x, 1-x, y, 1-y)
                
                # Try to increase radius as much as possible while avoiding overlaps
                if max_r > r + improvement_threshold:
                    # Try to increase radius as much as possible
                    new_r = max_r
                    
                    # Check overlap constraints with existing circles
                    valid_radius = True
                    for j in range(n):
                        if i != j:
                            x2, y2, r2 = circles_copy[j]
                            distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                            if distance < new_r + r2:
                                valid_radius = False
                                break
                    
                    if valid_radius:
                        circles_copy[i, 2] = new_r
                        improved = True
                    else:
                        # Try a smaller increase
                        step_size = 0.0005
                        test_r = min(r + step_size, max_r)
                        while test_r > r + improvement_threshold and not valid_radius:
                            valid_radius = True
                            for j in range(n):
                                if i != j:
                                    x2, y2, r2 = circles_copy[j]
                                    distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                                    if distance < test_r + r2:
                                        valid_radius = False
                                        break
                            
                            if valid_radius:
                                circles_copy[i, 2] = test_r
                                improved = True
                                break
                            else:
                                test_r -= step_size
                
                # Position adjustment with limited frequency
                if iteration % 8 == 0 and improved:
                    # Simple gradient-based adjustment - calculate repulsive forces
                    force_x, force_y = 0.0, 0.0
                    current_r = r
                    
                    # Calculate repulsive forces from neighbors
                    for j in range(n):
                        if i != j:
                            x2, y2, r2 = circles_copy[j]
                            distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                            if distance < current_r + r2:
                                # Repulsive force to move away from overlapping circle
                                dx = x - x2
                                dy = y - y2
                                dist = np.sqrt(dx*dx + dy*dy)
                                if dist > 0:
                                    # Normalize and apply repulsive force
                                    force_x += dx / dist * (current_r + r2 - distance) * 0.02
                                    force_y += dy / dist * (current_r + r2 - distance) * 0.02
                    
                    # Apply forces to position
                    new_x = x + force_x * 0.5
                    new_y = y + force_y * 0.5
                    
                    # Keep within bounds
                    new_x = np.clip(new_x, current_r, 1-current_r)
                    new_y = np.clip(new_y, current_r, 1-current_r)
                    
                    # Check if new position is valid
                    temp_circles = circles_copy.copy()
                    temp_circles[i, 0] = new_x
                    temp_circles[i, 1] = new_y
                    
                    valid_position = True
                    for j in range(n):
                        if i != j:
                            x2, y2, r2 = temp_circles[j]
                            distance = np.sqrt((new_x - x2)**2 + (new_y - y2)**2)
                            if distance < current_r + r2:
                                valid_position = False
                                break
                    
                    if valid_position:
                        circles_copy[i, 0] = new_x
                        circles_copy[i, 1] = new_y
                        improved = True
            
            if not improved:
                improvement_count += 1
                if improvement_count > 10:  # Early termination if no improvement for 10 iterations
                    break
            else:
                improvement_count = 0
                
        return circles_copy

class EvolutionOperator:
    """Handles evolutionary operations: selection, crossover, mutation."""
    
    @staticmethod
    def tournament_selection(population: List[np.ndarray], fitnesses: List[float],
                             tournament_size: int) -> np.ndarray:
        """Select individual using tournament selection."""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_index].copy()

    @staticmethod
    def uniform_crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Perform uniform crossover between two parents."""
        n = len(parent1)
        child = np.zeros_like(parent1)
        
        # Use vectorized operations for crossover
        mask = np.random.random(n) < 0.5
        child[mask] = parent1[mask]
        child[~mask] = parent2[~mask]
        
        return child

    @staticmethod
    def mutate(individual: np.ndarray) -> np.ndarray:
        """Apply mutation to an individual."""
        mutated = individual.copy()
        n = len(mutated)

        # Vectorized mutation
        mutation_mask = np.random.random(n) < MUTATION_RATE
        
        # Separate position and radius mutations
        pos_mutations = mutation_mask & (np.random.random(n) < 0.5)
        rad_mutations = mutation_mask & ~pos_mutations
        
        # Mutate positions
        if np.any(pos_mutations):
            mutated[pos_mutations, 0] += (np.random.random(np.sum(pos_mutations)) - 0.5) * 0.1
            mutated[pos_mutations, 1] += (np.random.random(np.sum(pos_mutations)) - 0.5) * 0.1
            
            # Keep within bounds
            mutated[pos_mutations, 0] = np.clip(mutated[pos_mutations, 0], 0.01, 0.99)
            mutated[pos_mutations, 1] = np.clip(mutated[pos_mutations, 1], 0.01, 0.99)
        
        # Mutate radii
        if np.any(rad_mutations):
            mutated[rad_mutations, 2] += (np.random.random(np.sum(rad_mutations)) - 0.5) * 0.05
            # Ensure positive radius
            mutated[rad_mutations, 2] = np.maximum(0.001, mutated[rad_mutations, 2])
        
        # Repair any constraint violations
        repaired = CircleRepair.repair_constraints(mutated)
        return repaired

class CircleRepair:
    """Handles constraint repair operations."""
    
    @staticmethod
    def repair_constraints(circles: np.ndarray) -> np.ndarray:
        """Repair any constraint violations."""
        repaired = circles.copy()
        n = len(repaired)
        
        # Ensure all circles are within bounds
        for i in range(n):
            x, y, r = repaired[i]
            r = max(0.001, r)
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            repaired[i] = [x, y, r]
        
        # Apply constraint repair iteration with reduced iterations for performance
        for _ in range(5):  # Reduced from 10 for performance
            any_changes = False
            for i in range(n):
                x, y, r = repaired[i]
                # Check overlaps and adjust if needed
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = repaired[j]
                        distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                        min_distance = r + r2
                        if distance < min_distance:
                            # Move circle away from overlapping one
                            dx = x2 - x
                            dy = y2 - y
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist > 0:
                                factor = (min_distance - distance) / dist * 0.1
                                x += dx * factor
                                y += dy * factor
                                any_changes = True
                
                # Keep within bounds
                r = max(0.001, r)
                x = np.clip(x, r, 1-r)
                y = np.clip(y, r, 1-r)
                repaired[i] = [x, y, r]
            
            if not any_changes:
                break
        
        return repaired

class CirclePack26:
    """Main controller class for the circle packing optimization."""
    
    def __init__(self):
        self.best_solution = None
        self.best_fitness = -np.inf
        self.population = []
    
    def evaluate_population(self, population: List[np.ndarray]) -> List[float]:
        """Evaluate fitness for entire population."""
        return [CircleValidator.validate_and_score(individual)[1] for individual in population]
    
    def evolve(self) -> np.ndarray:
        """Main evolutionary loop."""
        # Create initial population
        self.population = CircleInitializer.create_initial_population(POPULATION_SIZE, 26)
        
        for generation in range(GENERATIONS):
            # Evaluate fitness of each individual
            fitnesses = self.evaluate_population(self.population)
            
            # Track best solution so far
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > self.best_fitness:
                self.best_fitness = fitnesses[max_fitness_idx]
                self.best_solution = self.population[max_fitness_idx].copy()
            
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitnesses)[-ELITISM_COUNT:]
            elites = [self.population[i].copy() for i in elite_indices]
            
            # Create new population
            new_population = elites.copy()
            
            # Generate offspring through selection, crossover, and mutation
            while len(new_population) < POPULATION_SIZE:
                # Selection
                parent1 = EvolutionOperator.tournament_selection(self.population, fitnesses, TOURNAMENT_SIZE)
                parent2 = EvolutionOperator.tournament_selection(self.population, fitnesses, TOURNAMENT_SIZE)
                
                # Crossover
                child = EvolutionOperator.uniform_crossover(parent1, parent2)
                
                # Mutation
                child = EvolutionOperator.mutate(child)
                
                # Add to new population
                new_population.append(child)
            
            self.population = new_population[:POPULATION_SIZE]
        
        # Return the best solution found
        if self.best_solution is not None:
            return self.best_solution
        else:
            # Fallback to final population if no valid solution was found
            return self.population[0]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates
                 of the i-th circle of radius r.
    """
    packer = CirclePack26()
    return packer.evolve()

# EVOLVE-BLOCK-END