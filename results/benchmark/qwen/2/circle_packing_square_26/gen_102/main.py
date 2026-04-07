# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List, Optional
import time

# Global constants
POPULATION_SIZE = 100
GENERATIONS = 200
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.1
ELITISM_COUNT = 5
MAX_ATTEMPTS = 1000

class CirclePacker:
    def __init__(self, num_circles: int = 26):
        self.num_circles = num_circles
        self.best_solution = None
        self.best_fitness = -np.inf
        
    def validate_placement(self, circles: np.ndarray) -> bool:
        """Check if all circles are within bounds and don't overlap"""
        n = len(circles)
        
        # Check containment constraints
        for i in range(n):
            x, y, r = circles[i]
            if r <= 0 or x < r or x > 1-r or y < r or y > 1-r:
                return False

        # Check overlap constraints using KDTree for efficiency
        points = circles[:, :2]
        tree = cKDTree(points)

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

    def calculate_sum_radii(self, circles: np.ndarray) -> float:
        """Calculate sum of all radii"""
        return np.sum(circles[:, 2])
    
    def create_grid_initialization(self, rows: int, cols: int) -> np.ndarray:
        """Create a grid-based initialization for circles"""
        circles = np.zeros((self.num_circles, 3))
        
        # Create a grid of positions
        grid_positions = []
        for i in range(rows):
            for j in range(cols):
                if len(grid_positions) >= self.num_circles:
                    break
                x = (j + 0.5) / cols
                y = (i + 0.5) / rows
                grid_positions.append((x, y))
        
        # Fill circles with grid positions
        for i in range(self.num_circles):
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
    
    def create_initial_population(self, pop_size: int) -> List[np.ndarray]:
        """Create initial population with improved initialization"""
        population = []
        
        # Try multiple initialization strategies
        strategies = [
            lambda: self.create_grid_initialization(int(np.ceil(np.sqrt(self.num_circles))), 
                                                   int(np.ceil(self.num_circles / np.ceil(np.sqrt(self.num_circles))))),
            lambda: self.create_grid_initialization(5, 6),
            lambda: self.create_grid_initialization(6, 5),
            lambda: self.create_grid_initialization(4, 7)
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
                improved = self.local_improvement(circles)
                
                if self.validate_placement(improved):
                    circles = improved
                    success = True
                else:
                    attempts += 1
                    
            if not success:
                # Fallback to random initialization
                circles = np.zeros((self.num_circles, 3))
                for i in range(self.num_circles):
                    x = random.uniform(0.05, 0.95)
                    y = random.uniform(0.05, 0.95)
                    r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
                    circles[i] = [x, y, r]
                    
            population.append(circles)
            
        return population

    def local_improvement(self, circles: np.ndarray) -> np.ndarray:
        """Apply enhanced local improvement to maximize sum of radii"""
        n = len(circles)
        circles_copy = circles.copy()
        
        # More sophisticated optimization using iterative improvement
        for iteration in range(100):
            improved = False
            
            # Try to improve each circle systematically
            for i in range(n):
                x, y, r = circles_copy[i]
                
                # Calculate maximum possible radius at current position
                max_r = min(x, 1-x, y, 1-y)
                
                # If we can increase radius, try to do so
                if max_r > r + 1e-6:
                    # Try to increase radius as much as possible while avoiding overlaps
                    new_r = max_r
                    
                    # Check overlap constraints with all other circles
                    valid_radius = True
                    for j in range(n):
                        if i != j:
                            x2, y2, r2 = circles_copy[j]
                            distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                            if distance < new_r + r2:
                                valid_radius = False
                                break
                    
                    # If we can increase radius, do so
                    if valid_radius:
                        circles_copy[i, 2] = new_r
                        improved = True
                    else:
                        # Try a smaller increase in a more systematic way
                        step_size = 0.001
                        test_r = min(r + step_size, max_r)
                        while test_r > r + 1e-6 and not valid_radius:
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
                
                # Try to slightly adjust position for better packing
                if iteration % 10 == 0:  # Only adjust position occasionally
                    best_x, best_y = x, y
                    best_r = r
                    best_score = self.calculate_sum_radii(circles_copy)
                    
                    # Try small shifts in 4 directions
                    for dx in [-0.005, 0, 0.005]:
                        for dy in [-0.005, 0, 0.005]:
                            if abs(dx) + abs(dy) == 0:
                                continue
                                
                            test_x = x + dx
                            test_y = y + dy
                            
                            # Keep within bounds
                            if test_x < r or test_x > 1-r or test_y < r or test_y > 1-r:
                                continue
                                
                            # Check validity of new configuration
                            valid = True
                            temp_circles = circles_copy.copy()
                            temp_circles[i, 0] = test_x
                            temp_circles[i, 1] = test_y
                            
                            for j in range(n):
                                if i != j:
                                    x2, y2, r2 = temp_circles[j]
                                    distance = np.sqrt((test_x - x2)**2 + (test_y - y2)**2)
                                    if distance < r + r2:
                                        valid = False
                                        break
                                        
                            if valid:
                                # Calculate how much we would gain by changing to this position
                                temp_circles[i, 2] = r  # reset radius to original
                                test_score = self.calculate_sum_radii(temp_circles)
                                if test_score > best_score:
                                    best_score = test_score
                                    best_x, best_y = test_x, test_y
                    
                    # Update position if beneficial
                    if best_x != x or best_y != y:
                        circles_copy[i, 0] = best_x
                        circles_copy[i, 1] = best_y
                        improved = True
                        
            # Stop if no significant improvement
            if not improved:
                break
                
        return circles_copy

    def tournament_selection(self, population: List[np.ndarray], fitnesses: List[float],
                             tournament_size: int) -> np.ndarray:
        """Select individual using tournament selection"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_index].copy()

    def uniform_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Perform uniform crossover between two parents"""
        n = len(parent1)
        child = np.zeros_like(parent1)

        for i in range(n):
            # Each element has 50% chance of coming from parent1 or parent2
            if random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]

        return child

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """Apply mutation to an individual"""
        mutated = individual.copy()
        n = len(mutated)

        for i in range(n):
            if random.random() < MUTATION_RATE:
                # Mutate either position or radius
                if random.random() < 0.5:
                    # Mutate position
                    mutated[i, 0] += (random.random() - 0.5) * 0.1
                    mutated[i, 1] += (random.random() - 0.5) * 0.1

                    # Keep within bounds
                    mutated[i, 0] = np.clip(mutated[i, 0], 0.01, 0.99)
                    mutated[i, 1] = np.clip(mutated[i, 1], 0.01, 0.99)
                else:
                    # Mutate radius
                    mutated[i, 2] += (random.random() - 0.5) * 0.05

                    # Ensure positive radius
                    mutated[i, 2] = max(0.001, mutated[i, 2])

        # Repair any constraint violations
        repaired = self.repair_constraints(mutated)
        return repaired

    def repair_constraints(self, circles: np.ndarray) -> np.ndarray:
        """Repair any constraint violations"""
        repaired = circles.copy()
        n = len(repaired)

        # Ensure all circles are within bounds
        for i in range(n):
            x, y, r = repaired[i]
            r = max(0.001, r)
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            repaired[i] = [x, y, r]

        # Apply constraint repair iteration
        for _ in range(10):
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

    def evolve(self) -> np.ndarray:
        """Main evolutionary loop"""
        # Create initial population
        population = self.create_initial_population(POPULATION_SIZE)

        for generation in range(GENERATIONS):
            # Evaluate fitness of each individual
            fitnesses = []
            for individual in population:
                if self.validate_placement(individual):
                    fitness = self.calculate_sum_radii(individual)
                    fitnesses.append(fitness)
                else:
                    # Invalid solutions get very low fitness
                    fitnesses.append(-1000000)

            # Track best solution so far
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > self.best_fitness:
                self.best_fitness = fitnesses[max_fitness_idx]
                self.best_solution = population[max_fitness_idx].copy()

            # Elitism: keep best individuals
            elite_indices = np.argsort(fitnesses)[-ELITISM_COUNT:]
            elites = [population[i].copy() for i in elite_indices]

            # Create new population
            new_population = elites.copy()

            # Generate offspring through selection, crossover, and mutation
            while len(new_population) < POPULATION_SIZE:
                # Selection
                parent1 = self.tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
                parent2 = self.tournament_selection(population, fitnesses, TOURNAMENT_SIZE)

                # Crossover
                child = self.uniform_crossover(parent1, parent2)

                # Mutation
                child = self.mutate(child)

                # Add to new population
                new_population.append(child)

            population = new_population[:POPULATION_SIZE]

        # Return the best solution found
        if self.best_solution is not None:
            return self.best_solution
        else:
            # Fallback to final population if no valid solution was found
            return population[0]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates
                 of the i-th circle of radius r.
    """
    packer = CirclePacker(26)
    return packer.evolve()

# EVOLVE-BLOCK-END