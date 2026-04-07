# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import time

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

class OptimizationStrategy(Enum):
    GREEDY = "greedy"
    LOCAL_SEARCH = "local_search"
    PHYSICS_BASED = "physics_based"

@dataclass
class CircleConfig:
    """Configuration parameters for circle packing optimization"""
    num_circles: int = 26
    population_size: int = 50
    generations: int = 100
    elite_size: int = 5
    tournament_size: int = 3
    mutation_rate: float = 0.1
    adaptive_mutation_threshold: int = 25
    refinement_iterations: int = 100

@dataclass
class EvolutionStats:
    """Statistics tracking for evolution process"""
    best_fitness: float = -float('inf')
    generation_times: List[float] = None
    fitness_history: List[float] = None
    
    def __post_init__(self):
        if self.generation_times is None:
            self.generation_times = []
        if self.fitness_history is None:
            self.fitness_history = []

class CircleValidator:
    """Handles circle validation and constraint checking"""
    
    @staticmethod
    def validate_circles(circles: np.ndarray) -> bool:
        """
        Validates that all circles are within bounds and don't overlap.
        
        Args:
            circles: np.array of shape (n, 3) where each row is (x, y, r)
            
        Returns:
            True if all circles are valid, False otherwise
        """
        n = len(circles)
        
        # Check containment constraints
        for i in range(n):
            x, y, r = circles[i]
            if x < r or x > 1 - r or y < r or y > 1 - r:
                return False

        # Check overlap constraints using KDTree for efficiency
        points = circles[:, :2]  # Get (x, y) coordinates
        tree = cKDTree(points)

        # For each circle, check overlap with others
        for i in range(n):
            x1, y1, r1 = circles[i]
            # Find nearby circles (within 2*(r1+r2) distance)
            nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))

            # Check overlap with each nearby circle
            for j in nearby_indices:
                if i != j:
                    x2, y2, r2 = circles[j]
                    distance_sq = (x1 - x2)**2 + (y1 - y2)**2
                    min_distance_sq = (r1 + r2)**2

                    if distance_sq < min_distance_sq:
                        return False

        return True

    @staticmethod
    def calculate_sum_radii(circles: np.ndarray) -> float:
        """Calculate the sum of all radii"""
        return np.sum(circles[:, 2])

class Initializer:
    """Handles population initialization strategies"""
    
    @staticmethod
    def create_initial_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
        """Create initial population with enhanced diversity"""
        population = []
        
        # Multiple initialization strategies
        strategies = [
            Initializer._grid_strategy,
            Initializer._corner_strategy, 
            Initializer._random_strategy,
            Initializer._distributed_strategy
        ]
        
        for _ in range(pop_size):
            strategy = random.choice(strategies)
            circles = strategy(n_circles)
            # Apply basic repulsion to reduce overlaps
            circles = Initializer._apply_repulsion(circles, iterations=30)
            # Add small perturbations
            circles = Initializer._perturb_configuration(circles)
            population.append(circles)
        
        return population

    @staticmethod
    def _grid_strategy(n_circles: int) -> np.ndarray:
        """Grid-based initialization"""
        circles = np.zeros((n_circles, 3))
        
        # Try different grid configurations
        grid_configs = [(3, 3), (4, 4), (5, 5), (6, 6)]
        grid_h, grid_w = random.choice(grid_configs)
        
        actual_cols = min(grid_w, n_circles // grid_h + (1 if n_circles % grid_h else 0))
        actual_rows = min(grid_h, (n_circles + actual_cols - 1) // actual_cols)
        
        spacing_x = 1.0 / (actual_cols + 1)
        spacing_y = 1.0 / (actual_rows + 1)
        
        idx = 0
        for i in range(actual_rows):
            for j in range(actual_cols):
                if idx >= n_circles:
                    break
                x = (j + 1) * spacing_x + np.random.uniform(-spacing_x/4, spacing_x/4)
                y = (i + 1) * spacing_y + np.random.uniform(-spacing_y/4, spacing_y/4)
                x = np.clip(x, 0.01, 0.99)
                y = np.clip(y, 0.01, 0.99)
                min_dist_to_edge = min(x, 1-x, y, 1-y)
                r = min(0.05, min_dist_to_edge * np.random.uniform(0.7, 0.95))
                circles[idx] = [x, y, r]
                idx += 1
            if idx >= n_circles:
                break
        
        # Fill remaining
        for i in range(idx, n_circles):
            x = np.random.uniform(0.01, 0.99)
            y = np.random.uniform(0.01, 0.99)
            min_dist_to_edge = min(x, 1-x, y, 1-y)
            r = min(0.05, min_dist_to_edge * np.random.uniform(0.5, 0.8))
            circles[i] = [x, y, r]
        
        return circles

    @staticmethod
    def _corner_strategy(n_circles: int) -> np.ndarray:
        """Corner and edge initialization"""
        circles = np.zeros((n_circles, 3))
        
        # Place in corners for larger potential radii
        corner_positions = [(0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9)]
        num_corners = min(4, n_circles)
        
        for i in range(num_corners):
            x, y = corner_positions[i]
            x += np.random.uniform(-0.05, 0.05)
            y += np.random.uniform(-0.05, 0.05)
            x = np.clip(x, 0.01, 0.99)
            y = np.clip(y, 0.01, 0.99)
            min_dist_to_edge = min(x, 1-x, y, 1-y)
            r = min(0.07, min_dist_to_edge * np.random.uniform(0.7, 0.95))
            circles[i] = [x, y, r]
        
        # Fill remaining with center
        center_start = num_corners
        for i in range(center_start, n_circles):
            x = np.random.uniform(0.3, 0.7)
            y = np.random.uniform(0.3, 0.7)
            min_dist_to_edge = min(x, 1-x, y, 1-y)
            r = min(0.04, min_dist_to_edge * np.random.uniform(0.4, 0.7))
            circles[i] = [x, y, r]
        
        return circles

    @staticmethod
    def _random_strategy(n_circles: int) -> np.ndarray:
        """Pure random initialization"""
        circles = np.zeros((n_circles, 3))
        for i in range(n_circles):
            x = np.random.uniform(0.01, 0.99)
            y = np.random.uniform(0.01, 0.99)
            min_dist_to_edge = min(x, 1-x, y, 1-y)
            r = min(0.06, min_dist_to_edge * np.random.uniform(0.3, 0.8))
            circles[i] = [x, y, r]
        return circles

    @staticmethod
    def _distributed_strategy(n_circles: int) -> np.ndarray:
        """Distributed initialization across regions"""
        circles = np.zeros((n_circles, 3))
        
        regions = [
            (0.05, 0.35, 0.05, 0.35),  # bottom left
            (0.65, 0.95, 0.05, 0.35),  # bottom right
            (0.05, 0.35, 0.65, 0.95),  # top left
            (0.65, 0.95, 0.65, 0.95),  # top right
            (0.35, 0.65, 0.35, 0.65),  # center
        ]
        
        circles_per_region = n_circles // 5 + (1 if n_circles % 5 else 0)
        
        for region_idx, (x_min, x_max, y_min, y_max) in enumerate(regions):
            if region_idx * circles_per_region >= n_circles:
                break
            region_circles = min(circles_per_region, n_circles - region_idx * circles_per_region)
            for i in range(region_circles):
                if region_idx * circles_per_region + i >= n_circles:
                    break
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
                min_dist_to_edge = min(x, 1-x, y, 1-y)
                r = min(0.06, min_dist_to_edge * np.random.uniform(0.5, 0.9))
                circles[region_idx * circles_per_region + i] = [x, y, r]
        
        return circles

    @staticmethod
    def _apply_repulsion(circles: np.ndarray, iterations: int = 30) -> np.ndarray:
        """Apply physics-inspired repulsion to reduce overlaps"""
        repelled = circles.copy()
        n = len(repelled)
        
        points = repelled[:, :2]
        tree = cKDTree(points)
        
        for _ in range(iterations):
            any_changes = False
            for i in range(n):
                x1, y1, r1 = repelled[i]
                nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))
                
                for j in nearby_indices:
                    if i != j:
                        x2, y2, r2 = repelled[j]
                        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        min_distance = r1 + r2
                        
                        if distance < min_distance:
                            if distance > 0.001:
                                dx = (x1 - x2) / distance
                                dy = (y1 - y2) / distance
                                
                                overlap = min_distance - distance
                                force_scale = 0.8 * (1 - distance/min_distance) + 0.2
                                move_amount = overlap * force_scale * 0.5
                                
                                repelled[i, 0] += dx * move_amount
                                repelled[i, 1] += dy * move_amount
                                repelled[j, 0] -= dx * move_amount
                                repelled[j, 1] -= dy * move_amount
                                
                                # Clamp to bounds
                                repelled[i, 0] = np.clip(repelled[i, 0], r1, 1 - r1)
                                repelled[i, 1] = np.clip(repelled[i, 1], r1, 1 - r1)
                                repelled[j, 0] = np.clip(repelled[j, 0], r2, 1 - r2)
                                repelled[j, 1] = np.clip(repelled[j, 1], r2, 1 - r2)
                            
                            any_changes = True
            
            if not any_changes:
                break
        
        return repelled

    @staticmethod
    def _perturb_configuration(circles: np.ndarray) -> np.ndarray:
        """Apply small random perturbations to add diversity"""
        perturbed = circles.copy()
        n = len(perturbed)
        
        for i in range(n):
            if np.random.rand() < 0.6:
                if np.random.rand() < 0.7:
                    # Perturb position
                    perturbed[i, 0] += np.random.uniform(-0.005, 0.005)
                    perturbed[i, 1] += np.random.uniform(-0.005, 0.005)
                    perturbed[i, 0] = np.clip(perturbed[i, 0], perturbed[i, 2], 1 - perturbed[i, 2])
                    perturbed[i, 1] = np.clip(perturbed[i, 1], perturbed[i, 2], 1 - perturbed[i, 2])
                else:
                    # Perturb radius
                    perturbed[i, 2] += np.random.uniform(-0.003, 0.003)
                    perturbed[i, 2] = max(0.001, perturbed[i, 2])
        
        return perturbed

class Selector:
    """Handles selection operations in genetic algorithm"""
    
    @staticmethod
    def tournament_selection(population: List[np.ndarray], fitnesses: List[float],
                             tournament_size: int = 3) -> np.ndarray:
        """Select parent using tournament selection"""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_index].copy()

class Operator:
    """Handles genetic operators"""
    
    @staticmethod
    def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Uniform crossover between two parents"""
        child = parent1.copy()
        mask = np.random.rand(*parent1.shape) > 0.5
        child[mask] = parent2[mask]
        return child

    @staticmethod
    def mutate(circles: np.ndarray, generation: int, max_generations: int, 
               base_mutation_rate: float = 0.1) -> np.ndarray:
        """Apply mutation with adaptive rate"""
        mutation_rate = base_mutation_rate * (0.5**(generation/max_generations))
        mutation_rate = max(0.02, mutation_rate)
        
        mutated = circles.copy()
        
        for i in range(len(mutated)):
            if np.random.rand() < mutation_rate:
                if np.random.rand() < 0.7:
                    # Position mutation
                    scale = 0.02 * (1 - generation/max_generations) + 0.005
                    mutated[i, 0] += np.random.normal(0, scale)
                    mutated[i, 1] += np.random.normal(0, scale)
                    mutated[i, 0] = np.clip(mutated[i, 0], mutated[i, 2], 1 - mutated[i, 2])
                    mutated[i, 1] = np.clip(mutated[i, 1], mutated[i, 2], 1 - mutated[i, 2])
                else:
                    # Radius mutation
                    scale = 0.01 * (1 - generation/max_generations) + 0.001
                    mutated[i, 2] += np.random.normal(0, scale)
                    mutated[i, 2] = max(0.001, mutated[i, 2])
        
        return mutated

class Repairer:
    """Handles circle configuration repair and overlap resolution"""
    
    @staticmethod
    def repair_circles(circles: np.ndarray) -> np.ndarray:
        """Repair invalid configurations with enhanced overlap resolution"""
        repaired = circles.copy()
        
        # Ensure bounds and positive radii
        for i in range(len(repaired)):
            x, y, r = repaired[i]
            repaired[i, 0] = np.clip(x, r, 1 - r)
            repaired[i, 1] = np.clip(y, r, 1 - r)
            repaired[i, 2] = max(0.001, repaired[i, 2])
        
        # Resolve overlaps with aggressive repulsion
        points = repaired[:, :2]
        tree = cKDTree(points)
        
        for _ in range(40):
            any_changes = False
            for i in range(len(repaired)):
                x1, y1, r1 = repaired[i]
                nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))
                
                for j in nearby_indices:
                    if i != j:
                        x2, y2, r2 = repaired[j]
                        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        min_distance = r1 + r2
                        
                        if distance < min_distance:
                            if distance > 0.001:
                                dx = (x1 - x2) / distance
                                dy = (y1 - y2) / distance
                                
                                move_amount = (min_distance - distance) * 0.6
                                repaired[i, 0] += dx * move_amount
                                repaired[i, 1] += dy * move_amount
                                repaired[j, 0] -= dx * move_amount
                                repaired[j, 1] -= dy * move_amount
                                
                                # Bounds
                                repaired[i, 0] = np.clip(repaired[i, 0], r1, 1 - r1)
                                repaired[i, 1] = np.clip(repaired[i, 1], r1, 1 - r1)
                                repaired[j, 0] = np.clip(repaired[j, 0], r2, 1 - r2)
                                repaired[j, 1] = np.clip(repaired[j, 1], r2, 1 - r2)
                            
                            any_changes = True
            
            if not any_changes:
                break
        
        return repaired

class Refiner:
    """Handles local refinement operations"""
    
    @staticmethod
    def local_refinement(circles: np.ndarray, max_iterations: int = 100) -> np.ndarray:
        """
        Enhanced local refinement with multi-level optimization strategies
        """
        refined = circles.copy()
        n_circles = len(refined)
        
        # Strategy 1: Greedy radius expansion
        improved = True
        iteration = 0
        
        while improved and iteration < 60:
            improved = False
            iteration += 1
            
            for i in range(n_circles):
                original_r = refined[i, 2]
                x, y, _ = refined[i]
                max_possible_r = min(x, 1-x, y, 1-y)
                
                steps = [0.008, 0.005, 0.003, 0.002, 0.001]
                
                for step in steps:
                    test_r = min(original_r + step, max_possible_r * 0.95)
                    
                    if test_r > original_r + 1e-6:
                        temp_circles = refined.copy()
                        temp_circles[i, 2] = test_r
                        
                        if CircleValidator.validate_circles(temp_circles):
                            refined = temp_circles
                            improved = True
                            break
        
        # Strategy 2: Position adjustment
        for _ in range(30):
            improved_local = False
            indices = list(range(n_circles))
            np.random.shuffle(indices)
            
            for i in indices:
                x, y, r = refined[i]
                
                best_x, best_y, best_r = x, y, r
                best_radius = r
                best_improvement = 0
                
                adjustments = [
                    (0, 0, 0),
                    (0.003, 0, 0),
                    (-0.003, 0, 0),
                    (0, 0.003, 0),
                    (0, -0.003, 0),
                    (0.002, 0.002, 0),
                    (-0.002, -0.002, 0),
                    (0.002, -0.002, 0),
                    (-0.002, 0.002, 0),
                    (0.001, 0.001, 0),
                    (-0.001, -0.001, 0),
                    (0.001, -0.001, 0),
                    (-0.001, 0.001, 0)
                ]
                
                for dx, dy, dr in adjustments:
                    test_x = max(0.001, min(0.999, x + dx))
                    test_y = max(0.001, min(0.999, y + dy))
                    test_r = max(0.001, min(r + dr,
                                      min(test_x, test_y, 1-test_x, 1-test_y) * 0.98))
                    
                    temp_circles = refined.copy()
                    temp_circles[i] = [test_x, test_y, test_r]
                    
                    if CircleValidator.validate_circles(temp_circles) and test_r > best_radius:
                        best_x, best_y, best_r = test_x, test_y, test_r
                        best_radius = test_r
                        best_improvement = test_r - r
                
                if best_improvement > 1e-6:
                    refined[i] = [best_x, best_y, best_r]
                    improved_local = True
            
            if not improved_local:
                break
        
        # Final repair
        final_repaired = Repairer.repair_circles(refined)
        
        if CircleValidator.validate_circles(final_repaired):
            return final_repaired
        else:
            result = circles.copy()
            for i in range(n_circles):
                result[i, 0] = np.clip(result[i, 0], result[i, 2], 1 - result[i, 2])
                result[i, 1] = np.clip(result[i, 1], result[i, 2], 1 - result[i, 2])
                result[i, 2] = max(0.001, result[i, 2])
            return result

class EvolutionController:
    """Main controller for the evolutionary algorithm"""
    
    def __init__(self, config: CircleConfig):
        self.config = config
        self.validator = CircleValidator()
        self.initializer = Initializer()
        self.selector = Selector()
        self.operator = Operator()
        self.repairer = Repairer()
        self.refiner = Refiner()
        self.stats = EvolutionStats()
    
    def run_evolution(self) -> np.ndarray:
        """Run the complete evolutionary process"""
        # Create initial population
        population = self.initializer.create_initial_population(
            self.config.population_size, 
            self.config.num_circles
        )
        
        # Evolution loop
        best_fitness = -np.inf
        best_individual = None
        
        for generation in range(self.config.generations):
            start_time = time.time()
            
            # Calculate fitness for all individuals
            fitnesses = []
            valid_individuals = []
            
            for circles in population:
                if self.validator.validate_circles(circles):
                    fitness = self.validator.calculate_sum_radii(circles)
                    fitnesses.append(fitness)
                    valid_individuals.append(circles)
                else:
                    # Repair invalid individuals
                    repaired = self.repairer.repair_circles(circles)
                    if self.validator.validate_circles(repaired):
                        fitness = self.validator.calculate_sum_radii(repaired)
                        fitnesses.append(fitness)
                        valid_individuals.append(repaired)
                    else:
                        fitnesses.append(-np.inf)
                        valid_individuals.append(circles)
            
            # Track best individual
            if valid_individuals:
                max_idx = np.argmax(fitnesses)
                if fitnesses[max_idx] > best_fitness:
                    best_fitness = fitnesses[max_idx]
                    best_individual = valid_individuals[max_idx].copy()
            
            # Elitism: keep top individuals
            elite_indices = np.argsort(fitnesses)[-self.config.elite_size:]
            elites = [valid_individuals[i] for i in elite_indices if fitnesses[i] > -np.inf]
            
            # Generate new population
            new_population = elites[:]
            
            # Fill remaining slots with offspring
            while len(new_population) < self.config.population_size:
                # Tournament selection
                parent1 = self.selector.tournament_selection(valid_individuals, fitnesses)
                parent2 = self.selector.tournament_selection(valid_individuals, fitnesses)
                
                # Crossover
                child = self.operator.crossover(parent1, parent2)
                
                # Mutation with adaptive rate
                mutation_rate = self.config.mutation_rate
                if generation > self.config.adaptive_mutation_threshold:
                    mutation_rate = 0.05
                elif generation > self.config.adaptive_mutation_threshold // 2:
                    mutation_rate = 0.07
                
                child = self.operator.mutate(child, generation, self.config.generations, mutation_rate)
                
                # Repair
                child = self.repairer.repair_circles(child)
                
                new_population.append(child)
            
            population = new_population[:self.config.population_size]
            
            # Record stats
            end_time = time.time()
            self.stats.generation_times.append(end_time - start_time)
            if valid_individuals:
                self.stats.fitness_history.append(best_fitness)
        
        # Return the best solution found with final refinement
        if best_individual is not None:
            refined_solution = self.refiner.local_refinement(best_individual)
            if self.validator.validate_circles(refined_solution):
                return refined_solution
            else:
                return best_individual
        else:
            # If no valid solution found, return the best from final population
            fitnesses = [self.validator.calculate_sum_radii(circles) 
                        for circles in population if self.validator.validate_circles(circles)]
            if fitnesses:
                best_idx = np.argmax(fitnesses)
                refined_solution = self.refiner.local_refinement(population[best_idx])
                if self.validator.validate_circles(refined_solution):
                    return refined_solution
                else:
                    return population[best_idx]
            else:
                # Fallback: return a valid random solution
                circles = np.zeros((self.config.num_circles, 3))
                for i in range(self.config.num_circles):
                    circles[i] = [0.5, 0.5, 0.01]
                return circles

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    config = CircleConfig()
    controller = EvolutionController(config)
    return controller.run_evolution()

# EVOLVE-BLOCK-END