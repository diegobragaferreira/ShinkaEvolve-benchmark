# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import random
from typing import Tuple, List, Optional
import time

# Global constants
POPULATION_SIZE = 100
GENERATIONS = 300
MUTATION_RATE_INITIAL = 0.15
MUTATION_RATE_FINAL = 0.01
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 5
ELITE_COUNT = 10
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

class CircleValidator:
    """Validates circle configurations for containment and overlap constraints."""
    
    @staticmethod
    def validate_containment(circles: np.ndarray) -> bool:
        """Vectorized containment validation."""
        if len(circles) == 0:
            return True
            
        radii = circles[:, 2]
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        
        # Check if any radius violates containment
        containment_check = (
            (radii <= x_coords) &
            (radii <= y_coords) &
            (radii <= 1 - x_coords) &
            (radii <= 1 - y_coords)
        )
        
        return np.all(containment_check)
    
    @staticmethod
    def validate_overlaps(circles: np.ndarray) -> bool:
        """Efficient overlap validation using spatial indexing."""
        if len(circles) <= 1:
            return True
            
        try:
            # Use KDTree for efficient neighbor search
            tree = KDTree(circles[:, :2])
            max_radius = np.max(circles[:, 2])
            
            # For each circle, find neighbors within 2*(max_radius) distance
            for i in range(len(circles)):
                # Find neighbors that could potentially overlap
                potential_neighbors = tree.query_ball_point(circles[i, :2], 2 * max_radius)
                # Skip self
                potential_neighbors = [idx for idx in potential_neighbors if idx != i]
                
                for j in potential_neighbors:
                    # Calculate actual distance
                    dist = np.sqrt((circles[i, 0] - circles[j, 0])**2 + (circles[i, 1] - circles[j, 1])**2)
                    min_dist = circles[i, 2] + circles[j, 2]
                    if dist < min_dist:
                        return False
                        
            return True
        except Exception:
            # Fallback to brute force if KDTree fails
            distances = cdist(circles[:, :2], circles[:, :2])
            mask = np.triu(np.ones((len(circles), len(circles)), dtype=bool), k=1)
            min_distances = (circles[:, 2][:, np.newaxis] + circles[:, 2][np.newaxis, :]) * mask
            overlaps = distances < min_distances
            return not np.any(overlaps)
    
    @staticmethod
    def is_valid(circles: np.ndarray) -> bool:
        """Complete validation of circle configuration."""
        return CircleValidator.validate_containment(circles) and CircleValidator.validate_overlaps(circles)

class CircleInitializer:
    """Creates initial population with various strategies."""
    
    @staticmethod
    def create_grid_initialization(n_circles: int) -> np.ndarray:
        """Create grid-based initialization."""
        circles = np.zeros((n_circles, 3))
        
        grid_size = int(np.ceil(np.sqrt(n_circles)))
        spacing = 1.0 / (grid_size + 1)
        
        idx = 0
        for row in range(grid_size):
            for col in range(grid_size):
                if idx >= n_circles:
                    break
                x = (col + 1) * spacing
                y = (row + 1) * spacing
                r = spacing / 3
                # Add small randomness
                x += np.random.uniform(-spacing/8, spacing/8)
                y += np.random.uniform(-spacing/8, spacing/8)
                r = max(0.01, min(r, x, y, 1-x, 1-y))
                circles[idx] = [x, y, r]
                idx += 1
                
        return circles
    
    @staticmethod
    def create_corner_initialization(n_circles: int) -> np.ndarray:
        """Create initialization with corner positioning."""
        circles = np.zeros((n_circles, 3))
        
        # Place key circles at strategic positions
        key_positions = [
            (0.1, 0.1, 0.05),      # bottom-left
            (0.9, 0.1, 0.05),      # bottom-right
            (0.1, 0.9, 0.05),      # top-left
            (0.9, 0.9, 0.05),      # top-right
            (0.5, 0.5, 0.1),       # center
        ]
        
        # Fill remaining positions with grid pattern
        grid_size = int(np.ceil(np.sqrt(n_circles - len(key_positions))))
        spacing = 1.0 / (grid_size + 1)
        
        idx = 0
        for pos in key_positions:
            if idx >= n_circles:
                break
            circles[idx] = list(pos)
            idx += 1
            
        # Fill remaining positions with grid pattern
        remaining_count = n_circles - idx
        for i in range(remaining_count):
            row = i // grid_size
            col = i % grid_size
            x = (col + 1) * spacing
            y = (row + 1) * spacing
            r = spacing / 3
            # Add small randomness
            x += np.random.uniform(-spacing/8, spacing/8)
            y += np.random.uniform(-spacing/8, spacing/8)
            r = max(0.01, min(r, x, y, 1-x, 1-y))
            circles[idx] = [x, y, r]
            idx += 1
            
        return circles
    
    @staticmethod
    def create_random_initialization(n_circles: int) -> np.ndarray:
        """Create random initialization with overlap avoidance."""
        circles = np.zeros((n_circles, 3))
        
        for i in range(n_circles):
            attempts = 0
            while attempts < 100:
                # Random placement in unit square
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                
                # Radius based on distance to closest boundary
                min_dist = min(x, y, 1-x, 1-y)
                r = np.random.uniform(0.01, min_dist/2)
                
                # Check if it overlaps with existing circles
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
                # Fallback to simple grid if failed
                grid_size = int(np.ceil(np.sqrt(n_circles)))
                spacing = 1.0 / (grid_size + 1)
                row = i // grid_size
                col = i % grid_size
                x = (col + 1) * spacing
                y = (row + 1) * spacing
                r = spacing / 4
                circles[i] = [x, y, r]
                
        return circles
    
    @classmethod
    def create_population(cls, pop_size: int, n_circles: int) -> List[np.ndarray]:
        """Create diverse initial population."""
        population = []
        
        # Strategy mapping
        strategies = [
            cls.create_grid_initialization,
            cls.create_corner_initialization,
            cls.create_random_initialization
        ]
        
        for _ in range(pop_size):
            # Select strategy based on index
            strategy = strategies[len(population) % len(strategies)]
            circles = strategy(n_circles)
            
            # Ensure valid configuration
            if not CircleValidator.is_valid(circles):
                # Fallback to simple grid
                circles = cls.create_grid_initialization(n_circles)
                
            population.append(circles)
            
        return population

class LocalOptimizer:
    """Performs local optimization to improve circle configurations."""
    
    @staticmethod
    def refine_solution(circles: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Advanced local optimization with adaptive strategies."""
        circles = circles.copy()
        n = len(circles)
        
        # Classify solution based on overlap severity
        def classify_solution(circles: np.ndarray) -> str:
            if n <= 1:
                return "no-overlap"
                
            distances = cdist(circles[:, :2], circles[:, :2])
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            min_distances = (circles[:, 2][:, np.newaxis] + circles[:, 2][np.newaxis, :]) * mask
            overlaps = distances < min_distances
            overlap_count = np.sum(overlaps)
            
            if overlap_count == 0:
                return "no-overlap"
            elif overlap_count <= 3:
                return "low-overlap"
            elif overlap_count <= 10:
                return "medium-overlap"
            else:
                return "high-overlap"
        
        # Classify solution type
        solution_type = classify_solution(circles)
        
        # Determine refinement strategy
        if solution_type == "no-overlap":
            max_refinement_iter = max_iter // 2
        elif solution_type == "low-overlap":
            max_refinement_iter = max_iter // 3
        elif solution_type == "medium-overlap":
            max_refinement_iter = max_iter // 2
        else:  # high-overlap
            max_refinement_iter = max_iter
            
        # Track improvements for early termination
        for iteration in range(max_refinement_iter):
            improved = False
            
            # Strategy 1: Try to expand radii
            if solution_type == "no-overlap" or solution_type == "low-overlap":
                increments = [0.01, 0.005]
            else:
                increments = [0.002, 0.001]
            
            # Process in order of available space (least constrained first)
            space_left = np.min([
                circles[:, 0], 
                circles[:, 1], 
                1 - circles[:, 0], 
                1 - circles[:, 1]
            ], axis=0)
            sorted_indices = np.argsort(space_left)
            
            for i in sorted_indices:
                original_radius = circles[i][2]
                original_x, original_y = circles[i][0], circles[i][1]
                
                # Calculate maximum possible radius
                max_radius = min(
                    original_x,
                    original_y,
                    1 - original_x,
                    1 - original_y
                )
                
                # Try different increments
                for increment in increments:
                    if max_radius > original_radius:
                        new_radius = min(original_radius + increment, max_radius)
                        if new_radius > original_radius:
                            circles[i][2] = new_radius
                            
                            if CircleValidator.is_valid(circles):
                                improved = True
                                break
                            else:
                                circles[i][2] = original_radius
                else:
                    continue
                break  # Break if we made an improvement
            
            # Strategy 2: Position adjustments for overlap resolution
            if improved or solution_type != "no-overlap":
                adjustments = [
                    (0.001, 0), (-0.001, 0), (0, 0.001), (0, -0.001),
                    (0.0005, 0.0005), (-0.0005, -0.0005),
                    (0.0005, -0.0005), (-0.0005, 0.0005)
                ]
                
                # Try adjustments in a specific order
                for i in range(n):
                    original_x, original_y = circles[i][0], circles[i][1]
                    
                    for dx, dy in adjustments:
                        new_x = np.clip(original_x + dx, 0, 1)
                        new_y = np.clip(original_y + dy, 0, 1)
                        
                        if new_x != original_x or new_y != original_y:
                            circles[i][0] = new_x
                            circles[i][1] = new_y
                            
                            if CircleValidator.is_valid(circles):
                                improved = True
                                break
                            else:
                                circles[i][0] = original_x
                                circles[i][1] = original_y
                    else:
                        continue
                    break  # Break if we made an improvement
                    
            if not improved:
                break
                
        return circles

class EvolutionEngine:
    """Handles the evolutionary algorithm operations."""
    
    @staticmethod
    def tournament_selection(population: List[np.ndarray], 
                           fitnesses: List[float],
                           tournament_size: int = TOURNAMENT_SIZE) -> np.ndarray:
        """Tournament selection with adaptive size."""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_index].copy()
    
    @staticmethod
    def uniform_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform uniform crossover."""
        if np.random.random() > CROSSOVER_RATE:
            return parent1.copy(), parent2.copy()

        n = len(parent1)
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)
        
        # Uniform crossover
        for i in range(n):
            if np.random.random() < 0.5:
                child1[i] = parent2[i].copy()
                child2[i] = parent1[i].copy()
            else:
                child1[i] = parent1[i].copy()
                child2[i] = parent2[i].copy()

        return child1, child2
    
    @staticmethod
    def mutate(circles: np.ndarray, generation: int, 
               mutation_rate_initial: float = MUTATION_RATE_INITIAL,
               mutation_rate_final: float = MUTATION_RATE_FINAL,
               generations: int = GENERATIONS) -> np.ndarray:
        """Apply mutation with adaptive rate."""
        mutated = circles.copy()
        n = len(mutated)
        
        # Adaptive mutation rate based on generation
        mutation_rate = mutation_rate_initial - (mutation_rate_initial - mutation_rate_final) * (generation / generations)
        
        for i in range(n):
            if np.random.random() < mutation_rate:
                # Mutate either position or radius
                if np.random.random() < 0.7:  # 70% chance to mutate position
                    # Mutate position
                    mutated[i][0] = np.clip(mutated[i][0] + np.random.normal(0, 0.03), 0, 1)
                    mutated[i][1] = np.clip(mutated[i][1] + np.random.normal(0, 0.03), 0, 1)
                else:
                    # Mutate radius
                    mutated[i][2] = np.clip(mutated[i][2] + np.random.normal(0, 0.01), 0.01, 0.5)

        return mutated

class CirclePackingOptimizer:
    """Core optimizer class for circle packing problem."""
    
    def __init__(self, n_circles: int = 26):
        self.n_circles = n_circles
        self.best_solution = None
        self.best_fitness = -float('inf')
        self.start_time = time.time()
        self.max_time = 60  # seconds
    
    def calculate_sum_radii(self, circles: np.ndarray) -> float:
        """Calculate the sum of all radii."""
        return np.sum(circles[:, 2])
    
    def evaluate_fitness(self, circles: np.ndarray) -> float:
        """Evaluate fitness with penalty for invalid configurations."""
        if CircleValidator.is_valid(circles):
            return self.calculate_sum_radii(circles)
        else:
            # Invalid configurations get very low fitness
            return 0.0
    
    def should_terminate(self) -> bool:
        """Check if we should terminate due to time limit."""
        return time.time() - self.start_time > self.max_time * 0.95
    
    def run_evolution(self) -> np.ndarray:
        """Run the complete evolutionary algorithm."""
        # Initialize population
        population = CircleInitializer.create_population(POPULATION_SIZE, self.n_circles)
        
        if not population:
            # Fallback to simple initialization
            return CircleInitializer.create_grid_initialization(self.n_circles)
        
        best_solution = None
        best_fitness = -1
        no_improvement_count = 0
        max_no_improvement = 20

        for generation in range(GENERATIONS):
            # Check early termination
            if self.should_terminate():
                break
            
            # Evaluate fitness for all individuals
            fitnesses = [self.evaluate_fitness(circles) for circles in population]
            
            # Track best solution
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > best_fitness:
                best_fitness = fitnesses[max_fitness_idx]
                best_solution = population[max_fitness_idx].copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            # Early termination if no improvement
            if no_improvement_count > max_no_improvement:
                break
            
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitnesses)[-ELITE_COUNT:]
            elites = [population[i].copy() for i in elite_indices]
            
            # Create new population
            new_population = elites.copy()
            
            # Generate offspring
            while len(new_population) < POPULATION_SIZE:
                # Selection
                parent1 = EvolutionEngine.tournament_selection(population, fitnesses)
                parent2 = EvolutionEngine.tournament_selection(population, fitnesses)

                # Crossover
                child1, child2 = EvolutionEngine.uniform_crossover(parent1, parent2)

                # Mutation
                child1 = EvolutionEngine.mutate(child1, generation)
                child2 = EvolutionEngine.mutate(child2, generation)

                # Local optimization
                child1 = LocalOptimizer.refine_solution(child1)
                child2 = LocalOptimizer.refine_solution(child2)

                # Add to new population
                new_population.extend([child1, child2])

            # Trim population to exact size
            population = new_population[:POPULATION_SIZE]

        # Return the best solution found
        if best_solution is not None:
            return best_solution
        else:
            # Fallback to a simple configuration
            return CircleInitializer.create_grid_initialization(self.n_circles)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    optimizer = CirclePackingOptimizer(26)
    return optimizer.run_evolution()

# EVOLVE-BLOCK-END