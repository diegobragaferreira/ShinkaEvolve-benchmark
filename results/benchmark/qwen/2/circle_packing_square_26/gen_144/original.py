# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import KDTree
import random
import time
from typing import Tuple, List, Optional
import math

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

class CirclePackingOptimizer:
    def __init__(self, n_circles: int = 26, pop_size: int = 50, max_generations: int = 1000):
        self.n_circles = n_circles
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.best_fitness_history = []
        
    def validate_circles(self, circles: np.ndarray) -> bool:
        """Validate that circles are within bounds and non-overlapping."""
        if len(circles) != self.n_circles:
            return False
            
        # Check containment constraints
        for i in range(self.n_circles):
            x, y, r = circles[i]
            if r <= 0 or x < r or x > 1 - r or y < r or y > 1 - r:
                return False
        
        # Check overlap constraints using KDTree for efficiency
        points = circles[:, :2]
        tree = KDTree(points)
        
        for i in range(self.n_circles):
            x, y, r = circles[i]
            # Find nearby circles (within 2*r distance)
            nearby = tree.query_ball_point([x, y], 2 * r)
            for j in nearby:
                if i != j:
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if distance < r + r2:
                        return False
        
        return True

    def calculate_fitness(self, circles: np.ndarray) -> float:
        """Calculate total radius sum as fitness."""
        return np.sum(circles[:, 2])

    def initialize_population(self) -> List[np.ndarray]:
        """Generate initial population of circle arrangements with enhanced strategy."""
        population = []
        
        # Use enhanced initialization strategy inspired by multiple approaches
        for _ in range(self.pop_size):
            circles = self._create_enhanced_circles()
            population.append(circles)
            
        return population

    def _create_enhanced_circles(self) -> np.ndarray:
        """Create a valid configuration of circles with enhanced initialization."""
        circles = np.zeros((self.n_circles, 3))
        
        # Create grid-based initialization with better distribution
        grid_size = max(1, int(np.ceil(np.sqrt(self.n_circles))))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= self.n_circles:
                    break
                    
                x = (i + 1) * spacing_x
                y = (j + 1) * spacing_y
                # Initial radius with better distribution
                r = min(spacing_x, spacing_y) * np.random.uniform(0.25, 0.45)
                
                # Add randomness with controlled variation
                r = max(0.005, r * np.random.uniform(0.8, 1.2))
                x = max(r, min(1-r, x + np.random.uniform(-spacing_x*0.1, spacing_x*0.1)))
                y = max(r, min(1-r, y + np.random.uniform(-spacing_y*0.1, spacing_y*0.1)))
                
                circles[idx] = [x, y, r]
                idx += 1
                
        # Fill remaining circles with intelligent random placement
        for i in range(idx, self.n_circles):
            max_attempts = 1000
            placed = False
            attempts = 0
            
            while not placed and attempts < max_attempts:
                # Use better probability distribution for positions (triangular)
                x = np.random.triangular(0.05, 0.5, 0.95)
                y = np.random.triangular(0.05, 0.5, 0.95)
                # Use log-uniform for radius to get better distribution
                r = np.random.loguniform(0.005, 0.15)
                
                # Check if valid placement
                valid_placement = True
                if r <= x <= 1 - r and r <= y <= 1 - r:
                    # Check overlap with existing circles using optimized approach
                    for j in range(i):
                        existing_x, existing_y, existing_r = circles[j]
                        distance = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                        if distance < r + existing_r:
                            valid_placement = False
                            break
                else:
                    valid_placement = False
                
                if valid_placement:
                    circles[i] = [x, y, r]
                    placed = True
                attempts += 1
            
            # If failed to place, use fallback with better positioning
            if not placed:
                x = 0.5 + np.random.normal(0, 0.1)
                y = 0.5 + np.random.normal(0, 0.1)
                r = 0.01
                x = max(r, min(1-r, x))
                y = max(r, min(1-r, y))
                circles[i] = [x, y, r]
        
        return circles

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Perform crossover between two parents with improved strategy."""
        child = np.copy(parent1)
        
        # Use more sophisticated crossover with selective swapping
        # Mix positions and radii separately to preserve structure
        for i in range(self.n_circles):
            if np.random.random() < 0.5:
                # Take from parent2 with probability 0.5
                child[i] = parent2[i].copy()
        
        # Ensure child remains valid by repositioning circles if necessary
        for i in range(self.n_circles):
            x, y, r = child[i]
            # Adjust position if out of bounds
            if x < r:
                x = r
            elif x > 1 - r:
                x = 1 - r
            if y < r:
                y = r
            elif y > 1 - r:
                y = 1 - r
            child[i] = [x, y, r]
        
        # Recheck for overlaps and fix them with improved repair
        return self._repair_overlaps(child)

    def _repair_overlaps(self, circles: np.ndarray) -> np.ndarray:
        """Repair overlapping circles by adjusting positions and radii."""
        # Try several iterations to resolve overlaps with progressive approach
        for iteration in range(10):
            if self.validate_circles(circles):
                return circles
                
            # Progressive overlap resolution: start aggressive then become conservative
            reduction_factor = 0.95 - (iteration * 0.02)  # Gradually reduce aggressiveness
            reduction_factor = max(0.8, reduction_factor)  # Minimum factor
            
            # More aggressive repair: reduce radii and slightly adjust positions
            for i in range(self.n_circles):
                x, y, r = circles[i]
                # Reduce radius to resolve overlap
                circles[i] = [x, y, max(0.001, r * reduction_factor)]
                
        # Final adjustment if still invalid
        for i in range(self.n_circles):
            x, y, r = circles[i]
            # Ensure boundaries
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            circles[i] = [x, y, r]
            
        return circles

    def mutate(self, circles: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
        """Apply mutations to circles with enhanced strategies."""
        mutated = np.copy(circles)
        
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                # Mutate either position or radius with higher probability for position
                if np.random.random() < 0.7:  # 70% chance to mutate position
                    # Mutate position
                    x, y, r = mutated[i]
                    # Perturbations tuned for better exploration
                    x += np.random.normal(0, 0.015)
                    y += np.random.normal(0, 0.015)
                    # Keep within bounds
                    x = max(r, min(1 - r, x))
                    y = max(r, min(1 - r, y))
                    mutated[i] = [x, y, r]
                else:
                    # Mutate radius with log-normal to ensure positivity and control
                    x, y, r = mutated[i]
                    # Log-normal mutation - keeps radius positive and allows larger changes
                    old_r = r
                    r = np.exp(np.log(old_r) + np.random.normal(0, 0.15))
                    # Keep positive with minimum
                    r = max(0.001, r)
                    mutated[i] = [x, y, r]
        
        # Ensure validity after mutation
        return self._repair_overlaps(mutated)

    def evaluate_population(self, population: List[np.ndarray]) -> List[float]:
        """Evaluate fitness of entire population."""
        return [self.calculate_fitness(circles) for circles in population]

    def select_parents(self, population: List[np.ndarray], fitnesses: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Select two parents using tournament selection with variable size."""
        # Tournament selection with variable size based on population
        tournament_size = min(max(2, len(population) // 5), 6)  # Adaptive tournament size
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitnesses[i] for i in tournament_indices]
        winner1_idx = tournament_indices[np.argmax(tournament_fitness)]
        
        # Tournament selection for second parent (different from first)
        tournament_indices2 = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness2 = [fitnesses[i] for i in tournament_indices2]
        winner2_idx = tournament_indices2[np.argmax(tournament_fitness2)]
        
        return population[winner1_idx], population[winner2_idx]

    def adaptive_mutation_rate(self, generation: int, diversity: float) -> float:
        """Adaptively adjust mutation rate based on generation and population diversity."""
        base_rate = 0.15  # Increased base rate for more exploration
        # Decrease mutation rate over time with faster decay
        time_factor = 1.0 - (generation / self.max_generations) ** 1.2
        # Increase mutation rate if diversity is low (to escape local optima)
        diversity_factor = max(0.8, 1.0 - diversity) if diversity < 0.3 else 1.0
        
        return base_rate * time_factor * diversity_factor

    def get_population_diversity(self, population: List[np.ndarray]) -> float:
        """Calculate population diversity based on spread of radii."""
        if len(population) < 2:
            return 0.0
            
        all_radii = np.concatenate([circles[:, 2] for circles in population])
        mean_radius = np.mean(all_radii)
        if mean_radius == 0:
            return 0.0
        return np.std(all_radii) / (mean_radius + 1e-8)  # Normalize by mean

    def evolve(self) -> np.ndarray:
        """Main evolutionary algorithm for circle packing."""
        # Initialize population
        population = self.initialize_population()
        
        # Evolution loop
        for generation in range(self.max_generations):
            # Evaluate fitness
            fitnesses = self.evaluate_population(population)
            
            # Track best fitness
            best_fitness = max(fitnesses)
            self.best_fitness_history.append(best_fitness)
            
            # Print progress every 100 generations
            if generation % 100 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")
            
            # Create new population
            new_population = []
            
            # Elitism: keep the best individual
            best_idx = np.argmax(fitnesses)
            new_population.append(population[best_idx])
            
            # Calculate population diversity for adaptive parameters
            diversity = self.get_population_diversity(population)
            
            # Generate offspring
            while len(new_population) < self.pop_size:
                # Select parents
                parent1, parent2 = self.select_parents(population, fitnesses)
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutation with adaptive rate
                mutation_rate = self.adaptive_mutation_rate(generation, diversity)
                child = self.mutate(child, mutation_rate)
                
                new_population.append(child)
            
            population = new_population[:self.pop_size]  # Ensure exact population size
            
            # Early stopping if fitness improves very little
            if len(self.best_fitness_history) > 10:
                recent_improvement = self.best_fitness_history[-1] - self.best_fitness_history[-10]
                if recent_improvement < 1e-6:
                    break
        
        # Return best solution
        final_fitnesses = self.evaluate_population(population)
        best_idx = np.argmax(final_fitnesses)
        return population[best_idx]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    start_time = time.time()
    
    try:
        # Create optimizer instance with optimized parameters
        optimizer = CirclePackingOptimizer(n_circles=26, pop_size=50, max_generations=1000)
        
        # Run evolution
        circles = optimizer.evolve()
        
        # Validate result
        if not optimizer.validate_circles(circles):
            # If validation fails, try a simpler approach
            print("Validation failed on evolved solution, using fallback...")
            circles = np.zeros((26, 3))
            # Use a simple heuristic: distribute evenly with decreasing radii
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
