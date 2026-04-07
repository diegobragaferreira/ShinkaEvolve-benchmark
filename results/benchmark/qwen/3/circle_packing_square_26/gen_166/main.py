# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List, Dict, Any, Optional
import math
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class CircleConfig:
    """Configuration parameters for circle packing evolution."""
    population_size: int = 150
    generations: int = 200
    tournament_size: int = 5
    mutation_rate_start: float = 0.2
    mutation_rate_end: float = 0.005
    crossover_prob: float = 0.9
    validity_threshold: float = 1e-6
    initial_grid_size: int = 20
    benchmark_target: float = 2.6358627564136983
    adaptive_grid_start_gen: int = 50
    adaptive_grid_fine_res: int = 30

class CircleConfiguration:
    """Represents a configuration of circles in the unit square."""
    
    def __init__(self, circles: np.ndarray):
        self.circles = circles.copy()  # Shape: (n, 3) where each row is [x, y, r]
        self._grid_cache = None
        self._grid_valid = False
    
    @property
    def positions(self) -> np.ndarray:
        return self.circles[:, :2]
    
    @property
    def radii(self) -> np.ndarray:
        return self.circles[:, 2]
    
    @property
    def total_radius(self) -> float:
        return np.sum(self.radii)
    
    def clone(self) -> 'CircleConfiguration':
        return CircleConfiguration(self.circles)
    
    def get_grid(self, grid_size: int = None, adaptive: bool = False, generation: int = 0) -> Dict[Tuple[int, int], List[int]]:
        """Get spatial grid for efficient neighbor lookups."""
        if grid_size is None:
            grid_size = 20
            
        # Adjust grid size for adaptive resolution
        if adaptive and generation >= 50:
            grid_size = 30
            
        if self._grid_cache is not None and self._grid_valid:
            return self._grid_cache
            
        grid = {}
        cell_size = 1.0 / grid_size
        
        for i, (x, y, r) in enumerate(self.circles):
            # Determine which grid cells this circle might occupy
            min_x_cell = max(0, int((x - r) / cell_size))
            max_x_cell = min(grid_size - 1, int((x + r) / cell_size))
            min_y_cell = max(0, int((y - r) / cell_size))
            max_y_cell = min(grid_size - 1, int((y + r) / cell_size))
            
            for gx in range(min_x_cell, max_x_cell + 1):
                for gy in range(min_y_cell, max_y_cell + 1):
                    if (gx, gy) not in grid:
                        grid[(gx, gy)] = []
                    grid[(gx, gy)].append(i)
        
        self._grid_cache = grid
        self._grid_valid = True
        return grid
    
    def invalidate_grid(self):
        """Invalidate cached grid."""
        self._grid_cache = None
        self._grid_valid = False

class FitnessEvaluator:
    """Handles fitness evaluation and constraint checking."""
    
    @staticmethod
    def check_containment(circles: np.ndarray) -> bool:
        """Check if all circles are fully contained in the unit square."""
        for x, y, r in circles:
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        return True
    
    @staticmethod
    def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    @staticmethod
    def check_overlap_efficient(circles: np.ndarray, grid: Dict[Tuple[int, int], List[int]] = None,
                               validity_threshold: float = 1e-6) -> bool:
        """Check if any circles overlap using spatial grid indexing."""
        if len(circles) <= 1:
            return False
        
        if grid is None:
            # Create temporary grid for checking
            temp_grid = {}
            cell_size = 1.0 / 20
            for i, (x, y, r) in enumerate(circles):
                min_x_cell = max(0, int((x - r) / cell_size))
                max_x_cell = min(19, int((x + r) / cell_size))
                min_y_cell = max(0, int((y - r) / cell_size))
                max_y_cell = min(19, int((y + r) / cell_size))
                
                for gx in range(min_x_cell, max_x_cell + 1):
                    for gy in range(min_y_cell, max_y_cell + 1):
                        if (gx, gy) not in temp_grid:
                            temp_grid[(gx, gy)] = []
                        temp_grid[(gx, gy)].append(i)
            grid = temp_grid
        
        # For each cell, check pairs of circles
        for (gx, gy), indices in grid.items():
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    idx1, idx2 = indices[i], indices[j]
                    x1, y1, r1 = circles[idx1]
                    x2, y2, r2 = circles[idx2]
                    
                    distance = FitnessEvaluator.calculate_distance((x1, y1), (x2, y2))
                    if distance < (r1 + r2 - validity_threshold):
                        return True
        
        return False
    
    @classmethod
    def compute_penalty(cls, circles: np.ndarray, generation: int = 0, 
                       total_generations: int = 100, penalty_scale: float = 1.0) -> float:
        """Compute penalty based on constraint violations with progressive scaling."""
        penalty = 0.0
        
        # Check containment violations with scaled penalties
        for x, y, r in circles:
            # Boundary violations  
            if x - r < 0:
                penalty += (abs(x - r) ** 2) * 10000 * penalty_scale
            elif x + r > 1:
                penalty += (abs(x + r - 1) ** 2) * 10000 * penalty_scale
            if y - r < 0:
                penalty += (abs(y - r) ** 2) * 10000 * penalty_scale
            elif y + r > 1:
                penalty += (abs(y + r - 1) ** 2) * 10000 * penalty_scale
        
        # Check overlap violations with scaled penalties
        if cls.check_overlap_efficient(circles):
            penalty += 10000000.0 * penalty_scale
        
        return penalty
    
    @classmethod
    def evaluate_fitness(cls, circles: np.ndarray, generation: int = 0, 
                        total_generations: int = 100, use_penalty: bool = True) -> float:
        """Evaluate fitness of a circle configuration."""
        # If invalid, heavily penalize
        if not cls.check_containment(circles) or cls.check_overlap_efficient(circles):
            if use_penalty:
                penalty_scale = 1.0 + (generation / total_generations) * 5.0
                penalty = cls.compute_penalty(circles, generation, total_generations, penalty_scale)
                return -penalty
            else:
                return -float('inf')
        
        # Otherwise, return total radius
        total_radius = np.sum(circles[:, 2])
        return total_radius

class EvolutionEngine:
    """Main evolutionary engine for circle packing optimization."""
    
    def __init__(self, config: CircleConfig):
        self.config = config
        self.fitness_evaluator = FitnessEvaluator()
        random.seed(42)
        np.random.seed(42)
        
    def poisson_disk_sampling(self, n_points: int, min_distance: float = 0.1) -> List[Tuple[float, float]]:
        """Generate points using Poisson disk sampling for better uniformity."""
        points = []
        active_list = []
        
        # Start with a random point
        points.append((random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)))
        active_list.append(0)
        
        while len(points) < n_points:
            if not active_list:
                break
                
            # Pick a random active point
            idx = random.choice(active_list)
            x, y = points[idx]
            
            # Try to generate a new point
            found = False
            for _ in range(30):  # Limit attempts
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(min_distance, 2 * min_distance)
                
                new_x = x + radius * math.cos(angle)
                new_y = y + radius * math.sin(angle)
                
                # Check bounds
                if new_x < 0.05 or new_x > 0.95 or new_y < 0.05 or new_y > 0.95:
                    continue
                    
                # Check distance to existing points
                too_close = False
                for px, py in points:
                    dist = math.sqrt((new_x - px)**2 + (new_y - py)**2)
                    if dist < min_distance:
                        too_close = True
                        break
                
                if not too_close:
                    points.append((new_x, new_y))
                    active_list.append(len(points) - 1)
                    found = True
                    break
            
            if not found:
                active_list.remove(idx)
        
        # If we didn't get enough points, fill with random ones
        while len(points) < n_points:
            points.append((random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)))
        
        return points[:n_points]
    
    def initialize_population(self, n: int, population_size: int) -> List[CircleConfiguration]:
        """Initialize population with improved Voronoi-based distribution using Poisson disk sampling."""
        population = []
        
        # Generate points using Poisson disk sampling for better distribution
        sample_points = self.poisson_disk_sampling(n, 0.15)
        
        # Create multiple populations with variation
        for _ in range(population_size):
            circles = np.zeros((n, 3))
            
            # Distribute circles using the sample points
            for i in range(min(n, len(sample_points))):
                x_base, y_base = sample_points[i]
                
                # Add jitter for diversity
                x = max(0.01, min(0.99, x_base + random.uniform(-0.03, 0.03)))
                y = max(0.01, min(0.99, y_base + random.uniform(-0.03, 0.03)))
                
                # Initial radius - start with moderately large values
                circles[i] = [x, y, 0.06]
            
            # Fill remaining circles
            for i in range(len(sample_points), n):
                # Place remaining circles more randomly but still with some structure
                if random.random() < 0.4:
                    # Near an existing circle
                    idx = random.randint(0, min(i-1, len(sample_points)-1))
                    x_base, y_base = sample_points[idx]
                    x = max(0.01, min(0.99, x_base + random.uniform(-0.08, 0.08)))
                    y = max(0.01, min(0.99, y_base + random.uniform(-0.08, 0.08)))
                else:
                    # Completely random
                    x = random.uniform(0.05, 0.95)
                    y = random.uniform(0.05, 0.95)
                
                circles[i] = [x, y, 0.025]
            
            # Ensure circles don't overlap initially
            circles = self.resolve_initial_overlaps(circles)
            population.append(CircleConfiguration(circles))
        
        return population
    
    def resolve_initial_overlaps(self, circles: np.ndarray) -> np.ndarray:
        """Resolve overlaps in initial configuration using force-based approach."""
        resolved = circles.copy()
        
        # Iteratively resolve overlaps
        for _ in range(10):
            changed = False
            grid = self.get_grid_cells(resolved, self.config.initial_grid_size)
            
            for i in range(len(resolved)):
                for j in range(i+1, len(resolved)):
                    xi, yi, ri = resolved[i]
                    xj, yj, rj = resolved[j]
                    dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                    
                    if dist < (ri + rj - self.config.validity_threshold):
                        # Move circles apart
                        dx = xj - xi
                        dy = yj - yi
                        distance = max(self.config.validity_threshold, dist)
                        
                        # Normalize
                        dx /= distance
                        dy /= distance
                        
                        # Move based on inverse radius ratio
                        move_amount = (ri + rj - dist) * 0.5
                        
                        # Apply movement in opposite directions
                        resolved[i, 0] -= dx * move_amount * 0.4
                        resolved[i, 1] -= dy * move_amount * 0.4
                        resolved[j, 0] += dx * move_amount * 0.4
                        resolved[j, 1] += dy * move_amount * 0.4
                        changed = True
            
            # Ensure bounds
            for i in range(len(resolved)):
                x, y, r = resolved[i]
                # Clamp to valid range
                x = max(r, min(1-r, x))
                y = max(r, min(1-r, y))
                resolved[i] = [x, y, r]
                
            if not changed:
                break
        
        return resolved
    
    def get_grid_cells(self, circles: np.ndarray, grid_size: int = 20) -> Dict[Tuple[int, int], List[int]]:
        """Create a spatial grid for fast neighbor lookups."""
        grid = {}
        cell_size = 1.0 / grid_size
        
        for i, (x, y, r) in enumerate(circles):
            # Determine which grid cells this circle might occupy
            min_x_cell = max(0, int((x - r) / cell_size))
            max_x_cell = min(grid_size - 1, int((x + r) / cell_size))
            min_y_cell = max(0, int((y - r) / cell_size))
            max_y_cell = min(grid_size - 1, int((y + r) / cell_size))
            
            for gx in range(min_x_cell, max_x_cell + 1):
                for gy in range(min_y_cell, max_y_cell + 1):
                    if (gx, gy) not in grid:
                        grid[(gx, gy)] = []
                    grid[(gx, gy)].append(i)
        
        return grid
    
    def enforce_constraints(self, circles: np.ndarray) -> np.ndarray:
        """Enforce constraints on circle positions and radii."""
        result = circles.copy()
        
        # Adjust positions and radii to satisfy bounds
        for i in range(len(result)):
            x, y, r = result[i]
            
            # Ensure circle fits in the unit square
            r = min(r, x, 1-x, y, 1-y)
            r = max(0.001, min(0.49, r))
            
            # Clamp coordinates to valid range
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            
            result[i] = [x, y, r]
        
        return result
    
    def mutate(self, circles: np.ndarray, generation: int, total_generations: int) -> np.ndarray:
        """Mutate a circle configuration with adaptive rates."""
        mutated = circles.copy()
        
        # Adaptive mutation rate using sigmoid decay
        mutation_rate = self.config.mutation_rate_start + (
            self.config.mutation_rate_end - self.config.mutation_rate_start
        ) * (1 / (1 + math.exp(-10 * (generation / total_generations - 0.5))))
        
        n = len(mutated)
        
        # Mutate some circles
        for i in range(n):
            if random.random() < mutation_rate:
                # Randomly choose what to mutate
                choice = random.randint(0, 2)
                
                if choice == 0:  # X coordinate
                    mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0] + random.gauss(0, 0.025)))
                elif choice == 1:  # Y coordinate
                    mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1] + random.gauss(0, 0.025)))
                else:  # Radius
                    mutated[i, 2] = max(0.001, min(0.49, mutated[i, 2] + random.gauss(0, 0.02)))
        
        # Ensure valid configuration after mutation
        return self.enforce_constraints(mutated)
    
    def crossover(self, parent1: CircleConfiguration, parent2: CircleConfiguration) -> CircleConfiguration:
        """Perform crossover between two parent configurations with enhanced recombination."""
        if random.random() > self.config.crossover_prob:
            # Return one of the parents randomly
            return parent1.clone() if random.random() < 0.5 else parent2.clone()
        
        n = len(parent1.circles)
        child_circles = np.zeros_like(parent1.circles)
        
        # Use uniform crossover for better recombination
        for i in range(n):
            # Uniform crossover for each parameter
            if random.random() < 0.5:
                child_circles[i] = parent1.circles[i].copy()
            else:
                child_circles[i] = parent2.circles[i].copy()
            
            # Add some blending for better exploration
            if random.random() < 0.3:  # 30% chance of blending
                alpha = random.random()
                # Blend positions and radii
                child_circles[i][0] = parent1.circles[i][0] + alpha * (parent2.circles[i][0] - parent1.circles[i][0])
                child_circles[i][1] = parent1.circles[i][1] + alpha * (parent2.circles[i][1] - parent1.circles[i][1])
                child_circles[i][2] = parent1.circles[i][2] + alpha * (parent2.circles[i][2] - parent1.circles[i][2])
        
        # Apply local refinement to ensure validity
        refined_child = self.local_refinement(child_circles)
        return CircleConfiguration(refined_child)
    
    def local_refinement(self, circles: np.ndarray, max_iterations: int = 20, generation: int = 0) -> np.ndarray:
        """Apply intensive local refinement to improve overlap resolution and constraint satisfaction."""
        refined = circles.copy()

        # Adaptive parameters based on generation
        if generation < 50:
            max_iter = max_iterations // 2  # Fewer iterations early on
            base_move_scale = 0.3
            force_multiplier = 0.5
        elif generation < 100:
            max_iter = max_iterations  # Normal iterations
            base_move_scale = 0.5
            force_multiplier = 0.7
        else:
            max_iter = max_iterations * 2  # More iterations later for fine-tuning
            base_move_scale = 0.7
            force_multiplier = 1.0

        for iteration in range(max_iter):
            grid = self.get_grid_cells(refined, self.config.initial_grid_size)

            # Check for overlaps and resolve them with more aggressive force-based approach
            resolved = False
            overlap_count = 0
            
            # Track how many overlaps we're resolving per iteration
            for i in range(len(refined)):
                for j in range(i+1, len(refined)):
                    xi, yi, ri = refined[i]
                    xj, yj, rj = refined[j]
                    dist = self.fitness_evaluator.calculate_distance((xi, yi), (xj, yj))

                    if dist < (ri + rj - self.config.validity_threshold):
                        overlap_count += 1
                        # More aggressive for severe overlaps
                        overlap_severity = (ri + rj - dist) / (ri + rj + 0.001)
                        move_scale = base_move_scale * (1.0 + overlap_severity * 0.5)

                        # More aggressive approach for later generations
                        if generation > 150:
                            move_scale *= 1.2

                        # More aggressive overlap resolution with better force distribution
                        dx = xj - xi
                        dy = yj - yi
                        distance = max(self.config.validity_threshold, dist)

                        # Normalize direction vector
                        dx /= distance
                        dy /= distance

                        # Move circles apart with enhanced force scaling
                        move_amount = (ri + rj - dist) * force_multiplier * move_scale

                        # Scale by inverse radii to better balance movement
                        scale_factor = min(1.0, ri / (ri + rj + 0.001))
                        refined[i, 0] -= dx * move_amount * scale_factor * 0.5
                        refined[i, 1] -= dy * move_amount * scale_factor * 0.5
                        refined[j, 0] += dx * move_amount * (1 - scale_factor) * 0.5
                        refined[j, 1] += dy * move_amount * (1 - scale_factor) * 0.5
                        resolved = True

            # Enforce bounds with more precise constraint handling
            for i in range(len(refined)):
                x, y, r = refined[i]
                # Ensure the circle fits properly in the unit square
                max_radius = min(x, 1-x, y, 1-y)
                r = min(r, max_radius)
                r = max(0.001, r)

                # Clamp coordinates to valid range
                x = max(r, min(1-r, x))
                y = max(r, min(1-r, y))
                refined[i] = [x, y, r]

            # Early stopping criteria:
            # 1. No overlaps detected
            # 2. Very few overlaps (stabilizing)
            # 3. Many iterations with no changes (converged)
            if not resolved or overlap_count < 2:
                break

        return refined
    
    def tournament_selection(self, population: List[CircleConfiguration], 
                           fitnesses: List[float], tournament_size: int) -> CircleConfiguration:
        """Select parent using tournament selection."""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_index]
    
    def evolve(self) -> CircleConfiguration:
        """Run the evolutionary algorithm to find optimal circle packing."""
        n = 26
        population = self.initialize_population(n, self.config.population_size)
        
        # Evaluate initial population
        fitnesses = [self.fitness_evaluator.evaluate_fitness(individual.circles, 0, self.config.generations) 
                    for individual in population]
        
        # Evolution loop
        for gen in range(self.config.generations):
            # Selection, crossover, and mutation
            new_population = []
            
            for _ in range(self.config.population_size):
                # Tournament selection
                parent1 = self.tournament_selection(population, fitnesses, self.config.tournament_size)
                parent2 = self.tournament_selection(population, fitnesses, self.config.tournament_size)
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutation
                child.circles = self.mutate(child.circles, gen, self.config.generations)
                child.invalidate_grid()
                
                # Apply local refinement in later generations for better exploitation
                if gen >= self.config.generations * 0.7:  # Start local refinement in later generations
                    child.circles = self.local_refinement(child.circles, generation=gen)
                    child.invalidate_grid()
                
                new_population.append(child)
            
            # Evaluate new population
            population = new_population
            fitnesses = [self.fitness_evaluator.evaluate_fitness(individual.circles, gen, self.config.generations) 
                        for individual in population]
            
            # Print progress
            best_fitness = max(fitnesses)
            if gen % 25 == 0:
                print(f"Generation {gen}: Best fitness = {best_fitness}")
        
        # Return the best individual
        best_index = np.argmax(fitnesses)
        best_solution = population[best_index]
        
        # Final refinement to improve the best solution
        refined_best = self.local_refinement(best_solution.circles, max_iterations=50, generation=self.config.generations)
        return CircleConfiguration(refined_best)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    config = CircleConfig()
    engine = EvolutionEngine(config)
    best_solution = engine.evolve()
    return best_solution.circles

# EVOLVE-BLOCK-END