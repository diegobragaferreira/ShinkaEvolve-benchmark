# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List, Dict, Any
import math
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class CircleConfig:
    """Configuration parameters for circle packing optimization."""
    population_size: int = 150
    generations: int = 200
    tournament_size: int = 5
    mutation_rate_start: float = 0.2
    mutation_rate_end: float = 0.005
    crossover_prob: float = 0.9
    validity_threshold: float = 1e-6
    initial_grid_size: int = 20
    adaptive_grid_start_gen: int = 50
    adaptive_grid_fine_res: int = 30

class Circle:
    """Represents a circle with position and radius."""
    
    def __init__(self, x: float, y: float, r: float):
        self.x = max(0.001, min(0.999, x))
        self.y = max(0.001, min(0.999, y))
        self.r = max(0.001, min(0.49, r))
        # Ensure valid bounds
        self._enforce_bounds()
    
    def _enforce_bounds(self):
        """Enforce boundary constraints."""
        max_radius = min(self.x, 1-self.x, self.y, 1-self.y)
        self.r = min(self.r, max_radius)
        self.x = max(self.r, min(1-self.r, self.x))
        self.y = max(self.r, min(1-self.r, self.y))
    
    def copy(self) -> 'Circle':
        return Circle(self.x, self.y, self.r)
    
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Return (min_x, max_x, min_y, max_y) bounds."""
        return (self.x - self.r, self.x + self.r, self.y - self.r, self.y + self.r)

class SpatialIndexer:
    """Efficient spatial indexing for circle overlap detection."""
    
    def __init__(self, grid_size: int = 20):
        self.grid_size = grid_size
        self.grid_cells = {}
        self.cell_size = 1.0 / grid_size

    def build_index(self, circles: List[Circle], generation: int = 0) -> Dict[Tuple[int, int], List[int]]:
        """Build spatial grid index for efficient neighbor queries."""
        self.grid_cells.clear()
        
        # Adaptive grid resolution based on generation
        adaptive_grid_size = max(15, self.grid_size - int(generation / 50))
        cell_size = 1.0 / adaptive_grid_size
        
        for i, circle in enumerate(circles):
            # Determine which grid cells this circle might occupy
            min_x = max(0, int((circle.x - circle.r) / cell_size))
            max_x = min(adaptive_grid_size - 1, int((circle.x + circle.r) / cell_size))
            min_y = max(0, int((circle.y - circle.r) / cell_size))
            max_y = min(adaptive_grid_size - 1, int((circle.y + circle.r) / cell_size))

            for gx in range(min_x, max_x + 1):
                for gy in range(min_y, max_y + 1):
                    key = (gx, gy)
                    if key not in self.grid_cells:
                        self.grid_cells[key] = []
                    self.grid_cells[key].append(i)
        return self.grid_cells

    def get_neighbors(self, circle: Circle) -> List[int]:
        """Get candidate neighbors within a search radius."""
        neighbors = []
        center_cell = (int(circle.x * self.grid_size), int(circle.y * self.grid_size))

        # Check nearby cells in a 3x3 grid around center
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell = (center_cell[0] + dx, center_cell[1] + dy)
                if cell in self.grid_cells:
                    neighbors.extend(self.grid_cells[cell])

        return neighbors

class ConstraintValidator:
    """Handles all constraint validation efficiently."""
    
    @staticmethod
    def validate_containment(circles: List[Circle]) -> bool:
        """Check if all circles are fully contained in the unit square."""
        if len(circles) == 0:
            return True
            
        # Vectorized check for better performance with many circles
        positions = np.array([(c.x, c.y, c.r) for c in circles])
        x_vals, y_vals, r_vals = positions[:, 0], positions[:, 1], positions[:, 2]
        
        # Check containment in vectorized form
        invalid_mask = (x_vals - r_vals < 0) | (x_vals + r_vals > 1) | \
                      (y_vals - r_vals < 0) | (y_vals + r_vals > 1)
        
        return not np.any(invalid_mask)

    @staticmethod
    def validate_overlap(circles: List[Circle], spatial_indexer: SpatialIndexer = None) -> bool:
        """Check for circle overlaps using spatial indexing for efficiency."""
        n = len(circles)
        if n <= 1:
            return True

        # Brute force for small populations
        if n <= 50:
            for i in range(n):
                for j in range(i+1, n):
                    circle1, circle2 = circles[i], circles[j]
                    distance = math.sqrt((circle1.x - circle2.x)**2 + (circle1.y - circle2.y)**2)
                    if distance < (circle1.r + circle2.r - 1e-8):
                        return False
            return True

        # Use spatial indexing for larger populations
        if spatial_indexer is not None and n > 50:
            # Use grid-based approach for efficiency
            for i, circle in enumerate(circles):
                neighbors = spatial_indexer.get_neighbors(circle)
                for j in neighbors:
                    if i != j:
                        other = circles[j]
                        distance = math.sqrt((circle.x - other.x)**2 + (circle.y - other.y)**2)
                        if distance < (circle.r + other.r - 1e-8):
                            return False
            return True
        
        # Fallback brute force
        for i in range(n):
            for j in range(i+1, n):
                circle1, circle2 = circles[i], circles[j]
                distance = math.sqrt((circle1.x - circle2.x)**2 + (circle1.y - circle2.y)**2)
                if distance < (circle1.r + circle2.r - 1e-8):
                    return False
        return True

    @staticmethod
    def enforce_bounds(circles: List[Circle]) -> List[Circle]:
        """Enforce boundary constraints by adjusting positions and radii."""
        result = [circle.copy() for circle in circles]
        
        for circle in result:
            # Ensure circle fits in the unit square
            max_radius = min(circle.x, 1-circle.x, circle.y, 1-circle.y)
            circle.r = min(circle.r, max_radius)
            circle.r = max(0.001, min(0.49, circle.r))
            
            # Clamp coordinates to valid range
            circle.x = max(circle.r, min(1-circle.r, circle.x))
            circle.y = max(circle.r, min(1-circle.r, circle.y))
            
        return result

class FitnessEvaluator:
    """Evaluates fitness with optimized penalty system."""
    
    def __init__(self):
        self._cache = {}  # Simple caching for repeated evaluations

    def evaluate(self, circles: List[Circle], spatial_indexer: SpatialIndexer = None) -> float:
        """Evaluate fitness with constraint penalties."""
        # Create a hashable representation for caching
        cache_key = tuple((c.x, c.y, c.r) for c in circles)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Check constraints
        if not ConstraintValidator.validate_containment(circles):
            penalty = self._compute_boundary_penalty(circles)
            result = -penalty
            self._cache[cache_key] = result
            return result

        if not ConstraintValidator.validate_overlap(circles, spatial_indexer):
            penalty = self._compute_overlap_penalty(circles)
            result = -penalty
            self._cache[cache_key] = result
            return result
            
        # Valid solution - return sum of radii
        result = float(sum(c.r for c in circles))
        self._cache[cache_key] = result
        return result

    def _compute_boundary_penalty(self, circles: List[Circle]) -> float:
        """Compute penalty based on boundary violations."""
        penalty = 0.0
        
        # Vectorized computation for better performance
        positions = np.array([(c.x, c.y, c.r) for c in circles])
        x_vals, y_vals, r_vals = positions[:, 0], positions[:, 1], positions[:, 2]
        
        # Calculate violations for all circles at once
        left_violation = np.maximum(0, r_vals - x_vals)
        right_violation = np.maximum(0, r_vals - (1 - x_vals))
        bottom_violation = np.maximum(0, r_vals - y_vals)
        top_violation = np.maximum(0, r_vals - (1 - y_vals))
        
        penalty = (np.sum(left_violation) + np.sum(right_violation) +
                  np.sum(bottom_violation) + np.sum(top_violation)) * 1000.0
        
        return penalty

    def _compute_overlap_penalty(self, circles: List[Circle]) -> float:
        """Compute penalty based on overlap violations."""
        penalty = 0.0
        
        # Use spatial indexing for efficiency
        if len(circles) > 50:
            # Build spatial index for efficient neighbor search
            indexer = SpatialIndexer()
            indexer.build_index(circles)
            
            # Only check pairs that could possibly overlap based on spatial index
            checked_pairs = set()
            for i, circle in enumerate(circles):
                neighbors = indexer.get_neighbors(circle)
                for j in neighbors:
                    if i < j and (i, j) not in checked_pairs:
                        circle1, circle2 = circles[i], circles[j]
                        distance = math.sqrt((circle1.x - circle2.x)**2 + (circle1.y - circle2.y)**2)
                        
                        if distance < (circle1.r + circle2.r):
                            overlap = (circle1.r + circle2.r - distance)
                            penalty += overlap * 100000.0
                            checked_pairs.add((i, j))
        else:
            # Brute force for smaller populations
            n = len(circles)
            for i in range(n):
                for j in range(i+1, n):
                    circle1, circle2 = circles[i], circles[j]
                    distance = math.sqrt((circle1.x - circle2.x)**2 + (circle1.y - circle2.y)**2)
                    
                    if distance < (circle1.r + circle2.r):
                        overlap = (circle1.r + circle2.r - distance)
                        penalty += overlap * 100000.0
                        
        return penalty

class CircleInitializer:
    """Implements advanced circle initialization strategies."""
    
    @staticmethod
    def poisson_disk_sampling(n_points: int, min_distance: float = 0.1) -> List[Tuple[float, float]]:
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

    @classmethod
    def initialize_population(cls, n: int, population_size: int) -> List[List[Circle]]:
        """Initialize population with improved Voronoi-based distribution using Poisson disk sampling."""
        population = []

        # Generate points using Poisson disk sampling for better distribution
        sample_points = cls.poisson_disk_sampling(n, 0.15)

        # Create multiple populations with variation
        for _ in range(population_size):
            circles = []

            # Distribute circles using the sample points
            for i in range(min(n, len(sample_points))):
                x_base, y_base = sample_points[i]

                # Add jitter for diversity
                x = max(0.01, min(0.99, x_base + random.uniform(-0.03, 0.03)))
                y = max(0.01, min(0.99, y_base + random.uniform(-0.03, 0.03)))

                # Initial radius - start with moderately large values
                circles.append(Circle(x, y, 0.06))

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

                circles.append(Circle(x, y, 0.025))

            # Ensure circles don't overlap initially
            circles = cls.resolve_initial_overlaps(circles)
            population.append(circles)

        return population

    @staticmethod
    def resolve_initial_overlaps(circles: List[Circle]) -> List[Circle]:
        """Resolve overlaps in initial configuration using force-based approach."""
        resolved = [circle.copy() for circle in circles]

        # Iteratively resolve overlaps
        for iteration in range(10):
            changed = False
            indexer = SpatialIndexer(20)
            indexer.build_index(resolved)

            for (gx, gy), indices in indexer.grid_cells.items():
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        idx1, idx2 = indices[i], indices[j]
                        c1, c2 = resolved[idx1], resolved[idx2]
                        dist = math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)

                        if dist < (c1.r + c2.r - 1e-6):
                            # Move circles apart with more aggressive force
                            dx = c2.x - c1.x
                            dy = c2.y - c1.y
                            distance = max(1e-6, dist)

                            # Normalize
                            dx /= distance
                            dy /= distance

                            # Move based on inverse radius ratio with stronger force
                            move_amount = (c1.r + c2.r - dist) * 0.7

                            # Apply movement in opposite directions with stronger push
                            resolved[idx1].x -= dx * move_amount * 0.5
                            resolved[idx1].y -= dy * move_amount * 0.5
                            resolved[idx2].x += dx * move_amount * 0.5
                            resolved[idx2].y += dy * move_amount * 0.5
                            changed = True

            # Ensure bounds
            for circle in resolved:
                circle.x = max(circle.r, min(1-circle.r, circle.x))
                circle.y = max(circle.r, min(1-circle.r, circle.y))

            if not changed:
                break

        return resolved

class GeneticAlgorithmOptimizer:
    """Main evolutionary optimization engine."""
    
    def __init__(self, config: CircleConfig):
        self.config = config
        self.evaluator = FitnessEvaluator()
        self.indexer = SpatialIndexer(config.initial_grid_size)
        random.seed(42)
        np.random.seed(42)
        
    def adapt_mutation_rate(self, generation: int, total_generations: int) -> float:
        """Adaptive mutation rate with sigmoid decay."""
        progress = generation / total_generations
        return self.config.mutation_rate_start + (self.config.mutation_rate_end - self.config.mutation_rate_start) * \
               (1 / (1 + math.exp(-10 * (progress - 0.5))))

    def tournament_selection(self, population: List[List[Circle]],
                           fitness_scores: List[float]) -> List[Circle]:
        """Select parent using tournament selection."""
        tournament_indices = random.sample(range(len(population)), self.config.tournament_size)
        tournament_fitnesses = [fitness_scores[i] for i in tournament_indices]

        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_index]

    def crossover(self, parent1: List[Circle], parent2: List[Circle]) -> List[Circle]:
        """Perform crossover between two parent configurations."""
        if random.random() > self.config.crossover_prob:
            # Return one of the parents randomly
            return [circle.copy() for circle in (parent1 if random.random() < 0.5 else parent2)]

        n = len(parent1)
        child = [None] * n

        # Use uniform crossover for better recombination
        for i in range(n):
            # Uniform crossover for each parameter
            if random.random() < 0.5:
                child[i] = parent1[i].copy()
            else:
                child[i] = parent2[i].copy()

            # Add some blending for better exploration
            if random.random() < 0.3:  # 30% chance of blending
                alpha = random.random()
                # Blend positions and radii
                child[i].x = parent1[i].x + alpha * (parent2[i].x - parent1[i].x)
                child[i].y = parent1[i].y + alpha * (parent2[i].y - parent1[i].y)
                child[i].r = parent1[i].r + alpha * (parent2[i].r - parent1[i].r)

        # Apply local refinement to ensure validity
        return self.refine_configuration(child)

    def mutate(self, circles: List[Circle], generation: int, total_generations: int) -> List[Circle]:
        """Mutate a circle configuration with adaptive rates."""
        mutated = [circle.copy() for circle in circles]

        # Adaptive mutation rate with better curve
        mutation_rate = self.adapt_mutation_rate(generation, total_generations)
        
        # Dynamic mutation strength based on generation
        if generation < 50:
            # Early generations: moderate mutation
            mutation_strength = 0.03
        elif generation < 150:
            # Mid generations: smaller mutation
            mutation_strength = 0.015
        else:
            # Late generations: very small mutation
            mutation_strength = 0.005

        # Mutate some circles
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Randomly choose what to mutate
                choice = random.randint(0, 2)

                if choice == 0:  # X coordinate
                    mutated[i].x = max(0.01, min(0.99, mutated[i].x + random.gauss(0, mutation_strength)))
                elif choice == 1:  # Y coordinate
                    mutated[i].y = max(0.01, min(0.99, mutated[i].y + random.gauss(0, mutation_strength)))
                else:  # Radius
                    mutated[i].r = max(0.001, min(0.49, mutated[i].r + random.gauss(0, mutation_strength/2)))

        # Ensure valid configuration after mutation
        return self.enforce_constraints(mutated)

    def refine_configuration(self, circles: List[Circle]) -> List[Circle]:
        """Refine configuration to remove overlaps and correct constraints."""
        refined = [circle.copy() for circle in circles]

        # More aggressive refinement in earlier stages
        max_iterations = 15
        for iteration in range(max_iterations):
            resolved = False
            
            # Use direct overlap checking for simplicity
            for i in range(len(refined)):
                for j in range(i+1, len(refined)):
                    c1, c2 = refined[i], refined[j]
                    dist = math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)

                    if dist < (c1.r + c2.r - 1e-6):
                        # Resolve overlap by moving circles apart with force-based approach
                        dx = c2.x - c1.x
                        dy = c2.y - c1.y
                        distance = max(1e-6, dist)

                        # Normalize direction vector
                        dx /= distance
                        dy /= distance

                        # Move circles apart with more aggressive force
                        move_amount = (c1.r + c2.r - dist) * 0.7

                        # Scale by inverse radii to balance movement
                        scale_factor = min(1.0, c1.r / (c1.r + c2.r + 0.001))
                        refined[i].x -= dx * move_amount * scale_factor * 0.4
                        refined[i].y -= dy * move_amount * scale_factor * 0.4
                        refined[j].x += dx * move_amount * (1 - scale_factor) * 0.4
                        refined[j].y += dy * move_amount * (1 - scale_factor) * 0.4
                        resolved = True

            # Enforce bounds
            for circle in refined:
                circle.x = max(circle.r, min(1-circle.r, circle.x))
                circle.y = max(circle.r, min(1-circle.r, circle.y))

            # Early stopping if no changes made
            if not resolved:
                break

        return refined

    def enforce_constraints(self, circles: List[Circle]) -> List[Circle]:
        """Enforce constraints on circle positions and radii."""
        result = [circle.copy() for circle in circles]

        # Adjust positions and radii to satisfy bounds
        for circle in result:
            # Ensure circle fits in the unit square
            circle.r = min(circle.r, circle.x, 1-circle.x, circle.y, 1-circle.y)
            circle.r = max(0.001, min(0.49, circle.r))

            # Clamp coordinates to valid range
            circle.x = max(circle.r, min(1-circle.r, circle.x))
            circle.y = max(circle.r, min(1-circle.r, circle.y))

        return result

    def optimize(self, initial_population: List[List[Circle]]) -> List[Circle]:
        """Run the genetic algorithm optimization."""
        population = initial_population
        best_fitness_history = []

        # Evaluate initial population
        fitness_scores = []
        for individual in population:
            fitness = self.evaluator.evaluate(individual)
            fitness_scores.append(fitness)

        # Evolution loop
        for gen in range(self.config.generations):
            # Selection, crossover, and mutation
            new_population = []

            # Elitism: keep the best individual
            best_index = np.argmax(fitness_scores)
            new_population.append(population[best_index])

            # Generate offspring
            while len(new_population) < self.config.population_size:
                # Tournament selection
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)

                # Crossover
                child = self.crossover(parent1, parent2)

                # Mutation
                child = self.mutate(child, gen, self.config.generations)

                new_population.append(child)

            # Trim to exact population size
            population = new_population[:self.config.population_size]

            # Evaluate new population
            fitness_scores = []
            for individual in population:
                fitness = self.evaluator.evaluate(individual)
                fitness_scores.append(fitness)

            # Track best fitness
            best_fitness = max(fitness_scores)
            best_fitness_history.append(best_fitness)

            # Print progress
            if gen % 25 == 0:
                print(f"Generation {gen}: Best fitness = {best_fitness}")

        # Return the best individual
        best_index = np.argmax(fitness_scores)
        return population[best_index]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    config = CircleConfig()
    initializer = CircleInitializer()
    optimizer = GeneticAlgorithmOptimizer(config)
    
    # Initialize population
    population = initializer.initialize_population(26, config.population_size)

    # Run optimization
    best_solution = optimizer.optimize(population)

    # Convert back to numpy array format
    result = np.array([[c.x, c.y, c.r] for c in best_solution])
    return result

# EVOLVE-BLOCK-END