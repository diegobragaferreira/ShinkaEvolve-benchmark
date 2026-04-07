# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import random
from typing import Tuple, List
import time
import warnings
from itertools import combinations

# Global constants
RECT_PERIMETER = 4.0
RECT_WIDTH = 1.2
RECT_HEIGHT = 0.8
NUM_CIRCLES = 21
SEED = 42

class CirclePackingOptimizer:
    def __init__(self, width: float = RECT_WIDTH, height: float = RECT_HEIGHT,
                 num_circles: int = NUM_CIRCLES):
        self.width = width
        self.height = height
        self.num_circles = num_circles
        self.rect_area = width * height
        self.best_solution = None
        self.best_fitness = float('-inf')

        # Initialize random seed for reproducibility
        np.random.seed(SEED)
        random.seed(SEED)

    def is_valid_position(self, x: float, y: float, r: float) -> bool:
        """Check if circle center is within bounds"""
        return (r <= x <= self.width - r and
                r <= y <= self.height - r)

    def is_valid_circle(self, x: float, y: float, r: float) -> bool:
        """Check if circle is valid (within bounds and positive radius)"""
        return (0 < r and
                self.is_valid_position(x, y, r))

    def check_overlap_simple(self, circles: np.ndarray, idx1: int, idx2: int) -> bool:
        """Simple overlap check using precomputed squared distances"""
        x1, y1, r1 = circles[idx1]
        x2, y2, r2 = circles[idx2]
        dx = x1 - x2
        dy = y1 - y2
        dist_sq = dx*dx + dy*dy
        radius_sum = r1 + r2
        return dist_sq < radius_sum * radius_sum

    def calculate_total_radius_sum(self, circles: np.ndarray) -> float:
        """Calculate sum of all circle radii"""
        return np.sum(circles[:, 2])

    def calculate_fitness(self, circles: np.ndarray) -> Tuple[float, int]:
        """
        Calculate fitness: sum of radii with penalty for constraint violations

        Returns:
            Tuple of (fitness_score, number_of_violations)
        """
        total_radius = self.calculate_total_radius_sum(circles)

        # Calculate constraint violations with proper penalty scaling
        boundary_violations = 0
        overlap_violations = 0
        total_penalty = 0

        # Check boundary violations with precise measurement
        for i in range(self.num_circles):
            x, y, r = circles[i]
            if not self.is_valid_circle(x, y, r):
                # Calculate exact boundary violation amounts
                left_violation = max(0, r - x)
                right_violation = max(0, x + r - self.width)
                bottom_violation = max(0, r - y)
                top_violation = max(0, y + r - self.height)
                violation_amount = left_violation + right_violation + bottom_violation + top_violation
                total_penalty += 1000 * violation_amount  # Reduced penalty for better exploration
                boundary_violations += 1

        # Check overlap violations with better penalty calculation
        try:
            points = circles[:, :2]
            tree = cKDTree(points)

            max_radius = np.max(circles[:, 2])
            if max_radius > 0:
                pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')

                for i, j in pairs:
                    if i < j:  # Avoid double counting
                        x1, y1, r1 = circles[i]
                        x2, y2, r2 = circles[j]

                        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        if distance < (r1 + r2):
                            # Calculate overlap amount and weight penalty appropriately
                            overlap = (r1 + r2) - distance
                            penalty_scaling = 200.0 * (r1 + r2)  # Reduced penalty for better exploration
                            total_penalty += overlap * penalty_scaling
                            overlap_violations += 1

        except Exception as e:
            warnings.warn(f"Error in overlap checking: {e}")
            # Fallback to brute force when spatial indexing fails
            for i in range(self.num_circles):
                for j in range(i+1, self.num_circles):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]

                    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    if distance < (r1 + r2):
                        overlap = (r1 + r2) - distance
                        penalty_scaling = 200.0 * (r1 + r2)
                        total_penalty += overlap * penalty_scaling
                        overlap_violations += 1

        # Return fitness score with properly weighted penalties
        return total_radius - total_penalty, boundary_violations + overlap_violations

    def generate_hexagonal_grid_initialization(self) -> np.ndarray:
        """Generate initial configuration using hexagonal lattice approach"""
        circles = np.zeros((self.num_circles, 3))

        # Calculate grid parameters based on rectangle aspect ratio
        aspect_ratio = self.width / self.height
        sqrt_n = np.sqrt(self.num_circles)

        # Determine grid dimensions for hexagonal packing
        cols = max(1, int(np.ceil(sqrt_n * np.sqrt(aspect_ratio))))
        rows = max(1, int(np.ceil(self.num_circles / cols)))

        # Ensure sufficient grid space
        while cols * rows < self.num_circles:
            cols += 1
            rows = max(1, int(np.ceil(self.num_circles / cols)))

        # Calculate spacing
        cell_width = self.width / cols
        cell_height = self.height / rows
        
        # Hexagonal spacing parameters
        hex_width = cell_width
        hex_height = cell_height * np.sqrt(3) / 2

        # Generate hexagonal grid with randomness
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= self.num_circles:
                    break
                # Hexagonal offset
                offset = (i % 2) * (hex_width / 2)
                x = (j + 0.5) * hex_width + offset + np.random.uniform(-hex_width * 0.1, hex_width * 0.1)
                y = (i + 0.5) * hex_height + np.random.uniform(-hex_height * 0.1, hex_height * 0.1)
                
                # Ensure within bounds
                x = max(0.01, min(self.width - 0.01, x))
                y = max(0.01, min(self.height - 0.01, y))

                # Base radius with variation
                base_radius = min(hex_width, hex_height) * 0.3
                r = base_radius + np.random.uniform(-base_radius * 0.15, base_radius * 0.15)
                r = max(0.01, min(0.2, r))

                circles[idx] = [x, y, r]
                idx += 1

        return circles

    def generate_voronoi_based_initialization(self) -> np.ndarray:
        """Generate initial configuration using Voronoi-inspired approach"""
        circles = np.zeros((self.num_circles, 3))

        # Start with random points
        points = np.random.rand(self.num_circles, 2)
        points[:, 0] *= self.width
        points[:, 1] *= self.height
        
        # Create Voronoi diagram to estimate density
        # For simplicity, just use points as circle centers with varying radii
        for i in range(self.num_circles):
            x, y = points[i]
            # Calculate distance to nearest neighbors for radius estimation
            distances = []
            for j in range(self.num_circles):
                if i != j:
                    dx = points[i, 0] - points[j, 0]
                    dy = points[i, 1] - points[j, 1]
                    distances.append(np.sqrt(dx*dx + dy*dy))
            
            # Set radius based on neighborhood density
            if len(distances) > 0:
                avg_dist = np.mean(distances)
                r = min(avg_dist * 0.3, 0.2)  # Cap at reasonable size
                r = max(0.01, min(0.2, r))
            else:
                r = np.random.uniform(0.02, 0.1)
            
            # Ensure within bounds
            x = max(r, min(self.width - r, x))
            y = max(r, min(self.height - r, y))
            
            circles[i] = [x, y, r]
        
        return circles

    def initialize_population(self, pop_size: int) -> List[np.ndarray]:
        """Generate diverse initial population with enhanced strategic placement"""
        population = []
        
        # Mix of initialization strategies
        strategies = [
            self.generate_hexagonal_grid_initialization,
            self.generate_voronoi_based_initialization
        ]
        
        for _ in range(pop_size):
            # Choose strategy randomly for diversity
            strategy = random.choice(strategies)
            circles = strategy()
            
            # Add small perturbation for variety
            for i in range(self.num_circles):
                if np.random.rand() < 0.3:  # 30% chance to perturb
                    circles[i, 0] += np.random.uniform(-0.05, 0.05)
                    circles[i, 1] += np.random.uniform(-0.05, 0.05)
                    circles[i, 0] = max(circles[i, 2], min(self.width - circles[i, 2], circles[i, 0]))
                    circles[i, 1] = max(circles[i, 2], min(self.height - circles[i, 2], circles[i, 1]))
            
            population.append(circles)

        return population

    def tournament_selection(self, population: List[np.ndarray],
                           fitness_scores: List[Tuple[float, int]]) -> np.ndarray:
        """Tournament selection with better diversity"""
        tournament_size = min(5, len(population))
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [(i, fitness_scores[i][0]) for i in tournament_indices]
        tournament_fitness.sort(key=lambda x: x[1], reverse=True)
        return population[tournament_fitness[0][0]].copy()

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Uniform crossover with improved trait preservation"""
        child = parent1.copy()

        # Use fitness-based bias toward better parent
        parent1_fitness, _ = self.calculate_fitness(parent1)
        parent2_fitness, _ = self.calculate_fitness(parent2)

        # Select better parent with higher probability
        better_parent = parent1 if parent1_fitness >= parent2_fitness else parent2
        worse_parent = parent2 if parent1_fitness >= parent2_fitness else parent1
        
        # Bias: 70% chance to inherit from better parent
        mask = np.random.rand(self.num_circles) > 0.3

        for i in range(self.num_circles):
            if mask[i]:
                child[i] = better_parent[i].copy()
            else:
                child[i] = worse_parent[i].copy()

        return child

    def mutate(self, individual: np.ndarray, mutation_rate: float, generation: int = 0) -> np.ndarray:
        """Mutation with adaptive step sizing and better balance"""
        mutated = individual.copy()

        for i in range(self.num_circles):
            if np.random.rand() < mutation_rate:
                # 65% position mutation, 35% radius mutation (slightly more position emphasis)
                if np.random.rand() < 0.65:
                    # Position mutation with adaptive step
                    # Decrease mutation strength as generation progresses
                    adaptive_step = 0.1 * (1.0 - generation/200) * mutation_rate
                    mutated[i, 0] = np.random.uniform(
                        max(0.001, mutated[i, 0] - adaptive_step),
                        min(self.width - 0.001, mutated[i, 0] + adaptive_step)
                    )
                    mutated[i, 1] = np.random.uniform(
                        max(0.001, mutated[i, 1] - adaptive_step),
                        min(self.height - 0.001, mutated[i, 1] + adaptive_step)
                    )
                else:
                    # Radius mutation with careful bounds
                    step = 0.03 * (1.0 - generation/200) * mutation_rate
                    mutated[i, 2] = np.random.uniform(
                        max(0.001, mutated[i, 2] - step),
                        min(0.2, mutated[i, 2] + step)
                    )

        return mutated

    def local_optimization(self, circles: np.ndarray) -> np.ndarray:
        """Advanced local optimization for fine-tuning with enhanced reliability"""
        optimized = circles.copy()

        # Sequential optimization focusing on improving each circle's radius
        improved_count = 0
        max_iter = 100
        iterations = 0
        
        while improved_count < self.num_circles and iterations < max_iter:
            improved_count = 0
            # Try to maximize each circle's radius
            for i in range(self.num_circles):
                current_x, current_y, current_r = optimized[i]

                # Create objective function that tries to maximize radius of circle i
                def radius_objective(r):
                    temp_circles = optimized.copy()
                    temp_circles[i, 2] = r[0]

                    # Check validity
                    if not self.is_valid_circle(temp_circles[i, 0], temp_circles[i, 1], r[0]):
                        return 1e10

                    # Check overlaps - more efficient version
                    try:
                        violations = 0
                        # Direct overlap check for efficiency
                        for j in range(self.num_circles):
                            if i != j and self.check_overlap_simple(temp_circles, i, j):
                                violations += 1
                        return -r[0] + 200 * violations  # Strong penalty
                    except:
                        return 1e10  # Fallback

                # Maximize radius with bounds - more conservative approach
                bounds = [(1e-6, min(self.width/2, self.height/2, current_r * 2))]
                try:
                    # Use L-BFGS-B for optimization
                    result = minimize(radius_objective, [current_r], bounds=bounds, method='L-BFGS-B', tol=1e-6)
                    if result.success:
                        new_r = max(1e-6, result.x[0])
                        # Only accept improvement if significant
                        if new_r > current_r + 1e-5:
                            optimized[i, 2] = new_r
                            improved_count += 1
                except:
                    # If optimization fails, continue
                    pass

            iterations += 1

        return optimized

    def optimize(self) -> np.ndarray:
        """Main optimization loop with adaptive approach"""
        start_time = time.time()

        # Phase 1: Initialize population with enhanced diversity
        population = self.initialize_population(120)

        # Phase 2: Multi-stage evolutionary optimization
        stagnant_generations = 0
        max_stagnant_generations = 25
        best_fitness_history = []
        
        for generation in range(200):  # Increased generations for better convergence
            
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness, violations = self.calculate_fitness(individual)
                fitness_scores.append((fitness, violations))

            # Track best solution
            gen_best_idx = np.argmax([f[0] for f in fitness_scores])
            gen_best_fitness = fitness_scores[gen_best_idx][0]

            if gen_best_fitness > self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.best_solution = population[gen_best_idx].copy()
                stagnant_generations = 0
                best_fitness_history.clear()
            else:
                stagnant_generations += 1
                
            best_fitness_history.append(gen_best_fitness)
            
            # Print progress
            if generation % 20 == 0:
                print(f"Generation {generation}: Best fitness = {gen_best_fitness:.6f}")

            # Early stopping based on convergence
            if len(best_fitness_history) >= 10:
                recent_avg = np.mean(best_fitness_history[-10:])
                prev_avg = np.mean(best_fitness_history[-20:-10])
                if abs(recent_avg - prev_avg) < 1e-5:
                    print(f"Converged at generation {generation}")
                    break

            # Adaptive population size and mutation rate adjustments
            adaptive_pop_size = max(100, 120 - generation // 5)
            mutation_rate = max(0.05, 0.3 * (1.0 - generation/200))
            
            # Create new population
            new_population = []

            # Elitism: keep best individual
            new_population.append(self.best_solution.copy())

            # Generate offspring
            while len(new_population) < adaptive_pop_size:
                # Selection
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)

                # Crossover
                child = self.crossover(parent1, parent2)

                # Mutation
                child = self.mutate(child, mutation_rate, generation)

                new_population.append(child)

            population = new_population[:adaptive_pop_size]

        end_time = time.time()
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")
        print(f"Best fitness achieved: {self.best_fitness:.6f}")

        # Phase 3: Final refinement
        if self.best_solution is not None:
            # Apply local optimization
            self.best_solution = self.local_optimization(self.best_solution)

            # Final verification
            final_fitness, _ = self.calculate_fitness(self.best_solution)
            print(f"Final refined fitness: {final_fitness:.6f}")

        return self.best_solution

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Create optimizer instance
    optimizer = CirclePackingOptimizer(width=RECT_WIDTH, height=RECT_HEIGHT, num_circles=NUM_CIRCLES)

    # Run optimization
    circles = optimizer.optimize()

    # Ensure valid output
    if circles is None or len(circles) != NUM_CIRCLES:
        circles = np.zeros((NUM_CIRCLES, 3))
        np.random.seed(SEED)
        for i in range(NUM_CIRCLES):
            circles[i, 0] = np.random.uniform(0.01, RECT_WIDTH - 0.01)
            circles[i, 1] = np.random.uniform(0.01, RECT_HEIGHT - 0.01)
            circles[i, 2] = np.random.uniform(0.01, 0.15)

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")