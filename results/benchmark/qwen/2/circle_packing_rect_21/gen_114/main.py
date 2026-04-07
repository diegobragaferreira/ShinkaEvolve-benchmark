# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import differential_evolution
import random
from typing import Tuple, List
import time
import warnings
from scipy.spatial.distance import cdist

# Global constants
RECT_PERIMETER = 4.0
NUM_CIRCLES = 21
POPULATION_SIZE = 75
MAX_GENERATIONS = 250
INITIAL_MUTATION_RATE = 0.2
FINAL_MUTATION_RATE = 0.03
TOURNAMENT_SIZE = 5
SEED = 42

class CirclePacker:
    def __init__(self, width: float = 1.0, height: float = 1.0,
                 num_circles: int = NUM_CIRCLES):
        self.width = width
        self.height = height
        self.num_circles = num_circles
        self.rect_area = width * height

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

    def check_overlap(self, circles: np.ndarray, idx1: int, idx2: int) -> bool:
        """Check if two circles overlap using Euclidean distance"""
        x1, y1, r1 = circles[idx1]
        x2, y2, r2 = circles[idx2]

        # Calculate squared distance to avoid sqrt computation
        dx = x1 - x2
        dy = y1 - y2
        dist_sq = dx*dx + dy*dy
        radius_sum = r1 + r2
        return dist_sq < radius_sum * radius_sum

    def efficient_overlap_check(self, circles: np.ndarray, tree: cKDTree = None) -> int:
        """Efficiently check all overlaps using spatial indexing"""
        violations = 0

        if tree is None:
            # Build KDTree for fast neighbor search
            points = circles[:, :2]  # Only x,y coordinates
            tree = cKDTree(points)

        # Get max radius to determine search radius
        max_radius = np.max(circles[:, 2])

        # Query pairs efficiently
        try:
            pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')

            for i, j in pairs:
                if self.check_overlap(circles, i, j):
                    violations += 1
        except Exception:
            # Fallback to brute force if spatial indexing fails
            for i in range(self.num_circles):
                for j in range(i+1, self.num_circles):
                    if self.check_overlap(circles, i, j):
                        violations += 1

        return violations

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

        # Count constraint violations
        violations = 0

        # Check boundary violations
        for i in range(self.num_circles):
            x, y, r = circles[i]
            if not self.is_valid_circle(x, y, r):
                violations += 100  # Heavy penalty for boundary violations

        # Check overlap violations using optimized spatial indexing
        violations += self.efficient_overlap_check(circles)

        # Return negative penalty (since we want to maximize) plus positive radius sum
        # Adjust penalty weight for better balance
        penalty_weight = 1000.0
        return total_radius - (penalty_weight * violations), violations

    def generate_optimal_rectangle_dimensions(self) -> Tuple[float, float]:
        """
        Compute optimal rectangle dimensions based on the circle packing problem
        For 21 circles, we want to find the aspect ratio that gives maximum packing efficiency.
        """
        # Empirical analysis suggests that for 21 circles, a near-square to slightly wider rectangle works well
        # We'll try several ratios and select the one that seems promising
        candidates = [
            (1.0, 1.0),     # Square
            (1.2, 0.8333),   # Slightly wider
            (1.4, 0.7143),   # Wider
            (1.6, 0.625),    # Even wider
            (1.8, 0.5556),   # Very wide
            (0.8333, 1.2),   # Slightly taller
            (0.7143, 1.4),   # Taller
        ]
        
        best_ratio = (1.0, 1.0)
        best_density = 0.0
        
        for w, h in candidates:
            if abs(w + h - 2.0) < 0.01:  # Ensure perimeter constraint
                # Estimate packing density (simplified)
                area_per_circle = (w * h) / self.num_circles
                max_radius = np.sqrt(area_per_circle / np.pi) * 0.9  # Conservative
                density = (self.num_circles * np.pi * max_radius * max_radius) / (w * h)
                if density > best_density:
                    best_density = density
                    best_ratio = (w, h)
        
        return best_ratio

    def generate_initial_population(self, pop_size: int) -> List[np.ndarray]:
        """Generate initial population of circle configurations"""
        population = []
        
        # Determine optimal rectangle dimensions
        optimal_width, optimal_height = self.generate_optimal_rectangle_dimensions()
        self.width = optimal_width
        self.height = optimal_height

        for _ in range(pop_size):
            circles = np.zeros((self.num_circles, 3))

            # Generate initial configuration using a more advanced grid-based approach
            # This follows a hexagonal packing pattern more carefully
            
            # Calculate grid dimensions more intelligently
            aspect_ratio = self.width / self.height
            
            # Try to create a grid that fits well with the aspect ratio
            if aspect_ratio >= 1:  # Width >= height
                sqrt_n = np.sqrt(self.num_circles)
                cols = int(np.ceil(sqrt_n * np.sqrt(aspect_ratio)))
                rows = int(np.ceil(self.num_circles / cols))
                
                # Ensure we have enough cells
                while cols * rows < self.num_circles:
                    cols += 1
            else:  # Height > width
                sqrt_n = np.sqrt(self.num_circles)
                rows = int(np.ceil(sqrt_n * np.sqrt(1/aspect_ratio)))
                cols = int(np.ceil(self.num_circles / rows))
                
                # Ensure we have enough cells
                while cols * rows < self.num_circles:
                    rows += 1

            # Calculate spacing with proper bounds
            spacing_x = self.width / (cols + 1) if cols > 0 else self.width
            spacing_y = self.height / (rows + 1) if rows > 0 else self.height

            # Create better hexagonal-like packing pattern
            placed_count = 0
            for i in range(rows):
                for j in range(cols):
                    if placed_count >= self.num_circles:
                        break

                    # Offset every other row for better packing
                    offset_x = spacing_x * 0.5 if i % 2 == 1 else 0
                    base_x = (j + 1) * spacing_x + offset_x
                    base_y = (i + 1) * spacing_y

                    # Apply tighter bounds for positioning
                    safe_x = max(0.01, min(self.width - 0.01, base_x))
                    safe_y = max(0.01, min(self.height - 0.01, base_y))

                    # Add adaptive perturbation based on available space
                    perturbation_factor = min(0.15, 0.2 * min(spacing_x, spacing_y))
                    x = np.clip(safe_x + np.random.uniform(-perturbation_factor, perturbation_factor),
                               0.01, self.width - 0.01)
                    y = np.clip(safe_y + np.random.uniform(-perturbation_factor, perturbation_factor),
                               0.01, self.height - 0.01)

                    # Better initial radius estimation
                    max_r = min(x, self.width - x, y, self.height - y)
                    # Use more sophisticated estimation for 21 circles
                    estimated_radius = min(0.15, max_r * 0.75)
                    # Small variation in initial radii to prevent premature convergence
                    r = np.random.uniform(estimated_radius * 0.6, estimated_radius * 1.2)

                    circles[placed_count] = [x, y, r]
                    placed_count += 1

                if placed_count >= self.num_circles:
                    break

            # Add some randomness to initial population
            for i in range(self.num_circles):
                if np.random.rand() < 0.1:  # 10% chance to modify
                    circles[i, 2] = np.clip(circles[i, 2] * np.random.uniform(0.8, 1.2), 0.001, 0.3)

            population.append(circles)

        return population

    def tournament_selection(self, population: List[np.ndarray],
                           fitness_scores: List[Tuple[float, int]]) -> np.ndarray:
        """Select individual using tournament selection"""
        tournament_indices = np.random.choice(len(population), TOURNAMENT_SIZE, replace=False)
        tournament_fitness = [(i, fitness_scores[i][0]) for i in tournament_indices]

        # Sort by fitness (descending)
        tournament_fitness.sort(key=lambda x: x[1], reverse=True)

        return population[tournament_fitness[0][0]].copy()

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Perform uniform crossover between two parents"""
        child = parent1.copy()

        # For each circle, randomly inherit from either parent
        mask = np.random.rand(self.num_circles) > 0.5

        for i in range(self.num_circles):
            if mask[i]:
                child[i] = parent2[i].copy()

        return child

    def mutate(self, individual: np.ndarray, generation: int, max_generations: int) -> np.ndarray:
        """Apply mutation to an individual with adaptive mutation rate"""
        # Adaptive mutation rate: decrease over generations
        adaptive_rate = INITIAL_MUTATION_RATE - (INITIAL_MUTATION_RATE - FINAL_MUTATION_RATE) * \
                       (generation / max_generations)

        mutated = individual.copy()

        for i in range(self.num_circles):
            if np.random.rand() < adaptive_rate:
                # Mutate either center position or radius
                if np.random.rand() < 0.6:  # 60% chance to mutate position
                    # Mutate position with stronger perturbations early, weaker later
                    mutation_strength = 0.1 * (1.0 - generation/max_generations) + 0.02
                    mutated[i, 0] = np.clip(
                        mutated[i, 0] + np.random.uniform(-mutation_strength, mutation_strength),
                        0.001, self.width - 0.001
                    )
                    mutated[i, 1] = np.clip(
                        mutated[i, 1] + np.random.uniform(-mutation_strength, mutation_strength),
                        0.001, self.height - 0.001
                    )
                else:  # 40% chance to mutate radius
                    # Mutate radius with adaptive strength
                    delta = 0.03 * (1.0 - generation/max_generations) + 0.005
                    mutated[i, 2] = np.clip(
                        mutated[i, 2] + np.random.uniform(-delta, delta),
                        0.001, 0.3
                    )

        return mutated

    def local_refinement(self, circles: np.ndarray, max_iterations: int = 50) -> np.ndarray:
        """
        Apply local refinement to improve the solution quality
        This applies fine-grained adjustments to increase radii where possible
        """
        refined_circles = circles.copy()
        
        for iteration in range(max_iterations):
            improved = False
            # Try to increase each circle's radius individually
            for i in range(self.num_circles):
                current_circles = refined_circles.copy()
                old_radius = current_circles[i, 2]
                
                # Try increasing radius by small increments
                for trial_radius in [old_radius + 0.001, old_radius + 0.002, old_radius + 0.005]:
                    if trial_radius > 0.3:
                        continue
                    
                    current_circles[i, 2] = trial_radius
                    
                    # Check constraints
                    valid = True
                    
                    # Check boundary
                    x, y, r = current_circles[i]
                    if not self.is_valid_circle(x, y, r):
                        valid = False
                    
                    if valid:
                        # Check overlaps
                        positions = current_circles[:, :2]
                        distances = cdist(positions, positions)
                        
                        for j in range(self.num_circles):
                            if j != i:
                                if distances[i, j] < (current_circles[i, 2] + current_circles[j, 2]) * 0.99:
                                    valid = False
                                    break
                    
                    if valid:
                        # Accept improvement
                        refined_circles = current_circles.copy()
                        improved = True
                        
            if not improved:
                break
                
        return refined_circles

    def optimize(self) -> np.ndarray:
        """Main optimization loop using evolutionary algorithm"""
        start_time = time.time()

        # Generate initial population
        population = self.generate_initial_population(POPULATION_SIZE)

        best_solution = None
        best_fitness = float('-inf')

        # Track convergence
        fitness_history = []

        for generation in range(MAX_GENERATIONS):
            # Evaluate fitness for entire population
            fitness_scores = []
            for individual in population:
                fitness, violations = self.calculate_fitness(individual)
                fitness_scores.append((fitness, violations))

            # Track best solution in this generation
            gen_best_idx = np.argmax([f[0] for f in fitness_scores])
            gen_best_fitness = fitness_scores[gen_best_idx][0]

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_solution = population[gen_best_idx].copy()

            fitness_history.append(gen_best_fitness)

            # Print progress every 25 generations
            if generation % 25 == 0:
                print(f"Generation {generation}: Best fitness = {gen_best_fitness:.6f}")

            # Create new population through selection, crossover, and mutation
            new_population = []

            # Elitism: keep best individual
            new_population.append(best_solution.copy())

            # Generate offspring
            while len(new_population) < POPULATION_SIZE:
                # Selection
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)

                # Crossover
                child = self.crossover(parent1, parent2)

                # Mutation with adaptive rate
                child = self.mutate(child, generation, MAX_GENERATIONS)

                new_population.append(child)

            population = new_population

            # Early stopping if converged
            if len(fitness_history) >= 15:
                recent_avg = np.mean(fitness_history[-15:])
                prev_avg = np.mean(fitness_history[-30:-15])
                if abs(recent_avg - prev_avg) < 1e-5:
                    print(f"Converged at generation {generation}")
                    break

        # Apply local refinement to the best solution
        if best_solution is not None:
            best_solution = self.local_refinement(best_solution)

        end_time = time.time()
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")
        print(f"Best fitness achieved: {best_fitness:.6f}")

        return best_solution

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Create packer instance with rectangle dimensions
    packer = CirclePacker(width=1.0, height=1.0, num_circles=21)

    # Run optimization
    circles = packer.optimize()

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")