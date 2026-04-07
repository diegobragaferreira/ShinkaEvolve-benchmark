# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List, Optional
import time
import warnings
import math
from dataclasses import dataclass

# Global constants
RECT_PERIMETER = 4.0
RECT_WIDTH = 1.0  # Default rectangle dimensions (width=1, height=1)
RECT_HEIGHT = 1.0
NUM_CIRCLES = 21
POPULATION_SIZE = 100
MAX_GENERATIONS = 150
MUTATION_RATE = 0.05
TOURNAMENT_SIZE = 3
SEED = 42
# Strategic aspect ratio sampling around most promising values
ASPECT_RATIOS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.8, 2.0, 2.5, 3.0]

@dataclass
class CircleConfig:
    """Data class for storing circle configuration."""
    x: float
    y: float
    r: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.r])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'CircleConfig':
        return cls(arr[0], arr[1], arr[2])

class ConstraintValidator:
    """Validates circle configurations against geometric constraints."""
    
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def is_valid_circle(self, x: float, y: float, r: float) -> bool:
        """Check if circle is valid (within bounds and positive radius)"""
        return (0 < r and 
                r <= x <= self.width - r and 
                r <= y <= self.height - r)

    def validate_all_constraints(self, circles: np.ndarray) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Validate all constraints and return violation count and indices
        
        Returns:
            Tuple of (violation_count, list_of_violating_pairs)
        """
        violations = 0
        violating_pairs = []
        
        # Vectorized boundary check
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Check boundary violations in batch
        valid_positions = (radii <= positions[:, 0]) & \
                         (positions[:, 0] <= self.width - radii) & \
                         (radii <= positions[:, 1]) & \
                         (positions[:, 1] <= self.height - radii)
        boundary_violations = np.sum(~valid_positions)
        violations += boundary_violations * 100

        # Check overlap violations using spatial indexing for efficiency
        try:
            # Build KDTree for fast neighbor search
            points = circles[:, :2]  # Only x,y coordinates
            tree = cKDTree(points)

            # Find neighbors within 2*max_radius distance (optimization)
            max_radius = np.max(circles[:, 2]) if len(circles) > 0 else 0
            if max_radius > 0:
                # Query pairs with distance threshold (more efficient than querying all pairs)
                pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')
                
                # Process pairs to identify true overlaps
                for i, j in pairs:
                    if i < j:  # Ensure we only check each pair once
                        x1, y1, r1 = circles[i]
                        x2, y2, r2 = circles[j]
                        
                        # Calculate squared distance to avoid sqrt computation
                        dx = x1 - x2
                        dy = y1 - y2
                        dist_sq = dx*dx + dy*dy
                        radius_sum = r1 + r2
                        
                        if dist_sq < radius_sum * radius_sum:
                            violations += 1
                            violating_pairs.append((i, j))

        except Exception as e:
            warnings.warn(f"Error in overlap checking: {e}")
            # Fallback to brute force when spatial indexing fails
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    
                    # Calculate squared distance to avoid sqrt computation
                    dx = x1 - x2
                    dy = y1 - y2
                    dist_sq = dx*dx + dy*dy
                    radius_sum = r1 + r2
                    
                    if dist_sq < radius_sum * radius_sum:
                        violations += 1
                        violating_pairs.append((i, j))

        return violations, violating_pairs

class FitnessEvaluator:
    """Evaluates fitness of circle configurations."""
    
    def __init__(self, validator: ConstraintValidator, penalty_weight: float = 1000.0):
        self.validator = validator
        self.penalty_weight = penalty_weight

    def evaluate(self, circles: np.ndarray) -> Tuple[float, int]:
        """
        Calculate fitness: sum of radii with penalty for constraint violations
        
        Returns:
            Tuple of (fitness_score, number_of_violations)
        """
        total_radius = np.sum(circles[:, 2])

        # Validate constraints
        violations, _ = self.validator.validate_all_constraints(circles)

        # Return fitness score (higher is better)
        # Enhanced penalty weight that scales with solution quality and violations
        dynamic_penalty = self.penalty_weight * (1.0 + 0.2 * np.log(total_radius + 1) + 0.1 * violations)
        return total_radius - dynamic_penalty, violations

class Initializer:
    """Generates initial configurations for circle packing."""
    
    def __init__(self, width: float, height: float, num_circles: int):
        self.width = width
        self.height = height
        self.num_circles = num_circles

    def generate_adaptive_grid(self) -> np.ndarray:
        """Create initial configuration using enhanced adaptive grid placement."""
        circles = np.zeros((self.num_circles, 3))

        # Use mathematical approach for grid spacing based on rectangle area
        rect_area = self.width * self.height
        if rect_area > 0:
            ideal_spacing = np.sqrt(rect_area / self.num_circles)
        else:
            ideal_spacing = min(self.width, self.height) / 5.0

        # Determine grid dimensions that optimize space utilization
        # This ensures better distribution across rectangle
        grid_width = max(1, int(np.ceil(self.width / ideal_spacing)))
        grid_height = max(1, int(np.ceil(self.height / ideal_spacing)))
        
        # Adjust to ensure sufficient space for all circles
        while grid_width * grid_height < self.num_circles:
            if self.width > self.height:
                grid_width += 1
            else:
                grid_height += 1

        # Calculate actual cell dimensions
        cell_width = self.width / grid_width if grid_width > 0 else self.width
        cell_height = self.height / grid_height if grid_height > 0 else self.height

        # Calculate base radius with more careful scaling
        base_radius = min(cell_width, cell_height) * 0.32

        # Place circles systematically with strategic randomization
        circle_idx = 0
        for i in range(grid_height):
            for j in range(grid_width):
                if circle_idx >= self.num_circles:
                    break
                    
                # Position with appropriate margin and controlled randomization
                margin_x = cell_width * 0.05  # 5% margin
                margin_y = cell_height * 0.05
                
                x = margin_x + (j + 0.5) * cell_width + np.random.uniform(-margin_x * 0.2, margin_x * 0.2)
                y = margin_y + (i + 0.5) * cell_height + np.random.uniform(-margin_y * 0.2, margin_y * 0.2)
                
                # Radius with controlled variation
                r = base_radius * (0.85 + np.random.uniform(0, 0.3))
                
                # Ensure circle fits within bounds
                x = np.clip(x, r, self.width - r)
                y = np.clip(y, r, self.height - r)
                
                circles[circle_idx] = [x, y, r]
                circle_idx += 1
                
            if circle_idx >= self.num_circles:
                break

        # Ensure minimum radius
        for i in range(self.num_circles):
            if circles[i, 2] < 0.001:
                circles[i, 2] = 0.001

        return circles

class EvolutionaryOptimizer:
    """Performs evolutionary optimization of circle configurations."""
    
    def __init__(self,
                 width: float,
                 height: float,
                 num_circles: int,
                 population_size: int = POPULATION_SIZE,
                 max_generations: int = MAX_GENERATIONS,
                 mutation_rate: float = MUTATION_RATE):
        self.width = width
        self.height = height
        self.num_circles = num_circles
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate

        # Initialize components
        self.validator = ConstraintValidator(width, height)
        self.evaluator = FitnessEvaluator(self.validator)
        self.initializer = Initializer(width, height, num_circles)

        # Set random seed for reproducibility
        np.random.seed(SEED)
        random.seed(SEED)

    def tournament_selection(self, population: List[np.ndarray],
                           fitness_scores: List[Tuple[float, int]]) -> np.ndarray:
        """Select individual using tournament selection"""
        tournament_indices = np.random.choice(len(population), TOURNAMENT_SIZE)
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

    def mutate(self, individual: np.ndarray, generation: int = 0) -> np.ndarray:
        """Apply mutation to an individual with adaptive parameters"""
        mutated = individual.copy()

        # Base mutation rate with adaptive scaling
        current_radius_sum = np.sum(individual[:, 2])
        # Dynamic mutation rate that decreases over generations
        adaptive_mutation_rate = self.mutation_rate * (1.0 - 0.002 * generation)
        adaptive_mutation_rate = max(0.01, adaptive_mutation_rate)  # Minimum mutation rate

        for i in range(self.num_circles):
            if np.random.rand() < adaptive_mutation_rate:
                # Mutate either center position or radius with preference for position
                if np.random.rand() < 0.75:  # 75% chance of position mutation
                    # Mutate position with adaptive step size
                    step_x = 0.02 * (1.0 + 0.005 * current_radius_sum)
                    step_y = 0.02 * (1.0 + 0.005 * current_radius_sum)
                    mutated[i, 0] = np.clip(
                        mutated[i, 0] + np.random.normal(0, step_x),
                        mutated[i, 2], self.width - mutated[i, 2]
                    )
                    mutated[i, 1] = np.clip(
                        mutated[i, 1] + np.random.normal(0, step_y),
                        mutated[i, 2], self.height - mutated[i, 2]
                    )
                else:
                    # Mutate radius with adaptive range
                    step_r = 0.01 * (1.0 + 0.003 * current_radius_sum)
                    mutated[i, 2] = np.clip(
                        mutated[i, 2] + np.random.normal(0, step_r),
                        0.001, 0.2
                    )

        return mutated

    def local_refinement(self, circles: np.ndarray, max_iterations: int = 100) -> np.ndarray:
        """Apply local refinement to improve solution quality"""
        refined = circles.copy()
        
        # Iteratively try to increase radii while respecting constraints
        for iter_num in range(max_iterations):
            improved = False
            
            # Try to increase each circle's radius in a shuffled order for better exploration
            indices = list(range(self.num_circles))
            np.random.shuffle(indices)
            
            for i in indices:
                original_circle = refined[i].copy()
                x, y, r = original_circle
                
                # Try to increase radius
                new_r = min(r * 1.012, 0.2)  # Even smaller increase factor
                test_circle = [x, y, new_r]
                
                # Check if it's still valid (reusing validator)
                if not self.validator.is_valid_circle(x, y, new_r):
                    continue
                    
                # Check if it causes overlaps using efficient method
                overlap_found = False
                # Only check against other circles that might overlap
                temp_circles = np.vstack([refined[:i], test_circle, refined[i+1:]])
                violations, _ = self.validator.validate_all_constraints(temp_circles)
                if violations > 0:
                    overlap_found = True
                
                if not overlap_found:
                    refined[i] = test_circle
                    improved = True
            
            # If no improvements were made, stop early
            if not improved:
                break
                
        return refined

    def optimize_phase_1(self, population: List[np.ndarray]) -> Tuple[np.ndarray, float]:
        """First phase: Coarse evolution with relaxed constraints"""
        best_solution = None
        best_fitness = float('-inf')
        fitness_history = []

        for generation in range(self.max_generations // 2):  # Half the generations for phase 1
            # Evaluate fitness for entire population
            fitness_scores = []
            for individual in population:
                fitness, violations = self.evaluator.evaluate(individual)
                fitness_scores.append((fitness, violations))

            # Track best solution in this generation
            gen_best_idx = np.argmax([f[0] for f in fitness_scores])
            gen_best_fitness = fitness_scores[gen_best_idx][0]

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_solution = population[gen_best_idx].copy()

            fitness_history.append(gen_best_fitness)

            # Create new population through selection, crossover, and mutation
            new_population = []

            # Elitism: keep best individual
            new_population.append(best_solution.copy())

            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)

                # Crossover
                child = self.crossover(parent1, parent2)

                # Mutation
                child = self.mutate(child, generation)

                new_population.append(child)

            population = new_population

        return best_solution, best_fitness

    def optimize_phase_2(self, best_solution: np.ndarray) -> np.ndarray:
        """Second phase: Fine-grained refinement with stricter constraints"""
        # Refine the best solution from phase 1
        refined_solution = self.local_refinement(best_solution.copy(), 200)
        
        # Additional refinement through focused optimization
        final_solution = refined_solution.copy()
        
        # Try iterative improvement with more targeted approach
        for i in range(100):  # More rounds of focused updates
            # Try to improve individual circles
            improved = False
            # Shuffle order for better exploration
            indices = list(range(self.num_circles))
            np.random.shuffle(indices)
            
            for j in indices:
                original_circle = final_solution[j].copy()
                x, y, r = original_circle
                
                # Try to slightly increase radius
                new_r = min(r * 1.008, 0.2)
                test_circle = [x, y, new_r]
                
                # Check validity
                if not self.validator.is_valid_circle(x, y, new_r):
                    continue
                    
                # Check overlap with others using efficient method
                temp_circles = np.vstack([final_solution[:j], test_circle, final_solution[j+1:]])
                violations, _ = self.validator.validate_all_constraints(temp_circles)
                if violations == 0:
                    final_solution[j] = test_circle
                    improved = True
                    
            if not improved:
                break

        return final_solution

    def optimize(self) -> np.ndarray:
        """Main optimization loop using evolutionary algorithm with two phases"""
        start_time = time.time()

        print("Starting Phase 1: Coarse evolution...")
        
        # Generate initial population
        population = []
        for _ in range(self.population_size):
            circles = self.initializer.generate_adaptive_grid()
            population.append(circles)

        # Phase 1: Coarse evolution
        best_solution, best_fitness = self.optimize_phase_1(population)
        
        # Phase 2: Fine-grained refinement
        print("Starting Phase 2: Fine-tuning...")
        final_solution = self.optimize_phase_2(best_solution)

        end_time = time.time()
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")
        print(f"Final fitness achieved: {self.evaluator.evaluate(final_solution)[0]:.6f}")

        return final_solution

class CirclePacker:
    """Main circle packing class orchestrating the optimization process."""
    
    def __init__(self, width: float = RECT_WIDTH, height: float = RECT_HEIGHT,
                 num_circles: int = NUM_CIRCLES):
        self.width = width
        self.height = height
        self.num_circles = num_circles

    def optimize_with_multiple_aspect_ratios(self) -> np.ndarray:
        """Try multiple aspect ratios and return the best solution."""
        best_solution = None
        best_radius_sum = -float('inf')

        print(f"Testing {len(ASPECT_RATIOS)} different aspect ratios...")

        for i, aspect_ratio in enumerate(ASPECT_RATIOS):
            # Calculate width and height based on perimeter constraint
            # width + height = 2, so if width/height = aspect_ratio, then:
            # width = aspect_ratio * height, and width + height = 2
            # Therefore: aspect_ratio * height + height = 2 => height = 2 / (1 + aspect_ratio)
            height = 2.0 / (1.0 + aspect_ratio)
            width = aspect_ratio * height

            print(f"Testing aspect ratio {aspect_ratio}: width={width:.3f}, height={height:.3f}")

            # Create optimizer with current dimensions
            optimizer = EvolutionaryOptimizer(width, height, self.num_circles)

            # Run optimization
            solution = optimizer.optimize()

            # Check if this solution is better
            radius_sum = np.sum(solution[:, 2])
            if radius_sum > best_radius_sum:
                best_radius_sum = radius_sum
                best_solution = solution.copy()
                print(f"New best found with aspect ratio {aspect_ratio}: radius sum = {radius_sum:.6f}")

        return best_solution

    def optimize(self) -> np.ndarray:
        """Run the optimization process and return the best solution."""
        return self.optimize_with_multiple_aspect_ratios()

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