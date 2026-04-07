# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List, Optional
import time
import math

# Fixed seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Optimal rectangle dimensions determined through analysis
RECT_WIDTH = 1.33
RECT_HEIGHT = 0.67

class AdaptiveGridCirclePacker:
    def __init__(self, width: float = RECT_WIDTH, height: float = RECT_HEIGHT, num_circles: int = 21):
        self.width = width
        self.height = height
        self.num_circles = num_circles
        self.rect_area = width * height

    def initialize_adaptive_grid(self) -> np.ndarray:
        """Initialize circles using an adaptive grid approach that considers aspect ratio and circle count"""
        circles = np.zeros((self.num_circles, 3))

        # Determine optimal grid density based on area and circle count
        # This considers both the total area and the expected packing efficiency
        target_density = 0.7  # Expected packing density for circle packing
        target_area_per_circle = self.rect_area / self.num_circles * (1/target_density)
        estimated_radius = math.sqrt(target_area_per_circle / math.pi)

        # Determine grid dimensions based on aspect ratio and estimated circle size
        aspect_ratio = self.width / self.height

        # Use a more sophisticated approach to grid sizing
        sqrt_n = math.sqrt(self.num_circles)
        cols = max(1, int(math.ceil(sqrt_n * math.sqrt(aspect_ratio))))
        rows = max(1, int(math.ceil(self.num_circles / cols)))

        # Adjust grid size to ensure adequate space for estimated circles
        # This helps avoid overly dense initial placements that are hard to improve
        if cols * rows < self.num_circles:
            cols = max(cols, int(math.ceil(self.num_circles / rows)))

        cell_width = self.width / cols
        cell_height = self.height / rows

        # Create grid with proper spacing and slight randomness
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= self.num_circles:
                    break

                # Position with adaptive randomness based on cell size
                x = (j + 0.5) * cell_width + random.uniform(-cell_width*0.2, cell_width*0.2)
                y = (i + 0.5) * cell_height + random.uniform(-cell_height*0.2, cell_height*0.2)

                # Keep within bounds with padding
                x = max(0.01, min(self.width - 0.01, x))
                y = max(0.01, min(self.height - 0.01, y))

                # Compute maximum possible radius for this position
                max_radius = min(x, self.width - x, y, self.height - y)
                # Use a more conservative initial radius to allow for better optimization
                initial_radius = max(0.01, min(max_radius * 0.25, estimated_radius * 0.8))

                circles[idx] = [x, y, initial_radius]
                idx += 1

        return circles

    def is_valid_position(self, x: float, y: float, r: float) -> bool:
        """Check if circle position is within bounds"""
        return (r <= x <= self.width - r and r <= y <= self.height - r)

    def calculate_overlap_penalty(self, circles: np.ndarray) -> float:
        """Calculate penalty for overlaps using spatial indexing for efficiency"""
        penalty = 0.0

        # Use spatial indexing for efficient neighbor queries
        try:
            points = circles[:, :2]
            tree = cKDTree(points)

            # Find pairs that might overlap based on maximum radius
            max_radius = np.max(circles[:, 2]) if len(circles) > 0 else 0.0
            pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')

            for i, j in pairs:
                if i < j:  # Avoid double counting
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]

                    # Calculate actual distance
                    dx = x1 - x2
                    dy = y1 - y2
                    distance_sq = dx*dx + dy*dy
                    radius_sum = r1 + r2

                    # If circles overlap
                    if distance_sq < radius_sum * radius_sum:
                        overlap = radius_sum - math.sqrt(distance_sq)
                        penalty += overlap * 1000.0

        except Exception:
            # Fallback to brute force if spatial indexing fails
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]

                    dx = x1 - x2
                    dy = y1 - y2
                    distance_sq = dx*dx + dy*dy
                    radius_sum = r1 + r2

                    if distance_sq < radius_sum * radius_sum:
                        overlap = radius_sum - math.sqrt(distance_sq)
                        penalty += overlap * 1000.0

        return penalty

    def calculate_boundary_penalty(self, circles: np.ndarray) -> float:
        """Calculate penalty for boundary violations"""
        penalty = 0.0

        for i in range(len(circles)):
            x, y, r = circles[i]
            # Calculate how much we're violating the boundary
            if x - r < 0:
                penalty += (0 - (x - r)) * 5000.0
            if x + r > self.width:
                penalty += ((x + r) - self.width) * 5000.0
            if y - r < 0:
                penalty += (0 - (y - r)) * 5000.0
            if y + r > self.height:
                penalty += ((y + r) - self.height) * 5000.0

        return penalty

    def calculate_fitness(self, circles: np.ndarray) -> Tuple[float, float]:
        """Calculate fitness function including both radius sum and penalties"""
        radius_sum = np.sum(circles[:, 2])

        # Calculate penalties
        overlap_penalty = self.calculate_overlap_penalty(circles)
        boundary_penalty = self.calculate_boundary_penalty(circles)

        # Combined fitness (higher is better)
        fitness = radius_sum - overlap_penalty - boundary_penalty

        return fitness, overlap_penalty + boundary_penalty

    def local_optimize_single_circle(self, circles: np.ndarray, idx: int,
                                   max_attempts: int = 100) -> bool:
        """Optimize a single circle's position and radius using systematic approach"""
        old_x, old_y, old_r = circles[idx]
        best_x, best_y, best_r = old_x, old_y, old_r
        best_fitness = float('-inf')

        # Get current fitness for comparison
        current_fitness, _ = self.calculate_fitness(circles)

        # Systematic exploration of radius adjustments
        radius_modifications = []
        if old_r < 0.3:  # Cap maximum radius
            # Try several radius adjustments
            radius_factors = [1.0, 1.05, 1.1, 1.15, 1.2, 0.95, 0.9, 0.85, 0.8]
            for factor in radius_factors:
                new_r = min(0.3, old_r * factor)
                if new_r > 0.001:
                    radius_modifications.append(('radius', new_r, old_x, old_y))

        # Systematic exploration of position adjustments
        position_modifications = []

        # Try different movement distances and directions
        move_distances = [0.01, 0.02, 0.03, 0.05]
        directions = [(0, 0), (0.02, 0), (-0.02, 0), (0, 0.02), (0, -0.02),
                     (0.01, 0.01), (0.01, -0.01), (-0.01, 0.01), (-0.01, -0.01)]

        # Generate position modifications with increased variety
        for dist in move_distances:
            for dx, dy in directions:
                new_x = old_x + dx * dist
                new_y = old_y + dy * dist
                # Keep within bounds
                new_x = max(old_r, min(self.width - old_r, new_x))
                new_y = max(old_r, min(self.height - old_r, new_y))
                position_modifications.append(('position', old_r, new_x, new_y))

        # Combine radius and position modifications
        combined_modifications = []
        for r_mod in radius_modifications:
            for pos_mod in position_modifications:
                if random.random() < 0.3:  # Only combine some of the time
                    # Take radius from radius mod, position from position mod
                    _, r_val, x_val, y_val = r_mod
                    combined_modifications.append(('combined', r_val, x_val, y_val))

        # Combine all modifications
        all_modifications = radius_modifications + position_modifications + combined_modifications

        # Test all modifications with a more systematic approach
        for mod_type, new_r, new_x, new_y in all_modifications:
            # Temporarily modify this circle
            circles[idx] = [new_x, new_y, new_r]

            # Check if modification is valid (within bounds)
            if self.is_valid_position(new_x, new_y, new_r):
                # Calculate fitness with this modification
                modified_fitness, _ = self.calculate_fitness(circles)

                # Accept improvement or with some probability
                if modified_fitness > best_fitness or (
                    random.random() < 0.05 and modified_fitness > current_fitness):
                    best_fitness = modified_fitness
                    best_x, best_y, best_r = new_x, new_y, new_r

            # Restore original values
            circles[idx] = [old_x, old_y, old_r]

        # Apply best modification if it improved fitness
        if best_fitness > current_fitness:
            circles[idx] = [best_x, best_y, best_r]
            return True

        return False

    def local_optimize_all(self, circles: np.ndarray, max_iterations: int = 100) -> bool:
        """Perform local optimization on all circles"""
        improved = False

        for iteration in range(max_iterations):
            # Shuffle indices for random order optimization
            indices = list(range(len(circles)))
            random.shuffle(indices)

            iteration_improved = False
            for idx in indices:
                if self.local_optimize_single_circle(circles, idx):
                    iteration_improved = True
                    improved = True

            # Stop early if no improvement
            if not iteration_improved:
                break

        return improved

    def optimize(self, max_generations: int = 1000) -> np.ndarray:
        """Main optimization routine combining initialization, local optimization, and evolutionary refinement"""
        # Phase 1: Adaptive grid initialization
        circles = self.initialize_adaptive_grid()

        # Phase 2: Intensive local optimization
        self.local_optimize_all(circles, max_iterations=200)

        # Phase 3: Evolutionary refinement with hybrid approach
        best_circles = circles.copy()
        best_fitness, _ = self.calculate_fitness(best_circles)

        # Evolutionary parameters
        population_size = 80
        elite_count = 10
        mutation_rate = 0.15

        # Create initial population
        population = [best_circles.copy()]
        for _ in range(population_size - 1):
            # Create diverse variants
            variant = best_circles.copy()
            for i in range(len(variant)):
                if random.random() < 0.3:  # 30% mutation rate
                    # Apply small random perturbations
                    variant[i][0] += random.uniform(-0.03, 0.03)
                    variant[i][1] += random.uniform(-0.03, 0.03)
                    variant[i][2] *= random.uniform(0.9, 1.1)

                    # Keep within bounds
                    variant[i][0] = max(variant[i][2], min(self.width - variant[i][2], variant[i][0]))
                    variant[i][1] = max(variant[i][2], min(self.height - variant[i][2], variant[i][1]))
                    variant[i][2] = max(0.001, min(0.3, variant[i][2]))

            population.append(variant)

        # Evolutionary loop
        for generation in range(max_generations):
            # Evaluate fitness for entire population
            fitness_scores = []
            for individual in population:
                fitness, _ = self.calculate_fitness(individual)
                fitness_scores.append(fitness)

            # Update best solution
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_circles = population[max_fitness_idx].copy()

            # Adaptive mutation rate that decreases with generations
            adaptive_mutation_rate = max(0.05, mutation_rate * (1.0 - generation / max_generations))

            # Selection with tournament - using a more robust approach with fitness scaling
            selected = []
            for _ in range(population_size):
                tournament_size = random.choice([3, 4, 5, 6])  # Variable tournament sizes
                tournament_indices = random.sample(range(len(population)), tournament_size)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                selected.append(population[winner_idx].copy())

            # Elitism - keep best individuals
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            elite = [population[i].copy() for i in elite_indices]

            # Create new population
            new_population = elite.copy()  # Start with elites

            # Fill remaining slots with offspring
            while len(new_population) < population_size:
                # Select parents
                parent1 = selected[random.randint(0, len(selected)-1)]
                parent2 = selected[random.randint(0, len(selected)-1)]

                # Create child through crossover with better strategy
                child = parent1.copy()

                # Use a more effective crossover approach - blend the best traits
                for i in range(len(parent1)):
                    if random.random() < 0.6:  # 60% chance of inheriting from parent2
                        # Blend based on whether parent2 is better
                        if fitness_scores[population.index(parent2)] > fitness_scores[population.index(parent1)]:
                            child[i] = parent2[i].copy()
                        else:
                            # Randomly decide which parent to take from with bias towards better parent
                            if random.random() < 0.7:
                                child[i] = parent2[i].copy()

                # Apply mutation with adaptive rate
                for i in range(len(child)):
                    if random.random() < adaptive_mutation_rate:
                        # Mutate position or radius with more diverse strategies
                        if random.random() < 0.7:  # 70% chance of position mutation
                            # Mutate position with variable magnitudes
                            delta_x = random.uniform(-0.03, 0.03) if generation < max_generations//2 else random.uniform(-0.01, 0.01)
                            delta_y = random.uniform(-0.03, 0.03) if generation < max_generations//2 else random.uniform(-0.01, 0.01)
                            child[i][0] += delta_x
                            child[i][1] += delta_y

                            # Keep within bounds
                            child[i][0] = max(child[i][2], min(self.width - child[i][2], child[i][0]))
                            child[i][1] = max(child[i][2], min(self.height - child[i][2], child[i][1]))
                        else:
                            # Mutate radius with adaptive factors
                            factor = random.uniform(0.8, 1.2) if generation < max_generations//2 else random.uniform(0.95, 1.05)
                            child[i][2] *= factor
                            child[i][2] = max(0.001, min(0.3, child[i][2]))

                # Validate child to ensure it meets constraints
                # If invalid, create a corrected version
                valid_child = self._validate_individual(child)
                new_population.append(valid_child)

            population = new_population[:population_size]

        # Final constraint validation
        best_circles = self._validate_individual(best_circles)

        # Final local optimization
        self.local_optimize_all(best_circles, max_iterations=100)

        return best_circles

    def _validate_individual(self, circles: np.ndarray) -> np.ndarray:
        """Validate and correct a circle configuration to ensure all constraints are met"""
        validated = circles.copy()

        # First pass: correct boundary violations
        for i in range(len(validated)):
            x, y, r = validated[i]
            # Correct boundary violations
            if x - r < 0:
                validated[i, 0] = r
            elif x + r > self.width:
                validated[i, 0] = self.width - r
            if y - r < 0:
                validated[i, 1] = r
            elif y + r > self.height:
                validated[i, 1] = self.height - r

        # Second pass: resolve overlaps through iterative adjustment
        max_iterations = 50
        for iteration in range(max_iterations):
            improved = False

            # Try to reduce radii for overlapping circles
            for i in range(len(validated)):
                x1, y1, r1 = validated[i]
                for j in range(len(validated)):
                    if i != j:
                        x2, y2, r2 = validated[j]
                        dx = x1 - x2
                        dy = y1 - y2
                        distance_sq = dx*dx + dy*dy
                        radius_sum = r1 + r2

                        if distance_sq < radius_sum * radius_sum:
                            # Circles are overlapping, try to reduce radius of circle i
                            overlap = radius_sum - math.sqrt(distance_sq)
                            # Reduce radius by a fraction of the overlap
                            new_r = max(0.001, r1 - overlap * 0.3)
                            if new_r < r1:
                                validated[i, 2] = new_r
                                improved = True

            if not improved:
                break

        return validated

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    packer = AdaptiveGridCirclePacker(width=RECT_WIDTH, height=RECT_HEIGHT, num_circles=21)
    circles = packer.optimize()
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")