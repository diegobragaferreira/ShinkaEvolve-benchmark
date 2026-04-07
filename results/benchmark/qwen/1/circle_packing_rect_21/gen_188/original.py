# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

class CirclePackingOptimizer:
    def __init__(self, rect_width: float = 1.0, rect_height: float = 1.0, n_circles: int = 21):
        self.rect_width = rect_width
        self.rect_height = rect_height
        self.n_circles = n_circles
        self.population_size = 50
        self.generations = 100
        self.elite_size = 8
        self.tournament_size = 7

    def check_constraints(self, circles: np.ndarray) -> bool:
        """Efficiently check if all circles satisfy the constraints with early termination."""
        # Check boundary constraints first
        for i in range(self.n_circles):
            x, y, r = circles[i]
            if x - r < 0 or x + r > self.rect_width or y - r < 0 or y + r > self.rect_height:
                return False

        # Check overlap constraints efficiently using vectorized operations
        if self.n_circles > 1:
            positions = circles[:, :2]
            radii = circles[:, 2]

            # Compute distance matrix (only upper triangle for efficiency)
            distances = cdist(positions, positions)
            
            # Use upper triangle indices to avoid double counting
            mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
            
            # Get overlap distances and required distances
            overlap_distances = distances[mask]
            overlap_radii = (radii[:, np.newaxis] + radii[np.newaxis, :])[mask]
            
            # Check for any overlaps efficiently
            if np.any(overlap_distances < overlap_radii):
                return False

        return True

    def evaluate_fitness(self, circles: np.ndarray) -> float:
        """Evaluate fitness as the sum of radii with constraint validation."""
        if not self.check_constraints(circles):
            return -np.inf

        return np.sum(circles[:, 2])

    def create_hexagonal_initial_solution(self) -> np.ndarray:
        """Create initial solution using hexagonal lattice pattern for better packing efficiency."""
        circles = np.zeros((self.n_circles, 3))

        # Determine grid dimensions based on circular arrangement
        rows = int(np.ceil(np.sqrt(self.n_circles)))
        cols = int(np.ceil(self.n_circles / rows))

        # Adjust for rectangular container
        grid_width = self.rect_width * 0.95
        grid_height = self.rect_height * 0.95

        # Calculate spacing based on available space
        cell_width = grid_width / cols
        cell_height = grid_height / rows
        min_cell_dim = min(cell_width, cell_height)

        # Hexagon radius (circles should fit comfortably)
        hex_radius = min_cell_dim * 0.45

        # Arrange in hexagonal pattern
        placed = 0
        for row in range(rows):
            if placed >= self.n_circles:
                break
            for col in range(cols):
                if placed >= self.n_circles:
                    break

                # Offset every other row for hexagonal pattern
                offset = (row % 2) * (cell_width / 2)
                x = offset + col * cell_width + cell_width / 2
                y = row * cell_height + cell_height / 2

                # Ensure we're within bounds (with padding)
                x = np.clip(x, hex_radius, self.rect_width - hex_radius)
                y = np.clip(y, hex_radius, self.rect_height - hex_radius)

                # Adjust radius to prevent boundary issues
                max_radius = min(x, y, self.rect_width - x, self.rect_height - y)
                r = min(hex_radius, max_radius * 0.9)

                circles[placed] = [x, y, r]
                placed += 1

        # Fill remaining positions with small random circles
        for i in range(placed, self.n_circles):
            # Place remaining circles randomly but within bounds
            x = np.random.uniform(hex_radius, self.rect_width - hex_radius)
            y = np.random.uniform(hex_radius, self.rect_height - hex_radius)
            r = np.random.uniform(0.001, hex_radius * 0.3)
            circles[i] = [x, y, r]

        return circles

    def create_random_solution(self) -> np.ndarray:
        """Create a random valid solution using greedy placement."""
        circles = np.zeros((self.n_circles, 3))

        # Try to place circles greedily with better spatial distribution
        placed = 0
        max_attempts = 10000

        # Pre-allocate arrays for efficiency
        attempts = 0
        while placed < self.n_circles and attempts < max_attempts:
            attempts += 1
            
            # Random position and radius
            x = np.random.uniform(0, self.rect_width)
            y = np.random.uniform(0, self.rect_height)
            r = np.random.uniform(0.001, 0.1)

            new_circle = np.array([x, y, r])

            # Check if it overlaps with existing circles efficiently
            valid_placement = True
            
            # Quick bounds check first
            if x - r < 0 or x + r > self.rect_width or y - r < 0 or y + r > self.rect_height:
                valid_placement = False
            else:
                # Vectorized overlap check with existing circles
                if placed > 0:
                    existing_positions = circles[:placed, :2]
                    existing_radii = circles[:placed, 2]
                    
                    # Compute distances efficiently
                    distances_squared = np.sum((existing_positions - [x, y])**2, axis=1)
                    overlap_radii = existing_radii + r
                    
                    # Check for any overlaps
                    overlaps = distances_squared < (overlap_radii ** 2)
                    if np.any(overlaps):
                        valid_placement = False

            if valid_placement:
                circles[placed] = new_circle
                placed += 1

        # If we couldn't place all circles, fill remaining with minimal valid ones
        for i in range(placed, self.n_circles):
            x = np.random.uniform(0.001, self.rect_width - 0.001)
            y = np.random.uniform(0.001, self.rect_height - 0.001)
            r = 0.001
            circles[i] = [x, y, r]

        return circles

    def mutate(self, circles: np.ndarray) -> np.ndarray:
        """Improved mutation operator with adaptive parameters."""
        mutated = circles.copy()

        # Compute constraint characteristics for adaptive mutation
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Adaptive mutation based on solution quality and distribution
        avg_radius = np.mean(radii)
        std_radius = np.std(radii)
        total_radius = np.sum(radii)
        
        # Dynamic mutation rates based on solution characteristics
        base_mutation_rate = 0.3
        density_factor = 1.0
        
        # Adjust mutation rate based on solution quality (higher quality = lower mutation)
        if total_radius > 1.0:
            density_factor = 0.7
        elif total_radius > 0.5:
            density_factor = 0.9
            
        adjusted_mutation_rate = base_mutation_rate * density_factor

        for i in range(self.n_circles):
            if np.random.random() < adjusted_mutation_rate:
                # Choose mutation type with preference for position
                mutation_type = np.random.choice(['position', 'radius'], p=[0.75, 0.25])

                if mutation_type == 'position':
                    # Mutate position with adaptive step size
                    step_size = np.random.uniform(0.01, 0.1) * (0.5 + 0.5 * np.random.random())
                    mutated[i, 0] += np.random.normal(0, step_size)
                    mutated[i, 1] += np.random.normal(0, step_size)
                else:
                    # Mutate radius with log-normal distribution
                    scale_factor = np.exp(np.random.normal(0, 0.1))
                    mutated[i, 2] *= scale_factor
                    mutated[i, 2] = max(0.001, mutated[i, 2])

        return mutated

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Improved crossover operator with better recombination."""
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Uniform crossover with better blending
        for i in range(self.n_circles):
            if np.random.random() < 0.5:
                child1[i] = parent2[i]
                child2[i] = parent1[i]
            else:
                # Blend positions and radii with better control
                blend_factor = 0.3 + 0.4 * np.random.random()
                child1[i, :2] = parent1[i, :2] * (1 - blend_factor) + parent2[i, :2] * blend_factor
                child2[i, :2] = parent1[i, :2] * blend_factor + parent2[i, :2] * (1 - blend_factor)
                
                child1[i, 2] = parent1[i, 2] * (1 - blend_factor) + parent2[i, 2] * blend_factor
                child2[i, 2] = parent1[i, 2] * blend_factor + parent2[i, 2] * (1 - blend_factor)
                
                # Ensure radii remain positive
                child1[i, 2] = max(0.001, child1[i, 2])
                child2[i, 2] = max(0.001, child2[i, 2])

        return child1, child2

    def repair_solution(self, circles: np.ndarray) -> np.ndarray:
        """Enhanced repair mechanism for fixing constraint violations."""
        repaired = circles.copy()

        # Ensure positive radii
        repaired[:, 2] = np.maximum(repaired[:, 2], 0.001)

        # Enforce bounds
        for i in range(len(repaired)):
            x, y, r = repaired[i]
            x = np.clip(x, r, self.rect_width - r)
            y = np.clip(y, r, self.rect_height - r)
            repaired[i] = [x, y, r]

        # Resolve overlaps iteratively with early exit conditions
        max_iter = 30
        for iteration in range(max_iter):
            # Calculate pairwise distances efficiently
            positions = repaired[:, :2]
            radii = repaired[:, 2]
            
            # Use vectorized approach for overlap detection
            distances = cdist(positions, positions)
            mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
            
            # Find overlapping pairs
            overlaps = distances[mask] < (radii[:, np.newaxis] + radii[np.newaxis, :])[mask]
            
            if not np.any(overlaps):
                break

            # Handle overlapping pairs with geometric correction
            overlap_indices = np.where(mask & (distances < (radii[:, np.newaxis] + radii[np.newaxis, :])))
            
            for i, j in zip(overlap_indices[0], overlap_indices[1]):
                x1, y1, r1 = repaired[i]
                x2, y2, r2 = repaired[j]

                dx = x2 - x1
                dy = y2 - y1
                distance = np.sqrt(dx*dx + dy*dy)

                if distance > 0:
                    # Move circles apart with geometric correction
                    move_distance = (r1 + r2 - distance) * 0.5
                    
                    # Apply movement with direction
                    repaired[i, 0] -= dx / distance * move_distance * 0.5
                    repaired[i, 1] -= dy / distance * move_distance * 0.5
                    repaired[j, 0] += dx / distance * move_distance * 0.5
                    repaired[j, 1] += dy / distance * move_distance * 0.5

            # Enforce bounds again after movement
            for i in range(len(repaired)):
                x, y, r = repaired[i]
                x = np.clip(x, r, self.rect_width - r)
                y = np.clip(y, r, self.rect_height - r)
                repaired[i] = [x, y, r]

        return repaired

    def evolve(self) -> np.ndarray:
        """Main evolutionary algorithm loop."""
        # Initialize population with diverse starting solutions
        population = []
        
        # Start with hexagonal initialization (better packing)
        for _ in range(self.population_size // 3):
            solution = self.create_hexagonal_initial_solution()
            population.append(solution)

        # Add random patterns
        for _ in range(self.population_size // 3):
            solution = self.create_random_solution()
            population.append(solution)

        # Fill remaining with slightly perturbed hexagonal solutions
        for _ in range(self.population_size - 2 * (self.population_size // 3)):
            solution = self.create_hexagonal_initial_solution()
            # Add slight random perturbations to get variety
            for i in range(self.n_circles):
                if np.random.random() < 0.3:
                    solution[i, 0] += np.random.uniform(-0.05, 0.05)
                    solution[i, 1] += np.random.uniform(-0.05, 0.05)
            population.append(solution)

        # Track best fitness for convergence detection
        previous_best = -np.inf
        stagnation_count = 0

        # Evolutionary algorithm
        start_time = time.time()
        for generation in range(self.generations):
            # Check time limit
            if time.time() - start_time > 55:  # Leave 5 seconds for cleanup
                break
                
            # Evaluate fitness
            fitness_scores = [self.evaluate_fitness(individual) for individual in population]

            # Sort by fitness (descending)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = [population[i] for i in sorted_indices]
            fitness_scores = [fitness_scores[i] for i in sorted_indices]

            # Keep elite
            elite = population[:self.elite_size]

            # Generate new population
            new_population = elite[:]

            # Create offspring using tournament selection and crossover
            while len(new_population) < self.population_size:
                # Tournament selection - select two parents
                parent1_idx = sorted_indices[np.random.choice(min(self.tournament_size, len(sorted_indices)))]
                parent2_idx = sorted_indices[np.random.choice(min(self.tournament_size, len(sorted_indices)))]

                parent1 = population[parent1_idx].copy()
                parent2 = population[parent2_idx].copy()

                # Crossover
                child1, child2 = self.crossover(parent1, parent2)

                # Mutate
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                # Repair
                child1 = self.repair_solution(child1)
                child2 = self.repair_solution(child2)

                new_population.extend([child1, child2])

            population = new_population[:self.population_size]

            # Convergence detection
            current_best = max(fitness_scores)
            if abs(current_best - previous_best) < 1e-6:
                stagnation_count += 1
            else:
                stagnation_count = 0
            previous_best = current_best

            # Early stopping if stagnated too long
            if stagnation_count > 15:
                print(f"Early stopping at generation {generation} due to convergence")
                break

            # Print progress
            if generation % 20 == 0:
                print(f"Generation {generation}: Best fitness = {current_best:.6f}")

        # Return the best solution
        fitness_scores = [self.evaluate_fitness(individual) for individual in population]
        best_idx = np.argmax(fitness_scores)
        best_solution = population[best_idx]

        # Final validation and minor improvement
        final_fitness = self.evaluate_fitness(best_solution)
        if final_fitness == -np.inf:
            print("Warning: Final solution violated constraints. Returning fallback.")
            # Fallback to best valid solution found during evolution
            for i in range(len(population)):
                if self.evaluate_fitness(population[i]) > -np.inf:
                    return population[i]

        # Final fine-tuning with local optimization
        final_solution = self.local_optimize(best_solution)
        return final_solution

    def local_optimize(self, circles: np.ndarray) -> np.ndarray:
        """Apply local optimization to improve final solution."""
        # Simple gradient descent style optimization
        optimized = circles.copy()
        
        # Run a few steps of local optimization
        for _ in range(20):
            # Calculate forces between circles
            forces = np.zeros_like(optimized)
            
            # Compute all pairwise center differences and distances
            positions = optimized[:, :2]
            radii = optimized[:, 2]
            
            # Vectorized computation
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff**2, axis=2))
            
            # Create mask for non-diagonal elements
            mask = ~np.eye(len(positions), dtype=bool)
            
            # Compute repulsion forces only for overlapping pairs
            overlap_mask = (distances < (radii[:, np.newaxis] + radii[np.newaxis, :])) & mask
            
            if np.any(overlap_mask):
                overlap_indices = np.where(overlap_mask)
                for i, j in zip(overlap_indices[0], overlap_indices[1]):
                    dx = diff[i, j, 0]
                    dy = diff[i, j, 1]
                    dist = distances[i, j]
                    
                    if dist > 0:
                        force_magnitude = 1.0 / (dist * dist + 1e-8)
                        forces[i, 0] -= force_magnitude * dx / dist
                        forces[i, 1] -= force_magnitude * dy / dist
                        
                        forces[j, 0] += force_magnitude * dx / dist
                        forces[j, 1] += force_magnitude * dy / dist
            
            # Apply forces with small step size
            learning_rate = 0.01
            for i in range(len(optimized)):
                x, y, r = optimized[i]
                new_x = x + learning_rate * forces[i, 0]
                new_y = y + learning_rate * forces[i, 1]
                
                # Keep within bounds
                new_x = np.clip(new_x, r, self.rect_width - r)
                new_y = np.clip(new_y, r, self.rect_height - r)
                
                optimized[i, 0] = new_x
                optimized[i, 1] = new_y
        
        return optimized

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    # Optimized rectangle dimensions - using a wider rectangle for better packing
    rect_width = 1.5
    rect_height = 0.5

    # Create optimizer instance
    optimizer = CirclePackingOptimizer(rect_width, rect_height, 21)

    # Run evolution
    circles = optimizer.evolve()

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")