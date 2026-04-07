# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

class CirclePackingOptimizer:
    def __init__(self, container_width=1.0, container_height=1.0, n_circles=21):
        self.container_width = container_width
        self.container_height = container_height
        self.n_circles = n_circles
        self.max_iterations = 5000
        self.population_size = 20
        self.elite_size = 4
        self.mutation_rate = 0.15
        self.boundary_strength = 50.0
        self.repulsion_strength = 100.0
        self.radius_adjustment_factor = 0.01
        self.dt = 0.01

    def initialize_population(self):
        """Generate diverse initial population with improved adaptive grid-based initialization"""
        population = []

        # Systematically optimize container dimensions for 21 circles
        # For perimeter = 4, width + height = 2
        best_ratio = 1.0
        best_efficiency = 0.0
        
        # Test ratios from 0.5 to 2.0 with finer resolution
        ratios_to_try = np.linspace(0.5, 2.0, 15)
        
        for ratio in ratios_to_try:
            width = 1.0
            height = width / ratio
            
            if width + height <= 2.0:
                # Estimate packing efficiency more realistically
                container_area = width * height
                avg_circle_area = container_area / self.n_circles
                avg_radius = np.sqrt(avg_circle_area / np.pi)
                
                # Heuristic for packing efficiency based on aspect ratio
                if ratio > 1.2:  # Width much greater than height
                    efficiency = min(1.0, 0.8 * (1.0 + 0.2 / ratio))
                elif ratio < 0.8:  # Height much greater than width
                    efficiency = min(1.0, 0.8 * (1.0 + 0.2 * ratio))
                else:  # Balanced
                    efficiency = 0.9
                
                # Adjust by fill ratio
                estimated_fill = min(1.0, (self.n_circles * avg_radius * avg_radius * np.pi) / container_area)
                efficiency *= estimated_fill
                
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_ratio = ratio

        # Use the best ratio for our main initialization
        container_width = 1.0
        container_height = container_width / best_ratio

        # Calculate optimized grid dimensions using fibonacci-inspired approach
        # Fibonacci-like grid for better packing density
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        grid_cols = max(3, int(np.ceil(np.sqrt(self.n_circles / phi))))
        grid_rows = max(3, int(np.ceil(self.n_circles / grid_cols)))
        
        # Ensure grid fits well within container
        grid_cols = min(grid_cols, 10)
        grid_rows = min(grid_rows, 10)

        for _ in range(self.population_size):
            circles = np.zeros((self.n_circles, 3))

            # Create structured grid with staggered rows (hexagonal packing approximation)
            cell_width = container_width / (grid_cols + 1)
            cell_height = container_height / (grid_rows + 1)

            idx = 0
            for i in range(grid_rows):
                # Stagger every other row for better packing
                offset = (i % 2) * (cell_width / 3)
                for j in range(grid_cols):
                    if idx >= self.n_circles:
                        break
                    x = (j + 1) * cell_width + offset + np.random.uniform(-cell_width*0.15, cell_width*0.15)
                    y = (i + 1) * cell_height + np.random.uniform(-cell_height*0.15, cell_height*0.15)

                    # Ensure within bounds with margin
                    x = np.clip(x, 0.01, container_width - 0.01)
                    y = np.clip(y, 0.01, container_height - 0.01)

                    circles[idx, 0] = x
                    circles[idx, 1] = y
                    idx += 1
                if idx >= self.n_circles:
                    break

            # Assign initial radii with log-normal distribution for better variance
            mean_radius = min(container_width, container_height) * 0.025
            for i in range(self.n_circles):
                # Use log-normal distribution to get varied radii (some larger, some smaller)
                circles[i, 2] = mean_radius * np.exp(np.random.normal(0, 0.4))
            
            # Apply spatial constraint checking to ensure initial feasibility
            if not self.check_constraints(circles):
                # Fallback to random placement with proper bounds
                circles[:, 0] = np.random.uniform(0.01, container_width - 0.01, self.n_circles)
                circles[:, 1] = np.random.uniform(0.01, container_height - 0.01, self.n_circles)
                circles[:, 2] = np.random.uniform(0.005, min(container_width, container_height) * 0.04, self.n_circles)

            # Normalize total radius to reasonable value to maintain balance
            total_radius = np.sum(circles[:, 2])
            if total_radius > 0:
                target_sum = 1.8  # Adjusted target for better performance
                scaling_factor = target_sum / total_radius
                circles[:, 2] *= scaling_factor

            population.append(circles.copy())
        return population

    def check_constraints(self, circles):
        """Check if all circles are within bounds and non-overlapping"""
        positions = circles[:, :2]
        radii = circles[:, 2]

        # Check boundary constraints efficiently
        if np.any(positions[:, 0] - radii < 0) or np.any(positions[:, 0] + radii > self.container_width) or \
           np.any(positions[:, 1] - radii < 0) or np.any(positions[:, 1] + radii > self.container_height):
            return False

        # Check overlap constraints more efficiently with early termination
        # First, get approximate pairs using spatial indexing
        tree = KDTree(positions)
        pairs = tree.query_pairs(radii.sum() * 1.5)  # Reduced query radius for efficiency

        # Verify actual overlaps
        for i, j in pairs:
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            distance = np.sqrt(dx*dx + dy*dy)
            if distance < (radii[i] + radii[j]):
                return False

        return True

    def calculate_fitness(self, circles):
        """Calculate fitness as sum of radii with penalties for violations"""
        if not self.check_constraints(circles):
            return -1.0  # Invalid solution

        return np.sum(circles[:, 2])

    def apply_physics(self, circles, max_steps=300):
        """Apply physics-based refinement to improve solution quality with faster convergence"""
        positions = circles[:, :2].copy()
        radii = circles[:, 2].copy()

        prev_positions = positions.copy()
        stable_count = 0

        for step in range(max_steps):
            # Compute forces using spatial indexing with reduced overhead
            forces = np.zeros_like(positions)

            # Build spatial index for neighbor search with radius-based pruning
            tree = KDTree(positions)
            # Query only relevant neighbors based on max radius to reduce computation
            max_radius = np.max(radii) if len(radii) > 0 else 1.0
            neighbors = tree.query_ball_tree(tree, 2 * max_radius)

            # Compute repulsion forces efficiently
            for i in range(self.n_circles):
                for j_idx in neighbors[i]:
                    j = j_idx
                    if i != j:
                        dx = positions[i, 0] - positions[j, 0]
                        dy = positions[i, 1] - positions[j, 1]
                        distance = np.sqrt(dx*dx + dy*dy)

                        if distance < (radii[i] + radii[j]) and distance > 1e-8:
                            # Repulsive force when overlapping
                            force_magnitude = self.repulsion_strength * (1.0 - distance/(radii[i] + radii[j]))
                            fx = force_magnitude * dx / distance
                            fy = force_magnitude * dy / distance
                            forces[i, 0] += fx
                            forces[i, 1] += fy

            # Boundary forces - optimized boundary enforcement
            for i in range(self.n_circles):
                x, y, r = positions[i, 0], positions[i, 1], radii[i]

                # Boundary forces - scaled for better stability
                if x - r < 0:
                    forces[i, 0] += self.boundary_strength * (0 - (x - r)) * 0.5
                if x + r > self.container_width:
                    forces[i, 0] += self.boundary_strength * (self.container_width - (x + r)) * 0.5
                if y - r < 0:
                    forces[i, 1] += self.boundary_strength * (0 - (y - r)) * 0.5
                if y + r > self.container_height:
                    forces[i, 1] += self.boundary_strength * (self.container_height - (y + r)) * 0.5

            # Update positions with velocity and bounded movement
            for i in range(self.n_circles):
                positions[i, 0] += forces[i, 0] * self.dt
                positions[i, 1] += forces[i, 1] * self.dt

                # Keep within bounds with tight constraints
                positions[i, 0] = np.clip(positions[i, 0], r, self.container_width - r)
                positions[i, 1] = np.clip(positions[i, 1], r, self.container_height - r)

            # Convergence check with earlier termination
            pos_change = np.mean(np.linalg.norm(positions - prev_positions, axis=1))
            if pos_change < 0.0005:  # Tighter convergence criteria
                stable_count += 1
                if stable_count > 30:
                    break
            else:
                stable_count = 0

            prev_positions = positions.copy()

        # Update circles with refined positions
        circles[:, :2] = positions
        return circles

    def mutate(self, circles):
        """Apply mutation to create new solution with adaptive rates"""
        mutated = circles.copy()

        # Mutate each circle with probability based on its radius
        for i in range(self.n_circles):
            # Higher mutation rate for smaller circles (more room to grow)
            adaptive_rate = self.mutation_rate * (1 + 0.5 * (1 - circles[i, 2] / np.max(circles[:, 2])))
            if random.random() < adaptive_rate:
                # Mutate position
                mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, 0.04),
                                      mutated[i, 2], self.container_width - mutated[i, 2])
                mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, 0.04),
                                      mutated[i, 2], self.container_height - mutated[i, 2])

                # Mutate radius with adaptive magnitude
                delta_radius = np.random.normal(0, 0.008)
                mutated[i, 2] = np.clip(mutated[i, 2] + delta_radius, 0.002, 0.4)

        return mutated

    def crossover(self, parent1, parent2):
        """Create offspring through crossover with uniform probability"""
        child = parent1.copy()
        # Uniform crossover - swap properties with 50% probability
        for i in range(self.n_circles):
            if random.random() < 0.5:
                child[i, :] = parent2[i, :]

        return child

    def optimize(self):
        """Main optimization loop using evolutionary approach with enhanced strategy"""
        # Initialize population
        population = self.initialize_population()

        # Evaluate initial population
        fitness_scores = [self.calculate_fitness(individual) for individual in population]

        # Evolution loop with adaptive parameters
        best_solution = None
        best_fitness = -float('inf')
        generation = 0
        max_generations = 80  # Reduced for faster execution

        while generation < max_generations:
            # Sort by fitness
            sorted_indices = np.argsort(fitness_scores)[::-1]
            sorted_population = [population[i] for i in sorted_indices]
            sorted_fitness = [fitness_scores[i] for i in sorted_indices]

            # Track best solution
            if sorted_fitness[0] > best_fitness:
                best_fitness = sorted_fitness[0]
                best_solution = sorted_population[0].copy()

            # Selection: keep elite and create offspring
            elite = sorted_population[:self.elite_size]

            # Create new population
            new_population = elite.copy()

            # Generate offspring through crossover and mutation
            while len(new_population) < self.population_size:
                # Tournament selection with adaptive tournament size
                tournament_size = max(2, min(5, int(self.population_size * 0.2)))
                parent1_idx = self.tournament_selection(sorted_fitness, tournament_size)
                parent2_idx = self.tournament_selection(sorted_fitness, tournament_size)

                parent1 = sorted_population[parent1_idx]
                parent2 = sorted_population[parent2_idx]

                # Crossover
                child = self.crossover(parent1, parent2)

                # Mutation
                child = self.mutate(child)

                # Physics refinement
                child = self.apply_physics(child, max_steps=200)

                new_population.append(child)

            # Replace population
            population = new_population[:self.population_size]

            # Recalculate fitness
            fitness_scores = [self.calculate_fitness(individual) for individual in population]

            generation += 1

        # Final refinement of best solution with more steps
        if best_solution is not None:
            best_solution = self.apply_physics(best_solution, max_steps=500)

        return best_solution

    def tournament_selection(self, fitness_scores, tournament_size=3):
        """Select individual using tournament selection with better distribution"""
        tournament_indices = np.random.choice(len(fitness_scores), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        return winner_index

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Uses a hybrid evolutionary physics-based optimization approach.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    optimizer = CirclePackingOptimizer(container_width=1.0, container_height=1.0, n_circles=21)
    circles = optimizer.optimize()

    # Ensure valid output
    if circles is None:
        circles = np.zeros((21, 3))
        np.random.seed(42)  # For reproducibility
        circles[:, 0] = np.random.uniform(0.01, 0.99, 21)  # x coordinates
        circles[:, 1] = np.random.uniform(0.01, 0.99, 21)  # y coordinates
        circles[:, 2] = np.random.uniform(0.01, 0.1, 21)  # Initial small radii

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")