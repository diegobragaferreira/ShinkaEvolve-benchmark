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
        self.population_size = 30  # Increased population size
        self.elite_size = 6  # Increased elite size
        self.mutation_rate = 0.15  # Increased mutation rate for better exploration
        self.boundary_strength = 50.0
        self.repulsion_strength = 100.0
        self.radius_adjustment_factor = 0.01
        self.dt = 0.01

    def initialize_population(self):
        """Generate diverse initial population with hexagonal packing"""
        population = []

        # Use square container for simplicity and good packing
        container_width = 1.0
        container_height = 1.0

        # Hexagonal packing parameters for 21 circles
        # Hexagonal packing efficiency ~90.69%
        n_cols = 5
        n_rows = 5
        # Adjust based on the number of circles needed
        if self.n_circles < 15:
            n_cols = 4
            n_rows = 4
        elif self.n_circles < 25:
            n_cols = 5
            n_rows = 5

        # Calculate spacing based on optimal hexagonal packing
        # For 21 circles, we can fit them in roughly 5x5 grid with staggering
        hex_radius = 0.1  # Initial estimate
        hex_spacing_x = hex_radius * 2.0
        hex_spacing_y = hex_radius * np.sqrt(3)

        # Adjust spacing to fit within container
        max_x = container_width - hex_radius
        max_y = container_height - hex_radius

        # Determine actual grid dimensions
        grid_cols = min(n_cols, int(max_x / hex_spacing_x) + 1)
        grid_rows = min(n_rows, int(max_y / hex_spacing_y) + 1)

        # Calculate actual spacing that fits container
        actual_spacing_x = max_x / grid_cols if grid_cols > 0 else hex_spacing_x
        actual_spacing_y = max_y / grid_rows if grid_rows > 0 else hex_spacing_y

        # Generate initial population with improved hexagonal packing
        for _ in range(self.population_size):
            circles = np.zeros((self.n_circles, 3))

            # Hexagonal grid with staggered rows
            idx = 0
            for i in range(grid_rows):
                x_offset = (i % 2) * (actual_spacing_x / 2)
                for j in range(grid_cols):
                    if idx >= self.n_circles:
                        break

                    x = x_offset + (j + 0.5) * actual_spacing_x
                    y = (i + 0.5) * actual_spacing_y

                    # Ensure within bounds with margin
                    x = np.clip(x, hex_radius, container_width - hex_radius)
                    y = np.clip(y, hex_radius, container_height - hex_radius)

                    circles[idx, 0] = x
                    circles[idx, 1] = y
                    idx += 1

                if idx >= self.n_circles:
                    break

            # Fill remaining circles if needed
            for i in range(idx, self.n_circles):
                x = np.random.uniform(hex_radius, container_width - hex_radius)
                y = np.random.uniform(hex_radius, container_height - hex_radius)
                circles[i, 0] = x
                circles[i, 1] = y

            # Assign initial radii with better distribution
            # Use a more balanced approach for initial radii
            base_radius = min(actual_spacing_x, actual_spacing_y) * 0.3
            for i in range(self.n_circles):
                # Use normal distribution around base radius
                circles[i, 2] = np.random.normal(base_radius, base_radius * 0.2)
                circles[i, 2] = max(0.01, circles[i, 2])  # Ensure positive radius

            # Normalize total radius to reasonable value (but maintain diversity)
            total_radius = np.sum(circles[:, 2])
            if total_radius > 0:
                # Scale to match typical successful solutions
                target_sum = 1.8  # Higher target to encourage bigger radii
                scaling_factor = target_sum / total_radius
                circles[:, 2] *= scaling_factor

            population.append(circles.copy())
        return population

    def check_constraints(self, circles):
        """Check if all circles are within bounds and non-overlapping"""
        positions = circles[:, :2]
        radii = circles[:, 2]

        # Check boundary constraints
        for i in range(self.n_circles):
            x, y, r = positions[i, 0], positions[i, 1], radii[i]
            if x - r < 0 or x + r > self.container_width or y - r < 0 or y + r > self.container_height:
                return False

        # Check overlap constraints using KDTree for efficiency
        # Only query neighbors that could possibly overlap
        tree = KDTree(positions)
        # Use a reasonable search radius based on largest radius
        max_radius = np.max(radii) if len(radii) > 0 else 1.0
        pairs = tree.query_pairs(max_radius * 3.0)

        for i, j in pairs:
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            distance = np.sqrt(dx*dx + dy*dy)
            if distance < (radii[i] + radii[j]) and distance > 0:
                return False

        return True

    def calculate_fitness(self, circles):
        """Calculate fitness as sum of radii with penalties for violations"""
        if not self.check_constraints(circles):
            return -1.0  # Invalid solution

        return np.sum(circles[:, 2])

    def apply_physics(self, circles, max_steps=500):
        """Apply physics-based refinement to improve solution quality"""
        positions = circles[:, :2].copy()
        radii = circles[:, 2].copy()

        prev_positions = positions.copy()
        stable_count = 0
        dt = self.dt

        for step in range(max_steps):
            # Compute forces using spatial indexing
            forces = np.zeros_like(positions)

            # Build spatial index for efficient neighbor search
            tree = KDTree(positions)
            # Query neighbors within a reasonable distance
            neighbors = tree.query_ball_tree(tree, 2 * max(radii) if len(radii) > 0 else 1.0)

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

            # Boundary forces
            for i in range(self.n_circles):
                x, y, r = positions[i, 0], positions[i, 1], radii[i]

                # Left boundary
                if x - r < 0:
                    forces[i, 0] += self.boundary_strength * (0 - (x - r))
                # Right boundary
                if x + r > self.container_width:
                    forces[i, 0] += self.boundary_strength * (self.container_width - (x + r))
                # Bottom boundary
                if y - r < 0:
                    forces[i, 1] += self.boundary_strength * (0 - (y - r))
                # Top boundary
                if y + r > self.container_height:
                    forces[i, 1] += self.boundary_strength * (self.container_height - (y + r))

            # Update positions
            for i in range(self.n_circles):
                positions[i, 0] += forces[i, 0] * dt
                positions[i, 1] += forces[i, 1] * dt

                # Keep within bounds
                positions[i, 0] = np.clip(positions[i, 0], r, self.container_width - r)
                positions[i, 1] = np.clip(positions[i, 1], r, self.container_height - r)

            # Convergence check with dynamic threshold
            pos_change = np.mean(np.linalg.norm(positions - prev_positions, axis=1))
            if pos_change < 0.0001:
                stable_count += 1
                if stable_count > 30:  # Reduced threshold for faster convergence
                    break
            else:
                stable_count = 0

            prev_positions = positions.copy()

        # Update circles with refined positions
        circles[:, :2] = positions
        return circles

    def mutate(self, circles):
        """Apply mutation to create new solution"""
        mutated = circles.copy()

        # Randomly decide which circles to mutate
        for i in range(self.n_circles):
            if random.random() < self.mutation_rate:
                # Mutate position
                mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, 0.05),
                                      mutated[i, 2], self.container_width - mutated[i, 2])
                mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, 0.05),
                                      mutated[i, 2], self.container_height - mutated[i, 2])

                # Mutate radius
                mutated[i, 2] = np.clip(mutated[i, 2] + np.random.normal(0, 0.01), 0.001, 0.5)

        return mutated

    def crossover(self, parent1, parent2):
        """Create offspring through crossover"""
        child = parent1.copy()
        # Single point crossover on circle properties
        crossover_point = random.randint(1, self.n_circles - 1)

        for i in range(crossover_point, self.n_circles):
            # Swap circle properties
            child[i, :] = parent2[i, :]

        return child

    def optimize(self):
        """Main optimization loop using evolutionary approach"""
        # Initialize population
        population = self.initialize_population()

        # Evaluate initial population
        fitness_scores = [self.calculate_fitness(individual) for individual in population]

        # Evolution loop
        best_solution = None
        best_fitness = -float('inf')
        generation = 0
        max_generations = 100

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
                # Tournament selection
                parent1_idx = self.tournament_selection(sorted_fitness)
                parent2_idx = self.tournament_selection(sorted_fitness)

                parent1 = sorted_population[parent1_idx]
                parent2 = sorted_population[parent2_idx]

                # Crossover
                child = self.crossover(parent1, parent2)

                # Mutation
                child = self.mutate(child)

                # Physics refinement
                child = self.apply_physics(child)

                new_population.append(child)

            # Replace population
            population = new_population[:self.population_size]

            # Recalculate fitness
            fitness_scores = [self.calculate_fitness(individual) for individual in population]

            generation += 1

        # Final refinement of best solution
        if best_solution is not None:
            best_solution = self.apply_physics(best_solution, max_steps=1000)

        return best_solution

    def tournament_selection(self, fitness_scores, tournament_size=3):
        """Select individual using tournament selection"""
        tournament_indices = np.random.choice(len(fitness_scores), tournament_size)
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