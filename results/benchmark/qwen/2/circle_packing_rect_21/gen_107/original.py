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
        self.mutation_rate = 0.1
        self.boundary_strength = 50.0
        self.repulsion_strength = 100.0
        self.radius_adjustment_factor = 0.01
        self.dt = 0.01

    def initialize_population(self):
        """Generate diverse initial population with adaptive grid-based initialization"""
        population = []

        # Calculate grid dimensions based on number of circles
        # For 21 circles, we want roughly a 4x5 grid (or similar)
        grid_rows = int(np.ceil(np.sqrt(self.n_circles)))
        grid_cols = int(np.ceil(self.n_circles / grid_rows))

        # Ensure we don't exceed reasonable grid dimensions
        grid_rows = min(grid_rows, 6)
        grid_cols = min(grid_cols, 6)

        for _ in range(self.population_size):
            circles = np.zeros((self.n_circles, 3))

            # Create adaptive grid spacing
            cell_width = self.container_width / (grid_cols + 1)
            cell_height = self.container_height / (grid_rows + 1)

            # Place circles on grid with slight randomization
            idx = 0
            for i in range(grid_rows):
                for j in range(grid_cols):
                    if idx >= self.n_circles:
                        break
                    x = (j + 1) * cell_width + np.random.uniform(-cell_width*0.1, cell_width*0.1)
                    y = (i + 1) * cell_height + np.random.uniform(-cell_height*0.1, cell_height*0.1)
                    circles[idx, 0] = x
                    circles[idx, 1] = y
                    idx += 1
                if idx >= self.n_circles:
                    break

            # Assign initial radii
            # Start with relatively large radii to encourage dense packing
            base_radius = min(self.container_width, self.container_height) * 0.05
            for i in range(self.n_circles):
                circles[i, 2] = base_radius * (0.8 + np.random.random() * 0.4)

            # Apply spatial constraint checking to ensure initial feasibility
            # If not initially feasible, use random fallback
            if not self.check_constraints(circles):
                # Fallback to random initialization for infeasible grids
                circles[:, 0] = np.random.uniform(0.01, self.container_width - 0.01, self.n_circles)
                circles[:, 1] = np.random.uniform(0.01, self.container_height - 0.01, self.n_circles)
                circles[:, 2] = np.random.uniform(0.01, 0.1, self.n_circles)

            # Normalize total radius to reasonable value
            total_radius = np.sum(circles[:, 2])
            if total_radius > 0:
                scaling_factor = 0.5 / total_radius
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
        tree = KDTree(positions)
        pairs = tree.query_pairs(radii.sum() * 2)

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

    def apply_physics(self, circles, max_steps=500):
        """Apply physics-based refinement to improve solution quality"""
        positions = circles[:, :2].copy()
        radii = circles[:, 2].copy()

        prev_positions = positions.copy()
        stable_count = 0

        for step in range(max_steps):
            # Compute forces using spatial indexing
            forces = np.zeros_like(positions)

            # Build spatial index for efficient neighbor search
            tree = KDTree(positions)
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
                positions[i, 0] += forces[i, 0] * self.dt
                positions[i, 1] += forces[i, 1] * self.dt

                # Keep within bounds
                positions[i, 0] = np.clip(positions[i, 0], r, self.container_width - r)
                positions[i, 1] = np.clip(positions[i, 1], r, self.container_height - r)

            # Convergence check
            pos_change = np.mean(np.linalg.norm(positions - prev_positions, axis=1))
            if pos_change < 0.001:
                stable_count += 1
                if stable_count > 50:
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