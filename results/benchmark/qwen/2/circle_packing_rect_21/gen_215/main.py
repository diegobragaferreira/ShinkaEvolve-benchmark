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
        self.population_size = 30
        self.elite_size = 6
        self.mutation_rate = 0.15
        self.boundary_strength = 100.0
        self.repulsion_strength = 200.0
        self.radius_adjustment_factor = 0.02
        self.dt = 0.005
        self.convergence_threshold = 1e-6

    def initialize_population(self):
        """Generate diverse initial population with improved adaptive grid initialization"""
        population = []

        # Test different aspect ratios to find optimal container dimensions
        best_ratio = 1.0
        best_fitness = -float('inf')

        # Test several aspect ratios from 0.5 to 2.0
        ratios = np.linspace(0.5, 2.0, 15)

        for ratio in ratios:
            width = 1.0
            height = width / ratio

            if width + height <= 2.0:  # Perimeter constraint
                # Estimate packing density for this configuration
                area = width * height
                avg_circle_area = area / self.n_circles

                # This is a simplified estimate - we'll evaluate actual performance later
                estimated_density = min(1.0, 0.8 * np.pi / (2 * np.sqrt(3)))  # Hexagonal packing efficiency
                fitness = estimated_density * (area / (avg_circle_area + 1e-8))

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_ratio = ratio

        # Use the best ratio found
        container_width = 1.0
        container_height = container_width / best_ratio

        for _ in range(self.population_size):
            circles = np.zeros((self.n_circles, 3))

            # Use a more sophisticated grid-based initialization
            # Calculate grid dimensions based on container aspect ratio
            aspect_ratio = container_width / container_height

            # Use a hexagonal-like packing pattern for better efficiency
            if aspect_ratio > 1:  # Wide container
                cols = max(3, int(np.ceil(np.sqrt(self.n_circles * aspect_ratio))))
                rows = max(3, int(np.ceil(self.n_circles / cols)))
            else:  # Tall container
                rows = max(3, int(np.ceil(np.sqrt(self.n_circles / aspect_ratio))))
                cols = max(3, int(np.ceil(self.n_circles / rows)))

            # Ensure reasonable grid dimensions
            cols = max(2, min(cols, 12))
            rows = max(2, min(rows, 12))

            cell_width = container_width / cols
            cell_height = container_height / rows

            # Place circles using a staggered grid pattern for better packing
            circle_idx = 0
            for i in range(rows):
                offset = (i % 2) * (cell_width / 2)  # Stagger every other row
                for j in range(cols):
                    if circle_idx >= self.n_circles:
                        break

                    # Grid position with systematic jitter
                    x_base = (j + 0.5) * cell_width + offset
                    y_base = (i + 0.5) * cell_height

                    # Add controlled jitter to improve diversity
                    x_jitter = np.random.normal(0, cell_width * 0.1)
                    y_jitter = np.random.normal(0, cell_height * 0.1)

                    x = np.clip(x_base + x_jitter, 0.01, container_width - 0.01)
                    y = np.clip(y_base + y_jitter, 0.01, container_height - 0.01)

                    # Set initial radius with consideration of local density
                    base_radius = min(cell_width, cell_height) * 0.25
                    # Use log-normal distribution to create some large and some small circles
                    r = np.clip(base_radius * np.exp(np.random.normal(0, 0.3)), 0.005, 0.15)

                    circles[circle_idx, 0] = x
                    circles[circle_idx, 1] = y
                    circles[circle_idx, 2] = r
                    circle_idx += 1

                if circle_idx >= self.n_circles:
                    break

            # Ensure we have exactly n_circles
            if circle_idx < self.n_circles:
                # Fill remaining circles with more strategic placement
                for i in range(circle_idx, self.n_circles):
                    # Better spatial distribution using stratified sampling
                    x = np.random.uniform(0.01, container_width - 0.01)
                    y = np.random.uniform(0.01, container_height - 0.01)
                    # Use smaller radii for remaining circles
                    r = np.random.uniform(0.005, 0.08)
                    circles[i, 0] = x
                    circles[i, 1] = y
                    circles[i, 2] = r

            # Normalize total radius to reasonable value with better scaling
            total_radius = np.sum(circles[:, 2])
            if total_radius > 0:
                # Scale to be closer to expected good solutions
                target_sum = 1.2  # Adjusted target for better performance
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
        try:
            tree = KDTree(positions)
            # Query pairs with a safe distance threshold
            pairs = tree.query_pairs(2 * max(radii) if len(radii) > 0 else 1.0)

            for i, j in pairs:
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                distance = np.sqrt(dx*dx + dy*dy)
                if distance < (radii[i] + radii[j]):
                    return False
        except Exception:
            # Fallback to brute force if KDTree fails
            for i in range(self.n_circles):
                for j in range(i+1, self.n_circles):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    if distance < (radii[i] + radii[j]):
                        return False

        return True

    def calculate_fitness(self, circles):
        """Calculate fitness as sum of radii with penalties for violations"""
        if not self.check_constraints(circles):
            # Heavy penalty for constraint violations
            return -1000.0

        # Base fitness is the sum of radii
        base_fitness = np.sum(circles[:, 2])

        # Additional penalty for very small radii (to encourage larger circles)
        small_radius_penalty = 0
        for r in circles[:, 2]:
            if r < 0.01:  # Very small radii penalized heavily
                small_radius_penalty -= 10 * (0.01 - r)

        # Add penalty for tight overlaps (circles nearly touching)
        overlap_penalty = 0
        positions = circles[:, :2]
        radii = circles[:, 2]

        try:
            tree = KDTree(positions)
            pairs = tree.query_pairs(2 * max(radii) if len(radii) > 0 else 1.0)

            for i, j in pairs:
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                distance = np.sqrt(dx*dx + dy*dy)
                overlap = (radii[i] + radii[j]) - distance
                if overlap > 0:
                    overlap_penalty -= 100 * overlap
        except Exception:
            pass  # Fallback if KDTree fails

        return base_fitness + small_radius_penalty + overlap_penalty

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
            try:
                tree = KDTree(positions)
                # Query all points within a distance that could potentially interact
                neighbors = tree.query_ball_tree(tree, 2 * max(radii) if len(radii) > 0 else 1.0)
            except Exception:
                # Fallback to full pairwise comparison if KDTree fails
                neighbors = []
                for i in range(self.n_circles):
                    neighbors.append([])
                    for j in range(self.n_circles):
                        if i != j:
                            dx = positions[i, 0] - positions[j, 0]
                            dy = positions[i, 1] - positions[j, 1]
                            distance = np.sqrt(dx*dx + dy*dy)
                            if distance < (radii[i] + radii[j]):
                                neighbors[i].append(j)

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

            # Boundary forces with smoother application
            for i in range(self.n_circles):
                x, y, r = positions[i, 0], positions[i, 1], radii[i]

                # Soft boundary repulsion (closer to boundary = stronger force)
                left_force = max(0, r - x) * self.boundary_strength * 0.5
                right_force = max(0, x + r - self.container_width) * self.boundary_strength * 0.5
                bottom_force = max(0, r - y) * self.boundary_strength * 0.5
                top_force = max(0, y + r - self.container_height) * self.boundary_strength * 0.5

                forces[i, 0] -= left_force - right_force
                forces[i, 1] -= bottom_force - top_force

            # Update positions with momentum
            for i in range(self.n_circles):
                # Add some damping to prevent oscillation
                velocity_x = forces[i, 0] * self.dt
                velocity_y = forces[i, 1] * self.dt

                # Apply velocity with damping
                positions[i, 0] += velocity_x * 0.7
                positions[i, 1] += velocity_y * 0.7

                # Keep within bounds
                positions[i, 0] = np.clip(positions[i, 0], r, self.container_width - r)
                positions[i, 1] = np.clip(positions[i, 1], r, self.container_height - r)

            # Convergence check with multiple criteria
            pos_change = np.mean(np.linalg.norm(positions - prev_positions, axis=1))
            if pos_change < self.convergence_threshold:
                stable_count += 1
                if stable_count > 20:
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
        """Main optimization loop using evolutionary approach with two-stage refinement"""
        # Stage 1: Coarse optimization with larger population and fewer generations
        # Initialize population
        population = self.initialize_population()

        # Evaluate initial population
        fitness_scores = [self.calculate_fitness(individual) for individual in population]

        # Evolution loop - coarse stage
        best_solution = None
        best_fitness = -float('inf')
        generation = 0
        max_generations_coarse = 50

        while generation < max_generations_coarse:
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
                child = self.apply_physics(child, max_steps=100)

                new_population.append(child)

            # Replace population
            population = new_population[:self.population_size]

            # Recalculate fitness
            fitness_scores = [self.calculate_fitness(individual) for individual in population]

            generation += 1

        # Stage 2: Fine tuning with focused optimization on best solution
        if best_solution is not None:
            # Refine the best solution further with more intensive physics simulation
            best_solution = self.apply_physics(best_solution, max_steps=1000)

            # Also create a few more refined versions of the best solution for diversity
            additional_solutions = []
            for _ in range(5):
                # Make a copy and perturb slightly
                perturbed = best_solution.copy()
                for i in range(self.n_circles):
                    # Small random perturbations
                    if np.random.random() < 0.3:  # 30% chance to perturb each circle
                        perturbed[i, 0] += np.random.normal(0, 0.05)
                        perturbed[i, 1] += np.random.normal(0, 0.05)
                        perturbed[i, 2] += np.random.normal(0, 0.02)

                        # Ensure bounds
                        perturbed[i, 0] = np.clip(perturbed[i, 0], perturbed[i, 2], self.container_width - perturbed[i, 2])
                        perturbed[i, 1] = np.clip(perturbed[i, 1], perturbed[i, 2], self.container_height - perturbed[i, 2])
                        perturbed[i, 2] = np.clip(perturbed[i, 2], 0.001, 0.5)

                # Apply final physics refinement
                perturbed = self.apply_physics(perturbed, max_steps=200)
                additional_solutions.append(perturbed)

            # Evaluate all solutions including the refined versions
            all_solutions = [best_solution] + additional_solutions
            all_fitness = [self.calculate_fitness(sol) for sol in all_solutions]

            # Select the best among all
            best_idx = np.argmax(all_fitness)
            best_solution = all_solutions[best_idx]

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