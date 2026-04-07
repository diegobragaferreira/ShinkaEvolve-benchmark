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
        self.boundary_strength = 80.0
        self.repulsion_strength = 150.0
        self.radius_adjustment_factor = 0.01
        self.dt = 0.01
        self.penalty_weight = 5000.0

    def initialize_population(self):
        """Generate diverse initial population with improved adaptive grid-based initialization"""
        population = []

        # Optimize container dimensions for 21 circles
        # For perimeter = 4, width + height = 2
        best_ratio = 1.0
        best_efficiency = 0.0

        # Test ratios more systematically
        ratios_to_try = np.linspace(0.5, 2.0, 15)
        
        for ratio in ratios_to_try:
            width = 1.0
            height = width / ratio
            
            if width + height <= 2.0:
                # Estimate packing efficiency for this aspect ratio
                if ratio > 1.2:  # Wide rectangle
                    efficiency = 0.8 + 0.1 / ratio
                elif ratio < 0.8:  # Tall rectangle
                    efficiency = 0.8 + 0.1 * ratio
                else:  # Balanced
                    efficiency = 0.9
                
                estimated_fill = min(1.0, (self.n_circles * 0.05 * 0.05 * np.pi) / (width * height))
                efficiency *= estimated_fill
                
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_ratio = ratio

        container_width = 1.0
        container_height = container_width / best_ratio

        # Create hexagonal-like grid for better packing efficiency
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        grid_cols = max(3, int(np.ceil(np.sqrt(self.n_circles / phi))))
        grid_rows = max(3, int(np.ceil(self.n_circles / grid_cols)))
        
        # Limit grid size to prevent excessive computation
        grid_cols = min(grid_cols, 10)
        grid_rows = min(grid_rows, 10)

        for _ in range(self.population_size):
            circles = np.zeros((self.n_circles, 3))

            # Create staggered grid for better packing
            cell_width = container_width / (grid_cols + 1.5)
            cell_height = container_height / (grid_rows + 1.5)

            idx = 0
            for i in range(grid_rows):
                offset = (i % 2) * (cell_width / 2)
                for j in range(grid_cols):
                    if idx >= self.n_circles:
                        break
                    x = (j + 1) * cell_width + offset + np.random.uniform(-cell_width*0.15, cell_width*0.15)
                    y = (i + 1) * cell_height + np.random.uniform(-cell_height*0.15, cell_height*0.15)

                    # Ensure within bounds with safety margin
                    x = np.clip(x, 0.01, container_width - 0.01)
                    y = np.clip(y, 0.01, container_height - 0.01)

                    circles[idx, 0] = x
                    circles[idx, 1] = y
                    idx += 1
                if idx >= self.n_circles:
                    break

            # Assign initial radii with log-normal distribution for diversity
            mean_radius = min(container_width, container_height) * 0.04
            for i in range(self.n_circles):
                circles[i, 2] = mean_radius * np.exp(np.random.normal(0, 0.4))

            # Check initial constraints and fallback if needed
            if not self.check_constraints(circles):
                # Use random placement with better bounds
                circles[:, 0] = np.random.uniform(0.01, container_width - 0.01, self.n_circles)
                circles[:, 1] = np.random.uniform(0.01, container_height - 0.01, self.n_circles)
                circles[:, 2] = np.random.uniform(0.01, min(container_width, container_height) * 0.06, self.n_circles)

            # Normalize to reasonable scale
            total_radius = np.sum(circles[:, 2])
            if total_radius > 0:
                target_sum = 1.2  # Typical good value
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
            penalty = 0
            
            # Calculate boundary violation penalty
            positions = circles[:, :2]
            radii = circles[:, 2]

            for i in range(self.n_circles):
                x, y, r = positions[i, 0], positions[i, 1], radii[i]
                
                # Calculate how much we're violating boundaries
                left_violation = max(0, r - x)
                right_violation = max(0, x + r - self.container_width)
                bottom_violation = max(0, r - y)
                top_violation = max(0, y + r - self.container_height)

                boundary_penalty = (left_violation + right_violation + bottom_violation + top_violation) * 500
                penalty += boundary_penalty

            # Calculate overlap penalty
            tree = KDTree(positions)
            pairs = tree.query_pairs(radii.sum() * 2)

            for i, j in pairs:
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                distance = np.sqrt(dx*dx + dy*dy)
                overlap = (radii[i] + radii[j]) - distance

                if overlap > 0:
                    overlap_penalty = overlap * 50000
                    penalty += overlap_penalty

            return -(np.sum(circles[:, 2]) + penalty)

        return np.sum(circles[:, 2])

    def apply_physics(self, circles, max_steps=300):
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

                # Boundary forces with stronger enforcement near edges
                if x - r < 0:
                    forces[i, 0] += self.boundary_strength * (0 - (x - r))
                if x + r > self.container_width:
                    forces[i, 0] += self.boundary_strength * (self.container_width - (x + r))
                if y - r < 0:
                    forces[i, 1] += self.boundary_strength * (0 - (y - r))
                if y + r > self.container_height:
                    forces[i, 1] += self.boundary_strength * (self.container_height - (y + r))

            # Apply forces and update positions
            for i in range(self.n_circles):
                # Apply forces with limited velocity
                force_magnitude = np.sqrt(forces[i, 0]**2 + forces[i, 1]**2)
                if force_magnitude > 0:
                    forces[i] = forces[i] * min(0.08, 0.05 / (force_magnitude + 1e-8))
                
                # Update positions
                positions[i, 0] += forces[i, 0] * dt
                positions[i, 1] += forces[i, 1] * dt

                # Keep within bounds with safety margin
                positions[i, 0] = np.clip(positions[i, 0], r + 0.005, self.container_width - r - 0.005)
                positions[i, 1] = np.clip(positions[i, 1], r + 0.005, self.container_height - r - 0.005)

            # Convergence check
            pos_change = np.mean(np.linalg.norm(positions - prev_positions, axis=1))
            if pos_change < 0.0005:
                stable_count += 1
                if stable_count > 30:
                    break
            else:
                stable_count = 0

            prev_positions = positions.copy()

            # Gradually reduce time step for stabilization
            if step > 100:
                dt *= 0.99

        # Update circles with refined positions
        circles[:, :2] = positions
        return circles

    def mutate(self, circles):
        """Apply mutation to create new solution"""
        mutated = circles.copy()

        # Apply mutation to some circles
        mutation_indices = np.random.choice(self.n_circles, 
                                           size=int(self.n_circles * self.mutation_rate * 0.5), 
                                           replace=False)
        
        for i in mutation_indices:
            # Randomly decide whether to mutate position or radius
            if random.random() < 0.7:  # 70% chance to mutate position
                # Mutate position with larger step size
                mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, 0.08),
                                      mutated[i, 2] + 0.005, self.container_width - mutated[i, 2] - 0.005)
                mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, 0.08),
                                      mutated[i, 2] + 0.005, self.container_height - mutated[i, 2] - 0.005)
            else:  # 30% chance to mutate radius
                mutated[i, 2] = np.clip(mutated[i, 2] + np.random.normal(0, 0.015), 0.001, 0.3)

        return mutated

    def crossover(self, parent1, parent2):
        """Create offspring through crossover"""
        child = parent1.copy()
        # Uniform crossover on circle properties
        mask = np.random.rand(self.n_circles) > 0.5

        for i in range(self.n_circles):
            if mask[i]:
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
        max_generations = 120

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
            best_solution = self.apply_physics(best_solution, max_steps=800)

        return best_solution

    def tournament_selection(self, fitness_scores, tournament_size=4):
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