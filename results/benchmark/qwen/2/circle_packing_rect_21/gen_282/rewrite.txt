# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import time
import random
import warnings

class CirclePackingOptimizer:
    def __init__(self, container_width=1.0, container_height=1.0, n_circles=21):
        self.container_width = container_width
        self.container_height = container_height
        self.n_circles = n_circles
        self.max_iterations = 5000
        self.population_size = 30  # Increased for better exploration
        self.elite_size = 6
        self.mutation_rate = 0.15  # Slightly higher for more exploration
        self.boundary_strength = 50.0
        self.repulsion_strength = 100.0
        self.radius_adjustment_factor = 0.01
        self.dt = 0.01
        self.seeds = [42, 123, 456, 789, 999]  # Multiple seeds for diversity

    def _calculate_optimal_dimensions(self):
        """Calculate theoretically optimal rectangle dimensions for 21 circles"""
        # For perimeter = 4, width + height = 2
        # Try different aspect ratios to find optimal
        ratios_to_try = [0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
        best_ratio = 1.0
        best_efficiency = 0.0
        
        for ratio in ratios_to_try:
            width = 1.0  # We'll vary height to maintain perimeter constraint
            height = 2.0 - width
            
            if height > 0 and abs(width/height - ratio) < 0.1:  # Allow some tolerance
                # Estimate packing efficiency based on theoretical hexagonal packing
                # For a 5x5 grid (25 positions) with 21 circles, this should work well
                efficiency = 1.0 / (1.0 + abs(ratio - 1.0)) * 0.9  # Favor square-like ratios
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_ratio = ratio
                    
        # Use the best found ratio
        if best_ratio <= 1.0:
            width = 1.0
            height = 2.0 - width
        else:
            height = 1.0
            width = 2.0 - height
            
        return width, height

    def _create_grid_layout(self, width, height, cols, rows):
        """Create a grid layout for circles with improved spacing"""
        circles = np.zeros((self.n_circles, 3))

        # Create regular grid with staggered rows for better packing
        cell_width = width / (cols + 1)
        cell_height = height / (rows + 1)

        idx = 0
        for i in range(rows):
            offset = (i % 2) * (cell_width / 2)  # Stagger alternate rows
            for j in range(cols):
                if idx >= self.n_circles:
                    break
                x = (j + 1) * cell_width + offset + np.random.uniform(-cell_width*0.05, cell_width*0.05)
                y = (i + 1) * cell_height + np.random.uniform(-cell_height*0.05, cell_height*0.05)

                # Ensure within bounds
                x = np.clip(x, 0.01, width - 0.01)
                y = np.clip(y, 0.01, height - 0.01)

                circles[idx, 0] = x
                circles[idx, 1] = y
                idx += 1
            if idx >= self.n_circles:
                break

        # Set initial radii using a gamma distribution for good spread
        avg_spacing = min(cell_width, cell_height) * 0.6
        for i in range(self.n_circles):
            # Create a mix of large and small circles for better packing
            radius = avg_spacing * np.random.gamma(1.8, 0.6)  # Gamma distribution gives good spread
            radius = np.clip(radius, 0.001, min(width, height) * 0.25)
            circles[i, 2] = radius

        return circles

    def _generate_grid_initialization(self):
        """Create initial configuration using adaptive grid placement optimized for the rectangle dimensions"""
        # Try several grid configurations to find the best one
        grid_configs = [(4, 6), (5, 5), (3, 7), (6, 4), (7, 3)]
        
        best_circles = None
        best_fitness = -float('inf')
        
        # Try multiple configurations
        for cols, rows in grid_configs:
            if cols * rows >= self.n_circles:
                circles = self._create_grid_layout(self.container_width, self.container_height, cols, rows)
                
                # Quick fitness check to see if this layout has promise
                fitness = self.calculate_fitness(circles)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_circles = circles.copy()
        
        # If no good configuration found, fall back to a simple approach
        if best_circles is None:
            best_circles = np.zeros((self.n_circles, 3))
            for i in range(self.n_circles):
                r = np.random.uniform(0.01, min(self.container_width, self.container_height) * 0.15)
                x = np.random.uniform(r, self.container_width - r)
                y = np.random.uniform(r, self.container_height - r)
                best_circles[i] = [x, y, r]
        
        return best_circles

    def initialize_population(self):
        """Generate diverse initial population with improved adaptive grid-based initialization"""
        population = []
        
        # Try to find optimal container dimensions
        container_width, container_height = self._calculate_optimal_dimensions()
        self.container_width = container_width
        self.container_height = container_height

        # Create diverse population using multiple strategies
        for i in range(self.population_size):
            # Alternate between different initialization methods
            if i % 3 == 0:
                # Grid-based with good parameters
                circles = self._generate_grid_initialization()
            elif i % 3 == 1:
                # Random initialization with better constraints
                circles = self._create_random_layout()
            else:
                # Hybrid approach using grid with mutations
                circles = self._generate_grid_initialization()
                # Apply some random mutations to diversify
                for j in range(min(5, self.n_circles)):  # Mutate only a few circles
                    if np.random.random() < 0.5:
                        idx = np.random.randint(0, self.n_circles)
                        circles[idx, 0] = np.clip(circles[idx, 0] + np.random.normal(0, 0.03), 
                                                circles[idx, 2], self.container_width - circles[idx, 2])
                        circles[idx, 1] = np.clip(circles[idx, 1] + np.random.normal(0, 0, 0.03), 
                                                circles[idx, 2], self.container_height - circles[idx, 2])
                        circles[idx, 2] = np.clip(circles[idx, 2] * np.random.uniform(0.9, 1.1), 
                                                0.001, min(self.container_width, self.container_height) * 0.3)
            
            # Validate and correct constraints
            if not self.check_constraints(circles):
                circles = self._create_fallback_layout()
                
            population.append(circles.copy())
            
        return population

    def _create_random_layout(self):
        """Create a random layout with better initial distribution"""
        circles = np.zeros((self.n_circles, 3))
        for i in range(self.n_circles):
            r = np.random.uniform(0.01, min(self.container_width, self.container_height) * 0.15)
            x = np.random.uniform(r, self.container_width - r)
            y = np.random.uniform(r, self.container_height - r)
            circles[i] = [x, y, r]
        return circles

    def _create_fallback_layout(self):
        """Create a fallback layout if initial methods fail"""
        circles = np.zeros((self.n_circles, 3))
        for i in range(self.n_circles):
            circles[i, 0] = np.random.uniform(0.01, self.container_width - 0.01)
            circles[i, 1] = np.random.uniform(0.01, self.container_height - 0.01)
            circles[i, 2] = np.random.uniform(0.01, min(self.container_width, self.container_height) * 0.1)
        return circles

    def check_constraints(self, circles):
        """Check if all circles are within bounds and non-overlapping with improved efficiency"""
        positions = circles[:, :2]
        radii = circles[:, 2]

        # Check boundary constraints
        for i in range(self.n_circles):
            x, y, r = positions[i, 0], positions[i, 1], radii[i]
            if x - r < 0 or x + r > self.container_width or y - r < 0 or y + r > self.container_height:
                return False

        # Efficient overlap checking using KDTree with better parameters
        try:
            # Use a more appropriate distance threshold for query
            max_radius = np.max(radii) if len(radii) > 0 else 1.0
            tree = cKDTree(positions)
            # Query pairs within 2 * max_radius distance
            pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')

            # Check actual overlap for all candidate pairs
            for i, j in pairs:
                if i < j:  # Only check each pair once
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
            # Calculate penalty based on severity of violations
            penalty = 0

            # Check boundary violations more carefully
            positions = circles[:, :2]
            radii = circles[:, 2]

            for i in range(self.n_circles):
                x, y, r = positions[i, 0], positions[i, 1], radii[i]

                # Calculate how much we're violating boundaries
                left_violation = max(0, r - x)
                right_violation = max(0, x + r - self.container_width)
                bottom_violation = max(0, r - y)
                top_violation = max(0, y + r - self.container_height)

                # Sum of all boundary violations (square for stronger penalty)
                boundary_penalty = (left_violation + right_violation + bottom_violation + top_violation) ** 2
                penalty += boundary_penalty * 1000  # Heavy penalty for boundary violations

            # Check overlap violations with more detail
            try:
                tree = cKDTree(positions)
                # Use more conservative query distance
                max_radius = np.max(radii) if len(radii) > 0 else 1.0
                pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')

                for i, j in pairs:
                    if i < j:  # Only consider each pair once
                        dx = positions[i, 0] - positions[j, 0]
                        dy = positions[i, 1] - positions[j, 1]
                        distance = np.sqrt(dx*dx + dy*dy)
                        overlap = (radii[i] + radii[j]) - distance

                        if overlap > 0:
                            # Penalty based on overlap amount, squared for stronger effect
                            penalty += (overlap ** 2) * 10000
            except Exception:
                # Fallback to brute force for overlap checking
                for i in range(self.n_circles):
                    for j in range(i+1, self.n_circles):
                        dx = positions[i, 0] - positions[j, 0]
                        dy = positions[i, 1] - positions[j, 1]
                        distance = np.sqrt(dx*dx + dy*dy)
                        overlap = (radii[i] + radii[j]) - distance

                        if overlap > 0:
                            penalty += (overlap ** 2) * 10000

            # Return negative fitness (since we minimize in scipy) minus penalty
            return -(np.sum(circles[:, 2]) + penalty)

        return np.sum(circles[:, 2])

    def apply_physics(self, circles, max_steps=500):
        """Apply physics-based refinement to improve solution quality with enhanced convergence"""
        positions = circles[:, :2].copy()
        radii = circles[:, 2].copy()

        prev_positions = positions.copy()
        stable_count = 0
        last_improvement_step = 0

        for step in range(max_steps):
            # Compute forces using spatial indexing
            forces = np.zeros_like(positions)

            # Build spatial index for efficient neighbor search
            try:
                tree = cKDTree(positions)
                # Query neighbors within a reasonable distance to limit computation
                max_radius = np.max(radii) if len(radii) > 0 else 1.0
                neighbors = tree.query_ball_tree(tree, 2 * max_radius, return_length=True)

                # Compute repulsion forces efficiently
                for i in range(self.n_circles):
                    for j_idx in tree.query_ball_point(positions[i], 2 * max_radius):
                        if i != j_idx:
                            dx = positions[i, 0] - positions[j_idx, 0]
                            dy = positions[i, 1] - positions[j_idx, 1]
                            distance = np.sqrt(dx*dx + dy*dy)

                            if distance < (radii[i] + radii[j_idx]) and distance > 1e-8:
                                # Repulsive force when overlapping
                                force_magnitude = self.repulsion_strength * (1.0 - distance/(radii[i] + radii[j_idx]))
                                fx = force_magnitude * dx / distance
                                fy = force_magnitude * dy / distance
                                forces[i, 0] += fx
                                forces[i, 1] += fy
            except Exception:
                # Fallback to simpler approach if spatial indexing fails
                for i in range(self.n_circles):
                    for j in range(self.n_circles):
                        if i != j:
                            dx = positions[i, 0] - positions[j, 0]
                            dy = positions[i, 1] - positions[j, 1]
                            distance = np.sqrt(dx*dx + dy*dy)

                            if distance < (radii[i] + radii[j]) and distance > 1e-8:
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

            # Convergence check with more sensitive criteria
            pos_change = np.mean(np.linalg.norm(positions - prev_positions, axis=1))
            if pos_change < 0.0001:
                stable_count += 1
                if stable_count > 30:
                    break
            else:
                stable_count = 0
                last_improvement_step = step

            # Early termination if no improvement for too long
            if step - last_improvement_step > 200:
                break

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
                # Mutate position with larger steps for better exploration
                mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, 0.07),
                                      mutated[i, 2], self.container_width - mutated[i, 2])
                mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, 0.07),
                                      mutated[i, 2], self.container_height - mutated[i, 2])

                # Mutate radius with wider range
                mutated[i, 2] = np.clip(mutated[i, 2] + np.random.normal(0, 0.015), 0.001, 0.5)

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
        """Main optimization loop using evolutionary approach with enhanced strategies"""
        # Initialize population with multiple runs to find best starting point
        best_solution = None
        best_fitness = -float('inf')
        best_seed = 0
        
        # Try with different seeds for better exploration
        for seed_idx, seed_value in enumerate(self.seeds):
            np.random.seed(seed_value)
            random.seed(seed_value)
            
            # Initialize population
            population = self.initialize_population()

            # Evaluate initial population
            fitness_scores = [self.calculate_fitness(individual) for individual in population]

            # Evolution loop
            generation = 0
            max_generations = 120  # Reduced generations for faster execution
            stagnation_counter = 0  # Track stagnation for early stopping

            while generation < max_generations:
                # Sort by fitness
                sorted_indices = np.argsort(fitness_scores)[::-1]
                sorted_population = [population[i] for i in sorted_indices]
                sorted_fitness = [fitness_scores[i] for i in sorted_indices]

                # Track best solution
                if sorted_fitness[0] > best_fitness:
                    best_fitness = sorted_fitness[0]
                    best_solution = sorted_population[0].copy()
                    best_seed = seed_idx
                    stagnation_counter = 0  # Reset stagnation counter
                else:
                    stagnation_counter += 1

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

                # Early stopping if stagnation occurs
                if stagnation_counter > 25:  # Stop if no improvement for 25 generations
                    break

                generation += 1

        # Final refinement of best solution with enhanced physics
        if best_solution is not None:
            # Apply more intensive physics refinement for final touch
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