# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import time
from typing import Tuple, List
import copy

class PointDistributionOptimizer:
    """Optimizes point distribution to maximize min/max distance ratio."""

    def __init__(self, n_points: int = 16, dimensions: int = 2):
        self.n_points = n_points
        self.dimensions = dimensions
        self.benchmark_ratio = 1 / np.sqrt(12.889266112)  # 0.2786
        self.max_time = 180.0  # seconds

    def calculate_min_max_ratio(self, points: np.ndarray) -> float:
        """Calculate the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0.0

        distances = pdist(points)

        # Handle edge cases
        if len(distances) == 0 or np.max(distances) <= 0:
            return 0.0

        d_min = np.min(distances)
        d_max = np.max(distances)

        # Avoid division by zero
        if d_max <= 0:
            return 0.0

        return d_min / d_max

    def initialize_population(self, population_size: int) -> List[np.ndarray]:
        """Initialize diverse population of point configurations."""
        population = []
        np.random.seed(42)

        # Generate multiple diverse initial configurations
        for i in range(population_size):
            # Mix of different initialization strategies
            if i % 4 == 0:
                # Hexagonal grid pattern
                points = self._generate_hexagonal_grid()
            elif i % 4 == 1:
                # Spiral pattern
                points = self._generate_spiral_pattern()
            elif i % 4 == 2:
                # Random uniform distribution
                points = np.random.rand(self.n_points, self.dimensions)
            else:
                # Grid with random jitter
                points = self._generate_grid_with_jitter()

            # Add small random noise to increase diversity
            noise_level = 0.01 * (1 + i * 0.05)
            points += np.random.normal(0, noise_level, points.shape)

            # Clip to valid bounds
            points = np.clip(points, 0, 1)
            population.append(points)

        return population

    def _generate_hexagonal_grid(self) -> np.ndarray:
        """Generate points in a hexagonal grid pattern."""
        # Create a grid that approximates hexagonal packing
        rows = int(np.ceil(np.sqrt(self.n_points)))
        cols = int(np.ceil(self.n_points / rows))

        points = []
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows

        for i in range(rows):
            for j in range(cols):
                if len(points) >= self.n_points:
                    break

                # Offset odd rows for hexagonal arrangement
                offset = 0.5 * (i % 2)
                x = (j + offset) * spacing_x
                y = i * spacing_y

                # Ensure we don't exceed bounds
                x = min(x, 0.99)
                y = min(y, 0.99)

                points.append([x, y])

        # Trim to exact number of points
        points = np.array(points[:self.n_points])

        # Normalize to fit properly in [0,1] box
        if len(points) > 0:
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)

            if x_max > x_min and y_max > y_min:
                points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min) * 0.9 + 0.05
                points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min) * 0.9 + 0.05

        return points

    def _generate_spiral_pattern(self) -> np.ndarray:
        """Generate points in a spiral pattern."""
        points = []
        angle_step = 2 * np.pi / 10
        radius_step = 1.0 / 10

        for i in range(self.n_points):
            if i == 0:
                points.append([0.5, 0.5])  # Center point
            else:
                angle = i * angle_step
                radius = min(0.45, i * radius_step)
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                points.append([x, y])

        # Fill remaining points with random if needed
        while len(points) < self.n_points:
            points.append([np.random.rand(), np.random.rand()])

        return np.array(points[:self.n_points])

    def _generate_grid_with_jitter(self) -> np.ndarray:
        """Generate grid pattern with random jitter."""
        points = []
        grid_size = int(np.ceil(np.sqrt(self.n_points)))

        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) >= self.n_points:
                    break
                x = i / (grid_size - 1) if grid_size > 1 else 0.5
                y = j / (grid_size - 1) if grid_size > 1 else 0.5
                # Add slight randomness to avoid perfect grid
                x += (np.random.rand() - 0.5) * 0.1
                y += (np.random.rand() - 0.5) * 0.1
                points.append([x, y])

        return np.clip(np.array(points[:self.n_points]), 0, 1)

    def evaluate_individual(self, points: np.ndarray) -> float:
        """Evaluate fitness of individual point configuration."""
        return self.calculate_min_max_ratio(points)

    def global_optimization(self, initial_points: np.ndarray, max_iter: int = 300) -> Tuple[np.ndarray, float]:
        """Use global optimization to explore promising regions."""
        def objective(x_flat):
            points = x_flat.reshape(-1, self.dimensions)
            return -self.calculate_min_max_ratio(points)

        # Use differential evolution for broad exploration
        bounds = [(0, 1) for _ in range(len(initial_points.flatten()))]

        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=max_iter,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False
            )

            if result.success:
                optimized_points = result.x.reshape(-1, self.dimensions)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = self.calculate_min_max_ratio(optimized_points)
                return optimized_points, ratio
        except Exception:
            pass

        return initial_points, self.evaluate_individual(initial_points)

    def optimize_individual(self, points: np.ndarray, max_iter: int = 500) -> Tuple[np.ndarray, float]:
        """Refine a single point configuration using adaptive local optimization."""
        def objective(x_flat):
            points_candidate = x_flat.reshape(-1, self.dimensions)
            # Return negative ratio for minimization
            return -self.calculate_min_max_ratio(points_candidate)

        best_points = points.copy()
        best_ratio = self.calculate_min_max_ratio(best_points)

        # Try multiple optimization methods with adaptive parameters
        methods_and_params = [
            ('L-BFGS-B', {'maxiter': max_iter // 2}),
            ('Nelder-Mead', {'maxiter': max_iter // 2, 'adaptive': True}),
        ]

        try:
            for method, options in methods_and_params:
                # Add small random perturbation to avoid local minima
                perturbed = points + np.random.normal(0, 0.001, points.shape)
                perturbed = np.clip(perturbed, 0, 1)

                result = minimize(
                    objective,
                    perturbed.flatten(),
                    method=method,
                    bounds=[(0, 1) for _ in range(len(perturbed.flatten()))],
                    options=options,
                    tol=1e-6
                )

                if result.success:
                    optimized_points = result.x.reshape(-1, self.dimensions)
                    optimized_points = np.clip(optimized_points, 0, 1)
                    ratio = self.calculate_min_max_ratio(optimized_points)

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()

        except Exception as e:
            # Fall back to original points if optimization fails
            pass

        return best_points, best_ratio

    def evolve(self) -> np.ndarray:
        """Main evolutionary optimization loop."""
        start_time = time.time()

        # Start with diverse initial configurations
        initial_configs = self.initialize_population(10)
        best_ratio = -np.inf
        best_points = None

        # Try each initial configuration with hybrid optimization
        for i, initial_config in enumerate(initial_configs):
            if time.time() - start_time > self.max_time - 10:
                break

            # First global optimization to find promising regions
            global_points, global_ratio = self.global_optimization(initial_config, max_iter=200)

            # Then fine-tune with local optimization
            local_points, local_ratio = self.optimize_individual(global_points, max_iter=300)

            # Further refine with additional local optimization
            final_points, final_ratio = self.optimize_individual(local_points, max_iter=200)

            # Keep track of the best solution found
            if final_ratio > best_ratio:
                best_ratio = final_ratio
                best_points = copy.deepcopy(final_points)

        # If no good solution found, fallback to a simple approach
        if best_points is None:
            # Simple approach: start with hexagonal grid and optimize
            initial_points = self._generate_hexagonal_grid()
            best_points, _ = self.optimize_individual(initial_points, max_iter=500)

        return best_points

    def _tournament_selection(self, population: List[np.ndarray],
                            fitness_scores: List[float], tournament_size: int) -> np.ndarray:
        """Select individual using tournament selection."""
        selected_indices = np.random.choice(len(population), tournament_size)
        best_idx = selected_indices[np.argmax([fitness_scores[i] for i in selected_indices])]
        return copy.deepcopy(population[best_idx])

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Perform crossover between two parents."""
        # Blend crossover (BLX-α)
        alpha = 0.5
        child = np.zeros_like(parent1)

        for i in range(len(parent1)):
            # Get bounds
            low = min(parent1[i][0], parent2[i][0]) - alpha * abs(parent1[i][0] - parent2[i][0])
            high = max(parent1[i][0], parent2[i][0]) + alpha * abs(parent1[i][0] - parent2[i][0])

            # Sample new value in range
            child[i][0] = np.random.uniform(low, high)

            low = min(parent1[i][1], parent2[i][1]) - alpha * abs(parent1[i][1] - parent2[i][1])
            high = max(parent1[i][1], parent2[i][1]) + alpha * abs(parent1[i][1] - parent2[i][1])

            child[i][1] = np.random.uniform(low, high)

        return np.clip(child, 0, 1)

    def _mutate(self, points: np.ndarray) -> np.ndarray:
        """Apply Gaussian mutation to points."""
        mutation_strength = 0.05
        mutated = points + np.random.normal(0, mutation_strength, points.shape)
        return np.clip(mutated, 0, 1)

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = PointDistributionOptimizer(n_points=16, dimensions=2)
    return optimizer.evolve()

# EVOLVE-BLOCK-END