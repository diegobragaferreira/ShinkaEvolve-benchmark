# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution, minimize

class PointConfiguration:
    """Represents a configuration of points and provides utility methods."""

    def __init__(self, points):
        self.points = np.array(points)
        self.n_points = len(points)

    def compute_min_max_ratio(self):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if self.n_points < 2:
            return 0

        # Compute pairwise distances with enhanced numerical stability
        distance_matrix = squareform(pdist(self.points))

        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(distance_matrix, np.inf)

        # Get all finite distances (excluding NaN and inf values)
        finite_distances = distance_matrix[np.isfinite(distance_matrix)]

        if len(finite_distances) == 0:
            return 0

        # Get min and max distances
        dmin = np.min(finite_distances)
        dmax = np.max(finite_distances)

        # Avoid division by zero
        if dmax == 0:
            return 0

        return dmin / dmax

    def compute_distance_matrix(self):
        """Compute full pairwise distance matrix."""
        return squareform(pdist(self.points))

    def get_clipped_points(self, lower=0, upper=1):
        """Get points clipped to specified bounds."""
        return np.clip(self.points, lower, upper)

    def copy(self):
        """Create a copy of this configuration."""
        return PointConfiguration(self.points.copy())

class OptimizationEngine:
    """Manages the optimization process with different strategies."""

    def __init__(self):
        self.best_ratio = 0
        self.best_points = None

    def objective_function(self, x_flat):
        """Objective function to maximize (negative because we minimize)."""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)

        # Ensure points are within bounds [0,1]
        points = np.clip(points, 0, 1)

        # Create temporary configuration
        temp_config = PointConfiguration(points)

        # Compute ratio
        ratio = temp_config.compute_min_max_ratio()

        # Return negative because we want to maximize
        return -ratio

    def generate_hexagonal_grid(self):
        """Generate a hexagonal grid arrangement."""
        points = []
        rows = 4
        cols = 4

        # Hexagonal packing parameters
        spacing_x = 1.0 / (cols - 1)
        spacing_y = np.sqrt(3) / 2 / (rows - 1)  # Height of equilateral triangle

        for i in range(rows):
            for j in range(cols):
                x = j * spacing_x + (i % 2) * spacing_x / 2
                y = i * spacing_y
                points.append([x, y])

        return np.array(points)

    def generate_golden_spiral(self):
        """Generate a golden spiral arrangement for better initial distribution."""
        indices = np.arange(16)
        golden_angle = 2.399963229728653
        angles = golden_angle * indices
        # Use logarithmic distribution for better point spreading
        radii = np.log(indices + 1) / np.log(16)
        golden_spiral = np.column_stack([
            0.5 + 0.45 * radii * np.cos(angles),
            0.5 + 0.45 * radii * np.sin(angles)
        ])
        return np.clip(golden_spiral, 0, 1)

    def generate_initial_strategies(self):
        """Generate multiple initial point configurations."""
        strategies = {}

        # Strategy 1: Hexagonal grid (primary)
        strategies['hex'] = self.generate_hexagonal_grid()

        # Strategy 2: Perturbed hexagonal grid
        np.random.seed(42)
        perturbed_hex = strategies['hex'] + np.random.normal(0, 0.02, strategies['hex'].shape)
        strategies['hex_perturbed'] = np.clip(perturbed_hex, 0, 1)

        # Strategy 3: Regular grid with jitter
        regular_grid = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0
                y = (j + 0.5) / 4.0
                regular_grid.append([x, y])
        regular_grid = np.array(regular_grid)
        jittered_grid = regular_grid + np.random.normal(0, 0.01, regular_grid.shape)
        strategies['grid_jittered'] = np.clip(jittered_grid, 0, 1)

        # Strategy 4: Golden spiral (better radial distribution)
        strategies['spiral'] = self.generate_golden_spiral()

        # Strategy 5: Random points with edge avoidance
        np.random.seed(123)
        random_points = np.random.rand(16, 2)
        strategies['random'] = np.clip(random_points, 0.05, 0.95)

        # Strategy 6: More structured random pattern
        np.random.seed(999)
        structured_random = np.random.rand(16, 2) * 0.8 + 0.1  # Centered in [0.1, 0.9]
        strategies['structured_random'] = structured_random

        return strategies

    def evaluate_all_strategies(self, strategies):
        """Evaluate all initial strategies and return the best one."""
        best_strategy = None
        best_ratio = 0

        for name, points in strategies.items():
            config = PointConfiguration(points)
            ratio = config.compute_min_max_ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_strategy = points.copy()

        return best_strategy, best_ratio

    def robust_optimization(self, initial_points, max_evaluations=1000):
        """Perform robust optimization with multiple strategies."""
        # Flatten for optimization
        x0 = initial_points.flatten()

        # Define bounds for each coordinate (0 to 1)
        bounds = [(0, 1) for _ in range(32)]

        # Track convergence for early stopping
        prev_best = float('inf')
        stagnation_count = 0
        max_stagnation = 10

        try:
            # First aggressive differential evolution with good parameters
            de_result = differential_evolution(
                self.objective_function,
                bounds,
                maxiter=max_evaluations // 10,
                popsize=20,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False,
                tol=1e-8,
                strategy='best1bin'
            )

            # Try local refinement with L-BFGS-B
            if de_result.success:
                refined_result = minimize(
                    self.objective_function,
                    de_result.x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'ftol': 1e-12, 'gtol': 1e-12},
                    tol=1e-12
                )

                if refined_result.success:
                    final_points = refined_result.x.reshape(-1, 2)
                    final_points = np.clip(final_points, 0, 1)
                    return final_points

            # If local refinement fails, return DE result
            final_points = de_result.x.reshape(-1, 2)
            final_points = np.clip(final_points, 0, 1)
            return final_points

        except Exception:
            # If all else fails, return the initial points
            return initial_points

    def run_multi_start_optimization(self):
        """Run optimization with multiple restarts and strategies."""
        # Generate initial strategies
        strategies = self.generate_initial_strategies()

        # Find the best initial configuration
        best_initial, initial_ratio = self.evaluate_all_strategies(strategies)

        # Initialize best results
        self.best_points = best_initial.copy()
        self.best_ratio = initial_ratio

        # Multi-start optimization with different initial variations
        for restart in range(5):
            # Generate new variation of the initial points
            np.random.seed(restart + 1000)
            perturbed_points = best_initial.copy()
            noise_level = 0.03 + restart * 0.005  # Gradually increasing noise
            perturbed_points += np.random.normal(0, noise_level, best_initial.shape)
            perturbed_points = np.clip(perturbed_points, 0, 1)

            # Optimize this variant
            optimized_points = self.robust_optimization(perturbed_points, max_evaluations=500)
            optimized_ratio = PointConfiguration(optimized_points).compute_min_max_ratio()

            if optimized_ratio > self.best_ratio:
                self.best_ratio = optimized_ratio
                self.best_points = optimized_points.copy()

        # Final optimization on the best configuration found
        final_points = self.robust_optimization(self.best_points, max_evaluations=300)
        final_ratio = PointConfiguration(final_points).compute_min_max_ratio()

        # One final refinement attempt with a more focused approach
        np.random.seed(9999)
        last_attempt = final_points + np.random.normal(0, 0.005, final_points.shape)
        last_attempt = np.clip(last_attempt, 0, 1)
        refined_final = self.robust_optimization(last_attempt, max_evaluations=200)
        refined_ratio = PointConfiguration(refined_final).compute_min_max_ratio()

        if refined_ratio > final_ratio:
            return refined_final
        else:
            return final_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    engine = OptimizationEngine()
    return engine.run_multi_start_optimization()

# EVOLVE-BLOCK-END