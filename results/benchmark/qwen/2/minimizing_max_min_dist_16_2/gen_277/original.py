# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

class PointConfigurationGenerator:
    """Handles generation of various initial point configurations."""

    @staticmethod
    def create_structured_grid(perturbation_magnitude=0.05):
        """Create a structured 4x4 grid with adaptive perturbation."""
        grid_x = np.linspace(0.1, 0.9, 4)
        grid_y = np.linspace(0.1, 0.9, 4)
        grid_points = np.array([[x, y] for x in grid_x for y in grid_y])

        # Apply adaptive perturbation
        noise = np.random.normal(0, perturbation_magnitude/3, (16, 2))
        return np.clip(grid_points + noise, 0, 1)

    @staticmethod
    def create_random_with_clustering_avoidance():
        """Create random configuration with clustering avoidance."""
        config = np.random.uniform(0.05, 0.95, (16, 2))
        # Add some structure to avoid very tight clusters
        for i in range(0, 16, 4):  # Group every 4 points
            group_center = np.mean(config[i:i+4], axis=0)
            config[i:i+4] += np.random.normal(0, 0.03, (4, 2))
            config[i:i+4] = np.clip(config[i:i+4], 0, 1)
        return config

    @staticmethod
    def create_fibonacci_spiral():
        """Create Fibonacci spiral-like arrangement."""
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radii = np.sqrt(np.linspace(0.05, 0.45, 16))  # Square root for uniform distribution
        fib_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return np.clip((fib_points + 1) / 2, 0, 1)  # Normalize to [0,1]

    @staticmethod
    def create_hexagonal_approximation(perturbation_magnitude=0.03):
        """Create hexagonal grid approximation with perturbation."""
        hex_x = np.array([0.15, 0.45, 0.75, 0.3, 0.6, 0.15, 0.45, 0.75, 0.225, 0.525, 0.825, 0.375, 0.675, 0.225, 0.525, 0.825])
        hex_y = np.array([0.15, 0.15, 0.15, 0.45, 0.45, 0.75, 0.75, 0.75, 0.3, 0.3, 0.3, 0.6, 0.6, 0.9, 0.9, 0.9])
        hex_points = np.column_stack([hex_x, hex_y])

        # Apply adaptive perturbation
        noise = np.random.normal(0, perturbation_magnitude/3, (16, 2))
        return np.clip(hex_points + noise, 0, 1)

    @classmethod
    def generate_all_configurations(cls):
        """Generate all types of initial configurations."""
        configs = []

        # Configuration 1: Structured 4x4 grid
        np.random.seed(42)
        configs.append(cls.create_structured_grid())

        # Configuration 2: Random with clustering avoidance
        np.random.seed(123)
        configs.append(cls.create_random_with_clustering_avoidance())

        # Configuration 3: Fibonacci spiral
        np.random.seed(456)
        configs.append(cls.create_fibonacci_spiral())

        # Configuration 4: Hexagonal grid approximation
        np.random.seed(789)
        configs.append(cls.create_hexagonal_approximation())

        return configs

class OptimizationPipeline:
    """Handles the optimization procedures with multiple refinement stages."""

    @staticmethod
    def objective(x):
        """Objective function to maximize the ratio of minimum to maximum distance."""
        points = x.reshape(-1, 2)
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)

        if d_max == 0:
            return -np.inf
        return -d_min / d_max

    @staticmethod
    def constraint(x):
        """Constraint function ensuring points stay within bounds."""
        points = x.reshape(-1, 2)
        return np.concatenate([
            points[:, 0],           # x coordinates >= 0
            1 - points[:, 0],       # x coordinates <= 1
            points[:, 1],           # y coordinates >= 0
            1 - points[:, 1]        # y coordinates <= 1
        ])

    @classmethod
    def optimize_with_refinement(cls, x0):
        """Perform sequential optimization with refinement stages."""
        try:
            # Stage 1: Fast optimization with L-BFGS-B
            bounds = [(0, 1) for _ in range(32)]
            result1 = minimize(
                cls.objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-6, 'gtol': 1e-6}
            )

            if not result1.success:
                return None

            # Stage 2: Precise optimization with SLSQP
            result2 = minimize(
                cls.objective,
                result1.x,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': cls.constraint},
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
            )

            if result2.success:
                return result2.x
        except Exception:
            return None
        return None

class SolutionEvaluator:
    """Handles solution evaluation and ratio computation."""

    @staticmethod
    def compute_ratio(points):
        """Compute the min/max distance ratio for given points."""
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return 0.0
        return d_min / d_max

class MultiStartOptimizer:
    """Manages the multi-start optimization strategy."""

    def __init__(self):
        self.config_generator = PointConfigurationGenerator()
        self.optimizer = OptimizationPipeline()
        self.evaluator = SolutionEvaluator()

    def find_best_solution(self):
        """Find the best solution using multi-start optimization."""
        best_ratio = -np.inf
        best_points = None

        # Generate multiple initial configurations
        initial_configs = self.config_generator.generate_all_configurations()

        # Run optimizations with different initial configurations
        for i, initial_points in enumerate(initial_configs):
            try:
                # Optimize using the refinement approach
                optimized_x = self.optimizer.optimize_with_refinement(initial_points.flatten())

                if optimized_x is not None:
                    optimized_points = optimized_x.reshape(-1, 2)
                    final_ratio = self.evaluator.compute_ratio(optimized_points)

                    if final_ratio > best_ratio:
                        best_ratio = final_ratio
                        best_points = optimized_points.copy()

            except Exception:
                continue

        # If no successful optimization, return the best initial configuration
        if best_points is None:
            best_points = initial_configs[0] if initial_configs else np.random.uniform(0, 1, (16, 2))

        return best_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    # Initialize the multi-start optimizer
    optimizer = MultiStartOptimizer()

    # Find and return the best solution
    return optimizer.find_best_solution()

# EVOLVE-BLOCK-END