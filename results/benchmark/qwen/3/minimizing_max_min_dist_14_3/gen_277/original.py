# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import SphericalVoronoi
import time

class PointOptimizer:
    """Main optimizer class that orchestrates the point distribution optimization process."""

    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        self.best_solution = None
        self.best_ratio = 0.0
        self.max_time = 345  # seconds (leaving buffer)
        self.start_time = time.time()

    def compute_min_max_ratio(self, points):
        """Compute the min/max distance ratio for given points."""
        if len(points) < 2:
            return 0.0

        # Compute pairwise distances efficiently
        distances = pdist(points)

        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Avoid division by zero
        if d_max == 0:
            return 0.0

        return d_min / d_max

    def fibonacci_sphere(self, n):
        """Generate points on sphere using Fibonacci spiral."""
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))

        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = golden_angle * i  # Golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def spherical_constraint(self, points):
        """Normalize points to lie on the unit sphere."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return points / norms

    def evaluate_solution(self, points):
        """Evaluate a solution and update best if better."""
        ratio = self.compute_min_max_ratio(points)
        if ratio > self.best_ratio:
            self.best_ratio = ratio
            self.best_solution = points.copy()
        return ratio

class InitializationStrategies:
    """Collection of different initialization strategies."""

    @staticmethod
    def fibonacci_with_perturbation():
        """Fibonacci sphere with small random perturbation."""
        points = PointOptimizer().fibonacci_sphere(14)
        perturbed = points + np.random.normal(0, 0.03, points.shape)
        return PointOptimizer().spherical_constraint(perturbed)

    @staticmethod
    def random_spherical():
        """Random points on sphere."""
        points = np.random.randn(14, 3)
        return PointOptimizer().spherical_constraint(points)

    @staticmethod
    def structured_axis_distribution():
        """Structured distribution along axes."""
        points = np.zeros((14, 3))
        for i in range(14):
            if i < 3:
                # Along positive axes
                points[i] = [1 if j==i else 0 for j in range(3)]
            elif i < 6:
                # Along negative axes
                points[i] = [-1 if j==i-3 else 0 for j in range(3)]
            elif i < 9:
                # Diagonal combinations
                j = i - 6
                points[i] = [1 if k==j else -1 if k==(j+1)%3 else 0 for k in range(3)]
            else:
                # Random points on sphere
                points[i] = np.random.randn(3)
        return PointOptimizer().spherical_constraint(points)

    @staticmethod
    def perturbed_fibonacci_large():
        """Perturbed Fibonacci with larger variance."""
        points = PointOptimizer().fibonacci_sphere(14)
        perturbed = points + np.random.normal(0, 0.08, points.shape)
        return PointOptimizer().spherical_constraint(perturbed)

    @staticmethod
    def voronoi_distribution():
        """Spherical Voronoi distribution."""
        points = np.random.randn(14, 3)
        return PointOptimizer().spherical_constraint(points)

    @staticmethod
    def scaled_fibonacci_configurations():
        """Multiple scaled Fibonacci configurations."""
        configurations = []
        fib_points = PointOptimizer().fibonacci_sphere(14)
        for scale in [0.7, 0.85, 1.15, 1.3]:
            scaled_fib = fib_points * scale
            scaled_perturbed = scaled_fib + np.random.normal(0, 0.04, scaled_fib.shape)
            configurations.append(PointOptimizer().spherical_constraint(scaled_perturbed))
        return configurations

class OptimizationPipeline:
    """Handles the complete optimization pipeline with adaptive strategies."""

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def adaptive_differential_evolution(self, initial_points, maxiter=50):
        """Enhanced differential evolution with adaptive population sizing."""
        points = initial_points.copy()
        history = []
        stagnation_count = 0
        max_stagnation = 10
        min_popsize = 10
        max_popsize = 30
        popsize = 15

        bounds = [(-1, 1)] * (14 * 3)

        def adaptive_objective(x_flat):
            points = x_flat.reshape(-1, 3)
            points = self.optimizer.spherical_constraint(points)
            ratio = self.optimizer.compute_min_max_ratio(points)
            return -ratio

        try:
            for iteration in range(3):
                current_popsize = popsize

                # Adaptive population sizing logic
                if len(history) >= 2:
                    improvement = -history[-1] - (-history[-2])
                    if improvement < 1e-6:
                        stagnation_count += 1
                        if stagnation_count >= 3 and current_popsize < max_popsize:
                            current_popsize = min(current_popsize + 5, max_popsize)
                    else:
                        stagnation_count = 0
                        if current_popsize > min_popsize and iteration > 0:
                            current_popsize = max(current_popsize - 3, min_popsize)

                result = differential_evolution(
                    adaptive_objective,
                    bounds,
                    maxiter=maxiter//3,
                    popsize=current_popsize,
                    seed=self.optimizer.seed + iteration,
                    disp=False,
                    polish=True,
                    strategy='best1bin'
                )

                if result.success:
                    temp_points = result.x.reshape(-1, 3)
                    temp_points = self.optimizer.spherical_constraint(temp_points)
                    current_ratio = self.optimizer.compute_min_max_ratio(temp_points)
                    history.append(current_ratio)

                    if len(history) == 1 or current_ratio > history[-2]:
                        points = temp_points.copy()

                    if len(history) >= 2:
                        improvement = current_ratio - history[-2]
                        if improvement < 1e-8:
                            stagnation_count += 1
                            if stagnation_count >= max_stagnation:
                                break
                        else:
                            stagnation_count = 0
                else:
                    break

        except Exception:
            pass

        return points

    def enhanced_hill_climbing(self, initial_points, maxiter=200):
        """Enhanced hill climbing with adaptive step sizes."""
        points = initial_points.copy()
        last_ratio = self.optimizer.compute_min_max_ratio(points)
        patience = 0
        max_patience = 15
        step_size = 0.01
        improvement_history = []
        max_improvement_history = 5

        for iteration in range(maxiter):
            current_ratio = self.optimizer.compute_min_max_ratio(points)
            best_points = points.copy()
            best_ratio = current_ratio

            # Adaptive step size based on recent improvement
            if len(improvement_history) >= 2:
                avg_improvement = np.mean(improvement_history[-min(len(improvement_history), max_improvement_history):])
                if avg_improvement > 1e-6:
                    step_size = min(0.02, step_size * 1.1)
                elif avg_improvement < 1e-8:
                    step_size = max(0.0001, step_size * 0.9)

            improvement = current_ratio - last_ratio
            improvement_history.append(improvement)
            if len(improvement_history) > max_improvement_history:
                improvement_history.pop(0)

            # Try perturbations with adaptive step sizes
            for i in range(14):
                for dim in range(3):
                    # Positive direction
                    test_points = points.copy()
                    test_points[i, dim] += step_size
                    test_points = self.optimizer.spherical_constraint(test_points)
                    test_ratio = self.optimizer.compute_min_max_ratio(test_points)

                    if test_ratio > best_ratio:
                        best_ratio = test_ratio
                        best_points = test_points.copy()

                    # Negative direction
                    test_points = points.copy()
                    test_points[i, dim] -= step_size
                    test_points = self.optimizer.spherical_constraint(test_points)
                    test_ratio = self.optimizer.compute_min_max_ratio(test_points)

                    if test_ratio > best_ratio:
                        best_ratio = test_ratio
                        best_points = test_points.copy()

            if best_ratio <= current_ratio:
                patience += 1
                if patience > max_patience:
                    break
            else:
                patience = 0
                points = best_points

            last_ratio = current_ratio

        return points

    def local_refinement(self, initial_points):
        """Local refinement with L-BFGS-B and adaptive tolerance tightening."""
        def local_obj(x_flat):
            points = x_flat.reshape(-1, 3)
            points = self.optimizer.spherical_constraint(points)
            ratio = self.optimizer.compute_min_max_ratio(points)
            return -ratio

        try:
            x0 = initial_points.flatten()
            bounds = [(-1, 1)] * (14 * 3)

            # Start with looser tolerances for faster initial convergence
            ftol = 1e-6
            gtol = 1e-6

            # Try with initial loose tolerances
            result = minimize(
                local_obj,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 60, 'ftol': ftol, 'gtol': gtol, 'disp': False}
            )

            if result.success:
                points = result.x.reshape(-1, 3)
                points = self.optimizer.spherical_constraint(points)

                # If we got a good improvement, try tighter tolerances for final precision
                # We use a heuristic to determine if it's worth the extra computation
                current_ratio = self.optimizer.compute_min_max_ratio(points)
                if current_ratio > 0.35:  # Only refine if we're getting reasonable results
                    # Try with tighter tolerances
                    ftol_tight = 1e-9
                    gtol_tight = 1e-9
                    result_tight = minimize(
                        local_obj,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 60, 'ftol': ftol_tight, 'gtol': gtol_tight, 'disp': False}
                    )
                    if result_tight.success:
                        points_tight = result_tight.x.reshape(-1, 3)
                        points_tight = self.optimizer.spherical_constraint(points_tight)
                        tight_ratio = self.optimizer.compute_min_max_ratio(points_tight)
                        if tight_ratio > current_ratio:
                            points = points_tight.copy()

                return points
        except:
            pass
        return initial_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    # Initialize core components
    optimizer = PointOptimizer(seed=42)
    pipeline = OptimizationPipeline(optimizer)

    # Generate diverse initial configurations using factory methods
    initial_strategies = [
        InitializationStrategies.fibonacci_with_perturbation,
        InitializationStrategies.random_spherical,
        InitializationStrategies.structured_axis_distribution,
        InitializationStrategies.perturbed_fibonacci_large,
        InitializationStrategies.voronoi_distribution,
    ]

    # Add scaled fibonacci configurations which returns list
    scaled_configs = InitializationStrategies.scaled_fibonacci_configurations()
    initial_strategies.extend(scaled_configs)

    # Process all initial configurations
    for i, strategy in enumerate(initial_strategies):
        # Handle both single return and list returns
        if callable(strategy):
            try:
                points = strategy()
            except Exception:
                continue
        else:
            points = strategy

        # Pipeline optimization process
        try:
            # Stage 1: Global optimization with adaptive DE
            global_result = pipeline.adaptive_differential_evolution(points, maxiter=30)

            # Stage 2: Local refinement
            local_result = pipeline.local_refinement(global_result)

            # Stage 3: Enhanced hill climbing
            final_result = pipeline.enhanced_hill_climbing(local_result, maxiter=60)

            # Evaluate final solution
            optimizer.evaluate_solution(final_result)

        except Exception:
            continue

    # If no successful optimization was performed, fall back to Fibonacci with perturbation
    if optimizer.best_solution is None:
        fib_points = optimizer.fibonacci_sphere(14)
        fib_points = fib_points + np.random.normal(0, 0.05, fib_points.shape)
        optimizer.best_solution = optimizer.spherical_constraint(fib_points)

    return optimizer.best_solution

# EVOLVE-BLOCK-END