# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

class SpatialOptimizer:
    def __init__(self, n_points=14, dimensions=3, seed=42):
        self.n_points = n_points
        self.dimensions = dimensions
        self.seed = seed
        np.random.seed(seed)

    def _initialize_spherical_points(self):
        """Initialize points on a unit sphere using Fibonacci spiral method"""
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # Golden angle

        for i in range(self.n_points):
            y = 1 - (i / float(self.n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def _initialize_cube_grid_points(self):
        """Initialize points in a 3D cube grid"""
        # Find appropriate grid size
        grid_size = int(np.ceil(self.n_points**(1/3)))
        coords = np.linspace(0, 1, grid_size)
        grid_points = []

        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if len(grid_points) < self.n_points:
                        grid_points.append([coords[i], coords[j], coords[k]])

        return np.array(grid_points[:self.n_points])

    def _evaluate_initialization(self, points):
        """Fast evaluation of initialization quality"""
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max > 1e-12:
            return d_min / d_max
        return 0

    def _generate_initial_strategies(self):
        """Generate multiple initialization strategies"""
        strategies = []

        # Strategy 1: Spherical Fibonacci points
        fib_points = self._initialize_spherical_points()
        fib_points = (fib_points + 1) / 2  # Normalize to [0,1]^3
        strategies.append(("fibonacci", fib_points.copy()))

        # Strategy 2: Cube grid points
        cube_points = self._initialize_cube_grid_points()
        strategies.append(("cube_grid", cube_points.copy()))

        # Strategy 3: Random points
        random_points = np.random.rand(self.n_points, self.dimensions)
        strategies.append(("random", random_points.copy()))

        # Strategy 4: Perturbed spherical points
        perturbed_points = fib_points + np.random.normal(0, 0.05, (self.n_points, self.dimensions))
        perturbed_points = np.clip(perturbed_points, 0, 1)
        strategies.append(("perturbed", perturbed_points.copy()))

        # Strategy 5: Alternative perturbed spherical points
        alt_perturbed_points = fib_points + np.random.normal(0, 0.03, (self.n_points, self.dimensions))
        alt_perturbed_points = np.clip(alt_perturbed_points, 0, 1)
        strategies.append(("alt_perturbed", alt_perturbed_points.copy()))

        return strategies

    def _adaptive_differential_evolution(self, objective_func, bounds, maxiter=300):
        """Enhanced differential evolution with adaptive parameters"""
        # Starting parameters with more careful initialization
        current_popsize = 25  # Slightly larger starting population
        current_tol = 1e-6
        current_mutation = (0.5, 1.0)
        current_recombination = 0.7

        # Track performance for adaptive adjustments
        prev_obj_values = []
        improvement_threshold = 1e-8

        # Use a more sophisticated adaptive scheme
        population_schedule = [
            (0, 20, 25),   # Early iterations: medium-high population
            (20, 60, 20),  # Middle iterations: standard population
            (60, 100, 15), # Later iterations: smaller population for fine-tuning
            (100, float('inf'), 10)  # Final iterations: smallest population
        ]

        for iteration in range(maxiter // 10):
            # Determine appropriate population size based on iteration
            for start_iter, end_iter, pop_size in population_schedule:
                if start_iter <= iteration < end_iter:
                    current_popsize = pop_size
                    break

            try:
                result = differential_evolution(
                    objective_func,
                    bounds,
                    seed=self.seed + iteration,
                    maxiter=10,
                    popsize=current_popsize,
                    tol=current_tol,
                    mutation=current_mutation,
                    recombination=current_recombination,
                    disp=False
                )

                # Track objective values
                prev_obj_values.append(-result.fun)
                if len(prev_obj_values) > 5:
                    prev_obj_values.pop(0)

                # Adaptive adjustments based on progress
                if len(prev_obj_values) >= 2:
                    improvement = prev_obj_values[-1] - prev_obj_values[-2]

                    # Adjust tolerance based on improvement
                    if improvement < improvement_threshold:
                        current_tol = max(1e-12, current_tol * 0.8)
                    else:
                        current_tol = max(1e-12, current_tol * 0.95)

                    # Adjust mutation rate based on convergence
                    if improvement < improvement_threshold * 0.1:
                        # Slow convergence - increase mutation for more exploration
                        current_mutation = (0.7, 1.0)
                    elif improvement > improvement_threshold:
                        # Fast convergence - reduce mutation for exploitation
                        current_mutation = (0.3, 0.7)
                    else:
                        # Moderate convergence - standard mutation
                        current_mutation = (0.5, 1.0)

                # Early stopping if progress is minimal
                if len(prev_obj_values) >= 5:
                    recent_improvements = [prev_obj_values[i+1] - prev_obj_values[i]
                                         for i in range(len(prev_obj_values)-1)]
                    if all(abs(imp) < improvement_threshold for imp in recent_improvements):
                        break

                # Update parameters for next iteration
                current_recombination = 0.7  # Keep recombination constant

            except Exception as e:
                # Fallback to simpler parameters if something goes wrong
                current_popsize = max(10, current_popsize - 5)
                current_tol = max(1e-10, current_tol * 2)
                if current_popsize < 10:
                    break
                continue

        return result

    def _local_refinement(self, points):
        """Apply local refinement to improve final solution"""
        def objective_local(x):
            points_local = x.reshape(-1, 3)
            distances = pdist(points_local)

            if len(distances) == 0:
                return -np.inf

            d_min = np.min(distances)
            d_max = np.max(distances)

            if d_max > 1e-12:
                return -(d_min / d_max)
            else:
                return -np.inf

        try:
            x0_refine = points.flatten()
            bounds = [(0, 1)] * self.n_points * 3

            result_refine = minimize(
                objective_local,
                x0_refine,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-10, 'gtol': 1e-10},
                tol=1e-10
            )

            refined_points = result_refine.x.reshape(-1, 3)
            refined_points = np.clip(refined_points, 0, 1)
            return refined_points
        except:
            return points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    optimizer = SpatialOptimizer(n_points=14, dimensions=3, seed=42)

    # Generate all initialization strategies
    strategies = optimizer._generate_initial_strategies()

    # Evaluate all strategies and select the best
    best_initialization = None
    best_ratio = -np.inf

    for name, points in strategies:
        ratio = optimizer._evaluate_initialization(points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_initialization = points.copy()

    # Use the best initialization as starting point
    x0 = best_initialization.flatten()

    # Bounds for each coordinate: [0, 1] for all 14 points × 3 coordinates
    bounds = [(0, 1)] * 14 * 3

    # Objective function for optimization
    def objective(x):
        points = x.reshape(-1, 3)
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)

        if d_max == 0:
            return -np.inf

        return -(d_min / d_max)

    # Run adaptive differential evolution optimization
    result = optimizer._adaptive_differential_evolution(objective, bounds, maxiter=200)

    # Extract optimized points
    optimized_points = result.x.reshape(-1, 3)

    # Apply local refinement
    final_points = optimizer._local_refinement(optimized_points)

    # Final clipping to ensure bounds are respected
    final_points = np.clip(final_points, 0, 1)

    return final_points

# EVOLVE-BLOCK-END