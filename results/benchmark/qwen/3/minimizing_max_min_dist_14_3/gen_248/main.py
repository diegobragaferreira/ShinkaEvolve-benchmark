# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

class SpatialPointOptimizer:
    def __init__(self, n_points=14, dimensions=3, seed=42):
        self.n_points = n_points
        self.dimensions = dimensions
        self.seed = seed
        np.random.seed(seed)

    def _initialize_fibonacci_points(self):
        """Initialize points on a unit sphere using improved Fibonacci spiral method"""
        points = []
        # More sophisticated golden ratio for better distribution
        golden_ratio = (1 + np.sqrt(5)) / 2

        # Use a more uniform distribution approach
        for i in range(self.n_points):
            # Improved latitude calculation to minimize clustering
            phi = np.arccos(1 - 2 * (i / (self.n_points - 1)))
            # Add slight perturbation to avoid regular patterns
            theta = 2 * np.pi * i / golden_ratio + np.random.uniform(-0.1, 0.1)

            # Convert to Cartesian coordinates
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

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

    def _initialize_random_points(self):
        """Initialize random points in 3D space"""
        return np.random.rand(self.n_points, self.dimensions)

    def _initialize_voronoi_uniform_points(self):
        """Initialize points with good Voronoi uniformity using iterative approach"""
        # Start with random points
        points = np.random.rand(self.n_points, self.dimensions)

        # Simple iterative improvement: move points to increase minimum distance
        for _ in range(50):  # Limited iterations to avoid excessive computation
            distances = pdist(points)
            if len(distances) > 0:
                # Move each point away from its nearest neighbor
                for i in range(self.n_points):
                    # Find closest neighbor
                    dist_row = distances[i*(self.n_points-1):(i+1)*(self.n_points-1)]
                    if len(dist_row) > 0:
                        closest_idx = np.argmin(dist_row)
                        # Move point away from neighbor
                        if closest_idx < i:
                            neighbor = points[closest_idx]
                        else:
                            neighbor = points[closest_idx + 1]

                        direction = points[i] - neighbor
                        norm_dir = np.linalg.norm(direction)
                        if norm_dir > 1e-12:
                            points[i] += 0.01 * direction / norm_dir
                # Keep within bounds
                points = np.clip(points, 0, 1)

        return points

    def _perturb_points(self, points, magnitude=0.05):
        """Add small random perturbations to break symmetry"""
        noise = np.random.normal(0, magnitude, points.shape)
        perturbed = points + noise
        # Clip to maintain bounds
        perturbed = np.clip(perturbed, 0, 1)
        return perturbed

    def _project_to_unit_cube(self, points):
        """Project points to [0,1]^3 bounds"""
        return np.clip(points, 0, 1)

    def _compute_distance_ratio(self, points):
        """Compute the minimum/maximum distance ratio"""
        if len(points) < 2:
            return 0

        distances = pdist(points)
        if len(distances) == 0:
            return 0

        d_min = np.min(distances)
        d_max = np.max(distances)

        # Avoid division by zero
        if d_max <= 1e-12:
            return 0

        return d_min / d_max

    def _evaluate_initialization(self, points):
        """Fast evaluation of initialization quality with uniformity consideration"""
        # Primary metric: distance ratio
        ratio = self._compute_distance_ratio(points)

        # Secondary metric: estimate uniformity using pairwise distance variance
        # Lower variance in distances suggests more uniform distribution
        try:
            distances = pdist(points)
            if len(distances) > 0:
                # Better uniformity metric: inverse of distance variance normalized by mean
                distance_mean = np.mean(distances)
                if distance_mean > 1e-12:
                    distance_variance = np.var(distances)
                    # Uniformity score (higher is better): inverse of variance scaled by mean
                    uniformity_score = 1.0 / (distance_variance + 1e-12) * distance_mean
                    # Combine with ratio (weight 0.8 for ratio, 0.2 for uniformity)
                    return 0.8 * ratio + 0.2 * uniformity_score
                else:
                    return ratio
            else:
                return ratio
        except:
            return ratio

    def _objective_function(self, x):
        """Objective function that returns negative ratio to minimize"""
        points = x.reshape(-1, 3)

        # Ensure points are within bounds [0,1]^3
        points = self._project_to_unit_cube(points)

        # Compute distances
        distances = pdist(points)

        if len(distances) == 0:
            return -np.inf

        d_min = np.min(distances)
        d_max = np.max(distances)

        # Avoid division by zero
        if d_max <= 1e-12:
            return -np.inf

        # Return negative because we want to maximize the ratio
        return -(d_min / d_max)

    def _penalty_objective(self, x, penalty_weight=1e6):
        """Objective with penalty for boundary violations"""
        points = x.reshape(-1, 3)

        # Apply penalty for points outside bounds using vectorized operations
        penalty = 0
        penalty += np.sum(np.maximum(0, -points)**2) * penalty_weight  # Below 0
        penalty += np.sum(np.maximum(0, points - 1)**2) * penalty_weight  # Above 1

        # Original objective
        original_obj = self._objective_function(x)

        return original_obj + penalty

    def _adaptive_differential_evolution(self, objective_func, bounds, maxiter=300):
        """Enhanced differential evolution with adaptive parameters"""
        current_popsize = 20
        prev_best = -np.inf
        stagnation_count = 0
        improvement_threshold = 1e-8
        min_improvement = 1e-12
        recent_improvements = []
        improvement_window = 5

        for iteration in range(maxiter // 10):
            # More sophisticated adaptive population sizing
            if len(recent_improvements) >= improvement_window:
                # Calculate recent average improvement
                recent_avg_improvement = np.mean(recent_improvements[-improvement_window:])

                # If improvement is consistently high, reduce population size to exploit
                if recent_avg_improvement > improvement_threshold * 10 and current_popsize > 10:
                    current_popsize = max(10, current_popsize - 3)
                # If improvement is low or stagnant, increase population size to explore
                elif recent_avg_improvement < improvement_threshold * 0.1 and current_popsize < 30:
                    current_popsize = min(30, current_popsize + 5)
                # If moderate improvement, maintain population size
                elif recent_avg_improvement < improvement_threshold and current_popsize < 25:
                    current_popsize = min(25, current_popsize + 2)
                elif recent_avg_improvement > improvement_threshold * 5 and current_popsize > 15:
                    current_popsize = max(15, current_popsize - 2)

            try:
                result = differential_evolution(
                    objective_func,
                    bounds,
                    seed=self.seed + iteration,
                    maxiter=10,
                    popsize=current_popsize,
                    tol=1e-12,
                    mutation=(0.5, 1.0),
                    recombination=0.9,
                    disp=False
                )
            except Exception:
                # Fallback to smaller population if needed
                try:
                    result = differential_evolution(
                        objective_func,
                        bounds,
                        seed=self.seed + iteration,
                        maxiter=10,
                        popsize=max(5, current_popsize - 5),
                        tol=1e-12,
                        mutation=(0.5, 1.0),
                        recombination=0.9,
                        disp=False
                    )
                except Exception:
                    # Last resort - use basic differential evolution
                    result = differential_evolution(
                        objective_func,
                        bounds,
                        seed=self.seed + iteration,
                        maxiter=10,
                        popsize=10,
                        tol=1e-12,
                        mutation=(0.5, 1.0),
                        recombination=0.7,
                        disp=False
                    )

            # Check for improvement
            current_best = -result.fun
            improvement = current_best - prev_best

            recent_improvements.append(improvement)
            if len(recent_improvements) > improvement_window:
                recent_improvements.pop(0)

            # Early stopping if improvement is minimal
            if len(recent_improvements) == improvement_window and all(abs(impr) < min_improvement for impr in recent_improvements):
                break

            if improvement > improvement_threshold:
                stagnation_count = 0
            else:
                stagnation_count += 1

            prev_best = current_best

        return result

    def _local_refinement(self, points):
        """Apply enhanced local refinement to improve final solution"""
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

            # Phase 1: Coarse refinement with relaxed tolerances
            result_refine = minimize(
                objective_local,
                x0_refine,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-9, 'gtol': 1e-9},
                tol=1e-9
            )

            refined_points = result_refine.x.reshape(-1, 3)
            refined_points = self._project_to_unit_cube(refined_points)

            # Phase 2: Fine refinement with stricter tolerances
            result_refine2 = minimize(
                objective_local,
                refined_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-12, 'gtol': 1e-12},
                tol=1e-12
            )

            refined_points = result_refine2.x.reshape(-1, 3)
            refined_points = self._project_to_unit_cube(refined_points)
            return refined_points
        except Exception:
            return points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Initialize optimizer
    optimizer = SpatialPointOptimizer(n_points=14, dimensions=3, seed=42)

    # Generate multiple initialization strategies
    strategies = []

    # Strategy 1: Spherical Fibonacci points
    fib_points = optimizer._initialize_fibonacci_points()
    fib_points = (fib_points + 1) / 2  # Normalize to [0,1]^3
    strategies.append(("fibonacci", fib_points))

    # Strategy 2: Cube grid points
    cube_points = optimizer._initialize_cube_grid_points()
    strategies.append(("cube_grid", cube_points))

    # Strategy 3: Random points
    random_points = optimizer._initialize_random_points()
    strategies.append(("random", random_points))

    # Strategy 4: Perturbed spherical points
    perturbed_points = optimizer._perturb_points(fib_points, 0.05)
    strategies.append(("perturbed", perturbed_points))

    # Strategy 5: Optimized version of spherical points
    optimized_fib_points = optimizer._perturb_points(fib_points, 0.02)
    strategies.append(("optimized_fib", optimized_fib_points))

    # Strategy 6: Voronoi uniform points
    voronoi_uniform_points = optimizer._initialize_voronoi_uniform_points()
    strategies.append(("voronoi_uniform", voronoi_uniform_points))

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

    # Run adaptive differential evolution optimization
    best_result = None
    best_ratio = -np.inf

    # Try 3 different random seeds for better exploration
    for seed_val in [42, 123, 456]:
        np.random.seed(seed_val)

        # Use adaptive differential evolution
        result = optimizer._adaptive_differential_evolution(
            optimizer._penalty_objective,
            bounds,
            maxiter=200
        )

        # Check if this result is better
        if -result.fun > best_ratio:
            best_ratio = -result.fun
            best_result = result

    # Extract optimized points
    optimized_points = best_result.x.reshape(-1, 3)

    # Apply local refinement
    final_points = optimizer._local_refinement(optimized_points)

    # Final clipping to ensure bounds are respected
    final_points = optimizer._project_to_unit_cube(final_points)

    return final_points

# EVOLVE-BLOCK-END