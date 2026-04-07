# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
from scipy.stats import qmc
import time


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def fibonacci_sphere(n):
        """Generate n points on a sphere using Fibonacci spiral method"""
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle

        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def sobol_points(n, seed=42):
        """Generate n points using Sobol sequence for better space-filling"""
        sampler = qmc.Sobol(d=3, seed=seed)
        points = sampler.random(n)
        return points

    def objective_with_tightening(x, iteration_step=0, max_steps=100):
        # Reshape x into 14 points in 3D
        points = x.reshape(-1, 3)

        # Calculate pairwise distances
        distances = pdist(points)

        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Adaptive constraint tightening - gradually move toward tighter constraints
        # Start with looser bounds (allowing larger diameters) and tighten over time
        if iteration_step < max_steps:
            # Linear decay from 2.8 to 2.0 diameter constraint
            max_diameter = 2.8 - (iteration_step / max_steps) * 0.8
            # Only apply the constraint if the current max distance exceeds the target
            if d_max > max_diameter:
                # Penalize solutions that exceed the constraint
                penalty = (d_max - max_diameter) * 1000
                return -d_min / d_max + penalty
            else:
                # No penalty when within constraint
                pass

        # Return negative ratio since we want to maximize
        # We add a small epsilon to avoid division by zero
        if d_max < 1e-10:
            return -1e10
        return -d_min / d_max

    def enhanced_initialization(n_points, seed=42):
        """Enhanced initialization using multiple strategies"""
        np.random.seed(seed)

        # Strategy 1: Fibonacci points on sphere
        fib_points = fibonacci_sphere(n_points)

        # Strategy 2: Sobol points for better space filling
        sobol_points_generated = sobol_points(n_points, seed=seed+1000)

        # Strategy 3: Random points
        random_points = np.random.rand(n_points, 3)

        # Use a weighted combination to get better diversity
        # Mix them with different weights to maintain good distribution
        mixed_points = (
            0.5 * fib_points +
            0.3 * sobol_points_generated +
            0.2 * random_points
        )

        # Normalize to unit cube [0,1]^3
        # First center around origin and scale appropriately
        mixed_points = mixed_points - np.mean(mixed_points, axis=0)
        max_coord = np.max(np.abs(mixed_points))
        if max_coord > 0:
            mixed_points = mixed_points / max_coord * 0.5
        # Then shift to [0,1]^3
        mixed_points = mixed_points + 0.5

        # Add controlled perturbation to break symmetry
        # Use smaller perturbation than before to maintain structure
        perturbation = np.random.normal(0, 0.02, mixed_points.shape)
        mixed_points += perturbation
        mixed_points = np.clip(mixed_points, 0, 1)

        return mixed_points

    # Multi-start optimization with different initializations
    best_result = None
    best_ratio = -np.inf

    # Try multiple initializations with different seeds and strategies
    for seed in [42, 123, 456, 789]:
        np.random.seed(seed)

        # Generate enhanced initial points
        initial_points = enhanced_initialization(14, seed=seed)

        # Flatten for optimization
        x0 = initial_points.flatten()

        # Set up bounds for optimization (0 to 1 for all coordinates)
        bounds = [(0.0, 1.0)] * 14 * 3

        # First stage: Differential Evolution for global search with reduced iterations for speed
        try:
            de_result = differential_evolution(
                objective_with_tightening,
                bounds,
                seed=seed,
                maxiter=250,  # Increased for better exploration
                popsize=12,   # Larger population for better diversity
                tol=1e-8,
                mutation=(0.5, 1.0),
                recombination=0.7,
                disp=False
            )

            # Second stage: Local refinement with L-BFGS-B for better convergence
            refined_result = minimize(
                objective_with_tightening,
                de_result.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 200},  # Increased maxiter
                callback=None
            )

            # Evaluate final result
            final_points = refined_result.x.reshape(-1, 3)
            distances = pdist(final_points)
            d_min = np.min(distances)
            d_max = np.max(distances)

            if d_max > 1e-10:
                ratio = d_min / d_max
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_result = refined_result.x.copy()

        except Exception as e:
            continue

    # If no good result was found, fallback to simple initialization with DE only
    if best_result is None:
        np.random.seed(42)
        points = np.random.rand(14, 3) * 0.8 + 0.1
        x0 = points.flatten()
        bounds = [(0.0, 1.0)] * 14 * 3

        result = differential_evolution(
            objective_with_tightening,
            bounds,
            seed=42,
            maxiter=250,  # Increased for better exploration
            popsize=12,   # Larger population
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            disp=False
        )
        return result.x.reshape(-1, 3)

    return best_result.reshape(-1, 3)


# EVOLVE-BLOCK-END