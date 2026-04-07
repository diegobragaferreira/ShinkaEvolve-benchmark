# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import math
import random

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def compute_min_max_ratio(points: np.ndarray) -> float:
        """Compute the ratio of minimum to maximum distances."""
        distances = pdist(points)
        return np.min(distances) / np.max(distances)

    def fibonacci_sphere(n: int) -> np.ndarray:
        """Generate n points on a sphere using Fibonacci spiral method."""
        points = []
        phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle

        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def initialize_points(n: int) -> np.ndarray:
        """Initialize points using Fibonacci sphere and perturbation."""
        # Start with Fibonacci sphere points
        points = fibonacci_sphere(n)

        # Add some randomness to avoid perfect symmetry issues
        np.random.seed(42)
        points += 0.01 * np.random.randn(n, 3)

        # Normalize to unit sphere
        points = points / np.linalg.norm(points, axis=1, keepdims=True)

        return points

    def perturb_point(point: np.ndarray, step_size: float) -> np.ndarray:
        """Perturb a single point on the unit sphere."""
        # Generate random perturbation
        delta = np.random.randn(3)
        # Project to tangent plane and normalize
        delta = delta - np.dot(delta, point) * point
        delta = delta / np.linalg.norm(delta)

        # Apply perturbation
        new_point = point + step_size * delta
        # Project back to sphere
        new_point = new_point / np.linalg.norm(new_point)

        return new_point

    def simulated_annealing():
        """Run simulated annealing optimization."""
        # Initialize
        points = initialize_points(14)
        current_ratio = compute_min_max_ratio(points)

        # Parameters
        T = 0.1  # Initial temperature
        Tmin = 1e-6  # Minimum temperature
        alpha = 0.999  # Cooling rate
        max_iter = 10000  # Maximum iterations
        step_size = 0.01  # Initial step size

        # Track best solution
        best_points = points.copy()
        best_ratio = current_ratio

        # Adaptive cooling parameters
        last_improvement = 0
        patience = 100

        for iteration in range(max_iter):
            # Adaptive cooling based on recent progress
            if iteration - last_improvement > patience:
                T = max(Tmin, T * 0.95)  # Cool faster if no improvement

            # Select random point to perturb
            idx = random.randint(0, 13)

            # Save current point
            old_point = points[idx].copy()

            # Perturb selected point
            points[idx] = perturb_point(points[idx], step_size)

            # Compute new ratio
            new_ratio = compute_min_max_ratio(points)

            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio:
                current_ratio = new_ratio
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = points.copy()
                    last_improvement = iteration
            else:
                # Calculate acceptance probability
                delta = new_ratio - current_ratio
                acceptance_prob = math.exp(delta / T)

                if random.random() < acceptance_prob:
                    current_ratio = new_ratio
                    if new_ratio > best_ratio:
                        best_ratio = new_ratio
                        best_points = points.copy()
                        last_improvement = iteration
                else:
                    # Revert change
                    points[idx] = old_point

            # Periodic local refinement
            if iteration % 500 == 0 and iteration > 0:
                # Local refinement with gradient-based approach
                refined_points = points.copy()
                for _ in range(10):
                    for i in range(14):
                        # Simple gradient-based refinement (steepest ascent)
                        old_ratio = compute_min_max_ratio(refined_points)
                        old_point = refined_points[i].copy()

                        # Small perturbations to estimate gradient
                        grad = np.zeros(3)
                        for j in range(3):
                            eps = 1e-4
                            test_points = refined_points.copy()
                            test_points[i, j] += eps
                            test_points[i] = test_points[i] / np.linalg.norm(test_points[i])
                            new_ratio = compute_min_max_ratio(test_points)
                            grad[j] = (new_ratio - old_ratio) / eps

                        # Move along gradient
                        if np.linalg.norm(grad) > 1e-10:
                            refined_points[i] = refined_points[i] + 0.1 * grad
                            refined_points[i] = refined_points[i] / np.linalg.norm(refined_points[i])

                    # Update if improved
                    new_ratio = compute_min_max_ratio(refined_points)
                    if new_ratio > compute_min_max_ratio(points):
                        points = refined_points.copy()
                        current_ratio = new_ratio
                        if new_ratio > best_ratio:
                            best_ratio = new_ratio
                            best_points = points.copy()
                            last_improvement = iteration

            # Reduce step size over time
            step_size = max(0.001, step_size * 0.9999)

            # Cool down
            T = max(Tmin, T * alpha)

            # Early stopping if we're not improving
            if iteration - last_improvement > 5000:
                break

        return best_points

    # Run optimization
    return simulated_annealing()

# EVOLVE-BLOCK-END