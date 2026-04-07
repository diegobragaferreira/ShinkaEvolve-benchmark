# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
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
        return np.min(distances) / np.max(distances) if np.max(distances) > 0 else 0

    def fibonacci_sphere(n):
        """Generate n points distributed approximately uniformly on a sphere."""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        for i in range(n):
            # Distribute points more evenly
            z = 1 - (i / (n - 1)) * 2  # z goes from 1 to -1
            radius = np.sqrt(1 - z*z)

            # Better distribution using Fibonacci sequence
            theta = np.arctan2(np.sin(i * 2 * np.pi / phi), np.cos(i * 2 * np.pi / phi))
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)

    def project_to_sphere(points):
        """Project points onto unit sphere."""
        norms = np.linalg.norm(points, axis=1)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return points / norms[:, np.newaxis]

    def simulated_annealing_optimization(initial_points, max_iter=10000):
        """Optimize points using improved simulated annealing with local search."""
        current_points = initial_points.copy()
        current_points = project_to_sphere(current_points)

        best_points = current_points.copy()
        best_ratio = compute_min_max_ratio(current_points)

        # Enhanced adaptive cooling schedule with better control
        temperature = 0.1
        cooling_rate = 0.9995
        min_temp = 1e-6

        # Track convergence for adaptive cooling
        last_improvement = 0
        stagnation_counter = 0
        max_stagnation = 1000

        # For tracking ratios to detect convergence patterns
        recent_ratios = []

        for iteration in range(max_iter):
            # Store current configuration
            old_points = current_points.copy()
            old_ratio = compute_min_max_ratio(current_points)

            # Select random point to perturb
            idx = np.random.randint(len(current_points))

            # Smart perturbation size that adapts to current solution quality
            # Larger perturbations early, smaller later
            base_perturbation = 0.02 * (1 - iteration / max_iter) + 0.001
            perturbation_size = base_perturbation * (1 + 0.5 * np.random.random())

            # Use a more informed direction for perturbation
            perturbation = np.random.normal(0, perturbation_size, 3)

            # For better sphere constraint handling, use the closest point approach
            # but also consider the influence of nearby points
            current_point = current_points[idx]

            # Tangent plane projection: remove component parallel to radius vector
            projection_factor = np.dot(perturbation, current_point)
            perturbation_tangent = perturbation - projection_factor * current_point

            # Apply perturbation
            current_points[idx] += perturbation_tangent

            # Project back to sphere
            current_points[idx] = project_to_sphere(current_points[idx:idx+1])[0]

            # Compute new ratio
            new_ratio = compute_min_max_ratio(current_points)

            # Accept or reject based on Metropolis criterion
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_points = current_points.copy()
                last_improvement = iteration
                stagnation_counter = 0
                recent_ratios.clear()  # Reset recent ratios on improvement
            elif np.random.random() < np.exp((new_ratio - old_ratio) / temperature):
                # Accept worse solution with probability
                pass  # Keep the new configuration
            else:
                # Revert to previous configuration
                current_points = old_points

            # Enhanced adaptive cooling logic
            if new_ratio > old_ratio:
                # Improvement occurred
                recent_ratios.append(new_ratio)
                if len(recent_ratios) > 10:
                    recent_ratios.pop(0)
            else:
                # No improvement, track stagnation
                stagnation_counter += 1
                if stagnation_counter > max_stagnation:
                    # If no improvement for a while, cool faster and diversify
                    temperature = max(min_temp, temperature * 0.9)
                    stagnation_counter = 0
                    # Add some random perturbations
                    for i in range(len(current_points)):
                        if np.random.random() < 0.15:  # 15% chance per point
                            perturbation = np.random.normal(0, 0.005, 3)
                            current_point = current_points[i]
                            projection_factor = np.dot(perturbation, current_point)
                            perturbation_tangent = perturbation - projection_factor * current_point
                            current_points[i] += perturbation_tangent
                            current_points[i] = project_to_sphere(current_points[i:i+1])[0]
                else:
                    # Normal cooling
                    temperature = max(min_temp, temperature * cooling_rate)

            # Periodic local search refinement (every 500 iterations)
            if iteration % 500 == 0 and iteration > 0:
                # Perform simple local search by trying small adjustments
                local_points = current_points.copy()
                for i in range(len(local_points)):
                    for j in range(3):  # Try small adjustments to each coordinate
                        old_val = local_points[i, j]
                        # Try small positive and negative perturbations
                        for delta in [-0.001, 0.001]:
                            local_points[i, j] = old_val + delta
                            local_points[i] = project_to_sphere(local_points[i:i+1])[0]
                            new_ratio = compute_min_max_ratio(local_points)
                            if new_ratio > best_ratio:
                                best_ratio = new_ratio
                                best_points = local_points.copy()
                            else:
                                local_points[i, j] = old_val  # Revert

        return best_points, best_ratio

    def objective_function(x_flat: np.ndarray) -> float:
        """Objective function for optimization (negative of min-max ratio)."""
        points = x_flat.reshape(-1, 3)
        return -compute_min_max_ratio(points)

    # Start with Fibonacci sphere placement for better initial distribution
    initial_points = fibonacci_sphere(14)

    # Normalize to unit sphere
    initial_points = project_to_sphere(initial_points)

    # Optimize using simulated annealing
    optimized_points, final_ratio = simulated_annealing_optimization(initial_points, 10000)

    # Try several random restarts with different initializations
    best_points = optimized_points.copy()
    best_ratio = final_ratio

    # Additional restarts with different Fibonacci sphere seeds
    for restart in range(5):
        # Create a slightly different Fibonacci sphere by offsetting the indices
        np.random.seed(restart)
        points_offset = fibonacci_sphere(14)
        points_offset = project_to_sphere(points_offset)

        # Optimize this initialization
        optimized_points_restart, ratio = simulated_annealing_optimization(points_offset, 5000)

        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points_restart

    # Final refinement using L-BFGS
    x0 = best_points.flatten()
    try:
        result = minimize(
            objective_function,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-9}
        )

        refined_points = result.x.reshape(-1, 3)
        refined_points = project_to_sphere(refined_points)

        ratio = compute_min_max_ratio(refined_points)
        if ratio > best_ratio:
            best_points = refined_points
    except:
        pass

    return best_points

# EVOLVE-BLOCK-END