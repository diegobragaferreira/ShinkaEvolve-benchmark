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
        # Avoid division by zero with a small epsilon
        eps = 1e-12
        norms = np.where(norms < eps, 1.0, norms)
        return points / norms[:, np.newaxis]

    def smart_perturbation_selection(current_points, current_ratio):
        """Select point for perturbation based on distance analysis."""
        # Analyze distances to determine which points might benefit most from perturbation
        distances = pdist(current_points)
        # Get upper triangular part (excluding diagonal)
        triu_indices = np.triu_indices_from(distances, k=1)
        distances_upper = distances[triu_indices]
        
        # Find points that are either very close or very far
        min_dist = np.min(distances_upper)
        max_dist = np.max(distances_upper)
        mean_dist = np.mean(distances_upper)
        
        # Select points that are either:
        # 1. Very close to another point (potential for expansion)
        # 2. Very far from others (potential for contraction)
        close_threshold = mean_dist * 0.5  # Points closer than half mean distance
        far_threshold = mean_dist * 1.5   # Points farther than 1.5 times mean distance
        
        # Analyze each point's neighbor distances
        point_distances = []
        for i in range(len(current_points)):
            # Get distances from point i to all other points
            dists = []
            for j in range(len(current_points)):
                if i != j:
                    # Calculate distance without recomputing full matrix
                    dist = np.sqrt(np.sum((current_points[i] - current_points[j])**2))
                    dists.append(dist)
            avg_dist = np.mean(dists) if dists else 1.0
            point_distances.append(avg_dist)
        
        # Choose point to perturb based on statistical properties
        point_distances = np.array(point_distances)
        # Prefer points with extreme average distances
        if np.std(point_distances) > 0:
            # Weighted selection based on distance variance
            weights = 1.0 / (point_distances + 1e-8)  # Inverse of distances
            weights = weights / np.sum(weights)
            idx = np.random.choice(len(current_points), p=weights)
        else:
            # Fallback to random selection
            idx = np.random.randint(len(current_points))
            
        return idx

    def enhanced_simulated_annealing_optimization(initial_points, max_iter=15000):
        """Optimize points using enhanced simulated annealing with adaptive cooling."""
        current_points = initial_points.copy()
        current_points = project_to_sphere(current_points)

        best_points = current_points.copy()
        best_ratio = compute_min_max_ratio(current_points)

        # Enhanced adaptive cooling schedule
        temperature = 0.1
        cooling_rate = 0.9997  # Slightly more aggressive cooling
        min_temp = 1e-6

        # Tracking variables for adaptive cooling
        last_improvement = 0
        stagnation_counter = 0
        max_stagnation = 500  # Reduced from 1000 for more responsive adaptation
        
        # Track recent ratios for convergence detection
        recent_ratios = []
        ratio_window_size = 10
        
        for iteration in range(max_iter):
            # Store current configuration
            old_points = current_points.copy()
            old_ratio = compute_min_max_ratio(current_points)

            # Select random point to perturb with smart selection
            idx = smart_perturbation_selection(current_points, old_ratio)

            # Adaptive perturbation size based on iteration and solution quality
            base_perturbation = 0.015 * (1 - iteration / max_iter) + 0.001
            perturbation_size = base_perturbation * (0.8 + 0.4 * np.random.random())
            
            # Generate small random perturbation in tangent plane (better for sphere)
            perturbation = np.random.normal(0, perturbation_size, 3)

            # Project perturbation to tangent plane of the sphere at current point
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
                if len(recent_ratios) > ratio_window_size:
                    recent_ratios.pop(0)
            else:
                # No improvement, track stagnation
                stagnation_counter += 1
                
                # Add diversity if stagnating
                if stagnation_counter > max_stagnation * 0.5:
                    # Occasionally add a small random perturbation to maintain diversity
                    for i in range(len(current_points)):
                        if np.random.random() < 0.05:  # 5% chance per point
                            perturbation = np.random.normal(0, 0.001, 3)
                            current_point = current_points[i]
                            projection_factor = np.dot(perturbation, current_point)
                            perturbation_tangent = perturbation - projection_factor * current_point
                            current_points[i] += perturbation_tangent
                            current_points[i] = project_to_sphere(current_points[i:i+1])[0]
                
                if stagnation_counter > max_stagnation:
                    # If no improvement for a while, cool faster and diversify
                    temperature = max(min_temp, temperature * 0.9)
                    stagnation_counter = 0
                    # Add more significant perturbations
                    for i in range(len(current_points)):
                        if np.random.random() < 0.1:  # 10% chance per point
                            perturbation = np.random.normal(0, 0.005, 3)
                            current_point = current_points[i]
                            projection_factor = np.dot(perturbation, current_point)
                            perturbation_tangent = perturbation - projection_factor * current_point
                            current_points[i] += perturbation_tangent
                            current_points[i] = project_to_sphere(current_points[i:i+1])[0]
                else:
                    # Normal cooling
                    temperature = max(min_temp, temperature * cooling_rate)

            # Occasionally rescale to maintain exploration
            if iteration % 2000 == 0 and iteration > 0:
                # Occasionally reset points to maintain exploration
                current_points = project_to_sphere(current_points)

        return best_points, best_ratio

    # Start with Fibonacci sphere placement for better initial distribution
    initial_points = fibonacci_sphere(14)

    # Normalize to unit sphere
    initial_points = project_to_sphere(initial_points)

    # Optimize using enhanced simulated annealing
    optimized_points, final_ratio = enhanced_simulated_annealing_optimization(initial_points, 15000)

    # Try several random restarts with different initializations
    best_points = optimized_points.copy()
    best_ratio = final_ratio

    # Additional restarts with more diverse initialization strategies
    for restart in range(10):  # Increase from 5 to 10 restarts
        # Strategy 1: Different Fibonacci sphere seed
        np.random.seed(restart * 17)  # Different prime for seeding
        points_offset = fibonacci_sphere(14)
        points_offset = project_to_sphere(points_offset)
        
        # Optimize this initialization
        optimized_points_restart, ratio = enhanced_simulated_annealing_optimization(points_offset, 7000)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points_restart

        # Strategy 2: Slight perturbation of best solution so far
        if restart % 3 == 0:  # Every third restart, try perturbing best solution
            perturbed = best_points + 0.02 * np.random.randn(*best_points.shape)
            perturbed = project_to_sphere(perturbed)
            optimized_points_pert, ratio = enhanced_simulated_annealing_optimization(perturbed, 5000)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points_pert

    return best_points

# EVOLVE-BLOCK-END