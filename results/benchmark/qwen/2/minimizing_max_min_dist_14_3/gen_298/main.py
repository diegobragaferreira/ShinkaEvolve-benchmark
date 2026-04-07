# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from numba import jit
import time

@jit(nopython=True)
def compute_min_max_ratio_numba(points):
    """Optimized distance computation using numba"""
    n = points.shape[0]
    min_dist = np.inf
    max_dist = 0.0

    for i in range(n):
        for j in range(i+1, n):
            # Compute squared distance to avoid sqrt computation
            dist_sq = (points[i,0]-points[j,0])**2 + (points[i,1]-points[j,1])**2 + (points[i,2]-points[j,2])**2
            dist = np.sqrt(dist_sq)
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist

    return min_dist, max_dist

def fibonacci_sphere(n: int) -> np.ndarray:
    """
    Generate n points distributed as evenly as possible on a unit sphere
    using Fibonacci spiral method.
    """
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle

    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

def compute_min_max_ratio(points: np.ndarray) -> tuple:
    """
    Compute the minimum and maximum distances between all pairs of points,
    and return their ratio.
    """
    if len(points) < 2:
        return 0.0, 0.0, 0.0

    # Use numba-optimized version
    min_distance, max_distance = compute_min_max_ratio_numba(points)

    # Avoid division by zero
    if max_distance == 0:
        ratio = 0.0
    else:
        ratio = min_distance / max_distance

    return min_distance, max_distance, ratio

def perturb_points_smart(points: np.ndarray, temperature: float, current_ratio: float) -> np.ndarray:
    """
    Apply smart perturbations based on distance distribution analysis.
    """
    # Create a copy of the points
    new_points = points.copy()

    # Analyze current distribution to decide which point to perturb
    # Get pairwise distances for analysis
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)

    # Find points that are closest to other points (potential bottlenecks)
    min_distances = np.min(distances, axis=1)
    max_distances = np.max(distances, axis=1)
    avg_distances = np.mean(distances, axis=1)

    # Calculate Voronoi entropy for distribution assessment
    from scipy.spatial import SphericalVoronoi
    try:
        sv = SphericalVoronoi(points)
        areas = sv.calculate_areas()
        # Normalize areas
        areas = areas / np.sum(areas)
        # Entropy calculation
        entropy = -np.sum(areas * np.log(areas + 1e-10))
        # Use entropy to inform perturbation strategy
        # Low entropy = more uniform, high entropy = less uniform
        entropy_factor = max(0.1, min(2.0, 1.0 + entropy/10.0))  # Normalize and bound
    except:
        entropy_factor = 1.0

    # Select multiple points to perturb for better exploration
    num_to_perturb = max(1, min(4, len(points) // 5))  # Perturb 1-4 points

    # More sophisticated selection:
    # 1. Prefer points that are in tight clusters (low min distance)
    # 2. Prefer points that are far from average (outliers)
    # 3. Use entropy-based weighting to balance uniformity and gap-filling

    # Weight points based on multiple criteria
    weights = np.zeros(len(points))

    # Cluster penalty: points with very small minimum distances get higher weight
    cluster_penalty = 1.0 / (min_distances + 1e-8)  # Avoid division by zero
    cluster_penalty = cluster_penalty / np.max(cluster_penalty)  # Normalize

    # Outlier bonus: points far from average get bonus
    outlier_bonus = np.abs(avg_distances - np.median(avg_distances))
    outlier_bonus = outlier_bonus / (np.max(outlier_bonus) + 1e-8)  # Normalize

    # Combine weights (weighted sum)
    weights = 0.6 * cluster_penalty + 0.4 * outlier_bonus

    # Add entropy factor - more perturbations when distribution is uneven
    weights = weights * entropy_factor

    # Normalize weights to probabilities
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    else:
        weights = np.ones(len(points)) / len(points)

    # Sample points to perturb based on weights
    target_indices = np.random.choice(len(points), size=num_to_perturb, replace=False, p=weights)

    # Apply perturbations with adaptive scaling
    for target_idx in target_indices:
        # Determine perturbation strength based on local configuration
        # Points that are too close get larger perturbations
        if min_distances[target_idx] < avg_distances[target_idx] * 0.3:
            perturbation_scale = 0.05 * temperature * entropy_factor  # Larger perturbation
        elif min_distances[target_idx] > avg_distances[target_idx] * 1.5:
            perturbation_scale = 0.01 * temperature  # Smaller perturbation for well-separated
        else:
            perturbation_scale = 0.03 * temperature * entropy_factor  # Medium perturbation

        # Generate random perturbation in tangent plane
        random_vec = np.random.randn(3)
        # Project onto sphere surface normal (tangent plane)
        normal_vec = new_points[target_idx]
        tangent_vec = random_vec - np.dot(random_vec, normal_vec) * normal_vec
        # Normalize tangent vector
        tangent_norm = np.linalg.norm(tangent_vec)
        if tangent_norm > 1e-10:
            tangent_vec = tangent_vec / tangent_norm
        # Apply perturbation with adaptive strength
        delta = tangent_vec * np.random.normal(0, perturbation_scale)

        # Add perturbation to selected point
        new_points[target_idx] += delta

    # Project back to unit sphere
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    # Handle case where norm might be zero (shouldn't happen but safety check)
    norms = np.where(norms == 0, 1, norms)
    new_points = new_points / norms

    return new_points

def adaptive_cooling(initial_temp, iteration, max_iterations, ratio_history):
    """
    Adaptive cooling schedule that adjusts based on convergence
    """
    # Base cooling rate
    base_cooling = 0.9995

    # Check recent convergence
    if len(ratio_history) > 10:
        recent_improvement = ratio_history[-1] - ratio_history[-10]
        if recent_improvement < 1e-8:
            # Slow improvement, cool faster
            return base_cooling * 1.05
        elif recent_improvement > 1e-6:
            # Fast improvement, cool slower
            return base_cooling * 0.95

    return base_cooling

def local_refinement_step(points, max_iterations=50):
    """
    Apply a simple local refinement using projected gradient descent
    to improve the current configuration around the best solution found.
    """
    from scipy.optimize import minimize
    from scipy.spatial.distance import cdist

    def objective(x_flat):
        # Reshape flat array back to points
        points_test = x_flat.reshape(-1, 3)

        # Compute pairwise distances
        distances = cdist(points_test, points_test)
        np.fill_diagonal(distances, np.inf)

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        if max_dist == 0:
            return 0.0
        return -min_dist / max_dist  # Negative because we want to maximize

    def constraint_func(x_flat):
        # Ensure all points lie on the unit sphere
        points_test = x_flat.reshape(-1, 3)
        norms = np.linalg.norm(points_test, axis=1)
        return norms - 1.0  # Should equal zero for unit sphere

    # Flatten points for optimization
    x0 = points.flatten()

    # Define constraints for unit sphere
    cons = {'type': 'eq', 'fun': constraint_func}

    try:
        # Use L-BFGS-B optimizer with bounds for better stability
        result = minimize(objective, x0, method='SLSQP', constraints=cons,
                         options={'maxiter': max_iterations, 'ftol': 1e-8, 'gtol': 1e-8})

        if result.success:
            refined_points = result.x.reshape(-1, 3)
            return refined_points
    except:
        pass

    return points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Set seed for reproducibility
    np.random.seed(42)

    # Multi-start optimization with multiple initialization strategies
    best_points = None
    best_ratio = -np.inf
    best_min_dist = 0
    best_max_dist = 0

    # Try multiple initialization strategies
    num_starts = 15  # Increased from 10 to 15 for better exploration
    for start_idx in range(num_starts):
        # Alternate between Fibonacci and random initialization
        if start_idx % 3 == 0:
            # Fibonacci sphere initialization
            np.random.seed(start_idx)
            points = fibonacci_sphere(14)
        elif start_idx % 3 == 1:
            # Random initialization
            np.random.seed(start_idx)
            points = np.random.randn(14, 3)
            # Normalize to unit sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            points = points / norms
        else:
            # Another Fibonacci variant with different seed
            np.random.seed(start_idx * 10)
            points = fibonacci_sphere(14)
            # Add small random jitter to break symmetry
            points += np.random.normal(0, 0.01, points.shape)

        # Optimization parameters
        max_iterations = 150000  # Increased for better convergence
        initial_temperature = 1.0
        cooling_rate = 0.9995
        min_temperature = 0.0001

        # Track best solution
        current_best_points = points.copy()
        current_best_min_dist, current_best_max_dist, current_best_ratio = compute_min_max_ratio(points)

        # Current state
        current_points = points.copy()
        current_min_dist, current_max_dist, current_ratio = current_best_ratio, current_best_max_dist, current_best_ratio

        # Track ratio history for adaptive cooling
        ratio_history = [current_ratio]

        # Simulated Annealing
        temp = initial_temperature
        last_improvement_iter = 0

        for iteration in range(max_iterations):
            # Perturb the current solution
            new_points = perturb_points_smart(current_points, temp, current_ratio)

            # Compute new ratio
            new_min_dist, new_max_dist, new_ratio = compute_min_max_ratio(new_points)

            # Accept or reject the new solution using Metropolis criterion
            if new_ratio > current_ratio:
                # Always accept better solutions
                current_points = new_points
                current_ratio = new_ratio
                current_min_dist = new_min_dist
                current_max_dist = new_max_dist

                # Update best solution if this is better
                if new_ratio > current_best_ratio:
                    current_best_points = new_points.copy()
                    current_best_ratio = new_ratio
                    current_best_min_dist = new_min_dist
                    current_best_max_dist = new_max_dist
                    last_improvement_iter = iteration
                    ratio_history.append(new_ratio)
            else:
                # Accept worse solutions with probability based on temperature
                if temp > 0:  # Avoid division by zero
                    acceptance_prob = np.exp((new_ratio - current_ratio) / temp)
                    if np.random.rand() < acceptance_prob:
                        current_points = new_points
                        current_ratio = new_ratio
                        current_min_dist = new_min_dist
                        current_max_dist = new_max_dist
                        ratio_history.append(new_ratio)

            # Apply adaptive cooling
            temp = max(temp * adaptive_cooling(initial_temperature, iteration, max_iterations, ratio_history), min_temperature)

            # Periodic local refinement (every 15000 iterations instead of 20000)
            if iteration % 15000 == 0 and iteration > 0:
                refined_points = local_refinement_step(current_best_points)
                _, _, refined_ratio = compute_min_max_ratio(refined_points)
                if refined_ratio > current_best_ratio:
                    current_best_points = refined_points.copy()
                    current_best_ratio = refined_ratio
                    # Update the current points to reflect the refinement
                    current_points = refined_points.copy()
                    # Update best distances
                    _, current_best_max_dist, _ = compute_min_max_ratio(refined_points)
                    current_best_min_dist = np.min(cdist(refined_points, refined_points))
                    current_min_dist = current_best_min_dist
                    current_max_dist = current_best_max_dist

            # Early stopping if no improvement in a long time
            if iteration - last_improvement_iter > 25000:
                break

        # Update global best if this run was better
        if current_best_ratio > best_ratio:
            best_ratio = current_best_ratio
            best_points = current_best_points.copy()
            best_min_dist = current_best_min_dist
            best_max_dist = current_best_max_dist

    # Final local refinement on the best solution found
    if best_points is not None:
        final_refined = local_refinement_step(best_points, max_iterations=100)
        _, _, final_ratio = compute_min_max_ratio(final_refined)
        if final_ratio > best_ratio:
            best_points = final_refined

    # Ensure the result is properly normalized
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    best_points = best_points / norms

    return best_points

# EVOLVE-BLOCK-END