# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from numba import jit
import time
import random

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

def project_to_unit_sphere(points):
    """Project points to the unit sphere"""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Handle case where norm might be zero
    norms = np.where(norms == 0, 1, norms)
    return points / norms

def spherical_perturb(points: np.ndarray, target_point_idx: int, temperature: float) -> np.ndarray:
    """
    Apply perturbation on the tangent plane of the unit sphere at target point,
    ensuring resulting point stays on unit sphere.
    """
    # Create a copy of the points
    new_points = points.copy()
    
    # Get the target point
    target_point = points[target_point_idx]
    
    # Generate random perturbation in tangent plane
    # We generate a random vector and subtract its projection onto the normal
    perturbation = np.random.normal(0, 0.01 * temperature, 3)
    
    # Project the perturbation onto the tangent plane (orthogonal to the normal)
    # Normal is just the point itself on unit sphere
    normal = target_point
    proj = np.dot(perturbation, normal)
    tangent_perturbation = perturbation - proj * normal
    
    # Apply the perturbation
    new_points[target_point_idx] = target_point + tangent_perturbation
    
    # Project back to unit sphere
    new_points = project_to_unit_sphere(new_points)
    
    return new_points

def adaptive_perturbation_strategy(points: np.ndarray, current_ratio: float, temperature: float) -> np.ndarray:
    """
    Apply adaptive perturbation based on current configuration analysis.
    """
    # Analyze current distribution to decide which point to perturb
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    
    # Get minimum and maximum distances for analysis
    min_distances = np.min(distances, axis=1)
    max_distances = np.max(distances, axis=1)
    
    # Calculate average distance per point for reference
    avg_distances = np.mean(distances, axis=1)
    
    # Score points based on their potential impact
    scores = np.zeros(len(points))
    for i in range(len(points)):
        # Weight by how close they are to min vs max distances
        min_dist = min_distances[i]
        max_dist = max_distances[i]
        avg_dist = avg_distances[i]
        
        # Score based on being too close (helps increase min) or too far (helps decrease max)
        if min_dist < avg_dist * 0.5:  # Very close - prioritize increasing their distance
            scores[i] = -min_dist
        elif max_dist > avg_dist * 2.0:  # Very far - prioritize decreasing their distance
            scores[i] = max_dist
        else:  # Medium distance - less critical
            scores[i] = 0
    
    # Choose point to perturb based on scores (higher score means more important)
    if np.sum(np.abs(scores)) > 0:
        # Use weighted probability based on scores
        probs = np.abs(scores)
        probs = probs / np.sum(probs)
        target_idx = np.random.choice(len(points), p=probs)
    else:
        # Fallback to random selection
        target_idx = np.random.randint(len(points))
    
    return spherical_perturb(points, target_idx, temperature)

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

def generate_initial_points(num_strategies: int = 5) -> list:
    """Generate multiple initialization strategies"""
    initial_points_list = []
    
    # Strategy 1: Fibonacci sphere
    points1 = fibonacci_sphere(14)
    initial_points_list.append(points1)
    
    # Strategy 2: Random points on sphere
    np.random.seed(123)
    points2 = np.random.randn(14, 3)
    points2 = project_to_unit_sphere(points2)
    initial_points_list.append(points2)
    
    # Strategy 3: Perturbed Fibonacci
    np.random.seed(456)
    points3 = fibonacci_sphere(14)
    noise = np.random.normal(0, 0.03, points3.shape)
    points3 = points3 + noise
    points3 = project_to_unit_sphere(points3)
    initial_points_list.append(points3)
    
    # Strategy 4: Another random distribution
    np.random.seed(789)
    points4 = np.random.randn(14, 3)
    points4 = project_to_unit_sphere(points4)
    initial_points_list.append(points4)
    
    # Strategy 5: Another Fibonacci variant
    np.random.seed(999)
    points5 = fibonacci_sphere(14)
    points5 = points5 * 0.9 + 0.05  # Shift away from boundaries
    points5 = project_to_unit_sphere(points5)
    initial_points_list.append(points5)
    
    return initial_points_list

def local_refinement_step(points, max_iterations=50):
    """
    Apply a simple local refinement using projected gradient descent
    to improve the current configuration around the best solution found.
    """
    try:
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
    random.seed(42)

    # Multi-start optimization with multiple initialization strategies
    best_points = None
    best_ratio = -np.inf

    # Generate multiple initial strategies
    initial_strategies = generate_initial_points()
    
    for strategy_idx, points in enumerate(initial_strategies):
        # Optimization parameters
        max_iterations = 150000  # Increased iterations for better search
        initial_temperature = 1.0
        cooling_rate = 0.9995
        min_temperature = 0.0001

        # Track best solution
        current_best_points = points.copy()
        current_best_ratio = compute_min_max_ratio(points)[2]  # Just get the ratio

        # Current state
        current_points = points.copy()
        current_ratio = current_best_ratio

        # Track ratio history for adaptive cooling
        ratio_history = [current_ratio]
        
        # Different temperature schedules for different phases
        temp_schedule = [
            {"temp": 1.0, "duration": 50000},   # High temperature for exploration
            {"temp": 0.5, "duration": 50000},   # Medium temperature for refinement  
            {"temp": 0.1, "duration": 50000}    # Low temperature for fine-tuning
        ]
        
        current_phase = 0
        phase_iterations = 0
        temp = initial_temperature

        # Simulated Annealing with multi-scale temperature
        last_improvement_iter = 0
        iteration = 0
        
        while iteration < max_iterations:
            # Check if we need to advance to next temperature phase
            if phase_iterations >= temp_schedule[current_phase]["duration"]:
                current_phase = min(current_phase + 1, len(temp_schedule) - 1)
                temp = temp_schedule[current_phase]["temp"]
                phase_iterations = 0
            
            # Perturb the current solution using spherical perturbations
            new_points = adaptive_perturbation_strategy(current_points, current_ratio, temp)

            # Compute new ratio
            new_min_dist, new_max_dist, new_ratio = compute_min_max_ratio(new_points)

            # Accept or reject the new solution using Metropolis criterion
            if new_ratio > current_ratio:
                # Always accept better solutions
                current_points = new_points
                current_ratio = new_ratio

                # Update best solution if this is better
                if new_ratio > current_best_ratio:
                    current_best_points = new_points.copy()
                    current_best_ratio = new_ratio
                    last_improvement_iter = iteration
                    ratio_history.append(new_ratio)
            else:
                # Accept worse solutions with probability based on temperature
                if temp > 0:  # Avoid division by zero
                    acceptance_prob = np.exp((new_ratio - current_ratio) / temp)
                    if np.random.rand() < acceptance_prob:
                        current_points = new_points
                        current_ratio = new_ratio
                        ratio_history.append(new_ratio)

            # Apply adaptive cooling
            temp = max(temp * adaptive_cooling(initial_temperature, iteration, max_iterations, ratio_history), min_temperature)
            
            # Periodic local refinement (every 50000 iterations)
            if iteration % 50000 == 0 and iteration > 0:
                refined_points = local_refinement_step(current_best_points)
                _, _, refined_ratio = compute_min_max_ratio(refined_points)
                if refined_ratio > current_best_ratio:
                    current_best_points = refined_points.copy()
                    current_best_ratio = refined_ratio
                    # Update the current points to reflect the refinement
                    current_points = refined_points.copy()

            # Increment counters
            iteration += 1
            phase_iterations += 1

            # Early stopping if no improvement in a long time
            if iteration - last_improvement_iter > 30000:
                break

        # Update global best if this run was better
        if current_best_ratio > best_ratio:
            best_ratio = current_best_ratio
            best_points = current_best_points.copy()

    # Ensure the result is properly normalized (should already be done, but extra safety)
    if best_points is not None:
        best_points = project_to_unit_sphere(best_points)
    else:
        # Fallback to Fibonacci if nothing worked
        best_points = fibonacci_sphere(14)
    
    return best_points

# EVOLVE-BLOCK-END