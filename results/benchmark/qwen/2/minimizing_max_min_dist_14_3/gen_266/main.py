# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math
import random
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0

        # Compute pairwise distances
        distances = pdist(points)

        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return 0.0

        return min_dist / max_dist

    def project_to_sphere(points):
        """Project points onto unit sphere while maintaining relative positions."""
        norms = np.linalg.norm(points, axis=1)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return points / norms[:, np.newaxis]

    def initialize_points_hybrid(seed=42):
        """Initialize points using hybrid approach combining icosahedral geometry with Fibonacci-like distribution."""
        np.random.seed(seed)
        
        # Start with icosahedral vertices (normalized)
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        icosahedron_vertices = [
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ]
        
        # Normalize vertices
        vertices = np.array(icosahedron_vertices)
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        # Use 12 vertices and add 2 more via Fibonacci-like distribution
        points = vertices.copy()
        
        # Add 2 more points using Fibonacci-inspired method for better distribution
        for i in range(2):
            # Use Fibonacci approach with adjusted parameters
            y = 1 - (i / 1.0) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y
            phi_angle = (i * (1 + math.sqrt(5)) / 2) % 14 * (2 * math.pi / 14)
            
            x = radius * math.cos(phi_angle)
            z = radius * math.sin(phi_angle)
            points = np.vstack([points, [x, y, z]])
            
        # Apply small random perturbations to break symmetries
        perturbation_magnitude = 0.03
        points += np.random.normal(0, perturbation_magnitude, points.shape)

        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1)
        safe_norms = np.where(norms == 0, 1.0, norms)
        points = points / safe_norms[:, np.newaxis]

        # Apply slight rotation to further break symmetries
        angle = np.pi / 12
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        # Rotate around y-axis
        for i in range(14):
            x, y, z = points[i]
            points[i] = [x * cos_a + z * sin_a, y, -x * sin_a + z * cos_a]

        return points

    def smart_perturb_point(point: np.ndarray, temp: float, points: np.ndarray) -> np.ndarray:
        """Smartly perturb a single point based on current configuration analysis."""
        # Generate random perturbation in tangent plane
        delta = np.random.randn(3)
        # Project to tangent plane (orthogonal to point)
        delta = delta - np.dot(delta, point) * point
        # Normalize the tangent vector
        delta_norm = np.linalg.norm(delta)
        if delta_norm > 1e-10:
            delta = delta / delta_norm
        else:
            # If the vector is essentially zero, use a random orthogonal vector
            perpendicular = np.array([1, 0, 0]) if abs(point[0]) < 0.9 else np.array([0, 1, 0])
            delta = np.cross(perpendicular, point)
            delta = delta / np.linalg.norm(delta)

        # Adaptive scaling based on temperature and current configuration
        # Higher temperature = larger perturbations, but with bounds
        perturbation_scale = min(temp * 0.03, 0.1)  # Cap the maximum perturbation
        new_point = point + perturbation_scale * delta
        # Project back to sphere
        new_point = new_point / np.linalg.norm(new_point)
        return new_point

    def adaptive_simulated_annealing(initial_points, max_iterations=100000):
        """Optimize point placement using enhanced simulated annealing with adaptive strategies."""
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_ratio = compute_min_max_ratio(current_points)

        # Enhanced adaptive cooling parameters
        current_temp = 1.0
        min_temp = 1e-10
        cooling_rate = 0.9997
        max_stagnation = 500
        stagnation_counter = 0

        # Track recent ratios for adaptive cooling
        recent_ratios = []
        last_improvement_iter = 0
        improvement_count = 0

        start_time = time.time()

        # Add a local refinement helper function
        def local_refinement(points, iterations=10):
            """Apply a simple local refinement to improve point distributions."""
            refined_points = points.copy()
            for _ in range(iterations):
                improved = False
                for i in range(14):
                    old_ratio = compute_min_max_ratio(refined_points)
                    old_point = refined_points[i].copy()

                    # Estimate gradient using finite differences
                    grad = np.zeros(3)
                    for j in range(3):
                        eps = 1e-4
                        test_points = refined_points.copy()
                        test_points[i, j] += eps
                        test_points[i] = test_points[i] / np.linalg.norm(test_points[i])
                        new_ratio = compute_min_max_ratio(test_points)
                        grad[j] = (new_ratio - old_ratio) / eps

                    # Move along gradient (scaled by learning rate)
                    if np.linalg.norm(grad) > 1e-10:
                        refined_points[i] = refined_points[i] + 0.1 * grad
                        refined_points[i] = refined_points[i] / np.linalg.norm(refined_points[i])
                        improved = True

                # If improvement was made, update the overall ratio
                if improved:
                    new_ratio = compute_min_max_ratio(refined_points)
                    if new_ratio > compute_min_max_ratio(points):
                        points = refined_points.copy()
                        return points
            return points

        for iteration in range(max_iterations):
            # Adaptive temperature cooling based on convergence
            if stagnation_counter > max_stagnation and current_temp > min_temp:
                current_temp *= 0.9  # Faster cooling when stagnating
                stagnation_counter = 0

            if current_temp < min_temp:
                break

            # Adaptive perturbation based on current distribution quality
            distances = pdist(current_points)
            if len(distances) > 0:
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                mean_dist = np.mean(distances)

                # If there are very small distances, focus on points that are too close
                if min_dist < mean_dist * 0.3:
                    # Select a point that's part of a very small distance pair
                    dist_matrix = pdist(current_points)
                    dist_square = squareform(dist_matrix)
                    min_indices = np.unravel_index(np.argmin(dist_square), dist_square.shape)
                    point_idx = min_indices[0] if np.random.random() < 0.7 else min_indices[1]  # Bias towards problematic points
                else:
                    # Otherwise, random selection works well
                    point_idx = np.random.randint(0, 14)
            else:
                point_idx = np.random.randint(0, 14)

            # Save current point
            old_point = current_points[point_idx].copy()

            # Perturb selected point with adaptive scaling
            current_points[point_idx] = smart_perturb_point(current_points[point_idx], current_temp, current_points)

            # Compute new ratio
            new_ratio = compute_min_max_ratio(current_points)

            # Accept or reject based on Metropolis criterion
            if new_ratio > best_ratio or np.random.random() < math.exp((new_ratio - best_ratio) / current_temp):
                current_points = current_points.copy()
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = current_points.copy()
                    last_improvement_iter = iteration
                    stagnation_counter = 0
                    improvement_count += 1
                else:
                    stagnation_counter += 1
            else:
                stagnation_counter += 1

            # Periodic local refinement to help escape local optima
            if iteration % 500 == 0 and iteration > 0:
                current_points = local_refinement(current_points, iterations=5)
                current_ratio = compute_min_max_ratio(current_points)
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = current_points.copy()
                    stagnation_counter = 0

            # Track recent ratios for adaptive cooling
            recent_ratios.append(new_ratio)
            if len(recent_ratios) > 20:
                recent_ratios.pop(0)

            # Occasionally print progress (only every 10000 iterations to reduce overhead)
            if iteration % 10000 == 0:
                elapsed = time.time() - start_time
                if elapsed > 350:  # Leave 10 seconds for cleanup
                    break

        return best_points, best_ratio

    def multi_start_optimization():
        """Run adaptive simulated annealing multiple times with different random seeds."""
        best_points = None
        best_ratio = 0.0

        # Try multiple restarts with different seeds
        seeds = [42, 123, 456, 789, 999, 111, 222, 333, 555, 666, 888, 999, 314, 271, 141] 

        start_time = time.time()
        for seed in seeds:
            if time.time() - start_time > 350:  # Leave 10 seconds for cleanup
                break
            try:
                np.random.seed(seed)
                initial_points = initialize_points_hybrid(seed)
                points, ratio = adaptive_simulated_annealing(initial_points, max_iterations=100000)

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = points.copy()
            except Exception as e:
                # Continue to next seed if this one fails
                continue

        # Fallback to the best solution if none worked
        if best_points is None:
            best_points = initialize_points_hybrid(42)

        return best_points

    # Run multi-start optimization
    return multi_start_optimization()

# EVOLVE-BLOCK-END