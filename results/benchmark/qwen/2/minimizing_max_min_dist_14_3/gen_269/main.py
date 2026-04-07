# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math
import random
import time
from scipy.spatial import SphericalVoronoi

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0.0
        return min_dist / max_dist

    def fibonacci_sphere(n: int, seed: int = 42) -> np.ndarray:
        """Generate n points on a sphere using improved Fibonacci spiral method."""
        np.random.seed(seed)
        points = []
        # Golden angle
        phi = math.pi * (3.0 - math.sqrt(5.0))

        for i in range(n):
            # Improved Fibonacci approach with better distribution
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y

            # Add small random offset to break symmetries
            theta = phi * i + np.random.normal(0, 0.1)  # golden angle increment

            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def initialize_points():
        """Initialize points using enhanced Fibonacci sphere method with perturbations."""
        # Start with improved Fibonacci sphere points
        points = fibonacci_sphere(14, seed=42)

        # Apply small random perturbations to break symmetries
        np.random.seed(42)
        perturbation_magnitude = 0.03
        points += np.random.normal(0, perturbation_magnitude, points.shape)

        # Normalize to ensure points are on unit sphere
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

    def project_to_sphere(points):
        """Project points onto unit sphere while maintaining exact constraint."""
        norms = np.linalg.norm(points, axis=1)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return points / norms[:, np.newaxis]

    def perturb_point_on_sphere(point: np.ndarray, step_size: float) -> np.ndarray:
        """Perturb a single point on the unit sphere using tangent plane method."""
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

        # Apply perturbation
        new_point = point + step_size * delta
        # Project back to sphere
        new_point = new_point / np.linalg.norm(new_point)
        return new_point

    def compute_voronoi_stats(points):
        """Compute Voronoi cell area statistics to evaluate distribution quality."""
        try:
            # Create spherical Voronoi diagram
            sv = SphericalVoronoi(points)
            # Get areas of Voronoi cells
            areas = sv.calculate_areas()
            # Return stats about Voronoi distribution (mean and std deviation)
            return np.mean(areas), np.std(areas)
        except:
            # If Voronoi computation fails, return neutral values
            return 1.0, 0.0

    def adaptive_point_selection(points, distances, current_ratio):
        """Select a point for perturbation based on local geometric properties."""
        # If we're in a region with poor distribution, select points that need adjustment
        if len(distances) > 0:
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            mean_dist = np.mean(distances)

            # Check if there are very small distances
            if min_dist < mean_dist * 0.4:
                # Use Voronoi analysis to get better insight into local structure
                mean_area, std_area = compute_voronoi_stats(points)

                # Prefer points in regions with very small distances
                dist_matrix = pdist(points)
                dist_square = squareform(dist_matrix)
                min_indices = np.unravel_index(np.argmin(dist_square), dist_square.shape)

                # Bias towards points with smaller distances - but also consider Voronoi quality
                if std_area > 0.1:  # If distribution is uneven, we might want to adjust
                    point_idx = min_indices[0] if np.random.random() < 0.7 else min_indices[1]
                else:
                    # Select a random point that's part of a small distance pair
                    point_idx = min_indices[0] if np.random.random() < 0.5 else min_indices[1]
            else:
                # Otherwise, use more strategic approach
                # Select point based on its contribution to minimum distance
                point_idx = random.randint(0, 13)
        else:
            point_idx = random.randint(0, 13)

        return point_idx

    def advanced_local_refinement(points, num_iterations=10):
        """Perform more advanced local refinement using gradient estimation and Voronoi analysis."""
        refined_points = points.copy()

        for iter_num in range(num_iterations):
            improved = False
            for i in range(14):
                # Estimate gradient of the ratio function
                old_ratio = compute_min_max_ratio(refined_points)
                old_point = refined_points[i].copy()

                # Simple gradient estimation by finite differences
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
                    improved = True

            # If we made improvements, recalculate to see if this helped
            if improved:
                new_ratio = compute_min_max_ratio(refined_points)
                if new_ratio > compute_min_max_ratio(points):
                    return refined_points

        return points

    def adaptive_simulated_annealing():
        """Enhanced simulated annealing with adaptive cooling, Voronoi refinement, and better point selection."""
        # Initialize with better starting configuration
        points = initialize_points()
        current_ratio = compute_min_max_ratio(points)
        distances = pdist(points)

        # Parameters
        T = 0.5  # Initial temperature (higher than before)
        Tmin = 1e-8  # Minimum temperature
        alpha = 0.9995  # Cooling rate
        max_iter = 100000  # Maximum iterations

        # Track best solution
        best_points = points.copy()
        best_ratio = current_ratio

        # Adaptive cooling parameters
        last_improvement = 0
        patience = 500
        stagnation_count = 0
        max_stagnation = 1000

        for iteration in range(max_iter):
            # Adaptive cooling based on recent progress
            if stagnation_count > max_stagnation and T > Tmin:
                T = max(Tmin, T * 0.95)  # Faster cooling when stagnating
                stagnation_count = 0
            else:
                T = max(Tmin, T * alpha)

            # Stop if temperature gets too low
            if T < Tmin:
                break

            # Adaptive point selection based on current configuration and geometric properties
            point_idx = adaptive_point_selection(points, distances, current_ratio)

            # Save current point
            old_point = points[point_idx].copy()

            # Perturb selected point with adaptive scaling
            step_size = T * 0.02
            points[point_idx] = perturb_point_on_sphere(points[point_idx], step_size)
            distances = pdist(points)

            # Compute new ratio
            new_ratio = compute_min_max_ratio(points)

            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio:
                current_ratio = new_ratio
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = points.copy()
                    last_improvement = iteration
                    stagnation_count = 0
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
                        stagnation_count = 0
                else:
                    # Revert change
                    points[point_idx] = old_point
                    stagnation_count += 1

            # Advanced periodic local refinement with Voronoi analysis
            if iteration % 1000 == 0 and iteration > 0:
                # Apply more intelligent local refinement using Voronoi-based insights
                refined_points = advanced_local_refinement(points, num_iterations=15)
                new_ratio = compute_min_max_ratio(refined_points)
                if new_ratio > compute_min_max_ratio(points):
                    points = refined_points.copy()
                    current_ratio = new_ratio
                    if new_ratio > best_ratio:
                        best_ratio = new_ratio
                        best_points = points.copy()
                        last_improvement = iteration
                        stagnation_count = 0

            # Early stopping if we're not improving
            if iteration - last_improvement > 5000:
                break

        return best_points

    def multi_start_optimization():
        """Run optimization multiple times with different seeds for better exploration."""
        best_points = None
        best_ratio = 0.0

        # Use more diverse seeds for better exploration
        seeds = [42, 123, 456, 789, 999, 111, 222, 333, 555, 666, 777, 888]

        start_time = time.time()
        for seed in seeds:
            if time.time() - start_time > 350:  # Leave 10 seconds for cleanup
                break
            try:
                np.random.seed(seed)
                points = adaptive_simulated_annealing()
                ratio = compute_min_max_ratio(points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = points.copy()
            except Exception as e:
                continue

        # Fallback to the best solution if none worked
        if best_points is None:
            best_points = initialize_points()

        return best_points

    # Run multi-start optimization
    return multi_start_optimization()

# EVOLVE-BLOCK-END