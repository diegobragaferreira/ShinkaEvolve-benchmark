# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import SphericalVoronoi
import math
import random
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def compute_min_max_ratio(points: np.ndarray) -> float:
        """Compute the ratio of minimum to maximum distances."""
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0.0
        return min_dist / max_dist

    def analyze_voronoi_regions(points: np.ndarray) -> np.ndarray:
        """Analyze point distribution using spherical Voronoi to identify dense/constrained regions."""
        try:
            # Create spherical Voronoi diagram
            sv = SphericalVoronoi(points)
            # Get areas of Voronoi cells
            areas = sv.calculate_areas()

            # Identify points in small cells (indicating dense regions)
            mean_area = np.mean(areas)
            std_area = np.std(areas)

            # Score points based on their Voronoi cell area
            # Points in small cells get higher scores (they're in tight clusters)
            voronoi_scores = np.zeros(len(points))
            for i, area in enumerate(areas):
                # Normalize score based on how much smaller the cell is than average
                if mean_area > 0:
                    norm_factor = (mean_area - area) / mean_area if area < mean_area else 0
                    voronoi_scores[i] = max(0, norm_factor)

            return voronoi_scores
        except Exception:
            # Fallback if Voronoi computation fails
            return np.zeros(len(points))

    def fibonacci_sphere(n: int, seed: int = 42) -> np.ndarray:
        """Generate n points on a sphere using improved Fibonacci spiral method."""
        np.random.seed(seed)
        points = []
        phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle

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

    def initialize_points(n: int, seed: int = 42) -> np.ndarray:
        """Initialize points using improved Fibonacci sphere and perturbation."""
        # Start with Fibonacci sphere points
        points = fibonacci_sphere(n, seed)

        # Add some randomness to avoid perfect symmetry issues
        points += 0.02 * np.random.randn(n, 3)

        # Normalize to unit sphere
        points = points / np.linalg.norm(points, axis=1, keepdims=True)

        return points

    def project_to_sphere(points: np.ndarray) -> np.ndarray:
        """Project points to unit sphere while preserving relative positions."""
        norms = np.linalg.norm(points, axis=1)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return points / norms[:, np.newaxis]

    def perturb_point(point: np.ndarray, step_size: float, is_targeted: bool = False) -> np.ndarray:
        """Perturb a single point on the unit sphere."""
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
            # Find a vector not parallel to point
            perpendicular = np.array([1, 0, 0]) if abs(point[0]) < 0.9 else np.array([0, 1, 0])
            delta = np.cross(perpendicular, point)
            delta = delta / np.linalg.norm(delta)

        # Apply perturbation
        new_point = point + step_size * delta
        # Project back to sphere
        new_point = new_point / np.linalg.norm(new_point)

        return new_point

    def targeted_perturbation(points: np.ndarray, point_idx: int, voronoi_scores: np.ndarray,
                           temperature: float, min_dist: float, mean_dist: float) -> np.ndarray:
        """Apply targeted perturbation based on Voronoi analysis and distance characteristics."""
        # Get the Voronoi score for this point
        voronoi_score = voronoi_scores[point_idx]

        # Base perturbation size
        base_step_size = temperature * 0.02

        # If this point is in a dense Voronoi region, apply larger perturbation
        if voronoi_score > 0.3:  # High density region
            step_size = base_step_size * 2.0  # Twice the normal step
        elif voronoi_score > 0.1:  # Moderate density
            step_size = base_step_size * 1.5
        else:  # Normal region
            step_size = base_step_size

        # Additional targeting based on distance characteristics
        distances = pdist(points)
        if len(distances) > 0:
            # Check if this point is part of a very small distance pair
            dist_matrix = squareform(distances)
            # Find all distances involving this point
            point_distances = dist_matrix[point_idx]
            # Get the minimum distance for this point
            min_point_dist = np.min(point_distances[point_distances > 0])

            # If this point has a very small distance to others, push it away
            if min_point_dist < mean_dist * 0.3:
                # Create repulsion vector away from nearby points
                repulsion = np.zeros(3)
                for i in range(len(points)):
                    if i != point_idx and point_distances[i] < mean_dist * 0.5:
                        diff = points[point_idx] - points[i]
                        dist = np.linalg.norm(diff)
                        if dist > 0:
                            repulsion += diff / dist * (1.0/dist)

                if np.linalg.norm(repulsion) > 0:
                    repulsion = repulsion / np.linalg.norm(repulsion)
                    # Apply stronger repulsion
                    step_size = max(step_size, 0.03)  # At least 0.03 step size
                    # Add repulsion component to original perturbation
                    original_delta = np.random.randn(3)
                    original_delta = original_delta - np.dot(original_delta, points[point_idx]) * points[point_idx]
                    original_delta = original_delta / np.linalg.norm(original_delta) if np.linalg.norm(original_delta) > 1e-10 else np.array([1, 0, 0])

                    # Combine both directions - repulsion is stronger
                    combined_delta = 0.7 * repulsion + 0.3 * original_delta
                    combined_delta = combined_delta / np.linalg.norm(combined_delta)
                    new_point = points[point_idx] + step_size * combined_delta
                    return new_point / np.linalg.norm(new_point)

        # Default perturbation
        return perturb_point(points[point_idx], step_size)

    def adaptive_perturbation(points: np.ndarray, temperature: float,
                           current_ratio: float, distances: np.ndarray) -> tuple:
        """Adaptively select which point to perturb based on current distribution."""
        # Analyze Voronoi regions first
        voronoi_scores = analyze_voronoi_regions(points)

        # Select point strategically - prefer points in dense regions or involved in small distances
        if len(distances) > 0:
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            mean_dist = np.mean(distances)

            # If there are very small distances, focus on points that are too close
            if min_dist < mean_dist * 0.4:
                # Select a point that's part of a very small distance pair
                # Find pairs with minimal distances
                dist_matrix = pdist(points)
                # Convert to square form for easier indexing
                dist_square = squareform(dist_matrix)
                # Find minimum distances and corresponding indices
                min_indices = np.unravel_index(np.argmin(dist_square), dist_square.shape)
                # Bias towards points in dense Voronoi regions
                if voronoi_scores[min_indices[0]] > voronoi_scores[min_indices[1]]:
                    point_idx = min_indices[0]
                else:
                    point_idx = min_indices[1]
            elif np.any(voronoi_scores > 0.1):  # If there are some dense regions
                # Select point from dense regions with probability proportional to score
                weights = voronoi_scores + 0.01  # Add small constant to avoid zeros
                point_idx = np.random.choice(len(points), p=weights/np.sum(weights))
            else:
                # Otherwise, random selection works well
                point_idx = random.randint(0, 13)
        else:
            point_idx = random.randint(0, 13)

        # Adaptive perturbation scale based on temperature and Voronoi score
        base_perturbation_scale = temperature * 0.02
        # Adjust based on Voronoi score (higher score = larger perturbation needed)
        voronoi_adjustment = 1.0 + voronoi_scores[point_idx] * 2.0
        perturbation_scale = base_perturbation_scale * voronoi_adjustment
        return point_idx, perturbation_scale

    def gradient_estimation(points: np.ndarray, i: int, eps: float = 1e-5) -> np.ndarray:
        """Estimate gradient of ratio with respect to point i."""
        old_ratio = compute_min_max_ratio(points)
        old_point = points[i].copy()

        grad = np.zeros(3)
        for j in range(3):
            # Forward difference
            test_points = points.copy()
            test_points[i, j] += eps
            test_points[i] = test_points[i] / np.linalg.norm(test_points[i])
            new_ratio = compute_min_max_ratio(test_points)
            grad[j] = (new_ratio - old_ratio) / eps

        return grad

    def simulated_annealing(seed: int = 42):
        """Run simulated annealing optimization with adaptive strategies."""
        # Initialize with better starting configuration
        points = initialize_points(14, seed)
        current_ratio = compute_min_max_ratio(points)
        distances = pdist(points)

        # Parameters
        T = 0.2  # Initial temperature
        Tmin = 1e-8  # Minimum temperature
        alpha = 0.9995  # Cooling rate
        max_iter = 100000  # Maximum iterations

        # Track best solution
        best_points = points.copy()
        best_ratio = current_ratio

        # Adaptive cooling parameters
        last_improvement = 0
        stagnation_count = 0
        max_stagnation = 1000

        for iteration in range(max_iter):
            # Adaptive cooling based on recent progress
            if stagnation_count > max_stagnation and T > Tmin:
                # Faster cooling when stagnating
                T = max(Tmin, T * 0.95)
                stagnation_count = 0
            else:
                T = max(Tmin, T * alpha)

            # Store recent improvements to detect convergence
            if iteration > 0 and iteration % 500 == 0:
                # Check for convergence
                if stagnation_count > 2000:
                    break

            # Adaptive perturbation selection
            point_idx, perturbation_scale = adaptive_perturbation(
                points, T, current_ratio, distances)

            # Save current point
            old_point = points[point_idx].copy()

            # Perturb selected point with targeted approach
            points[point_idx] = targeted_perturbation(points, point_idx,
                                                    analyze_voronoi_regions(points),
                                                    T, np.min(distances) if len(distances) > 0 else 1.0,
                                                    np.mean(distances) if len(distances) > 0 else 1.0)

            # Compute new ratio
            new_ratio = compute_min_max_ratio(points)
            distances = pdist(points)

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

            # Periodic local refinement with gradient-based approach
            if iteration % 1000 == 0 and iteration > 0:
                # Local refinement with better gradient estimation
                refined_points = points.copy()

                # Multiple passes of gradient-based refinement
                for pass_num in range(3):
                    improved = False
                    for i in range(14):
                        # Better gradient estimation with adaptive epsilon
                        eps = max(1e-6, 1e-4 * (T / 0.1))
                        grad = gradient_estimation(refined_points, i, eps)

                        # Move along gradient
                        if np.linalg.norm(grad) > 1e-10:
                            refined_points[i] = refined_points[i] + 0.1 * grad
                            refined_points[i] = refined_points[i] / np.linalg.norm(refined_points[i])
                            improved = True

                    # Update if improved
                    new_ratio = compute_min_max_ratio(refined_points)
                    if new_ratio > compute_min_max_ratio(points):
                        points = refined_points.copy()
                        current_ratio = new_ratio
                        if new_ratio > best_ratio:
                            best_ratio = new_ratio
                            best_points = points.copy()
                            last_improvement = iteration
                            stagnation_count = 0

            # Occasionally reset temperature to escape local minima
            if iteration % 5000 == 0 and iteration > 0:
                T = min(0.5, T * 1.2)  # Warm up occasionally

            # Early stopping if we're not improving
            if iteration - last_improvement > 5000:
                break

        return best_points

    # Multi-start optimization with different seeds
    best_points = None
    best_ratio = 0.0
    seeds = [42, 123, 456, 789]

    start_time = time.time()
    for seed in seeds:
        if time.time() - start_time > 350:  # Leave 10 seconds for cleanup
            break
        try:
            points = simulated_annealing(seed)
            ratio = compute_min_max_ratio(points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = points.copy()
        except Exception as e:
            continue

    # Fallback to the best solution if none worked
    if best_points is None:
        best_points = initialize_points(14, 42)

    return best_points

# EVOLVE-BLOCK-END