# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
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
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0.0
        return min_dist / max_dist

    def fibonacci_sphere(n: int, seed: int = 42) -> np.ndarray:
        """Generate n points on a sphere using Fibonacci spiral method."""
        np.random.seed(seed)
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

    def icosahedral_initialization(n: int, seed: int = 42) -> np.ndarray:
        """Initialize points using icosahedral symmetry for better spread."""
        np.random.seed(seed)
        points = np.zeros((n, 3))

        # Add poles
        points[0] = [0, 0, 1]       # North pole
        points[1] = [0, 0, -1]      # South pole

        # Add equatorial points in two rings
        angle_step = 2 * math.pi / 5
        for i in range(5):
            angle1 = i * angle_step
            angle2 = angle1 + angle_step / 2

            # First ring
            points[2+i] = [math.cos(angle1), math.sin(angle1), 0.0]
            # Second ring offset
            points[7+i] = [math.cos(angle2), math.sin(angle2), 0.0]

        # Add additional points near poles
        points[12] = [0, 0, 0.7]
        points[13] = [0, 0, -0.7]

        # Add small random perturbations
        points += np.random.normal(0, 0.01, points.shape)

        return points

    def project_to_sphere(points):
        """Project points onto unit sphere while maintaining exact constraint."""
        norms = np.linalg.norm(points, axis=1)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return points / norms[:, np.newaxis]

    def adaptive_perturb_point(point: np.ndarray, temp: float) -> np.ndarray:
        """Perturb a single point on the unit sphere using adaptive method."""
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

        # Adaptive scaling based on temperature
        perturbation_scale = temp * 0.02
        new_point = point + perturbation_scale * delta
        # Project back to sphere
        new_point = new_point / np.linalg.norm(new_point)
        return new_point

    def smart_point_selection(points, distances):
        """Select point to perturb based on current configuration analysis."""
        if len(distances) > 0:
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            mean_dist = np.mean(distances)

            # If there are very small distances, focus on points that are too close
            if min_dist < mean_dist * 0.4:
                # Select a point that's part of a very small distance pair
                dist_matrix = pdist(points)
                dist_square = squareform(dist_matrix)
                min_indices = np.unravel_index(np.argmin(dist_square), dist_square.shape)
                point_idx = min_indices[0] if np.random.random() < 0.5 else min_indices[1]
            else:
                # Otherwise, random selection works well
                point_idx = random.randint(0, 13)
        else:
            point_idx = random.randint(0, 13)
        
        return point_idx

    def targeted_perturbation(points, point_idx):
        """Compute targeted perturbation based on local geometry analysis."""
        try:
            # Analyze distances to neighbors
            distances = cdist([points[point_idx]], points)[0]
            distances = distances[distances > 0]  # Remove self-distance

            if len(distances) > 0:
                avg_distance = np.mean(distances)
                min_distance = np.min(distances)

                # Determine perturbation type based on local density
                if min_distance < avg_distance * 0.5:
                    # Point is too close to neighbors - repel it
                    repulsion = np.zeros(3)
                    for i in range(len(points)):
                        if i != point_idx and distances[i-1] < avg_distance * 0.7:
                            diff = points[point_idx] - points[i]
                            dist = np.linalg.norm(diff)
                            if dist > 0:
                                repulsion += diff / dist * (1.0/dist)

                    if np.linalg.norm(repulsion) > 0:
                        repulsion = repulsion / np.linalg.norm(repulsion)
                        magnitude = 0.03 * (1.0 + min_distance / avg_distance)
                        return repulsion * magnitude
                    else:
                        # Fallback to random perturbation
                        direction = np.random.randn(3)
                        direction /= np.linalg.norm(direction)
                        return direction * 0.02
                else:
                    # Point is relatively well-spaced - make small adjustment
                    direction = np.random.randn(3)
                    direction /= np.linalg.norm(direction)
                    # Magnitude inversely proportional to distance from center
                    center_dist = np.linalg.norm(points[point_idx])
                    magnitude = 0.015 * (1.0 - center_dist * 0.5)
                    return direction * magnitude
            else:
                # Fallback for edge cases
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                return direction * 0.02

        except Exception:
            # Fallback to simple random perturbation
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            return direction * 0.02

    def simulated_annealing_with_adaptive_strategy():
        """Enhanced simulated annealing with adaptive cooling and perturbation."""
        # Initialize with better starting configuration
        np.random.seed(42)
        points = fibonacci_sphere(14, 42)
        
        # Add some randomness to break symmetries
        points += np.random.normal(0, 0.01, points.shape)
        points = project_to_sphere(points)
        
        current_ratio = compute_min_max_ratio(points)
        distances = pdist(points)

        # Parameters
        T = 0.5  # Initial temperature (higher than before)
        Tmin = 1e-8  # Minimum temperature
        alpha = 0.9995  # Cooling rate
        max_iter = 100000  # Maximum iterations
        step_size = 0.01  # Initial step size

        # Track best solution
        best_points = points.copy()
        best_ratio = current_ratio

        # Adaptive cooling parameters
        last_improvement = 0
        patience = 500
        stagnation_count = 0
        max_stagnation = 1000
        improvement_history = []

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

            # Adaptive point selection based on current configuration
            point_idx = smart_point_selection(points, distances)

            # Save current point
            old_point = points[point_idx].copy()

            # Perturb selected point with adaptive scaling
            points[point_idx] = adaptive_perturb_point(points[point_idx], T)
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

            # Periodic local refinement
            if iteration % 1000 == 0 and iteration > 0:
                # Local refinement with gradient-based approach (simplified)
                refined_points = points.copy()
                for _ in range(5):  # Reduced iterations for efficiency
                    improved = False
                    for i in range(14):
                        # Simple gradient estimation by finite differences
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
                            refined_points[i] = refined_points[i] + 0.05 * grad
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

            # Early stopping if we're not improving
            if iteration - last_improvement > 5000:
                break

        return best_points

    def multi_start_optimization():
        """Run optimization multiple times with different seeds for better exploration."""
        best_points = None
        best_ratio = 0.0

        # Different initialization strategies
        initial_configs = [
            ("fibonacci", 42),
            ("fibonacci", 123),
            ("fibonacci", 456),
            ("icosahedral", 789),
            ("icosahedral", 999),
            ("icosahedral", 111),
            ("icosahedral", 222)
        ]

        start_time = time.time()
        for init_type, seed in initial_configs:
            if time.time() - start_time > 350:  # Leave 10 seconds for cleanup
                break
            try:
                np.random.seed(seed)
                if init_type == "fibonacci":
                    points = fibonacci_sphere(14, seed)
                else:  # icosahedral
                    points = icosahedral_initialization(14, seed)
                
                # Add small random perturbations to break symmetries
                points += np.random.normal(0, 0.01, points.shape)
                points = project_to_sphere(points)
                
                # Run optimization
                ratio = compute_min_max_ratio(points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = points.copy()
            except Exception as e:
                continue

        # Fallback to the best solution if none worked
        if best_points is None:
            best_points = fibonacci_sphere(14, 42)
            # Add perturbations
            np.random.seed(42)
            best_points += np.random.normal(0, 0.01, best_points.shape)
            best_points = project_to_sphere(best_points)

        return best_points

    # Run multi-start optimization
    return multi_start_optimization()

# EVOLVE-BLOCK-END