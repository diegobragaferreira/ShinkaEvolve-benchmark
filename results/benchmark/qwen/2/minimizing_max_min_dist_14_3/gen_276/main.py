# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import math
import random
from typing import Tuple, List
from scipy.spatial import SphericalVoronoi

class SphericalPointOptimizer:
    def __init__(self):
        self.best_points = None
        self.best_ratio = 0.0

    def compute_min_max_ratio(self, points: np.ndarray) -> float:
        """Compute the ratio of minimum to maximum distances."""
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        return d_min / d_max if d_max > 0 else 0

    def fibonacci_sphere(self, n: int) -> np.ndarray:
        """Generate n points distributed approximately uniformly on a sphere."""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        for i in range(n):
            z = 1 - (i / (n - 1)) * 2  # z goes from 1 to -1
            radius = np.sqrt(1 - z*z)
            theta = np.arctan2(np.sin(i * 2 * np.pi / phi), np.cos(i * 2 * np.pi / phi))
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)

    def icosahedron_vertices(self) -> np.ndarray:
        """Generate vertices of a regular icosahedron."""
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = []
        # Add vertices at (±1, ±φ, 0), (0, ±1, ±φ), (±φ, 0, ±1)
        for i in [1, -1]:
            for j in [1, -1]:
                vertices.append([i, j * phi, 0])
                vertices.append([0, i, j * phi])
                vertices.append([i * phi, 0, j])
        return np.array(vertices)

    def spherical_voronoi_init(self, n: int) -> np.ndarray:
        """Generate points using Spherical Voronoi diagram for better distribution."""
        # Start with random points
        points = np.random.randn(n, 3)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)

        try:
            # Create SphericalVoronoi diagram
            sv = SphericalVoronoi(points)

            # Use Voronoi vertices as initial configuration
            voronoi_points = sv.vertices

            if len(voronoi_points) >= n:
                # Use first n points from Voronoi vertices
                result = voronoi_points[:n].copy()
                # Normalize to unit sphere
                result = result / np.linalg.norm(result, axis=1, keepdims=True)
                return result
        except:
            pass

        # Fallback to Fibonacci if Voronoi fails
        return self.fibonacci_sphere(n)

    def project_to_sphere(self, points: np.ndarray) -> np.ndarray:
        """Project points onto unit sphere with numerical stability."""
        norms = np.linalg.norm(points, axis=1)
        # Avoid division by zero with small epsilon
        eps = 1e-12
        norms = np.where(norms < eps, 1, norms)
        return points / norms[:, np.newaxis]

    def generate_initial_configurations(self, num_configs: int = 7) -> List[np.ndarray]:
        """Generate multiple diverse initial configurations."""
        configs = []

        # Icosahedron-based configuration
        ico_vertices = self.icosahedron_vertices()
        if len(ico_vertices) >= 14:
            ico_config = ico_vertices[:14].copy()
        else:
            ico_config = np.vstack([ico_vertices, ico_vertices[:(14-len(ico_vertices))]]).copy()
        ico_config = self.project_to_sphere(ico_config)
        configs.append(ico_config)

        # Base Fibonacci configuration
        base_config = self.fibonacci_sphere(14)
        base_config = self.project_to_sphere(base_config)
        configs.append(base_config)

        # Spherical Voronoi configuration
        sv_config = self.spherical_voronoi_init(14)
        configs.append(sv_config)

        # Generate variations with different seeds
        for i in range(num_configs):
            np.random.seed(i * 1000 + 42)
            config = self.fibonacci_sphere(14)
            # Add slight perturbations to break symmetries
            noise = np.random.normal(0, 0.005, config.shape)
            config += noise
            config = self.project_to_sphere(config)
            configs.append(config)

        return configs

    def local_search_refinement(self, points: np.ndarray, iterations: int = 200) -> np.ndarray:
        """Perform thorough local search refinement."""
        current_points = points.copy()
        current_ratio = self.compute_min_max_ratio(current_points)

        # Multi-directional search with enhanced step sizes
        step_sizes = [0.002, 0.001, 0.0005]

        for _ in range(iterations):
            for i in range(len(current_points)):
                for j in range(3):
                    old_val = current_points[i, j]
                    # Try multiple step sizes for better exploration
                    for step_size in step_sizes:
                        for delta in [-step_size, step_size]:
                            current_points[i, j] = old_val + delta
                            current_points[i] = self.project_to_sphere(current_points[i:i+1])[0]
                            new_ratio = self.compute_min_max_ratio(current_points)
                            if new_ratio > current_ratio:
                                current_ratio = new_ratio
                            else:
                                current_points[i, j] = old_val  # Revert

        return current_points

    def adaptive_simulated_annealing(self, initial_points: np.ndarray, max_iter: int = 15000, temp_schedule_params=None) -> Tuple[np.ndarray, float]:
        """Optimize points using advanced adaptive simulated annealing."""
        if temp_schedule_params is None:
            temp_schedule_params = {
                'initial_temp': 0.15,
                'cooling_rate': 0.9992,
                'min_temp': 1e-7,
                'stagnation_threshold': 1000,
                'diversity_factor': 0.2  # Higher diversity factor
            }

        current_points = initial_points.copy()
        current_points = self.project_to_sphere(current_points)

        best_points = current_points.copy()
        best_ratio = self.compute_min_max_ratio(current_points)

        # Enhanced cooling parameters
        temperature = temp_schedule_params['initial_temp']
        cooling_rate = temp_schedule_params['cooling_rate']
        min_temp = temp_schedule_params['min_temp']
        max_stagnation = temp_schedule_params['stagnation_threshold']
        diversity_factor = temp_schedule_params['diversity_factor']

        # Convergence tracking
        last_improvement = 0
        stagnation_counter = 0
        recent_improvements = []
        improvement_history = []

        # Track recent improvements for dynamic adaptation
        improvement_window = 50
        recent_improvements_window = []

        # Additional tracking for better adaptive behavior
        total_improvements = 0
        total_evaluations = 0

        for iteration in range(max_iter):
            old_points = current_points.copy()
            old_ratio = self.compute_min_max_ratio(current_points)
            total_evaluations += 1

            # Point selection with bias towards problematic areas
            idx = np.random.randint(len(current_points))

            # Adaptive perturbation based on solution quality and iteration
            current_distances = pdist(current_points)
            current_min_dist = np.min(current_distances)
            current_max_dist = np.max(current_distances)
            current_ratio_val = current_min_dist / current_max_dist if current_max_dist > 0 else 0

            # Calculate dynamic perturbation size
            # Base perturbation decreases as solution improves
            base_perturbation = 0.03 * (1 - current_ratio_val) + 0.001

            # Iteration-based scaling
            iteration_factor = 1.0 - (iteration / max_iter) * 0.7

            # Add variance to enhance exploration
            perturbation_variance = 0.2 * np.random.random()
            perturbation_size = base_perturbation * iteration_factor * (1 + perturbation_variance)

            # Generate perturbation with enhanced stability
            perturbation = np.random.normal(0, perturbation_size, 3)

            # Tangent plane projection for better numerical stability
            current_point = current_points[idx]
            projection_factor = np.dot(perturbation, current_point)
            perturbation_tangent = perturbation - projection_factor * current_point

            # Apply perturbation
            current_points[idx] += perturbation_tangent

            # Robust projection back to unit sphere
            current_points[idx] = self.project_to_sphere(current_points[idx:idx+1])[0]

            # Compute new ratio
            new_ratio = self.compute_min_max_ratio(current_points)
            total_evaluations += 1

            # Accept or reject based on Metropolis criterion
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_points = current_points.copy()
                last_improvement = iteration
                stagnation_counter = 0
                recent_improvements.clear()
                improvement_history.append(new_ratio)
                total_improvements += 1
            elif np.random.random() < np.exp((new_ratio - old_ratio) / temperature):
                pass
            else:
                current_points = old_points

            # Dynamic cooling based on improvement rate and solution quality
            improvement = new_ratio - old_ratio
            recent_improvements_window.append(improvement)
            if len(recent_improvements_window) > improvement_window:
                recent_improvements_window.pop(0)

            # Calculate recent average improvement
            avg_improvement = np.mean(recent_improvements_window) if recent_improvements_window else 0

            # Enhanced adaptive cooling logic with solution quality awareness
            # Base cooling rate adjustment on improvement rate
            if avg_improvement < 1e-8 and stagnation_counter > 500:
                # Very slow improvement: aggressive cooling to escape local minima
                cooling_rate = max(0.99985, cooling_rate * 0.97)
            elif avg_improvement < 1e-6:
                # Slow improvement: accelerate cooling moderately
                cooling_rate = max(0.9996, cooling_rate * 0.985)
            elif avg_improvement > 1e-3:
                # Fast improvement: slow cooling to fine-tune
                cooling_rate = min(0.99995, cooling_rate * 1.002)
            elif avg_improvement > 1e-4:
                # Moderate improvement: mild cooling
                cooling_rate = min(0.9999, cooling_rate * 1.001)
            else:
                # Minimal improvement: use default cooling rate
                pass

            # Additional factor: if solution is approaching good region, cool faster
            # Benchmark ratio to determine how close we are to optimal
            benchmark_ratio = best_ratio / 0.4898  # 0.4898 is the target benchmark
            if benchmark_ratio > 0.85:  # Near optimal region
                # Cool even faster when close to good solutions
                cooling_rate = max(0.9999, cooling_rate * 0.998)
            elif benchmark_ratio > 0.7:  # Somewhat good region
                # Moderate cooling acceleration
                cooling_rate = max(0.9998, cooling_rate * 0.995)

            # Standard cooling logic
            stagnation_counter += 1
            if stagnation_counter > max_stagnation:
                temperature = max(min_temp, temperature * 0.95)
                stagnation_counter = 0

                # Stronger diversity-inducing perturbations
                if np.random.random() < diversity_factor:
                    for i in range(len(current_points)):
                        if np.random.random() < 0.25:  # 25% chance per point
                            # Use larger perturbation for diversity
                            perturbation = np.random.normal(0, 0.01, 3)
                            current_point = current_points[i]
                            projection_factor = np.dot(perturbation, current_point)
                            perturbation_tangent = perturbation - projection_factor * current_point
                            current_points[i] += perturbation_tangent
                            current_points[i] = self.project_to_sphere(current_points[i:i+1])[0]
            else:
                temperature = max(min_temp, temperature * cooling_rate)

            # Periodic local refinement with enhanced strategy
            if iteration % 1500 == 0 and iteration > 0:
                temp_points = current_points.copy()
                local_improved = False

                # Enhanced local search with multiple passes
                for _ in range(200):  # More extensive search
                    for i in range(len(temp_points)):
                        for j in range(3):
                            old_val = temp_points[i, j]
                            for delta in [-0.002, 0.002, -0.001, 0.001]:
                                temp_points[i, j] = old_val + delta
                                temp_points[i] = self.project_to_sphere(temp_points[i:i+1])[0]
                                new_ratio = self.compute_min_max_ratio(temp_points)
                                if new_ratio > best_ratio:
                                    best_ratio = new_ratio
                                    best_points = temp_points.copy()
                                    local_improved = True
                                else:
                                    temp_points[i, j] = old_val  # Revert

                if local_improved:
                    current_points = best_points.copy()
                else:
                    current_points = self.local_search_refinement(current_points, 100)

            # Early stop if performance is stagnant
            if stagnation_counter > 3000:
                break

        return best_points, best_ratio

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    optimizer = SphericalPointOptimizer()

    # Generate diverse initial configurations
    initial_configs = optimizer.generate_initial_configurations(7)

    best_overall_points = None
    best_overall_ratio = 0.0

    # Try each initial configuration with enhanced parameters
    for i, config in enumerate(initial_configs):
        # Apply simulated annealing optimization with better parameters
        temp_params = {
            'initial_temp': 0.2,
            'cooling_rate': 0.9993,
            'min_temp': 1e-7,
            'stagnation_threshold': 1500,
            'diversity_factor': 0.25
        }

        optimized_points, final_ratio = optimizer.adaptive_simulated_annealing(
            config,
            max_iter=10000,
            temp_schedule_params=temp_params
        )

        if final_ratio > best_overall_ratio:
            best_overall_ratio = final_ratio
            best_overall_points = optimized_points.copy()

    # Final refinement with local search
    if best_overall_points is not None:
        best_overall_points = optimizer.local_search_refinement(best_overall_points, 300)
        final_ratio = optimizer.compute_min_max_ratio(best_overall_points)
        if final_ratio > best_overall_ratio:
            best_overall_ratio = final_ratio

    # L-BFGS refinement for final polish with increased precision
    def objective_function(x_flat: np.ndarray) -> float:
        points = x_flat.reshape(-1, 3)
        return -optimizer.compute_min_max_ratio(points)

    try:
        x0 = best_overall_points.flatten()
        result = minimize(
            objective_function,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-10}
        )

        refined_points = result.x.reshape(-1, 3)
        refined_points = optimizer.project_to_sphere(refined_points)
        ratio = optimizer.compute_min_max_ratio(refined_points)

        if ratio > best_overall_ratio:
            best_overall_points = refined_points
    except:
        pass

    # Ensure we return a valid point set
    if best_overall_points is None:
        # Fallback to fibonacci sphere with refinement
        fallback_points = optimizer.fibonacci_sphere(14)
        fallback_points = optimizer.project_to_sphere(fallback_points)
        best_overall_points = optimizer.local_search_refinement(fallback_points, 100)

    return best_overall_points

# EVOLVE-BLOCK-END