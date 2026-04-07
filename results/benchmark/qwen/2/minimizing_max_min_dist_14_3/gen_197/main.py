# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist
import math
import random
from typing import Tuple, List

class AdaptiveVoronoiOptimizer:
    def __init__(self):
        self.best_points = None
        self.best_ratio = 0.0
        self.dimension = 3
        self.num_points = 14

    def compute_min_max_ratio(self, points: np.ndarray) -> float:
        """Compute the ratio of minimum to maximum distances."""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        return np.min(distances) / np.max(distances) if np.max(distances) > 0 else 0

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

    def project_to_sphere(self, points: np.ndarray) -> np.ndarray:
        """Project points onto unit sphere."""
        norms = np.linalg.norm(points, axis=1)
        norms = np.where(norms == 0, 1, norms)
        return points / norms[:, np.newaxis]

    def compute_voronoi_forces(self, points: np.ndarray) -> np.ndarray:
        """Compute forces based on Voronoi-like interactions."""
        n = len(points)
        forces = np.zeros_like(points)
        
        # Compute distance matrix
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        
        # Apply inverse square repulsion for all pairs
        for i in range(n):
            for j in range(n):
                if i != j:
                    diff = points[i] - points[j]
                    dist_sq = np.dot(diff, diff)
                    
                    # Avoid division by zero
                    if dist_sq > 1e-12:
                        # Repulsive force inversely proportional to square of distance
                        force_magnitude = 1.0 / (dist_sq * np.sqrt(dist_sq))
                        force_direction = diff / np.sqrt(dist_sq)
                        forces[i] += force_magnitude * force_direction
        
        return forces

    def generate_initial_configurations(self, num_configs: int = 8) -> List[np.ndarray]:
        """Generate diverse initial configurations."""
        configs = []
        
        # Base icosahedron configuration
        ico_vertices = self.icosahedron_vertices()
        if len(ico_vertices) >= 14:
            ico_config = ico_vertices[:14].copy()
        else:
            ico_config = np.vstack([ico_vertices, ico_vertices[:(14-len(ico_vertices))]]).copy()
        ico_config = self.project_to_sphere(ico_config)
        configs.append(ico_config)

        # Perturbed icosahedron variants
        for i in range(3):
            np.random.seed(i * 1000 + 42)
            base_config = ico_config.copy()
            noise_magnitude = 0.03 + i * 0.01  # Increasing noise
            noise = np.random.normal(0, noise_magnitude, base_config.shape)
            noisy_config = base_config + noise
            noisy_config = self.project_to_sphere(noisy_config)
            configs.append(noisy_config)

        # Fibonacci sphere configurations
        for i in range(3):
            np.random.seed(i * 2000 + 123)
            fib_config = self.fibonacci_sphere(14)
            fib_config = self.project_to_sphere(fib_config)
            configs.append(fib_config)

        # Random perturbed Fibonacci
        np.random.seed(999)
        random_fib = self.fibonacci_sphere(14)
        random_fib = self.project_to_sphere(random_fib)
        noise = np.random.normal(0, 0.05, random_fib.shape)
        perturbed_random = random_fib + noise
        perturbed_random = self.project_to_sphere(perturbed_random)
        configs.append(perturbed_random)

        return configs

    def adaptive_perturbation(self, points: np.ndarray, temp: float, current_ratio: float) -> np.ndarray:
        """Apply adaptive perturbations based on current solution quality."""
        new_points = points.copy()
        
        # Analyze current distribution to select which point to perturb
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        
        # Find points with minimum distances (potential bottlenecks)
        min_distances = np.min(distances, axis=1)
        
        # Prefer to perturb points that are near minimum distances
        # or points that have high influence on the overall configuration
        if np.random.rand() < 0.7:  # 70% chance to target bottlenecks
            # Select point with smallest minimum distance
            target_idx = np.argmin(min_distances)
        else:
            # Random selection otherwise
            target_idx = np.random.randint(len(points))
        
        # Adaptive perturbation size based on temperature and solution quality
        # Higher temperature or lower ratio = bigger perturbations
        adaptive_scale = temp * (1.0 - current_ratio) * 0.1 + 0.001
        
        # Generate perturbation in tangent plane
        delta = np.random.normal(0, adaptive_scale, 3)
        
        # Project perturbation to tangent plane
        current_point = new_points[target_idx]
        projection_factor = np.dot(delta, current_point)
        tangent_delta = delta - projection_factor * current_point
        
        # Apply perturbation
        new_points[target_idx] += tangent_delta
        
        # Project back to unit sphere
        new_points[target_idx] = self.project_to_sphere(new_points[target_idx:target_idx+1])[0]
        
        return new_points

    def adaptive_optimization(self, initial_points: np.ndarray, max_iter: int = 3000) -> Tuple[np.ndarray, float]:
        """Optimize points using adaptive simulated annealing with smart cooling."""
        current_points = initial_points.copy()
        current_points = self.project_to_sphere(current_points)

        best_points = current_points.copy()
        best_ratio = self.compute_min_max_ratio(current_points)

        # Enhanced parameters
        temperature = 0.1
        cooling_rate = 0.99995  # Faster cooling
        min_temperature = 1e-8
        max_stagnation = 500
        stagnation_counter = 0

        # Track convergence
        ratio_history = []
        last_improvement_iter = 0

        for iteration in range(max_iter):
            # Store previous state for potential rejection
            old_points = current_points.copy()
            old_ratio = self.compute_min_max_ratio(current_points)

            # Apply adaptive perturbation
            new_points = self.adaptive_perturbation(current_points, temperature, old_ratio)

            # Compute new ratio
            new_ratio = self.compute_min_max_ratio(new_points)

            # Accept or reject based on Metropolis criterion
            if new_ratio > best_ratio:
                # Always accept better solutions
                current_points = new_points
                current_ratio = new_ratio
                best_ratio = new_ratio
                best_points = new_points.copy()
                last_improvement_iter = iteration
                stagnation_counter = 0
                ratio_history.append(new_ratio)
            elif np.random.random() < np.exp((new_ratio - old_ratio) / temperature):
                # Accept worse solutions with probability
                current_points = new_points
                current_ratio = new_ratio
                ratio_history.append(new_ratio)
            else:
                # Revert to old state
                current_points = old_points
                current_ratio = old_ratio

            # Adaptive cooling based on recent improvement
            if len(ratio_history) > 10:
                recent_improvement = ratio_history[-1] - ratio_history[-10]
                if recent_improvement < 1e-7:
                    # Slow improvement, cool faster
                    cooling_rate = min(0.99999, cooling_rate * 1.001)
                elif recent_improvement > 1e-5:
                    # Fast improvement, cool slower  
                    cooling_rate = max(0.9999, cooling_rate * 0.999)
            temperature = max(min_temperature, temperature * cooling_rate)

            # Periodic local refinement
            if iteration % 500 == 0 and iteration > 0:
                local_refined = self.local_search_refinement(current_points, 100)
                local_ratio = self.compute_min_max_ratio(local_refined)
                if local_ratio > best_ratio:
                    current_points = local_refined
                    best_points = local_refined.copy()
                    best_ratio = local_ratio
                    last_improvement_iter = iteration

            # Check for stagnation and restart
            if iteration - last_improvement_iter > max_stagnation:
                # Restart from best known solution
                current_points = best_points.copy()
                current_ratio = best_ratio
                last_improvement_iter = iteration
                # Increase randomness during restart
                temperature = max(temperature * 0.5, 0.01)

            # Early stopping if converging
            if len(ratio_history) > 20:
                recent_changes = [ratio_history[i] - ratio_history[i-1] 
                                for i in range(len(ratio_history)-1, max(0, len(ratio_history)-10), -1)]
                if len(recent_changes) > 5:
                    avg_change = np.mean(np.abs(recent_changes[-5:]))
                    if avg_change < 1e-8:
                        break

        return best_points, best_ratio

    def local_search_refinement(self, points: np.ndarray, iterations: int = 100) -> np.ndarray:
        """Perform targeted local search refinement."""
        current_points = points.copy()
        current_ratio = self.compute_min_max_ratio(current_points)

        # Try more comprehensive local search
        for _ in range(iterations):
            # Try different perturbation sizes
            for delta_size in [0.0001, 0.0005, 0.001]:
                for i in range(len(current_points)):
                    for j in range(3):
                        old_val = current_points[i, j]
                        # Try multiple delta values
                        for delta in [-delta_size, delta_size]:
                            current_points[i, j] = old_val + delta
                            current_points[i] = self.project_to_sphere(current_points[i:i+1])[0]
                            new_ratio = self.compute_min_max_ratio(current_points)
                            if new_ratio > current_ratio:
                                current_ratio = new_ratio
                            else:
                                current_points[i, j] = old_val  # Revert

        return current_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    optimizer = AdaptiveVoronoiOptimizer()

    # Generate diverse initial configurations
    initial_configs = optimizer.generate_initial_configurations(8)

    best_overall_points = None
    best_overall_ratio = 0.0

    # Try each initial configuration
    for i, config in enumerate(initial_configs):
        # Apply adaptive optimization
        optimized_points, final_ratio = optimizer.adaptive_optimization(config, 3000)

        if final_ratio > best_overall_ratio:
            best_overall_ratio = final_ratio
            best_overall_points = optimized_points.copy()

    # Final refinement with more aggressive local search
    if best_overall_points is not None:
        best_overall_points = optimizer.local_search_refinement(best_overall_points, 300)
        final_ratio = optimizer.compute_min_max_ratio(best_overall_points)
        if final_ratio > best_overall_ratio:
            best_overall_ratio = final_ratio

    # L-BFGS refinement for final polish
    def objective_function(x_flat: np.ndarray) -> float:
        points = x_flat.reshape(-1, 3)
        return -optimizer.compute_min_max_ratio(points)

    try:
        x0 = best_overall_points.flatten()
        result = minimize(
            objective_function,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-9}
        )

        refined_points = result.x.reshape(-1, 3)
        refined_points = optimizer.project_to_sphere(refined_points)
        ratio = optimizer.compute_min_max_ratio(refined_points)

        if ratio > best_overall_ratio:
            best_overall_points = refined_points
    except:
        pass

    return best_overall_points

# EVOLVE-BLOCK-END
