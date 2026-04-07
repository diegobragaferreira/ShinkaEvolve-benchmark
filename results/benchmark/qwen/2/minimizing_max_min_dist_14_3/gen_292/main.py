# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist
import math
import random
from typing import Tuple, List, Callable
import time

class PointConfiguration:
    """Manages point configurations and their properties."""

    def __init__(self, points: np.ndarray):
        self.points = points.astype(np.float64)
        self._ratio = None
        self._distances = None

    @property
    def ratio(self) -> float:
        """Cached computation of min/max ratio."""
        if self._ratio is None:
            self._ratio = self._compute_ratio()
        return self._ratio

    @property
    def distances(self) -> np.ndarray:
        """Cached distance matrix."""
        if self._distances is None:
            self._distances = pdist(self.points)
        return self._distances

    def _compute_ratio(self) -> float:
        """Compute the ratio of minimum to maximum distances."""
        if len(self.points) < 2:
            return 0.0
        distances = self.distances
        max_dist = np.max(distances)
        if max_dist <= 0:
            return 0.0
        return np.min(distances) / max_dist

    def update_points(self, new_points: np.ndarray) -> 'PointConfiguration':
        """Create a new configuration with updated points."""
        new_config = PointConfiguration(new_points)
        new_config._ratio = None
        new_config._distances = None
        return new_config

    def copy(self) -> 'PointConfiguration':
        """Create a deep copy of the configuration."""
        return PointConfiguration(self.points.copy())

    @classmethod
    def from_array(cls, points: np.ndarray) -> 'PointConfiguration':
        """Create configuration from array."""
        return cls(points)

class OptimizationStrategy:
    """Base class for optimization strategies."""

    def optimize(self, config: PointConfiguration, max_iter: int) -> Tuple[PointConfiguration, float]:
        """Optimize the given configuration."""
        raise NotImplementedError

class SimulatedAnnealingStrategy(OptimizationStrategy):
    """Adaptive simulated annealing optimization strategy."""

    def __init__(self, **kwargs):
        self.cooling_rate = kwargs.get('cooling_rate', 0.9995)
        self.min_temp = kwargs.get('min_temp', 1e-6)
        self.max_stagnation = kwargs.get('max_stagnation', 500)
        self.diversity_prob = kwargs.get('diversity_prob', 0.15)
        self.initial_temp = kwargs.get('initial_temp', 0.1)

    def optimize(self, config: PointConfiguration, max_iter: int) -> Tuple[PointConfiguration, float]:
        """Optimize using adaptive simulated annealing."""
        current_config = config.copy()
        best_config = current_config.copy()
        best_ratio = best_config.ratio

        temperature = self.initial_temp
        stagnation_counter = 0
        recent_improvements = []
        avg_improvement_window = 100

        for iteration in range(max_iter):
            old_config = current_config.copy()

            # Point selection and perturbation
            idx = np.random.randint(len(current_config.points))

            # Adaptive perturbation calculation
            base_perturbation = 0.02 * (1 - best_ratio) + 0.001
            adaptive_factor = 1.0 - (iteration / max_iter) * 0.5
            perturbation_size = base_perturbation * adaptive_factor * (1 + 0.3 * np.random.random())

            perturbation = np.random.normal(0, perturbation_size, 3)

            # Tangent plane projection
            current_point = current_config.points[idx]
            projection_factor = np.dot(perturbation, current_point)
            perturbation_tangent = perturbation - projection_factor * current_point

            # Apply perturbation
            current_config.points[idx] += perturbation_tangent
            current_config.points[idx] = self._project_to_sphere(current_config.points[idx:idx+1])[0]

            # Compute new ratio
            new_ratio = current_config.ratio

            # Accept or reject based on Metropolis criterion
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_config = current_config.copy()
                stagnation_counter = 0
            elif np.random.random() < np.exp((new_ratio - best_ratio) / temperature):
                pass
            else:
                current_config = old_config

            # Adaptive cooling
            improvement = new_ratio - best_ratio if 'best_ratio' in locals() else 0
            recent_improvements.append(1 if improvement > 1e-10 else 0)

            if len(recent_improvements) > avg_improvement_window:
                recent_improvements.pop(0)

            # Dynamic cooling adjustment
            avg_improvement = np.mean(recent_improvements) if recent_improvements else 0
            if avg_improvement < 1e-6 and stagnation_counter > 100:
                self.cooling_rate = max(0.999, self.cooling_rate * 0.99)
            elif avg_improvement > 1e-4:
                self.cooling_rate = min(0.9999, self.cooling_rate * 1.01)

            # Standard cooling logic
            stagnation_counter += 1
            if stagnation_counter > self.max_stagnation:
                temperature = max(self.min_temp, temperature * 0.95)
                stagnation_counter = 0

                # Diversity injection
                for i in range(len(current_config.points)):
                    if np.random.random() < self.diversity_prob:
                        perturbation = np.random.normal(0, 0.005, 3)
                        current_point = current_config.points[i]
                        projection_factor = np.dot(perturbation, current_point)
                        perturbation_tangent = perturbation - projection_factor * current_point
                        current_config.points[i] += perturbation_tangent
                        current_config.points[i] = self._project_to_sphere(current_config.points[i:i+1])[0]
            else:
                temperature = max(self.min_temp, temperature * self.cooling_rate)

        return best_config, best_ratio

    def _project_to_sphere(self, points: np.ndarray) -> np.ndarray:
        """Project points onto unit sphere."""
        norms = np.linalg.norm(points, axis=1)
        norms = np.where(norms == 0, 1, norms)
        return points / norms[:, np.newaxis]

class ClusterBasedStrategy(OptimizationStrategy):
    """Cluster-based force field optimization strategy."""

    def __init__(self, **kwargs):
        self.learning_rate = kwargs.get('learning_rate', 0.01)
        self.damping_factor = kwargs.get('damping_factor', 0.95)
        self.min_temperature = kwargs.get('min_temperature', 1e-8)
        self.max_stagnation = kwargs.get('max_stagnation', 1000)
        self.convergence_threshold = kwargs.get('convergence_threshold', 1e-6)
        self.convergence_window = kwargs.get('convergence_window', 50)

    def optimize(self, config: PointConfiguration, max_iter: int) -> Tuple[PointConfiguration, float]:
        """Optimize using cluster-based force field dynamics."""
        current_config = config.copy()
        best_config = current_config.copy()
        best_ratio = best_config.ratio

        stagnation_counter = 0
        recent_ratios = []

        for iteration in range(max_iter):
            # Compute forces (simplified version for performance)
            forces = self._compute_cluster_forces(current_config.points)

            # Apply forces with adaptive learning rate
            current_learning_rate = self.learning_rate * (1.0 - iteration / max_iter) * 0.5 + 0.005

            # Apply forces
            velocities = forces * current_learning_rate
            current_config.points += velocities

            # Project back to sphere
            current_config.points = self._project_to_sphere(current_config.points)

            # Compute new ratio
            new_ratio = current_config.ratio

            # Update best if improved
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_config = current_config.copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Add noise for escaping local minima
            if stagnation_counter > 100 and stagnation_counter % 200 == 0:
                noise = np.random.normal(0, 0.01, current_config.points.shape)
                current_config.points += noise
                current_config.points = self._project_to_sphere(current_config.points)

            # Convergence checking
            recent_ratios.append(new_ratio)
            if len(recent_ratios) > self.convergence_window:
                recent_ratios.pop(0)
                if (len(recent_ratios) >= 2 and
                    abs(recent_ratios[-1] - recent_ratios[0]) < self.convergence_threshold):
                    break

            # Reduce learning rate gradually
            if iteration % 1000 == 0 and iteration > 0:
                self.learning_rate *= 0.95

        return best_config, best_ratio

    def _compute_cluster_forces(self, points: np.ndarray) -> np.ndarray:
        """Compute cluster-based repulsion forces."""
        n = len(points)
        forces = np.zeros_like(points)

        # Compute distance matrix
        try:
            distances = cdist(points, points)
            np.fill_diagonal(distances, np.inf)

            if len(distances) > 0:
                # Simple repulsion based on inverse square law
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            diff = points[i] - points[j]
                            dist_sq = np.dot(diff, diff)

                            if dist_sq > 1e-12:
                                force_magnitude = 1.0 / (dist_sq * np.sqrt(dist_sq))
                                force_direction = diff / np.sqrt(dist_sq)
                                forces[i] += force_magnitude * force_direction
        except:
            pass

        return forces

    def _project_to_sphere(self, points: np.ndarray) -> np.ndarray:
        """Project points onto unit sphere."""
        norms = np.linalg.norm(points, axis=1)
        norms = np.where(norms == 0, 1, norms)
        return points / norms[:, np.newaxis]

class LocalSearchStrategy(OptimizationStrategy):
    """Local search refinement strategy."""

    def __init__(self, **kwargs):
        self.iterations = kwargs.get('iterations', 100)

    def optimize(self, config: PointConfiguration, max_iter: int) -> Tuple[PointConfiguration, float]:
        """Refine configuration using local search."""
        current_config = config.copy()
        current_ratio = current_config.ratio

        for _ in range(self.iterations):
            # Try small adjustments to each coordinate
            for i in range(len(current_config.points)):
                for j in range(3):
                    old_val = current_config.points[i, j]
                    for delta in [-0.0005, 0.0005]:
                        current_config.points[i, j] = old_val + delta
                        current_config.points[i] = self._project_to_sphere(current_config.points[i:i+1])[0]
                        new_ratio = current_config.ratio
                        if new_ratio > current_ratio:
                            current_ratio = new_ratio
                        else:
                            current_config.points[i, j] = old_val  # Revert

        return current_config, current_ratio

    def _project_to_sphere(self, points: np.ndarray) -> np.ndarray:
        """Project points onto unit sphere."""
        norms = np.linalg.norm(points, axis=1)
        norms = np.where(norms == 0, 1, norms)
        return points / norms[:, np.newaxis]

class SphericalPointOptimizer:
    """Main optimizer class with evolutionary pipeline."""

    def __init__(self):
        self.strategies = {
            'sa': SimulatedAnnealingStrategy(),
            'cluster': ClusterBasedStrategy(),
            'local': LocalSearchStrategy()
        }
        # Set of seeds for multi-start diversification
        self.seeds = [42, 123, 456, 789, 999, 1001, 2002, 3003]

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

    def generate_initial_configurations(self, num_configs: int = 8) -> List[PointConfiguration]:
        """Generate diverse initial configurations."""
        configs = []

        # Multiple Fibonacci configurations with different seeds
        for i in range(num_configs // 2):
            seed_index = i % len(self.seeds)
            np.random.seed(self.seeds[seed_index])
            config = self.fibonacci_sphere(14)
            config = self.project_to_sphere(config)
            configs.append(PointConfiguration(config))

        # Icosahedron-based configuration
        ico_vertices = self.icosahedron_vertices()
        if len(ico_vertices) >= 14:
            ico_config = ico_vertices[:14].copy()
        else:
            ico_config = np.vstack([ico_vertices, ico_vertices[:(14-len(ico_vertices))]]).copy()
        ico_config = self.project_to_sphere(ico_config)
        configs.append(PointConfiguration(ico_config))

        # Perturbed configurations with different seeds
        for i in range(num_configs // 4):
            seed_index = (i + len(self.seeds)//2) % len(self.seeds)
            np.random.seed(self.seeds[seed_index])
            perturbed_config = ico_config + np.random.normal(0, 0.05, ico_config.shape)
            perturbed_config = self.project_to_sphere(perturbed_config)
            configs.append(PointConfiguration(perturbed_config))

        # Additional perturbed Fibonacci configurations
        for i in range(num_configs // 4):
            seed_index = (i + len(self.seeds)//4) % len(self.seeds)
            np.random.seed(self.seeds[seed_index])
            fib_config = self.fibonacci_sphere(14)
            fib_config = fib_config + np.random.normal(0, 0.03, fib_config.shape)
            fib_config = self.project_to_sphere(fib_config)
            configs.append(PointConfiguration(fib_config))

        return configs

    def optimize_with_strategy(self, config: PointConfiguration,
                              strategy_name: str, max_iter: int = 10000) -> Tuple[PointConfiguration, float]:
        """Optimize using specific strategy."""
        strategy = self.strategies.get(strategy_name)
        if strategy is None:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        return strategy.optimize(config, max_iter)

    def adaptive_simulated_annealing_optimize(self, config: PointConfiguration,
                                            max_iter: int = 10000) -> Tuple[PointConfiguration, float]:
        """Enhanced simulated annealing with adaptive cooling and checkpointing."""
        current_config = config.copy()
        best_config = current_config.copy()
        best_ratio = best_config.ratio

        # Initialize parameters
        temperature = 0.1
        min_temp = 1e-8
        base_cooling_rate = 0.9995
        max_stagnation = 1000
        stagnation_counter = 0
        last_improvement = 0
        improvement_history = []
        checkpoint_best = best_config.copy()
        checkpoint_ratio = best_ratio
        checkpoint_interval = 2000

        for iteration in range(max_iter):
            old_config = current_config.copy()

            # Point selection and perturbation
            idx = np.random.randint(len(current_config.points))

            # Adaptive perturbation calculation
            base_perturbation = 0.02 * (1 - best_ratio) + 0.001
            adaptive_factor = 1.0 - (iteration / max_iter) * 0.5
            perturbation_size = base_perturbation * adaptive_factor * (1 + 0.3 * np.random.random())

            perturbation = np.random.normal(0, perturbation_size, 3)

            # Tangent plane projection
            current_point = current_config.points[idx]
            projection_factor = np.dot(perturbation, current_point)
            perturbation_tangent = perturbation - projection_factor * current_point

            # Apply perturbation
            current_config.points[idx] += perturbation_tangent
            current_config.points[idx] = self._project_to_sphere(current_config.points[idx:idx+1])[0]

            # Compute new ratio
            new_ratio = current_config.ratio

            # Accept or reject based on Metropolis criterion
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_config = current_config.copy()
                last_improvement = iteration
                stagnation_counter = 0
                improvement_history.append(1)
            elif np.random.random() < np.exp((new_ratio - best_ratio) / temperature):
                improvement_history.append(1)
            else:
                current_config = old_config
                improvement_history.append(0)

            # Keep history window bounded
            if len(improvement_history) > 1000:
                improvement_history.pop(0)

            # Adaptive cooling based on recent improvement rate
            if len(improvement_history) >= 500:
                recent_improvement_rate = sum(improvement_history[-500:]) / 500.0
                if recent_improvement_rate < 0.05:  # Low improvement rate
                    # Increase cooling rate to escape local optima
                    cooling_rate = min(0.99995, base_cooling_rate * 1.05)
                elif recent_improvement_rate > 0.2:  # High improvement rate
                    # Decrease cooling rate to explore more carefully
                    cooling_rate = max(0.9994, base_cooling_rate * 0.95)
                else:
                    cooling_rate = base_cooling_rate
            else:
                cooling_rate = base_cooling_rate

            # Apply cooling
            temperature = max(min_temp, temperature * cooling_rate)

            # Checkpointing every checkpoint_interval iterations
            if iteration > 0 and iteration % checkpoint_interval == 0:
                if best_ratio > checkpoint_ratio:
                    checkpoint_best = best_config.copy()
                    checkpoint_ratio = best_ratio

            # Early stopping conditions
            if iteration - last_improvement > max_stagnation:
                break

            # Diversity injection
            if stagnation_counter > 500 and iteration % 100 == 0:
                for i in range(len(current_config.points)):
                    if np.random.random() < 0.2:  # Higher probability for diversity
                        perturbation = np.random.normal(0, 0.005, 3)
                        current_point = current_config.points[i]
                        projection_factor = np.dot(perturbation, current_point)
                        perturbation_tangent = perturbation - projection_factor * current_point
                        current_config.points[i] += perturbation_tangent
                        current_config.points[i] = self._project_to_sphere(current_config.points[i:i+1])[0]

            stagnation_counter += 1

        # Restore best checkpoint if it's better
        if checkpoint_ratio > best_ratio:
            best_config = checkpoint_best.copy()
            best_ratio = checkpoint_ratio

        return best_config, best_ratio

    def _project_to_sphere(self, points: np.ndarray) -> np.ndarray:
        """Project points onto unit sphere."""
        norms = np.linalg.norm(points, axis=1)
        norms = np.where(norms == 0, 1, norms)
        return points / norms[:, np.newaxis]

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    optimizer = SphericalPointOptimizer()

    # Generate diverse initial configurations
    initial_configs = optimizer.generate_initial_configurations(10)

    best_overall_config = None
    best_overall_ratio = 0.0

    # Multi-start: Try multiple configurations with different optimization strategies
    for i, config in enumerate(initial_configs):
        # Use different strategies based on initial configuration index
        strategy_idx = i % 3
        if strategy_idx == 0:
            # Use adaptive SA for this configuration
            optimized_config, final_ratio = optimizer.adaptive_simulated_annealing_optimize(config, 10000)
        elif strategy_idx == 1:
            # Use cluster-based optimization
            optimized_config, final_ratio = optimizer.optimize_with_strategy(config, 'cluster', 8000)
        else:
            # Use standard SA
            optimized_config, final_ratio = optimizer.optimize_with_strategy(config, 'sa', 8000)

        if final_ratio > best_overall_ratio:
            best_overall_ratio = final_ratio
            best_overall_config = optimized_config.copy()

    # Final refinement with all strategies to ensure we get the best possible solution
    if best_overall_config is not None:
        # Apply sequential refinement: cluster → local → SA
        refined_config, final_ratio = optimizer.optimize_with_strategy(
            best_overall_config, 'cluster', 5000)
        if final_ratio > best_overall_ratio:
            best_overall_ratio = final_ratio
            best_overall_config = refined_config

        refined_config, final_ratio = optimizer.optimize_with_strategy(
            best_overall_config, 'local', 200)
        if final_ratio > best_overall_ratio:
            best_overall_ratio = final_ratio
            best_overall_config = refined_config

        # Final adaptive SA refinement
        refined_config, final_ratio = optimizer.adaptive_simulated_annealing_optimize(
            best_overall_config, 5000)
        if final_ratio > best_overall_ratio:
            best_overall_ratio = final_ratio
            best_overall_config = refined_config

    # L-BFGS refinement for final polish
    def objective_function(x_flat: np.ndarray) -> float:
        points = x_flat.reshape(-1, 3)
        # Create temporary configuration for ratio calculation
        temp_config = PointConfiguration(points)
        return -temp_config.ratio

    try:
        x0 = best_overall_config.points.flatten()
        result = minimize(
            objective_function,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9}
        )

        refined_points = result.x.reshape(-1, 3)
        refined_points = optimizer.project_to_sphere(refined_points)

        # Recompute ratio properly
        final_config = PointConfiguration(refined_points)
        ratio = final_config.ratio

        if ratio > best_overall_ratio:
            best_overall_config = final_config
    except:
        pass

    return best_overall_config.points

# EVOLVE-BLOCK-END