# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time
import random
from typing import List, Tuple, Optional

class PointSetOptimizer:
    """Main optimizer class that orchestrates the point placement optimization process."""
    
    def __init__(self, n_points: int = 16, dimensions: int = 2, seed: int = 42):
        self.n_points = n_points
        self.dimensions = dimensions
        self.seed = seed
        np.random.seed(seed)
        
    def compute_min_max_ratio(self, points: np.ndarray) -> float:
        """Compute the ratio of minimum to maximum distances between all point pairs."""
        if len(points) < 2:
            return 0.0

        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        if max_dist == 0:
            return 0.0

        return min_dist / max_dist
    
    def compute_min_max_ratio_with_boundary_penalty(self, points: np.ndarray, penalty_factor: float = 1000.0) -> float:
        """Compute the ratio with boundary penalties to avoid edge violations."""
        penalty = 0.0
        for point in points:
            dist_to_left = point[0]
            dist_to_right = 1.0 - point[0]
            dist_to_bottom = point[1]
            dist_to_top = 1.0 - point[1]
            
            min_dist_to_edge = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
            
            if min_dist_to_edge < 0.01:
                penalty += penalty_factor * (0.01 - min_dist_to_edge)

        ratio = self.compute_min_max_ratio(points)
        return ratio - penalty / len(points)
    
    def generate_initial_configuration(self, config_type: str) -> np.ndarray:
        """Generate a specific initial configuration."""
        if config_type == "hexagonal":
            return self._generate_hexagonal_grid()
        elif config_type == "fibonacci":
            return self._generate_fibonacci_spiral_points()
        elif config_type == "random":
            return self._generate_random_points()
        elif config_type == "structured":
            return self._generate_structured_grid()
        else:
            raise ValueError(f"Unknown configuration type: {config_type}")
    
    def _generate_fibonacci_spiral_points(self) -> np.ndarray:
        """Generate points using Fibonacci spiral on sphere projected to 2D."""
        points = np.zeros((self.n_points, self.dimensions))
        golden_ratio = (1 + np.sqrt(5)) / 2

        for i in range(self.n_points):
            z = 1 - (i / (self.n_points - 1)) * 2
            radius = np.sqrt(1 - z*z)
            theta = np.arccos(z)
            phi = (i * golden_ratio) % (2 * np.pi)

            x = radius * np.cos(phi)
            y = radius * np.sin(phi)

            x_norm = (x + 1) / 2
            y_norm = (y + 1) / 2

            points[i] = [x_norm, y_norm]

        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        return points
    
    def _generate_hexagonal_grid(self) -> np.ndarray:
        """Generate points in a hexagonal grid pattern."""
        points = []
        rows = 4
        cols = 4
        spacing = np.sqrt(1.0 / (np.sqrt(3) * self.n_points)) * 1.5

        spacing_x = spacing
        spacing_y = spacing * np.sqrt(3) / 2

        for i in range(rows):
            for j in range(cols):
                if len(points) < self.n_points:
                    x = j * spacing_x + (i % 2) * spacing_x / 2
                    y = i * spacing_y

                    asymmetry_factor = 0.015
                    x_offset = asymmetry_factor * np.sin(i * 1.7) * np.cos(j * 3.1) * np.sin((i+j) * 0.5)
                    y_offset = asymmetry_factor * np.cos(i * 2.3) * np.sin(j * 1.9) * np.cos((i-j) * 0.7)

                    points.append([x + x_offset, y + y_offset])

        points = np.array(points[:self.n_points])
        points[:, 0] = np.clip(points[:, 0], 0, 1)
        points[:, 1] = np.clip(points[:, 1], 0, 1)
        points += np.random.normal(0, 0.002, points.shape)
        points = np.clip(points, 0, 1)
        return points
    
    def _generate_random_points(self) -> np.ndarray:
        """Generate random points."""
        return np.random.rand(self.n_points, self.dimensions)
    
    def _generate_structured_grid(self) -> np.ndarray:
        """Generate a structured grid pattern."""
        points = []
        for i in range(4):
            for j in range(4):
                x_base = 0.1 + j * 0.22
                y_base = 0.1 + i * 0.22
                
                asym_x = np.sin(i * 0.7) * 0.01 * (1 + j * 0.05)
                asym_y = np.cos(j * 0.5) * 0.01 * (1 + i * 0.05)
                
                x = x_base + asym_x + np.random.normal(0, 0.003)
                y = y_base + asym_y + np.random.normal(0, 0.003)
                points.append([x, y])
        
        points = np.array(points)
        points = np.clip(points, 0, 1)
        return points
    
    def _compute_gradient_approximation(self, points: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """Compute approximate gradient using finite differences."""
        grad = np.zeros_like(points)
        base_ratio = self.compute_min_max_ratio(points)

        for i in range(len(points)):
            for j in range(len(points[i])):
                points_plus = points.copy()
                points_minus = points.copy()
                points_plus[i,j] += epsilon
                points_minus[i,j] -= epsilon

                ratio_plus = self.compute_min_max_ratio(points_plus)
                ratio_minus = self.compute_min_max_ratio(points_minus)
                grad[i,j] = (ratio_plus - ratio_minus) / (2 * epsilon)

        return grad
    
    def optimize_with_gradient_refinement(self, initial_points: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Apply gradient-based refinement to improve solution quality."""
        points = initial_points.copy()
        current_ratio = self.compute_min_max_ratio(points)

        for iteration in range(max_iter):
            grad = self._compute_gradient_approximation(points)
            learning_rate = 0.02 * (1.0 - iteration / max_iter * 0.5)
            points = points + learning_rate * grad
            points = np.clip(points, 0, 1)
            
            new_ratio = self.compute_min_max_ratio(points)
            if new_ratio > current_ratio:
                current_ratio = new_ratio
            else:
                learning_rate *= 0.5
                points = points - learning_rate * grad
                points = np.clip(points, 0, 1)

            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-6:
                break

        return points
    
    def optimize_with_lbfgs(self, initial_points: np.ndarray) -> np.ndarray:
        """Optimize using L-BFGS-B method for local refinement."""
        def objective_function(params):
            points = params.reshape(-1, 2)
            ratio = self.compute_min_max_ratio(points)
            return -ratio

        initial_params = initial_points.flatten()
        bounds = [(0, 1)] * (self.n_points * self.dimensions)

        try:
            result = minimize(
                objective_function,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12},
                callback=None
            )

            optimized_points = result.x.reshape(-1, 2)
            optimized_points = np.clip(optimized_points, 0, 1)
            return optimized_points
        except Exception:
            return initial_points
    
    def optimize_points(self, initial_points: np.ndarray, max_time: float = 175) -> np.ndarray:
        """Optimize point positions using enhanced simulated annealing."""
        start_time = time.time()
        points = np.clip(initial_points, 0, 1)
        current_ratio = self.compute_min_max_ratio_with_boundary_penalty(points)

        temperature = 1.0
        cooling_rate = 0.99995
        min_temperature = 1e-8
        max_iterations = 500000
        iteration = 0

        best_points = points.copy()
        best_ratio = current_ratio

        recent_improvements = []
        patience = 0
        max_patience = 1000

        while temperature > min_temperature and iteration < max_iterations and (time.time() - start_time) < max_time:
            candidate_points = points.copy()

            move_type = random.random()
            if move_type < 0.3:
                progress = iteration / max_iterations
                if progress < 0.3:
                    num_points_to_move = random.randint(3, 5)
                elif progress < 0.7:
                    num_points_to_move = random.randint(2, 4)
                else:
                    num_points_to_move = random.randint(1, 3)

                selected_indices = random.sample(range(len(points)), num_points_to_move)
                centroid = np.mean(candidate_points[selected_indices], axis=0)
                move_vector = np.random.normal(0, 0.015, 2)
                new_centroid = np.clip(centroid + move_vector, 0, 1)
                delta = new_centroid - centroid

                for idx in selected_indices:
                    candidate_points[idx] += delta
            elif move_type < 0.4:
                candidate_points += np.random.normal(0, 0.01, candidate_points.shape)
            else:
                idx = np.random.randint(0, len(points))
                candidate_points[idx] += np.random.normal(0, 0.02, 2)

            candidate_points = np.clip(candidate_points, 0, 1)
            candidate_ratio = self.compute_min_max_ratio_with_boundary_penalty(candidate_points)

            if candidate_ratio > current_ratio or np.random.rand() < np.exp((candidate_ratio - current_ratio) / temperature):
                points = candidate_points
                current_ratio = candidate_ratio

                if current_ratio > best_ratio:
                    best_points = points.copy()
                    best_ratio = current_ratio
                    recent_improvements = []
                    patience = 0
                else:
                    patience += 1
                    recent_improvements.append(current_ratio)
                    if len(recent_improvements) > 50:
                        recent_improvements.pop(0)
            else:
                patience += 1

            if patience > max_patience:
                if len(recent_improvements) > 10:
                    recent_avg = np.mean(recent_improvements[-10:])
                    if recent_avg > 0.99 * best_ratio:
                        break

            if temperature > 0.1:
                temperature *= cooling_rate
            else:
                temperature *= 0.999995

            iteration += 1

        # Final local refinement pipeline
        lbfgs_points = self.optimize_with_lbfgs(best_points)
        lbfgs_ratio = self.compute_min_max_ratio_with_boundary_penalty(lbfgs_points)

        gradient_points = self.optimize_with_gradient_refinement(lbfgs_points)
        gradient_ratio = self.compute_min_max_ratio_with_boundary_penalty(gradient_points)

        final_points = self.optimize_with_lbfgs(gradient_points)
        final_ratio = self.compute_min_max_ratio_with_boundary_penalty(final_points)

        best_of_all = max(lbfgs_ratio, gradient_ratio, final_ratio)
        if best_of_all == lbfgs_ratio:
            return lbfgs_points
        elif best_of_all == gradient_ratio:
            return gradient_points
        else:
            return final_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    # Define initial configuration types to try
    config_types = ["hexagonal", "fibonacci", "random", "structured"]
    
    # Create optimizer instance
    optimizer = PointSetOptimizer(n_points=16, dimensions=2, seed=42)
    
    # Generate multiple initial configurations
    initial_configs = [optimizer.generate_initial_configuration(config_type) for config_type in config_types]
    
    # Run optimization from each configuration
    best_final_points = None
    best_final_ratio = -np.inf

    time_per_run = 175 / len(initial_configs)
    
    for i, initial_config in enumerate(initial_configs):
        config_points = optimizer.optimize_points(initial_config, max_time=time_per_run * 0.9)
        config_ratio = optimizer.compute_min_max_ratio_with_boundary_penalty(config_points)

        if config_ratio > best_final_ratio:
            best_final_ratio = config_ratio
            best_final_points = config_points.copy()

    # Final validation
    if best_final_points is None:
        fallback_points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25
                fallback_points.append([x, y])
        best_final_points = np.array(fallback_points)
        best_final_points = np.clip(best_final_points + np.random.normal(0, 0.01, (16, 2)), 0, 1)

    return best_final_points

# EVOLVE-BLOCK-END