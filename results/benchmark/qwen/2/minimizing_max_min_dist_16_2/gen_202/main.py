# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from numba import jit
import time
from typing import List, Tuple, Optional, Any

@jit(nopython=True)
def fast_pairwise_distances(points):
    """Fast computation of pairwise distances using numba."""
    n = points.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

class PointEvolutionOptimizer:
    """Stateful optimizer for maximizing min/max distance ratio in 2D point placement."""
    
    def __init__(self, num_points: int = 16, dimension: int = 2):
        self.num_points = num_points
        self.dimension = dimension
        self.bounds = [(0.001, 0.999) for _ in range(num_points * dimension)]
        self.best_solution = None
        self.best_ratio = -np.inf
        
    def _compute_ratio(self, points: np.ndarray) -> Tuple[float, float, float]:
        """Compute min/max distance ratio efficiently."""
        if len(points) < 2:
            return 0.0, 0.0, 0.0
        
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0, 0.0, 0.0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0.0, min_dist, max_dist
            
        ratio = min_dist / max_dist
        return ratio, min_dist, max_dist
    
    def _objective(self, x: np.ndarray) -> float:
        """Objective function to minimize (negative ratio)."""
        points = x.reshape(-1, self.dimension)
        ratio, _, _ = self._compute_ratio(points)
        return -ratio
    
    def _generate_hexagonal_lattice(self) -> np.ndarray:
        """Generate optimized hexagonal lattice configuration."""
        points = []
        rows = cols = 4
        
        spacing_x = 1.0 / (cols - 1) if cols > 1 else 1.0
        spacing_y = 1.0 / (rows - 1) if rows > 1 else 1.0
        
        spacing_x *= 0.88
        spacing_y *= 0.88
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= self.num_points:
                    break
                x_offset = spacing_x * 0.25 if i % 2 == 1 else 0.0
                x = (j * spacing_x) + x_offset
                y = i * spacing_y
                
                x = np.clip(x, 0.001, 0.999)
                y = np.clip(y, 0.001, 0.999)
                
                points.append([x, y])
        
        return np.array(points[:self.num_points])
    
    def _generate_fibonacci_distribution(self) -> np.ndarray:
        """Generate Fibonacci spiral distribution with improved spacing."""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        
        for i in range(self.num_points):
            theta = np.arccos(-1 + (2 * i) / (self.num_points - 1))
            phi_angle = (i * 2 * np.pi) / (phi * phi)
            
            x = np.sin(theta) * np.cos(phi_angle)
            y = np.sin(theta) * np.sin(phi_angle)
            
            x = 0.05 + 0.9 * (x + 1) / 2
            y = 0.05 + 0.9 * (y + 1) / 2
            
            points.append([x, y])
        
        return np.array(points)
    
    def _generate_regular_grid(self) -> np.ndarray:
        """Generate regular grid with boundary safety."""
        points = []
        side_length = int(np.ceil(np.sqrt(self.num_points)))
        
        for i in range(side_length):
            for j in range(side_length):
                if len(points) >= self.num_points:
                    break
                x = (i + 0.5) / side_length
                y = (j + 0.5) / side_length
                points.append([x, y])
        
        return np.array(points[:self.num_points])
    
    def _generate_polar_arrangement(self) -> np.ndarray:
        """Generate polar ring arrangement."""
        points = []
        radii = [0.15, 0.3, 0.45, 0.6]
        angles_per_ring = [4, 6, 8, 10]
        
        points.append([0.5, 0.5])
        
        for i, (radius, num_angles) in enumerate(zip(radii, angles_per_ring)):
            for j in range(num_angles):
                if len(points) >= self.num_points:
                    break
                angle = (j * 2 * np.pi) / num_angles
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                points.append([x, y])
            if len(points) >= self.num_points:
                break
        
        remaining = self.num_points - len(points)
        for _ in range(remaining):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            points.append([x, y])
        
        return np.array(points)
    
    def _generate_initial_configurations(self) -> List[np.ndarray]:
        """Generate diverse initial configurations systematically."""
        configs = []
        
        # Base configurations
        configs.append(self._generate_hexagonal_lattice())
        configs.append(self._generate_fibonacci_distribution())
        configs.append(self._generate_regular_grid())
        configs.append(self._generate_polar_arrangement())
        
        # Add variations with different perturbations
        for config in configs[:3]:  # Use first 3 base configs
            # Different perturbation magnitudes
            for mag in [0.01, 0.015, 0.02, 0.025]:
                perturbed = config + np.random.normal(0, mag, config.shape)
                perturbed = np.clip(perturbed, 0.001, 0.999)
                configs.append(perturbed)
        
        # Grid-based variations
        for _ in range(3):
            grid_points = []
            for i in range(4):
                for j in range(4):
                    x = (i + 0.5) / 4.0
                    y = (j + 0.5) / 4.0
                    grid_points.append([x, y])
            
            structured = np.array(grid_points[:self.num_points])
            structured += np.random.normal(0, 0.02, structured.shape)
            structured = np.clip(structured, 0.001, 0.999)
            configs.append(structured)
        
        return configs
    
    def _quick_optimize(self, x0: np.ndarray) -> Optional[np.ndarray]:
        """Light optimization for preliminary configuration assessment."""
        try:
            result = minimize(
                self._objective,
                x0,
                method='L-BFGS-B',
                bounds=self.bounds,
                options={'maxiter': 50, 'ftol': 1e-4, 'gtol': 1e-3}
            )
            
            if result.success:
                return result.x.reshape(-1, self.dimension)
        except Exception:
            return None
        return None
    
    def _thorough_optimize(self, x0: np.ndarray) -> Optional[np.ndarray]:
        """Full optimization for final refinement."""
        try:
            result = minimize(
                self._objective,
                x0,
                method='SLSQP',
                bounds=self.bounds,
                options={'maxiter': 150, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            
            if result.success:
                return result.x.reshape(-1, self.dimension)
        except Exception:
            return None
        return None
    
    def _refine_solution(self, points: np.ndarray, max_iterations: int = 3) -> np.ndarray:
        """Iteratively refine solution with adaptive perturbations."""
        current_points = points.copy()
        
        for iteration in range(max_iterations):
            # Generate adaptive perturbation
            distances = pdist(current_points)
            if len(distances) > 0:
                avg_dist = np.mean(distances)
                std_dist = np.std(distances)
                uniformity_ratio = std_dist / avg_dist if avg_dist > 0 else 1.0
                
                # Dynamic scaling based on distribution quality
                base_std = 0.025
                perturbation_std = base_std * (1.0 / (1.0 + iteration * 0.1))
                
                if uniformity_ratio < 0.15:  # Uniform distribution detected
                    perturbation_std *= 1.5
                
                # Apply perturbation
                perturbed = current_points + np.random.normal(0, perturbation_std, current_points.shape)
                perturbed = np.clip(perturbed, 0.001, 0.999)
                
                # Optimize perturbed version
                refined = self._thorough_optimize(perturbed.flatten())
                if refined is not None:
                    ratio, _, _ = self._compute_ratio(refined)
                    current_ratio, _, _ = self._compute_ratio(current_points)
                    
                    if ratio > current_ratio:
                        current_points = refined.copy()
        
        return current_points
    
    def _perform_multi_stage_optimization(self, initial_configs: List[np.ndarray]) -> np.ndarray:
        """Execute multi-stage optimization with progressive refinement."""
        # Stage 1: Quick assessments
        stage1_results = []
        
        for i, config in enumerate(initial_configs):
            quick_result = self._quick_optimize(config.flatten())
            if quick_result is not None:
                ratio, _, _ = self._compute_ratio(quick_result)
                stage1_results.append((ratio, i, quick_result))
        
        # Sort and select top performers
        if stage1_results:
            stage1_results.sort(reverse=True)
            top_configs = [result[2] for result in stage1_results[:5]]
        else:
            top_configs = initial_configs[:5] if len(initial_configs) >= 5 else initial_configs
        
        # Stage 2: Thorough optimization of top candidates
        for config in top_configs:
            full_result = self._thorough_optimize(config.flatten())
            if full_result is not None:
                ratio, _, _ = self._compute_ratio(full_result)
                if ratio > self.best_ratio:
                    self.best_ratio = ratio
                    self.best_solution = full_result.copy()
        
        # Stage 3: Refinement with adaptive perturbations
        if self.best_solution is not None:
            self.best_solution = self._refine_solution(self.best_solution)
            ratio, _, _ = self._compute_ratio(self.best_solution)
            if ratio > self.best_ratio:
                self.best_ratio = ratio
        
        return self.best_solution if self.best_solution is not None else initial_configs[0]
    
    def optimize(self) -> np.ndarray:
        """Main optimization entry point."""
        # Generate diverse initial configurations
        initial_configs = self._generate_initial_configurations()
        
        # Perform multi-stage optimization
        best_points = self._perform_multi_stage_optimization(initial_configs)
        
        # Final validation
        if best_points is not None:
            final_result = self._thorough_optimize(best_points.flatten())
            if final_result is not None:
                ratio, _, _ = self._compute_ratio(final_result)
                if ratio > 0.01:
                    best_points = final_result
        
        # Fallback mechanism
        if best_points is None or len(best_points) == 0:
            fallback_config = self._generate_regular_grid()
            fallback_config += np.random.normal(0, 0.01, fallback_config.shape)
            fallback_config = np.clip(fallback_config, 0.001, 0.999)
            best_points = fallback_config
        
        return best_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = PointEvolutionOptimizer(16, 2)
    return optimizer.optimize()

# EVOLVE-BLOCK-END