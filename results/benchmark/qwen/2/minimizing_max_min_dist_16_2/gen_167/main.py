# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist
import math
from numba import jit

@jit(nopython=True)
def fast_pdist_numba(points):
    """Fast pairwise distance calculation using numba."""
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

class PointDispersionOptimizer:
    """Optimized point dispersion optimizer for maximizing min/max distance ratio."""
    
    def __init__(self, num_points: int = 16, dimension: int = 2):
        self.num_points = num_points
        self.dimension = dimension
        self.bounds = [(0.001, 0.999) for _ in range(num_points * dimension)]
        
    def calculate_ratio(self, points: np.ndarray) -> tuple:
        """Calculate min/max distance ratio efficiently."""
        if len(points) < 2:
            return 0.0, 0.0, 0.0
        
        distances = fast_pdist_numba(points)
        # Extract upper triangle for unique distances
        triu_indices = np.triu_indices_from(distances, k=1)
        unique_distances = distances[triu_indices]
        
        if len(unique_distances) == 0:
            return 0.0, 0.0, 0.0
            
        min_dist = np.min(unique_distances)
        max_dist = np.max(unique_distances)
        
        if max_dist == 0:
            return 0.0, min_dist, max_dist
            
        ratio = min_dist / max_dist
        return ratio, min_dist, max_dist
    
    def objective_function(self, x: np.ndarray) -> float:
        """Objective function to minimize (negative ratio)."""
        points = x.reshape(-1, self.dimension)
        ratio, _, _ = self.calculate_ratio(points)
        return -ratio
    
    def generate_hexagonal_lattice(self) -> np.ndarray:
        """Generate high-quality hexagonal lattice with optimized spacing."""
        points = []
        rows, cols = 4, 4
        spacing_x = 1.0 / (cols - 1) if cols > 1 else 1.0
        spacing_y = 1.0 / (rows - 1) if rows > 1 else 1.0
        
        # Enhanced spacing for better distribution
        spacing_x *= 0.92
        spacing_y *= 0.92
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= self.num_points:
                    break
                x_offset = spacing_x * 0.25 if i % 2 == 1 else 0.0
                x = (j * spacing_x) + x_offset
                y = i * spacing_y
                
                # Maintain bounds precisely
                x = np.clip(x, 0.001, 0.999)
                y = np.clip(y, 0.001, 0.999)
                
                points.append([x, y])
        
        return np.array(points[:self.num_points])
    
    def generate_fibonacci_spiral(self) -> np.ndarray:
        """Generate points using Fibonacci spiral with improved distribution."""
        points = []
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        
        for i in range(self.num_points):
            # Better spiral parameterization
            theta = math.acos(-1 + (2 * i) / (self.num_points - 1))
            phi_angle = (i * 2 * math.pi) / (phi * phi)
            
            # Cartesian conversion with better mapping
            x = math.sin(theta) * math.cos(phi_angle)
            y = math.sin(theta) * math.sin(phi_angle)
            
            # Map to [0.05, 0.95] range with boundary safety
            x = 0.05 + 0.9 * (x + 1) / 2
            y = 0.05 + 0.9 * (y + 1) / 2
            
            points.append([x, y])
        
        return np.array(points)
    
    def generate_regular_grid(self) -> np.ndarray:
        """Generate regular grid with proper boundary handling."""
        points = []
        side_length = int(math.ceil(math.sqrt(self.num_points)))
        
        for i in range(side_length):
            for j in range(side_length):
                if len(points) >= self.num_points:
                    break
                x = (i + 0.5) / side_length
                y = (j + 0.5) / side_length
                points.append([x, y])
        
        return np.array(points[:self.num_points])
    
    def generate_polar_arrangement(self) -> np.ndarray:
        """Generate polar arrangement with concentric rings."""
        points = []
        # Concentric circles with increasing angular density
        radii = [0.15, 0.3, 0.45, 0.6]
        angles_per_ring = [4, 6, 8, 10]
        
        # Center point
        points.append([0.5, 0.5])
        
        # Ring points
        for i, (radius, num_angles) in enumerate(zip(radii, angles_per_ring)):
            for j in range(num_angles):
                if len(points) >= self.num_points:
                    break
                angle = (j * 2 * math.pi) / num_angles
                x = 0.5 + radius * math.cos(angle)
                y = 0.5 + radius * math.sin(angle)
                points.append([x, y])
            if len(points) >= self.num_points:
                break
        
        # Fill remaining spots with random distribution
        remaining = self.num_points - len(points)
        for _ in range(remaining):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            points.append([x, y])
        
        return np.array(points)
    
    def generate_structured_initials(self) -> list:
        """Generate diverse structured initial configurations."""
        configs = []
        
        # Base configurations
        configs.append(self.generate_hexagonal_lattice())
        configs.append(self.generate_fibonacci_spiral())
        configs.append(self.generate_regular_grid())
        configs.append(self.generate_polar_arrangement())
        
        # Enhanced variations with different perturbations
        np.random.seed(42)
        for base_config in configs:
            # Different perturbation magnitudes
            for mag in [0.01, 0.015, 0.02]:
                perturbed = base_config + np.random.normal(0, mag, base_config.shape)
                perturbed = np.clip(perturbed, 0.001, 0.999)
                configs.append(perturbed)
        
        return configs
    
    def adaptive_perturbation(self, points: np.ndarray, iteration: int = 0) -> np.ndarray:
        """Apply adaptive perturbation with sophisticated control."""
        distances = fast_pdist_numba(points)
        triu_indices = np.triu_indices_from(distances, k=1)
        unique_distances = distances[triu_indices]
        
        if len(unique_distances) > 0:
            avg_dist = np.mean(unique_distances)
            std_dist = np.std(unique_distances)
            
            # Dynamic perturbation scaling
            base_std = 0.025
            perturbation_std = base_std * (1.0 / (1.0 + iteration * 0.1))
            
            # Adaptive factor based on distribution uniformity
            uniformity_ratio = std_dist / avg_dist if avg_dist > 0 else 1.0
            if uniformity_ratio < 0.15:  # Uniform distribution detected
                perturbation_std *= 1.5
            
            # Apply perturbation
            perturbed = points + np.random.normal(0, perturbation_std, points.shape)
            perturbed = np.clip(perturbed, 0.001, 0.999)
            return perturbed
        return points
    
    def multi_stage_optimization(self, configs: list) -> np.ndarray:
        """Multi-stage optimization with progressive refinement."""
        best_ratio = -np.inf
        best_points = None
        
        # Stage 1: Coarse evaluation of all configurations using L-BFGS-B
        stage1_results = []
        for i, config in enumerate(configs):
            try:
                # Light optimization for quick assessment
                result = minimize(
                    self.objective_function,
                    config.flatten(),
                    method='L-BFGS-B',
                    bounds=self.bounds,
                    options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-4}
                )
                
                if result.success:
                    optimized_points = result.x.reshape(-1, self.dimension)
                    ratio, _, _ = self.calculate_ratio(optimized_points)
                    stage1_results.append((ratio, i, optimized_points))
            except Exception:
                continue
        
        # Sort by quality and keep top performers
        if stage1_results:
            stage1_results.sort(reverse=True)
            top_configs = [result[2] for result in stage1_results[:5]]  # Top 5
            
            # Stage 2: Thorough optimization of top candidates using SLSQP
            for i, config in enumerate(top_configs):
                try:
                    # Full optimization for best candidates
                    result = minimize(
                        self.objective_function,
                        config.flatten(),
                        method='SLSQP',
                        bounds=self.bounds,
                        options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
                    )
                    
                    if result.success:
                        optimized_points = result.x.reshape(-1, self.dimension)
                        ratio, _, _ = self.calculate_ratio(optimized_points)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = optimized_points.copy()
                            
                except Exception:
                    continue
        
        # Stage 3: Additional refinement with adaptive perturbations
        if best_points is not None:
            for iteration in range(3):
                try:
                    # Generate new configurations via adaptive perturbation
                    perturbed = self.adaptive_perturbation(best_points, iteration)
                    result = minimize(
                        self.objective_function,
                        perturbed.flatten(),
                        method='SLSQP',
                        bounds=self.bounds,
                        options={'maxiter': 150, 'ftol': 1e-8, 'gtol': 1e-6}
                    )
                    
                    if result.success:
                        optimized_points = result.x.reshape(-1, self.dimension)
                        ratio, _, _ = self.calculate_ratio(optimized_points)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = optimized_points.copy()
                            
                except Exception:
                    continue
        
        # Return best found or fallback
        return best_points if best_points is not None else configs[0] if configs else np.random.rand(16, 2)

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Initialize optimizer with optimized parameters
    optimizer = PointDispersionOptimizer(16, 2)
    
    # Generate diverse initial configurations
    initial_configs = optimizer.generate_structured_initials()
    
    # Perform multi-stage optimization
    best_points = optimizer.multi_stage_optimization(initial_configs)
    
    # Final validation and refinement
    if best_points is not None:
        # Additional optimization to ensure quality
        try:
            result = minimize(
                optimizer.objective_function,
                best_points.flatten(),
                method='SLSQP',
                bounds=optimizer.bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                ratio, _, _ = optimizer.calculate_ratio(final_points)
                # If the improvement is significant, use the final result
                if ratio > 0.01:  # Only accept meaningful improvements
                    best_points = final_points
        except Exception:
            pass
    
    # Ensure we always return a valid configuration
    if best_points is None:
        # Fallback to a well-known good configuration
        fallback_config = optimizer.generate_regular_grid()
        # Add small perturbation to break symmetry
        fallback_config += np.random.normal(0, 0.01, fallback_config.shape)
        fallback_config = np.clip(fallback_config, 0.001, 0.999)
        best_points = fallback_config
    
    return best_points

# EVOLVE-BLOCK-END