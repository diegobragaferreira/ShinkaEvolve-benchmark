# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution
import math
from typing import Tuple, List, Optional
from numba import jit

@jit(nopython=True)
def calculate_distances_numba(points):
    """Fast distance calculation using numba for performance."""
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

class EvolvePointDispersionOptimizer:
    """Optimizes point distribution to maximize min/max distance ratio."""
    
    def __init__(self, num_points: int = 16, dimension: int = 2):
        self.num_points = num_points
        self.dimension = dimension
        self.bounds = [(0.001, 0.999) for _ in range(num_points * dimension)]
        self.max_evaluations = 10000
        
    def calculate_ratio(self, points: np.ndarray) -> Tuple[float, float, float]:
        """Calculate min/max distance ratio along with actual values."""
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
    
    def objective_function(self, x: np.ndarray) -> float:
        """Objective function to minimize (negative ratio)."""
        points = x.reshape(-1, self.dimension)
        ratio, _, _ = self.calculate_ratio(points)
        return -ratio
    
    def generate_hexagonal_grid(self) -> np.ndarray:
        """Generate hexagonal lattice initial configuration with optimized spacing."""
        points = []
        rows = 4
        cols = 4
        
        # Use optimized spacing for better dispersion
        spacing_x = 1.0 / (cols - 1) if cols > 1 else 1.0
        spacing_y = 1.0 / (rows - 1) if rows > 1 else 1.0
        
        # Enhanced spacing factor
        spacing_x *= 0.88
        spacing_y *= 0.88
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= self.num_points:
                    break
                x_offset = spacing_x * 0.25 if i % 2 == 1 else 0.0
                x = (j * spacing_x) + x_offset
                y = i * spacing_y
                
                # Ensure bounds and add slight perturbation
                x = max(0.001, min(0.999, x))
                y = max(0.001, min(0.999, y))
                
                points.append([x, y])
        
        return np.array(points[:self.num_points])
    
    def generate_fibonacci_spiral(self) -> np.ndarray:
        """Generate points using Fibonacci spiral with better distribution."""
        points = []
        # Golden ratio for improved point distribution
        phi = (1 + math.sqrt(5)) / 2
        
        for i in range(self.num_points):
            # Improved spiral generation with better parameterization
            theta = math.acos(-1 + (2 * i) / (self.num_points - 1))
            # Use a modified phi to get better spacing
            phi_angle = (i * 2 * math.pi) / (phi * 1.1)
            
            # Convert to cartesian coordinates
            x = math.sin(theta) * math.cos(phi_angle)
            y = math.sin(theta) * math.sin(phi_angle)
            
            # Map to [0.05, 0.95] range to avoid boundaries
            x = 0.05 + 0.9 * (x + 1) / 2
            y = 0.05 + 0.9 * (y + 1) / 2
            
            points.append([x, y])
        
        return np.array(points)
    
    def generate_regular_grid(self) -> np.ndarray:
        """Generate regular grid initial configuration."""
        points = []
        side_length = int(math.ceil(math.sqrt(self.num_points)))
        
        for i in range(side_length):
            for j in range(side_length):
                if len(points) >= self.num_points:
                    break
                x = (i + 0.5) / side_length
                y = (j + 0.5) / side_length
                points.append([x, y])
        
        return np.array(points)[:self.num_points]
    
    def generate_radial_pattern(self) -> np.ndarray:
        """Generate points in radial arrangement with adaptive angular distribution."""
        points = []
        # Concentric circles with varying angular density
        radii = [0.12, 0.25, 0.4, 0.55]
        angles_per_ring = [6, 8, 10, 12]
        
        # Add center point
        points.append([0.5, 0.5])
        
        # Add points in rings
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
        
        # Fill remaining spots with careful random placement
        while len(points) < self.num_points:
            # Place closer to edges to avoid clustering
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            points.append([x, y])
        
        return np.array(points[:self.num_points])
    
    def generate_modified_hex_grid(self) -> np.ndarray:
        """Generate a modified hexagonal grid with strategic perturbations."""
        points = []
        rows = 4
        cols = 4
        
        spacing_x = 1.0 / (cols - 1) if cols > 1 else 1.0
        spacing_y = 1.0 / (rows - 1) if rows > 1 else 1.0
        
        # Slightly adjusted spacing for better distribution
        spacing_x *= 0.9
        spacing_y *= 0.9
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= self.num_points:
                    break
                x_offset = spacing_x * 0.3 if i % 2 == 1 else 0.0
                x = (j * spacing_x) + x_offset
                y = i * spacing_y
                
                # Add strategic perturbation to avoid regular patterns
                if (i + j) % 3 == 0:
                    x += np.random.uniform(-0.01, 0.01)
                    y += np.random.uniform(-0.01, 0.01)
                
                # Ensure bounds
                x = max(0.001, min(0.999, x))
                y = max(0.001, min(0.999, y))
                
                points.append([x, y])
        
        return np.array(points[:self.num_points])
    
    def generate_initial_configurations(self) -> List[np.ndarray]:
        """Generate multiple diverse initial configurations."""
        configs = []
        
        # Generate different base configurations
        configs.append(self.generate_hexagonal_grid())
        configs.append(self.generate_fibonacci_spiral())
        configs.append(self.generate_regular_grid())
        configs.append(self.generate_radial_pattern())
        configs.append(self.generate_modified_hex_grid())
        
        # Add perturbed versions with different magnitudes
        np.random.seed(42)
        perturbed_configs = []
        for config in configs:
            # Four levels of perturbations for more diversity
            for perturbation_magnitude in [0.008, 0.012, 0.016, 0.02]:
                perturbed = config + np.random.normal(0, perturbation_magnitude, config.shape)
                perturbed = np.clip(perturbed, 0.001, 0.999)
                perturbed_configs.append(perturbed)
        
        return perturbed_configs
    
    def multi_stage_optimization(self, x0: np.ndarray) -> Optional[np.ndarray]:
        """Multi-stage optimization with adaptive approaches."""
        try:
            # Stage 1: Quick coarse optimization with L-BFGS-B
            result1 = minimize(
                self.objective_function,
                x0,
                method='L-BFGS-B',
                bounds=self.bounds,
                options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-4}
            )
            
            if result1.success:
                # Stage 2: Refinement with SLSQP
                refined_x = result1.x
                result2 = minimize(
                    self.objective_function,
                    refined_x,
                    method='SLSQP',
                    bounds=self.bounds,
                    options={'maxiter': 200}
                )
                
                if result2.success:
                    return result2.x.reshape(-1, self.dimension)
                else:
                    # Fallback to the previous result
                    return result1.x.reshape(-1, self.dimension)
            else:
                # Fallback to direct optimization with different method
                result3 = minimize(
                    self.objective_function,
                    x0,
                    method='TNC',
                    bounds=self.bounds,
                    options={'maxiter': 250}
                )
                
                if result3.success:
                    return result3.x.reshape(-1, self.dimension)
                    
        except Exception as e:
            # If all optimization fails, return original points
            return x0.reshape(-1, self.dimension)
        
        # Last resort - return original points
        return x0.reshape(-1, self.dimension)
    
    def adaptive_perturbation(self, points: np.ndarray, iteration: int = 0) -> np.ndarray:
        """Apply adaptive perturbation based on current configuration."""
        distances = pdist(points)
        if len(distances) > 0:
            avg_dist = np.mean(distances)
            std_dist = np.std(distances)
            
            # Perturbation magnitude decreases with iterations
            perturbation_std = 0.03 * (1.0 / (1.0 + iteration * 0.25))
            
            # Increase perturbation if distribution is too uniform
            if std_dist / avg_dist < 0.15:
                perturbation_std *= 2.5
            
            # Add more variance in early iterations
            if iteration < 2:
                perturbation_std *= 1.3
            
            perturbed = points + np.random.normal(0, perturbation_std, points.shape)
            perturbed = np.clip(perturbed, 0.001, 0.999)
            return perturbed
        return points
    
    def get_best_solution(self, configs: List[np.ndarray]) -> np.ndarray:
        """Find best solution among all starting configurations with hierarchical multi-start optimization."""
        best_ratio = -np.inf
        best_points = None
        
        # Phase 1: Fast coarse screening
        phase1_results = []
        for i, config in enumerate(configs[:len(configs)//2]):  # Use half configs for fast screening
            try:
                # Fast optimization for screening
                result = minimize(
                    self.objective_function,
                    config.flatten(),
                    method='L-BFGS-B',
                    bounds=self.bounds,
                    options={'maxiter': 50, 'ftol': 1e-4, 'gtol': 1e-3}
                )
                
                if result.success:
                    optimized_points = result.x.reshape(-1, self.dimension)
                    ratio, _, _ = self.calculate_ratio(optimized_points)
                    
                    phase1_results.append((ratio, i, optimized_points))
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
            except Exception:
                continue
        
        # Phase 2: Thorough refinement of top performers
        if best_points is not None and len(phase1_results) > 0:
            # Sort by ratio and take top 3
            sorted_results = sorted(phase1_results, key=lambda x: x[0], reverse=True)
            top_configs = [result[2] for result in sorted_results[:3]]
            
            # Refine the top configurations
            for config in top_configs:
                refined_points = self.multi_stage_optimization(config.flatten())
                if refined_points is not None:
                    ratio, _, _ = self.calculate_ratio(refined_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = refined_points.copy()
        
        # Phase 3: Advanced local search
        if best_points is not None:
            # Multiple local search strategies
            local_search_configs = []
            
            # Strategy 1: Adaptive perturbations
            for iter_num in range(2):
                adapted = self.adaptive_perturbation(best_points, iter_num)
                local_search_configs.append(adapted)
            
            # Strategy 2: Gaussian perturbations  
            for _ in range(3):
                gaussian = best_points + np.random.normal(0, 0.01, best_points.shape)
                gaussian = np.clip(gaussian, 0.001, 0.999)
                local_search_configs.append(gaussian)
            
            # Strategy 3: Uniform perturbations
            for _ in range(3):
                uniform = best_points + np.random.uniform(-0.015, 0.015, best_points.shape)
                uniform = np.clip(uniform, 0.001, 0.999)
                local_search_configs.append(uniform)
            
            # Optimize all local search configurations
            for config in local_search_configs:
                local_points = self.multi_stage_optimization(config.flatten())
                if local_points is not None:
                    ratio, _, _ = self.calculate_ratio(local_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = local_points.copy()
        
        return best_points if best_points is not None else configs[0] if configs else np.random.rand(16, 2)

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Initialize optimizer
    optimizer = EvolvePointDispersionOptimizer(16, 2)
    
    # Generate initial configurations
    initial_configs = optimizer.generate_initial_configurations()
    
    # Add evolutionary restart for better exploration
    try:
        de_bounds = [(0.001, 0.999) for _ in range(32)]
        de_result = differential_evolution(
            optimizer.objective_function,
            de_bounds,
            maxiter=25,          # Fewer iterations for speed
            popsize=10,          # Smaller population for faster execution
            seed=42,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7
        )
        
        if de_result.success:
            de_points = de_result.x.reshape(-1, 2)
            initial_configs.append(de_points)
    except Exception:
        pass
    
    # Find best solution with comprehensive optimization
    best_points = optimizer.get_best_solution(initial_configs)
    
    # Final refinement step with enhanced optimization
    final_points = optimizer.multi_stage_optimization(best_points.flatten())
    
    return final_points

# EVOLVE-BLOCK-END