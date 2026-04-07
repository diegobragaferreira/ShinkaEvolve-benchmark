# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution
import math
from typing import Tuple, List, Optional
from numba import jit
import time

@jit(nopython=True)
def calculate_distances_fast(points):
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

class AdaptiveEvolutionaryOptimizer:
    """Enhanced optimizer with adaptive strategies for point dispersion problems."""
    
    def __init__(self, num_points: int = 16, dimension: int = 2):
        self.num_points = num_points
        self.dimension = dimension
        self.bounds = [(0.001, 0.999) for _ in range(num_points * dimension)]
        self.start_time = time.time()
        self.max_time = 170.0  # Leave 10 seconds for final processing

    def calculate_ratio(self, points: np.ndarray) -> Tuple[float, float, float]:
        """Calculate min/max distance ratio with proper error handling."""
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

    def generate_hexagonal_lattice(self) -> np.ndarray:
        """Generate high-quality hexagonal lattice with optimized spacing."""
        points = []
        rows = cols = 4
        
        spacing_x = 1.0 / (cols - 1) if cols > 1 else 1.0
        spacing_y = 1.0 / (rows - 1) if rows > 1 else 1.0
        
        # Enhanced spacing for better distribution
        spacing_x *= 0.88
        spacing_y *= 0.88
        
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

    def generate_fibonacci_distribution(self) -> np.ndarray:
        """Generate points using Fibonacci spiral with improved distribution."""
        points = []
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        
        for i in range(self.num_points):
            # Better spiral parameterization with enhanced distribution
            theta = math.acos(-1 + (2 * i) / (self.num_points - 1))
            phi_angle = (i * 2 * math.pi) / (phi * phi * 1.2)  # Slight modification
            
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

    def generate_triangular_lattice(self) -> np.ndarray:
        """Generate triangular lattice pattern."""
        points = []
        rows = 4
        cols = 4
        
        # Triangular spacing
        spacing_x = 1.0 / (cols - 1) if cols > 1 else 1.0
        spacing_y = 1.0 / (rows - 1) if rows > 1 else 1.0
        spacing_y *= math.sqrt(3) / 2  # Height of equilateral triangle
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= self.num_points:
                    break
                x_offset = spacing_x * 0.5 if i % 2 == 1 else 0.0
                x = (j * spacing_x) + x_offset
                y = i * spacing_y
                
                # Maintain bounds precisely
                x = np.clip(x, 0.001, 0.999)
                y = np.clip(y, 0.001, 0.999)
                
                points.append([x, y])
        
        return np.array(points[:self.num_points])

    def generate_centered_pattern(self) -> np.ndarray:
        """Generate pattern with strong center concentration."""
        points = []
        
        # Strong center concentration
        for _ in range(4):
            points.append([0.5, 0.5])
        
        # Surrounding points
        for i in range(12):
            angle = (i * 2 * math.pi) / 12
            radius = 0.3 + 0.1 * np.random.random()
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            points.append([x, y])
        
        return np.array(points[:self.num_points])

    def adaptive_perturbation(self, points: np.ndarray, iteration: int = 0, ratio_quality: float = 0.0) -> np.ndarray:
        """Apply adaptive perturbation with sophisticated control."""
        distances = pdist(points)
        if len(distances) > 0:
            avg_dist = np.mean(distances)
            std_dist = np.std(distances)
            
            # Dynamic perturbation scaling based on solution quality
            if ratio_quality < 0.1:
                # Poor solution: larger perturbations to escape local minima
                base_std = 0.03
                perturbation_std = base_std * (1.0 / (1.0 + iteration * 0.05))
            elif ratio_quality < 0.2:
                # Moderate solution: medium perturbations
                base_std = 0.02
                perturbation_std = base_std * (1.0 / (1.0 + iteration * 0.1))
            else:
                # Good solution: small perturbations for fine tuning
                base_std = 0.01
                perturbation_std = base_std * (1.0 / (1.0 + iteration * 0.2))
            
            # Adaptive factor based on distribution uniformity
            uniformity_ratio = std_dist / avg_dist if avg_dist > 0 else 1.0
            if uniformity_ratio < 0.15:  # Uniform distribution detected
                perturbation_std *= 1.5
            
            # Apply perturbation
            perturbed = points + np.random.normal(0, perturbation_std, points.shape)
            perturbed = np.clip(perturbed, 0.001, 0.999)
            return perturbed
        return points

    def generate_diverse_initial_configs(self) -> List[np.ndarray]:
        """Generate highly diverse set of initial configurations."""
        configs = []
        
        # Base configurations - increased from 4 to 8 for better diversity
        configs.append(self.generate_hexagonal_lattice())
        configs.append(self.generate_fibonacci_distribution())
        configs.append(self.generate_regular_grid())
        configs.append(self.generate_polar_arrangement())
        configs.append(self.generate_triangular_lattice())
        configs.append(self.generate_centered_pattern())
        
        # Enhanced variations with multiple perturbation levels
        np.random.seed(42)
        for base_config in configs[:6]:  # First 6 base configs for variation
            # Different perturbation magnitudes
            for mag in [0.01, 0.015, 0.02, 0.025]:
                perturbed = base_config + np.random.normal(0, mag, base_config.shape)
                perturbed = np.clip(perturbed, 0.001, 0.999)
                configs.append(perturbed)
        
        # Add structured variations with different distributions
        for _ in range(4):
            # Grid-based with different offsets and perturbation types
            grid_points = []
            for i in range(4):
                for j in range(4):
                    x = (i + 0.5) / 4.0
                    y = (j + 0.5) / 4.0
                    grid_points.append([x, y])
            
            # Apply some structured perturbation with beta distribution
            structured = np.array(grid_points[:self.num_points])
            structured += np.random.beta(2, 2, structured.shape) * 0.03 - 0.015
            structured = np.clip(structured, 0.001, 0.999)
            configs.append(structured)
        
        return configs

    def optimize_single_config(self, x0: np.ndarray, methods: List[str] = None, 
                              tolerance: float = 1e-8) -> Optional[np.ndarray]:
        """Perform single optimization with intelligent method selection."""
        if methods is None:
            methods = ['L-BFGS-B', 'SLSQP', 'TNC']

        for method in methods:
            # Check if we're running out of time
            if time.time() - self.start_time > self.max_time:
                return None
                
            try:
                if method == 'L-BFGS-B':
                    result = minimize(
                        self.objective_function,
                        x0,
                        method=method,
                        bounds=self.bounds,
                        options={'maxiter': 100, 'ftol': tolerance, 'gtol': 1e-5}
                    )
                else:
                    result = minimize(
                        self.objective_function,
                        x0,
                        method=method,
                        bounds=self.bounds,
                        options={'maxiter': 100, 'ftol': tolerance}
                    )
                
                if result.success:
                    return result.x.reshape(-1, self.dimension)
            except Exception:
                continue
        
        return None

    def evolutionary_restart(self, initial_configs: List[np.ndarray]) -> List[np.ndarray]:
        """Use enhanced differential evolution to find good starting points for local optimization."""
        # Check time left
        if time.time() - self.start_time > self.max_time - 10:
            return initial_configs
            
        try:
            # Enhanced DE with better parameters for exploration
            de_bounds = [(0.001, 0.999) for _ in range(self.num_points * self.dimension)]
            de_result = differential_evolution(
                self.objective_function,
                de_bounds,
                maxiter=50,          # More iterations for better exploration
                popsize=20,          # Larger population for better diversity
                seed=42,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7
            )

            if de_result.success:
                de_points = de_result.x.reshape(-1, self.dimension)
                # Add this to our initial configurations
                initial_configs.append(de_points)
        except Exception:
            pass

        return initial_configs

    def multi_stage_optimization(self, configs: List[np.ndarray]) -> np.ndarray:
        """Multi-stage optimization with progressive refinement."""
        best_ratio = -np.inf
        best_points = None
        
        # Stage 1: Coarse evaluation of all configurations
        stage1_results = []
        
        for i, config in enumerate(configs):
            # Check time limit
            if time.time() - self.start_time > self.max_time - 10:
                break
                
            # Light optimization for quick assessment
            light_result = self.optimize_single_config(config.flatten(), ['L-BFGS-B'], 1e-6)
            if light_result is not None:
                ratio, _, _ = self.calculate_ratio(light_result)
                stage1_results.append((ratio, i, light_result))
        
        # Sort by quality and keep top performers
        stage1_results.sort(reverse=True)
        top_configs = [result[2] for result in stage1_results[:6]]  # Top 6 for more thorough optimization
        
        # Stage 2: Thorough optimization of top candidates
        for i, config in enumerate(top_configs):
            # Check time limit
            if time.time() - self.start_time > self.max_time - 10:
                break
                
            # Full optimization for best candidates
            full_result = self.optimize_single_config(config.flatten(), ['SLSQP'], 1e-8)
            if full_result is not None:
                ratio, _, _ = self.calculate_ratio(full_result)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = full_result.copy()
        
        # Stage 3: Additional refinement with adaptive perturbations
        if best_points is not None:
            # Multiple refinement passes with decreasing perturbations
            for iteration in range(5):
                # Check time limit
                if time.time() - self.start_time > self.max_time - 10:
                    break
                    
                # Generate new configurations via adaptive perturbation
                perturbed = self.adaptive_perturbation(best_points, iteration, best_ratio)
                refined_result = self.optimize_single_config(perturbed.flatten(), ['SLSQP'], 1e-8)
                
                if refined_result is not None:
                    ratio, _, _ = self.calculate_ratio(refined_result)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = refined_result.copy()
                elif iteration < 3:  # Don't give up too early on good solutions
                    # Try alternate method if primary fails
                    refined_result_alt = self.optimize_single_config(perturbed.flatten(), ['TNC'], 1e-8)
                    if refined_result_alt is not None:
                        ratio, _, _ = self.calculate_ratio(refined_result_alt)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = refined_result_alt.copy()

        # Return best found or fallback
        return best_points if best_points is not None else configs[0] if configs else np.random.rand(self.num_points, self.dimension)

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Initialize optimizer with optimized parameters
    optimizer = AdaptiveEvolutionaryOptimizer(16, 2)
    
    # Generate diverse initial configurations
    initial_configs = optimizer.generate_diverse_initial_configs()
    
    # Integrate enhanced evolutionary restart strategy to improve starting points
    initial_configs = optimizer.evolutionary_restart(initial_configs)
    
    # Perform multi-stage optimization
    best_points = optimizer.multi_stage_optimization(initial_configs)
    
    # Final validation and refinement
    if best_points is not None:
        # Additional optimization to ensure quality
        final_result = optimizer.optimize_single_config(best_points.flatten(), ['SLSQP'], 1e-8)
        if final_result is not None:
            ratio, _, _ = optimizer.calculate_ratio(final_result)
            # If the improvement is significant, use the final result
            if ratio > 0.05:  # More reasonable threshold
                best_points = final_result
    
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