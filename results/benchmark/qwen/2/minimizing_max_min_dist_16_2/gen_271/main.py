# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import math
from typing import Tuple, List, Optional

class EfficientPointDispersionOptimizer:
    """Efficient optimizer for point distribution to maximize min/max distance ratio."""
    
    def __init__(self, num_points: int = 16, dimension: int = 2):
        self.num_points = num_points
        self.dimension = dimension
        self.bounds = [(0.001, 0.999) for _ in range(num_points * dimension)]
        
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
    
    def generate_smart_grid(self) -> np.ndarray:
        """Generate optimized 4x4 grid with improved spacing."""
        points = []
        rows, cols = 4, 4
        
        # Use golden ratio inspired spacing for better distribution
        phi = (1 + math.sqrt(5)) / 2
        spacing_x = 1.0 / (cols - 1) if cols > 1 else 1.0
        spacing_y = 1.0 / (rows - 1) if rows > 1 else 1.0
        
        # Adjust spacing for better geometric properties
        spacing_x *= 0.92
        spacing_y *= 0.92
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= self.num_points:
                    break
                # Hexagonal offset
                x_offset = spacing_x * 0.25 if i % 2 == 1 else 0.0
                x = (j * spacing_x) + x_offset
                y = i * spacing_y
                
                # Clamp to bounds  
                x = np.clip(x, 0.001, 0.999)
                y = np.clip(y, 0.001, 0.999)
                
                points.append([x, y])
        
        return np.array(points[:self.num_points])
    
    def generate_fibonacci_distribution(self) -> np.ndarray:
        """Generate points using Fibonacci spiral with excellent distribution properties."""
        points = []
        # Golden ratio for optimal distribution
        phi = (1 + math.sqrt(5)) / 2
        
        for i in range(self.num_points):
            # Spiral with improved parameterization for 2D
            theta = math.acos(-1 + (2 * i) / (self.num_points - 1))
            phi_angle = (i * 2 * math.pi) / (phi * phi)
            
            # Convert to cartesian
            x = math.sin(theta) * math.cos(phi_angle)
            y = math.sin(theta) * math.sin(phi_angle)
            
            # Map to [0.05, 0.95] for boundary safety
            x = 0.05 + 0.9 * (x + 1) / 2
            y = 0.05 + 0.9 * (y + 1) / 2
            
            points.append([x, y])
        
        return np.array(points)
    
    def generate_regular_grid(self) -> np.ndarray:
        """Generate regular grid with boundary handling."""
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
        
        # Fill remaining spots
        remaining = self.num_points - len(points)
        for _ in range(remaining):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            points.append([x, y])
        
        return np.array(points)
    
    def adaptive_perturbation(self, points: np.ndarray, iteration: int = 0) -> np.ndarray:
        """Apply adaptive perturbation based on distribution characteristics."""
        distances = pdist(points)
        if len(distances) > 0:
            avg_dist = np.mean(distances)
            std_dist = np.std(distances)
            
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
    
    def generate_diverse_initial_configs(self) -> List[np.ndarray]:
        """Generate highly diverse initial configurations."""
        configs = []
        
        # Base configurations
        configs.append(self.generate_smart_grid())
        configs.append(self.generate_fibonacci_distribution())
        configs.append(self.generate_regular_grid())
        configs.append(self.generate_polar_arrangement())
        
        # Enhanced variations with multiple perturbation levels
        np.random.seed(42)
        for base_config in configs[:3]:  # Use first 3 base configs for variation
            # Different perturbation magnitudes
            for mag in [0.01, 0.015, 0.02]:
                perturbed = base_config + np.random.normal(0, mag, base_config.shape)
                perturbed = np.clip(perturbed, 0.001, 0.999)
                configs.append(perturbed)
        
        # Add structured variations
        for _ in range(2):
            # Grid-based with different offsets
            grid_points = []
            for i in range(4):
                for j in range(4):
                    x = (i + 0.5) / 4.0
                    y = (j + 0.5) / 4.0
                    grid_points.append([x, y])
            
            # Apply structured perturbation
            structured = np.array(grid_points[:self.num_points])
            structured += np.random.normal(0, 0.02, structured.shape)
            structured = np.clip(structured, 0.001, 0.999)
            configs.append(structured)
        
        return configs
    
    def advanced_multi_start_optimization(self, configs: List[np.ndarray]) -> np.ndarray:
        """Advanced multi-start optimization with smart selection and refinement."""
        best_ratio = -np.inf
        best_points = None
        
        # Phase 1: Fast screening with differential evolution for global search
        de_results = []
        for i, config in enumerate(configs):
            try:
                # Use differential evolution for global search with reduced iterations
                de_result = differential_evolution(
                    self.objective_function,
                    self.bounds,
                    seed=42 + i,
                    maxiter=100,  # Reduced iterations for speed
                    popsize=15,    # Smaller population for faster execution
                    mutation=(0.5, 1),
                    recombination=0.7,
                    tol=1e-6,
                    disp=False
                )
                
                if de_result.success:
                    refined_points = de_result.x.reshape(-1, self.dimension)
                    ratio, _, _ = self.calculate_ratio(refined_points)
                    de_results.append((ratio, i, refined_points))
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = refined_points.copy()
                        
            except Exception:
                continue
        
        # Phase 2: Local refinement of top configurations
        if best_points is not None:
            # Refine the best configurations using multiple methods
            top_configs = []
            if len(de_results) > 0:
                # Select top 3 based on DE results
                sorted_results = sorted(de_results, reverse=True)
                top_configs = [result[2] for result in sorted_results[:3]]
            
            # Try local optimizations on top performers
            for config in top_configs:
                # Try L-BFGS-B first for fast convergence
                try:
                    result = minimize(
                        self.objective_function,
                        config.flatten(),
                        method='L-BFGS-B',
                        bounds=self.bounds,
                        options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5}
                    )
                    
                    if result.success:
                        refined_points = result.x.reshape(-1, self.dimension)
                        ratio, _, _ = self.calculate_ratio(refined_points)
                        
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = refined_points.copy()
                            
                except Exception:
                    continue
                
                # Then try SLSQP for more thorough refinement
                try:
                    result = minimize(
                        self.objective_function,
                        config.flatten(),
                        method='SLSQP',
                        bounds=self.bounds,
                        options={'maxiter': 100}
                    )
                    
                    if result.success:
                        refined_points = result.x.reshape(-1, self.dimension)
                        ratio, _, _ = self.calculate_ratio(refined_points)
                        
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = refined_points.copy()
                            
                except Exception:
                    continue
        
        # Phase 3: Additional adaptive refinement
        if best_points is not None:
            # Apply progressive refinement with adaptive perturbations
            current_points = best_points.copy()
            for iteration in range(3):
                # Generate new configurations via adaptive perturbation
                perturbed = self.adaptive_perturbation(current_points, iteration)
                
                # Try optimization from perturbed configuration
                try:
                    result = minimize(
                        self.objective_function,
                        perturbed.flatten(),
                        method='L-BFGS-B',
                        bounds=self.bounds,
                        options={'maxiter': 80, 'ftol': 1e-8, 'gtol': 1e-5}
                    )
                    
                    if result.success:
                        refined_points = result.x.reshape(-1, self.dimension)
                        ratio, _, _ = self.calculate_ratio(refined_points)
                        
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = refined_points.copy()
                            
                except Exception:
                    continue
                    
                # Update current for next iteration
                current_points = best_points.copy()
        
        # Return best found or fallback
        return best_points if best_points is not None else configs[0]

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Initialize optimizer
    optimizer = EfficientPointDispersionOptimizer(16, 2)
    
    # Generate diverse initial configurations
    initial_configs = optimizer.generate_diverse_initial_configs()
    
    # Perform advanced multi-start optimization
    best_points = optimizer.advanced_multi_start_optimization(initial_configs)
    
    # Final validation and refinement
    if best_points is not None:
        # Try final local optimization
        try:
            result = minimize(
                optimizer.objective_function,
                best_points.flatten(),
                method='L-BFGS-B',
                bounds=optimizer.bounds,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                ratio, _, _ = optimizer.calculate_ratio(final_points)
                if ratio > 0.01:  # Only accept meaningful improvements
                    best_points = final_points
                    
        except Exception:
            pass
    
    # Ensure we always return a valid configuration
    if best_points is None:
        # Fallback to smart grid configuration
        fallback_config = optimizer.generate_smart_grid()
        # Add small perturbation to break symmetry
        fallback_config += np.random.normal(0, 0.01, fallback_config.shape)
        fallback_config = np.clip(fallback_config, 0.001, 0.999)
        best_points = fallback_config
    
    return best_points

# EVOLVE-BLOCK-END