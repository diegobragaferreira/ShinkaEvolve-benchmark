# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist
import math
from typing import Tuple, List, Optional

class PointEvolutionOptimizer:
    """Novel evolutionary point dispersion optimizer that maximizes min/max distance ratio."""
    
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
    
    def generate_hierarchical_grid(self) -> np.ndarray:
        """Generate a novel hierarchical grid with adaptive spacing and strategic perturbations."""
        points = []
        
        # Create multiple layers of grids with different densities
        layer_configs = [
            # Dense inner grid (4x4)
            {"rows": 4, "cols": 4, "spacing_factor": 0.9, "perturbation": 0.015},
            # Medium outer grid (3x3)
            {"rows": 3, "cols": 3, "spacing_factor": 1.2, "perturbation": 0.02},
            # Sparse perimeter grid (2x2)
            {"rows": 2, "cols": 2, "spacing_factor": 1.5, "perturbation": 0.025}
        ]
        
        # Start with dense inner grid
        layer = layer_configs[0]
        spacing_x = 1.0 / (layer["cols"] - 1) if layer["cols"] > 1 else 1.0
        spacing_y = 1.0 / (layer["rows"] - 1) if layer["rows"] > 1 else 1.0
        
        spacing_x *= layer["spacing_factor"]
        spacing_y *= layer["spacing_factor"]
        
        for i in range(layer["rows"]):
            for j in range(layer["cols"]):
                if len(points) >= self.num_points:
                    break
                x_offset = spacing_x * 0.25 if i % 2 == 1 else 0.0
                x = (j * spacing_x) + x_offset
                y = i * spacing_y
                
                # Add controlled perturbations
                x += np.random.uniform(-layer["perturbation"], layer["perturbation"])
                y += np.random.uniform(-layer["perturbation"], layer["perturbation"])
                
                # Ensure bounds
                x = np.clip(x, 0.001, 0.999)
                y = np.clip(y, 0.001, 0.999)
                
                points.append([x, y])
        
        # Add remaining points in sparse pattern
        remaining = self.num_points - len(points)
        for _ in range(remaining):
            # Strategic placement to avoid clustering
            x = 0.2 + np.random.exponential(0.1) * 0.6  # Prefer middle region
            y = 0.2 + np.random.exponential(0.1) * 0.6
            x = np.clip(x, 0.001, 0.999)
            y = np.clip(y, 0.001, 0.999)
            points.append([x, y])
            
        return np.array(points[:self.num_points])
    
    def generate_fibonacci_enhanced(self) -> np.ndarray:
        """Enhanced Fibonacci spiral approach for better distribution."""
        points = []
        # Modified golden ratio for better spacing
        phi = (1 + math.sqrt(5)) / 2
        golden_ratio = 1.618
        
        # Use modified parameters for improved distribution
        for i in range(self.num_points):
            # Improve spiral parameterization
            theta = math.acos(-1 + (2 * i) / (self.num_points - 1))
            phi_angle = (i * 2 * math.pi) / (golden_ratio * golden_ratio)
            
            # Convert to cartesian coordinates
            x = math.sin(theta) * math.cos(phi_angle)
            y = math.sin(theta) * math.sin(phi_angle)
            
            # Map to [0.05, 0.95] range with better boundary handling
            x = 0.05 + 0.9 * (x + 1) / 2
            y = 0.05 + 0.9 * (y + 1) / 2
            
            # Add slight perturbations to break perfect symmetry
            x += np.random.uniform(-0.005, 0.005)
            y += np.random.uniform(-0.005, 0.005)
            
            x = np.clip(x, 0.001, 0.999)
            y = np.clip(y, 0.001, 0.999)
            
            points.append([x, y])
        
        return np.array(points)
    
    def generate_symmetric_pattern(self) -> np.ndarray:
        """Generate a symmetric pattern with controlled randomness to avoid degenerate solutions."""
        points = []
        
        # Place points in a symmetric pattern with strategic randomness
        # Start with a central ring structure
        center_radius = 0.3
        
        # Points on concentric rings
        rings = [0.15, 0.3, 0.45, 0.6]
        points_per_ring = [4, 6, 8, 10]
        
        # Add center point
        points.append([0.5, 0.5])
        
        # Add points on rings
        for i, (radius, num_points) in enumerate(zip(rings, points_per_ring)):
            for j in range(num_points):
                if len(points) >= self.num_points:
                    break
                    
                angle = (j * 2 * math.pi) / num_points
                # Add some randomness to angles to break perfect symmetry
                angle += np.random.uniform(-0.1, 0.1)
                
                x = 0.5 + radius * math.cos(angle)
                y = 0.5 + radius * math.sin(angle)
                
                x = np.clip(x, 0.001, 0.999)
                y = np.clip(y, 0.001, 0.999)
                
                points.append([x, y])
                
            if len(points) >= self.num_points:
                break
        
        # Fill remaining spots with controlled randomness
        remaining = self.num_points - len(points)
        for _ in range(remaining):
            # Prefer outer regions with some bias
            x = 0.1 + np.random.beta(2, 2) * 0.8
            y = 0.1 + np.random.beta(2, 2) * 0.8
            x = np.clip(x, 0.001, 0.999)
            y = np.clip(y, 0.001, 0.999)
            points.append([x, y])
        
        return np.array(points[:self.num_points])
    
    def generate_multi_scale_pattern(self) -> np.ndarray:
        """Generate points using a multi-scale approach combining different distribution strategies."""
        points = []
        
        # Combine elements from different strategies for maximum diversity
        
        # 1. Grid-based core (8 points)
        grid_points = []
        for i in range(3):
            for j in range(3):
                if len(grid_points) >= 8:
                    break
                x = (i + 0.5) / 3.0
                y = (j + 0.5) / 3.0
                grid_points.append([x, y])
        
        # Add some randomness to grid points
        for i in range(len(grid_points)):
            grid_points[i][0] += np.random.uniform(-0.03, 0.03)
            grid_points[i][1] += np.random.uniform(-0.03, 0.03)
            
        grid_points = [[np.clip(x, 0.001, 0.999), np.clip(y, 0.001, 0.999)] for x, y in grid_points]
        points.extend(grid_points[:8])
        
        # 2. Fibonacci-like spiral (6 points)
        fib_points = self.generate_fibonacci_enhanced()
        points.extend(fib_points[:6])
        
        # 3. Random distribution for remaining (2 points)
        remaining = self.num_points - len(points)
        for _ in range(remaining):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            points.append([np.clip(x, 0.001, 0.999), np.clip(y, 0.001, 0.999)])
        
        return np.array(points[:self.num_points])
    
    def generate_initial_configurations(self) -> List[np.ndarray]:
        """Generate diverse initial configurations using novel approaches."""
        configs = []
        
        # Generate different base configurations using our new methods
        configs.append(self.generate_hierarchical_grid())
        configs.append(self.generate_fibonacci_enhanced())
        configs.append(self.generate_symmetric_pattern())
        configs.append(self.generate_multi_scale_pattern())
        
        # Add perturbed versions
        np.random.seed(42)
        perturbed_configs = []
        for config in configs:
            # Multiple levels of perturbations
            for perturbation_magnitude in [0.01, 0.015, 0.02]:
                perturbed = config + np.random.normal(0, perturbation_magnitude, config.shape)
                perturbed = np.clip(perturbed, 0.001, 0.999)
                perturbed_configs.append(perturbed)
        
        return perturbed_configs
    
    def evolutionary_refinement(self, x0: np.ndarray, max_iterations: int = 50) -> np.ndarray:
        """Implement evolutionary refinement process that progressively improves the configuration."""
        current_points = x0.reshape(-1, self.dimension).copy()
        best_points = current_points.copy()
        best_ratio, _, _ = self.calculate_ratio(best_points)
        
        # Evolutionary phases
        for iteration in range(max_iterations):
            # Phase 1: Local optimization
            try:
                result = minimize(
                    self.objective_function,
                    current_points.flatten(),
                    method='L-BFGS-B',
                    bounds=self.bounds,
                    options={'maxiter': 20, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                if result.success:
                    local_points = result.x.reshape(-1, self.dimension)
                    local_ratio, _, _ = self.calculate_ratio(local_points)
                    
                    if local_ratio > best_ratio:
                        best_ratio = local_ratio
                        best_points = local_points.copy()
                        current_points = local_points.copy()
            except Exception:
                pass
            
            # Phase 2: Adaptive neighborhood search
            if iteration % 5 == 0 and iteration > 0:
                # Create neighborhood configurations and evaluate
                neighbor_configs = []
                
                # Create variations with different perturbation schemes
                for _ in range(3):
                    # Gaussian perturbations
                    perturbed = current_points + np.random.normal(0, 0.005, current_points.shape)
                    perturbed = np.clip(perturbed, 0.001, 0.999)
                    neighbor_configs.append(perturbed)
                
                for neighbor in neighbor_configs:
                    try:
                        ratio, _, _ = self.calculate_ratio(neighbor)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = neighbor.copy()
                            current_points = neighbor.copy()
                    except Exception:
                        continue
            
            # Phase 3: Periodic global restart (every 10 iterations)
            if iteration % 10 == 0 and iteration > 0:
                # Restart from best configuration found so far
                current_points = best_points.copy()
        
        return best_points
    
    def get_best_solution(self, configs: List[np.ndarray]) -> np.ndarray:
        """Find the best solution among all starting configurations using evolutionary approach."""
        best_ratio = -np.inf
        best_points = None
        
        # Phase 1: Evaluate all configurations with fast local optimization
        evaluated_configs = []
        for i, config in enumerate(configs):
            try:
                result = minimize(
                    self.objective_function,
                    config.flatten(),
                    method='L-BFGS-B',
                    bounds=self.bounds,
                    options={'maxiter': 30, 'ftol': 1e-6, 'gtol': 1e-4}
                )
                
                if result.success:
                    optimized_points = result.x.reshape(-1, self.dimension)
                    ratio, _, _ = self.calculate_ratio(optimized_points)
                    evaluated_configs.append((ratio, i, optimized_points))
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
            except Exception:
                continue
        
        # Phase 2: Evolutionary refinement of top configurations
        if best_points is not None:
            # Refine the best found so far using our evolutionary approach
            refined_points = self.evolutionary_refinement(best_points.flatten(), max_iterations=30)
            refined_ratio, _, _ = self.calculate_ratio(refined_points)
            
            if refined_ratio > best_ratio:
                best_ratio = refined_ratio
                best_points = refined_points.copy()
        
        # Phase 3: Additional local search with multiple restarts
        if best_points is not None:
            # Try additional local optimizations from different starting points
            for i in range(3):
                # Slightly perturb the best configuration
                perturbed = best_points + np.random.normal(0, 0.005, best_points.shape)
                perturbed = np.clip(perturbed, 0.001, 0.999)
                
                try:
                    result = minimize(
                        self.objective_function,
                        perturbed.flatten(),
                        method='SLSQP',
                        bounds=self.bounds,
                        options={'maxiter': 50}
                    )
                    
                    if result.success:
                        optimized_points = result.x.reshape(-1, self.dimension)
                        ratio, _, _ = self.calculate_ratio(optimized_points)
                        
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = optimized_points.copy()
                except Exception:
                    continue
        
        return best_points if best_points is not None else configs[0] if configs else np.random.rand(16, 2)

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Initialize optimizer
    optimizer = PointEvolutionOptimizer(16, 2)
    
    # Generate initial configurations with novel approaches
    initial_configs = optimizer.generate_initial_configurations()
    
    # Add evolutionary restart (different from previous versions)
    try:
        de_bounds = [(0.001, 0.999) for _ in range(32)]
        de_result = differential_evolution(
            optimizer.objective_function,
            de_bounds,
            maxiter=20,          # Fewer iterations for speed
            popsize=8,           # Smaller population for faster execution
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
    
    # Find best solution using evolutionary refinement approach
    best_points = optimizer.get_best_solution(initial_configs)
    
    # Final enhancement: one more round of evolutionary refinement
    final_points = optimizer.evolutionary_refinement(best_points.flatten(), max_iterations=20)
    
    return final_points

# EVOLVE-BLOCK-END