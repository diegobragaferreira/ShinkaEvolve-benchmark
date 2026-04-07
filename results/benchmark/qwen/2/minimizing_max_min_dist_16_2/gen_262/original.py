# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist
import time
import warnings

class PointOptimizer:
    def __init__(self, num_points=16, dimensions=2):
        self.num_points = num_points
        self.dimensions = dimensions
        self.best_ratio = -np.inf
        self.best_points = None
        self.benchmark_threshold = 1 / np.sqrt(12.889266112)  # AlphaEvolve benchmark
        
    def objective(self, x):
        """Objective function to maximize min/max distance ratio."""
        points = x.reshape(self.num_points, self.dimensions)
        distances = pdist(points)
        
        if len(distances) == 0:
            return -np.inf
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max <= 0:
            return -np.inf
            
        return -d_min / d_max
    
    def constraint(self, x):
        """Boundary constraints to keep points within [0,1]^2."""
        points = x.reshape(self.num_points, self.dimensions)
        return np.concatenate([
            points[:, 0],           # x coordinates >= 0
            1 - points[:, 0],       # x coordinates <= 1
            points[:, 1],           # y coordinates >= 0
            1 - points[:, 1]        # y coordinates <= 1
        ])
    
    def compute_ratio(self, points):
        """Compute the min/max distance ratio for given points."""
        if len(points) < 2:
            return 0.0
            
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max <= 0:
            return 0.0
            
        return d_min / d_max
    
    def adaptive_perturbation(self, initial_points, current_ratio):
        """Adaptively scale perturbations based on current distance distribution."""
        base_perturbation = 0.08
        adaptive_scale = max(0.1, 1.0 / (current_ratio + 0.01))
        perturbation_magnitude = base_perturbation * adaptive_scale
        
        noise = np.random.normal(0, perturbation_magnitude/3, (self.num_points, self.dimensions))
        perturbed_points = np.clip(initial_points + noise, 0, 1)
        return perturbed_points
    
    def create_initial_configurations(self):
        """Create multiple high-quality initial configurations."""
        configs = []
        
        # Configuration 1: Structured 4x4 grid with adaptive perturbations
        np.random.seed(42)
        grid_x = np.linspace(0.1, 0.9, 4)
        grid_y = np.linspace(0.1, 0.9, 4)
        grid_points = np.array([[x, y] for x in grid_x for y in grid_y])
        config1 = self.adaptive_perturbation(grid_points, 0.1)
        configs.append(config1)
        
        # Configuration 2: Random with clustering avoidance
        np.random.seed(123)
        config2 = np.random.uniform(0.05, 0.95, (self.num_points, self.dimensions))
        # Add some structure to avoid very tight clusters
        for i in range(0, self.num_points, 4):
            group_center = np.mean(config2[i:i+4], axis=0)
            config2[i:i+4] += np.random.normal(0, 0.03, (4, self.dimensions))
            config2[i:i+4] = np.clip(config2[i:i+4], 0, 1)
        configs.append(config2)
        
        # Configuration 3: Fibonacci spiral-like arrangement
        np.random.seed(456)
        angles = np.linspace(0, 2*np.pi, self.num_points, endpoint=False)
        radii = np.sqrt(np.linspace(0.05, 0.45, self.num_points))
        fib_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        fib_points = np.clip((fib_points + 1) / 2, 0, 1)
        configs.append(fib_points)
        
        # Configuration 4: Hexagonal grid approximation
        np.random.seed(789)
        hex_x = np.array([0.15, 0.45, 0.75, 0.3, 0.6, 0.15, 0.45, 0.75, 0.225, 0.525, 0.825, 0.375, 0.675, 0.225, 0.525, 0.825])
        hex_y = np.array([0.15, 0.15, 0.15, 0.45, 0.45, 0.75, 0.75, 0.75, 0.3, 0.3, 0.3, 0.6, 0.6, 0.9, 0.9, 0.9])
        hex_points = np.column_stack([hex_x, hex_y])
        config4 = self.adaptive_perturbation(hex_points, 0.1)
        configs.append(config4)
        
        return configs
    
    def optimize_stage(self, x0, method='L-BFGS-B'):
        """Perform single optimization stage with specified method."""
        bounds = [(0, 1) for _ in range(self.num_points * self.dimensions)]
        
        try:
            if method == 'L-BFGS-B':
                result = minimize(
                    self.objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-6, 'gtol': 1e-6}
                )
            else:  # SLSQP
                result = minimize(
                    self.objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints={'type': 'ineq', 'fun': self.constraint},
                    options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
                )
            
            if result.success:
                return result.x
        except Exception:
            pass
            
        return None
    
    def hierarchical_optimization(self, initial_points_list, max_time=170):
        """Run optimization with coarse-to-fine strategy to improve efficiency."""
        start_time = time.time()
        best_ratio = -np.inf
        best_points = None
        
        # Phase 1: Coarse optimization with fewer iterations
        phase1_configs = []
        for i, initial_points in enumerate(initial_points_list[:2]):
            if time.time() - start_time > max_time:
                break
                
            try:
                # Quick optimization with fewer iterations
                bounds = [(0, 1) for _ in range(self.num_points * self.dimensions)]
                result = minimize(
                    self.objective,
                    initial_points.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-5, 'gtol': 1e-5}
                )
                
                if result.success:
                    phase1_configs.append(result.x.reshape(self.num_points, self.dimensions))
                    
            except Exception:
                continue
        
        # Phase 2: Detailed optimization on promising configurations
        configs_to_optimize = phase1_configs if phase1_configs else initial_points_list[:2]
        
        for i, initial_points in enumerate(configs_to_optimize):
            if time.time() - start_time > max_time:
                break
                
            try:
                # Stage 1: Fast optimization with L-BFGS-B
                optimized_x = self.optimize_stage(initial_points.flatten(), 'L-BFGS-B')
                
                if optimized_x is not None:
                    optimized_points = optimized_x.reshape(self.num_points, self.dimensions)
                    
                    # Stage 2: Precise optimization with SLSQP
                    refined_x = self.optimize_stage(optimized_points.flatten(), 'SLSQP')
                    
                    if refined_x is not None:
                        final_points = refined_x.reshape(self.num_points, self.dimensions)
                        final_ratio = self.compute_ratio(final_points)
                        
                        if final_ratio > best_ratio:
                            best_ratio = final_ratio
                            best_points = final_points.copy()
                            
            except Exception:
                continue
        
        return best_points if best_points is not None else initial_points_list[0]
    
    def execute_optimization(self):
        """Main optimization execution method."""
        # Generate multiple initial configurations
        initial_configs = self.create_initial_configurations()
        
        # Integrate evolutionary algorithm restarts
        try:
            bounds = [(0, 1) for _ in range(self.num_points * self.dimensions)]
            de_result = differential_evolution(
                self.objective,
                bounds,
                maxiter=30,
                popsize=8,
                seed=42,
                tol=1e-5,
                mutation=(0.5, 1),
                recombination=0.7
            )

            if de_result.success:
                de_points = de_result.x.reshape(self.num_points, self.dimensions)
                # Add evolutionary result as another initial configuration
                initial_configs.append(de_points)
        except Exception:
            pass
        
        # Run hierarchical optimization for better efficiency
        best_points = self.hierarchical_optimization(initial_configs, max_time=170)
        
        # Final refinement of the best result
        if best_points is not None:
            try:
                final_x = self.optimize_stage(best_points.flatten(), 'SLSQP')
                if final_x is not None:
                    final_points = final_x.reshape(self.num_points, self.dimensions)
                    final_ratio = self.compute_ratio(final_points)
                    if final_ratio > self.best_ratio:
                        self.best_points = final_points
                        self.best_ratio = final_ratio
            except Exception:
                pass
        
        # If no successful optimization, return the best initial configuration
        if self.best_points is None:
            self.best_points = initial_configs[0] if initial_configs else np.random.uniform(0, 1, (self.num_points, self.dimensions))
        
        return self.best_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    # Suppress warnings for cleaner output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        optimizer = PointOptimizer(num_points=16, dimensions=2)
        return optimizer.execute_optimization()

# EVOLVE-BLOCK-END
