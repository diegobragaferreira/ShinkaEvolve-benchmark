# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time


class PointArrangementOptimizer:
    """Optimizes point arrangement to maximize min/max distance ratio."""
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        
    def calculate_min_max_ratio(self, points):
        """Calculate the ratio of minimum to maximum distances between all point pairs."""
        if len(points) < 2:
            return 0

        distances = pdist(points)
        dmin = np.min(distances)
        dmax = np.max(distances)

        if dmax == 0:
            return 0

        return dmin / dmax

    def objective_function(self, points):
        """Objective function to maximize (negative because we minimize in scipy)."""
        return -self.calculate_min_max_ratio(points)

    def create_hexagonal_initialization(self):
        """Create a hexagonal-like arrangement of points."""
        points = np.zeros((16, 2))
        
        # Create a roughly hexagonal arrangement with 4 rows and 4 columns
        rows, cols = 4, 4
        spacing_x = 1.0 / (cols + 1)
        spacing_y = spacing_x * np.sqrt(3) / 2.0

        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx < 16:
                    # Offset every other row
                    x = (j + 0.5 * (i % 2)) * spacing_x
                    y = i * spacing_y
                    points[idx] = [x, y]
                    idx += 1
        
        return points

    def create_ring_initialization(self):
        """Create a concentric ring-like arrangement."""
        points = np.zeros((16, 2))

        # Place points in concentric rings with 4 layers of 4 points each
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radii = np.linspace(0.1, 0.8, 4)  # Four layers
        layer_points = [4, 4, 4, 4]  # 4 points per layer

        idx = 0
        for i, radius in enumerate(radii):
            num_points_in_layer = layer_points[i]
            layer_angles = np.linspace(0, 2*np.pi, num_points_in_layer, endpoint=False)
            for angle in layer_angles:
                if idx < 16:
                    x = 0.5 + radius * np.cos(angle)
                    y = 0.5 + radius * np.sin(angle)
                    points[idx] = [x, y]
                    idx += 1

        return points

    def create_fibonacci_initialization(self):
        """Create a Fibonacci-like arrangement for better point distribution."""
        points = np.zeros((16, 2))

        # Use Fibonacci-inspired pattern in 2D
        golden_ratio = (1 + np.sqrt(5)) / 2.0
        for i in range(16):
            theta = 2 * np.pi * i / golden_ratio
            r = np.sqrt(i / 15.0)  # Normalize to [0,1]
            x = 0.5 + r * np.cos(theta) * 0.8
            y = 0.5 + r * np.sin(theta) * 0.8
            points[i] = [x, y]

        return points

    def create_grid_initialization(self):
        """Create a regular grid initialization."""
        points = np.zeros((16, 2))
        idx = 0
        
        # Create 4x4 grid
        for i in range(4):
            for j in range(4):
                if idx < 16:
                    x = j / 3.0 if j > 0 else 0.0
                    y = i / 3.0 if i > 0 else 0.0
                    points[idx] = [x, y]
                    idx += 1
        
        return points

    def create_random_initialization(self):
        """Create a random initialization."""
        np.random.seed(self.seed)
        return np.random.rand(16, 2)

    def perturb_points(self, points, perturbation_magnitude=0.015):
        """Apply random perturbation to points and ensure bounds."""
        perturbed = points.copy()
        # Add random perturbation
        perturbed += np.random.normal(0, perturbation_magnitude, points.shape)
        # Ensure points stay within bounds
        perturbed = np.clip(perturbed, 0, 1)
        return perturbed

    def adaptive_perturbation(self, initial_ratio, iteration=0, max_iterations=100):
        """Calculate adaptive perturbation magnitude based on solution quality."""
        # Base perturbation scale
        base_perturbation = 0.015
        
        # Scale based on current solution quality
        if initial_ratio < 0.1:
            # Poor initial solution - use larger perturbations
            scale = 1.5
        elif initial_ratio < 0.2:
            # Moderate solution - medium perturbations
            scale = 1.0
        elif initial_ratio < 0.3:
            # Good solution - smaller perturbations
            scale = 0.6
        else:
            # Excellent solution - smallest perturbations
            scale = 0.3
            
        # Further reduce perturbation over iterations
        decay_factor = 1.0 - (iteration / max_iterations)
        scale *= decay_factor
        
        return base_perturbation * max(0.1, scale)

    def local_refinement(self, initial_points, max_iter=500):
        """Perform local optimization refinement on initial configuration."""
        # Flatten for optimization
        initial_flat = initial_points.flatten()
        
        # Define bounds for each coordinate (0 to 1)
        bounds = [(0, 1) for _ in range(len(initial_flat))]
        
        # Optimize using L-BFGS-B method with strict tolerances
        try:
            result = minimize(
                lambda flat_points: self.objective_function(flat_points.reshape(-1, 2)),
                initial_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-10},
                callback=None
            )
            
            # Extract optimized points
            optimized_points = result.x.reshape(-1, 2)
            
            # Ensure all points are within bounds
            optimized_points = np.clip(optimized_points, 0, 1)
            
            return optimized_points
        except Exception:
            # Return original points if optimization fails
            return initial_points

    def global_optimization(self, initial_points, max_iter=300):
        """Perform global optimization using differential evolution."""
        # Flatten for optimization
        flat_points = initial_points.flatten()
        bounds = [(0, 1) for _ in range(len(flat_points))]
        
        try:
            de_result = differential_evolution(
                lambda x: self.objective_function(x.reshape(-1, 2)),
                bounds,
                seed=self.seed,
                maxiter=max_iter,
                popsize=25,
                mutation=(0.5, 1),
                recombination=0.7,
                tol=1e-8,
                disp=False
            )
            
            return de_result.x.reshape(-1, 2)
        except Exception:
            # Return initial points if optimization fails
            return initial_points

    def get_initial_configurations(self):
        """Generate diverse initial configurations."""
        configs = []
        
        # Different initialization strategies
        configs.append(("hexagonal", self.create_hexagonal_initialization()))
        configs.append(("ring", self.create_ring_initialization()))
        configs.append(("fibonacci", self.create_fibonacci_initialization()))
        configs.append(("grid", self.create_grid_initialization()))
        configs.append(("random", self.create_random_initialization()))
        
        # Add perturbed versions to increase diversity
        perturbed_configs = []
        for name, config in configs:
            # Apply perturbation to each configuration
            perturbed = self.perturb_points(config, 0.02)
            perturbed_configs.append((f"{name}_perturbed", perturbed))
            
        configs.extend(perturbed_configs)
        
        return configs

    def optimize_single_configuration(self, name, initial_points, iteration=0):
        """Optimize a single initial configuration."""
        try:
            # Global optimization step
            global_result = self.global_optimization(initial_points, max_iter=200)
            
            # Local refinement
            refined_result = self.local_refinement(global_result, max_iter=300)
            
            # Evaluate final result
            ratio = self.calculate_min_max_ratio(refined_result)
            
            return ratio, refined_result
        except Exception:
            return -np.inf, initial_points

    def optimize_all_configurations(self):
        """Perform multi-start optimization across all configurations."""
        best_ratio = -np.inf
        best_points = None
        
        # Get all initial configurations
        configurations = self.get_initial_configurations()
        
        # Optimize each configuration
        for name, initial_config in configurations:
            try:
                ratio, result = self.optimize_single_configuration(name, initial_config)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = result
                    
            except Exception:
                continue
        
        # Fallback to the best configuration if none succeeded
        if best_points is None:
            # Use the hexagonal configuration as fallback
            hex_initial = self.create_hexagonal_initialization()
            best_points = self.local_refinement(hex_initial, max_iter=500)
            
        return best_points


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    optimizer = PointArrangementOptimizer(seed=42)
    return optimizer.optimize_all_configurations()


# EVOLVE-BLOCK-END