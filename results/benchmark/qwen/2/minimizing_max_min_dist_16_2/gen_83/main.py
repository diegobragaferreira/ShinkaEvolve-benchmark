# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

class PointConfigurationGenerator:
    """Generates various initial point configurations for optimization."""
    
    @staticmethod
    def generate_hexagonal_grid(seed=42):
        """Generate points in a hexagonal lattice pattern."""
        np.random.seed(seed)
        points = []
        rows, cols = 4, 4
        
        for i in range(rows):
            for j in range(cols):
                x_offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + x_offset) / 3.0
                y = i / 3.0
                # Add small random perturbation
                x += np.random.normal(0, 0.01)
                y += np.random.normal(0, 0.01)
                # Ensure within bounds
                x = np.clip(x, 0.001, 0.999)
                y = np.clip(y, 0.001, 0.999)
                points.append([x, y])
        
        return np.array(points)
    
    @staticmethod
    def generate_ring_distribution(seed=42):
        """Generate points in concentric rings."""
        np.random.seed(seed)
        points = []
        # Two rings with 8 points each
        radii = [0.3, 0.7]
        angles_per_ring = [8, 8]
        
        for r_idx, (radius, num_angles) in enumerate(zip(radii, angles_per_ring)):
            for i in range(num_angles):
                angle = 2 * np.pi * i / num_angles
                x = 0.5 + radius * np.cos(angle) * 0.4
                y = 0.5 + radius * np.sin(angle) * 0.4
                # Ensure within bounds
                x = np.clip(x, 0.001, 0.999)
                y = np.clip(y, 0.001, 0.999)
                points.append([x, y])
        
        return np.array(points)
    
    @staticmethod
    def generate_fibonacci_spiral(seed=42):
        """Generate points using Fibonacci spiral-like arrangement."""
        np.random.seed(seed)
        points = []
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        
        for i in range(16):
            # Modified Fibonacci approach for better distribution
            theta = np.arccos(-1 + (2 * i) / 15)  # elevation angle
            phi_angle = (i * 2 * np.pi) / (phi * phi)  # azimuthal angle
            
            # Convert to cartesian coordinates
            x = np.sin(theta) * np.cos(phi_angle)
            y = np.sin(theta) * np.sin(phi_angle)
            
            # Map to [0.05, 0.95] range to avoid boundaries
            x = 0.05 + 0.9 * (x + 1) / 2
            y = 0.05 + 0.9 * (y + 1) / 2
            
            points.append([x, y])
        
        return np.array(points)
    
    @staticmethod
    def generate_regular_grid(seed=42):
        """Generate a regular grid with controlled perturbations."""
        np.random.seed(seed)
        points = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0
                y = (j + 0.5) / 4.0
                # Add small random perturbation
                x += np.random.normal(0, 0.02)
                y += np.random.normal(0, 0.02)
                # Ensure within bounds
                x = np.clip(x, 0.001, 0.999)
                y = np.clip(y, 0.001, 0.999)
                points.append([x, y])
        return np.array(points)


class Optimizer:
    """Handles the optimization process with multiple stages."""
    
    def __init__(self, bounds):
        self.bounds = bounds
    
    def objective(self, x):
        """Objective function to maximize the min/max distance ratio."""
        points = x.reshape(-1, 2)
        distances = pdist(points)
        
        if len(distances) == 0:
            return 0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max <= 1e-12:
            return 0
            
        ratio = d_min / d_max
        return -ratio  # Negative because we want to maximize ratio
    
    def optimize_single(self, initial_points, max_iter=100):
        """Perform single optimization from given initial points."""
        try:
            result = minimize(
                self.objective,
                initial_points.flatten(),
                method='L-BFGS-B',
                bounds=self.bounds,
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            
            if result.success:
                return result.x.reshape(-1, 2)
        except Exception:
            return None
        return None


class MultiStagePointOptimizer:
    """Main optimizer class that orchestrates the entire optimization process."""
    
    def __init__(self):
        self.bounds = [(0.001, 0.999) for _ in range(32)]
        self.config_generator = PointConfigurationGenerator()
        self.optimizer = Optimizer(self.bounds)
    
    def evaluate_configuration(self, points):
        """Evaluate a configuration and return its min/max ratio."""
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max <= 1e-12:
            return 0
        return d_min / d_max
    
    def get_best_initial_config(self):
        """Generate and evaluate multiple initial configurations."""
        # Generate different types of configurations
        configs = [
            self.config_generator.generate_hexagonal_grid(42),
            self.config_generator.generate_ring_distribution(43),
            self.config_generator.generate_fibonacci_spiral(44),
            self.config_generator.generate_regular_grid(45)
        ]
        
        best_ratio = -np.inf
        best_config = None
        
        for config in configs:
            ratio = self.evaluate_configuration(config)
            if ratio > best_ratio:
                best_ratio = ratio
                best_config = config.copy()
        
        return best_config
    
    def optimize_with_fallback(self, initial_points):
        """Run optimization with fallback strategies."""
        # Stage 1: Initial coarse optimization
        coarse_points = self.optimizer.optimize_single(initial_points, max_iter=50)
        if coarse_points is None:
            return initial_points
            
        # Evaluate coarse result
        coarse_ratio = self.evaluate_configuration(coarse_points)
        
        # Stage 2: Fine optimization if coarse is reasonable
        if coarse_ratio > 0.1:  # Only proceed if coarse solution is decent
            fine_points = self.optimizer.optimize_single(coarse_points, max_iter=100)
            if fine_points is not None:
                fine_ratio = self.evaluate_configuration(fine_points)
                return fine_points if fine_ratio > coarse_ratio else coarse_points
        else:
            # Even if coarse is poor, return it as fallback
            return coarse_points
    
    def run_optimization(self):
        """Main optimization routine."""
        # Get best initial configuration
        initial_config = self.get_best_initial_config()
        
        # Run optimization with fallback
        final_points = self.optimize_with_fallback(initial_config)
        
        # Final evaluation and cleanup
        final_ratio = self.evaluate_configuration(final_points)
        
        # Ensure all points are within bounds
        final_points[:, 0] = np.clip(final_points[:, 0], 0, 1)
        final_points[:, 1] = np.clip(final_points[:, 1], 0, 1)
        
        return final_points


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = MultiStagePointOptimizer()
    return optimizer.run_optimization()

# EVOLVE-BLOCK-END