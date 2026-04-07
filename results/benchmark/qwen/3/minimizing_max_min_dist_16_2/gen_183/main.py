# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist, pdist
import math
import time
import warnings
from typing import List, Tuple

class ConfigurationGenerator:
    """Generates diverse initial point configurations."""
    
    @staticmethod
    def hexagonal_grid(n_points: int = 16) -> np.ndarray:
        """Generate points in a hexagonal pattern with optimal spacing."""
        points = []
        
        # Create hexagonal arrangement for 16 points
        # Use 4x4 grid with offset rows for better packing
        rows, cols = 4, 4
        spacing = 1.0 / 3.0
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= n_points:
                    break
                x = j * spacing
                y = i * spacing
                
                # Offset every other row
                if i % 2 == 1:
                    x += spacing * 0.5
                
                points.append([x, y])
        
        points = np.array(points[:n_points])
        
        # Normalize to [0,1] range
        if np.max(points) > 0:
            points = points / np.max(points)
        
        # Add strategic perturbations to break symmetry
        noise_magnitude = 0.02
        for i in range(len(points)):
            # Different noise for corner points
            if i in [0, 3, 12, 15]:  # Corner indices
                points[i, 0] += np.random.normal(0, noise_magnitude * 2.0)
                points[i, 1] += np.random.normal(0, noise_magnitude * 2.0)
            else:
                points[i, 0] += np.random.normal(0, noise_magnitude)
                points[i, 1] += np.random.normal(0, noise_magnitude)
        
        # Clip to bounds
        points = np.clip(points, 0, 1)
        return points
    
    @staticmethod
    def perturbed_grid(n_points: int = 16) -> np.ndarray:
        """Generate perturbed grid configuration."""
        # Create regular grid and add noise
        grid_size = int(np.ceil(np.sqrt(n_points)))
        spacing = 1.0 / (grid_size - 1) if grid_size > 1 else 1.0
        
        points = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) >= n_points:
                    break
                x = i * spacing + np.random.normal(0, 0.05 * spacing)
                y = j * spacing + np.random.normal(0, 0.05 * spacing)
                points.append([x, y])
        
        points = np.clip(np.array(points[:n_points]), 0, 1)
        return points
    
    @staticmethod
    def random_configuration(n_points: int = 16) -> np.ndarray:
        """Generate random point configuration."""
        return np.random.rand(n_points, 2)
    
    @staticmethod
    def generate_all_configurations(n_points: int = 16) -> List[np.ndarray]:
        """Generate multiple diverse initial configurations."""
        configs = []
        
        # Various initialization strategies
        configs.append(ConfigurationGenerator.hexagonal_grid(n_points))
        
        # Different perturbations
        for i in range(3):
            config = ConfigurationGenerator.perturbed_grid(n_points)
            # Apply different noise levels
            noise_level = 0.015 + i * 0.005
            config += np.random.normal(0, noise_level, config.shape)
            config = np.clip(config, 0, 1)
            configs.append(config)
        
        # Random configuration
        configs.append(ConfigurationGenerator.random_configuration(n_points))
        
        return configs

class FitnessCalculator:
    """Calculates fitness metrics for point configurations."""
    
    @staticmethod
    def min_max_ratio(points: np.ndarray) -> float:
        """Compute the ratio of minimum to maximum distance."""
        if len(points) < 2:
            return 0.0
            
        # Use efficient distance computation
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max <= 0:
            return 0.0
            
        return d_min / d_max

class Optimizer:
    """Base optimizer interface."""
    
    def optimize(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        raise NotImplementedError

class SimulatedAnnealing(Optimizer):
    """Advanced simulated annealing optimization."""
    
    def __init__(self, max_iterations: int = 5000):
        self.max_iterations = max_iterations
    
    def optimize(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Optimize using advanced simulated annealing."""
        def compute_ratio(points_array):
            if len(points_array) < 2:
                return 0.0
            distances = cdist(points_array, points_array)
            np.fill_diagonal(distances, np.inf)
            d_min = np.min(distances)
            d_max = np.max(distances)
            if d_max <= 0:
                return 0.0
            return d_min / d_max
        
        current_points = points.copy()
        current_ratio = compute_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Adaptive temperature schedule
        temp = 0.1
        min_temp = 1e-8
        cooling_rate = 0.999
        
        for iteration in range(self.max_iterations):
            # Dynamic temperature adjustment
            if iteration > 0 and iteration % 100 == 0:
                # Check recent improvement
                temp *= cooling_rate
            
            temp = max(temp, min_temp)
            
            # Generate neighbor with adaptive strategy
            new_points = current_points.copy()
            
            # Choose move type based on iteration
            if iteration < self.max_iterations * 0.3:
                # Early: aggressive moves
                move_type = np.random.choice(['single', 'cluster'], p=[0.7, 0.3])
                step_size = 0.03
            elif iteration < self.max_iterations * 0.7:
                # Middle: balanced
                move_type = np.random.choice(['single', 'pair', 'cluster'], p=[0.5, 0.3, 0.2])
                step_size = 0.015
            else:
                # Late: fine-tuning
                move_type = np.random.choice(['single', 'pair'], p=[0.7, 0.3])
                step_size = 0.005
            
            if move_type == 'single':
                idx = np.random.randint(len(new_points))
                new_points[idx, 0] += np.random.normal(0, step_size)
                new_points[idx, 1] += np.random.normal(0, step_size)
            elif move_type == 'pair':
                # Move two nearby points together
                distances = cdist(new_points, new_points)
                np.fill_diagonal(distances, np.inf)
                min_indices = np.unravel_index(np.argmin(distances), distances.shape)
                idx1, idx2 = min_indices
                movement = np.random.normal(0, step_size * 0.8, 2)
                new_points[idx1] += movement
                new_points[idx2] += movement
            else:  # cluster
                num_cluster = min(3, len(new_points))
                cluster_indices = np.random.choice(len(new_points), num_cluster, replace=False)
                centroid = np.mean(new_points[cluster_indices], axis=0)
                movement = np.random.normal(0, step_size * 0.6, 2)
                for idx in cluster_indices:
                    new_points[idx] += movement
            
            # Boundary constraints
            new_points = np.clip(new_points, 0, 1)
            
            # Evaluate
            new_ratio = compute_ratio(new_points)
            
            # Accept or reject
            if new_ratio > current_ratio or np.random.rand() < math.exp((new_ratio - current_ratio) / temp):
                current_points = new_points.copy()
                current_ratio = new_ratio
                
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = current_points.copy()
            
            if temp < min_temp:
                break
        
        return best_points, best_ratio

class PointOptimizer:
    """Main optimization controller."""
    
    def __init__(self, n_points: int = 16, seed: int = 42, max_time_seconds: int = 180):
        self.n_points = n_points
        self.seed = seed
        self.max_time_seconds = max_time_seconds
        np.random.seed(seed)
    
    def optimize(self) -> np.ndarray:
        """Main optimization loop."""
        start_time = time.time()
        
        # Generate initial configurations
        configurations = ConfigurationGenerator.generate_all_configurations(self.n_points)
        
        best_points = None
        best_ratio = -float('inf')
        
        sa_optimizer = SimulatedAnnealing(max_iterations=3000)
        
        # Try each configuration
        for i, initial_config in enumerate(configurations):
            if time.time() - start_time > self.max_time_seconds - 5:
                break
                
            try:
                # Apply simulated annealing optimization
                optimized_points, ratio = sa_optimizer.optimize(initial_config)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
            except Exception as e:
                warnings.warn(f"Optimization error for config {i}: {str(e)}")
                continue
        
        # Return best solution found
        if best_points is not None:
            return best_points
        else:
            # Fallback to first configuration
            return configurations[0] if configurations else np.random.rand(self.n_points, 2)

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    optimizer = PointOptimizer(n_points=16, seed=42, max_time_seconds=180)
    points = optimizer.optimize()
    return points

# EVOLVE-BLOCK-END