# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution, minimize
import time

class PointDistributionOptimizer:
    """
    Optimizer for distributing 16 points in 2D space to maximize the ratio of 
    minimum to maximum pairwise distances.
    """
    
    def __init__(self, n_points=16, dimension=2, bounds=(0, 1), seed=42):
        self.n_points = n_points
        self.dimension = dimension
        self.bounds = bounds
        self.seed = seed
        np.random.seed(seed)
        
    def calculate_distance_matrix(self, points):
        """Calculate pairwise distance matrix with numerical stability."""
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        return distances
    
    def fitness_function(self, points_flat):
        """Calculate fitness as min/max distance ratio."""
        points = points_flat.reshape(-1, self.dimension)
        
        # Ensure points are within bounds with small epsilon
        epsilon = 1e-8
        points = np.clip(points, self.bounds[0] + epsilon, self.bounds[1] - epsilon)
        
        try:
            dist_matrix = self.calculate_distance_matrix(points)
            min_dist = np.min(dist_matrix)
            max_dist = np.max(dist_matrix)
            
            if max_dist < 1e-12:
                return 0
                
            ratio = min_dist / max_dist
            return ratio
            
        except Exception:
            return 0
    
    def generate_hexagonal_initial(self):
        """Generate initial configuration based on hexagonal packing."""
        points = np.zeros((self.n_points, self.dimension))
        rows = 4
        cols = 4
        spacing = 0.25
        idx = 0
        
        for row in range(rows):
            for col in range(cols):
                if idx < self.n_points:
                    x = col * spacing + (row % 2) * spacing * 0.5
                    y = row * spacing * np.sqrt(3) / 2
                    points[idx] = [x, y]
                    idx += 1
        
        # Normalize to [0,1] range and add noise
        points[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min()) * 0.8 + 0.1
        points[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min()) * 0.8 + 0.1
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        return points
    
    def generate_golden_spiral_initial(self):
        """Generate initial configuration based on golden spiral."""
        indices = np.arange(self.n_points)
        golden_angle = 2.399963229728653
        angles = golden_angle * indices
        radii = np.log(indices + 1) / np.log(self.n_points)
        points = np.column_stack([
            0.5 + 0.45 * radii * np.cos(angles),
            0.5 + 0.45 * radii * np.sin(angles)
        ])
        points = np.clip(points, 0, 1)
        return points
    
    def generate_grid_initial(self):
        """Generate initial configuration as perturbed square grid."""
        points = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0 + np.random.normal(0, 0.015)
                y = (j + 0.5) / 4.0 + np.random.normal(0, 0.015)
                points.append([x, y])
        points = np.array(points)
        points = np.clip(points, 0, 1)
        return points
    
    def generate_random_initial(self):
        """Generate random initial configuration with boundary awareness."""
        points = np.random.rand(self.n_points, self.dimension)
        points = np.clip(points, 0.05, 0.95)
        return points
    
    def generate_multiple_initializations(self):
        """Generate multiple initial point configurations and select the best."""
        strategies = [
            self.generate_hexagonal_initial(),
            self.generate_golden_spiral_initial(),
            self.generate_grid_initial(),
            self.generate_random_initial()
        ]
        
        best_points = None
        best_ratio = 0
        
        for points_strategy in strategies:
            ratio = self.fitness_function(points_strategy.flatten())
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = points_strategy.copy()
                
        return best_points if best_points is not None else strategies[0]
    
    def optimize_with_de_and_lbfgs(self, initial_points):
        """Perform optimization using differential evolution followed by L-BFGS-B."""
        flat_bounds = [(self.bounds[0], self.bounds[1]) for _ in range(self.n_points * self.dimension)]
        
        try:
            # Global optimization with differential evolution
            de_result = differential_evolution(
                self.fitness_function,
                flat_bounds,
                maxiter=300,
                popsize=30,
                tol=1e-8,
                mutation=(0.7, 1.2),
                recombination=0.8,
                seed=self.seed,
                disp=False
            )
            
            if de_result.success:
                # Local refinement with L-BFGS-B
                lbfgs_result = minimize(
                    self.fitness_function,
                    de_result.x,
                    method='L-BFGS-B',
                    bounds=flat_bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                
                if lbfgs_result.success:
                    return lbfgs_result.x.reshape(-1, 2)
                    
        except Exception as e:
            pass
            
        # If optimization fails, return initial points
        return initial_points
    
    def get_optimized_points(self):
        """Main method to compute optimized point distribution."""
        # Generate initial configurations
        initial_points = self.generate_multiple_initializations()
        
        # Optimize using multi-stage approach
        optimized_points = self.optimize_with_de_and_lbfgs(initial_points)
        
        # Ensure final points are within bounds
        optimized_points = np.clip(optimized_points, self.bounds[0], self.bounds[1])
        
        return optimized_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = PointDistributionOptimizer()
    return optimizer.get_optimized_points()

# EVOLVE-BLOCK-END
