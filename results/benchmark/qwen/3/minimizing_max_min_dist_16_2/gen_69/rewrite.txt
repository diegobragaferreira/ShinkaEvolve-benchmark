# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import math
import time

class HexagonalSphericalOptimizer:
    """Optimizes point distribution using hexagonal grid with spherical coordinate transformation."""
    
    def __init__(self, n_points=16, dimension=2, seed=42):
        self.n_points = n_points
        self.dimension = dimension
        self.seed = seed
        np.random.seed(seed)
        
    def _initialize_hexagonal_sphere(self):
        """Initialize points using hexagonal grid on a sphere, then project to 2D."""
        # Create hexagonal grid points on a sphere using Fibonacci spiral approach
        # This provides better uniformity than simple grid placement
        points = []
        phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
        
        for i in range(self.n_points):
            y = 1 - (i / float(self.n_points - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            # Convert to Cartesian coordinates on sphere
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            
            points.append([x, y, z])
        
        # Project to 2D (remove z-coordinate)
        points_2d = np.array([[p[0], p[1]] for p in points])
        
        # Normalize to [0,1] range
        x_range = np.max(points_2d[:, 0]) - np.min(points_2d[:, 0])
        y_range = np.max(points_2d[:, 1]) - np.min(points_2d[:, 1])
        
        if x_range > 0:
            points_2d[:, 0] = (points_2d[:, 0] - np.min(points_2d[:, 0])) / x_range
        if y_range > 0:
            points_2d[:, 1] = (points_2d[:, 1] - np.min(points_2d[:, 1])) / y_range
            
        # Add small random perturbations to break symmetry
        noise_magnitude = 0.01
        noise = np.random.normal(0, noise_magnitude, points_2d.shape)
        points_2d += noise
        points_2d = np.clip(points_2d, 0, 1)
        
        return points_2d
    
    def _calculate_distances(self, points: np.ndarray) -> tuple:
        """Calculate minimum and maximum distances between all point pairs."""
        if len(points) < 2:
            return 0, 0
            
        # Calculate pairwise distances
        distances = pdist(points)
        
        # Get min and max distances
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        
        return min_distance, max_distance
    
    def _evaluate_ratio(self, points: np.ndarray) -> float:
        """Evaluate the min/max distance ratio."""
        min_d, max_d = self._calculate_distances(points)
        
        if max_d <= 0:
            return 0
            
        return min_d / max_d
    
    def _perturb_point(self, points: np.ndarray, idx: int, step_size: float = 0.01) -> np.ndarray:
        """Perturb a specific point."""
        new_points = points.copy()
        
        # Random perturbation
        delta = np.random.uniform(-step_size, step_size, self.dimension)
        new_points[idx] = points[idx] + delta
        
        # Boundary check
        new_points[idx] = np.clip(new_points[idx], 0, 1)
        
        return new_points
    
    def _perturb_neighborhood(self, points: np.ndarray, indices: list, step_size: float = 0.01) -> np.ndarray:
        """Perturb a group of points together to preserve local structure."""
        new_points = points.copy()
        
        # Calculate centroid of the neighborhood
        centroid = np.mean(points[indices], axis=0)
        
        # Apply coordinated perturbations relative to centroid
        for idx in indices:
            delta = np.random.uniform(-step_size, step_size, self.dimension)
            new_points[idx] = points[idx] + delta
            new_points[idx] = np.clip(new_points[idx], 0, 1)
            
        return new_points
    
    def _global_search_phase(self, points: np.ndarray, temperature: float) -> np.ndarray:
        """Perform global search with larger perturbations."""
        # Try various neighborhood sizes and move types
        if np.random.random() < 0.7:
            # Neighborhood move with 2-3 points
            neighborhood_size = np.random.randint(2, min(4, self.n_points))
            indices = np.random.choice(self.n_points, neighborhood_size, replace=False).tolist()
            return self._perturb_neighborhood(points, indices, step_size=temperature * 0.03)
        else:
            # Single point move
            point_idx = np.random.randint(0, self.n_points)
            return self._perturb_point(points, point_idx, step_size=temperature * 0.03)
    
    def _local_search_phase(self, points: np.ndarray, temperature: float) -> np.ndarray:
        """Perform local search with smaller perturbations."""
        # More focused, smaller moves
        if np.random.random() < 0.6:
            # Single point move with very small perturbation
            point_idx = np.random.randint(0, self.n_points)
            return self._perturb_point(points, point_idx, step_size=temperature * 0.005)
        else:
            # Small neighborhood move
            neighborhood_size = np.random.randint(2, min(3, self.n_points))
            indices = np.random.choice(self.n_points, neighborhood_size, replace=False).tolist()
            return self._perturb_neighborhood(points, indices, step_size=temperature * 0.01)
    
    def optimize(self, max_iterations: int = 5000) -> np.ndarray:
        """Optimize point distribution using hybrid approach."""
        
        # Initial configuration
        current_points = self._initialize_hexagonal_sphere()
        current_ratio = self._evaluate_ratio(current_points)
        
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Multiple phases with different cooling schedules
        phase = 0
        iteration = 0
        
        # Different cooling rates for different phases
        cooling_rates = [0.999, 0.9995, 0.9999]
        
        while iteration < max_iterations:
            # Determine which phase we're in (adjust cooling rate accordingly)
            phase = min(iteration // 1500, len(cooling_rates) - 1)
            cooling_rate = cooling_rates[phase]
            
            # Temperature decreases faster in later phases
            temperature = max(0.001, 1.0 * (cooling_rate ** (iteration // 50)))
            
            # Alternate between global and local search based on iteration
            if iteration % 10 == 0:
                # Global search phase
                new_points = self._global_search_phase(current_points, temperature)
            else:
                # Local search phase  
                new_points = self._local_search_phase(current_points, temperature)
            
            # Evaluate new solution
            new_ratio = self._evaluate_ratio(new_points)
            
            # Accept or reject the move
            if new_ratio > current_ratio:
                # Always accept better solutions
                current_points = new_points
                current_ratio = new_ratio
                
                if new_ratio > best_ratio:
                    best_points = new_points.copy()
                    best_ratio = new_ratio
            else:
                # Accept worse solutions with probability based on temperature
                if np.random.random() < math.exp((new_ratio - current_ratio) / temperature):
                    current_points = new_points
                    current_ratio = new_ratio
            
            iteration += 1
            
            # Early termination condition
            if iteration > 1000 and abs(current_ratio - best_ratio) < 1e-8:
                break
                
        return best_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Create optimizer instance
    optimizer = HexagonalSphericalOptimizer(n_points=16, dimension=2, seed=42)
    
    # Run optimization
    optimized_points = optimizer.optimize(max_iterations=5000)
    
    return optimized_points

# EVOLVE-BLOCK-END