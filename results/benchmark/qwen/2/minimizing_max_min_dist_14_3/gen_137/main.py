# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import math
import time
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

class PointOptimizer:
    def __init__(self, num_points: int = 14, dimension: int = 3):
        self.num_points = num_points
        self.dimension = dimension
        self.best_points = None
        self.best_ratio = 0.0
        self.max_evaluations = 100000
        
    def fibonacci_sphere(self, samples: int = 14) -> np.ndarray:
        """Generate points distributed evenly on a sphere using Fibonacci method"""
        points = []
        phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
        
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            
            points.append([x, y, z])
            
        return np.array(points)
    
    def project_to_sphere(self, points: np.ndarray) -> np.ndarray:
        """Project points onto unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return points / norms

    def initialize_fibonacci_sphere(self, seed: int) -> np.ndarray:
        """Initialize points using Fibonacci sphere method"""
        np.random.seed(seed)
        initial_points = self.fibonacci_sphere(self.num_points)
        # Scale and shift to unit cube [0.05, 0.95]^3 to avoid boundary issues
        initial_points = initial_points * 0.9 + 0.05
        return self.project_to_sphere(initial_points)

    def initialize_polyhedron(self, seed: int) -> np.ndarray:
        """Initialize points based on regular icosahedron vertices"""
        np.random.seed(seed)
        # Regular icosahedron vertices (normalized)
        phi = (1 + np.sqrt(5)) / 2
        vertices = [
            (-1, 0, phi), (1, 0, phi), (-1, 0, -phi), (1, 0, -phi),
            (0, phi, 1), (0, phi, -1), (0, -phi, 1), (0, -phi, -1),
            (phi, 1, 0), (-phi, 1, 0), (phi, -1, 0), (-phi, -1, 0)
        ]
        vertices = np.array(vertices)
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices[0])
        
        # Add more points by taking edge midpoints for better distribution
        edges = []
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                dist = np.linalg.norm(vertices[i] - vertices[j])
                if abs(dist - 2) < 0.1:  # approximately the edge length
                    edges.append((i, j))
        
        # Add midpoints of edges
        additional_points = []
        for i, j in edges[:2]:  # Take first 2 edges
            midpoint = (vertices[i] + vertices[j]) / 2
            midpoint = midpoint / np.linalg.norm(midpoint)
            additional_points.append(midpoint)
        
        # Combine and ensure we have proper number of points
        all_points = np.vstack([vertices, additional_points])
        if len(all_points) > self.num_points:
            # Select points that are well spread
            return self.project_to_sphere(all_points[:self.num_points])
        elif len(all_points) < self.num_points:
            # Fill with fibonacci points
            fib_points = self.fibonacci_sphere(self.num_points - len(all_points))
            return self.project_to_sphere(np.vstack([all_points, fib_points]))
        else:
            return self.project_to_sphere(all_points)
    
    def initialize_spherical_voronoi(self, seed: int) -> np.ndarray:
        """Initialize points using SphericalVoronoi approach"""
        np.random.seed(seed)
        try:
            # Generate random points on sphere
            points = np.random.randn(self.num_points, self.dimension)
            points = self.project_to_sphere(points)
            
            # Create SphericalVoronoi diagram (this is simplified for performance)
            # Use direct random points with noise instead of complex Voronoi computation
            noise = np.random.normal(0, 0.05, points.shape)
            points += noise
            points = self.project_to_sphere(points)
            
            return points
        except Exception:
            # Fallback to Fibonacci initialization
            return self.initialize_fibonacci_sphere(seed)
    
    def initialize_points(self, seed: int) -> np.ndarray:
        """Initialize points using multiple strategies"""
        np.random.seed(seed)
        
        # Try different initialization strategies
        strategies = [
            self.initialize_fibonacci_sphere,
            self.initialize_polyhedron,
            self.initialize_spherical_voronoi
        ]
        
        # Pick a random strategy based on seed
        strategy = strategies[seed % len(strategies)]
        return strategy(seed)
    
    def calculate_ratio(self, points: np.ndarray) -> float:
        """Calculate the min/max distance ratio"""
        if len(points) < 2:
            return 0.0
            
        try:
            distances = pdist(points)
            
            if len(distances) == 0:
                return 0.0
                
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            # Avoid division by zero
            if d_max <= 0:
                return 0.0
                
            return d_min / d_max
        except Exception:
            return 0.0
    
    def apply_constraints(self, points: np.ndarray) -> np.ndarray:
        """Ensure points stay within unit cube [0,1]^3"""
        return np.clip(points, 0, 1)
    
    def perturb_point(self, point: np.ndarray, delta: float = 0.01) -> np.ndarray:
        """Perturb a single point slightly"""
        noise = np.random.normal(0, delta, 3)
        return point + noise
    
    def adaptive_perturbation(self, current_ratio: float, temperature: float) -> float:
        """Calculate adaptive perturbation size based on current state"""
        # Start with base perturbation size
        base_size = 0.005
        
        # Scale based on current ratio (larger perturbations when ratio is poor)
        ratio_factor = min(1.0, max(0.1, current_ratio / 0.3))
        
        # Scale based on temperature (larger perturbations when temperature is high)
        temp_factor = min(1.0, max(0.1, temperature))
        
        return base_size * ratio_factor * temp_factor
    
    def optimize_single_start(self, seed: int) -> Tuple[np.ndarray, float]:
        """Perform optimization from a single starting configuration"""
        # Initialize points
        points = self.initialize_points(seed)
        
        # Simulated Annealing parameters
        current_ratio = self.calculate_ratio(points)
        best_ratio_local = current_ratio
        best_points_local = points.copy()
        
        # Cooling schedule parameters
        T = 1.0  # Initial temperature
        T_min = 1e-8  # Minimum temperature
        base_alpha = 0.999  # Base cooling rate
        max_iter = 20000  # Max iterations
        iter_without_improvement = 0
        max_no_improvement = 2000  # Early stopping threshold
        
        # Track recent improvements for adaptive cooling
        recent_improvements = []
        improvement_window = 100
        
        # Track improvement history
        improvement_history = []
        history_length = 500
        
        for iteration in range(max_iter):
            # Perturb one random point
            idx = np.random.randint(0, self.num_points)
            old_point = points[idx].copy()
            
            # Calculate adaptive perturbation size
            perturbation_size = self.adaptive_perturbation(current_ratio, T)
            new_point = self.perturb_point(old_point, perturbation_size)
            points[idx] = new_point
            
            # Apply constraints
            points = self.apply_constraints(points)
            
            # Calculate new ratio
            new_ratio = self.calculate_ratio(points)
            
            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio:
                current_ratio = new_ratio
                if new_ratio > best_ratio_local:
                    best_ratio_local = new_ratio
                    best_points_local = points.copy()
                iter_without_improvement = 0
                recent_improvements.append(True)
                improvement_history.append(iteration)
            else:
                # Accept with probability based on temperature
                delta = new_ratio - current_ratio
                if np.random.rand() < np.exp(delta / T):
                    current_ratio = new_ratio
                    if new_ratio > best_ratio_local:
                        best_ratio_local = new_ratio
                        best_points_local = points.copy()
                    iter_without_improvement = 0
                    recent_improvements.append(True)
                    improvement_history.append(iteration)
                else:
                    # Revert the change
                    points[idx] = old_point
                    recent_improvements.append(False)
            
            # Update temperature with adaptive cooling
            # Dynamic cooling rate based on recent improvements
            alpha = base_alpha
            if len(recent_improvements) > improvement_window:
                recent_improvements.pop(0)
                improvement_rate = sum(recent_improvements) / len(recent_improvements)
                
                # Adapt cooling rate based on improvement rate
                if improvement_rate < 0.05:  # Very slow improvement
                    alpha = min(0.9999, base_alpha * 1.02)  # Faster cooling
                elif improvement_rate < 0.15:  # Slow improvement
                    alpha = min(0.9998, base_alpha * 1.01)  # Moderate cooling
                elif improvement_rate > 0.4:  # Fast improvement
                    alpha = max(0.999, base_alpha * 0.99)  # Slower cooling
            
            T = max(T * alpha, T_min)
            
            # Early stopping if no improvement for too long
            iter_without_improvement += 1
            if iter_without_improvement > max_no_improvement:
                break
            
            # Early stopping based on convergence rate
            if len(improvement_history) > history_length:
                recent_improvements_count = len([i for i in improvement_history if i > iteration - history_length])
                if recent_improvements_count < 5 and iteration > history_length:
                    break
                    
        return best_points_local, best_ratio_local
    
    def multi_start_optimization(self) -> np.ndarray:
        """Run multi-start optimization to find best configuration"""
        # Use thread pool executor for parallel processing of seeds
        seeds = [42, 123, 456, 789, 999, 555, 111, 222]
        
        best_result = None
        best_ratio = 0.0
        
        # Process seeds in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_seed = {
                executor.submit(self.optimize_single_start, seed): seed 
                for seed in seeds
            }
            
            for future in as_completed(future_to_seed):
                try:
                    points, ratio = future.result(timeout=300)  # 5 minute timeout per seed
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_result = points.copy()
                except Exception as e:
                    continue
        
        # If no results found, use fallback
        if best_result is None:
            points, ratio = self.optimize_single_start(42)
            best_result = points
            best_ratio = ratio
            
        return best_result, best_ratio
    
    def run_optimization(self) -> np.ndarray:
        """Run the main optimization routine"""
        try:
            points, ratio = self.multi_start_optimization()
            
            if ratio > self.best_ratio:
                self.best_ratio = ratio
                self.best_points = points.copy()
                
        except Exception as e:
            # Fallback to basic approach if anything goes wrong
            np.random.seed(42)
            self.best_points = np.random.rand(self.num_points, self.dimension)
            self.best_points = self.project_to_sphere(self.best_points)
        
        return self.best_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    optimizer = PointOptimizer(num_points=14, dimension=3)
    return optimizer.run_optimization()

# EVOLVE-BLOCK-END