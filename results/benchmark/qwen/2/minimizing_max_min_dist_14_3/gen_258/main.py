# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
import math
import time
from typing import Tuple, Optional
from scipy.spatial import SphericalVoronoi
from numba import jit, prange

@jit(nopython=True)
def fast_distance_matrix(points):
    """Fast computation of distance matrix using numba"""
    n = points.shape[0]
    distances = np.zeros((n, n))
    for i in prange(n):
        for j in range(i+1, n):
            dist = 0.0
            for k in range(3):
                diff = points[i,k] - points[j,k]
                dist += diff * diff
            dist = np.sqrt(dist)
            distances[i,j] = dist
            distances[j,i] = dist
    return distances

@jit(nopython=True)
def fast_min_max_ratio(distances):
    """Fast computation of min/max ratio"""
    n = distances.shape[0]
    if n < 2:
        return 0.0
    
    min_dist = np.inf
    max_dist = 0.0
    
    for i in range(n):
        for j in range(i+1, n):
            dist = distances[i,j]
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist
    
    if max_dist <= 0:
        return 0.0
    return min_dist / max_dist

class PointOptimizer:
    def __init__(self, num_points: int = 14, dimension: int = 3):
        self.num_points = num_points
        self.dimension = dimension
        self.best_points = None
        self.best_ratio = 0.0

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

    def calculate_ratio(self, points: np.ndarray) -> float:
        """Calculate the min/max distance ratio"""
        if len(points) < 2:
            return 0.0

        try:
            # Use fast numba-based distance matrix calculation
            distances = fast_distance_matrix(points)
            
            # Use fast numba-based ratio calculation
            ratio = fast_min_max_ratio(distances)
            return ratio
        except Exception:
            return 0.0

    def apply_constraints(self, points: np.ndarray) -> np.ndarray:
        """Ensure points stay within unit cube [0,1]^3"""
        return np.clip(points, 0, 1)

    def project_to_sphere(self, points: np.ndarray) -> np.ndarray:
        """Project points onto unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return points / norms

    def perturb_point(self, point: np.ndarray, delta: float = 0.01) -> np.ndarray:
        """Perturb a single point slightly"""
        noise = np.random.normal(0, delta, 3)
        return point + noise

    def initialize_points(self, seed: int) -> np.ndarray:
        """Initialize points using multiple strategies"""
        np.random.seed(seed)
        
        # Strategy 1: Fibonacci sphere with small noise
        initial_points = self.fibonacci_sphere(self.num_points)
        noise = np.random.normal(0, 0.01, initial_points.shape)
        initial_points += noise
        initial_points = self.project_to_sphere(initial_points)
        
        # Scale to unit cube [0.05, 0.95]^3
        initial_points = initial_points * 0.9 + 0.05
        
        # Apply constraints to ensure they're within bounds
        return self.apply_constraints(initial_points)

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
        alpha = 0.999  # Cooling rate
        max_iter = 20000  # Increased iterations for better convergence
        iter_without_improvement = 0
        max_no_improvement = 2000  # Increased early stopping threshold

        # Track recent improvements for adaptive cooling
        recent_improvements = []
        improvement_window = 100

        # Local refinement parameters
        refine_frequency = 1500  # How often to attempt local refinement
        refine_iterations = 100   # Number of local refinement steps

        # Improved adaptation: track improvement rates more carefully
        last_improvement_iteration = 0
        improvement_counts = [0] * 10  # Recent improvement counts

        for iteration in range(max_iter):
            # Perturb one random point with adaptive perturbation based on temperature
            idx = np.random.randint(0, self.num_points)
            old_point = points[idx].copy()
            
            # Temperature-dependent perturbation size
            perturbation_size = 0.008 * (1.0 + 0.5 * (T / 1.0))
            new_point = self.perturb_point(old_point, perturbation_size)
            
            # Project to sphere to maintain constraint
            new_point = self.project_to_sphere(new_point.reshape(1, 3)).reshape(-1)
            points[idx] = new_point

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
                last_improvement_iteration = iteration
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
                    last_improvement_iteration = iteration
                else:
                    # Revert the change
                    points[idx] = old_point
                    recent_improvements.append(False)

            # Update temperature with adaptive cooling based on performance
            if len(recent_improvements) > improvement_window:
                recent_improvements.pop(0)
                
                # Track recent improvement rate
                improvement_count = sum(recent_improvements[-improvement_window:])
                improvement_rate = improvement_count / improvement_window
                
                # Adjust cooling rate based on improvement rate
                if improvement_rate < 0.1:  # Low improvement rate
                    alpha = max(0.9995, alpha * 1.02)  # Accelerate cooling
                elif improvement_rate > 0.4:  # High improvement rate
                    alpha = min(0.9999, alpha * 0.98)  # Slow down cooling
                else:
                    alpha = min(0.9998, alpha * 0.995)  # Moderate cooling

            T = max(T * alpha, T_min)

            # Periodic local refinement to polish the solution
            if iteration % refine_frequency == 0 and iteration > 0:
                refined_points = self.local_refinement(best_points_local.copy(), refine_iterations)
                refined_ratio = self.calculate_ratio(refined_points)
                if refined_ratio > best_ratio_local:
                    best_ratio_local = refined_ratio
                    best_points_local = refined_points.copy()

            # Early stopping if no improvement for too long
            if iteration - last_improvement_iteration > max_no_improvement:
                break

        return best_points_local, best_ratio_local

    def local_refinement(self, points: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Apply enhanced local refinement to improve a point configuration"""
        current_points = points.copy()
        current_ratio = self.calculate_ratio(current_points)

        # Use a smart sampling approach for local refinement
        for iteration in range(max_iter):
            # For each point, try moves that might improve ratio
            best_move = None
            best_ratio_change = 0
            
            # Consider more directions for better exploration
            move_samples = 15
            
            for i in range(self.num_points):
                original_point = current_points[i].copy()
                
                # Sample various perturbations
                for _ in range(move_samples):
                    # Generate random perturbation
                    perturbation = np.random.normal(0, 0.001, 3)
                    
                    # Apply perturbation
                    new_point = original_point + perturbation
                    
                    # Project back to sphere
                    new_point = self.project_to_sphere(new_point.reshape(1, 3)).reshape(-1)
                    
                    # Test this move
                    test_points = current_points.copy()
                    test_points[i] = new_point
                    new_ratio = self.calculate_ratio(test_points)
                    
                    # Check improvement
                    ratio_change = new_ratio - current_ratio
                    
                    if ratio_change > best_ratio_change:
                        best_ratio_change = ratio_change
                        best_move = (i, new_point.copy())

            # Apply the best move if it improves the ratio
            if best_move is not None and best_ratio_change > 1e-12:
                idx, new_point = best_move
                current_points[idx] = new_point
                current_ratio += best_ratio_change
            else:
                break  # No significant improvement

        return current_points

    def spherical_voronoi_init(self, n: int, num_attempts: int = 5) -> np.ndarray:
        """Generate diverse initial points using SphericalVoronoi for better distribution."""
        best_points = None
        best_ratio = 0.0

        for attempt in range(num_attempts):
            # Generate random points on sphere
            np.random.seed(attempt + 1000)
            points = np.random.randn(n, 3)
            # Normalize to unit sphere
            norms = np.linalg.norm(points, axis=1)
            points = points / norms[:, np.newaxis]

            try:
                # Create SphericalVoronoi diagram
                sv = SphericalVoronoi(points, radius=1.0)

                # Get the centroids of the Voronoi cells as new candidate points
                voronoi_points = sv.vertices

                # If we got enough points, use them; otherwise fall back to original
                if len(voronoi_points) >= n:
                    # Take first n points, but make sure they're properly normalized
                    selected_points = voronoi_points[:n]
                    selected_points = selected_points / np.linalg.norm(selected_points, axis=1)[:, np.newaxis]
                    # Add small noise to break degeneracies
                    noise = np.random.normal(0, 0.001, selected_points.shape)
                    selected_points += noise
                    selected_points = selected_points / np.linalg.norm(selected_points, axis=1)[:, np.newaxis]

                    # Evaluate this configuration
                    ratio = self.calculate_ratio(selected_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = selected_points.copy()
                else:
                    # If Voronoi didn't give us enough points, use a simple approach
                    noise = np.random.normal(0, 0.005, points.shape)
                    noisy_points = points + noise
                    noisy_points = noisy_points / np.linalg.norm(noisy_points, axis=1)[:, np.newaxis]

                    ratio = self.calculate_ratio(noisy_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = noisy_points.copy()

            except Exception:
                # If SphericalVoronoi fails, fall back to regular noise
                noise = np.random.normal(0, 0.005, points.shape)
                noisy_points = points + noise
                noisy_points = noisy_points / np.linalg.norm(noisy_points, axis=1)[:, np.newaxis]

                ratio = self.calculate_ratio(noisy_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = noisy_points.copy()

        return best_points if best_points is not None else points

    def run_optimization(self) -> np.ndarray:
        """Run multi-start optimization to find best configuration"""
        # Try multiple starting configurations with diverse strategies
        seeds = [42, 123, 456, 789, 999, 555, 111, 222, 333, 666]
        
        # Add a spherical Voronoi initialization
        try:
            sv_points = self.spherical_voronoi_init(14)
            ratio = self.calculate_ratio(sv_points)
            if ratio > self.best_ratio:
                self.best_ratio = ratio
                self.best_points = sv_points.copy()
        except:
            pass

        # Run for multiple seeds
        for seed in seeds:
            try:
                points, ratio = self.optimize_single_start(seed)
                
                if ratio > self.best_ratio:
                    self.best_ratio = ratio
                    self.best_points = points.copy()
            except Exception:
                continue

        # Final validation
        if self.best_points is None:
            # Fallback to random initialization if something went wrong
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