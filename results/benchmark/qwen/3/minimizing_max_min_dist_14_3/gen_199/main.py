# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import SphericalVoronoi
from scipy.spatial.distance import pdist
import warnings
from typing import Tuple, Optional, Callable, Any

class PointOptimizer:
    """A modular optimizer for 3D point dispersion problems."""
    
    def __init__(self, n_points: int = 14):
        self.n_points = n_points
        self.best_solution = None
        self.best_ratio = 0.0
        
    def spherical_map(self, points: np.ndarray) -> np.ndarray:
        """Map points from 3D space to unit sphere using normalization."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return points / norms
    
    def spherical_voronoi_quality(self, sphere_points: np.ndarray) -> float:
        """Calculate quality based on Voronoi cell areas on sphere."""
        if len(sphere_points) < 2:
            return 0
        
        # Create spherical Voronoi diagram
        try:
            sv = SphericalVoronoi(sphere_points)
            # Calculate total area of Voronoi cells
            cell_areas = sv.calculate_areas()
            # Quality is inversely related to variance of cell areas
            # More uniform areas indicate better distribution
            if len(cell_areas) > 0:
                mean_area = np.mean(cell_areas)
                if mean_area > 0:
                    variance = np.var(cell_areas)
                    # Return inverse variance (higher is better)
                    return 1.0 / (1.0 + variance / mean_area**2)
        except Exception:
            pass
        return 0
    
    def min_max_ratio(self, points: np.ndarray) -> float:
        """Calculate the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0
        
        # Calculate pairwise distances
        distances = pdist(points)
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max == 0:
            return 0
        
        return d_min / d_max
    
    def penalty_objective(self, x_flat: np.ndarray, penalty_weight: float = 1e6) -> float:
        """Objective function with penalty for out-of-bounds points."""
        # Reshape flat array back to points
        points = x_flat.reshape((self.n_points, 3))
        
        # Apply penalty for constraint violations
        penalty = 0
        for i in range(self.n_points):
            for j in range(3):  # x, y, z coordinates
                if points[i, j] < 0:
                    penalty += penalty_weight * (0 - points[i, j])**2
                elif points[i, j] > 1:
                    penalty += penalty_weight * (points[i, j] - 1)**2
        
        # Calculate min/max ratio
        ratio = self.min_max_ratio(points)
        
        # Return value to minimize (negative ratio + penalty)
        return -ratio + penalty
    
    def adaptive_penalty_objective(self, x_flat: np.ndarray, iteration: int = 0, 
                                 penalty_weight: float = 1e6) -> float:
        """Objective function with adaptive penalty for out-of-bounds points."""
        # Reshape flat array back to points
        points = x_flat.reshape((self.n_points, 3))
        
        # Apply penalty for constraint violations
        penalty = 0
        for i in range(self.n_points):
            for j in range(3):  # x, y, z coordinates
                if points[i, j] < 0:
                    penalty += penalty_weight * (0 - points[i, j])**2 * (1 + iteration * 0.1)
                elif points[i, j] > 1:
                    penalty += penalty_weight * (points[i, j] - 1)**2 * (1 + iteration * 0.1)
        
        # Calculate min/max ratio
        ratio = self.min_max_ratio(points)
        
        # Return value to minimize (negative ratio + penalty)
        return -ratio + penalty
    
    def fibonacci_sphere(self, n: int) -> np.ndarray:
        """Generate n points evenly distributed on a unit sphere using Fibonacci spiral method."""
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle
        
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def latin_hypercube_sampling(self, n: int, d: int, seed: int = 42) -> np.ndarray:
        """Generate n points using Latin Hypercube Sampling in d dimensions."""
        np.random.seed(seed)
        samples = np.zeros((n, d))
        
        for i in range(d):
            # Generate random permutation for each dimension
            perm = np.random.permutation(n)
            samples[:, i] = perm
        
        # Normalize to [0, 1]
        samples = samples / (n - 1)
        
        return samples
    
    def create_initial_placement(self) -> np.ndarray:
        """Create initial point placement using enhanced spherical code approach."""
        # Method: Generate points using Fibonacci-like distribution but with better uniformity
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle
        
        for i in range(self.n_points):
            y = 1 - (i / float(self.n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        initial_points = np.array(points)
        
        # Improve distribution by applying multiple small perturbations
        np.random.seed(42)
        for _ in range(15):  # More perturbations for better uniformity
            # Add small random perturbations
            perturbation = np.random.normal(0, 0.015, (self.n_points, 3))
            initial_points += perturbation
            
            # Project back to sphere surface using normalization
            norms = np.linalg.norm(initial_points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            initial_points = initial_points / norms
        
        # Normalize to unit sphere and scale to unit cube [0,1]^3
        initial_points = (initial_points + 1) / 2
        
        return initial_points
    
    def get_initialization_strategies(self) -> list:
        """Return list of initialization strategy functions."""
        return [
            lambda: self.create_initial_placement(),  # Enhanced Fibonacci sphere
            lambda: (self.fibonacci_sphere(self.n_points) + 1) / 2,  # Basic Fibonacci
            lambda: self.latin_hypercube_sampling(self.n_points, 3, seed=42),  # LHS
            lambda: np.random.rand(self.n_points, 3)  # Random
        ]
    
    def global_optimization_phase(self, initial_points: np.ndarray, 
                                bounds: list, restart: int = 0) -> Optional[np.ndarray]:
        """Perform global optimization using differential evolution."""
        try:
            # Use dynamic population size and adaptive parameters
            popsize = 20 + restart * 5  # Increase population size with restarts
            maxiter = 150 + restart * 50  # More iterations with restarts
            
            result = differential_evolution(
                lambda x: self.adaptive_penalty_objective(x, iteration=restart),
                bounds,
                seed=42 + restart * 10,
                maxiter=maxiter,
                popsize=popsize,
                tol=1e-7,
                mutation=(0.5, 1),
                recombination=0.8,
                disp=False
            )
            
            # Extract optimized points
            optimized_points = result.x.reshape((self.n_points, 3))
            return optimized_points
            
        except Exception as e:
            warnings.warn(f"Global optimization failed: {str(e)}")
            return None
    
    def local_refinement_phase(self, points: np.ndarray, bounds: list) -> np.ndarray:
        """Perform local refinement with L-BFGS-B."""
        try:
            # Flatten the points
            x0_refine = points.flatten()
            
            # Refinement with L-BFGS-B using tighter tolerances
            result_refined = minimize(
                lambda x: self.adaptive_penalty_objective(x, iteration=10),
                x0_refine,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-9, 'gtol': 1e-9, 'maxiter': 1000},
                tol=1e-9
            )
            
            refined_points = result_refined.x.reshape((self.n_points, 3))
            return refined_points
            
        except Exception as e:
            warnings.warn(f"Local refinement failed: {str(e)}")
            return points
    
    def optimize(self) -> np.ndarray:
        """Main optimization loop with multiple strategies and restarts."""
        # Define bounds for each coordinate (0 to 1)
        bounds = [(0, 1) for _ in range(self.n_points * 3)]
        
        # Try different combinations of strategies with multiple restarts
        best_ratio = 0
        best_points = None
        
        # Track improvement for early stopping
        improvement_streak = 0
        max_improvement_streak = 10
        
        initialization_strategies = self.get_initialization_strategies()
        
        for restart in range(3):  # 3 restart rounds for better exploration
            for i, init_func in enumerate(initialization_strategies):
                # Generate initial points
                try:
                    initial_points = init_func()
                except Exception as e:
                    warnings.warn(f"Initialization failed: {str(e)}")
                    continue
                
                # Phase 1: Global optimization with differential evolution
                optimized_points = self.global_optimization_phase(initial_points, bounds, restart)
                
                if optimized_points is not None:
                    # Calculate final ratio
                    final_ratio = self.min_max_ratio(optimized_points)
                    
                    # Store best result
                    if final_ratio > best_ratio:
                        best_ratio = final_ratio
                        best_points = optimized_points.copy()
                        improvement_streak = 0  # Reset streak
                    else:
                        improvement_streak += 1
                    
                    # Early stopping if no improvement for too many attempts
                    if improvement_streak >= max_improvement_streak:
                        break
                        
                    # Early stopping if we reach good enough quality
                    if best_ratio >= 0.4898 * 0.98:
                        break
                        
        # Phase 2: Local refinement with L-BFGS-B if we found a good candidate
        if best_points is not None and best_ratio > 0:
            try:
                refined_points = self.local_refinement_phase(best_points, bounds)
                final_ratio = self.min_max_ratio(refined_points)
                
                # Update if improved
                if final_ratio > best_ratio:
                    best_points = refined_points
            except Exception as e:
                warnings.warn(f"Final refinement failed: {str(e)}")
        
        # Ensure we return valid points even if optimization failed
        if best_points is None:
            # Fallback to enhanced Fibonacci initialization
            best_points = self.create_initial_placement()
        
        return best_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    optimizer = PointOptimizer(n_points=14)
    return optimizer.optimize()

# EVOLVE-BLOCK-END