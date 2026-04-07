# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi
from sklearn.cluster import KMeans
import warnings
import time
from typing import Tuple, List, Callable, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PointDistributionOptimizer:
    """
    A modular optimizer for distributing 14 points in 3D space to maximize
    the ratio of minimum to maximum pairwise distances.
    """
    
    def __init__(self, n_points: int = 14, dimensions: int = 3, seed: int = 42):
        self.n_points = n_points
        self.dimensions = dimensions
        self.seed = seed
        np.random.seed(seed)
        
    def _initialize_spherical_fibonacci(self) -> np.ndarray:
        """Generate points on sphere using Fibonacci spiral method"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        for i in range(self.n_points):
            # Latitude
            phi = np.arccos(1 - 2*i/(self.n_points-1))
            # Longitude
            theta = 2 * np.pi * i / golden_ratio
            
            # Convert to Cartesian coordinates
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            
            points.append([x, y, z])
            
        return np.array(points)
    
    def _initialize_cube_grid(self) -> np.ndarray:
        """Initialize points in a 3D cube grid"""
        # Find appropriate grid size
        grid_size = int(np.ceil(self.n_points**(1/3)))
        coords = np.linspace(0, 1, grid_size)
        grid_points = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if len(grid_points) < self.n_points:
                        grid_points.append([coords[i], coords[j], coords[k]])
                        
        return np.array(grid_points[:self.n_points])
    
    def _initialize_spherical_code(self) -> np.ndarray:
        """Initialize using known spherical code configurations"""
        # Known good configuration for 14 points on sphere
        spherical_points = np.array([
            [0.0000, 0.0000, 1.0000],
            [0.0000, 0.0000, -1.0000],
            [0.9343, 0.0000, 0.3564],
            [-0.9343, 0.0000, 0.3564],
            [0.0000, 0.9343, 0.3564],
            [0.0000, -0.9343, 0.3564],
            [0.0000, 0.9343, -0.3564],
            [0.0000, -0.9343, -0.3564],
            [0.9343, 0.0000, -0.3564],
            [-0.9343, 0.0000, -0.3564],
            [0.3564, 0.9343, 0.0000],
            [-0.3564, 0.9343, 0.0000],
            [0.3564, -0.9343, 0.0000],
            [-0.3564, -0.9343, 0.0000]
        ])
        
        # Normalize to unit sphere
        norms = np.linalg.norm(spherical_points, axis=1, keepdims=True)
        spherical_points = spherical_points / np.where(norms == 0, 1, norms)
        
        # Add small perturbations
        perturbation = np.random.normal(0, 0.01, spherical_points.shape)
        spherical_points = spherical_points + perturbation
        
        # Normalize again after perturbation
        norms = np.linalg.norm(spherical_points, axis=1, keepdims=True)
        spherical_points = spherical_points / np.where(norms == 0, 1, norms)
        
        return spherical_points
    
    def _initialize_kmeans(self) -> np.ndarray:
        """Initialize using KMeans clustering"""
        # Generate more samples for better clustering
        kmeans_points = np.random.rand(50, self.dimensions)
        kmeans = KMeans(n_clusters=self.n_points, random_state=self.seed, n_init=20)
        kmeans.fit(kmeans_points)
        return kmeans.cluster_centers_
    
    def _initialize_random(self) -> np.ndarray:
        """Initialize with random points"""
        return np.random.rand(self.n_points, self.dimensions)
    
    def _initialize_perturbed(self, base_points: np.ndarray) -> np.ndarray:
        """Apply perturbations to base points"""
        perturbed = base_points + np.random.normal(0, 0.03, base_points.shape)
        return np.clip(perturbed, 0, 1)
    
    def _evaluate_initialization(self, points: np.ndarray) -> float:
        """Fast evaluation of initialization quality"""
        if len(points) < 2:
            return 0
            
        distances = pdist(points)
        if len(distances) == 0:
            return 0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max > 1e-12:
            return d_min / d_max
        return 0
    
    def _generate_initial_strategies(self) -> List[Tuple[str, np.ndarray]]:
        """Generate multiple initialization strategies"""
        strategies = []
        
        # Strategy 1: Spherical Fibonacci points
        fib_points = self._initialize_spherical_fibonacci()
        fib_points = (fib_points + 1) / 2  # Normalize to [0,1]^3
        strategies.append(("fibonacci", fib_points.copy()))
        
        # Strategy 2: Spherical code points
        sph_code_points = self._initialize_spherical_code()
        sph_code_points = (sph_code_points + 1) / 2  # Normalize to [0,1]^3
        strategies.append(("spherical_code", sph_code_points.copy()))
        
        # Strategy 3: Cube grid points
        cube_points = self._initialize_cube_grid()
        strategies.append(("cube_grid", cube_points.copy()))
        
        # Strategy 4: Random points
        random_points = self._initialize_random()
        strategies.append(("random", random_points.copy()))
        
        # Strategy 5: Perturbed Fibonacci points
        perturbed_points = self._initialize_perturbed(fib_points)
        strategies.append(("perturbed_fibonacci", perturbed_points.copy()))
        
        # Strategy 6: KMeans points
        kmeans_points = self._initialize_kmeans()
        strategies.append(("kmeans", kmeans_points.copy()))
        
        # Strategy 7: Perturbed spherical code points
        alt_perturbed_points = self._initialize_perturbed(sph_code_points)
        strategies.append(("perturbed_spherical_code", alt_perturbed_points.copy()))
        
        return strategies
    
    def _adaptive_differential_evolution(self, objective_func: Callable, bounds: List[Tuple[float, float]], 
                                      maxiter: int = 300) -> Tuple[np.ndarray, float]:
        """Enhanced differential evolution with adaptive parameters"""
        # Starting parameters
        current_popsize = 20
        current_tol = 1e-6
        current_mutation = (0.5, 1.0)
        current_recombination = 0.7
        
        # Track performance for adaptive adjustments
        prev_obj_values = []
        improvement_threshold = 1e-8
        max_stagnation = 10
        
        try:
            for iteration in range(maxiter // 10):
                try:
                    result = differential_evolution(
                        objective_func,
                        bounds,
                        seed=self.seed + iteration,
                        maxiter=10,
                        popsize=current_popsize,
                        tol=current_tol,
                        mutation=current_mutation,
                        recombination=current_recombination,
                        disp=False
                    )
                    
                    # Track objective values
                    prev_obj_values.append(-result.fun)
                    if len(prev_obj_values) > 5:
                        prev_obj_values.pop(0)
                    
                    # Adaptive adjustments based on progress
                    if len(prev_obj_values) >= 2:
                        improvement = prev_obj_values[-1] - prev_obj_values[-2]
                        if improvement < improvement_threshold:
                            # Reduce population size for faster convergence if stagnating
                            current_popsize = max(10, current_popsize - 2)
                        else:
                            # Increase population size for better exploration
                            current_popsize = min(30, current_popsize + 1)
                            
                        # Tighten tolerance for better precision
                        current_tol = max(1e-12, current_tol * 0.9)
                        
                    # Early stopping if progress is minimal
                    if len(prev_obj_values) >= 5:
                        recent_improvements = [prev_obj_values[i+1] - prev_obj_values[i] 
                                             for i in range(len(prev_obj_values)-1)]
                        if all(abs(imp) < improvement_threshold for imp in recent_improvements):
                            break
                            
                    # Update parameters for next iteration
                    current_mutation = (0.5, 1.0)  # Reset to standard
                    current_recombination = 0.7
                    
                except Exception as e:
                    logger.warning(f"Differential evolution iteration {iteration} failed: {e}")
                    # Fallback to simpler parameters if something goes wrong
                    current_popsize = max(10, current_popsize - 5)
                    current_tol = max(1e-10, current_tol * 2)
                    if current_popsize < 10:
                        break
                    continue
                    
            return result.x, -result.fun
            
        except Exception as e:
            logger.error(f"Optimization failed completely: {e}")
            # Return a reasonable fallback
            return np.random.rand(self.n_points * self.dimensions), 0.0
    
    def _calculate_min_max_ratio(self, points: np.ndarray) -> Tuple[float, float, float]:
        """Calculate minimum, maximum, and ratio of distances"""
        if len(points) < 2:
            return 0.0, 0.0, 0.0
            
        distances = pdist(points)
        
        if len(distances) == 0:
            return 0.0, 0.0, 0.0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max <= 0:
            return 0.0, 0.0, 0.0
            
        ratio = d_min / d_max
        return d_min, d_max, ratio
    
    def _penalty_objective(self, x: np.ndarray, penalty_weight: float = 1e6) -> float:
        """Objective with penalty for boundary violations"""
        points = x.reshape(-1, 3)
        
        # Vectorized penalty calculation
        below_penalty = np.sum(np.maximum(0, -points)**2) * penalty_weight
        above_penalty = np.sum(np.maximum(0, points - 1)**2) * penalty_weight
        
        # Calculate main objective
        distances = pdist(points)
        
        if len(distances) == 0:
            return float('inf') + below_penalty + above_penalty
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max <= 0:
            return float('inf') + below_penalty + above_penalty
            
        # Return negative ratio plus penalty (to minimize)
        return -(d_min / d_max) + below_penalty + above_penalty
    
    def _local_refinement(self, points: np.ndarray, max_iter: int = 300) -> np.ndarray:
        """Apply local refinement to improve final solution"""
        def objective_local(x):
            points_local = x.reshape(-1, 3)
            distances = pdist(points_local)
            
            if len(distances) == 0:
                return float('inf')
                
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            if d_max > 1e-12:
                return -(d_min / d_max)  # Negative because we minimize
            else:
                return float('inf')
                
        try:
            x0_refine = points.flatten()
            bounds = [(0, 1)] * self.n_points * 3
            
            result_refine = minimize(
                objective_local,
                x0_refine,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-12, 'gtol': 1e-12},
                tol=1e-12
            )
            
            refined_points = result_refine.x.reshape(-1, 3)
            refined_points = np.clip(refined_points, 0, 1)
            return refined_points
        except Exception as e:
            logger.warning(f"Local refinement failed: {e}")
            return points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Initialize optimizer
    optimizer = PointDistributionOptimizer(n_points=14, dimensions=3, seed=42)
    
    # Phase 1: Generate all initialization strategies
    logger.info("Generating initialization strategies...")
    strategies = optimizer._generate_initial_strategies()
    
    # Phase 2: Evaluate all strategies and select the best
    logger.info("Evaluating initial strategies...")
    best_initialization = None
    best_ratio = -np.inf
    
    for name, points in strategies:
        ratio = optimizer._evaluate_initialization(points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_initialization = points.copy()
    
    if best_initialization is None:
        # Fallback to random initialization
        best_initialization = optimizer._initialize_random()
        best_ratio = optimizer._evaluate_initialization(best_initialization)
    
    logger.info(f"Best initial strategy ratio: {best_ratio}")
    
    # Phase 3: Setup bounds for optimization
    bounds = [(0, 1)] * 14 * 3
    x0 = best_initialization.flatten()
    
    # Phase 4: Run adaptive differential evolution optimization
    logger.info("Running adaptive differential evolution...")
    start_time = time.time()
    
    # Use the custom objective with penalty
    best_result_x, best_result_ratio = optimizer._adaptive_differential_evolution(
        optimizer._penalty_objective, 
        bounds, 
        maxiter=200
    )
    
    # Extract optimized points
    optimized_points = best_result_x.reshape(-1, 3)
    
    logger.info(f"Optimization completed in {time.time() - start_time:.2f}s")
    logger.info(f"Ratio after optimization: {best_result_ratio}")
    
    # Phase 5: Apply local refinement
    logger.info("Applying local refinement...")
    final_points = optimizer._local_refinement(optimized_points)
    
    # Final clipping to ensure bounds are respected
    final_points = np.clip(final_points, 0, 1)
    
    # Final validation
    final_min, final_max, final_ratio = optimizer._calculate_min_max_ratio(final_points)
    logger.info(f"Final validation - Min: {final_min:.6f}, Max: {final_max:.6f}, Ratio: {final_ratio:.6f}")
    
    return final_points

# EVOLVE-BLOCK-END