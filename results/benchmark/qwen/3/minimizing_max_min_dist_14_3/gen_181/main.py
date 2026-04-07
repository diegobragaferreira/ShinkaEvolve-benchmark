# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class PointInitializer:
    """Handles various initialization strategies for point placement."""
    
    @staticmethod
    def fibonacci_sphere(n_points):
        """Initialize points on a unit sphere using Fibonacci spiral method."""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2

        for i in range(n_points):
            phi = np.arccos(1 - 2*i/(n_points-1))
            theta = 2 * np.pi * i / golden_ratio

            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

            points.append([x, y, z])

        return np.array(points)
    
    @staticmethod
    def cube_grid(n_points):
        """Initialize points in a 3D cube grid."""
        grid_size = int(np.ceil(n_points**(1/3)))
        coords = np.linspace(0, 1, grid_size)
        grid_points = []

        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if len(grid_points) < n_points:
                        grid_points.append([coords[i], coords[j], coords[k]])

        return np.array(grid_points[:n_points])
    
    @staticmethod
    def random_points(n_points, dimensions=3):
        """Initialize random points in 3D space."""
        return np.random.rand(n_points, dimensions)
    
    @staticmethod
    def perturb_points(points, magnitude=0.05):
        """Add small random perturbations to break symmetry."""
        noise = np.random.normal(0, magnitude, points.shape)
        perturbed = points + noise
        return np.clip(perturbed, 0, 1)

class DistanceAnalyzer:
    """Analyzes distance properties of point configurations."""
    
    @staticmethod
    def compute_distance_ratio(points):
        """Compute the minimum/maximum distance ratio."""
        if len(points) < 2:
            return 0
            
        distances = pdist(points)
        if len(distances) == 0:
            return 0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max <= 1e-12:
            return 0
            
        return d_min / d_max
    
    @staticmethod
    def voronoi_uniformity_score(points):
        """Evaluate how uniformly distributed points are using distance variance."""
        try:
            distances = pdist(points)
            if len(distances) == 0:
                return 0
            # High uniformity means distances vary less - we'll use inverse variance
            return 1.0 / (np.var(distances) + 1e-12)
        except:
            return 0

class OptimizationPipeline:
    """Manages the complete optimization workflow."""
    
    def __init__(self, n_points=14, dimensions=3, seed=42):
        self.n_points = n_points
        self.dimensions = dimensions
        self.seed = seed
        np.random.seed(seed)
        
    def _initialize_strategies(self):
        """Generate multiple initialization strategies."""
        strategies = {}
        
        # Strategy 1: Spherical Fibonacci points
        fib_points = PointInitializer.fibonacci_sphere(self.n_points)
        fib_points = (fib_points + 1) / 2  # Normalize to [0,1]^3
        strategies["fibonacci"] = fib_points
        
        # Strategy 2: Cube grid points
        cube_points = PointInitializer.cube_grid(self.n_points)
        strategies["cube_grid"] = cube_points
        
        # Strategy 3: Random points
        random_points = PointInitializer.random_points(self.n_points)
        strategies["random"] = random_points
        
        # Strategy 4: Perturbed spherical points
        perturbed_points = PointInitializer.perturb_points(fib_points, 0.05)
        strategies["perturbed"] = perturbed_points
        
        # Strategy 5: Optimized version of spherical points
        optimized_fib_points = PointInitializer.perturb_points(fib_points, 0.02)
        strategies["optimized_fib"] = optimized_fib_points
        
        # Strategy 6: K-means optimized spherical points
        try:
            temp_points = PointInitializer.fibonacci_sphere(30)
            temp_points = (temp_points + 1) / 2
            kmeans = KMeans(n_clusters=self.n_points, random_state=self.seed, n_init=10)
            kmeans.fit(temp_points)
            kmeans_points = kmeans.cluster_centers_
            strategies["kmeans_optimized"] = kmeans_points
        except:
            strategies["kmeans_optimized"] = PointInitializer.perturb_points(fib_points, 0.03)

        # Strategy 7: Iterative spreading approach
        np.random.seed(self.seed)
        uniform_init = PointInitializer.random_points(self.n_points)
        # Simple iterative spreading
        for _ in range(10):
            distances = pdist(uniform_init)
            if len(distances) > 0:
                for i in range(self.n_points):
                    dist_row = distances[i*(self.n_points-1):(i+1)*(self.n_points-1)]
                    if len(dist_row) > 0:
                        closest_idx = np.argmin(dist_row)
                        if closest_idx < i:
                            neighbor = uniform_init[closest_idx]
                        else:
                            neighbor = uniform_init[closest_idx + 1]
                        direction = uniform_init[i] - neighbor
                        if np.linalg.norm(direction) > 1e-12:
                            uniform_init[i] += 0.01 * direction / np.linalg.norm(direction)
                uniform_init = np.clip(uniform_init, 0, 1)
        strategies["iterative_spread"] = uniform_init
        
        return strategies
    
    def _select_best_initialization(self, strategies):
        """Select the best initialization based on combined evaluation."""
        best_initialization = None
        best_score = -np.inf

        for name, points in strategies.items():
            ratio = DistanceAnalyzer.compute_distance_ratio(points)
            uniformity = DistanceAnalyzer.voronoi_uniformity_score(points)
            # Combined score: weighted sum of ratio and uniformity
            combined_score = ratio + 0.1 * uniformity  # Weight uniformity less heavily
            if combined_score > best_score:
                best_score = combined_score
                best_initialization = points.copy()

        return best_initialization
    
    def _objective_function(self, x):
        """Core objective function that returns negative ratio to minimize."""
        points = x.reshape(-1, 3)
        
        # Ensure points are within bounds [0,1]^3
        points = np.clip(points, 0, 1)
        
        # Compute distances
        distances = pdist(points)
        
        if len(distances) == 0:
            return -np.inf
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max <= 1e-12:
            return -np.inf
            
        # Return negative because we want to maximize the ratio
        return -(d_min / d_max)
    
    def _penalty_objective(self, x, penalty_weight=1e6):
        """Objective with penalty for boundary violations."""
        points = x.reshape(-1, 3)
        
        # Apply penalty for points outside bounds using vectorized operations
        penalty = 0
        penalty += np.sum(np.maximum(0, -points)**2) * penalty_weight  # Below 0
        penalty += np.sum(np.maximum(0, points - 1)**2) * penalty_weight  # Above 1
        
        # Original objective
        original_obj = self._objective_function(x)
        
        return original_obj + penalty
    
    def _adaptive_differential_evolution(self, objective_func, bounds, maxiter=300):
        """Enhanced differential evolution with adaptive parameters and early stopping."""
        current_popsize = 20
        prev_best = -np.inf
        stagnation_count = 0
        improvement_threshold = 1e-8
        min_improvement = 1e-12
        recent_improvements = []
        
        for iteration in range(maxiter // 10):
            # Adjust population size based on convergence
            if stagnation_count > 3 and current_popsize < 30:
                current_popsize = min(current_popsize + 5, 30)
            
            try:
                result = differential_evolution(
                    objective_func,
                    bounds,
                    seed=self.seed + iteration,
                    maxiter=10,
                    popsize=current_popsize,
                    tol=1e-12,
                    mutation=(0.5, 1.0),
                    recombination=0.9,
                    disp=False
                )
            except Exception:
                # Fallback to smaller population if needed
                try:
                    result = differential_evolution(
                        objective_func,
                        bounds,
                        seed=self.seed + iteration,
                        maxiter=10,
                        popsize=max(5, current_popsize - 5),
                        tol=1e-12,
                        mutation=(0.5, 1.0),
                        recombination=0.9,
                        disp=False
                    )
                except Exception:
                    # Last resort - use basic differential evolution
                    result = differential_evolution(
                        objective_func,
                        bounds,
                        seed=self.seed + iteration,
                        maxiter=10,
                        popsize=10,
                        tol=1e-12,
                        mutation=(0.5, 1.0),
                        recombination=0.7,
                        disp=False
                    )
            
            # Check for improvement
            current_best = -result.fun
            improvement = current_best - prev_best
            
            recent_improvements.append(improvement)
            if len(recent_improvements) > 5:
                recent_improvements.pop(0)
            
            # Early stopping if improvement is minimal
            if len(recent_improvements) == 5 and all(abs(impr) < min_improvement for impr in recent_improvements):
                break
                
            if improvement > improvement_threshold:
                stagnation_count = 0
            else:
                stagnation_count += 1
                
            prev_best = current_best
            
        return result
    
    def _local_refinement(self, points):
        """Apply local refinement to improve final solution."""
        def objective_local(x):
            points_local = x.reshape(-1, 3)
            distances = pdist(points_local)
            
            if len(distances) == 0:
                return -np.inf
                
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            if d_max > 1e-12:
                return -(d_min / d_max)
            else:
                return -np.inf
                
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
        except Exception:
            return points
    
    def optimize(self):
        """Execute the complete optimization pipeline."""
        # Generate and evaluate initialization strategies
        strategies = self._initialize_strategies()
        best_initialization = self._select_best_initialization(strategies)
        
        # Prepare for optimization
        x0 = best_initialization.flatten()
        bounds = [(0, 1)] * self.n_points * 3

        # Run adaptive differential evolution optimization
        best_result = None
        best_ratio = -np.inf

        # Try 3 different random seeds for better exploration
        for seed_val in [42, 123, 456]:
            np.random.seed(seed_val)
            
            # Use adaptive differential evolution
            result = self._adaptive_differential_evolution(
                self._penalty_objective, 
                bounds, 
                maxiter=200
            )
            
            # Check if this result is better
            if -result.fun > best_ratio:
                best_ratio = -result.fun
                best_result = result

        # Extract optimized points
        optimized_points = best_result.x.reshape(-1, 3)

        # Apply local refinement
        final_points = self._local_refinement(optimized_points)

        # Final clipping to ensure bounds are respected
        final_points = np.clip(final_points, 0, 1)

        return final_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    optimizer = OptimizationPipeline(n_points=14, dimensions=3, seed=42)
    return optimizer.optimize()

# EVOLVE-BLOCK-END