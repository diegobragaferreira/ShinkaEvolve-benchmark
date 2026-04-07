# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

class PointOptimizer3D:
    def __init__(self, n_points=14, dimension=3, max_time_seconds=360):
        self.n_points = n_points
        self.dimension = dimension
        self.max_time_seconds = max_time_seconds
        self.best_score = -np.inf
        self.best_points = None
        
    def _compute_distances(self, points):
        """Compute pairwise distances efficiently"""
        return pdist(points)
    
    def _calculate_ratio(self, points):
        """Calculate min/max distance ratio"""
        distances = self._compute_distances(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0.0
            
        return min_dist / max_dist
    
    def _objective_function(self, x):
        """Objective function returning negative ratio for maximization"""
        points = x.reshape((self.n_points, self.dimension))
        ratio = self._calculate_ratio(points)
        return -ratio
    
    def _penalty_objective(self, x, penalty_weight=1e6):
        """Objective with penalty for boundary violations"""
        points = x.reshape((self.n_points, self.dimension))
        
        # Calculate base objective
        ratio = self._calculate_ratio(points)
        base_obj = -ratio
        
        # Add penalty for points outside [0,1]^3 bounds
        penalty = 0
        for i in range(self.n_points):
            for j in range(self.dimension):
                coord = points[i, j]
                if coord < 0:
                    penalty += penalty_weight * (0 - coord) ** 2
                elif coord > 1:
                    penalty += penalty_weight * (coord - 1) ** 2
        
        return base_obj + penalty
    
    def _fibonacci_sphere_sampling(self, n):
        """Generate points on a unit sphere using Fibonacci spiral method"""
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle
        
        for i in range(n):
            y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def _initialize_spherical_points(self):
        """Initialize points using Fibonacci sphere sampling"""
        # Generate points on a unit sphere
        points = self._fibonacci_sphere_sampling(self.n_points)
        
        # Add small random perturbations to escape local minima
        np.random.seed(42)
        perturbations = np.random.normal(0, 0.01, points.shape)
        points += perturbations
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Scale appropriately to have reasonable distances
        points *= 0.8
        
        # Transform to [0,1]^3 space
        # Map from [-1,1]^3 to [0,1]^3
        points = (points + 1) / 2
        
        return points
    
    def _initialize_random_points(self):
        """Alternative random initialization"""
        np.random.seed(42)
        points = np.random.rand(self.n_points, self.dimension)
        return points
    
    def _initialize_points(self):
        """Multi-strategy initialization"""
        # Try spherical initialization first
        spherical_points = self._initialize_spherical_points()
        random_points = self._initialize_random_points()
        
        # Evaluate both initializations
        spherical_ratio = self._calculate_ratio(spherical_points)
        random_ratio = self._calculate_ratio(random_points)
        
        # Choose the better initialization
        if spherical_ratio > random_ratio:
            return spherical_points
        else:
            return random_points
    
    def _adaptive_differential_evolution(self, initial_points):
        """Perform differential evolution with adaptive population sizing"""
        bounds = [(0, 1)] * self.n_points * self.dimension
        
        # Start with smaller population and adapt
        popsize = 10
        maxiter = 300
        
        try:
            result = differential_evolution(
                self._penalty_objective,
                bounds,
                seed=42,
                maxiter=maxiter,
                popsize=popsize,
                mutation=(0.5, 1.0),
                recombination=0.7,
                tol=1e-8,
                callback=None
            )
            
            return result.x.reshape((self.n_points, self.dimension))
            
        except Exception:
            # Fallback to basic optimization
            return initial_points
    
    def _local_refinement(self, points):
        """Apply local optimization refinement"""
        try:
            # Use L-BFGS-B for fine-tuning
            result = minimize(
                self._objective_function,
                points.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1)] * self.n_points * self.dimension,
                options={'ftol': 1e-9, 'gtol': 1e-9}
            )
            
            refined_points = result.x.reshape((self.n_points, self.dimension))
            return refined_points
            
        except Exception:
            return points
    
    def _validate_and_correct_bounds(self, points):
        """Ensure all points are within [0,1]^3 bounds"""
        corrected_points = np.clip(points, 0, 1)
        return corrected_points
    
    def _monitor_progress(self, points, iteration):
        """Monitor optimization progress"""
        ratio = self._calculate_ratio(points)
        if ratio > self.best_score:
            self.best_score = ratio
            self.best_points = points.copy()
        return ratio
    
    def optimize(self):
        """Main optimization routine with hierarchical approach"""
        # Phase 1: Initialization
        initial_points = self._initialize_points()
        
        # Phase 2: Global optimization
        global_optimized = self._adaptive_differential_evolution(initial_points)
        
        # Phase 3: Local refinement
        local_optimized = self._local_refinement(global_optimized)
        
        # Final validation
        final_points = self._validate_and_correct_bounds(local_optimized)
        
        # Final check of quality
        final_ratio = self._calculate_ratio(final_points)
        
        # Store best solution found
        if final_ratio > self.best_score:
            self.best_score = final_ratio
            self.best_points = final_points.copy()
        
        return self.best_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    optimizer = PointOptimizer3D(n_points=14, dimension=3)
    return optimizer.optimize()

# EVOLVE-BLOCK-END
