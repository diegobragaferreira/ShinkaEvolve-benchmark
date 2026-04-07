# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist
from scipy.spatial import Voronoi
import time
from typing import Tuple

class GeometricPackingOptimizer:
    def __init__(self, num_points: int = 16, dimensions: int = 2):
        self.num_points = num_points
        self.dimensions = dimensions
        self.benchmark_threshold = 1 / np.sqrt(12.889266112)
        self.best_ratio = -np.inf
        self.best_points = None
        
    def _compute_min_max_ratio(self, points) -> float:
        """Efficiently compute the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0
            
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max <= 0:
            return 0.0
            
        return d_min / d_max
    
    def _hexagonal_init(self) -> np.ndarray:
        """Initialize points in hexagonal packing pattern for better spacing."""
        points = []
        rows, cols = 4, 4
        
        for i in range(rows):
            for j in range(cols):
                # Offset every other row for hexagonal packing
                x_offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + x_offset) * 0.25 + 0.125
                y = i * 0.25 + 0.125
                points.append([x, y])
                
        return np.array(points)
    
    def _voronoi_cluster_detection(self, points: np.ndarray) -> np.ndarray:
        """Detect clustered regions using Voronoi diagrams and mark them for relaxation."""
        try:
            vor = Voronoi(points)
            # Simple heuristic: find regions with many nearby points
            # This helps identify where clustering occurs
            return vor.points
        except:
            # Fallback if Voronoi fails
            return points
    
    def _distance_weighted_gradient_step(self, points: np.ndarray, prev_ratio: float) -> np.ndarray:
        """Apply gradient descent with step sizes weighted by current distance distribution."""
        # Compute pairwise distances
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        
        # Calculate gradient contribution from each point pair
        grad = np.zeros_like(points)
        
        for i in range(len(points)):
            for j in range(len(points)):
                if i != j:
                    diff = points[i] - points[j]
                    dist = np.linalg.norm(diff)
                    if dist > 0:
                        # Weight gradient by inverse of distance (closer points have stronger influence)
                        weight = 1.0 / (dist * dist + 1e-10)
                        grad[i] -= weight * diff / dist
                        grad[j] += weight * diff / dist
        
        # Adaptive step sizing based on ratio quality
        if prev_ratio < 0.1:
            step_size = 0.02
        elif prev_ratio < 0.2:
            step_size = 0.01
        else:
            step_size = 0.005
            
        # Apply gradient with bounds checking
        new_points = points - step_size * grad
        new_points = np.clip(new_points, 0, 1)
        
        return new_points
    
    def _centroid_projection(self, points: np.ndarray) -> np.ndarray:
        """Project points to break symmetry using centroid-based constraints."""
        # Calculate centroid
        centroid = np.mean(points, axis=0)
        
        # Move points away from centroid to break symmetric configurations
        # Only move points that are close to centroid
        moved_points = points.copy()
        for i in range(len(points)):
            dist_to_centroid = np.linalg.norm(points[i] - centroid)
            if dist_to_centroid < 0.3:  # If point is near centroid
                # Push away from centroid
                direction = points[i] - centroid
                if np.linalg.norm(direction) > 1e-10:
                    moved_points[i] = points[i] + 0.02 * direction / np.linalg.norm(direction)
        
        return np.clip(moved_points, 0, 1)
    
    def _multi_scale_refinement(self, points: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Refine solution using multi-scale approach."""
        current_points = points.copy()
        current_ratio = self._compute_min_max_ratio(current_points)
        
        # Coarse refinement (larger steps)
        for i in range(30):
            if i % 10 == 0:  # Update ratio periodically
                current_ratio = self._compute_min_max_ratio(current_points)
            current_points = self._distance_weighted_gradient_step(current_points, current_ratio)
            current_points = self._centroid_projection(current_points)
        
        # Medium refinement (medium steps)
        for i in range(30):
            if i % 10 == 0:
                current_ratio = self._compute_min_max_ratio(current_points)
            current_points = self._distance_weighted_gradient_step(current_points, current_ratio)
            current_points = self._centroid_projection(current_points)
        
        # Fine refinement (small steps)
        for i in range(40):
            if i % 10 == 0:
                current_ratio = self._compute_min_max_ratio(current_points)
            current_points = self._distance_weighted_gradient_step(current_points, current_ratio)
            current_points = self._centroid_projection(current_points)
            
        return current_points
    
    def _smart_initialization(self) -> np.ndarray:
        """Create smart initial configuration using hexagonal pattern with enhancements."""
        # Start with hexagonal packing
        points = self._hexagonal_init()
        
        # Add slight random perturbations to break some symmetries
        np.random.seed(42)
        noise = np.random.normal(0, 0.01, points.shape)
        points += noise
        points = np.clip(points, 0, 1)
        
        # Apply iterative local refinement to improve initial quality
        refined_points = self._multi_scale_refinement(points, max_iter=50)
        return refined_points
    
    def _hybrid_local_optimization(self, points: np.ndarray) -> np.ndarray:
        """Combine multiple local optimization strategies."""
        # First, do a coarse optimization using L-BFGS-B
        def objective(x):
            pts = x.reshape(self.num_points, self.dimensions)
            ratio = self._compute_min_max_ratio(pts)
            return -ratio  # Negative because we minimize
            
        bounds = [(0, 1) for _ in range(self.num_points * self.dimensions)]
        
        try:
            result1 = minimize(objective, points.flatten(), method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if result1.success:
                refined_points = result1.x.reshape(self.num_points, self.dimensions)
            else:
                refined_points = points.copy()
        except:
            refined_points = points.copy()
            
        # Then apply multi-scale refinement
        final_points = self._multi_scale_refinement(refined_points, max_iter=100)
        
        return final_points
    
    def optimize(self) -> np.ndarray:
        """Main optimization routine with geometric insights."""
        # Set fixed seed for reproducibility
        np.random.seed(42)
        
        # Strategy 1: Smart initialization with hexagonal pattern
        initial_points = self._smart_initialization()
        initial_ratio = self._compute_min_max_ratio(initial_points)
        
        # Store best so far
        if initial_ratio > self.best_ratio:
            self.best_ratio = initial_ratio
            self.best_points = initial_points.copy()
            
        # Strategy 2: Multiple restarts with different initializations
        restart_strategies = [
            lambda: self._smart_initialization(),
            lambda: self._smart_initialization() + np.random.normal(0, 0.01, (16, 2)),
            lambda: self._smart_initialization() + np.random.uniform(-0.02, 0.02, (16, 2))
        ]
        
        for i, strategy in enumerate(restart_strategies):
            if i > 0:  # Don't reseed for first strategy since it's already seeded
                np.random.seed(100 + i)
                
            try:
                points = strategy()
                # Apply hybrid optimization
                optimized_points = self._hybrid_local_optimization(points)
                ratio = self._compute_min_max_ratio(optimized_points)
                
                if ratio > self.best_ratio:
                    self.best_ratio = ratio
                    self.best_points = optimized_points.copy()
                    
                # Early exit if we've beaten the benchmark
                if ratio >= self.benchmark_threshold:
                    break
                    
            except Exception:
                continue
        
        # Strategy 3: Final optimization with enhanced refinement
        if self.best_points is not None:
            # Apply final multi-scale refinement with more iterations
            final_refinement = self._multi_scale_refinement(self.best_points, max_iter=200)
            final_ratio = self._compute_min_max_ratio(final_refinement)
            
            if final_ratio > self.best_ratio:
                self.best_ratio = final_ratio
                self.best_points = final_refinement.copy()
        
        # Fallback if nothing worked
        if self.best_points is None:
            fallback = self._hexagonal_init()
            self.best_points = self._hybrid_local_optimization(fallback)
            self.best_ratio = self._compute_min_max_ratio(self.best_points)
            
        return self.best_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    # Initialize the geometric packing optimizer
    optimizer = GeometricPackingOptimizer(16, 2)
    
    # Perform optimization
    points = optimizer.optimize()
    
    return points

# EVOLVE-BLOCK-END