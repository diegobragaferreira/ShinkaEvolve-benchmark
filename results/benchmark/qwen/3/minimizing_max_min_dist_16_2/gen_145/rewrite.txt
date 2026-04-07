# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial import Voronoi
import time
from scipy.optimize import minimize
import warnings

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    class VoronoiLatticeOptimizer:
        def __init__(self, n_points=16, seed=42):
            self.n_points = n_points
            self.seed = seed
            np.random.seed(seed)
            
        def _construct_lattice_points(self):
            """Construct initial points using a mathematical lattice approach."""
            # Create points arranged in a pattern that maximizes uniformity
            # Using a modified hexagonal lattice with controlled distortions
            points = []
            
            # Generate points in a grid-like structure that can be adjusted for optimality
            rows = 4
            cols = 4
            
            # Create a base grid with alternating offsets to form a triangular lattice
            for i in range(rows):
                for j in range(cols):
                    if len(points) >= self.n_points:
                        break
                        
                    # Base positions with triangular offset
                    x_base = j * 0.25 + (i % 2) * 0.125
                    y_base = i * 0.25
                    
                    # Add systematic perturbations to break symmetries
                    # Use prime-based perturbations for uniqueness
                    prime_i = [2, 3, 5, 7][i % 4] if i < 4 else 11
                    prime_j = [13, 17, 19, 23][j % 4] if j < 4 else 29
                    
                    x_pert = np.sin(prime_i * 0.7) * 0.005
                    y_pert = np.cos(prime_j * 0.5) * 0.005
                    
                    x = x_base + x_pert
                    y = y_base + y_pert
                    
                    points.append([x, y])
            
            # Trim or pad to exact number of points
            points = points[:self.n_points]
            while len(points) < self.n_points:
                points.append([np.random.random(), np.random.random()])
            
            points = np.array(points)
            
            # Normalize to [0,1] bounds
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])
            
            if x_range > 0:
                points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
            if y_range > 0:
                points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
                
            # Scale to fit within unit square
            points = points * 0.9 + 0.05  # Center in [0.05, 0.95] range
            points = np.clip(points, 0, 1)
            
            return points
            
        def _compute_ratio(self, points):
            """Compute the ratio of minimum to maximum pairwise distances."""
            if len(points) < 2:
                return 0.0

            distances = pdist(points)
            d_min = np.min(distances)
            d_max = np.max(distances)

            if d_max == 0:
                return 0.0

            return d_min / d_max
            
        def _compute_voronoi_uniformity(self, points):
            """Compute uniformity of Voronoi cells (higher is better)."""
            try:
                vor = Voronoi(points)
                areas = []
                for region in vor.regions:
                    if not any(v == -1 for v in region) and len(region) >= 3:
                        # Calculate area of polygon using shoelace formula
                        polygon = [vor.vertices[i] for i in region]
                        if len(polygon) >= 3:
                            area = 0.5 * abs(sum(polygon[i][0] * polygon[(i+1)%len(polygon)][1] - 
                                               polygon[(i+1)%len(polygon)][0] * polygon[i][1] 
                                               for i in range(len(polygon))))
                            areas.append(area)
                
                if not areas:
                    return 0.0
                    
                mean_area = np.mean(areas)
                if mean_area == 0:
                    return 0.0
                    
                # Standard deviation of areas as measure of uniformity (lower = more uniform)
                std_area = np.std(areas)
                return 1.0 / (1.0 + std_area / mean_area) if mean_area > 0 else 0.0
                
            except:
                return 0.0
                
        def _objective_function(self, points_flat):
            """Combined objective function that balances ratio and Voronoi uniformity."""
            points = points_flat.reshape(-1, 2)
            
            # Compute primary objective: ratio of min/max distances
            ratio = self._compute_ratio(points)
            
            # Compute secondary objective: Voronoi uniformity
            uniformity = self._compute_voronoi_uniformity(points)
            
            # Combined objective (higher is better)
            # We weight uniformity to encourage balanced point distribution
            combined = ratio + 0.2 * uniformity
            
            # We minimize, so return negative
            return -combined
            
        def _project_to_bounds(self, points):
            """Ensure points remain within [0,1] bounds."""
            return np.clip(points, 0, 1)
            
        def _gradient_descent_refinement(self, points, max_iter=1000):
            """Refine solution using gradient-based optimization."""
            # Define the objective function for scipy optimize
            def obj_func(x_flat):
                return self._objective_function(x_flat)
                
            # Flatten points for optimization
            x0 = points.flatten()
            
            # Bounds for optimization
            bounds = [(0, 1) for _ in range(len(x0))]
            
            try:
                # Use L-BFGS-B for bound-constrained optimization
                result = minimize(
                    obj_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-10}
                )
                
                if result.success:
                    refined_points = result.x.reshape(-1, 2)
                    return self._project_to_bounds(refined_points)
                    
            except Exception as e:
                warnings.warn(f"Gradient descent failed: {str(e)}")
                
            return points
            
        def _local_search_refinement(self, points, max_iter=500):
            """Refine solution using local search with geometric considerations."""
            current_points = points.copy()
            current_ratio = self._compute_ratio(current_points)
            best_points = current_points.copy()
            best_ratio = current_ratio
            
            # Use a more sophisticated local search
            for iteration in range(max_iter):
                # Try small adjustments to points
                for i in range(len(current_points)):
                    # Save current point
                    original_point = current_points[i].copy()
                    
                    # Try small random perturbations
                    perturbation = np.random.normal(0, 0.001, 2)
                    current_points[i] += perturbation
                    current_points[i] = self._project_to_bounds(current_points[i])
                    
                    # Evaluate new configuration
                    new_ratio = self._compute_ratio(current_points)
                    
                    if new_ratio > current_ratio:
                        current_ratio = new_ratio
                        if new_ratio > best_ratio:
                            best_ratio = new_ratio
                            best_points = current_points.copy()
                    else:
                        # Revert
                        current_points[i] = original_point
                        
                # Occasionally try larger perturbations
                if iteration % 50 == 0 and iteration > 0:
                    # Try to improve by moving a few points more aggressively
                    move_indices = np.random.choice(len(current_points), 
                                                  size=min(3, len(current_points)), 
                                                  replace=False)
                    for idx in move_indices:
                        perturbation = np.random.normal(0, 0.005, 2)
                        current_points[idx] += perturbation
                        current_points[idx] = self._project_to_bounds(current_points[idx])
                        
                    new_ratio = self._compute_ratio(current_points)
                    if new_ratio > current_ratio:
                        current_ratio = new_ratio
                        if new_ratio > best_ratio:
                            best_ratio = new_ratio
                            best_points = current_points.copy()
                            
            return best_points
            
        def optimize(self):
            """Main optimization routine."""
            # Step 1: Generate initial points using lattice construction
            initial_points = self._construct_lattice_points()
            
            # Step 2: Refine using gradient-based optimization
            refined_points = self._gradient_descent_refinement(initial_points, max_iter=500)
            
            # Step 3: Further refine with local search
            final_points = self._local_search_refinement(refined_points, max_iter=300)
            
            # Step 4: Additional local refinement for fine-tuning
            final_points = self._local_search_refinement(final_points, max_iter=200)
            
            return final_points
    
    # Create optimizer and run optimization
    optimizer = VoronoiLatticeOptimizer(n_points=16, seed=42)
    points = optimizer.optimize()
    
    return points

# EVOLVE-BLOCK-END