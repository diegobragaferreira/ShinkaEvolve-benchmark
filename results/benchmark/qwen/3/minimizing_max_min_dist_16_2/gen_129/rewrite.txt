# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import time

class SphereSamplingEvolutionOptimizer:
    """Novel optimizer using spherical sampling and adaptive clustering refinement."""
    
    def __init__(self, n_points=16, dimensions=2, seed=42):
        self.n_points = n_points
        self.dimensions = dimensions
        self.seed = seed
        np.random.seed(seed)

    def _compute_ratio(self, points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0
        
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max == 0:
            return 0
            
        return d_min / d_max

    def _generate_sphere_points(self):
        """Generate points using Fibonacci spiral on 3D sphere, then project to 2D."""
        points_3d = np.zeros((self.n_points, 3))
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        for i in range(self.n_points):
            # Distribute points evenly on sphere using Fibonacci spiral
            z = 1 - (i / (self.n_points - 1)) * 2  # z coordinate from -1 to 1
            radius = np.sqrt(1 - z*z)
            theta = np.arccos(z)
            phi = (i * golden_ratio) % (2 * np.pi)
            
            # Convert to Cartesian coordinates on unit sphere
            x = radius * np.cos(phi)
            y = radius * np.sin(phi)
            
            points_3d[i] = [x, y, z]
        
        # Project 3D points to 2D using stereographic projection
        points_2d = np.zeros((self.n_points, 2))
        for i in range(self.n_points):
            x, y, z = points_3d[i]
            # Stereographic projection from south pole
            denom = 1 - z
            if abs(denom) < 1e-10:
                points_2d[i] = [0, 0]
            else:
                points_2d[i] = [x / denom, y / denom]
        
        # Normalize to [0,1] x [0,1] range
        x_range = np.max(points_2d[:, 0]) - np.min(points_2d[:, 0])
        y_range = np.max(points_2d[:, 1]) - np.min(points_2d[:, 1])
        
        if x_range > 0:
            points_2d[:, 0] = (points_2d[:, 0] - np.min(points_2d[:, 0])) / x_range
        if y_range > 0:
            points_2d[:, 1] = (points_2d[:, 1] - np.min(points_2d[:, 1])) / y_range
            
        # Add small perturbations to break symmetries
        points_2d += np.random.normal(0, 0.005, points_2d.shape)
        points_2d = np.clip(points_2d, 0, 1)
        
        return points_2d

    def _adaptive_clustering_refinement(self, points, max_iter=500):
        """Refine points using adaptive clustering-based approach."""
        current_points = points.copy()
        current_ratio = self._compute_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Precompute distance matrix for efficiency
        dist_matrix = squareform(pdist(current_points))
        
        # Iterative refinement
        for iteration in range(max_iter):
            # Identify problematic points - those with small minimum distances
            min_distances = np.min(dist_matrix[np.triu_indices_from(dist_matrix, k=1)], axis=1)
            
            # Find cluster centers (points with smallest average distances to neighbors)
            avg_distances = np.mean(dist_matrix, axis=1)
            cluster_centers = np.argsort(avg_distances)[:min(5, self.n_points//2)]
            
            # Refine points in clusters
            for center_idx in cluster_centers:
                # Collect nearby points within a certain radius
                radius = 0.2  # Adjust based on point density
                nearby_mask = dist_matrix[center_idx] < radius
                nearby_indices = np.where(nearby_mask)[0]
                
                if len(nearby_indices) > 1:  # Need at least one other point
                    # Compute centroid of nearby points
                    centroid = np.mean(current_points[nearby_indices], axis=0)
                    
                    # Move center point towards centroid
                    movement_direction = centroid - current_points[center_idx]
                    movement_magnitude = np.linalg.norm(movement_direction)
                    
                    if movement_magnitude > 0:
                        # Apply adaptive movement based on density
                        density_factor = len(nearby_indices) / len(current_points)
                        step_size = 0.02 * (1 - iteration/max_iter) * density_factor
                        
                        new_pos = current_points[center_idx] + step_size * movement_direction
                        new_pos = np.clip(new_pos, 0, 1)  # Ensure bounds
                        
                        # Check if this improves the ratio
                        test_points = current_points.copy()
                        test_points[center_idx] = new_pos
                        
                        test_ratio = self._compute_ratio(test_points)
                        if test_ratio > current_ratio:
                            current_points[center_idx] = new_pos
                            current_ratio = test_ratio
                            
                            # Update best if improved
                            if current_ratio > best_ratio:
                                best_ratio = current_ratio
                                best_points = current_points.copy()
            
            # Recalculate distance matrix for next iteration
            dist_matrix = squareform(pdist(current_points))
            
            # Early stopping criteria
            if iteration > 10 and abs(current_ratio - best_ratio) < 1e-6:
                break
                
        return best_points

    def _local_optimization_refinement(self, points, max_iter=200):
        """Apply local optimization to fine-tune the solution."""
        def objective(x_flat):
            points = x_flat.reshape(-1, self.dimensions)
            points = np.clip(points, 0, 1)
            return -self._compute_ratio(points)  # Minimize negative ratio
            
        # Flatten points for optimization
        x0 = points.flatten()
        bounds = [(0, 1) for _ in range(len(x0))]
        
        try:
            # Use L-BFGS-B for smooth local optimization
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, self.dimensions)
                refined_points = np.clip(refined_points, 0, 1)
                return refined_points
        except:
            pass
            
        return points

    def optimize(self):
        """Main optimization routine using sphere sampling and clustering refinement."""
        best_solution = None
        best_ratio = -np.inf
        
        # Strategy 1: Sphere sampling initialization
        try:
            sphere_points = self._generate_sphere_points()
            refined_points = self._adaptive_clustering_refinement(sphere_points)
            final_points = self._local_optimization_refinement(refined_points)
            
            final_ratio = self._compute_ratio(final_points)
            if final_ratio > best_ratio:
                best_ratio = final_ratio
                best_solution = final_points.copy()
                
        except Exception as e:
            pass
        
        # Strategy 2: Random initialization with clustering
        try:
            np.random.seed(self.seed)
            random_points = np.random.rand(self.n_points, self.dimensions)
            refined_points = self._adaptive_clustering_refinement(random_points)
            final_points = self._local_optimization_refinement(refined_points)
            
            final_ratio = self._compute_ratio(final_points)
            if final_ratio > best_ratio:
                best_ratio = final_ratio
                best_solution = final_points.copy()
                
        except Exception as e:
            pass
        
        # Strategy 3: Perturbed regular grid with clustering
        try:
            # Create regular grid
            n_per_side = int(np.ceil(np.sqrt(self.n_points)))
            x = np.linspace(0.1, 0.9, n_per_side)
            y = np.linspace(0.1, 0.9, n_per_side)
            xx, yy = np.meshgrid(x, y)
            points = np.column_stack([xx.ravel(), yy.ravel()])[:self.n_points]
            
            # Add perturbations
            points += np.random.normal(0, 0.01, points.shape)
            points = np.clip(points, 0, 1)
            
            refined_points = self._adaptive_clustering_refinement(points)
            final_points = self._local_optimization_refinement(refined_points)
            
            final_ratio = self._compute_ratio(final_points)
            if final_ratio > best_ratio:
                best_ratio = final_ratio
                best_solution = final_points.copy()
                
        except Exception as e:
            pass
        
        # Return best solution found
        if best_solution is None:
            # Fallback to sphere sampling
            return self._generate_sphere_points()
            
        return best_solution

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = SphereSamplingEvolutionOptimizer(n_points=16, dimensions=2, seed=42)
    points = optimizer.optimize()
    return points

# EVOLVE-BLOCK-END