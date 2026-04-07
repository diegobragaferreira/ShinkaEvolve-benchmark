# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
import time
from typing import Tuple, Optional

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses sphere packing inspired evolutionary approach.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    class SpherePackingOptimizer:
        def __init__(self, n_points: int = 16, max_iterations: int = 2000):
            self.n_points = n_points
            self.max_iterations = max_iterations
            self.best_ratio = -np.inf
            self.best_points = None
            
        def compute_min_max_ratio(self, points: np.ndarray) -> float:
            """Compute the ratio of minimum to maximum pairwise distances"""
            if len(points) < 2:
                return 0.0
            distances = cdist(points, points)
            # Set diagonal to large value to ignore self-distances
            np.fill_diagonal(distances, np.inf)
            max_dist = np.max(distances)
            if max_dist == 0:
                return 0.0
            min_dist = np.min(distances)
            return min_dist / max_dist
            
        def compute_voronoi_constraints(self, points: np.ndarray) -> float:
            """Compute penalty based on Voronoi cell sizes to encourage uniform distribution"""
            try:
                vor = Voronoi(points)
                # Compute areas of Voronoi cells
                areas = []
                for region in vor.point_regions:
                    if -1 not in region:  # Skip infinite regions
                        polygon = [vor.vertices[i] for i in region]
                        if len(polygon) >= 3:
                            # Simple area calculation (this is approximate but effective)
                            area = self._polygon_area(polygon)
                            areas.append(area)
                
                if areas:
                    # Penalize variance in cell areas
                    mean_area = np.mean(areas)
                    if mean_area > 0:
                        variance = np.var(areas)
                        return -variance / (mean_area * mean_area)  # Negative for minimization
            except:
                pass
            return 0.0
            
        def _polygon_area(self, vertices) -> float:
            """Calculate polygon area using shoelace formula"""
            if len(vertices) < 3:
                return 0.0
            x = [v[0] for v in vertices]
            y = [v[1] for v in vertices]
            area = 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(len(x)-1)) + x[-1] * y[0] - x[0] * y[-1])
            return area
            
        def initialize_grid_points(self) -> np.ndarray:
            """Initialize points on a structured grid with adaptive spacing"""
            # Create a 4x4 grid with better packing
            points = []
            spacing = 0.8
            row_spacing = spacing * np.sqrt(3) / 2.0
            col_spacing = spacing
            
            for i in range(4):
                for j in range(4):
                    if len(points) < self.n_points:
                        x = j * col_spacing + (i % 2) * col_spacing / 2.0
                        y = i * row_spacing
                        points.append([x, y])
            
            points = np.array(points[:self.n_points])
            
            # Normalize to [0,1] range
            if len(points) > 0:
                min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
                min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
                if max_x > min_x and max_y > min_y:
                    scale_x = 1.0 / (max_x - min_x)
                    scale_y = 1.0 / (max_y - min_y)
                    scale = min(scale_x, scale_y, 1.0)
                    points[:, 0] = (points[:, 0] - min_x) * scale
                    points[:, 1] = (points[:, 1] - min_y) * scale
                    
                # Center in unit square
                center_shift = 0.5 - np.mean(points, axis=0)
                points = points + center_shift
                
            # Ensure bounds
            points = np.clip(points, 0, 1)
            
            # Apply symmetry breaking with mathematical rotations and perturbations
            center = np.mean(points, axis=0)
            np.random.seed(42)
            
            # Apply multiple rotations with golden ratio angles
            golden_ratio = (1 + np.sqrt(5)) / 2
            angles = [0, np.pi/golden_ratio, np.pi/(2*golden_ratio), np.pi/(3*golden_ratio)]
            
            for i in range(len(points)):
                angle = angles[i % len(angles)]
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                points[i] = rotation_matrix @ (points[i] - center) + center
                
            # Apply sophisticated perturbations
            for i in range(len(points)):
                dist_from_center = np.linalg.norm(points[i] - center)
                max_dist = np.sqrt(2)  # Max possible in [0,1]x[0,1]
                norm_dist = dist_from_center / max_dist if max_dist > 0 else 0
                
                # Create position-dependent perturbation
                base_perturbation = 0.005 * (1 - norm_dist)  # Smaller near center
                perturbation = np.random.normal(0, base_perturbation, 2)
                points[i] += perturbation
                
            points = np.clip(points, 0, 1)
            return points
            
        def adaptive_sphere_relaxation(self, points: np.ndarray, iteration: int) -> np.ndarray:
            """Adaptive relaxation that modifies point positions based on local geometry"""
            new_points = points.copy()
            n = len(points)
            
            # Progressive refinement - use larger steps early, smaller later
            step_size = 0.02 * (1.0 - iteration / self.max_iterations * 0.8)
            
            # Sample points for local optimization
            sample_indices = np.random.choice(n, min(8, n), replace=False)
            
            for idx in sample_indices:
                # Compute forces from neighbors
                forces = np.zeros(2)
                current_point = points[idx]
                
                # Get points that are relatively close (to reduce computation)
                distances = np.linalg.norm(points - current_point, axis=1)
                # Consider only nearby points for force computation
                nearby_mask = (distances > 1e-6) & (distances < 0.5)
                nearby_indices = np.where(nearby_mask)[0]
                
                if len(nearby_indices) > 0:
                    # Apply repulsion forces from nearby points
                    for neighbor_idx in nearby_indices:
                        diff = points[neighbor_idx] - current_point
                        dist = np.linalg.norm(diff)
                        if dist > 1e-6:
                            # Repulsive force (inverse distance squared)
                            force_magnitude = 1.0 / (dist * dist + 1e-8)
                            forces += (diff / dist) * force_magnitude * 0.5
                    
                    # Apply attraction to center to prevent clustering
                    center_force = 0.1 * (0.5 - current_point)
                    forces += center_force
                    
                # Apply forces with adaptive step size
                new_points[idx] += forces * step_size
                
            # Ensure points stay within bounds
            new_points = np.clip(new_points, 0, 1)
            
            # Apply boundary-aware correction
            for i in range(n):
                # Boundary correction - push away from boundaries
                boundary_margin = 0.02
                if new_points[i][0] < boundary_margin:
                    new_points[i][0] += 0.005
                elif new_points[i][0] > 1 - boundary_margin:
                    new_points[i][0] -= 0.005
                    
                if new_points[i][1] < boundary_margin:
                    new_points[i][1] += 0.005
                elif new_points[i][1] > 1 - boundary_margin:
                    new_points[i][1] -= 0.005
                    
            return new_points
            
        def neighborhood_search(self, points: np.ndarray, iteration: int) -> np.ndarray:
            """Implement neighborhood search with adaptive patterns"""
            new_points = points.copy()
            n = len(points)
            
            # Use different search patterns based on iteration
            pattern_type = iteration % 3
            
            if pattern_type == 0:  # Local clustering search
                # Group points into clusters and move them together
                for i in range(0, n, 3):
                    cluster_points = points[i:i+3]
                    if len(cluster_points) >= 2:
                        centroid = np.mean(cluster_points, axis=0)
                        # Move cluster towards median of all points
                        all_points_centroid = np.mean(points, axis=0)
                        direction = all_points_centroid - centroid
                        movement = direction * 0.001 * (1 - iteration / self.max_iterations)
                        
                        for j in range(min(3, len(cluster_points))):
                            if i + j < n:
                                new_points[i + j] += movement
                        
            elif pattern_type == 1:  # Randomized exploration
                # Randomly perturb some points
                indices_to_move = np.random.choice(n, size=max(1, n//4), replace=False)
                for idx in indices_to_move:
                    # Adaptive perturbation based on current configuration
                    current_ratio = self.compute_min_max_ratio(new_points)
                    scale = max(0.001, 0.01 * (1 - current_ratio * 0.5))
                    new_points[idx] += np.random.normal(0, scale, 2)
                    
            else:  # Systematic refinement
                # Move points along Voronoi directions
                try:
                    vor = Voronoi(new_points)
                    for i in range(n):
                        # Move each point towards its Voronoi centroid if valid
                        if i < len(vor.point_regions) and -1 not in vor.point_regions[i]:
                            region = vor.point_regions[i]
                            if len(region) > 0:
                                vertex_indices = [j for j in region if j >= 0 and j < len(vor.vertices)]
                                if len(vertex_indices) > 0:
                                    vertices = [vor.vertices[j] for j in vertex_indices]
                                    # Compute centroid of Voronoi cell
                                    cell_centroid = np.mean(vertices, axis=0)
                                    # Move point towards cell centroid
                                    direction = cell_centroid - new_points[i]
                                    new_points[i] += direction * 0.002
                except:
                    pass
                    
            new_points = np.clip(new_points, 0, 1)
            return new_points
            
        def optimize(self) -> np.ndarray:
            """Main optimization loop using sphere packing inspired approach"""
            # Initialize with multiple strategies
            strategies = [
                self.initialize_grid_points,
                lambda: np.random.rand(self.n_points, 2),
            ]
            
            best_points = None
            best_ratio = -np.inf
            
            # Try multiple initialization strategies
            for strategy in strategies:
                points = strategy()
                
                # Apply progressive refinement cycles
                for iteration in range(self.max_iterations):
                    # Every 50 iterations, apply neighborhood search
                    if iteration % 50 == 0 and iteration > 0:
                        points = self.neighborhood_search(points, iteration)
                    
                    # Apply adaptive sphere relaxation
                    points = self.adaptive_sphere_relaxation(points, iteration)
                    
                    # Occasionally reinitialize with better configuration
                    if iteration % 100 == 0 and iteration > 0:
                        # Try a simple random restart with better spacing
                        points = np.random.rand(self.n_points, 2)
                        # Apply some structured pattern
                        points = self.initialize_grid_points()
                        
                    # Periodically evaluate and save best solution
                    if iteration % 10 == 0:
                        current_ratio = self.compute_min_max_ratio(points)
                        if current_ratio > best_ratio:
                            best_ratio = current_ratio
                            best_points = points.copy()
                            
                        # Early stopping if we get very good solution
                        if current_ratio > 0.27 and iteration > 500:
                            break
                
                # Final assessment
                current_ratio = self.compute_min_max_ratio(points)
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = points.copy()
                    
            # Return the best configuration found
            if best_points is None:
                return self.initialize_grid_points()
            return best_points
    
    # Execute the optimization
    optimizer = SpherePackingOptimizer(n_points=16, max_iterations=2000)
    return optimizer.optimize()

# EVOLVE-BLOCK-END