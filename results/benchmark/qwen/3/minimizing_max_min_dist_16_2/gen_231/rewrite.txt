# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import Voronoi
import math
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    class GeometricSpatialOptimizer:
        def __init__(self):
            self.n_points = 16
            self.boundary_margin = 0.02
            self.max_iterations = 1000
            
        def compute_distance_matrix(self, points):
            """Compute pairwise distance matrix efficiently."""
            return squareform(pdist(points))
            
        def compute_min_max_ratio(self, distance_matrix):
            """Calculate the ratio of minimum to maximum distances."""
            off_diagonal = distance_matrix[distance_matrix > 0]
            if len(off_diagonal) == 0:
                return 0.0
            d_min = np.min(off_diagonal)
            d_max = np.max(off_diagonal)
            return d_min / d_max if d_max > 0 else 0.0
            
        def compute_voronoi_quality(self, points):
            """Compute quality metric based on Voronoi cell uniformity."""
            try:
                vor = Voronoi(points)
                areas = []
                for i, region in enumerate(vor.regions):
                    if not region or -1 in region:
                        continue
                    vertices = [vor.vertices[j] for j in region if j >= 0]
                    if len(vertices) >= 3:
                        vertices = np.array(vertices)
                        # Compute polygon area using shoelace formula
                        area = 0.5 * np.abs(np.dot(vertices[:, 0], np.roll(vertices[:, 1], 1)) - 
                                           np.dot(vertices[:, 1], np.roll(vertices[:, 0], 1)))
                        areas.append(area)
                if areas:
                    return np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0.0
                return 0.0
            except:
                return 0.0
                
        def initialize_mathematical_structure(self):
            """Initialize points using mathematical principles for optimal distribution."""
            # Create a mathematically optimized hexagonal pattern
            points = []
            sqrt3 = math.sqrt(3)
            row_spacing = sqrt3 / 2
            col_spacing = 1.0
            
            # Generate points in hexagonal lattice pattern with 16 points
            rows, cols = 4, 4
            for i in range(rows):
                for j in range(cols):
                    if len(points) >= self.n_points:
                        break
                    x = j * col_spacing + (i % 2) * col_spacing / 2
                    y = i * row_spacing
                    points.append([x, y])
            
            points = np.array(points[:self.n_points])
            
            # Normalize to fit well in unit square
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])
            
            if x_range > 0:
                points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
            if y_range > 0:
                points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
                
            # Scale and center in [0.05, 0.95] range
            points[:, 0] = 0.05 + 0.9 * points[:, 0]
            points[:, 1] = 0.05 + 0.9 * points[:, 1]
            
            # Apply sophisticated symmetry breaking using prime sequences
            np.random.seed(42)
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
            
            for i in range(len(points)):
                prime_idx = i % len(primes)
                base_magnitude = 0.01
                # Use prime-based perturbation with mathematical variation
                magnitude = base_magnitude * (1.0 + 0.1 * (primes[prime_idx] % 7))
                # Add directional component
                angle = i * 0.785398  # pi/4 increments
                noise_x = np.random.normal(0, magnitude, 1)[0] * math.cos(angle)
                noise_y = np.random.normal(0, magnitude, 1)[0] * math.sin(angle)
                points[i] += [noise_x, noise_y]
                
            # Ensure points stay within boundary margins
            points[:, 0] = np.clip(points[:, 0], self.boundary_margin, 1 - self.boundary_margin)
            points[:, 1] = np.clip(points[:, 1], self.boundary_margin, 1 - self.boundary_margin)
            
            return points
            
        def geometric_constraint_satisfaction(self, points, max_iter=50):
            """Apply geometric constraint satisfaction to improve point distribution."""
            current_points = points.copy()
            
            for iteration in range(max_iter):
                try:
                    # Calculate current state
                    dist_matrix = self.compute_distance_matrix(current_points)
                    ratio = self.compute_min_max_ratio(dist_matrix)
                    
                    if ratio < 1e-8:  # Prevent numerical issues
                        break
                        
                    # Calculate Voronoi quality for reference
                    voronoi_quality = self.compute_voronoi_quality(current_points)
                    
                    # Apply geometric correction to each point
                    new_points = current_points.copy()
                    updated = False
                    
                    # Sample points in a specific geometric pattern for better coverage
                    sample_indices = list(range(0, len(current_points), 3)) + [len(current_points)-1]
                    if len(sample_indices) > len(current_points):
                        sample_indices = list(range(len(current_points)))
                    
                    # Apply corrections to sampled points to maintain geometric balance
                    for idx in sample_indices:
                        # Calculate forces from neighbors
                        neighbor_distances = dist_matrix[idx]
                        # Ignore self-distance (0) and very close neighbors
                        mask = (neighbor_distances > 1e-6) & (neighbor_distances < 1.0)
                        neighbors = np.where(mask)[0]
                        
                        if len(neighbors) == 0:
                            continue
                            
                        # Calculate average influence from neighbors
                        avg_force = np.zeros(2)
                        for n_idx in neighbors:
                            if n_idx != idx:
                                diff = current_points[n_idx] - current_points[idx]
                                distance = np.linalg.norm(diff)
                                if distance > 1e-6:
                                    force = diff / distance
                                    # Use inverse square law for force strength
                                    force_strength = 1.0 / (distance * distance + 1e-8)
                                    avg_force += force * force_strength
                        
                        # Apply constraint to move point away from dense areas
                        if np.linalg.norm(avg_force) > 0:
                            # Move in direction opposite to average force
                            avg_force = -avg_force
                            avg_force = avg_force / (np.linalg.norm(avg_force) + 1e-8)
                            # Apply step size based on how well we're doing
                            step_size = 0.005 * (1.0 + 0.5 * voronoi_quality)
                            new_points[idx] += avg_force * step_size
                            updated = True
                    
                    # Apply boundary corrections
                    new_points[:, 0] = np.clip(new_points[:, 0], self.boundary_margin, 1 - self.boundary_margin)
                    new_points[:, 1] = np.clip(new_points[:, 1], self.boundary_margin, 1 - self.boundary_margin)
                    
                    current_points = new_points
                    
                    # Early stopping if improvement is minimal
                    if not updated:
                        break
                        
                except Exception:
                    break
                    
            return current_points
            
        def hybrid_gradient_refinement(self, points, max_iter=100):
            """Refine using a hybrid gradient approach with geometric awareness."""
            current_points = points.copy()
            
            for iteration in range(max_iter):
                try:
                    # Calculate current ratio and derivatives
                    dist_matrix = self.compute_distance_matrix(current_points)
                    ratio = self.compute_min_max_ratio(dist_matrix)
                    
                    if ratio < 1e-8:
                        break
                        
                    # Calculate gradient for each point (simplified geometric gradient)
                    new_points = current_points.copy()
                    updated = False
                    
                    # Use a simplified gradient approximation based on neighbor distances
                    for i in range(len(current_points)):
                        # Calculate gradient based on neighbors' influence
                        grad = np.zeros(2)
                        distances = dist_matrix[i]
                        
                        # Focus on immediate neighbors for gradient calculation
                        neighbor_indices = np.argsort(distances)[1:6]  # Top 5 neighbors
                        
                        for j in neighbor_indices:
                            if j != i:
                                diff = current_points[j] - current_points[i]
                                distance = np.linalg.norm(diff)
                                if distance > 1e-6:
                                    # Gradient pushes points apart when they're close
                                    # and pulls them together when they're far
                                    force = diff / (distance * distance + 1e-8)
                                    force *= (distance - 0.5)  # Prefer medium distances
                                    grad += force
                                    
                        # Apply gradient descent step
                        if np.linalg.norm(grad) > 0:
                            step_size = 0.002 * (1.0 + 0.2 * ratio)  # Adaptive step size
                            new_points[i] -= grad * step_size
                            updated = True
                    
                    # Apply boundary constraints
                    new_points[:, 0] = np.clip(new_points[:, 0], self.boundary_margin, 1 - self.boundary_margin)
                    new_points[:, 1] = np.clip(new_points[:, 1], self.boundary_margin, 1 - self.boundary_margin)
                    
                    current_points = new_points
                    
                    if not updated:
                        break
                        
                except Exception:
                    break
                    
            return current_points
            
        def optimize(self):
            """Main optimization procedure."""
            # Phase 1: Mathematical initialization
            points = self.initialize_mathematical_structure()
            
            # Phase 2: Constraint satisfaction to improve geometric structure
            points = self.geometric_constraint_satisfaction(points, 50)
            
            # Phase 3: Hybrid gradient refinement
            points = self.hybrid_gradient_refinement(points, 100)
            
            # Phase 4: Iterative geometric improvement
            for phase in range(3):
                points = self.geometric_constraint_satisfaction(points, 30)
                points = self.hybrid_gradient_refinement(points, 50)
            
            # Final cleanup and boundary enforcement
            points[:, 0] = np.clip(points[:, 0], self.boundary_margin, 1 - self.boundary_margin)
            points[:, 1] = np.clip(points[:, 1], self.boundary_margin, 1 - self.boundary_margin)
            
            return points
    
    optimizer = GeometricSpatialOptimizer()
    return optimizer.optimize()

# EVOLVE-BLOCK-END