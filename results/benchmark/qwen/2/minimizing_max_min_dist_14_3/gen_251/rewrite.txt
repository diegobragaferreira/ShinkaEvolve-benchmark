# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
import math
import time
from typing import Tuple, Optional
from scipy.spatial import SphericalVoronoi

class PointOptimizer:
    def __init__(self, num_points: int = 14, dimension: int = 3):
        self.num_points = num_points
        self.dimension = dimension
        self.best_points = None
        self.best_ratio = 0.0

    def icosahedral_arrangement(self) -> np.ndarray:
        """Generate points based on icosahedral symmetry - a well-known good starting configuration"""
        # Golden ratio
        phi = (1 + math.sqrt(5)) / 2
        
        # Vertices of icosahedron scaled to unit sphere
        vertices = []
        
        # 12 vertices of icosahedron
        vertices.append([0, 1, phi])
        vertices.append([0, -1, phi])
        vertices.append([0, 1, -phi])
        vertices.append([0, -1, -phi])
        vertices.append([1, phi, 0])
        vertices.append([-1, phi, 0])
        vertices.append([1, -phi, 0])
        vertices.append([-1, -phi, 0])
        vertices.append([phi, 0, 1])
        vertices.append([phi, 0, -1])
        vertices.append([-phi, 0, 1])
        vertices.append([-phi, 0, -1])
        
        # Normalize to unit sphere
        points = np.array(vertices)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # If we need fewer points, take first n points
        if self.num_points < 12:
            return points[:self.num_points]
        elif self.num_points == 12:
            return points
        else:
            # For more than 12 points, add additional points using a spherical spiral approach
            extra_points = []
            for i in range(self.num_points - 12):
                # Generate points in spherical spiral pattern
                k = i + 12
                phi_spiral = math.acos(-1 + 2 * k / (self.num_points - 1))
                theta_spiral = math.sqrt(self.num_points * math.pi) * phi_spiral
                
                x = math.sin(phi_spiral) * math.cos(theta_spiral)
                y = math.sin(phi_spiral) * math.sin(theta_spiral)
                z = math.cos(phi_spiral)
                
                extra_points.append([x, y, z])
            
            return np.vstack([points, np.array(extra_points)])

    def project_to_sphere(self, points: np.ndarray) -> np.ndarray:
        """Project points onto unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return points / norms

    def calculate_ratio(self, points: np.ndarray) -> float:
        """Calculate the min/max distance ratio"""
        if len(points) < 2:
            return 0.0

        try:
            distances = pdist(points)

            if len(distances) == 0:
                return 0.0

            d_min = np.min(distances)
            d_max = np.max(distances)

            if d_max <= 0:
                return 0.0

            return d_min / d_max
        except Exception:
            return 0.0

    def distance_weighted_relaxation(self, points: np.ndarray, iterations: int = 50) -> np.ndarray:
        """Apply distance-weighted relaxation to optimize point distribution"""
        current_points = points.copy()
        
        for _ in range(iterations):
            # Calculate all pairwise distances
            distances = cdist(current_points, current_points)
            
            # Create repulsion forces (inverse distance weighted)
            # Set diagonal to large value to avoid self-interaction
            np.fill_diagonal(distances, 1e10)
            
            # Compute forces for each point
            forces = np.zeros_like(current_points)
            
            for i in range(len(current_points)):
                # Calculate repulsion from all other points
                for j in range(len(current_points)):
                    if i != j:
                        # Direction from point j to point i
                        direction = current_points[i] - current_points[j]
                        dist = np.linalg.norm(direction)
                        
                        if dist > 0:
                            # Inverse square law with distance weighting
                            force_magnitude = 1.0 / (dist * dist + 1e-10)
                            force_direction = direction / dist
                            forces[i] += force_magnitude * force_direction
            
            # Apply forces with adaptive step size
            step_size = 0.01 / (1.0 + np.linalg.norm(forces, axis=1)**0.5)
            
            # Update positions with bounded movement
            new_positions = current_points + np.multiply(step_size.reshape(-1, 1), forces)
            
            # Project back to sphere
            current_points = self.project_to_sphere(new_positions)
            
            # Add small random jitter occasionally to escape local minima
            if _ % 10 == 0:
                jitter = np.random.normal(0, 0.0001, current_points.shape)
                current_points += jitter
                current_points = self.project_to_sphere(current_points)
        
        return current_points

    def hierarchical_optimize(self, initial_points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Perform hierarchical optimization with multiple levels of refinement"""
        points = initial_points.copy()
        
        # Level 1: Coarse global optimization
        points = self.distance_weighted_relaxation(points, iterations=30)
        
        # Level 2: Fine-grained optimization with local refinement
        points = self.distance_weighted_relaxation(points, iterations=50)
        
        # Level 3: Local optimization around promising areas
        # Identify clusters or problematic regions and focus optimization there
        for _ in range(3):
            # Apply local relaxation focused on specific points
            local_points = points.copy()
            for i in range(len(points)):
                # Save current point
                original_point = local_points[i].copy()
                
                # Try small perturbations
                best_point = original_point.copy()
                best_ratio = self.calculate_ratio(local_points)
                
                for _ in range(10):
                    # Small random perturbation
                    perturbation = np.random.normal(0, 0.005, 3)
                    test_point = original_point + perturbation
                    test_point = self.project_to_sphere(test_point.reshape(1, 3)).reshape(-1)
                    
                    # Test this perturbation
                    local_points[i] = test_point
                    new_ratio = self.calculate_ratio(local_points)
                    
                    if new_ratio > best_ratio:
                        best_ratio = new_ratio
                        best_point = test_point.copy()
                    
                    # Restore original for next test
                    local_points[i] = original_point
                
                points[i] = best_point
        
        # Final relaxation
        points = self.distance_weighted_relaxation(points, iterations=20)
        
        return points, self.calculate_ratio(points)

    def symmetry_breaking(self, points: np.ndarray, num_attempts: int = 5) -> np.ndarray:
        """Apply symmetry breaking techniques to escape symmetric local optima"""
        best_points = points.copy()
        best_ratio = self.calculate_ratio(points)
        
        for attempt in range(num_attempts):
            # Create a slightly perturbed version
            perturbed_points = points.copy()
            
            # Add symmetry-breaking noise
            noise = np.random.normal(0, 0.01, perturbed_points.shape)
            perturbed_points += noise
            
            # Project back to sphere
            perturbed_points = self.project_to_sphere(perturbed_points)
            
            # Optimize the perturbed version
            optimized_points, ratio = self.hierarchical_optimize(perturbed_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
        
        return best_points

    def run_optimization(self) -> np.ndarray:
        """Run the spherical tiling optimization approach"""
        # Start with a good initial configuration
        initial_points = self.icosahedral_arrangement()
        
        # Apply our hierarchical optimization approach
        optimized_points, ratio = self.hierarchical_optimize(initial_points)
        
        # Try symmetry breaking to get better results
        sym_break_points = self.symmetry_breaking(optimized_points)
        sym_break_ratio = self.calculate_ratio(sym_break_points)
        
        if sym_break_ratio > ratio:
            self.best_points = sym_break_points
            self.best_ratio = sym_break_ratio
        else:
            self.best_points = optimized_points
            self.best_ratio = ratio
        
        # Perform one final comprehensive optimization
        final_points, final_ratio = self.hierarchical_optimize(self.best_points)
        
        if final_ratio > self.best_ratio:
            self.best_points = final_points
            self.best_ratio = final_ratio
            
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