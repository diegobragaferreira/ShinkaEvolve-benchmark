# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import minimize
from scipy.spatial import SphericalVoronoi
import warnings
import time
from typing import Tuple

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def fibonacci_sphere(n: int) -> np.ndarray:
        """Generate points on sphere using Fibonacci spiral method"""
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle

        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)
    
    def construct_icosahedron_vertices() -> np.ndarray:
        """Construct vertices of regular icosahedron scaled to unit sphere"""
        # Golden ratio
        phi = (1 + np.sqrt(5)) / 2
        
        # Vertices of icosahedron
        vertices = np.array([
            [-1,  phi,  0],
            [ 1,  phi,  0],
            [-1, -phi,  0],
            [ 1, -phi,  0],
            [ 0, -1,  phi],
            [ 0,  1,  phi],
            [ 0, -1, -phi],
            [ 0,  1, -phi],
            [ phi,  0, -1],
            [ phi,  0,  1],
            [-phi,  0, -1],
            [-phi,  0,  1]
        ])
        
        # Normalize to unit sphere
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        vertices = vertices / norms
        
        return vertices
    
    def initialize_points_spherical_voronoi(n: int) -> np.ndarray:
        """Initialize points using spherical Voronoi construction with better spread"""
        # Start with icosahedron vertices
        if n <= 12:
            vertices = construct_icosahedron_vertices()
            # Use subset of vertices
            points = vertices[:n]
            
            # Add some random perturbation
            noise = np.random.normal(0, 0.05, points.shape)
            points += noise
            
            # Normalize to unit sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / np.maximum(norms, 1e-10)
            
            # Scale to unit cube [0,1]^3
            points = (points + 1) / 2
            
            return points
        else:
            # For more points, use fibonacci approach with perturbation
            base_points = fibonacci_sphere(n)
            
            # Add perturbation to avoid degeneracy
            noise = np.random.normal(0, 0.03, base_points.shape)
            base_points += noise
            
            # Project to sphere and normalize
            norms = np.linalg.norm(base_points, axis=1, keepdims=True)
            base_points = base_points / np.maximum(norms, 1e-10)
            
            # Scale to unit cube [0,1]^3
            base_points = (base_points + 1) / 2
            
            return base_points
    
    def compute_min_max_distances(points: np.ndarray) -> Tuple[float, float]:
        """Compute minimum and maximum distances between all point pairs."""
        if len(points) < 2:
            return 0.0, 0.0

        try:
            # Compute pairwise distances
            distances = pdist(points)

            if len(distances) == 0:
                return 0.0, 0.0

            min_dist = np.min(distances)
            max_dist = np.max(distances)

            # Avoid division by zero
            if max_dist <= 0:
                return 0.0, 0.0

            return min_dist, max_dist
        except Exception:
            return 0.0, 0.0
    
    def distance_weighted_repulsion_gradient(points: np.ndarray, alpha: float = 2.0, beta: float = 1.0) -> np.ndarray:
        """
        Compute gradient for distance-weighted repulsion that maximizes min/max ratio.
        This is a novel approach that directly optimizes the desired ratio.
        """
        n = len(points)
        if n < 2:
            return np.zeros_like(points)
        
        grad = np.zeros_like(points)
        
        # Compute distance matrix
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances
        
        # Get min distance for each point
        min_distances = np.min(distances, axis=1)
        
        # Get max distance for each point
        max_distances = np.max(distances, axis=1)
        
        # Compute weights based on distance ratios
        weights = np.zeros(n)
        for i in range(n):
            if max_distances[i] > 0:
                weights[i] = min_distances[i] / max_distances[i]
        
        # Gradient calculation for distance-weighted repulsion
        for i in range(n):
            for j in range(n):
                if i != j:
                    diff = points[i] - points[j]
                    dist = np.linalg.norm(diff)
                    
                    if dist > 0:
                        # Weighted attraction-repulsion force
                        force_magnitude = 1.0 / (dist ** alpha) if dist > 0 else 0
                        force_direction = diff / dist
                        
                        # Apply weights based on current configuration
                        weight_factor = 1.0  # Simple weight factor
                        
                        grad[i] += weight_factor * force_magnitude * force_direction
        
        return grad
    
    def ratio_based_objective(points_flat: np.ndarray) -> float:
        """Objective function that directly maximizes min/max distance ratio"""
        points = points_flat.reshape(-1, 3)
        
        # Ensure points are within bounds [0,1]^3
        points = np.clip(points, 0, 1)
        
        # Handle edge cases
        if len(points) < 2:
            return 1.0  # Very bad value
    
        try:
            distances = pdist(points)
            if len(distances) == 0:
                return 1.0  # Very bad value
                
            distances = distances[np.isfinite(distances)]
            if len(distances) == 0:
                return 1.0  # Very bad value
            
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            if d_max <= 0:
                return 1.0  # Very bad value
                
            # We want to maximize min/max ratio, so minimize -ratio
            ratio = d_min / d_max
            return -ratio  # Negative because we're minimizing
            
        except:
            return 1.0  # Very bad value
    
    def spherical_project(points: np.ndarray) -> np.ndarray:
        """Project points onto unit sphere maintaining their relative relationships"""
        # Center around origin
        centered = points - np.mean(points, axis=0)
        
        # Project to unit sphere
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        projected = centered / norms
        
        return projected
    
    def spherical_energy_optimization(initial_points: np.ndarray, max_iter: int = 1000) -> np.ndarray:
        """Optimize using spherical energy minimization approach"""
        points = initial_points.copy()
        n = len(points)
        
        # Convert to spherical coordinates and back to maintain spherical constraint
        def project_to_unit_sphere(pts):
            # Make sure points are on unit sphere
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return pts / norms
        
        def calculate_energy_and_gradients(points):
            """Calculate energy and gradients for optimization"""
            if len(points) < 2:
                return 0.0, np.zeros_like(points)
            
            try:
                # Compute pairwise distances
                distances = cdist(points, points)
                np.fill_diagonal(distances, np.inf)
                
                # Energy based on inverse distance
                energy = 0.0
                gradients = np.zeros_like(points)
                
                for i in range(n):
                    for j in range(i+1, n):
                        dist = distances[i,j]
                        if dist > 0:
                            # Energy contribution (inverse distance squared)
                            energy += 1.0 / (dist * dist)
                            
                            # Gradient contribution
                            diff = points[i] - points[j]
                            grad_i = diff / (dist * dist * dist)
                            grad_j = -grad_i
                            
                            gradients[i] += grad_i
                            gradients[j] += grad_j
                
                return energy, gradients
                
            except Exception as e:
                return 0.0, np.zeros_like(points)
        
        # Gradient descent with adaptive step size
        for iteration in range(max_iter):
            try:
                energy, gradients = calculate_energy_and_gradients(points)
                if np.isnan(energy) or np.any(np.isnan(gradients)):
                    break
                    
                # Adaptive step size
                step_size = 0.01 / (1.0 + iteration * 0.001)
                
                # Update points
                points -= step_size * gradients
                
                # Project back to unit sphere
                points = project_to_unit_sphere(points)
                
                # Keep within [0,1]^3 bounds
                points = np.clip(points, 0, 1)
                
            except Exception:
                break
                
        return points
    
    # Main optimization routine
    try:
        # Initialize with spherical Voronoi-inspired configuration
        np.random.seed(42)
        initial_points = initialize_points_spherical_voronoi(14)
        
        # First phase: Spherical energy optimization
        points = spherical_energy_optimization(initial_points.copy(), 500)
        
        # Second phase: Fine-tune with gradient-based optimization  
        bounds = [(0, 1) for _ in range(42)]
        
        # Use scipy minimize with L-BFGS-B for final refinement
        try:
            result = minimize(
                ratio_based_objective,
                points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                points = result.x.reshape(-1, 3)
        except Exception:
            # If optimization fails, proceed with existing points
            pass
        
        # Third phase: Final validation and projection
        points = np.clip(points, 0, 1)
        
        # Ensure we have a valid solution
        min_dist, max_dist = compute_min_max_distances(points)
        if max_dist <= 0 or min_dist <= 0:
            # Revert to initial configuration if invalid
            points = initial_points
        
        return points
        
    except Exception as e:
        # Fallback to simple initialization
        warnings.warn(f"Optimization failed: {e}")
        np.random.seed(42)
        return np.random.rand(14, 3)

# EVOLVE-BLOCK-END