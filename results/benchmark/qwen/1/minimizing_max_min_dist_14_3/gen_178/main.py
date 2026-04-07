# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import math
from typing import Tuple

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a novel torus-based evolutionary approach for improved convergence.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def distance_ratio(points_flat):
        """Calculate the ratio of minimum to maximum distance"""
        points = points_flat.reshape(-1, 3)
        distances = squareform(pdist(points))
        # Set diagonal to large value so it doesn't affect min/max
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return min_dist / max_dist
    
    def torus_to_sphere_mapping(torus_coords):
        """
        Map torus coordinates to sphere coordinates using parametric equations
        Torus parameters: R = 1 (major radius), r = 0.5 (minor radius)
        """
        # torus_coords: [theta1, theta2, ...] - angles in torus space
        points = []
        for i in range(0, len(torus_coords), 2):
            theta1 = torus_coords[i]
            theta2 = torus_coords[i+1]
            
            # Torus parametrization with major radius R=1, minor radius r=0.5
            R = 1.0
            r = 0.5
            
            x = (R + r * np.cos(theta2)) * np.cos(theta1)
            y = (R + r * np.cos(theta2)) * np.sin(theta1)
            z = r * np.sin(theta2)
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def sphere_to_torus_mapping(sphere_points):
        """
        Convert spherical coordinates to torus representation
        Note: This is a simplified inverse mapping for the optimization process
        """
        torus_coords = []
        for point in sphere_points:
            x, y, z = point
            
            # Project to torus surface (approximate inverse)
            # Using azimuthal angle theta1 and polar angle theta2
            theta1 = np.arctan2(y, x)
            theta2 = np.arctan2(z, np.sqrt(x*x + y*y))
            
            # Normalize angles to [0, 2π)
            if theta1 < 0:
                theta1 += 2 * np.pi
            if theta2 < 0:
                theta2 += 2 * np.pi
                
            torus_coords.extend([theta1, theta2])
        
        return np.array(torus_coords)
    
    def torus_objective_function(torus_coords):
        """
        Objective function optimized in torus parameter space
        """
        # Convert torus coords to sphere coords
        sphere_points = torus_to_sphere_mapping(torus_coords)
        
        # Ensure points are normalized to unit sphere
        norms = np.linalg.norm(sphere_points, axis=1)
        sphere_points = sphere_points / norms[:, np.newaxis]
        
        # Calculate distance ratio
        distances = squareform(pdist(sphere_points))
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return min_dist / max_dist
    
    def modified_hybrid_initialization(n):
        """
        Generate initial points using modified hybrid approach in torus space
        """
        # Start with icosahedron vertices for better symmetry in 3D space
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        vertices = [
            (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
            (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
            (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1)
        ]
        
        # Convert to numpy array and normalize
        points = np.array(vertices, dtype=float)
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        
        # Add extra points using Fibonacci spiral but in 3D
        remaining = n - len(points)
        if remaining > 0:
            for i in range(remaining):
                # Improved Fibonacci-like distribution with better spread
                theta = math.acos(1 - 2 * (i / (remaining - 1)))
                # Golden ratio multiple for better distribution
                phi_coord = (i * 2.414213562) % (2 * math.pi)
                
                x = math.sin(theta) * math.cos(phi_coord)
                y = math.sin(theta) * math.sin(phi_coord)
                z = math.cos(theta)
                points = np.vstack([points, [x, y, z]])
        
        # Apply jittering to break symmetry
        np.random.seed(42)
        noise = np.random.normal(0, 0.015, points.shape)
        points += noise
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        
        return points
    
    def adaptive_optimization(torus_coords, maxiter=800):
        """
        Adaptive optimization in torus space with multiple phases
        """
        # Phase 1: Coarse optimization with broad search
        bounds = [(-2*np.pi, 2*np.pi)] * len(torus_coords)
        
        def opt_func(coords):
            return -torus_objective_function(coords)
        
        # Use L-BFGS-B with adaptive settings
        result = minimize(
            opt_func,
            torus_coords,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-4, 'gtol': 1e-4},
            tol=1e-4
        )
        
        # Phase 2: Medium optimization
        if result.success:
            torus_coords = result.x
            
        bounds = [(-2*np.pi, 2*np.pi)] * len(torus_coords)
        result = minimize(
            opt_func,
            torus_coords,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-6, 'gtol': 1e-6},
            tol=1e-6
        )
        
        # Phase 3: Fine optimization
        if result.success:
            torus_coords = result.x
            
        bounds = [(-2*np.pi, 2*np.pi)] * len(torus_coords)
        result = minimize(
            opt_func,
            torus_coords,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8},
            tol=1e-8
        )
        
        return result.x if result.success else torus_coords
    
    def evolution_mutation(current_torus_coords, mutation_rate=0.05):
        """
        Specialized mutation operator for torus coordinates to maintain diversity
        """
        mutated = current_torus_coords.copy()
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                # Apply torus-aware perturbation
                mutated[i] += np.random.normal(0, 0.3)
                # Wrap around torus boundaries
                mutated[i] = mutated[i] % (2 * np.pi)
        return mutated
    
    best_ratio = -np.inf
    best_points = None
    
    # Multi-start optimization with torus-based approach
    for restart in range(12):  # More restarts for better exploration
        np.random.seed(42 + restart)
        
        # Create initial points using hybrid approach
        initial_sphere_points = modified_hybrid_initialization(14)
        
        # Convert to torus representation
        initial_torus_coords = sphere_to_torus_mapping(initial_sphere_points)
        
        # Add small random perturbation to torus coordinates
        perturbation = np.random.normal(0, 0.1, initial_torus_coords.shape)
        initial_torus_coords += perturbation
        
        # Optimize in torus space
        try:
            optimized_torus_coords = adaptive_optimization(initial_torus_coords, maxiter=800)
            
            # Convert back to sphere coordinates
            optimized_sphere_points = torus_to_sphere_mapping(optimized_torus_coords)
            
            # Ensure normalization
            norms = np.linalg.norm(optimized_sphere_points, axis=1)
            optimized_sphere_points = optimized_sphere_points / norms[:, np.newaxis]
            
            # Calculate final ratio
            ratio = distance_ratio(optimized_sphere_points.flatten())
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_sphere_points.copy()
                
        except Exception as e:
            continue
    
    # Fallback to hybrid initialization if optimization fails
    if best_points is None:
        initial_points = modified_hybrid_initialization(14)
        best_points = initial_points
    
    return best_points

# EVOLVE-BLOCK-END
