# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import SphericalVoronoi
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import distance

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def normalize_points(points):
        """Normalize points to unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return points / norms
    
    def calculate_ratio(points):
        """Calculate min/max distance ratio"""
        if len(points) < 2:
            return 0.0
            
        # Calculate pairwise distances
        distances = pdist(points)
        
        if len(distances) == 0:
            return 0.0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Handle edge case where all points are identical
        if d_max == 0:
            return 0.0
            
        return d_min / d_max
    
    def spherical_voronoi_initialization(n_points):
        """Generate initial points using spherical Voronoi-like approach"""
        # Start with points on a sphere
        np.random.seed(42)
        
        # Use a variation of Fibonacci spiral with small perturbations
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle
        
        for i in range(n_points):
            y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
            
        points = np.array(points)
        
        # Add slight perturbations to ensure non-regularity
        noise = np.random.normal(0, 0.01, points.shape)
        points += noise
        
        # Normalize to unit sphere
        points = normalize_points(points)
        
        return points
    
    def objective_spherical(x):
        """Objective function for spherical optimization"""
        points = x.reshape(-1, 3)
        # Normalize to unit sphere
        points = normalize_points(points)
        
        distances = pdist(points)
        
        if len(distances) == 0:
            return -1.0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max == 0:
            return -1.0
            
        # Maximize min/max ratio
        return -d_min / d_max
    
    def objective_3d(x):
        """Objective function for 3D optimization (minimize negative ratio)"""
        points = x.reshape(-1, 3)
        
        # Ensure points are within bounds [0,1]^3
        points = np.clip(points, 0, 1)
        
        distances = pdist(points)
        
        if len(distances) == 0:
            return 1.0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max == 0:
            return 1.0
            
        # Maximize min/max ratio
        return -d_min / d_max
    
    # Phase 1: Spherical Voronoi initialization and global optimization
    print("Initializing with spherical Voronoi approach...")
    initial_points = spherical_voronoi_initialization(14)
    
    # Flatten for optimization
    x0 = initial_points.flatten()
    
    # Define bounds for optimization (coordinates in [0,1])
    bounds = [(0, 1) for _ in range(42)]
    
    # Run coarse differential evolution on spherical surface
    try:
        result = differential_evolution(
            objective_spherical,
            [(0, 1) for _ in range(42)],  # Bounds are handled in objective
            seed=42,
            maxiter=50,
            popsize=10,
            mutation=(0.5, 1),
            recombination=0.7,
            atol=1e-6,
            tol=1e-6,
            disp=False
        )
        
        # Convert back to points and normalize
        optimized_points = result.x.reshape(-1, 3)
        optimized_points = normalize_points(optimized_points)
        
        # Fine-tune with local optimization
        # Try several local optimization methods
        
        # Method 1: Local optimization using Nelder-Mead
        try:
            result_fine = minimize(
                objective_3d,
                optimized_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 100, 'disp': False}
            )
            fine_points = result_fine.x.reshape(-1, 3)
            fine_points = np.clip(fine_points, 0, 1)
            
            # Check if refined version is better
            current_ratio = calculate_ratio(optimized_points)
            refined_ratio = calculate_ratio(fine_points)
            
            if refined_ratio > current_ratio:
                optimized_points = fine_points
                
        except:
            pass
            
        # Method 2: Another round of differential evolution for final refinement
        try:
            final_result = differential_evolution(
                objective_3d,
                [(0, 1) for _ in range(42)],
                seed=42,
                maxiter=30,
                popsize=8,
                mutation=(0.7, 1),
                recombination=0.8,
                atol=1e-8,
                tol=1e-8,
                disp=False
            )
            
            final_points = final_result.x.reshape(-1, 3)
            final_points = np.clip(final_points, 0, 1)
            
            # Final comparison
            current_ratio = calculate_ratio(optimized_points)
            final_ratio = calculate_ratio(final_points)
            
            if final_ratio > current_ratio:
                optimized_points = final_points
                
        except:
            pass
            
    except Exception as e:
        # Fallback to simpler approach if optimization fails
        print(f"Optimization failed with error: {e}")
        # Return initial spherical points
        optimized_points = initial_points
        
    # Final normalization to ensure they're on the unit sphere
    optimized_points = normalize_points(optimized_points)
    
    # Ensure all points are within [0,1]^3 (they should already be)
    optimized_points = np.clip(optimized_points, 0, 1)
    
    return optimized_points

# EVOLVE-BLOCK-END