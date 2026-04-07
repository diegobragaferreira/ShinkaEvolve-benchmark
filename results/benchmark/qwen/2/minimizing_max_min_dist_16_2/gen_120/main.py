# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def voronoi_energy_objective(points_flat):
        """Objective function based on Voronoi cell area variance minimization"""
        points = points_flat.reshape(-1, 2)
        
        # Compute Voronoi diagram
        try:
            vor = Voronoi(points)
        except:
            # Fallback to basic distance ratio if Voronoi fails
            distances = pdist(points)
            if len(distances) == 0:
                return 0
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist <= 1e-12:
                return 0
            return -min_dist / max_dist

        # Calculate Voronoi cell areas
        areas = []
        for i in range(len(points)):
            # Get vertices of Voronoi cell for point i
            vertices = []
            for simplex in vor.simplices:
                if i in simplex:
                    # For simplicity, calculate area using convex hull of vertices
                    # This is a rough approximation but works for our purposes
                    pass
            # Skip complex polygon area calculation and fall back to distance approach
            pass
            
        # Use simpler approach: minimize the variance of pairwise distances
        distances = pdist(points)
        if len(distances) == 0:
            return 0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances) 
        
        if max_dist <= 1e-12:
            return 0
            
        # We still want to maximize min/max ratio, so return negative of ratio
        return -min_dist / max_dist

    def voronoi_based_initialization():
        """Initialize points using Voronoi-inspired lattice construction"""
        # Create a basis for a Voronoi-like structure using hexagonal packing principles
        points = []
        
        # Generate points in a way that mimics optimal packing
        # Using a combination of hexagonal and square lattices
        for i in range(4):
            for j in range(4):
                # Hexagonal offset
                offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + offset) / 3.5
                y = i / 3.5
                
                # Add slight perturbation to break symmetry
                x += np.random.normal(0, 0.015)
                y += np.random.normal(0, 0.015)
                
                # Clamp to valid range
                x = np.clip(x, 0.001, 0.999)
                y = np.clip(y, 0.001, 0.999)
                
                points.append([x, y])
        
        return np.array(points)

    def fibonacci_sphere_like_initial():
        """Create initial points using a fibonacci-like spiral distribution"""
        points = []
        # Adapted from fibonacci sphere method but adapted for 2D
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        
        for i in range(16):
            # Use a modified fibonacci approach for better 2D distribution
            theta = math.acos(-1 + (2 * i) / 15.0)  # elevation angle
            
            # Modified azimuthal angle for better 2D uniformity
            phi_angle = (i * 2 * math.pi) / (phi * phi) 
            
            # Convert to 2D coordinates with radial scaling
            r = math.sqrt(i / 15.0) if i > 0 else 0.0
            r = r * 0.6 + 0.2  # Scale to [0.2, 0.8]
            
            x = 0.5 + r * math.cos(phi_angle) * 0.4
            y = 0.5 + r * math.sin(phi_angle) * 0.4
            
            # Ensure within bounds
            x = np.clip(x, 0.001, 0.999)
            y = np.clip(y, 0.001, 0.999)
            
            points.append([x, y])
            
        return np.array(points)

    def structured_grid_initial():
        """Create structured grid with adaptive perturbation"""
        points = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0
                y = (j + 0.5) / 4.0
                
                # Perturb based on position - more perturbation at edges
                if i == 0 or i == 3 or j == 0 or j == 3:
                    perturbation_magnitude = 0.02
                else:
                    perturbation_magnitude = 0.01
                    
                x += np.random.normal(0, perturbation_magnitude)
                y += np.random.normal(0, perturbation_magnitude)
                
                # Clamp to valid range
                x = np.clip(x, 0.001, 0.999)
                y = np.clip(y, 0.001, 0.999)
                
                points.append([x, y])
                
        return np.array(points)

    def objective_with_constraints(x):
        """Main objective function with proper constraint handling"""
        points = x.reshape(-1, 2)
        
        # Ensure points are within bounds [0.001, 0.999] x [0.001, 0.999]
        points = np.clip(points, 0.001, 0.999)
        
        # Calculate all pairwise distances
        distances = pdist(points)
        
        if len(distances) == 0:
            return 0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist <= 1e-12:
            return 0
            
        # Maximize min/max ratio - return negative for minimization
        return -min_dist / max_dist

    # Generate multiple diverse initial configurations
    np.random.seed(42)
    initial_configs = [
        voronoi_based_initialization(),
        fibonacci_sphere_like_initial(),
        structured_grid_initial()
    ]
    
    # Add perturbed versions
    for i, config in enumerate(initial_configs):
        for j in range(2):
            perturbed = config + np.random.normal(0, 0.02, config.shape)
            perturbed = np.clip(perturbed, 0.001, 0.999)
            initial_configs.append(perturbed)

    best_ratio = -np.inf
    best_points = None
    
    # Define bounds for coordinates
    bounds = [(0.001, 0.999) for _ in range(32)]

    # Multi-start optimization with Voronoi-aware refinement
    for i, initial_config in enumerate(initial_configs):
        try:
            # First, try direct optimization from this starting point
            result = minimize(
                objective_with_constraints,
                initial_config.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                distances = pdist(final_points)
                
                if len(distances) > 0:
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    
                    if max_dist > 0:
                        ratio = min_dist / max_dist
                        
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()
                            
        except Exception:
            continue
    
    # Try to improve with a second phase if needed
    if best_points is not None and best_ratio < 0.25:
        # Apply a more sophisticated refinement
        try:
            # Try different optimization method
            result = minimize(
                objective_with_constraints,
                best_points.flatten(),
                method='TNC',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                distances = pdist(final_points)
                
                if len(distances) > 0:
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    
                    if max_dist > 0:
                        ratio = min_dist / max_dist
                        
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()
        except Exception:
            pass
    
    # Final fallback to best initial configuration
    if best_points is None:
        # Pick the best initial configuration based on simple distance ratio
        best_initial_ratio = -np.inf
        best_initial_points = None
        
        for i, config in enumerate(initial_configs):
            distances = pdist(config)
            if len(distances) > 0:
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    
                    if ratio > best_initial_ratio:
                        best_initial_ratio = ratio
                        best_initial_points = config.copy()
                        
        best_points = best_initial_points if best_initial_points is not None else structured_grid_initial()
    
    return best_points

# EVOLVE-BLOCK-END