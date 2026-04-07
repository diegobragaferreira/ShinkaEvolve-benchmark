# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import Voronoi
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a Voronoi-guided optimization approach.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the min/max distance ratio for given point configuration."""
        # Ensure points are within unit square
        points = np.clip(points, 0, 1)

        # Compute pairwise distances
        distances = squareform(pdist(points))

        # Set diagonal to large value so it doesn't affect min
        np.fill_diagonal(distances, np.inf)

        # Get minimum and maximum distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Return ratio
        if max_dist > 0:
            return min_dist / max_dist
        else:
            return 0.0

    def compute_voronoi_cell_areas(points):
        """Compute the areas of Voronoi cells for given points."""
        try:
            # Create Voronoi diagram with extended bounds
            vor = Voronoi(points)
            
            # Compute areas of finite Voronoi cells
            areas = []
            for region in vor.regions:
                if len(region) > 0 and not any(r == -1 for r in region):
                    # Calculate polygon area using shoelace formula
                    vertices = [vor.vertices[r] for r in region]
                    if len(vertices) >= 3:
                        x_coords = [v[0] for v in vertices]
                        y_coords = [v[1] for v in vertices]
                        area = 0.5 * abs(sum(x_coords[i] * y_coords[i+1] - x_coords[i+1] * y_coords[i] 
                                            for i in range(len(vertices)-1)) + 
                                        x_coords[-1] * y_coords[0] - x_coords[0] * y_coords[-1])
                        areas.append(area)
            
            return areas
        except:
            return [0.0] * len(points)

    def objective_function(points_flat):
        # Reshape flat array back to 16x2 points
        points = points_flat.reshape(-1, 2)

        # Ensure points are within unit square
        points = np.clip(points, 0, 1)

        # Compute pairwise distances
        distances = squareform(pdist(points))

        # Set diagonal to large value so it doesn't affect min
        np.fill_diagonal(distances, np.inf)

        # Get minimum and maximum distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Return negative ratio since we want to maximize
        if max_dist > 0:
            return -min_dist / max_dist
        else:
            return -np.inf

    def voronoi_based_initialization():
        """Create initial configuration based on Voronoi cell uniformity."""
        np.random.seed(42)
        
        # Start with a hexagonal grid pattern (which naturally forms Voronoi structures)
        points = []
        rows = 4
        cols = 4
        
        # Create a staggered hexagonal grid
        for i in range(rows):
            for j in range(cols):
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25
                
                # Add small perturbation to make it less regular
                x += (np.random.random() - 0.5) * 0.02
                y += (np.random.random() - 0.5) * 0.02
                
                points.append([x, y])
        
        points = np.array(points)
        
        # Apply constraints to break symmetry and avoid degenerate solutions
        # Fix corner points to ensure boundary constraints are met properly
        points[0] = [0.01, 0.01]      # bottom-left
        points[3] = [0.99, 0.01]      # bottom-right
        points[12] = [0.01, 0.99]     # top-left
        points[15] = [0.99, 0.99]     # top-right
        
        # Ensure all points are within bounds
        points = np.clip(points, 0.01, 0.99)
        return points

    def adaptive_perturbation(points, iteration=0):
        """Apply adaptive perturbations based on Voronoi cell quality."""
        # Compute current Voronoi cell areas to assess distribution quality
        cell_areas = compute_voronoi_cell_areas(points)
        
        if len(cell_areas) > 0:
            mean_area = np.mean(cell_areas) if np.mean(cell_areas) > 0 else 1.0
            std_area = np.std(cell_areas) if len(cell_areas) > 1 else 1.0
            
            # Determine perturbation magnitude based on cell uniformity
            # Lower uniformity requires more aggressive adjustment
            uniformity_factor = 1.0 - (std_area / mean_area if mean_area > 0 else 0.0)
            max_perturbation = 0.05 * (1.0 + uniformity_factor * 0.5)
            
            # Reduce perturbation over iterations
            perturbation_reduction = 1.0 / (1.0 + iteration * 0.2)
            actual_perturbation = max_perturbation * perturbation_reduction
            
            # Apply perturbations
            perturbed = points + np.random.normal(0, actual_perturbation, points.shape)
            
            # Ensure bounds
            perturbed = np.clip(perturbed, 0.01, 0.99)
            return perturbed
        
        return points

    # Generate initial configuration
    initial_points = voronoi_based_initialization()
    
    # Multi-start optimization approach
    best_points = initial_points.copy()
    best_ratio = compute_min_max_ratio(initial_points)
    
    # Strategy 1: Differential Evolution for global search (multiple restarts)
    bounds = [(0.01, 0.99) for _ in range(32)]
    
    # Run multiple DE restarts with different seeds
    for restart in range(5):
        # Create variation of initial configuration
        restart_points = adaptive_perturbation(initial_points, restart)
        
        try:
            de_result = differential_evolution(
                objective_function,
                bounds,
                maxiter=50,
                popsize=12,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42 + restart,
                disp=False
            )
            
            de_ratio = -objective_function(de_result.x)
            if de_ratio > best_ratio:
                best_ratio = de_ratio
                best_points = de_result.x.reshape(-1, 2)
                best_points = np.clip(best_points, 0.01, 0.99)
                
                # Reapply boundary constraints
                best_points[0] = [0.01, 0.01]   # bottom-left
                best_points[3] = [0.99, 0.01]   # bottom-right
                best_points[12] = [0.01, 0.99]  # top-left
                best_points[15] = [0.99, 0.99]  # top-right
                
        except:
            continue

    # Strategy 2: Local optimization refinement
    try:
        refined_result = minimize(
            objective_function,
            best_points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-9}
        )

        final_points = refined_result.x.reshape(-1, 2)
        final_points = np.clip(final_points, 0.01, 0.99)
        
        # Reapply boundary constraints
        final_points[0] = [0.01, 0.01]
        final_points[3] = [0.99, 0.01]
        final_points[12] = [0.01, 0.99]
        final_points[15] = [0.99, 0.99]
        
        final_ratio = compute_min_max_ratio(final_points)
        if final_ratio > best_ratio:
            best_points = final_points
            
    except:
        pass

    # Strategy 3: Additional refinement with adaptive perturbations
    try:
        # Create multiple variants and optimize each
        for variant_iter in range(3):
            variant_points = adaptive_perturbation(best_points, variant_iter)
            
            variant_result = minimize(
                objective_function,
                variant_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-9}
            )
            
            variant_points_opt = variant_result.x.reshape(-1, 2)
            variant_points_opt = np.clip(variant_points_opt, 0.01, 0.99)
            
            # Reapply boundary constraints
            variant_points_opt[0] = [0.01, 0.01]
            variant_points_opt[3] = [0.99, 0.01]
            variant_points_opt[12] = [0.01, 0.99]
            variant_points_opt[15] = [0.99, 0.99]
            
            variant_ratio = compute_min_max_ratio(variant_points_opt)
            if variant_ratio > best_ratio:
                best_ratio = variant_ratio
                best_points = variant_points_opt
                
    except:
        pass

    # Final constraint enforcement
    best_points[0] = [0.01, 0.01]   # bottom-left
    best_points[3] = [0.99, 0.01]   # bottom-right
    best_points[12] = [0.01, 0.99]  # top-left
    best_points[15] = [0.99, 0.99]  # top-right
    
    return best_points

# EVOLVE-BLOCK-END