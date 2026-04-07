# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution, minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    class PointConfiguration:
        """Represents a configuration of points and provides utility methods."""
        
        def __init__(self, points):
            self.points = np.array(points)
            self.n_points = len(points)
            
        def compute_min_max_ratio(self):
            """Compute the ratio of minimum to maximum distance between all point pairs."""
            if self.n_points < 2:
                return 0
                
            # Compute pairwise distances with enhanced numerical stability
            distance_matrix = squareform(pdist(self.points))
            
            # Set diagonal to infinity to exclude self-distances
            np.fill_diagonal(distance_matrix, np.inf)
            
            # Get all finite distances (excluding NaN and inf values)
            finite_distances = distance_matrix[np.isfinite(distance_matrix)]
            
            if len(finite_distances) == 0:
                return 0
                
            # Get min and max distances
            dmin = np.min(finite_distances)
            dmax = np.max(finite_distances)
            
            # Avoid division by zero
            if dmax == 0:
                return 0
                
            return dmin / dmax
            
        def compute_distance_matrix(self):
            """Compute full pairwise distance matrix."""
            return squareform(pdist(self.points))
            
        def get_clipped_points(self, lower=0, upper=1):
            """Get points clipped to specified bounds."""
            return np.clip(self.points, lower, upper)
            
        def copy(self):
            """Create a copy of this configuration."""
            return PointConfiguration(self.points.copy())
    
    def generate_golden_spiral_points(n_points=16):
        """Generate points arranged in a golden spiral pattern."""
        indices = np.arange(n_points)
        golden_angle = 2.399963229728653
        angles = golden_angle * indices
        radii = np.log(indices + 1) / np.log(n_points)
        points = np.column_stack([
            0.5 + 0.45 * radii * np.cos(angles),
            0.5 + 0.45 * radii * np.sin(angles)
        ])
        return np.clip(points, 0, 1)
    
    def generate_hexagonal_grid():
        """Generate a hexagonal grid arrangement."""
        points = []
        rows = 4
        cols = 4
        
        # Hexagonal packing parameters
        spacing_x = 1.0 / (cols - 1)
        spacing_y = np.sqrt(3) / 2 / (rows - 1)  # Height of equilateral triangle
        
        for i in range(rows):
            for j in range(cols):
                x = j * spacing_x + (i % 2) * spacing_x / 2
                y = i * spacing_y
                points.append([x, y])
                
        return np.array(points)
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0

        # Compute pairwise distances with numerical stability
        distances = pdist(points)

        # Remove any near-zero distances that could cause numerical issues
        distances = distances[distances > 1e-12]

        if len(distances) == 0:
            return 0

        # Get min and max distances
        dmin = np.min(distances)
        dmax = np.max(distances)

        # Avoid division by zero
        if dmax == 0:
            return 0

        return dmin / dmax
    
    def objective_function(x_flat, points_to_keep=None):
        """Objective function to maximize (negative because we minimize)."""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        
        # Ensure points are within bounds [0,1]
        points = np.clip(points, 0, 1)
        
        # If specific points to keep, restore them
        if points_to_keep is not None:
            # For now, we'll just return the ratio of the full configuration
            pass
        
        # Compute ratio
        ratio = compute_min_max_ratio(points)
        
        # Return negative because we want to maximize
        return -ratio
    
    def progressive_construction_approach():
        """Construct points with geometric progression and adaptive refinement."""
        # Start with a golden spiral for good initial distribution
        base_points = generate_golden_spiral_points(16)
        
        # Try several progressive construction strategies
        best_points = base_points.copy()
        best_ratio = compute_min_max_ratio(base_points)
        
        # Strategy 1: Refine the golden spiral with local optimization
        try:
            x0 = base_points.flatten()
            bounds = [(0, 1) for _ in range(32)]
            
            # Differential evolution first
            de_result = differential_evolution(
                objective_function,
                bounds,
                maxiter=50,
                popsize=15,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False,
                tol=1e-8
            )
            
            if de_result.success:
                # Local refinement
                refined_result = minimize(
                    objective_function,
                    de_result.x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'ftol': 1e-12, 'gtol': 1e-12},
                    tol=1e-12
                )
                
                if refined_result.success:
                    refined_points = refined_result.x.reshape(-1, 2)
                    refined_points = np.clip(refined_points, 0, 1)
                    refined_ratio = compute_min_max_ratio(refined_points)
                    
                    if refined_ratio > best_ratio:
                        best_ratio = refined_ratio
                        best_points = refined_points.copy()
                        
        except Exception:
            pass
        
        return best_points
    
    def hybrid_refinement_with_constraints():
        """Use hybrid approach with constraint awareness."""
        # Start with a good base configuration
        initial_points = generate_golden_spiral_points(16)
        
        # Multi-start approach with different seeds
        best_points = initial_points.copy()
        best_ratio = compute_min_max_ratio(initial_points)
        
        # Run multiple optimization attempts with different random seeds
        for seed_val in [123, 456, 789, 987, 654]:
            np.random.seed(seed_val)
            
            # Slightly perturb the initial points
            perturbed = initial_points + np.random.normal(0, 0.01, initial_points.shape)
            perturbed = np.clip(perturbed, 0, 1)
            
            # Try a two-phase optimization
            try:
                x0 = perturbed.flatten()
                bounds = [(0, 1) for _ in range(32)]
                
                # Global optimization
                de_result = differential_evolution(
                    objective_function,
                    bounds,
                    maxiter=30,
                    popsize=10,
                    mutation=(0.5, 1),
                    recombination=0.7,
                    seed=seed_val,
                    disp=False,
                    tol=1e-8
                )
                
                if de_result.success:
                    # Local refinement
                    refined_result = minimize(
                        objective_function,
                        de_result.x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'ftol': 1e-12, 'gtol': 1e-12},
                        tol=1e-12
                    )
                    
                    if refined_result.success:
                        refined_points = refined_result.x.reshape(-1, 2)
                        refined_points = np.clip(refined_points, 0, 1)
                        refined_ratio = compute_min_max_ratio(refined_points)
                        
                        if refined_ratio > best_ratio:
                            best_ratio = refined_ratio
                            best_points = refined_points.copy()
                            
            except Exception:
                continue  # Skip this attempt if optimization fails
        
        return best_points
    
    # Main optimization logic
    # Try progressive construction approach
    points1 = progressive_construction_approach()
    ratio1 = compute_min_max_ratio(points1)
    
    # Try hybrid refinement approach
    points2 = hybrid_refinement_with_constraints()
    ratio2 = compute_min_max_ratio(points2)
    
    # Return the better of the two approaches
    if ratio1 > ratio2:
        return points1
    else:
        return points2

# EVOLVE-BLOCK-END