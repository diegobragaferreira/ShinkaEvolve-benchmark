# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.optimize import differential_evolution, minimize
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Combines Voronoi relaxation with differential evolution optimization for optimal results.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum distance with numerical stability."""
        if len(points) < 2:
            return 0.0

        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        if max_dist == 0:
            return 0.0

        return min_dist / max_dist

    def initialize_structured_points():
        """Initialize points using structured patterns for better starting configuration."""
        np.random.seed(42)
        
        # Create a 4x4 grid pattern with perturbations
        grid_size = 4
        x = np.linspace(0.1, 0.9, grid_size)
        y = np.linspace(0.1, 0.9, grid_size)
        
        # Generate grid points
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Add small random perturbations to break symmetry
        noise = np.random.normal(0, 0.01, points.shape)
        points += noise
        
        # Clip to ensure points stay within bounds
        points = np.clip(points, 0, 1)
        
        return points

    def voronoi_relaxation(points, max_iterations=100, tolerance=1e-6):
        """Perform Voronoi relaxation to improve point distribution."""
        current_points = points.copy()
        best_ratio = calculate_min_max_ratio(current_points)
        best_points = current_points.copy()
        
        for iteration in range(max_iterations):
            try:
                # Compute Voronoi diagram
                vor = Voronoi(current_points)
                
                # Calculate new positions as centroids of Voronoi cells
                new_points = np.zeros_like(current_points)
                converged = True
                
                # Process each point
                for i in range(len(current_points)):
                    # Get vertices of Voronoi cell for point i
                    region = vor.regions[vor.point_region[i]]
                    
                    if -1 in region or len(region) < 3:
                        # Handle unbounded regions (use current position with slight adjustment)
                        new_points[i] = current_points[i] + np.random.normal(0, 0.001, 2)
                        continue
                        
                    # Extract vertices of the Voronoi cell
                    vertices = np.array([vor.vertices[j] for j in region if j >= 0])
                    
                    if len(vertices) < 3:
                        # Not enough vertices, use current position
                        new_points[i] = current_points[i]
                        continue
                        
                    # Compute centroid of polygon (Voronoi cell)
                    centroid = np.mean(vertices, axis=0)
                    
                    # Apply boundary constraints
                    centroid = np.clip(centroid, 0, 1)
                    
                    # Update point position
                    new_points[i] = centroid
                    
                    # Check for convergence
                    if np.linalg.norm(new_points[i] - current_points[i]) > tolerance:
                        converged = False
                
                # Apply cooling schedule for better convergence
                cooling_factor = 0.95 ** iteration
                current_points = current_points + cooling_factor * (new_points - current_points)
                
                # Ensure points stay within bounds
                current_points = np.clip(current_points, 0, 1)
                
                # Track best solution
                if iteration % 10 == 0:
                    ratio = calculate_min_max_ratio(current_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = current_points.copy()
                
                # Early stopping if converged
                if converged:
                    break
                    
            except Exception as e:
                # Fallback to simple perturbation
                current_points += np.random.normal(0, 0.001, current_points.shape)
                current_points = np.clip(current_points, 0, 1)
        
        return best_points

    def objective(x):
        """Objective function to minimize (negative ratio to maximize ratio)."""
        # Reshape into points
        points = x.reshape(-1, 2)

        # Calculate pairwise distances using squareform for numerical stability
        distances = squareform(pdist(points))

        # Set diagonal to large value to ignore self-distances
        np.fill_diagonal(distances, np.inf)

        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Avoid division by zero
        if d_max == 0:
            return -1e10

        # Return negative ratio to minimize (we want to maximize ratio)
        return -d_min / d_max

    # Set up bounds (0 to 1 for each coordinate)
    bounds = [(0, 1)] * 32

    # Method 1: Voronoi relaxation + differential evolution
    try:
        # Initialize with structured points
        initial_points = initialize_structured_points()
        
        # Apply Voronoi relaxation for global optimization
        relaxed_points = voronoi_relaxation(initial_points, max_iterations=50)
        
        # Fine-tune with differential evolution
        result_de = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=150,
            popsize=25,
            atol=1e-10,
            rtol=1e-10,
            mutation=(0.8, 1.0),
            recombination=0.9
        )
        
        if result_de.success:
            # Apply local refinement with L-BFGS-B
            refined = minimize(
                objective,
                result_de.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            
            if refined.success:
                final_points = refined.x.reshape(-1, 2)
                final_points = np.clip(final_points, 0, 1)
                ratio = calculate_min_max_ratio(final_points)
                
                # Return the better of the two approaches
                if ratio > calculate_min_max_ratio(relaxed_points):
                    return final_points
                else:
                    return relaxed_points
    except Exception:
        pass

    # Method 2: Direct optimization with multiple strategies
    try:
        # Try multiple optimization strategies
        strategies = [
            {
                'method': 'Differential Evolution',
                'params': {
                    'maxiter': 100,
                    'popsize': 20,
                    'mutation': (0.7, 1.0),
                    'recombination': 0.7
                }
            },
            {
                'method': 'Differential Evolution',
                'params': {
                    'maxiter': 120,
                    'popsize': 25,
                    'mutation': (0.8, 1.0),
                    'recombination': 0.9
                }
            }
        ]
        
        best_points = None
        best_ratio = -np.inf
        
        for strategy in strategies:
            try:
                # Initialize with structured points
                x0 = initialize_structured_points().flatten()
                
                # Run differential evolution
                result = differential_evolution(
                    objective,
                    bounds,
                    x0=x0,
                    seed=42,
                    maxiter=strategy['params']['maxiter'],
                    popsize=strategy['params']['popsize'],
                    mutation=strategy['params']['mutation'],
                    recombination=strategy['params']['recombination'],
                    atol=1e-12,
                    rtol=1e-12
                )
                
                if result.success:
                    # Local refinement
                    refined = minimize(
                        objective,
                        result.x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
                    )
                    
                    if refined.success:
                        final_points = refined.x.reshape(-1, 2)
                        final_points = np.clip(final_points, 0, 1)
                        ratio = calculate_min_max_ratio(final_points)
                        
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()
                            
            except Exception:
                continue
        
        if best_points is not None:
            return best_points
            
    except Exception:
        pass

    # Fallback to basic structured initialization
    points = initialize_structured_points()
    return points

# EVOLVE-BLOCK-END