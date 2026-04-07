# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import time

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

def compute_voronoi_centroids(points, bounds=(0, 1)):
    """
    Compute new point positions as centroids of Voronoi cells.
    """
    points = np.array(points)
    # Clip points to valid bounds to prevent Voronoi errors
    points = np.clip(points, bounds[0], bounds[1])
    
    try:
        vor = Voronoi(points)
        new_points = []
        
        # For each point, find its Voronoi cell centroid
        for i in range(len(points)):
            # Get vertices of the Voronoi cell for point i
            region_indices = np.where(np.array(vor.point_region) == i)[0]
            if len(region_indices) > 0:
                region_id = region_indices[0]
                vertices = vor.vertices[vor.regions[region_id]]
                if len(vertices) > 0:
                    # Compute centroid of polygon
                    vertices = np.array(vertices)
                    centroid = np.mean(vertices, axis=0)
                    # Check if centroid is inside bounds
                    if bounds[0] <= centroid[0] <= bounds[1] and bounds[0] <= centroid[1] <= bounds[1]:
                        new_points.append(centroid)
                    else:
                        # If not, keep the original point
                        new_points.append(points[i])
                else:
                    # No vertices, keep original point
                    new_points.append(points[i])
            else:
                # No region found, keep original point
                new_points.append(points[i])
        
        return np.array(new_points)
    except:
        # Fall back to original points if Voronoi computation fails
        return points.copy()

def voronoi_relaxation(points, max_iter=50, tolerance=1e-6):
    """
    Apply Voronoi relaxation to distribute points more evenly.
    """
    points = np.array(points)
    previous_points = None
    
    for i in range(max_iter):
        new_points = compute_voronoi_centroids(points)
        
        # Check for convergence
        if previous_points is not None:
            diff = np.mean(np.abs(new_points - previous_points))
            if diff < tolerance:
                break
                
        points = new_points
        previous_points = new_points.copy()
    
    return points

def compute_objective_gradient(points_flat, points_shape):
    """
    Compute gradient of the negative min/max ratio function.
    """
    points = points_flat.reshape(points_shape)
    
    # Simple finite difference approximation for gradient
    eps = 1e-8
    grad = np.zeros_like(points_flat)
    
    # For each point component, compute partial derivative
    for i in range(len(points_flat)):
        points_plus = points_flat.copy()
        points_minus = points_flat.copy()
        points_plus[i] += eps
        points_minus[i] -= eps
        
        # Compute difference of objective functions
        obj_plus = -PointConfiguration(points_plus.reshape(points_shape)).compute_min_max_ratio()
        obj_minus = -PointConfiguration(points_minus.reshape(points_shape)).compute_min_max_ratio()
        
        grad[i] = (obj_plus - obj_minus) / (2 * eps)
    
    return grad

def hybrid_voronoi_optimization(initial_points, max_evaluations=500):
    """
    Hybrid optimization combining Voronoi relaxation with gradient-based optimization.
    """
    # Start with Voronoi relaxation for good initial distribution
    relaxed_points = voronoi_relaxation(initial_points, max_iter=20)
    
    # Convert to flat array for optimization
    x0 = relaxed_points.flatten()
    bounds = [(0, 1) for _ in range(len(x0))]
    
    # Use L-BFGS-B for fine-tuning
    def objective(x_flat):
        points = x_flat.reshape(-1, 2)
        points = np.clip(points, 0, 1)
        ratio = PointConfiguration(points).compute_min_max_ratio()
        return -ratio  # Negative because we want to maximize
    
    def gradient(x_flat):
        points = x_flat.reshape(-1, 2)
        points = np.clip(points, 0, 1)
        return -compute_objective_gradient(x_flat, (-1, 2))  # Negative because we're minimizing
    
    # First, try L-BFGS-B with gradient
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            jac=gradient,
            options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': max_evaluations//2}
        )
        
        if result.success:
            final_points = result.x.reshape(-1, 2)
            final_points = np.clip(final_points, 0, 1)
            return final_points
    except:
        pass
    
    # Fallback to just L-BFGS-B without gradient
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': max_evaluations//2}
        )
        
        if result.success:
            final_points = result.x.reshape(-1, 2)
            final_points = np.clip(final_points, 0, 1)
            return final_points
    except:
        pass
    
    # Final fallback to just the relaxed points
    return relaxed_points

def generate_initial_strategies():
    """Generate multiple initial point configurations."""
    strategies = {}
    
    # Strategy 1: Grid-based (hexagonal-like)
    points = []
    rows = 4
    cols = 4
    spacing_x = 1.0 / (cols - 1)
    spacing_y = np.sqrt(3) / 2 / (rows - 1)
    
    for i in range(rows):
        for j in range(cols):
            x = j * spacing_x + (i % 2) * spacing_x / 2
            y = i * spacing_y
            points.append([x, y])
    
    strategies['hex_grid'] = np.array(points)
    
    # Strategy 2: Golden spiral
    indices = np.arange(16)
    golden_angle = 2.399963229728653
    angles = golden_angle * indices
    radii = np.log(indices + 1) / np.log(16)
    golden_spiral = np.column_stack([
        0.5 + 0.45 * radii * np.cos(angles),
        0.5 + 0.45 * radii * np.sin(angles)
    ])
    strategies['spiral'] = np.clip(golden_spiral, 0, 1)
    
    # Strategy 3: Perturbed grid
    np.random.seed(42)
    perturbed_hex = strategies['hex_grid'] + np.random.normal(0, 0.02, strategies['hex_grid'].shape)
    strategies['perturbed_grid'] = np.clip(perturbed_hex, 0, 1)
    
    # Strategy 4: Random with edge avoidance
    np.random.seed(123)
    random_points = np.random.rand(16, 2)
    strategies['random'] = np.clip(random_points, 0.05, 0.95)
    
    # Strategy 5: Structured random
    np.random.seed(999)
    structured_random = np.random.rand(16, 2) * 0.8 + 0.1
    strategies['structured'] = structured_random
    
    return strategies

def evaluate_all_strategies(strategies):
    """Evaluate all initial strategies and return the best one."""
    best_strategy = None
    best_ratio = 0

    for name, points in strategies.items():
        config = PointConfiguration(points)
        ratio = config.compute_min_max_ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_strategy = points.copy()

    return best_strategy, best_ratio

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    # Generate initial strategies
    strategies = generate_initial_strategies()
    
    # Find the best initial configuration
    best_initial, initial_ratio = evaluate_all_strategies(strategies)
    
    # Multi-start optimization with Voronoi relaxation hybrid
    best_points = best_initial.copy()
    best_ratio = initial_ratio
    
    # Try multiple variants with different randomness
    for restart in range(5):
        # Generate new variation of the initial points
        np.random.seed(restart + 1000)
        perturbed_points = best_initial.copy()
        noise_level = 0.03 + restart * 0.005  # Gradually increasing noise
        perturbed_points += np.random.normal(0, noise_level, best_initial.shape)
        perturbed_points = np.clip(perturbed_points, 0, 1)
        
        # Apply hybrid optimization
        optimized_points = hybrid_voronoi_optimization(perturbed_points, max_evaluations=300)
        optimized_ratio = PointConfiguration(optimized_points).compute_min_max_ratio()
        
        if optimized_ratio > best_ratio:
            best_ratio = optimized_ratio
            best_points = optimized_points.copy()
    
    # Final refinement
    final_points = hybrid_voronoi_optimization(best_points, max_evaluations=200)
    final_ratio = PointConfiguration(final_points).compute_min_max_ratio()
    
    # Additional final attempt with slightly different noise
    np.random.seed(9999)
    last_attempt = best_points + np.random.normal(0, 0.005, best_points.shape)
    last_attempt = np.clip(last_attempt, 0, 1)
    refined_final = hybrid_voronoi_optimization(last_attempt, max_evaluations=100)
    refined_ratio = PointConfiguration(refined_final).compute_min_max_ratio()
    
    if refined_ratio > final_ratio:
        return refined_final
    else:
        return final_points

# EVOLVE-BLOCK-END