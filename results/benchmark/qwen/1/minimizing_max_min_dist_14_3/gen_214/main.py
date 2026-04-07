# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import SphericalVoronoi
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a spherical Voronoi evolution approach for improved convergence.
    """
    
    def normalize_to_unit_sphere(points):
        """Normalize points to lie on unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return points / norms

    def calculate_voronoi_uniformity(points):
        """Calculate how uniform the Voronoi cells are"""
        if len(points) < 2:
            return 0.0
        try:
            # Create spherical Voronoi diagram
            sv = SphericalVoronoi(points)
            # Calculate areas of all cells
            areas = sv.calculate_areas()
            # Return normalized standard deviation of areas
            if len(areas) > 0:
                mean_area = np.mean(areas)
                if mean_area > 0:
                    std_area = np.std(areas)
                    return 1.0 - (std_area / mean_area)
            return 0.0
        except:
            return 0.0

    def voronoi_objective(x):
        """Combined objective: maximize min/max ratio AND Voronoi uniformity"""
        points = x.reshape(-1, 3)
        points = normalize_to_unit_sphere(points)
        
        # Calculate pairwise distances
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        
        if len(distances[distances > 0]) == 0:
            return np.inf
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max <= 0:
            return np.inf
            
        ratio = d_min / d_max
        
        # Add Voronoi uniformity penalty (minimize negative)
        voronoi_uniformity = calculate_voronoi_uniformity(points)
        
        # Combined objective: prioritize ratio but penalize non-uniform Voronoi
        # We want to maximize ratio, so return negative value for minimization
        # The Voronoi component adds a small penalty for non-uniform distributions
        return -(ratio + 0.1 * voronoi_uniformity)

    def generate_fibonacci_points(n):
        """Generate points using Fibonacci spiral on sphere"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            theta = np.arccos(1 - 2*(i/(n-1)))
            phi = i * 2 * np.pi / golden_ratio
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            points.append([x, y, z])
        return np.array(points)

    def spherical_perturbation(points, delta=0.05):
        """Perturb points on sphere while maintaining unit sphere constraint"""
        # Generate random perturbations
        perturbations = np.random.normal(0, delta, points.shape)
        # Project back onto sphere
        new_points = points + perturbations
        return normalize_to_unit_sphere(new_points)

    def constraint_sphere(x):
        """Constraint function to ensure points stay on unit sphere"""
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return 1 - norms  # Should be >= 0

    # Main evolutionary optimization
    np.random.seed(42)
    
    # Initialize with multiple good starting configurations
    configs = []
    
    # Fibonacci points as baseline
    configs.append(generate_fibonacci_points(14))
    
    # Add multiple perturbed versions
    for i in range(5):
        np.random.seed(100 + i)
        configs.append(spherical_perturbation(generate_fibonacci_points(14), 0.03))
    
    # Use K-means clustering to identify promising regions
    # Generate a dense sample for analysis
    sample_points = generate_fibonacci_points(100)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(sample_points)
    
    # Add cluster centers as starting points
    for center in kmeans.cluster_centers_:
        configs.append(normalize_to_unit_sphere(center.reshape(1, 3).repeat(14, axis=0)))
    
    best_points = None
    best_score = -np.inf
    
    # Multi-stage evolutionary approach
    for stage in range(3):
        stage_configs = configs.copy()
        
        # Stage 1: Global exploration with larger perturbations
        if stage == 0:
            stage_delta = 0.1
            num_iterations = 20
        elif stage == 1:
            stage_delta = 0.05
            num_iterations = 30
        else:  # stage == 2
            stage_delta = 0.01
            num_iterations = 50
            
        # Each configuration gets optimized multiple times
        for config_idx, initial_config in enumerate(stage_configs):
            for iter_num in range(num_iterations):
                # Apply spherical perturbation to create new candidate
                candidate = spherical_perturbation(initial_config, stage_delta)
                
                # Optimize this candidate using local method
                try:
                    x0 = candidate.flatten()
                    cons = [{'type': 'ineq', 'fun': constraint_sphere}]
                    
                    result = minimize(voronoi_objective, x0, method='SLSQP', 
                                    constraints=cons, options={'ftol': 1e-8, 'maxiter': 500})
                    
                    if result.success:
                        optimized_points = result.x.reshape(-1, 3)
                        optimized_points = normalize_to_unit_sphere(optimized_points)
                        
                        # Calculate actual ratio
                        distances = cdist(optimized_points, optimized_points)
                        np.fill_diagonal(distances, np.inf)
                        
                        if len(distances[distances > 0]) > 0:
                            d_min = np.min(distances)
                            d_max = np.max(distances)
                            
                            if d_max > 0:
                                ratio = d_min / d_max
                                
                                if ratio > best_score:
                                    best_score = ratio
                                    best_points = optimized_points.copy()
                                    
                except Exception:
                    continue
                    
                # Occasionally apply a larger perturbation for exploration
                if iter_num % 10 == 0 and iter_num > 0:
                    new_candidate = spherical_perturbation(initial_config, stage_delta * 3)
                    configs.append(new_candidate)
    
    # If we didn't find anything, return a good Fibonacci configuration
    if best_points is None:
        return normalize_to_unit_sphere(generate_fibonacci_points(14))
    
    return best_points

# EVOLVE-BLOCK-END