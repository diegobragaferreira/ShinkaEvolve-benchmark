# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import SphericalVoronoi, distance
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0

        # Compute pairwise distances efficiently
        distances = pdist(points)

        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return 0.0

        return min_dist / max_dist
    
    def compute_ratio_gradient(points):
        """Compute gradient of the ratio function for optimization."""
        n = len(points)
        if n < 2:
            return np.zeros_like(points)
        
        # Compute pairwise distances
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return np.zeros_like(points)
        
        # Compute gradient with respect to the ratio
        ratios = distances / max_dist
        return ratios
    
    def project_to_sphere(points):
        """Project points onto unit sphere while maintaining relative positions."""
        norms = np.linalg.norm(points, axis=1)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return points / norms[:, np.newaxis] * 0.99
    
    def fibonacci_sphere(n):
        """Generate n points distributed approximately uniformly on a sphere."""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        for i in range(n):
            z = 1 - (i / (n - 1)) * 2  # z goes from 1 to -1
            radius = np.sqrt(1 - z*z)

            theta = np.arctan2(np.sin(i * 2 * np.pi / phi), np.cos(i * 2 * np.pi / phi))
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)
    
    def create_voronoi_force_adjustment(points, alpha=0.01, beta=0.005):
        """
        Compute force-based adjustment using SphericalVoronoi.
        This creates a repulsion/attraction force field based on Voronoi cells.
        """
        if len(points) < 2:
            return points
            
        try:
            # Create SphericalVoronoi diagram
            sv = SphericalVoronoi(points, radius=1.0)
            
            # Get Voronoi vertices
            vertices = sv.vertices
            
            # For each point, compute adjustments based on Voronoi structure
            adjusted_points = points.copy()
            
            # Compute forces from Voronoi geometry
            for i in range(len(points)):
                # Find Voronoi cell vertices associated with this point
                # In practice, we'll compute simpler distance-based forces
                
                # Compute distance to all other points
                distances = np.linalg.norm(points[i] - points, axis=1)
                distances[i] = np.inf  # Ignore self-distance
                
                # Find nearest and furthest points
                nearest_idx = np.argmin(distances)
                furthest_idx = np.argmax(distances)
                
                # Apply force to balance distances
                # Move towards nearest to increase min distance
                if distances[nearest_idx] > 0:
                    direction = points[nearest_idx] - points[i]
                    force_magnitude = alpha * (1.0 / distances[nearest_idx]) 
                    adjusted_points[i] += force_magnitude * direction / np.linalg.norm(direction)
                
                # Move away from furthest to decrease max distance
                if distances[furthest_idx] > 0:
                    direction = points[i] - points[furthest_idx]
                    force_magnitude = beta * (1.0 / distances[furthest_idx])
                    adjusted_points[i] -= force_magnitude * direction / np.linalg.norm(direction)
                    
            return adjusted_points
            
        except:
            # Fallback to simple distance-based adjustment
            return points
    
    def adaptive_local_optimization(points, max_iter=100):
        """Apply adaptive local optimization to improve the current configuration."""
        current_points = points.copy()
        best_points = points.copy()
        best_ratio = compute_min_max_ratio(points)
        
        for iteration in range(max_iter):
            # Compute current ratio
            current_ratio = compute_min_max_ratio(current_points)
            
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = current_points.copy()
            
            # Apply Voronoi-based adjustment
            adjusted_points = create_voronoi_force_adjustment(current_points)
            
            # Apply some random perturbations to maintain diversity
            if iteration % 10 == 0:
                for i in range(len(adjusted_points)):
                    # Small random perturbation
                    perturbation = np.random.normal(0, 0.001, 3)
                    adjusted_points[i] += perturbation
                    
            # Ensure points stay on sphere
            norms = np.linalg.norm(adjusted_points, axis=1)
            adjusted_points = adjusted_points / norms[:, np.newaxis] * 0.99
            
            # Check if we should accept this adjustment
            test_ratio = compute_min_max_ratio(adjusted_points)
            
            if test_ratio > current_ratio:
                current_points = adjusted_points.copy()
            elif np.random.random() < 0.1:  # Accept some bad moves occasionally
                current_points = adjusted_points.copy()
                
        return best_points, best_ratio
    
    def hybrid_optimization(initial_points, max_iterations=3000):
        """Combine multiple optimization strategies."""
        current_points = initial_points.copy()
        best_points = initial_points.copy()
        best_ratio = compute_min_max_ratio(initial_points)
        
        # Track improvement history
        improvement_history = []
        patience = 0
        max_patience = 500
        
        for iteration in range(max_iterations):
            # Adaptive cooling schedule
            cooling_rate = max(0.99, 1.0 - iteration / (max_iterations * 2.0))
            
            # Sometimes do large jump for exploration
            if iteration % 100 == 0 and iteration > 0:
                # Random perturbation
                noise = np.random.normal(0, 0.01, current_points.shape)
                current_points += noise
                norms = np.linalg.norm(current_points, axis=1)
                current_points = current_points / norms[:, np.newaxis] * 0.99
            
            # Apply force-based adjustment
            adjusted_points = create_voronoi_force_adjustment(current_points)
            
            # Apply local optimization periodically
            if iteration % 50 == 0:
                optimized_points, optimized_ratio = adaptive_local_optimization(adjusted_points, 50)
                if optimized_ratio > compute_min_max_ratio(adjusted_points):
                    adjusted_points = optimized_points
                    current_points = adjusted_points.copy()
            
            # Accept or reject with probability based on improvement
            test_ratio = compute_min_max_ratio(adjusted_points)
            
            if test_ratio > best_ratio:
                best_ratio = test_ratio
                best_points = adjusted_points.copy()
                improvement_history.append(1)
                patience = 0
            elif np.random.random() < np.exp((test_ratio - best_ratio) * 0.5):
                current_points = adjusted_points.copy()
                improvement_history.append(1)
                patience = 0
            else:
                improvement_history.append(0)
                patience += 1
                
            # Early stopping if no improvement for too long
            if len(improvement_history) > 100:
                if np.sum(improvement_history[-100:]) < 5:
                    break
                    
            # Reduce cooling rate after significant improvement
            if patience > 100:
                cooling_rate *= 0.99
                
        return best_points, best_ratio

    # Multiple initialization strategies
    best_points = None
    best_ratio = 0.0
    
    # Strategy 1: Fibonacci Sphere Initialization
    try:
        fib_points = fibonacci_sphere(14)
        # Add small noise to break symmetry
        noise = np.random.normal(0, 0.01, fib_points.shape)
        fib_points += noise
        norms = np.linalg.norm(fib_points, axis=1)
        fib_points = fib_points / norms[:, np.newaxis] * 0.99
        
        # Optimize the Fibonacci configuration
        optimized_points, ratio = hybrid_optimization(fib_points, 2000)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()
    except:
        pass
    
    # Strategy 2: Multiple random restarts with different perturbations
    for seed in [1, 2, 3, 4, 5]:
        try:
            np.random.seed(seed)
            # Start with uniform random points
            random_points = np.random.uniform(-1, 1, (14, 3))
            norms = np.linalg.norm(random_points, axis=1)
            random_points = random_points / norms[:, np.newaxis] * 0.99
            
            # Perturb the points
            perturbation = np.random.normal(0, 0.02, random_points.shape)
            perturbed_points = random_points + perturbation
            norms = np.linalg.norm(perturbed_points, axis=1)
            perturbed_points = perturbed_points / norms[:, np.newaxis] * 0.99
            
            # Optimize this configuration
            optimized_points, ratio = hybrid_optimization(perturbed_points, 1500)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
        except:
            continue
            
    # Strategy 3: Refinement of best existing result
    if best_points is not None:
        try:
            # Apply more intensive optimization
            refined_points, ratio = hybrid_optimization(best_points, 1000)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined_points.copy()
        except:
            pass
    
    # Final fallback
    if best_points is None:
        # Use fibonacci with some additional refinement
        np.random.seed(42)
        points = fibonacci_sphere(14)
        noise = np.random.normal(0, 0.015, points.shape)
        points += noise
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis] * 0.99
        best_points = points

    return best_points

# EVOLVE-BLOCK-END