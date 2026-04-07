# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import SphericalVoronoi, distance
from scipy.spatial.distance import pdist
import math
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0, 0.0

        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max == 0:
            return 0.0, 0.0
            
        return d_min / d_max, d_min

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

    def project_to_sphere(points):
        """Project points onto unit sphere."""
        norms = np.linalg.norm(points, axis=1)
        norms = np.where(norms == 0, 1.0, norms)
        return points / norms[:, np.newaxis]

    def analyze_voronoi_structure(points):
        """Analyze Voronoi structure to identify optimization opportunities."""
        try:
            # Create Spherical Voronoi diagram
            sv = SphericalVoronoi(points)
            
            # Get Voronoi cell areas and centroids
            cell_areas = sv.volume
            centroids = sv.vertices
            
            # Find points in dense regions (small Voronoi cells)
            mean_area = np.mean(cell_areas)
            dense_points = np.where(cell_areas < 0.1 * mean_area)[0]
            
            # Find points in sparse regions (large Voronoi cells)
            sparse_points = np.where(cell_areas > 1.5 * mean_area)[0]
            
            return {
                'cell_areas': cell_areas,
                'centroids': centroids,
                'dense_points': dense_points,
                'sparse_points': sparse_points
            }
        except:
            return None

    def select_point_for_perturbation(points, voronoi_analysis):
        """Select which point to perturb based on Voronoi analysis."""
        if voronoi_analysis is None or len(voronoi_analysis['dense_points']) == 0:
            # If no Voronoi info or no dense points, select randomly
            return np.random.randint(0, len(points))
        
        # Priority: perturb points in dense regions first
        dense_points = voronoi_analysis['dense_points']
        if len(dense_points) > 0 and np.random.random() < 0.7:
            # 70% chance to select from dense points
            return np.random.choice(dense_points)
        else:
            # Otherwise, select randomly from all points
            return np.random.randint(0, len(points))

    def compute_adaptive_perturbation(points, point_idx, voronoi_analysis):
        """Compute perturbation that improves the min/max ratio."""
        # Get neighbors for the selected point
        current_point = points[point_idx]
        
        # Calculate all distances to other points
        distances = np.array([np.linalg.norm(current_point - points[i]) for i in range(len(points)) if i != point_idx])
        
        if len(distances) == 0:
            return np.random.normal(0, 0.005, 3)
        
        # Find closest and furthest neighbors
        closest_idx = np.argmin(distances)
        furthest_idx = np.argmax(distances)
        
        # Compute ideal perturbation direction
        perturbation = np.zeros(3)
        
        # Push away from the furthest neighbor to reduce max distance
        if furthest_idx < len(points):
            direction = current_point - points[furthest_idx]
            dist = np.linalg.norm(direction)
            if dist > 0:
                perturbation -= 0.5 * direction / dist
        
        # Pull towards the closest neighbor to increase min distance
        if closest_idx < len(points):
            direction = points[closest_idx] - current_point
            dist = np.linalg.norm(direction)
            if dist > 0:
                perturbation += 0.3 * direction / dist
        
        # Add some random component for exploration
        random_component = np.random.normal(0, 0.003, 3)
        perturbation += random_component
        
        # Scale based on Voronoi information and current state
        base_scale = 0.005
        if voronoi_analysis is not None:
            # If point is in a very dense region, make larger moves
            if point_idx in voronoi_analysis['dense_points']:
                base_scale *= 1.5
        
        return perturbation * base_scale

    def guided_simulated_annealing(initial_points, max_iter=20000):
        """Optimize using guided simulated annealing based on Voronoi analysis."""
        points = initial_points.copy()
        current_ratio, _ = compute_min_max_ratio(points)
        
        best_points = points.copy()
        best_ratio = current_ratio
        
        # Cooling schedule with adaptive parameters
        temp = 1.0
        min_temp = 1e-8
        cooling_rate = 0.9995
        max_no_improvement = 2000
        no_improvement_counter = 0
        
        # Track recent improvements for adaptive cooling
        recent_improvements = []
        improvement_window = 100
        
        for iteration in range(max_iter):
            # Adaptive cooling based on recent progress
            if len(recent_improvements) >= improvement_window:
                recent_improvements.pop(0)
                
            if len(recent_improvements) > 0 and np.mean(recent_improvements) < 1e-6:
                cooling_rate = min(0.9999, cooling_rate * 1.01)  # Slow down cooling
            else:
                cooling_rate = max(0.999, cooling_rate * 0.9999)  # Normal cooling
            
            # Calculate temperature
            temp = max(min_temp, temp * cooling_rate)
            
            # Stop if temperature is too low
            if temp < min_temp:
                break
                
            # Analyze current Voronoi structure
            voronoi_analysis = analyze_voronoi_structure(points)
            
            # Select point to perturb
            point_to_perturb = select_point_for_perturbation(points, voronoi_analysis)
            
            # Compute adaptive perturbation
            perturbation = compute_adaptive_perturbation(points, point_to_perturb, voronoi_analysis)
            
            # Apply perturbation
            new_points = points.copy()
            new_points[point_to_perturb] += perturbation
            
            # Project back to sphere
            new_points = project_to_sphere(new_points)
            
            # Calculate new ratio
            new_ratio, _ = compute_min_max_ratio(new_points)
            
            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio:
                points = new_points
                current_ratio = new_ratio
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = new_points.copy()
                no_improvement_counter = 0
                recent_improvements.append(1)
            else:
                # Accept with probability based on temperature
                if np.random.random() < math.exp((new_ratio - current_ratio) / temp):
                    points = new_points
                    current_ratio = new_ratio
                    if new_ratio > best_ratio:
                        best_ratio = new_ratio
                        best_points = new_points.copy()
                    no_improvement_counter = 0
                    recent_improvements.append(1)
                else:
                    recent_improvements.append(0)
                    no_improvement_counter += 1
                    
            # Early stopping if no improvement
            if no_improvement_counter > max_no_improvement:
                break
                
        return best_points, best_ratio

    # Multi-start strategy with different initialization methods
    best_points = None
    best_ratio = 0.0
    
    # Strategy 1: Fibonacci sphere with various perturbations
    np.random.seed(42)
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        
        # Generate Fibonacci sphere points
        points = fibonacci_sphere(14)
        
        # Add random noise
        noise = np.random.normal(0, 0.01, points.shape)
        points += noise
        
        # Project to sphere
        points = project_to_sphere(points)
        
        # Optimize
        optimized_points, ratio = guided_simulated_annealing(points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()
    
    # Strategy 2: Perturbed regular configuration
    if best_points is None:
        np.random.seed(42)
        # Start with a regular distribution and add noise
        points = fibonacci_sphere(14)
        # Add more significant noise for diversity
        noise = np.random.normal(0, 0.02, points.shape)
        points += noise
        points = project_to_sphere(points)
        
        # Optimize
        optimized_points, ratio = guided_simulated_annealing(points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()
    
    # Strategy 3: Spherical Voronoi based initialization
    try:
        np.random.seed(42)
        # Generate random points on sphere
        random_points = np.random.randn(14, 3)
        random_points = project_to_sphere(random_points)
        
        # Create SphericalVoronoi and get vertices
        sv = SphericalVoronoi(random_points)
        voronoi_points = sv.vertices[:14]  # Take first 14 vertices
        voronoi_points = project_to_sphere(voronoi_points)
        
        # Optimize
        optimized_points, ratio = guided_simulated_annealing(voronoi_points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()
    except:
        pass
    
    # Final refinement with local search
    if best_points is not None:
        # Run a few iterations of local search to refine
        final_points = best_points.copy()
        
        # Simple local optimization: try small adjustments to each point
        for _ in range(100):
            for i in range(len(final_points)):
                # Try small perturbations
                original_point = final_points[i].copy()
                for _ in range(5):  # Try several perturbations
                    noise = np.random.normal(0, 0.001, 3)
                    test_point = original_point + noise
                    test_point = project_to_sphere(test_point.reshape(1, 3))[0]
                    
                    # Test this move
                    test_points = final_points.copy()
                    test_points[i] = test_point
                    test_ratio, _ = compute_min_max_ratio(test_points)
                    
                    if test_ratio > compute_min_max_ratio(final_points)[0]:
                        final_points = test_points.copy()
                        break
        
        # Final verification
        _, final_ratio = compute_min_max_ratio(final_points)
        if final_ratio > best_ratio:
            best_points = final_points.copy()
            best_ratio = final_ratio

    # Fallback to ensure valid result
    if best_points is None:
        points = fibonacci_sphere(14)
        points = project_to_sphere(points)
        best_points = points

    return best_points

# EVOLVE-BLOCK-END