# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import SphericalVoronoi
from scipy.optimize import minimize
import math
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Implements a hybrid approach combining Fibonacci initialization, multi-start simulated annealing,
    and advanced refinement techniques for superior convergence.
    """

    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return 0.0
        return d_min / d_max

    def fibonacci_sphere(samples=14):
        """Generate points distributed evenly on a sphere using Fibonacci method"""
        points = []
        phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
        
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            
            points.append([x, y, z])
            
        return np.array(points)

    def voronoi_entropy_score(points):
        """
        Calculate entropy-based score of Voronoi cell distribution.
        High entropy indicates more uniform cell distribution.
        """
        try:
            sv = SphericalVoronoi(points)
            areas = sv.calculate_areas()
            # Normalize areas
            areas = areas / np.sum(areas)
            # Entropy calculation
            entropy = -np.sum(areas * np.log(areas + 1e-10))
            return entropy
        except:
            return 0.0

    def project_to_unit_sphere(points):
        """Project points to unit sphere ensuring no zero-norm vectors"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return points / norms

    def perturb_point_smart(points, temp, current_ratio):
        """
        Smart perturbation that targets points based on distance distribution
        and respects spherical constraint.
        """
        new_points = points.copy()
        
        # Analyze current distribution to decide which point to perturb
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        
        # Prefer perturbing points that are near the minimum distance
        if np.random.rand() < 0.7:  # 70% chance to target close points
            target_idx = np.argmin(min_distances)
        else:
            target_idx = np.random.randint(len(points))
        
        # Generate small random perturbation
        delta = np.random.normal(0, 0.01 * temp, 3)
        
        # Add perturbation to selected point
        new_points[target_idx] += delta
        
        # Project back to unit sphere
        new_points = project_to_unit_sphere(new_points)
        
        return new_points

    def adaptive_cooling(initial_temp, iteration, max_iterations, ratio_history):
        """
        Adaptive cooling schedule that adjusts based on convergence
        """
        base_cooling = 0.9995
        
        if len(ratio_history) > 10:
            recent_improvement = ratio_history[-1] - ratio_history[-10]
            if recent_improvement < 1e-8:
                return base_cooling * 1.05
            elif recent_improvement > 1e-6:
                return base_cooling * 0.95
                
        return base_cooling

    def multi_start_optimization():
        """Multi-start optimization with multiple initialization strategies"""
        best_points = None
        best_ratio = -np.inf
        best_min_dist = 0
        best_max_dist = 0
        
        # Try multiple initialization strategies
        num_starts = 10
        for start_idx in range(num_starts):
            # Alternate between Fibonacci and random initialization
            if start_idx % 2 == 0:
                # Fibonacci sphere initialization
                np.random.seed(start_idx)
                points = fibonacci_sphere(14)
            else:
                # Random initialization with normalization
                np.random.seed(start_idx)
                points = np.random.randn(14, 3)
                points = project_to_unit_sphere(points)
            
            # Optimization parameters
            max_iterations = 100000
            initial_temperature = 1.0
            cooling_rate = 0.9995
            min_temperature = 0.0001
            
            # Track best solution
            current_best_points = points.copy()
            current_best_min_dist, current_best_max_dist, current_best_ratio = calculate_min_max_ratio(points)
            
            # Current state
            current_points = points.copy()
            current_min_dist, current_max_dist, current_ratio = current_best_ratio, current_best_max_dist, current_best_ratio
            
            # Track ratio history for adaptive cooling
            ratio_history = [current_ratio]
            
            # Simulated Annealing
            temp = initial_temperature
            last_improvement_iter = 0
            
            for iteration in range(max_iterations):
                # Perturb the current solution
                new_points = perturb_point_smart(current_points, temp, current_ratio)
                
                # Compute new ratio
                new_min_dist, new_max_dist, new_ratio = calculate_min_max_ratio(new_points)
                
                # Accept or reject the new solution using Metropolis criterion
                if new_ratio > current_ratio:
                    # Always accept better solutions
                    current_points = new_points
                    current_ratio = new_ratio
                    current_min_dist = new_min_dist
                    current_max_dist = new_max_dist
                    
                    # Update best solution if this is better
                    if new_ratio > current_best_ratio:
                        current_best_points = new_points.copy()
                        current_best_ratio = new_ratio
                        current_best_min_dist = new_min_dist
                        current_best_max_dist = new_max_dist
                        last_improvement_iter = iteration
                        ratio_history.append(new_ratio)
                else:
                    # Accept worse solutions with probability based on temperature
                    if temp > 0:  # Avoid division by zero
                        acceptance_prob = np.exp((new_ratio - current_ratio) / temp)
                        if np.random.rand() < acceptance_prob:
                            current_points = new_points
                            current_ratio = new_ratio
                            current_min_dist = new_min_dist
                            current_max_dist = new_max_dist
                            ratio_history.append(new_ratio)
                
                # Apply adaptive cooling
                temp = max(temp * adaptive_cooling(initial_temperature, iteration, max_iterations, ratio_history), min_temperature)
                
                # Early stopping if no improvement in a long time
                if iteration - last_improvement_iter > 20000:
                    break
            
            # Update global best if this run was better
            if current_best_ratio > best_ratio:
                best_ratio = current_best_ratio
                best_points = current_best_points.copy()
                best_min_dist = current_best_min_dist
                best_max_dist = current_best_max_dist
        
        return best_points

    def advanced_gradient_refinement(points):
        """Multi-stage gradient-based refinement with adaptive tolerances."""
        def objective(x_flat):
            points_local = x_flat.reshape(-1, 3)
            # Keep points on unit sphere constraint
            points_local = project_to_unit_sphere(points_local)
            return -calculate_min_max_ratio(points_local)
        
        try:
            refined_points = points.copy()
            
            # Stage 1: Coarse refinement with relaxed tolerances
            result1 = minimize(objective, refined_points.flatten(), method='L-BFGS-B',
                             options={'maxiter': 100, 'ftol': 1e-4, 'gtol': 1e-4})
            refined_points = result1.x.reshape(-1, 3)
            refined_points = project_to_unit_sphere(refined_points)
            
            # Stage 2: Medium refinement with tighter tolerances
            result2 = minimize(objective, refined_points.flatten(), method='L-BFGS-B',
                             options={'maxiter': 150, 'ftol': 1e-6, 'gtol': 1e-6})
            refined_points = result2.x.reshape(-1, 3)
            refined_points = project_to_unit_sphere(refined_points)
            
            # Stage 3: Fine refinement with extremely tight tolerances
            result3 = minimize(objective, refined_points.flatten(), method='L-BFGS-B',
                             options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
            refined_points = result3.x.reshape(-1, 3)
            refined_points = project_to_unit_sphere(refined_points)
            
            return refined_points
        except Exception as e:
            # Fallback to iterative refinement if optimization fails
            current_points = points.copy()
            best_ratio = calculate_min_max_ratio(current_points)
            
            # Iterative improvement with sphere constraint
            for iteration in range(500):
                neighbor_points = current_points.copy()
                point_idx = np.random.randint(len(neighbor_points))
                
                # Small perturbation
                perturbation = np.random.normal(0, 0.0005, 3)
                neighbor_points[point_idx] += perturbation
                
                # Project back to sphere
                neighbor_points = project_to_unit_sphere(neighbor_points)
                
                new_ratio = calculate_min_max_ratio(neighbor_points)
                
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    current_points = neighbor_points.copy()
            
            return current_points

    def project_to_unit_cube(points):
        """Project points to unit cube [0,1]^3"""
        # Find min/max along each axis
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # Handle case where there's no variation
        ranges = max_coords - min_coords
        if np.any(ranges == 0):
            # If any dimension has no variation, return points centered at 0.5
            return np.full_like(points, 0.5)
        
        # Scale to [0,1] range
        normalized = (points - min_coords) / ranges
        
        # Ensure they're clipped to [0,1]
        return np.clip(normalized, 0, 1)

    # Main execution flow
    np.random.seed(42)
    
    # Phase 1: Multi-start optimization with diverse initialization
    optimized_points = multi_start_optimization()
    
    # Phase 2: Advanced gradient refinement
    final_points = advanced_gradient_refinement(optimized_points)
    
    # Phase 3: Final projection to unit cube [0,1]^3
    points_in_cube = project_to_unit_cube(final_points)
    
    return points_in_cube

# EVOLVE-BLOCK-END