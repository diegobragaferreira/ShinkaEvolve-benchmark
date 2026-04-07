# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import math

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Implements a novel spherical discrete optimization approach using distance-weighted gradients
    and multi-scale refinement for superior convergence.
    """
    
    def cartesian_to_spherical(xyz):
        """Convert Cartesian coordinates to spherical coordinates (r, theta, phi)"""
        r = np.linalg.norm(xyz, axis=-1, keepdims=True)
        theta = np.arccos(xyz[..., 2] / (r + 1e-12))  # Avoid division by zero
        phi = np.arctan2(xyz[..., 1], xyz[..., 0])
        return np.stack([r, theta, phi], axis=-1)
    
    def spherical_to_cartesian(rtp):
        """Convert spherical coordinates (r, theta, phi) to Cartesian coordinates"""
        r, theta, phi = rtp[..., 0], rtp[..., 1], rtp[..., 2]
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.stack([x, y, z], axis=-1)
    
    def compute_min_max_ratio(points):
        """Compute the minimum and maximum distances between all pairs of points"""
        if len(points) < 2:
            return 0.0, 0.0, 0.0
        
        # Compute pairwise distances
        distances = cdist(points, points)
        
        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(distances, np.inf)
        
        # Find min and max distances
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        
        # Avoid division by zero
        if max_distance == 0:
            ratio = 0.0
        else:
            ratio = min_distance / max_distance
        
        return min_distance, max_distance, ratio
    
    def distance_weighted_gradient(points, ratio):
        """Compute a gradient that weights updates based on how much they can improve the ratio"""
        n = len(points)
        if n < 2:
            return np.zeros_like(points)
        
        # Compute distance matrix
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        
        # Gradient computation
        gradient = np.zeros_like(points)
        
        # For each point, compute contribution to improvement
        for i in range(n):
            # Find nearest neighbors 
            nearest_indices = np.argsort(distances[i])[:min(5, n-1)]
            for j in nearest_indices:
                if i != j:
                    # Direction vector from point i to point j
                    direction = points[j] - points[i]
                    distance_ij = distances[i, j]
                    
                    # Weight by inverse of distance - closer points have more impact
                    weight = 1.0 / (distance_ij + 1e-10)
                    
                    # Add to gradient (negative because we want to minimize distance differences)
                    gradient[i] += weight * direction / (np.linalg.norm(direction) + 1e-10)
        
        # Normalize gradient
        grad_norms = np.linalg.norm(gradient, axis=1, keepdims=True)
        grad_norms = np.where(grad_norms == 0, 1, grad_norms)
        gradient = gradient / grad_norms
        
        return gradient
    
    def configuration_entropy(points):
        """Calculate entropy of the point distribution to encourage uniformity"""
        if len(points) < 2:
            return 0.0
            
        # Compute pairwise distances
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        
        # Compute mean distance
        mean_dist = np.mean(distances[distances != np.inf])
        
        # Compute variance of distances (lower variance = more uniform)
        var_dist = np.var(distances[distances != np.inf])
        
        # Entropy-like measure (lower variance = higher entropy for uniform distribution)
        if mean_dist > 0:
            entropy = 1.0 / (var_dist + 1e-10)  # Inverse variance as entropy measure
        else:
            entropy = 0.0
            
        return entropy
    
    def objective_function(points_flat, points_count=14):
        """Objective function that considers both ratio and uniformity"""
        # Reshape flat array to points
        points = points_flat.reshape(points_count, 3)
        
        # Ensure points are on unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        points = points / norms
        
        # Compute basic metrics
        min_dist, max_dist, ratio = compute_min_max_ratio(points)
        
        # Compute uniformity metric
        entropy = configuration_entropy(points)
        
        # Combined objective (minimize negative ratio, maximize entropy)
        # The entropy component penalizes non-uniform distributions
        combined_objective = -ratio + 0.05 * entropy
        
        return combined_objective
    
    def perturb_points_adaptive(points, iteration):
        """Adaptive perturbation that changes based on optimization stage"""
        # Create a copy of the points
        new_points = points.copy()
        
        # Different perturbation strategies based on iteration count
        if iteration < 1000:
            # Early stage: larger perturbations for exploration
            perturbation_magnitude = 0.05
            num_perturbed = max(2, min(6, len(points) // 3))
        elif iteration < 5000:
            # Mid stage: medium perturbations
            perturbation_magnitude = 0.02
            num_perturbed = max(1, min(4, len(points) // 4))
        else:
            # Late stage: small perturbations for fine-tuning
            perturbation_magnitude = 0.005
            num_perturbed = max(1, min(3, len(points) // 5))
        
        # Select random points to perturb
        indices_to_perturb = np.random.choice(len(points), num_perturbed, replace=False)
        
        for idx in indices_to_perturb:
            # Generate perturbation in tangent plane
            random_vec = np.random.randn(3)
            # Project onto sphere surface normal (tangent plane)
            normal_vec = new_points[idx]
            tangent_vec = random_vec - np.dot(random_vec, normal_vec) * normal_vec
            
            # Normalize tangent vector
            tangent_norm = np.linalg.norm(tangent_vec)
            if tangent_norm > 1e-10:
                tangent_vec = tangent_vec / tangent_norm
            
            # Apply perturbation
            perturbation = tangent_vec * np.random.normal(0, perturbation_magnitude)
            new_points[idx] += perturbation
            
            # Project back to unit sphere
            norm = np.linalg.norm(new_points[idx])
            if norm > 0:
                new_points[idx] = new_points[idx] / norm
                
        return new_points
    
    def initialize_points():
        """Initialize points using a combination of strategies"""
        # Strategy 1: Fibonacci sphere (good initial distribution)
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle
        
        for i in range(14):
            y = 1 - (i / float(13)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        points = np.array(points)
        
        # Strategy 2: Add some randomness to break perfect symmetry
        noise_magnitude = 0.02
        points += np.random.normal(0, noise_magnitude, points.shape)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        points = points / norms
        
        return points
    
    def multi_scale_optimization(initial_points, max_iterations=10000):
        """Perform multi-scale optimization with different resolutions"""
        # Keep track of best solution
        best_points = initial_points.copy()
        best_ratio = compute_min_max_ratio(initial_points)[2]
        
        # Multi-scale optimization with decreasing resolution
        scales = [1.0, 0.7, 0.4, 0.2, 0.1]
        
        for scale_idx, scale_factor in enumerate(scales):
            # Scale down the number of iterations for coarser scales
            iter_count = int(max_iterations * scale_factor)
            
            if iter_count == 0:
                continue
                
            current_points = initial_points.copy()
            
            for iteration in range(iter_count):
                # Apply adaptive perturbation
                new_points = perturb_points_adaptive(current_points, iteration)
                
                # Compute new ratio
                new_min_dist, new_max_dist, new_ratio = compute_min_max_ratio(new_points)
                
                # Accept or reject the new solution
                if new_ratio > best_ratio:
                    # Always accept better solutions
                    current_points = new_points
                    best_points = new_points.copy()
                    best_ratio = new_ratio
                elif np.random.rand() < 0.1:  # Sometimes accept worse solutions
                    current_points = new_points
                
                # Every 500 iterations, fine-tune with local optimization
                if iteration % 500 == 0 and iteration > 0:
                    # Local refinement using scipy minimize
                    try:
                        # Flatten the points for optimization
                        x0 = current_points.flatten()
                        
                        # Define constraints for unit sphere
                        def constraint_func(x_flat):
                            points_test = x_flat.reshape(-1, 3)
                            norms = np.linalg.norm(points_test, axis=1)
                            return norms - 1.0
                        
                        cons = {'type': 'eq', 'fun': constraint_func}
                        
                        # Optimize with L-BFGS-B
                        result = minimize(objective_function, x0, args=(14,), method='L-BFGS-B', 
                                        constraints=cons, options={'maxiter': 20, 'ftol': 1e-8})
                        
                        if result.success:
                            refined_points = result.x.reshape(-1, 3)
                            # Ensure they're on unit sphere
                            norms = np.linalg.norm(refined_points, axis=1, keepdims=True)
                            norms = np.where(norms == 0, 1, norms)
                            refined_points = refined_points / norms
                            
                            # Check if refinement improved
                            _, _, refined_ratio = compute_min_max_ratio(refined_points)
                            if refined_ratio > best_ratio:
                                best_points = refined_points.copy()
                                best_ratio = refined_ratio
                                current_points = refined_points.copy()
                    except:
                        pass
            
            # Update initial points for next scale
            initial_points = current_points.copy()
            
        return best_points, best_ratio
    
    # Main optimization process
    np.random.seed(42)
    
    # Initialize points
    points = initialize_points()
    
    # Multi-scale optimization
    points, ratio = multi_scale_optimization(points, max_iterations=8000)
    
    # Final refinement with gradient-based method
    try:
        # Flatten points
        x0 = points.flatten()
        
        # Define constraint for unit sphere
        def constraint_func(x_flat):
            points_test = x_flat.reshape(-1, 3)
            norms = np.linalg.norm(points_test, axis=1)
            return norms - 1.0
        
        cons = {'type': 'eq', 'fun': constraint_func}
        
        # Use L-BFGS-B for final polishing
        result = minimize(objective_function, x0, args=(14,), method='L-BFGS-B', 
                         constraints=cons, options={'maxiter': 100, 'ftol': 1e-9})
        
        if result.success:
            final_points = result.x.reshape(-1, 3)
            # Ensure unit sphere constraint
            norms = np.linalg.norm(final_points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            final_points = final_points / norms
            _, _, final_ratio = compute_min_max_ratio(final_points)
            if final_ratio > ratio:
                points = final_points
                ratio = final_ratio
    except:
        pass
    
    # Ensure normalization to unit cube [0,1]^3
    # Project to unit cube [0,1]^3
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    ranges = max_coords - min_coords
    
    # Handle case where there's no variation
    if np.any(ranges == 0):
        # If any dimension has no variation, return points centered at 0.5
        points_in_cube = np.full_like(points, 0.5)
    else:
        # Scale to [0,1] range
        points_in_cube = (points - min_coords) / ranges
        # Ensure they're clipped to [0,1]
        points_in_cube = np.clip(points_in_cube, 0, 1)
    
    return points_in_cube

# EVOLVE-BLOCK-END