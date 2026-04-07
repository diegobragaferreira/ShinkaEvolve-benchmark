# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import SphericalVoronoi
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a novel alternating optimization approach with constraint relaxation.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)
    n = 14
    d = 3

    def generate_initial_configuration():
        """Generate high-quality initial configuration using mathematical principles."""
        # Use a known good configuration for 14 points on sphere
        # Based on the principle of maximizing minimum distance on sphere
        # This is a simplified version of known optimal configurations
        points = []
        
        # Generate points in a way that approximates uniform distribution
        # Using a variant of Fibonacci-based distribution but tuned for 14 points
        phi = np.pi * (3 - np.sqrt(5))  # golden angle
        for i in range(n):
            # Distribute points more evenly
            y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def project_to_cube(points):
        """Project points from unit sphere to unit cube [0,1]^3."""
        # Center and scale to fit nicely in cube
        center = np.mean(points, axis=0)
        points_centered = points - center
        
        # Find the maximum absolute coordinate to properly scale
        max_extent = np.max(np.abs(points_centered))
        if max_extent > 0:
            points_scaled = points_centered / max_extent * 0.5
        else:
            points_scaled = points_centered
            
        # Shift to [0,1]^3
        points_final = points_scaled + 0.5
        return points_final
    
    def calculate_metrics(points):
        """Calculate key metrics for optimization."""
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0, 1.0, 0.0
        
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Handle edge case where all points are identical
        if d_max < 1e-12:
            ratio = 0.0
        else:
            ratio = d_min / d_max
            
        return ratio, d_min, d_max
    
    def get_distance_gradient(points):
        """Compute directional gradients based on distance relationships."""
        n = len(points)
        if n < 2:
            return np.zeros_like(points)
        
        # Compute pairwise distance matrix
        dist_matrix = cdist(points, points)
        np.fill_diagonal(dist_matrix, np.inf)
        
        # For each point, compute repulsion force from closest neighbors
        gradients = np.zeros_like(points)
        
        for i in range(n):
            # Get distances from point i to all others
            dists_i = dist_matrix[i]
            
            # Find the closest neighbor (excluding itself)
            closest_idx = np.argmin(dists_i)
            closest_dist = dists_i[closest_idx]
            
            if closest_dist > 1e-12:
                # Direction from point i to its closest neighbor
                direction = points[closest_idx] - points[i]
                # Normalize and scale by inverse distance (stronger repulsion at closer distances)
                grad = direction / (closest_dist * closest_dist + 1e-12)
                gradients[i] = grad
        
        return gradients
    
    def objective_with_regularization(x_flat, alpha=0.1):
        """Enhanced objective function with regularization terms."""
        points = x_flat.reshape(-1, 3)
        
        # Standard distance-based objective
        distances = pdist(points)
        if len(distances) == 0:
            return 1e10
        
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max < 1e-12:
            return 1e10
            
        ratio = d_min / d_max
        
        # Add regularization term to penalize extreme differences
        # This helps maintain a more balanced configuration
        dist_variance = np.var(distances)
        regularization = alpha * dist_variance / (d_max * d_max + 1e-12)
        
        # Also add penalty for points near boundaries
        boundary_penalty = 0.0
        for coord in range(3):
            coord_values = points[:, coord]
            boundary_penalty += np.sum(np.minimum(coord_values, 1 - coord_values)**2)
        
        # Return weighted sum (minimize negative ratio plus penalties)
        return -(ratio - regularization) + 0.1 * boundary_penalty
    
    def constraint_violation(x_flat):
        """Measure constraint violations (points outside [0,1]^3)."""
        points = x_flat.reshape(-1, 3)
        violations = 0.0
        
        # Count violations for each coordinate
        for i in range(3):
            violations += np.sum(np.maximum(0, points[:, i] - 1.0))  # Above 1
            violations += np.sum(np.maximum(0, -points[:, i]))       # Below 0
            
        return violations
    
    def adaptive_optimization(initial_points, max_iter=1000):
        """Alternating optimization between different strategies."""
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_ratio, _, _ = calculate_metrics(current_points)
        
        # Parameters for adaptive behavior
        learning_rate = 0.01
        momentum = 0.9
        velocity = np.zeros_like(current_points)
        
        # Store history for adaptive adjustments
        history = []
        
        for iteration in range(max_iter):
            # Alternate between different optimization modes
            mode = iteration % 4
            
            if mode == 0:  # Gradient descent with momentum
                # Calculate gradients
                grad = get_distance_gradient(current_points)
                
                # Update using momentum
                velocity = momentum * velocity - learning_rate * grad
                current_points += velocity
                
            elif mode == 1:  # Boundary handling
                # Snap points back to valid region
                current_points = np.clip(current_points, 0, 1)
                
            elif mode == 2:  # Regularized optimization
                # Flatten points for optimization
                flat_current = current_points.flatten()
                
                # Define local objective function with constraints
                def local_obj(x_flat):
                    # Project back to valid space
                    points = np.clip(x_flat.reshape(-1, 3), 0, 1)
                    return objective_with_regularization(points.flatten())
                
                # Simple local update
                current_points = np.clip(current_points + np.random.normal(0, 0.001, current_points.shape), 0, 1)
                
            else:  # Constraint relaxation
                # Relax constraints temporarily to allow exploration
                temp_points = current_points.copy()
                temp_points = np.clip(temp_points + np.random.normal(0, 0.005, temp_points.shape), 0, 1)
                
                # Only accept if it improves quality
                temp_ratio, _, _ = calculate_metrics(temp_points)
                current_ratio, _, _ = calculate_metrics(current_points)
                
                if temp_ratio > current_ratio:
                    current_points = temp_points
                    
            # Maintain boundary constraints
            current_points = np.clip(current_points, 0, 1)
            
            # Evaluate and track best
            cur_ratio, _, _ = calculate_metrics(current_points)
            if cur_ratio > best_ratio:
                best_ratio = cur_ratio
                best_points = current_points.copy()
                
            # Adaptive learning rate adjustment
            if len(history) > 2:
                recent_improvements = [history[-i][0] - history[-i-1][0] for i in range(1, min(3, len(history)))]
                avg_improvement = np.mean(recent_improvements) if recent_improvements else 0
                
                if avg_improvement < 0:
                    learning_rate *= 0.95  # Reduce if not improving
                else:
                    learning_rate = min(learning_rate * 1.05, 0.1)  # Increase if improving
                    
            history.append((cur_ratio, iteration))
            
            # Early stopping condition
            if iteration > 50 and len(history) > 10:
                recent_ratios = [h[0] for h in history[-10:]]
                if np.std(recent_ratios) < 1e-6:
                    break
                    
        return best_points
    
    def advanced_refinement(initial_points, max_iter=300):
        """Advanced refinement using multiple optimization passes."""
        # First pass: Global optimization with constraints
        refined_points = adaptive_optimization(initial_points, max_iter=max_iter//3)
        
        # Second pass: More aggressive local refinement
        for _ in range(5):
            # Small random perturbations
            noise = np.random.normal(0, 0.001, refined_points.shape)
            perturbed = refined_points + noise
            perturbed = np.clip(perturbed, 0, 1)
            
            # Evaluate and accept if better
            orig_ratio, _, _ = calculate_metrics(refined_points)
            pert_ratio, _, _ = calculate_metrics(perturbed)
            
            if pert_ratio > orig_ratio:
                refined_points = perturbed
                
        return refined_points
    
    # Main optimization workflow
    try:
        # Generate initial configuration
        initial_points = generate_initial_configuration()
        
        # Project to cube
        initial_cube_points = project_to_cube(initial_points)
        
        # Apply advanced refinement
        optimized_points = advanced_refinement(initial_cube_points, max_iter=500)
        
        # Final verification and cleanup
        final_points = np.clip(optimized_points, 0, 1)
        
        # Ensure we have a valid solution
        ratio, _, _ = calculate_metrics(final_points)
        if ratio <= 0:
            raise ValueError("Invalid solution")
            
        return final_points
        
    except Exception as e:
        # Fallback to simple initialization
        print(f"Fallback due to error: {e}")
        points = np.random.rand(14, 3) * 0.8 + 0.1
        return points

# EVOLVE-BLOCK-END