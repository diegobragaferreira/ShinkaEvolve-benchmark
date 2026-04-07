# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import ConvexHull
import warnings
import time
from typing import Tuple

class SphereMomentumOptimizer:
    """Physics-inspired optimizer using momentum and repulsive forces to maximize min/max distance ratio."""
    
    def __init__(self, num_points=14, max_iterations=1000):
        self.num_points = num_points
        self.max_iterations = max_iterations
        self.best_solution = None
        self.best_ratio = -np.inf
        
    def fibonacci_sphere(self, samples=14):
        """Generate points on sphere using Fibonacci spiral method."""
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle

        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)
    
    def generate_initial_configurations(self):
        """Create multiple diverse initial configurations."""
        configs = []
        np.random.seed(42)
        
        # Config 1: Standard Fibonacci sphere
        fib_points = self.fibonacci_sphere(self.num_points)
        # Scale to unit cube [0,1]^3
        fib_points = fib_points - np.mean(fib_points, axis=0)
        max_coord = np.max(np.abs(fib_points))
        if max_coord > 0:
            fib_points = fib_points / (2 * max_coord) + 0.5
        configs.append(("fibonacci", fib_points))
        
        # Config 2: Perturbed Fibonacci for better escape from local optima  
        perturbed = fib_points + np.random.normal(0, 0.03, fib_points.shape)
        # Clamp to [0,1]^3
        perturbed = np.clip(perturbed, 0, 1)
        configs.append(("perturbed_fibonacci", perturbed))
        
        # Config 3: Random uniform distribution
        random_points = np.random.rand(self.num_points, 3)
        configs.append(("random", random_points))
        
        # Config 4: Grid-based with jitter
        grid_coords = np.linspace(0.05, 0.95, 3)  # Avoid edges
        grid_points = []
        for x in grid_coords:
            for y in grid_coords:
                for z in grid_coords:
                    grid_points.append([x, y, z])
        # Take first N points and add jitter
        grid_array = np.array(grid_points[:self.num_points]) + np.random.normal(0, 0.02, (self.num_points, 3))
        # Clamp to [0,1]^3
        grid_array = np.clip(grid_array, 0, 1)
        configs.append(("grid", grid_array))
        
        # Config 5: Different random seed
        np.random.seed(2468)
        random_points2 = np.random.rand(self.num_points, 3)
        configs.append(("random2", random_points2))
        
        return configs
    
    def calculate_ratio(self, points):
        """Calculate min/max distance ratio with robust error handling."""
        if len(points) < 2:
            return 0.0
        try:
            distances = pdist(points)
            if len(distances) == 0:
                return 0.0
            # Filter out invalid distances
            finite_distances = distances[np.isfinite(distances)]
            if len(finite_distances) == 0:
                return 0.0
            d_min = np.min(finite_distances)
            d_max = np.max(finite_distances)
            if d_max <= 0:
                return 0.0
            return d_min / d_max
        except:
            return 0.0

    def compute_forces(self, points: np.ndarray, momenta: np.ndarray, 
                       base_strength: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute repulsive forces between points and update momenta.
        Uses inverse distance law with momentum-based acceleration.
        """
        # Compute pairwise distances
        n = points.shape[0]
        
        # Calculate all pairwise distances efficiently
        dist_matrix = cdist(points, points)
        
        # Zero out diagonal (point to itself)
        np.fill_diagonal(dist_matrix, 1.0)
        
        # Avoid division by zero and very small distances
        safe_dist = np.maximum(dist_matrix, 1e-10)
        
        # Compute forces as inverse of distance (repulsive)
        # F ∝ 1/r^2, but we'll use 1/r^(alpha) with alpha = 2
        force_magnitudes = base_strength / (safe_dist ** 2)
        
        # Zero out diagonal again
        np.fill_diagonal(force_magnitudes, 0.0)
        
        # Compute direction vectors for all point pairs
        directions = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        # Normalize directions
        norms = np.linalg.norm(directions, axis=2, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized_directions = directions / norms
        
        # Compute total forces on each point
        forces = np.sum(force_magnitudes[:, :, np.newaxis] * normalized_directions, axis=1)
        
        # Apply momentum-based velocity updates
        # V = V_old * damping + force * dt
        damping_factor = 0.8
        dt = 0.01
        
        new_momenta = damping_factor * momenta + dt * forces
        new_positions = points + new_momenta
        
        return new_positions, new_momenta
    
    def adaptive_constraint_handling(self, positions: np.ndarray, 
                                   penalty_weight: float = 1000.0) -> np.ndarray:
        """Adaptively handle boundary constraints with dynamic penalty adjustment."""
        # Reflect points that go out of bounds
        reflected = positions.copy()
        penalty = 0.0

        # For each coordinate, reflect if out of bounds
        for dim in range(3):
            # Handle points going below 0
            below_zero = reflected[:, dim] < 0
            if np.any(below_zero):
                reflected[below_zero, dim] = -reflected[below_zero, dim]
                penalty += penalty_weight * np.sum(reflected[below_zero, dim]**2)
            
            # Handle points going above 1
            above_one = reflected[:, dim] > 1
            if np.any(above_one):
                reflected[above_one, dim] = 2 - reflected[above_one, dim]
                penalty += penalty_weight * np.sum((reflected[above_one, dim] - 1)**2)
                
        return reflected, penalty

    def multi_scale_optimization(self, initial_points: np.ndarray, time_limit: float):
        """Perform multi-scale optimization with different resolution levels."""
        start_time = time.time()
        
        # Define scale factors for coarse-to-fine optimization
        scales = [1.0, 0.7, 0.4, 0.2, 0.1]  # From coarse to fine
        current_points = initial_points.copy()
        current_momenta = np.zeros_like(current_points)
        
        # Track best solution
        best_points = current_points.copy()
        best_ratio = self.calculate_ratio(current_points)
        
        # Iteration counters for each scale
        scale_iterations = [200, 150, 100, 50, 25]
        
        for scale_idx, (scale_factor, iter_count) in enumerate(zip(scales, scale_iterations)):
            if time.time() - start_time > time_limit - 5:
                break
                
            # Adjust force strength based on scale
            force_strength = 1.0 * scale_factor
            
            # Optimize at this scale level
            for iteration in range(iter_count):
                # Compute forces
                new_points, new_momenta = self.compute_forces(
                    current_points, current_momenta, 
                    base_strength=force_strength
                )
                
                # Apply constraint handling
                constrained_points, penalty = self.adaptive_constraint_handling(new_points)
                
                # Update current state
                current_points = constrained_points
                current_momenta = new_momenta
                
                # Update best solution if needed
                current_ratio = self.calculate_ratio(current_points)
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = current_points.copy()
                    
                # Early termination if we're converging
                if iteration > 10 and iteration % 20 == 0:
                    # Check if we're not improving much
                    recent_ratios = [self.calculate_ratio(current_points) for _ in range(5)]
                    if len(recent_ratios) > 1:
                        if max(recent_ratios) - min(recent_ratios) < 1e-8:
                            break
            
            # Fine-tune at next scale level with more iterations
            if scale_idx < len(scales) - 1:
                # Smooth transition to finer scale
                current_points = best_points.copy()
                current_momenta = np.zeros_like(current_points)
        
        return best_points
    
    def local_refinement(self, points: np.ndarray, time_limit: float) -> np.ndarray:
        """Apply local optimization refinement using gradient-based methods."""
        start_time = time.time()
        
        # Try gradient-based refinement for fine-tuning
        try:
            # Simple local optimization with gradient descent
            current_points = points.copy()
            learning_rate = 0.01
            decay_factor = 0.95
            min_lr = 0.001
            
            # Perform gradient ascent with momentum
            momentum = np.zeros_like(current_points)
            momentum_decay = 0.8
            
            for iteration in range(1000):
                if time.time() - start_time > time_limit - 5:
                    break
                    
                # Compute gradients via finite differences
                grad = np.zeros_like(current_points)
                epsilon = 1e-6
                
                for i in range(self.num_points):
                    for j in range(3):  # x, y, z coordinates
                        # Forward difference approximation
                        current_points_perturbed = current_points.copy()
                        current_points_perturbed[i, j] += epsilon
                        
                        # Boundary handling
                        if current_points_perturbed[i, j] < 0:
                            current_points_perturbed[i, j] = 0
                        elif current_points_perturbed[i, j] > 1:
                            current_points_perturbed[i, j] = 1
                            
                        ratio_plus = self.calculate_ratio(current_points_perturbed)
                        
                        current_points_perturbed = current_points.copy()
                        current_points_perturbed[i, j] -= epsilon
                        
                        # Boundary handling
                        if current_points_perturbed[i, j] < 0:
                            current_points_perturbed[i, j] = 0
                        elif current_points_perturbed[i, j] > 1:
                            current_points_perturbed[i, j] = 1
                            
                        ratio_minus = self.calculate_ratio(current_points_perturbed)
                        
                        grad[i, j] = (ratio_plus - ratio_minus) / (2 * epsilon)
                
                # Update with momentum and gradient ascent (since we want to maximize)
                momentum = momentum_decay * momentum + grad
                current_points += learning_rate * momentum
                
                # Ensure points stay within bounds
                current_points = np.clip(current_points, 0, 1)
                
                # Reduce learning rate over time
                learning_rate = max(learning_rate * decay_factor, min_lr)
                
                # Early stopping if convergence
                if iteration > 10 and iteration % 20 == 0:
                    recent_ratios = [self.calculate_ratio(current_points) for _ in range(5)]
                    if len(recent_ratios) > 1:
                        if max(recent_ratios) - min(recent_ratios) < 1e-10:
                            break
                
        except Exception as e:
            warnings.warn(f"Local refinement failed: {e}")
            return points
            
        return current_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    # Initialize optimizer
    optimizer = SphereMomentumOptimizer(num_points=14)
    
    # Set time limit
    start_time = time.time()
    time_limit = 340  # seconds
    
    # Generate initial configuration
    np.random.seed(42)
    
    # Start with spherical arrangement
    initial_points = optimizer.fibonacci_sphere(14)
    
    # Scale to fit within unit cube [0,1]^3
    initial_points = initial_points - np.mean(initial_points, axis=0)
    max_coord = np.max(np.abs(initial_points))
    if max_coord > 0:
        initial_points = initial_points / (2 * max_coord) + 0.5

    # Run multi-scale momentum optimization
    optimized_points = optimizer.multi_scale_optimization(initial_points, time_limit)
    
    # Apply local refinement if time permits
    if time.time() - start_time < time_limit - 5:
        refined_points = optimizer.local_refinement(optimized_points, time_limit)
        
        # Evaluate which solution is better
        ratio_before = optimizer.calculate_ratio(optimized_points)
        ratio_after = optimizer.calculate_ratio(refined_points)
        
        if ratio_after > ratio_before:
            optimized_points = refined_points
    
    # Ensure final points are within bounds
    optimized_points = np.clip(optimized_points, 0, 1)
    
    # Final validation check
    if optimizer.calculate_ratio(optimized_points) <= 0:
        # If something went wrong, return original good initialization
        return initial_points
    
    return optimized_points

# EVOLVE-BLOCK-END