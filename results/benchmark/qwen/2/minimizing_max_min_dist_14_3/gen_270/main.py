# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
import time
import math

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def fibonacci_sphere(n, golden_ratio=1.618033988749895):
        """Generate n points distributed approximately uniformly on a sphere."""
        points = []
        for i in range(n):
            z = 1 - (i / (n - 1)) * 2  # z goes from 1 to -1
            radius = np.sqrt(1 - z*z)
            theta = np.arctan2(np.sin(i * 2 * np.pi / golden_ratio), np.cos(i * 2 * np.pi / golden_ratio))
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)

    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
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

    def project_to_sphere(points):
        """Project points onto unit sphere."""
        norms = np.linalg.norm(points, axis=1)
        norms = np.where(norms == 0, 1, norms)
        return points / norms[:, np.newaxis]

    def adaptive_simulated_annealing(initial_points, max_time=360):
        """Enhanced simulated annealing with adaptive cooling."""
        points = initial_points.copy()
        current_ratio = compute_min_max_ratio(points)
        
        # Parameters for adaptive simulated annealing
        temp = 1.0
        min_temp = 1e-10
        base_cooling_rate = 0.9995
        max_iter = 200000
        iter_count = 0
        
        # Track best solution
        best_points = points.copy()
        best_ratio = current_ratio
        
        start_time = time.time()
        
        # Track recent improvements for adaptive cooling
        recent_improvements = []
        max_recent = 100
        
        while temp > min_temp and iter_count < max_iter and time.time() - start_time < max_time:
            # Create neighbor by perturbing one point
            new_points = points.copy()
            point_idx = np.random.randint(len(new_points))
            
            # Calculate adaptive perturbation size based on current solution quality
            distances = pdist(new_points)
            if len(distances) > 0:
                avg_dist = np.mean(distances)
                # Perturbation scale inversely related to current solution quality
                perturbation_scale = min(0.05, max(0.001, 0.02 / (current_ratio + 1e-8)))
            else:
                perturbation_scale = 0.02
            
            # Apply perturbation
            perturbation = np.random.normal(0, perturbation_scale, 3)
            new_points[point_idx] += perturbation
            
            # Project back onto unit sphere
            new_points[point_idx] = project_to_sphere(new_points[point_idx:point_idx+1])[0]
            
            # Compute new ratio
            new_ratio = compute_min_max_ratio(new_points)
            
            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio or np.random.rand() < np.exp((new_ratio - current_ratio) / temp):
                points = new_points
                current_ratio = new_ratio
                
                # Update best solution if improved
                if new_ratio > best_ratio:
                    best_points = new_points.copy()
                    best_ratio = new_ratio
            
            # Adaptive cooling based on recent performance
            if len(recent_improvements) < max_recent:
                recent_improvements.append(new_ratio - current_ratio)
            else:
                recent_improvements.pop(0)
                recent_improvements.append(new_ratio - current_ratio)
            
            # Adjust cooling rate based on average recent improvement
            if len(recent_improvements) > 10:
                avg_improvement = np.mean(recent_improvements)
                if avg_improvement < 1e-8:  # Very slow improvement
                    cooling_rate = base_cooling_rate * 1.1  # Cool faster
                elif avg_improvement > 1e-4:  # Fast improvement
                    cooling_rate = base_cooling_rate * 0.98  # Cool slower
                else:
                    cooling_rate = base_cooling_rate
            else:
                cooling_rate = base_cooling_rate
            
            # Apply cooling
            temp = max(min_temp, temp * cooling_rate)
            iter_count += 1
            
        return best_points, best_ratio

    def generate_multiple_initializations():
        """Generate diverse initial configurations."""
        initial_configs = []
        
        # Original Fibonacci sphere
        points = fibonacci_sphere(14)
        initial_configs.append(("fibonacci", points))
        
        # Different golden ratio variations
        golden_ratios = [1.618033988749895, 1.4142135623730951, 1.7320508075688772]
        
        for i, gr in enumerate(golden_ratios):
            points = fibonacci_sphere(14, gr)
            initial_configs.append((f"fibonacci_{i}", points))
            
        # Add some random noise to create diversity
        for i in range(3):
            np.random.seed(1000 + i)
            points = fibonacci_sphere(14)
            noise = np.random.normal(0, 0.02, points.shape)
            points += noise
            points = project_to_sphere(points)
            initial_configs.append((f"noisy_fibonacci_{i}", points))
            
        return initial_configs

    best_points = None
    best_ratio = 0.0

    # Generate diverse initial configurations
    initial_configs = generate_multiple_initializations()

    # Try each initial configuration
    for config_name, initial_points in initial_configs:
        np.random.seed(hash(config_name) % 10000)
        
        # Apply enhanced simulated annealing
        optimized_points, final_ratio = adaptive_simulated_annealing(initial_points, max_time=300)
        
        if final_ratio > best_ratio:
            best_ratio = final_ratio
            best_points = optimized_points.copy()

    # Final refinement with more focused optimization
    if best_points is not None:
        # Run another round with slightly different settings for fine-tuning
        np.random.seed(42)
        refined_points, refined_ratio = adaptive_simulated_annealing(best_points, max_time=60)
        
        if refined_ratio > best_ratio:
            best_points = refined_points
            best_ratio = refined_ratio

    # Ensure points are on unit sphere
    if best_points is not None:
        best_points = project_to_sphere(best_points)
        
    # Fallback to Fibonacci initialization if nothing worked
    if best_points is None:
        np.random.seed(42)
        points = fibonacci_sphere(14)
        best_points = project_to_sphere(points)

    return best_points

# EVOLVE-BLOCK-END