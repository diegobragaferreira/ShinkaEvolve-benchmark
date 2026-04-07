# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist, pdist
import math
import time
import warnings

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    np.random.seed(42)
    
    # Enhanced hexagonal initialization with better geometric properties
    def initialize_hexagonal():
        # Create a 4x4 grid with strategic perturbations for better dispersion
        points = []
        
        # Generate points in a structured manner that maximizes minimum distance
        # Using offset hexagonal pattern for optimal distribution
        spacing = 1.0 / 3.0  # Grid spacing to fit within [0,1] box
        
        for i in range(4):
            for j in range(4):
                # Offset every other row for hexagonal packing
                x = j * spacing
                y = i * spacing
                
                # Apply strategic offset to create better spacing
                if i % 2 == 1:
                    x += spacing * 0.25
                    
                points.append([x, y])
        
        points = np.array(points)
        
        # Normalize to [0,1] range
        if np.max(points) > 0:
            points = points / np.max(points)
        
        # Add controlled perturbations to break symmetry and improve solution
        # Use different noise levels for different points to avoid uniform patterns
        noise_magnitude = 0.02
        
        # Apply stronger perturbation to corner points to encourage expansion
        corner_indices = [0, 3, 12, 15]  # Corners of 4x4 grid
        
        for i, point in enumerate(points):
            if i in corner_indices:
                # Stronger perturbation for corners
                point[0] += np.random.normal(0, noise_magnitude * 2.0)
                point[1] += np.random.normal(0, noise_magnitude * 2.0)
            else:
                # Standard perturbation
                point[0] += np.random.normal(0, noise_magnitude)
                point[1] += np.random.normal(0, noise_magnitude)
        
        # Clip to ensure all points stay within [0,1]
        points = np.clip(points, 0, 1)
        
        return points

    # More efficient distance ratio computation
    def compute_ratio(points):
        if len(points) < 2:
            return 0
            
        # Use cdist for efficient distance matrix computation
        distances = cdist(points, points)
        
        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(distances, np.inf)
        
        # Find minimum and maximum distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max <= 0:
            return 0
            
        return d_min / d_max

    # Enhanced neighbor generation with adaptive strategies
    def generate_neighbor(points, iteration, max_iter):
        new_points = points.copy()
        
        # Adaptive move selection based on iteration progress
        if iteration < max_iter * 0.3:
            # Early stage: aggressive moves to explore widely
            move_type = np.random.choice(['single', 'cluster'], p=[0.7, 0.3])
            step_size = 0.03
        elif iteration < max_iter * 0.7:
            # Middle stage: balanced exploration
            move_type = np.random.choice(['single', 'pair', 'cluster'], p=[0.5, 0.3, 0.2])
            step_size = 0.015
        else:
            # Late stage: fine-tuning
            move_type = np.random.choice(['single', 'pair'], p=[0.7, 0.3])
            step_size = 0.005
        
        if move_type == 'single':
            # Perturb a single point with adaptive step size
            idx = np.random.randint(len(points))
            new_points[idx, 0] += np.random.normal(0, step_size)
            new_points[idx, 1] += np.random.normal(0, step_size)
            
        elif move_type == 'pair':
            # Move two nearby points together
            # Find two nearby points to move together
            distances = cdist(points, points)
            np.fill_diagonal(distances, np.inf)
            min_indices = np.unravel_index(np.argmin(distances), distances.shape)
            idx1, idx2 = min_indices
            
            # Move both points in coordination
            movement = np.random.normal(0, step_size * 0.8, 2)
            new_points[idx1] += movement
            new_points[idx2] += movement
            
        else:  # cluster
            # Move a small cluster of points together
            num_cluster = min(3, len(points))
            cluster_indices = np.random.choice(len(points), num_cluster, replace=False)
            
            # Move them towards their centroid
            centroid = np.mean(new_points[cluster_indices], axis=0)
            movement = np.random.normal(0, step_size * 0.6, 2)
            
            for idx in cluster_indices:
                new_points[idx] += movement
        
        # Boundary constraint enforcement with clipping
        new_points = np.clip(new_points, 0, 1)
        
        return new_points

    # Advanced simulated annealing with adaptive cooling
    def adaptive_simulated_annealing(initial_points, max_iter=8000):
        points = initial_points.copy()
        current_ratio = compute_ratio(points)
        best_points = points.copy()
        best_ratio = current_ratio
        
        # Adaptive temperature schedule with performance monitoring
        temp = 0.05
        min_temp = 1e-8
        cooling_rate = 0.9995
        
        # Track recent improvements for adaptive cooling
        improvement_window = 50
        recent_improvements = []
        
        for iteration in range(max_iter):
            # Dynamic temperature adjustment based on recent performance
            if len(recent_improvements) >= improvement_window:
                avg_improvement = np.mean(recent_improvements[-improvement_window:])
                if avg_improvement < 1e-6:
                    # If stagnating, cool slower to allow more exploration
                    temp *= 0.995
                elif avg_improvement > 1e-4:
                    # If improving well, cool faster to converge
                    temp *= 0.999
                else:
                    # Normal cooling rate
                    temp *= cooling_rate
                    
            temp = max(temp, min_temp)
            
            # Generate neighbor solution
            new_points = generate_neighbor(points, iteration, max_iter)
            new_ratio = compute_ratio(new_points)
            
            # Track recent improvements
            if new_ratio > current_ratio:
                recent_improvements.append(new_ratio - current_ratio)
                if len(recent_improvements) > improvement_window * 2:
                    recent_improvements.pop(0)
            
            # Accept or reject using Metropolis criterion
            if new_ratio > current_ratio or np.random.rand() < math.exp((new_ratio - current_ratio) / temp):
                points = new_points.copy()
                current_ratio = new_ratio
                
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = points.copy()
            
            # Early stopping if temperature gets too low
            if temp < min_temp:
                break
        
        return best_points, best_ratio

    # Generate multiple diverse initial configurations
    initial_configs = []
    
    # 1. Hexagonal initialization (core strategy)
    initial_configs.append(initialize_hexagonal())
    
    # 2. Variants with different perturbations
    for i in range(3):
        points = initialize_hexagonal()
        # Different perturbation pattern for variety
        noise_level = 0.015 + i * 0.005
        for j in range(len(points)):
            points[j, 0] += np.random.normal(0, noise_level)
            points[j, 1] += np.random.normal(0, noise_level)
        points = np.clip(points, 0, 1)
        initial_configs.append(points)
    
    # 3. Random configuration for diversity
    initial_configs.append(np.random.rand(16, 2))
    
    # 4. Grid with slight perturbation
    grid_points = []
    spacing = 1.0 / 3.0
    for i in range(4):
        for j in range(4):
            x = i * spacing + np.random.normal(0, 0.01)
            y = j * spacing + np.random.normal(0, 0.01)
            grid_points.append([x, y])
    grid_points = np.clip(np.array(grid_points[:16]), 0, 1)
    initial_configs.append(grid_points)
    
    # Run optimization on multiple initial configurations
    best_points = None
    best_ratio = -np.inf
    
    # Process each initial configuration with adaptive SA
    for i, initial_config in enumerate(initial_configs):
        try:
            # Apply adaptive simulated annealing
            optimized_points, ratio = adaptive_simulated_annealing(initial_config.copy())
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
                
        except Exception as e:
            warnings.warn(f"Error optimizing initial config {i}: {str(e)}")
            continue
    
    # Final validation
    if best_points is not None:
        final_ratio = compute_ratio(best_points)
        print(f"Final optimized ratio: {final_ratio:.6f}")
        return best_points
    else:
        # Fallback to first configuration if nothing worked
        return initial_configs[0] if initial_configs else np.random.rand(16, 2)

# EVOLVE-BLOCK-END