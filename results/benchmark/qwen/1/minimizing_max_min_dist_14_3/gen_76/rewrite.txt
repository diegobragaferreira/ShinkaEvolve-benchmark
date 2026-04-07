# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import random
import time
import math

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses spherical simulated annealing optimization to find a good configuration.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)
    random.seed(42)

    n = 14
    d = 3

    # Generate initial points using Fibonacci spiral on sphere for good distribution
    def fibonacci_sphere(samples=14):
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

    # Initialize points on unit sphere
    points = fibonacci_sphere(n)
    
    # Normalize to fit in [0,1]^3 while preserving spherical properties
    # Center around origin and scale to unit sphere, then map to [0,1]^3
    points = points - np.mean(points, axis=0)
    max_coord = np.max(np.abs(points))
    if max_coord > 0:
        points = points / max_coord * 0.5
    points = points + 0.5  # Shift to [0,1]^3

    # Simulated Annealing parameters
    max_iter = 150000
    initial_temp = 0.1
    cooling_rate = 0.9998
    min_temp = 1e-10

    # Track best solution
    best_points = points.copy()
    best_ratio = 0.0

    # Calculate ratio and energy function
    def calculate_ratio_and_energy(points):
        distances = squareform(pdist(points))
        # Set diagonal to large value to avoid considering same points
        np.fill_diagonal(distances, np.inf)
        
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max > 0:
            ratio = d_min / d_max
        else:
            ratio = 0.0
            
        # Energy function: penalize both very small and very large distances
        # This helps in finding a balanced distribution
        energy = 0.0
        if d_min > 0 and d_max > 0:
            # Penalize ratio being too low (i.e., when min distance is too small relative to max)
            # We'll use log scaling to make small changes more significant near boundaries
            energy = -ratio  # Negative because we want to maximize ratio
            
            # Additional penalty for extreme distance variations
            if d_max > 0:
                # Scale the energy inversely with the ratio
                energy -= 0.1 * (d_max / d_min)  # Penalize wide variance
                
            # Add penalty for points being too close together
            # This encourages non-degenerate solutions
            if d_min < 0.01:
                energy -= 1000.0 * (0.01 - d_min)  # Strong penalty for very small distances
                
        return ratio, energy

    current_ratio, current_energy = calculate_ratio_and_energy(points)
    temperature = initial_temp

    # Optimizations for better performance
    # Precompute distance matrix for faster updates
    current_distances = squareform(pdist(points))
    np.fill_diagonal(current_distances, np.inf)
    current_d_min = np.min(current_distances)
    current_d_max = np.max(current_distances)

    # Optimization loop
    start_time = time.time()
    iter_count = 0
    for iteration in range(max_iter):
        iter_count += 1
        # Make small random perturbation
        new_points = points.copy()
        # Perturb one point at a time
        idx = random.randint(0, n-1)
        
        # Adaptive perturbation magnitude based on temperature
        perturbation_magnitude = temperature * 0.02
        
        # Add 3D normal distribution perturbation
        new_points[idx] += np.random.normal(0, perturbation_magnitude, d)
        
        # Keep points in [0,1]^3 bounds
        new_points[idx] = np.clip(new_points[idx], 0, 1)

        # Calculate new ratio and energy efficiently
        new_distances = squareform(pdist(new_points))
        np.fill_diagonal(new_distances, np.inf)
        new_d_min = np.min(new_distances)
        new_d_max = np.max(new_distances)
        
        if new_d_max > 0:
            new_ratio = new_d_min / new_d_max
        else:
            new_ratio = 0.0

        # Accept or reject the move
        if new_ratio > current_ratio:
            points = new_points
            current_ratio = new_ratio
            current_d_min = new_d_min
            current_d_max = new_d_max
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_points = new_points.copy()
        else:
            # Accept with probability based on temperature and energy difference
            delta_energy = new_ratio - current_ratio
            if random.random() < np.exp(delta_energy / temperature):
                points = new_points
                current_ratio = new_ratio
                current_d_min = new_d_min
                current_d_max = new_d_max

        # Cool down
        temperature *= cooling_rate

        # Check for early termination
        if temperature < min_temp:
            break

        # Check time limit
        if time.time() - start_time > 350:  # Leave some buffer for cleanup
            break
            
        # Occasionally reinitialize to escape local minima
        if iter_count % 5000 == 0 and temperature > 0.01:
            # Random reinitialization with some bias towards current solution
            points = best_points.copy() if random.random() < 0.7 else fibonacci_sphere(n)
            points = points - np.mean(points, axis=0)
            max_coord = np.max(np.abs(points))
            if max_coord > 0:
                points = points / max_coord * 0.5
            points = points + 0.5

    # Final validation to prevent degenerate cases
    final_distances = squareform(pdist(best_points))
    np.fill_diagonal(final_distances, np.inf)
    final_min = np.min(final_distances)
    final_max = np.max(final_distances)
    
    if final_min < 1e-6 or final_max < 1e-6:
        # If we have degenerate points, fall back to the initial spherical configuration
        initial_points = fibonacci_sphere(n)
        initial_points = initial_points - np.mean(initial_points, axis=0)
        max_coord = np.max(np.abs(initial_points))
        if max_coord > 0:
            initial_points = initial_points / max_coord * 0.5
        initial_points = initial_points + 0.5
        return initial_points

    return best_points

# EVOLVE-BLOCK-END