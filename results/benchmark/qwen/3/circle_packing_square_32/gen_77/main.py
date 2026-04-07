# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import random
from itertools import combinations
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    
    # Generate initial points using a modified grid-based approach
    n = 32
    
    # Create initial points using a more sophisticated sampling strategy
    initial_points = []
    
    # Use a grid with some randomness to get good initial distribution
    grid_size = int(np.ceil(np.sqrt(n)))
    spacing = 1.0 / (grid_size + 1)
    
    for i in range(grid_size):
        for j in range(grid_size):
            if len(initial_points) >= n:
                break
            x = (j + 1) * spacing + random.uniform(-spacing/4, spacing/4)
            y = (i + 1) * spacing + random.uniform(-spacing/4, spacing/4)
            # Ensure points are within bounds
            x = max(0.01, min(0.99, x))
            y = max(0.01, min(0.99, y))
            initial_points.append([x, y])
            
        if len(initial_points) >= n:
            break
    
    # If we don't have enough points, add random ones
    while len(initial_points) < n:
        x = random.uniform(0.01, 0.99)
        y = random.uniform(0.01, 0.99)
        initial_points.append([x, y])
    
    initial_points = np.array(initial_points[:n])
    
    # Apply Voronoi-based optimization
    result = optimize_voronoi_layout(initial_points, n)
    
    return result

def optimize_voronoi_layout(initial_points, n):
    """
    Optimize circle placement using Voronoi-based approach
    """
    # Start with Voronoi diagram of initial points
    points = initial_points.copy()
    
    # Iteratively improve the configuration
    best_sum_radii = 0
    best_circles = None
    
    # Multiple random restarts to avoid local optima
    for restart in range(5):
        # Create a copy of initial points and randomize slightly
        current_points = points.copy()
        for i in range(len(current_points)):
            current_points[i][0] += random.uniform(-0.02, 0.02)
            current_points[i][1] += random.uniform(-0.02, 0.02)
            # Clamp to bounds
            current_points[i][0] = max(0.01, min(0.99, current_points[i][0]))
            current_points[i][1] = max(0.01, min(0.99, current_points[i][1]))
        
        # Run optimization cycle
        circles = optimize_single_configuration(current_points, n)
        
        # Calculate sum of radii
        sum_radii = np.sum(circles[:, 2])
        
        if sum_radii > best_sum_radii:
            best_sum_radii = sum_radii
            best_circles = circles.copy()
    
    return best_circles

def optimize_single_configuration(points, n):
    """
    Optimize a single configuration with given point positions
    """
    # Start with equal radius for all circles
    circles = np.zeros((n, 3))
    circles[:, 0] = points[:, 0]  # x
    circles[:, 1] = points[:, 1]  # y
    circles[:, 2] = 0.05         # initial radius
    
    # Apply iterative optimization using a greedy approach
    improved = True
    iteration = 0
    max_iterations = 100
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try increasing each radius as much as possible
        for i in range(n):
            old_radius = circles[i, 2]
            
            # Compute maximum possible radius for this circle
            max_radius = compute_max_radius(circles, i)
            
            if max_radius > old_radius + 1e-6:
                circles[i, 2] = max_radius
                improved = True
    
    # Final cleanup - ensure no overlaps and boundary constraints
    circles = enforce_constraints(circles)
    
    return circles

def compute_max_radius(circles, index):
    """
    Compute the maximum radius for circle at given index without violating constraints
    """
    # Get the current position
    x, y = circles[index, 0], circles[index, 1]
    
    # Find minimum distance to other circles
    min_dist = float('inf')
    
    for i in range(len(circles)):
        if i != index:
            dx = circles[i, 0] - x
            dy = circles[i, 1] - y
            distance = np.sqrt(dx*dx + dy*dy)
            min_dist = min(min_dist, distance)
    
    # Minimum distance to boundaries
    boundary_dist = min(x, 1-x, y, 1-y)
    
    # Maximum radius is limited by both boundary and other circles
    if min_dist == float('inf'):
        max_radius = boundary_dist
    else:
        # The maximum radius is limited by either boundary or other circles
        max_radius = min(boundary_dist, min_dist / 2.0)
    
    return max(0.001, max_radius)

def enforce_constraints(circles):
    """
    Enforce all constraints and make sure circles don't overlap or go out of bounds
    """
    # First, make sure all circles are within bounds
    for i in range(len(circles)):
        # Clamp positions to valid range
        circles[i, 0] = max(circles[i, 2], min(1 - circles[i, 2], circles[i, 0]))
        circles[i, 1] = max(circles[i, 2], min(1 - circles[i, 2], circles[i, 1]))
    
    # Resolve overlaps through iterative adjustment
    changed = True
    iterations = 0
    
    while changed and iterations < 20:
        changed = False
        iterations += 1
        
        # Try to reduce overlapping circles
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                required_distance = circles[i, 2] + circles[j, 2]
                
                if distance < required_distance:
                    # Need to adjust radii
                    overlap = required_distance - distance
                    
                    # Reduce both radii proportionally
                    reduction = overlap / 2.0
                    if circles[i, 2] > reduction and circles[j, 2] > reduction:
                        circles[i, 2] -= reduction
                        circles[j, 2] -= reduction
                        changed = True
                        
                    # Ensure radii are positive
                    circles[i, 2] = max(0.001, circles[i, 2])
                    circles[j, 2] = max(0.001, circles[j, 2])
    
    # Clamp radii to keep them within reasonable bounds
    for i in range(len(circles)):
        circles[i, 2] = max(0.001, min(0.5, circles[i, 2]))
        # Ensure positions still respect boundaries
        circles[i, 0] = max(circles[i, 2], min(1 - circles[i, 2], circles[i, 0]))
        circles[i, 1] = max(circles[i, 2], min(1 - circles[i, 2], circles[i, 1]))
    
    return circles

# EVOLVE-BLOCK-END