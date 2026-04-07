# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
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
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Generate initial points using a structured approach
    # Start with corners and edges, then fill with random points
    initial_points = []
    
    # Add corner points
    initial_points.extend([(0, 0), (1, 0), (0, 1), (1, 1)])
    
    # Add edge centers
    initial_points.extend([(0.5, 0), (0.5, 1), (0, 0.5), (1, 0.5)])
    
    # Add a few more strategic points
    initial_points.extend([(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)])
    
    # Add 20 random points
    for _ in range(20):
        initial_points.append((random.random(), random.random()))
    
    # Create Voronoi diagram
    vor = Voronoi(initial_points)
    
    # Extract Voronoi vertices that are inside the unit square
    valid_vertices = []
    for vertex in vor.vertices:
        if 0 <= vertex[0] <= 1 and 0 <= vertex[1] <= 1:
            valid_vertices.append(vertex)
    
    # Use Voronoi vertices as initial circle centers
    # If we don't have enough vertices, add random points
    if len(valid_vertices) < 32:
        num_needed = 32 - len(valid_vertices)
        for _ in range(num_needed):
            valid_vertices.append((random.random(), random.random()))
    
    # Take first 32 points as our initial circle centers
    centers = valid_vertices[:32]
    
    # Initial radii assignment - start with small values
    radii = [0.02] * 32
    
    # Create initial solution
    circles = np.array([[centers[i][0], centers[i][1], radii[i]] for i in range(32)])
    
    # Refine using a local optimization approach
    # First, apply a greedy procedure to increase radii where possible
    improved = True
    max_iterations = 100
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try to increase each circle's radius one by one
        for i in range(32):
            current_x, current_y, current_r = circles[i]
            
            # Find maximum possible radius for this circle
            max_radius = min(current_x, 1-current_x, current_y, 1-current_y)
            
            # Check overlap constraints with other circles
            for j in range(32):
                if i != j:
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    # Maximum radius such that circles don't overlap
                    max_radius_to_avoid_overlap = distance - r2
                    max_radius = min(max_radius, max_radius_to_avoid_overlap)
            
            # Increase radius if beneficial
            if max_radius > current_r:
                new_r = min(max_radius, current_r + 0.01)
                if new_r > current_r:
                    circles[i] = [current_x, current_y, new_r]
                    improved = True
    
    # Final local optimization using scipy minimize
    # Define objective function to maximize sum of radii (minimize negative sum)
    def objective(params):
        # Reshape params back to circles format
        circles_local = params.reshape(-1, 3)
        return -np.sum(circles_local[:, 2])  # Negative because we want to maximize
    
    # Define constraints
    def constraint_containment(j):
        def func(params):
            circles_local = params.reshape(-1, 3)
            x, y, r = circles_local[j]
            # Circle must be fully contained
            return min(x - r, 1 - x - r, y - r, 1 - y - r)
        return func
    
    def constraint_overlap(i, j):
        def func(params):
            circles_local = params.reshape(-1, 3)
            x1, y1, r1 = circles_local[i]
            x2, y2, r2 = circles_local[j]
            # Distance between centers minus radii should be >= 0
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            return dist - r1 - r2
        return func
    
    # Create constraints
    constraints = []
    
    # Add containment constraints
    for i in range(32):
        constraints.append({'type': 'ineq', 'fun': constraint_containment(i)})
    
    # Add overlap constraints
    for i, j in combinations(range(32), 2):
        constraints.append({'type': 'ineq', 'fun': constraint_overlap(i, j)})
    
    # Flatten initial circles to parameters
    initial_params = circles.flatten()
    
    # Optimize using L-BFGS-B method
    try:
        result = minimize(objective, initial_params, method='L-BFGS-B', 
                         constraints=constraints, options={'maxiter': 1000})
        
        if result.success:
            circles = result.x.reshape(-1, 3)
    except:
        pass  # If optimization fails, keep the current solution
    
    # Final cleanup - ensure all constraints are met
    # Ensure containment
    for i in range(32):
        x, y, r = circles[i]
        circles[i] = [x, y, min(r, x, 1-x, y, 1-y)]
    
    # Ensure no overlaps
    for i in range(32):
        x1, y1, r1 = circles[i]
        for j in range(i+1, 32):
            x2, y2, r2 = circles[j]
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if dist < r1 + r2:
                # Reduce radii to avoid overlap
                total_reduction = (r1 + r2 - dist) * 0.5
                circles[i] = [x1, y1, max(0.001, r1 - total_reduction)]
                circles[j] = [x2, y2, max(0.001, r2 - total_reduction)]
    
    return circles

# EVOLVE-BLOCK-END