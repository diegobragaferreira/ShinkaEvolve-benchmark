# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.optimize import minimize
import random
from copy import deepcopy

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    n = 26
    
    # Stage 1: Generate initial configuration using Voronoi-based approach
    def generate_voronoi_initial():
        # Generate points that roughly form a Voronoi diagram
        # Create a grid of points and perturb them
        points = []
        grid_size = int(np.ceil(np.sqrt(n)))
        
        # Create regular grid points
        for i in range(grid_size):
            for j in range(grid_size):
                x = (i + 0.5) / grid_size
                y = (j + 0.5) / grid_size
                # Add small random perturbation
                x += (np.random.random() - 0.5) * 0.1
                y += (np.random.random() - 0.5) * 0.1
                # Clamp to valid range
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                points.append([x, y])
                
        # If we have too many points, keep only n of them
        if len(points) > n:
            points = points[:n]
        elif len(points) < n:
            # Fill with random points
            while len(points) < n:
                x = np.random.random()
                y = np.random.random()
                points.append([x, y])
        
        return np.array(points[:n])
    
    # Stage 2: Initialize circles with reasonable radii
    def initialize_circles(points):
        circles = np.zeros((len(points), 3))
        for i, (x, y) in enumerate(points):
            # Start with a reasonable initial radius
            circles[i] = [x, y, min(x, 1-x, y, 1-y) * 0.4]
        return circles
    
    # Stage 3: Check feasibility and fix overlaps
    def check_feasibility(circles):
        n = len(circles)
        # Check containment
        for i in range(n):
            x, y, r = circles[i]
            if r > x or r > (1-x) or r > y or r > (1-y):
                return False
                
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if dist < r1 + r2:
                    return False
        return True
    
    # Stage 4: Refinement function to optimize radii while keeping positions fixed
    def refine_radii(circles):
        # Simple greedy approach: increase radii while respecting constraints
        new_circles = circles.copy()
        changed = True
        iteration = 0
        
        while changed and iteration < 100:
            changed = False
            iteration += 1
            
            # Try increasing each radius
            for i in range(len(new_circles)):
                x, y, r = new_circles[i]
                # Calculate maximum possible radius
                max_radius = min(x, 1-x, y, 1-y)
                
                # Check overlap constraints
                for j in range(len(new_circles)):
                    if i != j:
                        x2, y2, r2 = new_circles[j]
                        dist = np.sqrt((x-x2)**2 + (y-y2)**2)
                        max_radius = min(max_radius, dist - r2)
                
                if max_radius > r:
                    new_r = min(max_radius, r + 0.01)
                    if new_r > r + 1e-6:
                        new_circles[i, 2] = new_r
                        changed = True
                        
        return new_circles
    
    # Stage 5: Multi-start local optimization
    def optimize_config(circles):
        def objective(args):
            # args contains [x1,y1,r1,x2,y2,r2,...]
            new_circles = circles.copy()
            for i in range(len(circles)):
                new_circles[i] = [args[3*i], args[3*i+1], args[3*i+2]]
            
            # Calculate negative of sum of radii (since we want to maximize)
            total_radius = sum(circle[2] for circle in new_circles)
            
            # Penalty for violations
            penalty = 0
            
            # Check containment violations
            for i in range(len(new_circles)):
                x, y, r = new_circles[i]
                if r > x or r > (1-x) or r > y or r > (1-y):
                    penalty += 1000
                    
            # Check overlap violations
            for i in range(len(new_circles)):
                for j in range(i+1, len(new_circles)):
                    x1, y1, r1 = new_circles[i]
                    x2, y2, r2 = new_circles[j]
                    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    if dist < r1 + r2:
                        penalty += 1000 * (r1 + r2 - dist)
                        
            return -total_radius + penalty
        
        # Initial guess
        initial_args = []
        for circle in circles:
            initial_args.extend(circle)
            
        # Optimize
        try:
            result = minimize(objective, initial_args, method='L-BFGS-B', 
                            bounds=[(0,1), (0,1), (0,0.5)] * len(circles),
                            options={'maxiter': 500})
            
            if result.success:
                optimized_circles = circles.copy()
                for i in range(len(circles)):
                    optimized_circles[i] = [result.x[3*i], result.x[3*i+1], result.x[3*i+2]]
                return optimized_circles
        except:
            pass
            
        return circles
    
    # Main algorithm
    best_sum = 0
    best_circles = None
    
    # Try multiple random initializations with Voronoi approach
    for trial in range(10):
        # Generate initial Voronoi-based configuration
        seed_points = generate_voronoi_initial()
        circles = initialize_circles(seed_points)
        
        # Refine radii
        circles = refine_radii(circles)
        
        # Optimize with local search
        final_circles = optimize_config(circles)
        
        # Calculate sum of radii
        total_radius = sum(circle[2] for circle in final_circles)
        
        if total_radius > best_sum:
            best_sum = total_radius
            best_circles = final_circles.copy()
    
    # Final refinement
    if best_circles is not None:
        # Apply one more optimization pass
        final_result = optimize_config(best_circles)
        return final_result
    
    # Fallback to simple method if nothing worked
    fallback_circles = np.zeros((n, 3))
    # Place circles in a grid pattern with decreasing sizes
    grid_size = int(np.ceil(np.sqrt(n)))
    count = 0
    for i in range(grid_size):
        if count >= n:
            break
        for j in range(grid_size):
            if count >= n:
                break
            x = (i + 0.5) / grid_size
            y = (j + 0.5) / grid_size
            r = min(x, 1-x, y, 1-y) * 0.3
            fallback_circles[count] = [x, y, r]
            count += 1
    
    return fallback_circles

# EVOLVE-BLOCK-END
