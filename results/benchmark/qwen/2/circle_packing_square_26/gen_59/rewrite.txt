# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import math

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def validate_circles(circles: np.ndarray) -> bool:
    """
    Validates that all circles are within bounds and don't overlap.
    """
    n = len(circles)

    # Check containment constraints first
    for i in range(n):
        x, y, r = circles[i]
        if x < r or x > 1 - r or y < r or y > 1 - r:
            return False

    # Check overlap constraints using pairwise distance calculation
    for i in range(n):
        x1, y1, r1 = circles[i]
        for j in range(i+1, n):
            x2, y2, r2 = circles[j]
            distance_sq = (x1 - x2)**2 + (y1 - y2)**2
            min_distance_sq = (r1 + r2)**2
            
            if distance_sq < min_distance_sq:
                return False
                
    return True

def calculate_sum_radii(circles: np.ndarray) -> float:
    """Calculate the sum of all radii"""
    return np.sum(circles[:, 2])

def initialize_voronoi_circles(n_circles: int) -> np.ndarray:
    """Initialize circles using a Voronoi-based approach for better spatial distribution"""
    circles = np.zeros((n_circles, 3))
    
    # Generate initial points using a low-discrepancy sequence for good coverage
    # Using a modified spiral pattern that distributes points well
    for i in range(n_circles):
        # Spiral-based positioning with randomness
        angle = 2 * np.pi * i / n_circles
        radius = 0.4 * (1.0 - 0.8 * (i / (n_circles - 1))) if n_circles > 1 else 0.5
        x = 0.5 + radius * np.cos(angle) * 0.8
        y = 0.5 + radius * np.sin(angle) * 0.8
        
        # Add some randomness to avoid perfect patterns
        x += np.random.normal(0, 0.02)
        y += np.random.normal(0, 0.02)
        
        # Ensure within bounds
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        
        # Initial radius estimation based on distance to edges
        max_radius = min(x, 1-x, y, 1-y)
        r = max(0.01, min(0.1, max_radius * 0.3))
        
        circles[i] = [x, y, r]
    
    return circles

def compute_voronoi_centroids(points: np.ndarray, bounds=(0, 1)) -> np.ndarray:
    """Compute centroids of Voronoi cells for given points"""
    # Create extended points to handle edge cases
    extended_points = np.vstack([
        points,
        [-1, -1], [-1, 2], [2, -1], [2, 2],  # Four corner points
        [-1, 0.5], [2, 0.5], [0.5, -1], [0.5, 2]  # Four side points
    ])
    
    try:
        vor = Voronoi(extended_points)
        
        # Only consider the first n points (original points)
        centroids = []
        for i in range(len(points)):
            region = vor.regions[vor.point_region[i]]
            if -1 not in region and len(region) > 0:
                # Compute centroid of the Voronoi cell
                cell_points = np.array([vor.vertices[j] for j in region if j >= 0])
                # Clip to bounds
                cell_points = np.clip(cell_points, bounds[0], bounds[1])
                if len(cell_points) > 1:
                    centroid = np.mean(cell_points, axis=0)
                else:
                    centroid = points[i]
            else:
                # Default to original point if Voronoi computation fails
                centroid = points[i]
            
            centroids.append(centroid)
        
        return np.array(centroids)
    except:
        # Fallback to simple averaging if Voronoi fails
        return points.copy()

def optimize_positions_voronoi(circles: np.ndarray, max_iterations: int = 10) -> np.ndarray:
    """Optimize circle positions using Voronoi-based guidance"""
    optimized = circles.copy()
    
    for iteration in range(max_iterations):
        # Compute Voronoi centroids for current positions
        positions = optimized[:, :2]
        centroids = compute_voronoi_centroids(positions)
        
        # Update positions towards Voronoi centroids with constraints
        for i in range(len(optimized)):
            # Move towards Voronoi centroid but respect boundaries
            target_x = centroids[i][0]
            target_y = centroids[i][1]
            
            # Apply some damping to avoid violent movements
            damping = 0.3
            optimized[i, 0] += damping * (target_x - optimized[i, 0])
            optimized[i, 1] += damping * (target_y - optimized[i, 1])
            
            # Keep within bounds respecting radius
            r = optimized[i, 2]
            optimized[i, 0] = np.clip(optimized[i, 0], r, 1 - r)
            optimized[i, 1] = np.clip(optimized[i, 1], r, 1 - r)
        
        # Refine using local optimization for overlaps
        refined = optimized.copy()
        converged = True
        
        for i in range(len(refined)):
            for j in range(i+1, len(refined)):
                x1, y1, r1 = refined[i]
                x2, y2, r2 = refined[j]
                
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                
                if dist_sq < min_dist_sq:
                    converged = False
                    # Move circles apart
                    if dist_sq > 0.0001:
                        dx = (x1 - x2) / np.sqrt(dist_sq)
                        dy = (y1 - y2) / np.sqrt(dist_sq)
                        
                        move_amount = (min_dist_sq - dist_sq) * 0.1
                        
                        # Apply movement proportional to radii
                        total_r = r1 + r2
                        refined[i, 0] += dx * move_amount * (r2 / total_r)
                        refined[i, 1] += dy * move_amount * (r2 / total_r)
                        refined[j, 0] -= dx * move_amount * (r1 / total_r)
                        refined[j, 1] -= dy * move_amount * (r1 / total_r)
                        
                        # Clamp to bounds
                        refined[i, 0] = np.clip(refined[i, 0], r1, 1 - r1)
                        refined[i, 1] = np.clip(refined[i, 1], r1, 1 - r1)
                        refined[j, 0] = np.clip(refined[j, 0], r2, 1 - r2)
                        refined[j, 1] = np.clip(refined[j, 1], r2, 1 - r2)
        
        optimized = refined
        
        if converged:
            break
    
    return optimized

def optimize_radii_local(circles: np.ndarray, max_iterations: int = 5) -> np.ndarray:
    """Locally optimize radii to increase sum while maintaining validity"""
    optimized = circles.copy()
    
    for iteration in range(max_iterations):
        improved = False
        
        # For each circle, try to increase radius if possible
        for i in range(len(optimized)):
            x, y, r = optimized[i]
            
            # Find nearest neighbors and get minimum distance to their boundaries
            min_dist_to_edge = min(x, 1-x, y, 1-y)
            
            # Calculate maximum allowable radius
            max_radius = min_dist_to_edge
            
            # Consider distances to other circles
            for j in range(len(optimized)):
                if i != j:
                    x2, y2, r2 = optimized[j]
                    dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if dist > 0:  # Avoid self-distance
                        max_radius = min(max_radius, dist - r2)
            
            # Try to increase radius (with some tolerance for numerical issues)
            if max_radius > r + 1e-6:
                # Increase radius but limit the change
                delta_r = min(0.02, max_radius - r)
                if delta_r > 1e-6:
                    optimized[i, 2] = r + delta_r
                    improved = True
        
        if not improved:
            break
    
    return optimized

def construct_better_initial_solution(n_circles: int) -> np.ndarray:
    """Construct a starting solution with better spatial distribution"""
    # Start with a few circles placed optimally
    circles = initialize_voronoi_circles(n_circles)
    
    # Perform a few optimization rounds
    for _ in range(3):
        circles = optimize_positions_voronoi(circles, max_iterations=5)
        circles = optimize_radii_local(circles, max_iterations=3)
    
    # Validate and fix
    if not validate_circles(circles):
        # Try several random repairs
        for _ in range(10):
            circles = optimize_positions_voronoi(circles, max_iterations=3)
            circles = optimize_radii_local(circles, max_iterations=2)
            if validate_circles(circles):
                break
    
    return circles

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n_circles = 26
    
    # Construct a good initial solution using Voronoi-guided approach
    best_circles = construct_better_initial_solution(n_circles)
    best_sum = calculate_sum_radii(best_circles)
    
    # Perform multiple improvement passes
    for pass_num in range(5):
        # Try several randomized approaches
        for attempt in range(10):
            # Create a slightly perturbed version
            circles = best_circles.copy()
            
            # Add some random perturbations to positions and radii
            for i in range(n_circles):
                # Perturb position
                circles[i, 0] += np.random.normal(0, 0.01)
                circles[i, 1] += np.random.normal(0, 0.01)
                
                # Perturb radius
                circles[i, 2] += np.random.normal(0, 0.005)
                
                # Keep within bounds
                r = circles[i, 2]
                circles[i, 0] = np.clip(circles[i, 0], r, 1 - r)
                circles[i, 1] = np.clip(circles[i, 1], r, 1 - r)
                circles[i, 2] = max(0.001, r)
            
            # Optimize with Voronoi-based techniques
            circles = optimize_positions_voronoi(circles, max_iterations=8)
            circles = optimize_radii_local(circles, max_iterations=5)
            
            # Validate and check improvement
            if validate_circles(circles):
                current_sum = calculate_sum_radii(circles)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = circles.copy()
        
        # After each pass, do a more thorough optimization
        best_circles = optimize_positions_voronoi(best_circles, max_iterations=10)
        best_circles = optimize_radii_local(best_circles, max_iterations=10)
        
        if not validate_circles(best_circles):
            # If corrupted, restore from previous valid state
            pass
    
    # Final validation and cleanup
    if not validate_circles(best_circles):
        # Create a fallback valid solution
        circles = np.zeros((n_circles, 3))
        # Distribute circles in a grid pattern
        grid_size = int(np.ceil(np.sqrt(n_circles)))
        spacing = 1.0 / grid_size
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= n_circles:
                    break
                x = (i + 0.5) * spacing
                y = (j + 0.5) * spacing
                r = 0.02
                circles[idx] = [x, y, r]
                idx += 1
        return circles
    
    return best_circles

# EVOLVE-BLOCK-END