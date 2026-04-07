# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import math
import random
from numba import jit

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Container setup (perimeter = 4, so width + height = 2)
    container_width, container_height = 1.2, 0.8

    # Number of circles
    n = 21

    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Phase 1: Voronoi-based Initialization
    # Generate strategic anchor points (corners and edges)
    anchors = []
    # Corners
    anchors.extend([(0.05, 0.05), (container_width - 0.05, 0.05),
                   (0.05, container_height - 0.05), (container_width - 0.05, container_height - 0.05)])
    # Edge centers
    anchors.extend([(container_width/2, 0.05), (container_width/2, container_height - 0.05),
                   (0.05, container_height/2), (container_width - 0.05, container_height/2)])

    # Generate candidate points for Voronoi
    candidate_points = []
    grid_density = 48
    for i in range(grid_density):
        for j in range(grid_density):
            x = (i + 0.5) / grid_density * container_width
            y = (j + 0.5) / grid_density * container_height
            candidate_points.append((x, y))

    # Add random points for diversity
    for _ in range(100):
        x = np.random.uniform(0.05, container_width - 0.05)
        y = np.random.uniform(0.05, container_height - 0.05)
        candidate_points.append((x, y))

    # Combine anchors with candidates and use k-means++ to get well-distributed seeds
    all_points = anchors + candidate_points
    all_array = np.array(all_points)
    
    # Use k-means++ to generate good initial placements
    kmeans = KMeans(n_clusters=min(len(all_points), n), init='k-means++', n_init=20, random_state=42)
    kmeans.fit(all_array)
    initial_positions = kmeans.cluster_centers_
    
    # If we have more positions than needed, keep only n of them
    if len(initial_positions) > n:
        initial_positions = initial_positions[:n]
    
    # Initialize circles with positions and small radii
    circles = np.zeros((n, 3))
    for i in range(n):
        x, y = initial_positions[i]
        circles[i] = [x, y, 0.02]

    # Phase 2: Voronoi-guided Optimization
    best_sum_radii = 0
    best_circles = None
    
    # Create Voronoi diagram for current configuration
    def compute_voronoi_circles(positions, radii):
        """Compute maximum possible radius for each circle given Voronoi cells"""
        vor = Voronoi(positions)
        new_radii = np.copy(radii)
        
        # For each Voronoi cell, compute maximum inscribed circle
        for i, (x, y) in enumerate(positions):
            # Get vertices of Voronoi cell around this point
            cell_vertices = []
            
            # Find all facets belonging to this cell
            for j, ridge in enumerate(vor.ridge_vertices):
                if -1 in ridge:  # Skip infinite ridges
                    continue
                # Check if this ridge belongs to our point
                if i in vor.ridge_points[j]:
                    # Collect vertices of this ridge
                    vertices = [vor.vertices[k] for k in ridge if k >= 0]
                    if vertices:
                        cell_vertices.extend(vertices)
            
            # If we have vertices, compute maximum inscribed circle
            if len(cell_vertices) >= 3:
                try:
                    # Simple polygon area approach to estimate space
                    # But let's just compute the minimum distance to boundaries
                    min_boundary_dist = min([
                        x, container_width - x, 
                        y, container_height - y
                    ])
                    
                    # Also check distance to other circles
                    for k, (ox, oy) in enumerate(positions):
                        if k != i:
                            dist = math.sqrt((x - ox)**2 + (y - oy)**2)
                            # We want to leave at least radius of both circles between them
                            min_boundary_dist = min(min_boundary_dist, dist - radii[k])
                    
                    # Ensure we don't go below minimum safe value
                    new_radii[i] = max(0.001, min(min_boundary_dist * 0.9, 0.4))
                except:
                    pass
            
            # Default case for simple computation
            else:
                min_boundary_dist = min([
                    x, container_width - x, 
                    y, container_height - y
                ])
                
                # Check distance to other circles
                for k, (ox, oy) in enumerate(positions):
                    if k != i:
                        dist = math.sqrt((x - ox)**2 + (y - oy)**2)
                        min_boundary_dist = min(min_boundary_dist, dist - radii[k])
                        
                new_radii[i] = max(0.001, min(min_boundary_dist * 0.9, 0.4))
        
        return new_radii

    # Phase 2b: Constraint-aware optimization using coordinate ascent
    # We'll perform alternating optimization
    max_iterations = 1000
    for iteration in range(max_iterations):
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Periodically save best solution
        if iteration % 50 == 0:
            current_sum = np.sum(radii)
            if current_sum > best_sum_radii:
                best_sum_radii = current_sum
                best_circles = circles.copy()
        
        # Optimizing radii based on Voronoi-like constraints
        # For each circle, compute max possible radius
        new_radii = np.zeros(n)
        for i in range(n):
            x, y = positions[i]
            
            # Boundary constraints
            boundary_radius = min(x, container_width - x, y, container_height - y)
            
            # Circle-to-circle constraints
            min_overlap_radius = boundary_radius
            for j in range(n):
                if i != j:
                    dx = x - positions[j, 0]
                    dy = y - positions[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist > 0.001:  # Avoid divide by zero
                        # Minimum radius to avoid overlap
                        overlap_radius = dist - radii[j]
                        min_overlap_radius = min(min_overlap_radius, overlap_radius)
            
            # Take the minimum among all constraints
            max_radius = min(boundary_radius, min_overlap_radius, 0.4)
            new_radii[i] = max(0.001, max_radius * 0.95)  # Small safety factor
        
        # Update radii
        circles[:, 2] = new_radii
        
        # Optimizing positions using gradient information
        # Create a simple optimization function
        def optimize_positions():
            # Simple coordinate ascent approach
            updated_positions = np.copy(positions)
            learning_rate = 0.1
            
            for i in range(n):
                x, y = positions[i]
                old_x, old_y = x, y
                
                # Compute attractive forces from neighbors (repulsion concept)
                # and boundary forces
                fx, fy = 0.0, 0.0
                
                # Add repulsion from overlapping circles
                for j in range(n):
                    if i != j:
                        dx = positions[j, 0] - x
                        dy = positions[j, 1] - y
                        dist = math.sqrt(dx*dx + dy*dy)
                        
                        if dist < (radii[i] + radii[j]) and dist > 0.001:
                            # Force to separate
                            force_magnitude = (radii[i] + radii[j] - dist) / (dist + 1e-8)
                            fx += force_magnitude * dx
                            fy += force_magnitude * dy
                
                # Add boundary forces
                if x < radii[i]:
                    fx += (radii[i] - x) * 2.0
                if x + radii[i] > container_width:
                    fx -= (x + radii[i] - container_width) * 2.0
                if y < radii[i]:
                    fy += (radii[i] - y) * 2.0
                if y + radii[i] > container_height:
                    fy -= (y + radii[i] - container_height) * 2.0
                
                # Apply movement with bounds
                new_x = max(radii[i], min(container_width - radii[i], x + learning_rate * fx))
                new_y = max(radii[i], min(container_height - radii[i], y + learning_rate * fy))
                
                updated_positions[i] = [new_x, new_y]
            
            return updated_positions
        
        # Update positions
        circles[:, :2] = optimize_positions()
        
        # Occasionally recompute to prevent getting stuck
        if iteration % 100 == 0 and iteration > 0:
            # Reinitialize with slightly perturbed versions
            for i in range(n):
                x, y, r = circles[i]
                circles[i] = [
                    max(r, min(container_width - r, x + np.random.normal(0, 0.01))),
                    max(r, min(container_height - r, y + np.random.normal(0, 0.01))),
                    r
                ]

    # Phase 3: Final refinement with constrained optimization
    # Use a more sophisticated approach for final optimization
    def objective_function(params):
        # params contains [x1, y1, r1, x2, y2, r2, ..., x21, y21, r21]
        circles_local = params.reshape(-1, 3)
        
        # Compute sum of radii (negative because we want to maximize)
        sum_radii = -np.sum(circles_local[:, 2])
        
        # Add penalty for constraint violations
        penalty = 0
        for i in range(n):
            x, y, r = circles_local[i]
            
            # Boundary penalties
            if x < r or x > container_width - r or y < r or y > container_height - r:
                penalty += 1000
            
            # Overlap penalties
            for j in range(i+1, n):
                x1, y1, r1 = circles_local[i]
                x2, y2, r2 = circles_local[j]
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < (r1 + r2) and dist > 0:
                    penalty += 1000 * (r1 + r2 - dist)
        
        return sum_radii + penalty
    
    # Final optimization using scipy minimize
    initial_params = circles.flatten()
    
    # Use L-BFGS-B for bounded optimization
    try:
        result = minimize(
            objective_function, 
            initial_params, 
            method='L-BFGS-B',
            bounds=[(r, container_width - r) if i % 3 == 0 or i % 3 == 1 else 
                   (0.001, 0.4) for i, r in enumerate(circles.flatten())],
            options={'maxiter': 500}
        )
        
        # Update with optimized solution if better
        optimized_circles = result.x.reshape(-1, 3)
        final_sum = np.sum(optimized_circles[:, 2])
        if final_sum > best_sum_radii:
            circles = optimized_circles
    except:
        pass  # If optimization fails, stick with previous best

    # Final cleanup
    for i in range(n):
        x, y, r = circles[i]
        # Ensure circles stay within bounds
        x = max(r, min(container_width - r, x))
        y = max(r, min(container_height - r, y))
        circles[i] = [x, y, r]

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")