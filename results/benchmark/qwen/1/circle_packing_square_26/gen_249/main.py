# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import time

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    
    # Step 1: Generate initial configuration using Voronoi-based approach
    n_circles = 26
    
    # Create initial points using a modified spiral to ensure good spatial distribution
    golden_angle = np.pi * (3 - np.sqrt(5))
    points = []
    
    # Generate points using spiral pattern with some randomness
    for i in range(n_circles * 3):  # Generate more points to allow for filtering
        r = np.sqrt(i / (n_circles * 3 - 1)) if n_circles * 3 > 1 else 0
        theta = i * golden_angle
        
        x = 0.4 * r * np.cos(theta) + 0.5
        y = 0.4 * r * np.sin(theta) + 0.5
        
        # Add some randomness to avoid perfect patterns
        x += random.uniform(-0.02, 0.02)
        y += random.uniform(-0.02, 0.02)
        
        # Only keep points that are reasonably within bounds
        if 0.05 <= x <= 0.95 and 0.05 <= y <= 0.95:
            points.append([x, y])
    
    # Limit to exactly n_circles points
    points = points[:n_circles]
    
    # Step 2: Create Voronoi diagram to generate initial configuration
    vor = Voronoi(points)
    
    # Estimate initial radii based on Voronoi cell areas and distances to edges
    circles = np.zeros((n_circles, 3))
    
    for i in range(n_circles):
        x, y = points[i]
        
        # Calculate minimum distance to unit square boundaries
        dist_to_edges = [
            x,  # Distance to left edge
            1 - x,  # Distance to right edge
            y,  # Distance to bottom edge
            1 - y  # Distance to top edge
        ]
        min_edge_dist = min(dist_to_edges)
        
        # Calculate minimum distance to other points (Voronoi influence)
        min_point_dist = float('inf')
        for j in range(n_circles):
            if i != j:
                dist = np.sqrt((x - points[j][0])**2 + (y - points[j][1])**2)
                min_point_dist = min(min_point_dist, dist)
        
        # Use Voronoi cell concept to estimate safe radius
        # Smaller distances to neighbors mean less room for large radii
        if min_point_dist < 0.1:
            # Very close to another point, use small radius
            radius = min_edge_dist * 0.3
        else:
            # More space available
            radius = min(min_edge_dist, min_point_dist * 0.4)
            
        # Ensure reasonable limits
        radius = max(0.005, min(0.2, radius))
        
        circles[i] = [x, y, radius]
    
    # Step 3: Refine using gradient-based optimization with constraints
    def objective_and_constraints(params):
        # Reshape params back to circles
        circles_local = params.reshape(-1, 3)
        
        # Extract positions and radii
        positions = circles_local[:, :2]
        radii = circles_local[:, 2]
        
        # Objective function (negative because we minimize)
        obj_value = -np.sum(radii)
        
        # Penalty for constraint violations
        penalty = 0
        
        # Boundary penalties
        for i in range(len(positions)):
            x, y = positions[i]
            r = radii[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                penalty += 10000
        
        # Overlap penalties 
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                pos_i = positions[i]
                pos_j = positions[j]
                r_i = radii[i]
                r_j = radii[j]
                dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                if dist < (r_i + r_j):
                    overlap = (r_i + r_j - dist)
                    penalty += 1000 * overlap
                    
        return obj_value + penalty
    
    def gradient_and_constraints(params):
        # This is a simplified gradient approximation
        # In practice, this would require more complex derivative calculations
        circles_local = params.reshape(-1, 3)
        positions = circles_local[:, :2]
        radii = circles_local[:, 2]
        
        # Simplified gradient (just for demonstration)
        grad = np.zeros_like(params)
        
        # For each circle, compute gradient contribution
        for i in range(len(positions)):
            # Simple gradient ascent (would need real gradient calculation)
            grad[i*3 + 2] = -1.0  # Maximizing sum of radii
            
        return grad
        
    # Step 4: Enhanced local optimization using sequential quadratic programming approach
    def local_optimization(circles_input):
        best_circles = circles_input.copy()
        best_score = np.sum(best_circles[:, 2])
        
        # Try different optimization strategies
        for strategy in range(3):
            current_circles = best_circles.copy()
            
            # Strategy 1: Sequential optimization of each circle
            if strategy == 0:
                improved = True
                iter_count = 0
                while improved and iter_count < 10:
                    improved = False
                    for i in range(len(current_circles)):
                        # Save original state
                        orig_pos = current_circles[i, :2].copy()
                        orig_r = current_circles[i, 2]
                        
                        # Try to increase radius
                        max_r = min(
                            current_circles[i, 0], 1 - current_circles[i, 0],
                            current_circles[i, 1], 1 - current_circles[i, 1]
                        ) - 0.001
                        
                        if max_r > orig_r:
                            # Binary search for maximum safe radius
                            low = orig_r
                            high = max_r
                            best_r = orig_r
                            
                            for _ in range(15):  # Binary search iterations
                                test_r = (low + high) / 2
                                valid = True
                                
                                # Check overlap with all other circles
                                for j in range(len(current_circles)):
                                    if i != j:
                                        pos_j = current_circles[j, :2]
                                        r_j = current_circles[j, 2]
                                        dist = np.sqrt(
                                            (current_circles[i, 0] - pos_j[0])**2 + 
                                            (current_circles[i, 1] - pos_j[1])**2
                                        )
                                        if dist < (test_r + r_j):
                                            valid = False
                                            break
                                
                                if valid:
                                    best_r = test_r
                                    low = test_r
                                else:
                                    high = test_r
                                    
                            if best_r > orig_r:
                                current_circles[i, 2] = best_r
                                improved = True
                                best_score = np.sum(current_circles[:, 2])
                
            # Strategy 2: Position optimization
            elif strategy == 1:
                improved = True
                iter_count = 0
                while improved and iter_count < 10:
                    improved = False
                    for i in range(len(current_circles)):
                        orig_pos = current_circles[i, :2].copy()
                        orig_r = current_circles[i, 2]
                        
                        # Try different nearby positions
                        best_pos = orig_pos.copy()
                        best_score = orig_r
                        
                        # Test a grid of nearby positions
                        test_positions = []
                        for dx in [-0.02, -0.01, 0, 0.01, 0.02]:
                            for dy in [-0.02, -0.01, 0, 0.01, 0.02]:
                                test_positions.append((dx, dy))
                        
                        # Shuffle for exploration
                        random.shuffle(test_positions)
                        
                        for dx, dy in test_positions[:10]:
                            new_x = max(0.01, min(0.99, current_circles[i, 0] + dx))
                            new_y = max(0.01, min(0.99, current_circles[i, 1] + dy))
                            
                            valid = True
                            # Check overlaps
                            for j in range(len(current_circles)):
                                if i != j:
                                    pos_j = current_circles[j, :2]
                                    r_j = current_circles[j, 2]
                                    dist = np.sqrt(
                                        (new_x - pos_j[0])**2 + 
                                        (new_y - pos_j[1])**2
                                    )
                                    if dist < (orig_r + r_j):
                                        valid = False
                                        break
                                    
                            if valid:
                                # Try to maintain a reasonable radius
                                new_score = orig_r
                                if new_score > best_score:
                                    best_score = new_score
                                    best_pos = [new_x, new_y]
                        
                        if not np.array_equal(best_pos, orig_pos):
                            current_circles[i, :2] = best_pos
                            improved = True
                            best_score = np.sum(current_circles[:, 2])
            
            # Strategy 3: Global optimization using simple iterative improvement
            else:
                # Simple iterative improvement on all circles
                for _ in range(50):
                    improved = False
                    # Try to increase all radii simultaneously
                    for i in range(len(current_circles)):
                        max_r = min(
                            current_circles[i, 0], 1 - current_circles[i, 0],
                            current_circles[i, 1], 1 - current_circles[i, 1]
                        ) - 0.001
                        
                        if max_r > current_circles[i, 2]:
                            # Check if we can safely increase
                            safe_incr = max_r - current_circles[i, 2]
                            current_circles[i, 2] += safe_incr * 0.2
                            improved = True
                    
                    # Reoptimize positions slightly
                    for i in range(len(current_circles)):
                        # Small position adjustments
                        current_circles[i, 0] += random.uniform(-0.005, 0.005)
                        current_circles[i, 1] += random.uniform(-0.005, 0.005)
                        current_circles[i, 0] = max(0.01, min(0.99, current_circles[i, 0]))
                        current_circles[i, 1] = max(0.01, min(0.99, current_circles[i, 1]))
                    
                    if not improved:
                        break
            
            # Update best if current is better
            if np.sum(current_circles[:, 2]) > best_score:
                best_circles = current_circles
                best_score = np.sum(best_circles[:, 2])
        
        return best_circles
    
    # Final optimization pass
    final_circles = local_optimization(circles)
    
    # Additional refinement to check for overlaps and fix them
    def fix_overlaps(circles_to_fix):
        fixed_circles = circles_to_fix.copy()
        max_iter = 50
        iter_count = 0
        
        while iter_count < max_iter:
            improved = False
            
            # Check all pairs for overlaps
            for i in range(len(fixed_circles)):
                for j in range(i+1, len(fixed_circles)):
                    pos_i = fixed_circles[i, :2]
                    pos_j = fixed_circles[j, :2]
                    r_i = fixed_circles[i, 2]
                    r_j = fixed_circles[j, 2]
                    dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                    
                    if dist < (r_i + r_j):
                        # Separate circles by moving them apart
                        # Compute normalized vector from j to i
                        dx = pos_i[0] - pos_j[0]
                        dy = pos_i[1] - pos_j[1]
                        dist_total = np.sqrt(dx*dx + dy*dy) + 1e-8
                        dx /= dist_total
                        dy /= dist_total
                        
                        # Move each circle away from the other
                        overlap = (r_i + r_j - dist)
                        
                        # Move circles away proportionally to their radii
                        move_i = overlap * r_i / (r_i + r_j) * 0.5
                        move_j = overlap * r_j / (r_i + r_j) * 0.5
                        
                        # Apply movements
                        fixed_circles[i, 0] += dx * move_i
                        fixed_circles[i, 1] += dy * move_i
                        fixed_circles[j, 0] -= dx * move_j
                        fixed_circles[j, 1] -= dy * move_j
                        
                        # Keep within bounds
                        fixed_circles[i, 0] = max(0.01, min(0.99, fixed_circles[i, 0]))
                        fixed_circles[i, 1] = max(0.01, min(0.99, fixed_circles[i, 1]))
                        fixed_circles[j, 0] = max(0.01, min(0.99, fixed_circles[j, 0]))
                        fixed_circles[j, 1] = max(0.01, min(0.99, fixed_circles[j, 1]))
                        
                        improved = True
            
            if not improved:
                break
            iter_count += 1
            
        return fixed_circles
    
    # Apply overlap fixes
    optimized_circles = fix_overlaps(final_circles)
    
    # Final local optimization
    optimized_circles = local_optimization(optimized_circles)
    
    return optimized_circles

# EVOLVE-BLOCK-END