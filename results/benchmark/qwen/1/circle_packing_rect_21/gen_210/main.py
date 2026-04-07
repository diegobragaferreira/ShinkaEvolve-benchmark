# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, cKDTree
from scipy.optimize import minimize
import random
from typing import Tuple, List
import time

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Rectangle dimensions (width + height = 2)
    rect_width = 1.0
    rect_height = 1.0

    # Number of circles
    n = 21

    def compute_voronoi_cell_areas(points, rect_width, rect_height):
        """Compute Voronoi cell areas for each point, accounting for boundaries"""
        # Add boundary points to make Voronoi meaningful at edges
        boundary_points = [
            [0, 0], [rect_width, 0], [0, rect_height], [rect_width, rect_height],
            [rect_width/2, 0], [rect_width/2, rect_height],
            [0, rect_height/2], [rect_width, rect_height/2]
        ]
        extended_points = np.vstack([points, boundary_points])
        
        try:
            vor = Voronoi(extended_points)
            
            # For each original point, compute Voronoi cell area
            areas = []
            for i in range(len(points)):
                region_idx = vor.point_region[i]
                if region_idx != -1 and region_idx < len(vor.regions):
                    region = vor.regions[region_idx]
                    if -1 not in region and len(region) >= 3:
                        # Compute area using shoelace formula
                        vertices = np.array([vor.vertices[j] for j in region if j < len(vor.vertices)])
                        if len(vertices) >= 3:
                            x = vertices[:, 0]
                            y = vertices[:, 1]
                            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                            areas.append(area)
                        else:
                            areas.append(1.0)
                    else:
                        areas.append(1.0)
                else:
                    areas.append(1.0)
            return np.array(areas)
        except:
            # Fallback
            return np.ones(len(points))

    def compute_forces(circles, rect_width, rect_height):
        """Compute forces acting on each circle based on Voronoi geometry"""
        n = len(circles)
        forces = np.zeros((n, 2))
        
        # Get centers
        centers = circles[:, :2]
        
        # Compute Voronoi cell areas for constraint density
        cell_areas = compute_voronoi_cell_areas(centers, rect_width, rect_height)
        
        # Normalize cell areas for force scaling (smaller areas = higher constraint density)
        normalized_areas = cell_areas / np.mean(cell_areas)
        
        # Compute pair-wise repulsion forces
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = centers[i]
                x2, y2 = centers[j]
                r1, r2 = circles[i, 2], circles[j, 2]
                
                # Distance between circle centers
                dx = x2 - x1
                dy = y2 - y1
                distance = np.sqrt(dx*dx + dy*dy)
                
                # Only consider if circles overlap or are close
                if distance < (r1 + r2):
                    # Overlapping case - strong repulsion
                    force_magnitude = 100.0 / (distance + 0.001)
                    fx = force_magnitude * dx / (distance + 0.001)
                    fy = force_magnitude * dy / (distance + 0.001)
                    forces[i] += np.array([fx, fy])
                    forces[j] -= np.array([fx, fy])
                elif distance < 2*(r1 + r2):  # Near contact case
                    # Gentle repulsion based on proximity
                    force_magnitude = 1.0 / (distance + 0.001)
                    fx = force_magnitude * dx / (distance + 0.001)
                    fy = force_magnitude * dy / (distance + 0.001)
                    forces[i] += np.array([fx, fy])
                    forces[j] -= np.array([fx, fy])
        
        # Boundary forces - push away from edges with force proportional to inverse cell area
        for i in range(n):
            x, y, r = circles[i]
            # Force towards center of box scaled by constraint density
            boundary_force = 0.1 * normalized_areas[i]
            
            # Left boundary
            if x < r:
                forces[i][0] += boundary_force * (r - x)
            # Right boundary  
            if x > rect_width - r:
                forces[i][0] -= boundary_force * (x - (rect_width - r))
            # Bottom boundary
            if y < r:
                forces[i][1] += boundary_force * (r - y)
            # Top boundary
            if y > rect_height - r:
                forces[i][1] -= boundary_force * (y - (rect_height - r))
        
        return forces

    def generate_initial_pattern():
        """Generate initial pattern using Voronoi-guided placement"""
        circles = np.zeros((n, 3))
        
        # Start with roughly uniform distribution
        max_radius = 0.1
        
        # Place circles in Voronoi-low-density areas
        # Try to fill space evenly
        attempts = 0
        placed_count = 0
        
        # Generate candidate positions
        candidates = []
        grid_size = 5
        for i in range(grid_size):
            for j in range(grid_size):
                x = (j + 0.5) * rect_width / grid_size
                y = (i + 0.5) * rect_height / grid_size
                candidates.append((x, y))
                
        # Shuffle candidates to avoid systematic bias
        random.shuffle(candidates)
        
        for cx, cy in candidates:
            if placed_count >= n:
                break
                
            # Check if this location is suitable
            valid = True
            for i in range(placed_count):
                px, py, pr = circles[i]
                distance = np.sqrt((cx - px)**2 + (cy - py)**2)
                if distance < (max_radius + pr):
                    valid = False
                    break
                    
            if valid:
                circles[placed_count] = [cx, cy, max_radius]
                placed_count += 1
                
        # Fill remaining with random valid positions
        while placed_count < n:
            attempts = 0
            valid = False
            while not valid and attempts < 1000:
                x = np.random.uniform(0.05, rect_width - 0.05)
                y = np.random.uniform(0.05, rect_height - 0.05)
                radius = np.random.uniform(0.01, max_radius)
                
                valid = True
                for i in range(placed_count):
                    px, py, pr = circles[i]
                    distance = np.sqrt((x - px)**2 + (y - py)**2)
                    if distance < (radius + pr):
                        valid = False
                        break
                        
                if valid:
                    circles[placed_count] = [x, y, radius]
                    placed_count += 1
                    
        return circles

    def validate_solution(circles, rect_width, rect_height):
        """Ensure solution is valid"""
        # Check boundaries
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
                return False
                
        # Check overlaps
        if len(circles) > 1:
            centers = circles[:, :2]
            radii = circles[:, 2]
            distances = np.sqrt(np.sum((centers[:, None] - centers[None, :])**2, axis=2))
            overlap_distances = distances[np.triu(np.ones_like(distances, dtype=bool), k=1)]
            overlap_radii = (radii[:, None] + radii[None, :])[np.triu(np.ones_like(distances, dtype=bool), k=1)]
            if np.any(overlap_distances < overlap_radii):
                return False
        return True

    def smooth_and_relax(circles, rect_width, rect_height, iterations=100):
        """Apply physics-inspired smoothing with Voronoi-guided forces"""
        current = circles.copy()
        
        # Progressive relaxation - start with coarse adjustments
        for iter_num in range(iterations):
            # Adaptive step size based on iteration (start large, decrease)
            step_size = 0.1 * (1.0 - iter_num / iterations) + 0.01
            
            # Compute forces
            forces = compute_forces(current, rect_width, rect_height)
            
            # Apply forces with adaptive damping
            damping = 0.1 + 0.9 * (1.0 - iter_num / iterations)
            
            for i in range(len(current)):
                x, y, r = current[i]
                
                # Apply force to position
                new_x = x + forces[i][0] * step_size * damping
                new_y = y + forces[i][1] * step_size * damping
                
                # Keep within bounds
                new_x = np.clip(new_x, r, rect_width - r)
                new_y = np.clip(new_y, r, rect_height - r)
                
                current[i] = [new_x, new_y, r]
                
            # Occasionally adjust radii to maximize sum (but respect constraints)
            if iter_num % 10 == 0:
                # Simple greedy radius increase for unconstrained circles
                for i in range(len(current)):
                    x, y, r = current[i]
                    # Only try to increase radius if we can safely do so
                    safe_radius = min(x, y, rect_width - x, rect_height - y)
                    # Only increase if there's room and we're not too close to neighbors
                    neighbors = []
                    centers = current[:, :2]
                    radii = current[:, 2]
                    for j in range(len(current)):
                        if i != j:
                            dist = np.sqrt((x - centers[j, 0])**2 + (y - centers[j, 1])**2)
                            if dist < (r + radii[j]):
                                neighbors.append(j)
                    
                    if len(neighbors) < 3:  # Only if not too constrained
                        # Try increasing radius
                        new_r = min(safe_radius, r + 0.005)
                        if new_r > r:
                            # Check if this still works
                            valid = True
                            for j in range(len(current)):
                                if j != i:
                                    dist = np.sqrt((x - centers[j, 0])**2 + (y - centers[j, 1])**2)
                                    if dist < (new_r + radii[j]):
                                        valid = False
                                        break
                            if valid:
                                current[i] = [x, y, new_r]
        
        return current

    def optimize_with_local_refinement(circles, rect_width, rect_height):
        """Enhanced optimization with local refinement"""
        best_solution = circles.copy()
        best_score = np.sum(circles[:, 2])
        
        # Multi-stage refinement
        for stage in range(3):
            # Stage 1: Global relaxation
            if stage == 0:
                refined = smooth_and_relax(best_solution, rect_width, rect_height, 50)
            # Stage 2: Fine-grained local optimization  
            elif stage == 1:
                refined = smooth_and_relax(best_solution, rect_width, rect_height, 100)
            else:  # Stage 3: Intensive local search
                refined = smooth_and_relax(best_solution, rect_width, rect_height, 150)
            
            # Validate and score
            if validate_solution(refined, rect_width, rect_height):
                score = np.sum(refined[:, 2])
                if score > best_score:
                    best_score = score
                    best_solution = refined.copy()
        
        return best_solution

    # Generate initial configuration
    initial_circles = generate_initial_pattern()
    
    # Apply multi-stage optimization
    optimized_circles = optimize_with_local_refinement(initial_circles, rect_width, rect_height)
    
    # Final validation
    if not validate_solution(optimized_circles, rect_width, rect_height):
        # Fallback to simple hexagonal pattern if invalid
        circles = np.zeros((n, 3))
        rows = 5
        cols = 5
        spacing_x = rect_width / (cols + 1)
        spacing_y = rect_height / (rows + 1)
        max_radius = 0.08
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                circles[idx] = [x, y, max_radius]
                idx += 1
        optimized_circles = circles[:n].copy()
    
    # Final polishing
    polished = optimize_with_local_refinement(optimized_circles, rect_width, rect_height)
    
    return polished

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")