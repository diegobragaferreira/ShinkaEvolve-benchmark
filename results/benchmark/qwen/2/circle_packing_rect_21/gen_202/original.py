# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
import random
import time

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses Voronoi diagram based approach for superior spatial distribution.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Rectangle dimensions: width + height = 2, optimized ratio
    rect_width = 1.2
    rect_height = 0.8

    n = 21

    def generate_voronoi_initial_placement(num_circles, width, height):
        """
        Generate initial circle placement using Voronoi diagram approach.
        """
        # Generate random points for Voronoi diagram
        # Add some padding to avoid edge issues
        padding = 0.1
        points = []
        
        # Create a structured initial set of points that will form good Voronoi cells
        grid_rows = int(np.sqrt(num_circles)) + 1
        grid_cols = int(np.ceil(num_circles / grid_rows))
        
        # Create a roughly uniform grid of points
        for i in range(grid_rows):
            for j in range(grid_cols):
                if len(points) >= num_circles:
                    break
                x = padding + (j + 0.5) * (width - 2*padding) / grid_cols
                y = padding + (i + 0.5) * (height - 2*padding) / grid_rows
                # Add slight randomness to avoid perfect grid patterns
                x += random.uniform(-0.02, 0.02)
                y += random.uniform(-0.02, 0.02)
                if 0 <= x <= width and 0 <= y <= height:
                    points.append([x, y])
        
        # If we didn't get enough points, add random ones
        while len(points) < num_circles:
            x = random.uniform(padding, width - padding)
            y = random.uniform(padding, height - padding)
            points.append([x, y])
        
        points = np.array(points[:num_circles])
        
        # Compute Voronoi diagram
        try:
            vor = Voronoi(points)
        except:
            # Fallback to random placement if Voronoi fails
            points = np.random.rand(num_circles, 2) * [width - 2*padding, height - 2*padding] + [padding, padding]
            vor = Voronoi(points)
        
        circles = []
        for i, (x, y) in enumerate(points):
            # Compute Voronoi cell area and determine radius
            # Find vertices of this cell
            if i < len(vor.point_region):
                region = vor.point_region[i]
                if region >= 0:
                    # Get vertices of this Voronoi cell
                    vertices = vor.vertices[vor.regions[region]]
                    if len(vertices) > 0:
                        # Find the minimum distance from centroid to cell boundary
                        min_dist = min(
                            distance.euclidean([x, y], vertex) for vertex in vertices
                        )
                        # Estimate radius based on cell size and proximity to boundaries
                        r = min(min_dist * 0.4, 0.15)
                    else:
                        r = 0.05
                else:
                    # Default case
                    r = 0.05
            else:
                r = 0.05
            
            # Ensure radius is within bounds
            r = max(0.005, min(r, width/4, height/4))
            
            # Ensure circle fits within rectangle
            r = min(r, x - 0.01, width - x - 0.01, y - 0.01, height - y - 0.01)
            
            circles.append([x, y, r])
        
        return np.array(circles)

    def calculate_fitness_voronoi_based(circles_array):
        """Fitness calculation with Voronoi-based penalties"""
        total_radius = np.sum(circles_array[:, 2])
        
        penalty = 0
        
        # Boundary penalties
        for i in range(len(circles_array)):
            x, y, r = circles_array[i]
            # More severe penalties near edges
            if x - r < 0.01:
                penalty += 100000 * (r - x)**2
            if x + r > rect_width - 0.01:
                penalty += 100000 * (x + r - rect_width)**2
            if y - r < 0.01:
                penalty += 100000 * (r - y)**2
            if y + r > rect_height - 0.01:
                penalty += 100000 * (y + r - rect_height)**2
        
        # Overlap penalties using direct distance computation
        for i in range(len(circles_array)):
            for j in range(i+1, len(circles_array)):
                x1, y1, r1 = circles_array[i]
                x2, y2, r2 = circles_array[j]
                
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                overlap = (r1 + r2) - dist
                
                if overlap > 0:
                    penalty += 200000 * overlap**2
        
        return total_radius - penalty

    def voronoi_force_based_optimization(circles_array, max_iter=500):
        """Optimize using force-based approach derived from Voronoi geometry"""
        best_circles = circles_array.copy()
        best_fitness = calculate_fitness_voronoi_based(best_circles)
        
        # Parameters for force-based optimization
        dt = 0.01
        repulsion_strength = 1000.0
        boundary_strength = 500.0
        attraction_strength = 10.0
        
        for iteration in range(max_iter):
            # Calculate forces for each circle
            forces = np.zeros_like(best_circles[:, :2])
            
            # Repulsion forces from overlaps
            positions = best_circles[:, :2]
            radii = best_circles[:, 2]
            
            for i in range(len(best_circles)):
                x1, y1 = positions[i]
                r1 = radii[i]
                
                # Check nearby circles for repulsion
                for j in range(len(best_circles)):
                    if i != j:
                        x2, y2 = positions[j]
                        r2 = radii[j]
                        dx = x2 - x1
                        dy = y2 - y1
                        dist = np.sqrt(dx*dx + dy*dy)
                        
                        if dist > 0 and dist < (r1 + r2):
                            # Repulsion force
                            force_mag = repulsion_strength * (1.0 - dist/(r1 + r2)) / (dist + 1e-8)
                            forces[i, 0] += force_mag * dx / dist
                            forces[i, 1] += force_mag * dy / dist
            
            # Boundary forces
            for i in range(len(best_circles)):
                x, y, r = best_circles[i]
                fx, fy = 0, 0
                
                # Left boundary
                if x - r < 0.01:
                    fx += boundary_strength * (0.01 - (x - r))
                # Right boundary
                if x + r > rect_width - 0.01:
                    fx -= boundary_strength * ((x + r) - (rect_width - 0.01))
                # Bottom boundary
                if y - r < 0.01:
                    fy += boundary_strength * (0.01 - (y - r))
                # Top boundary
                if y + r > rect_height - 0.01:
                    fy -= boundary_strength * ((y + r) - (rect_height - 0.01))
                
                forces[i, 0] += fx
                forces[i, 1] += fy
            
            # Move circles
            for i in range(len(best_circles)):
                # Apply forces
                best_circles[i, 0] += forces[i, 0] * dt
                best_circles[i, 1] += forces[i, 1] * dt
                
                # Keep within bounds
                x, y, r = best_circles[i]
                best_circles[i, 0] = np.clip(x, r + 0.01, rect_width - r - 0.01)
                best_circles[i, 1] = np.clip(y, r + 0.01, rect_height - r - 0.01)
            
            # Periodic fitness check for early stopping
            if iteration % 20 == 0:
                current_fitness = calculate_fitness_voronoi_based(best_circles)
                if current_fitness > best_fitness:
                    best_fitness = current_fitness
                else:
                    # If no progress for several iterations, reduce learning rate
                    dt *= 0.99
        
        return best_circles

    def adaptive_radius_enhancement(circles_array):
        """Enhance radii by analyzing Voronoi cells and expanding where possible"""
        best_circles = circles_array.copy()
        
        # For each circle, try to increase radius if safe
        for i in range(len(best_circles)):
            x, y, r = best_circles[i]
            
            # Compute maximum allowable radius
            max_radius = float('inf')
            
            # Boundary constraints
            max_radius = min(max_radius, x - 0.01)
            max_radius = min(max_radius, rect_width - x - 0.01)
            max_radius = min(max_radius, y - 0.01)
            max_radius = min(max_radius, rect_height - y - 0.01)
            
            # Overlap constraints
            for j in range(len(best_circles)):
                if i != j:
                    x2, y2, r2 = best_circles[j]
                    dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                    max_radius = min(max_radius, dist - r2 - 0.001)
            
            # Try increasing radius if beneficial
            if max_radius > r and max_radius > 0.001:
                new_r = min(r + 0.01, max_radius)
                if new_r > r:
                    # Test validity of new configuration
                    temp_circles = best_circles.copy()
                    temp_circles[i, 2] = new_r
                    
                    # Check overlaps
                    valid = True
                    for k in range(len(temp_circles)):
                        if k != i:
                            xk, yk, rk = temp_circles[k]
                            dist = np.sqrt((x - xk)**2 + (y - yk)**2)
                            if dist < new_r + rk:
                                valid = False
                                break
                    
                    if valid:
                        best_circles[i, 2] = new_r
        
        return best_circles

    def multi_stage_voronoi_optimization(initial_circles):
        """Full optimization pipeline using Voronoi approach"""
        
        # Stage 1: Voronoi-based initial placement
        stage1_circles = generate_voronoi_initial_placement(n, rect_width, rect_height)
        
        # Stage 2: Force-based optimization
        stage2_circles = voronoi_force_based_optimization(stage1_circles, max_iter=300)
        
        # Stage 3: Radius enhancement
        stage3_circles = adaptive_radius_enhancement(stage2_circles)
        
        # Stage 4: Final force-based optimization
        stage4_circles = voronoi_force_based_optimization(stage3_circles, max_iter=200)
        
        # Stage 5: Final radius enhancement
        stage5_circles = adaptive_radius_enhancement(stage4_circles)
        
        return stage5_circles

    # Execute the Voronoi-based optimization
    final_circles = multi_stage_voronoi_optimization(None)
    
    # Final validation
    total_radius = np.sum(final_circles[:, 2])
    fitness = calculate_fitness_voronoi_based(final_circles)
    
    # Ensure final constraints are met
    for i in range(n):
        x, y, r = final_circles[i]
        # Keep within bounds
        final_circles[i, 0] = np.clip(x, r + 0.01, rect_width - r - 0.01)
        final_circles[i, 1] = np.clip(y, r + 0.01, rect_height - r - 0.01)
        # Ensure positive radius
        final_circles[i, 2] = max(0.001, r)
    
    return final_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")