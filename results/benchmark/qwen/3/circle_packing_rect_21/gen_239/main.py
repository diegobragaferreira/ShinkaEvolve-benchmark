# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses Voronoi-based spatial partitioning for optimized circle placement.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    width, height = 1.2, 0.8

    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    def generate_voronoi_seed_points(n_points: int, width: float, height: float) -> np.ndarray:
        """Generate initial seed points using Voronoi-based approach"""
        # Start with strategic corner and edge points
        seed_points = [
            [0.1, 0.1],           # Bottom-left
            [width-0.1, 0.1],     # Bottom-right
            [0.1, height-0.1],    # Top-left
            [width-0.1, height-0.1], # Top-right
            [width/2, 0.1],       # Bottom-middle
            [width/2, height-0.1], # Top-middle
            [0.1, height/2],      # Left-middle
            [width-0.1, height/2], # Right-middle
        ]
        
        # Add random points in the interior
        while len(seed_points) < n_points:
            x = random.uniform(0.05, width - 0.05)
            y = random.uniform(0.05, height - 0.05)
            seed_points.append([x, y])
            
        return np.array(seed_points[:n_points])

    def compute_voronoi_cell_areas(points: np.ndarray, width: float, height: float) -> np.ndarray:
        """Compute areas of Voronoi cells for given points"""
        # Add boundary points to ensure finite cells
        boundary_points = [
            [-1, -1], [-1, height+1], [width+1, -1], [width+1, height+1],
            [width/2, -1], [width/2, height+1],
            [-1, height/2], [width+1, height/2]
        ]
        
        all_points = np.vstack([points, boundary_points])
        
        try:
            vor = Voronoi(all_points)
            areas = []
            
            # For each original point, compute the area of its Voronoi cell
            for i in range(len(points)):
                # Get vertices of the Voronoi cell for point i
                region_indices = vor.point_region[i]
                region = vor.regions[region_indices]
                
                # Skip infinite regions
                if -1 in region:
                    areas.append(0)
                    continue
                    
                # Extract vertices and compute polygon area
                vertices = np.array([vor.vertices[j] for j in region])
                # Filter vertices to be within bounds
                vertices = vertices[
                    (vertices[:, 0] >= 0) & (vertices[:, 0] <= width) &
                    (vertices[:, 1] >= 0) & (vertices[:, 1] <= height)
                ]
                
                if len(vertices) < 3:
                    areas.append(0)
                    continue
                    
                # Compute area using shoelace formula
                n = len(vertices)
                if n < 3:
                    areas.append(0)
                    continue
                    
                # Shoelace formula
                area = 0.5 * abs(sum(vertices[i][0] * vertices[(i+1)%n][1] - 
                                   vertices[(i+1)%n][0] * vertices[i][1] 
                                   for i in range(n)))
                areas.append(area)
            
            return np.array(areas)
        except:
            # Fallback for Voronoi computation issues
            return np.ones(len(points)) * (width * height / len(points))

    def initialize_from_voronoi(n_circles: int, width: float, height: float) -> np.ndarray:
        """Initialize circles based on Voronoi cell analysis"""
        # Generate initial points
        initial_points = generate_voronoi_seed_points(n_circles * 2, width, height)
        
        # Compute Voronoi cell areas
        cell_areas = compute_voronoi_cell_areas(initial_points, width, height)
        
        # Sort points by Voronoi cell area (largest first)
        sorted_indices = np.argsort(cell_areas)[::-1][:n_circles]
        
        # Create circles at selected points
        circles = np.zeros((n_circles, 3))
        
        # Initialize with larger radii at high-area Voronoi cells
        for i, idx in enumerate(sorted_indices):
            x, y = initial_points[idx]
            # Set initial radius based on Voronoi cell area (larger cells get larger initial radii)
            initial_radius = min(0.1, cell_areas[idx] * 0.05) if cell_areas[idx] > 0 else 0.02
            circles[i] = [x, y, max(0.005, min(0.1, initial_radius))]
            
        return circles

    def calculate_max_radius_fast(circles, index, width, height):
        """Fast calculation of maximum radius for circle at given index without overlapping others."""
        x, y, current_radius = circles[index]
        
        # Maximum radius based on container boundaries
        max_radius_bound = min(x, y, width - x, height - y)
        
        # Vectorized overlap checking for efficiency
        if len(circles) > 1:
            # Get other circles' positions and radii
            other_positions = circles[[i for i in range(len(circles)) if i != index], :2]
            other_radii = circles[[i for i in range(len(circles)) if i != index], 2]
            
            # Calculate distances to all other circles
            distances = np.sqrt(np.sum((other_positions - [x, y])**2, axis=1))
            
            # Maximum radius that avoids overlap with all other circles
            max_radius_overlap = np.min(distances - other_radii)
            
            max_radius = min(max_radius_bound, max_radius_overlap)
        else:
            max_radius = max_radius_bound
            
        return max(max_radius, 0.001)

    def calculate_max_radius_at_position_fast(circles, index, x, y, width, height):
        """Fast calculation of maximum radius for circle at given position without overlapping others."""
        # Maximum radius based on container boundaries
        max_radius_bound = min(x, y, width - x, height - y)
        
        # Vectorized overlap checking for efficiency
        if len(circles) > 1:
            # Get other circles' positions and radii
            other_positions = circles[[i for i in range(len(circles)) if i != index], :2]
            other_radii = circles[[i for i in range(len(circles)) if i != index], 2]
            
            # Calculate distances to all other circles
            distances = np.sqrt(np.sum((other_positions - [x, y])**2, axis=1))
            
            # Maximum radius that avoids overlap with all other circles
            max_radius_overlap = np.min(distances - other_radii)
            
            max_radius = min(max_radius_bound, max_radius_overlap)
        else:
            max_radius = max_radius_bound
            
        return max(max_radius, 0.001)

    def energy_based_optimization(circles: np.ndarray, width: float, height: float, 
                                 iterations: int = 1000) -> np.ndarray:
        """Optimize using an energy-based approach that considers attraction/repulsion forces"""
        current = circles.copy()
        
        for iter_num in range(iterations):
            # Adaptive learning rate
            lr = max(0.001, 0.05 * (1 - iter_num / iterations))
            
            # For each circle, compute forces and update positions
            for i in range(len(current)):
                x, y, r = current[i]
                
                # Compute attractive force towards center (encourage spreading)
                center_attraction = 0.001 * np.array([
                    (width/2 - x) * 0.1,
                    (height/2 - y) * 0.1
                ])
                
                # Compute repulsive forces from neighbors
                repulsion_force = np.array([0.0, 0.0])
                
                if len(current) > 1:
                    positions = current[:, :2]
                    radii = current[:, 2]
                    
                    # Calculate forces from all other circles
                    dx = positions[:, 0] - x
                    dy = positions[:, 1] - y
                    dists = np.sqrt(dx*dx + dy*dy)
                    
                    # Avoid division by zero
                    dists = np.maximum(dists, 1e-6)
                    
                    # Repulsion force (stronger when circles are close)
                    for j in range(len(current)):
                        if j != i:
                            # Force magnitude inversely proportional to distance squared
                            force_magnitude = 0.01 / (dists[j]**2 + 1e-6)
                            force_direction = np.array([dx[j], dy[j]]) / dists[j]
                            
                            # Scale by difference in radii (larger circles repel more)
                            radius_diff = radii[j] - r
                            force_scale = max(0.1, min(1.0, abs(radius_diff)))
                            
                            repulsion_force += force_magnitude * force_direction * force_scale
                
                # Compute boundary repulsion
                boundary_force = np.array([0.0, 0.0])
                boundary_repulsion_strength = 0.1
                
                if x < 0.05:
                    boundary_force[0] += boundary_repulsion_strength * (0.05 - x)
                elif x > width - 0.05:
                    boundary_force[0] += boundary_repulsion_strength * (width - 0.05 - x)
                    
                if y < 0.05:
                    boundary_force[1] += boundary_repulsion_strength * (0.05 - y)
                elif y > height - 0.05:
                    boundary_force[1] += boundary_repulsion_strength * (height - 0.05 - y)
                
                # Total force
                total_force = center_attraction + repulsion_force + boundary_force
                
                # Update position with learning rate
                new_x = x + lr * total_force[0]
                new_y = y + lr * total_force[1]
                
                # Clamp to boundaries
                new_x = np.clip(new_x, 0.05, width - 0.05)
                new_y = np.clip(new_y, 0.05, height - 0.05)
                
                # Update radius to maximize it
                max_radius = calculate_max_radius_at_position_fast(current, i, new_x, new_y, width, height)
                new_r = min(max_radius, max(0.001, r + lr * 0.1 * (max_radius - r)))
                
                current[i] = [new_x, new_y, new_r]
                
        return current

    def local_improvement_step(circles: np.ndarray, width: float, height: float) -> np.ndarray:
        """Refine configuration with local search improvements"""
        current = circles.copy()
        n = len(current)
        
        # Try to improve each circle individually
        for i in range(n):
            x, y, r = current[i]
            
            # Define search space
            search_steps = [0.01, 0.02, 0.03, 0.05, 0.08]
            best_x, best_y, best_r = x, y, r
            best_sum = np.sum(current[:, 2])
            
            # Grid search around current position
            for step in search_steps:
                # Search in a square pattern
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        new_x = x + dx
                        new_y = y + dy
                        
                        # Check bounds
                        if 0.05 <= new_x <= width - 0.05 and 0.05 <= new_y <= height - 0.05:
                            # Calculate max radius at new position
                            max_radius = calculate_max_radius_at_position_fast(current, i, new_x, new_y, width, height)
                            new_r = min(max_radius, max(0.001, r + random.uniform(-0.01, 0.01)))  # Slight random adjustment
                            
                            # Temporarily update
                            temp_current = current.copy()
                            temp_current[i] = [new_x, new_y, new_r]
                            
                            # Check if valid configuration
                            temp_sum = np.sum(temp_current[:, 2])
                            if temp_sum > best_sum:
                                best_x, best_y, best_r = new_x, new_y, new_r
                                best_sum = temp_sum
            
            # Apply best improvement
            current[i] = [best_x, best_y, best_r]
            
        return current

    # Main optimization workflow
    # Phase 1: Initialize using Voronoi-based approach
    circles = initialize_from_voronoi(21, width, height)
    
    # Phase 2: Energy-based optimization
    circles = energy_based_optimization(circles, width, height, 500)
    
    # Phase 3: Local improvement
    circles = local_improvement_step(circles, width, height)
    
    # Phase 4: Additional refining iterations
    for _ in range(100):
        circles = energy_based_optimization(circles, width, height, 100)
        circles = local_improvement_step(circles, width, height)
    
    # Final validation and cleanup
    for i in range(len(circles)):
        # Ensure minimum radius
        circles[i][2] = max(circles[i][2], 0.001)
        
        # Ensure circles stay within bounds
        circles[i][0] = np.clip(circles[i][0], 0.001, width - 0.001)
        circles[i][1] = np.clip(circles[i][1], 0.001, height - 0.001)

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")