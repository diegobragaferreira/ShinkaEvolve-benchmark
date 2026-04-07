# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
import random
import time
from typing import Tuple, List

# Set seeds for determinism
random.seed(42)
np.random.seed(42)

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_valid_solution(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> bool:
    """Check if all circles are within bounds and non-overlapping using efficient vectorized approach."""
    n = len(circles)
    
    # Check bounds - vectorized
    x_coords = circles[:, 0]
    y_coords = circles[:, 1]
    radii = circles[:, 2]
    
    if np.any(x_coords - radii < 0) or np.any(x_coords + radii > rect_width) or \
       np.any(y_coords - radii < 0) or np.any(y_coords + radii > rect_height):
        return False
    
    # Check overlaps - vectorized using broadcasting
    x_diff = x_coords[:, np.newaxis] - x_coords[np.newaxis, :]
    y_diff = y_coords[:, np.newaxis] - y_coords[np.newaxis, :]
    dists = np.sqrt(x_diff**2 + y_diff**2)
    sums = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Set diagonal to infinity to avoid self-comparison
    np.fill_diagonal(dists, np.inf)
    
    # Check if any distances are less than sum of radii
    if np.any(dists < sums):
        return False
    
    return True

def compute_voronoi_density(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """Compute Voronoi cell areas for each circle to estimate constraint density."""
    # Add boundary points for proper Voronoi calculation
    points = circles[:, :2].copy()
    
    # Add boundary points to make Voronoi more meaningful
    boundary_points = [
        [0, 0], [rect_width, 0], [0, rect_height], [rect_width, rect_height],
        [rect_width/2, 0], [rect_width/2, rect_height],
        [0, rect_height/2], [rect_width, rect_height/2]
    ]
    points = np.vstack([points, boundary_points])
    
    try:
        vor = Voronoi(points)
        
        # For each original point, compute Voronoi cell area
        areas = []
        for i in range(len(circles)):
            region_idx = np.where(vor.point_region == i)[0][0] if i in vor.point_region else -1
            
            if region_idx != -1 and region_idx < len(vor.regions):
                region = vor.regions[region_idx]
                if -1 not in region and len(region) >= 3:
                    # Compute area of polygon using shoelace formula
                    vertices = np.array([vor.vertices[i] for i in region])
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
        # Fallback to uniform distribution if Voronoi fails
        return np.ones(len(circles))

def compute_voronoi_metrics(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Voronoi cell areas and boundary distances for each circle."""
    # Add boundary points for proper Voronoi calculation
    points = circles[:, :2].copy()
    
    # Add boundary points to make Voronoi more meaningful
    boundary_points = [
        [0, 0], [rect_width, 0], [0, rect_height], [rect_width, rect_height],
        [rect_width/2, 0], [rect_width/2, rect_height],
        [0, rect_height/2], [rect_width, rect_height/2]
    ]
    points = np.vstack([points, boundary_points])
    
    try:
        vor = Voronoi(points)
        
        # For each original point, compute Voronoi cell area and boundary distance
        areas = []
        boundary_distances = []
        
        for i in range(len(circles)):
            region_idx = np.where(vor.point_region == i)[0][0] if i in vor.point_region else -1
            
            if region_idx != -1 and region_idx < len(vor.regions):
                region = vor.regions[region_idx]
                if -1 not in region and len(region) >= 3:
                    # Compute area of polygon using shoelace formula
                    vertices = np.array([vor.vertices[i] for i in region])
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
            
            # Compute minimum distance to boundary
            x, y = circles[i, 0], circles[i, 1]
            min_dist_to_boundary = min([
                x, rect_width - x, y, rect_height - y
            ])
            boundary_distances.append(min_dist_to_boundary)
                
        return np.array(areas), np.array(boundary_distances)
    except:
        # Fallback to uniform distribution if Voronoi fails
        return np.ones(len(circles)), np.ones(len(circles))

def evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate fitness as sum of radii."""
    return np.sum(circles[:, 2])

def generate_initial_pattern(n: int, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """Generate good initial pattern using a modified hexagonal approach."""
    circles = np.zeros((n, 3))
    
    # Use a more careful hexagonal approach
    sqrt_n = int(np.ceil(np.sqrt(n)))
    spacing_x = rect_width / (sqrt_n + 2)
    spacing_y = rect_height / (sqrt_n + 2)
    
    # Adjust spacing to ensure we have enough room
    min_radius = 0.02
    
    idx = 0
    for i in range(sqrt_n):
        for j in range(sqrt_n):
            if idx >= n:
                break
            x = (j + 1) * spacing_x
            y = (i + 1) * spacing_y
            # Slight offset for hexagonal pattern
            if i % 2 == 1:
                x += spacing_x / 2
            circles[idx] = [x, y, min_radius]
            idx += 1
        if idx >= n:
            break
    
    return circles

def adaptive_move_circle(circles: np.ndarray, idx: int, rect_width: float = 1.0, rect_height: float = 1.0,
                         density_scores: np.ndarray = None, voronoi_areas: np.ndarray = None,
                         boundary_distances: np.ndarray = None) -> np.ndarray:
    """Move a circle optimally based on Voronoi metrics."""
    new_circles = circles.copy()
    
    # Get current properties
    x, y, r = new_circles[idx]
    
    # Determine move strategy based on Voronoi information
    strategy_weight = 0.0
    
    # Prefer moving circles with larger Voronoi cells (more freedom)
    if voronoi_areas is not None and len(voronoi_areas) > idx:
        strategy_weight = voronoi_areas[idx]
    
    # Prefer moving circles near boundaries (they can grow more)
    boundary_weight = 0.0
    if boundary_distances is not None and len(boundary_distances) > idx:
        boundary_weight = boundary_distances[idx]
    
    # Base movement strategy
    base_delta = 0.02
    adaptive_delta = base_delta * (1.0 + strategy_weight * 0.5) * (1.0 + boundary_weight * 0.5)
    
    # Decide whether to move position or increase radius
    if np.random.random() < 0.3:  # 30% chance to adjust radius
        # Radius adjustment based on Voronoi quality
        max_radius_delta = adaptive_delta * 0.8
        delta_r = np.random.uniform(-max_radius_delta, max_radius_delta)
        new_r = r + delta_r
        
        # Ensure positive radius
        new_r = max(0.001, new_r)
        new_circles[idx, 2] = new_r
    else:
        # Position adjustment
        delta_x = np.random.uniform(-adaptive_delta, adaptive_delta)
        delta_y = np.random.uniform(-adaptive_delta, adaptive_delta)
        
        new_x = x + delta_x
        new_y = y + delta_y
        
        # Boundary checking with margin
        new_x = np.clip(new_x, r + 0.01, rect_width - r - 0.01)
        new_y = np.clip(new_y, r + 0.01, rect_height - r - 0.01)
        
        new_circles[idx, 0] = new_x
        new_circles[idx, 1] = new_y
    
    return new_circles

def adaptive_local_search(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0,
                          max_iter: int = 100) -> np.ndarray:
    """Perform adaptive local search using Voronoi-based strategy."""
    current_circles = circles.copy()
    best_circles = current_circles.copy()
    best_fitness = evaluate_fitness(current_circles)
    
    # Compute initial Voronoi metrics
    voronoi_areas, boundary_distances = compute_voronoi_metrics(current_circles, rect_width, rect_height)
    
    for iteration in range(max_iter):
        # Create a list of circle indices sorted by their Voronoi area (smaller first - more constrained)
        # This way, we prioritize fixing the most constrained circles first
        sort_indices = np.argsort(voronoi_areas)
        
        # Shuffle to add some randomness but maintain priority
        shuffled_indices = sort_indices.copy()
        np.random.shuffle(shuffled_indices)
        
        # Move multiple circles in each iteration for better exploration
        for i in shuffled_indices[:len(shuffled_indices)//2]:  # Move half the circles
            # Try to move this circle
            candidate_circles = adaptive_move_circle(
                current_circles, i, rect_width, rect_height, 
                voronoi_areas, voronoi_areas, boundary_distances
            )
            
            # Check if the move is valid
            if is_valid_solution(candidate_circles, rect_width, rect_height):
                current_circles = candidate_circles
                # Re-compute Voronoi metrics after each successful move
                voronoi_areas, boundary_distances = compute_voronoi_metrics(current_circles, rect_width, rect_height)
        
        # Update best solution
        current_fitness = evaluate_fitness(current_circles)
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_circles = current_circles.copy()
    
    return best_circles

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 21
    rect_width = 1.0  # Since perimeter = 4 and width + height = 2, we can set width = height = 1
    rect_height = 1.0
    
    # Generate initial solution
    initial_solution = generate_initial_pattern(n, rect_width, rect_height)
    
    # Perform adaptive local search
    final_solution = adaptive_local_search(initial_solution, rect_width, rect_height, max_iter=150)
    
    # Run a second pass for further improvement
    final_solution = adaptive_local_search(final_solution, rect_width, rect_height, max_iter=100)
    
    # Final validation and cleanup to ensure it's valid
    if not is_valid_solution(final_solution, rect_width, rect_height):
        # Fallback to original method if needed
        initial_solution = generate_initial_pattern(n, rect_width, rect_height)
        final_solution = adaptive_local_search(initial_solution, rect_width, rect_height, max_iter=200)
    
    # Ensure we have the best possible valid solution
    if not is_valid_solution(final_solution, rect_width, rect_height):
        final_solution = generate_initial_pattern(n, rect_width, rect_height)
    
    return final_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
