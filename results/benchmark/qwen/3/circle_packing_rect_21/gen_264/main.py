# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import time

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    width, height = 1.2, 0.8
    
    # Generate Voronoi-based initial configuration
    n = 21
    
    # Create a structured grid of points for Voronoi generation
    grid_size = 10
    x_coords = np.linspace(0.01, width - 0.01, grid_size)
    y_coords = np.linspace(0.01, height - 0.01, grid_size)
    
    # Generate regular grid points
    grid_points = []
    for x in x_coords:
        for y in y_coords:
            grid_points.append([x, y])
    
    # Add boundary corner and edge points for better coverage
    boundary_points = [
        [0.01, 0.01], [width-0.01, 0.01], [0.01, height-0.01], [width-0.01, height-0.01],
        [width/2, 0.01], [width/2, height-0.01], [0.01, height/2], [width-0.01, height/2]
    ]
    
    all_points = grid_points + boundary_points
    
    # Use Voronoi to analyze spatial distribution
    vor = Voronoi(all_points)
    
    # Select Voronoi cell centroids that fall within bounds as initial centers
    centroids = []
    for i, point in enumerate(vor.points):
        # Skip points that are near boundaries or already selected
        if 0.01 <= point[0] <= width - 0.01 and 0.01 <= point[1] <= height - 0.01:
            # Find corresponding Voronoi cell centroid
            region = vor.regions[vor.point_region[i]]
            if -1 not in region and len(region) > 0:
                try:
                    # Get vertices of Voronoi region
                    vertices = [vor.vertices[j] for j in region]
                    if len(vertices) > 0:
                        # Compute centroid
                        centroid = np.mean(vertices, axis=0)
                        if 0.01 <= centroid[0] <= width - 0.01 and 0.01 <= centroid[1] <= height - 0.01:
                            centroids.append([centroid[0], centroid[1]])
                except:
                    pass
    
    # If not enough centroids, fill with k-means
    if len(centroids) < n:
        from scipy.cluster.vq import kmeans2
        dense_grid_size = 50
        grid_x = np.linspace(0.01, width - 0.01, dense_grid_size)
        grid_y = np.linspace(0.01, height - 0.01, dense_grid_size)
        grid_points = np.array([[x, y] for x in grid_x for y in grid_y])
        centroids_kmeans, _ = kmeans2(grid_points, n - len(centroids), minit='points')
        centroids.extend(centroids_kmeans.tolist())
    
    # Ensure we have exactly n points
    if len(centroids) > n:
        centroids = centroids[:n]
    elif len(centroids) < n:
        # Fill with random points
        for i in range(n - len(centroids)):
            centroids.append([np.random.uniform(0.01, width - 0.01), 
                              np.random.uniform(0.01, height - 0.01)])
    
    # Initialize circles with these centroids
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i] = [centroids[i][0], centroids[i][1], 0.01]
    
    # Phase 1: Evolutionary enhancement with Voronoi-guided population diversity
    best_solution = circles.copy()
    best_sum = np.sum(circles[:, 2])
    
    # Create diverse initial solutions using Voronoi-inspired sampling
    for trial in range(8):
        np.random.seed(trial * 42)
        
        # Start with Voronoi-based configuration
        trial_circles = circles.copy()
        
        # Slightly perturb positions based on Voronoi insight
        for i in range(n):
            # Small random perturbation centered on Voronoi cell
            perturbation_factor = 0.1
            trial_circles[i][0] += np.random.uniform(-perturbation_factor, perturbation_factor) * width
            trial_circles[i][1] += np.random.uniform(-perturbation_factor, perturbation_factor) * height
            
            # Clip to bounds
            trial_circles[i][0] = np.clip(trial_circles[i][0], 0.01, width - 0.01)
            trial_circles[i][1] = np.clip(trial_circles[i][1], 0.01, height - 0.01)
            
            # Reset radius
            trial_circles[i][2] = 0.01

        # Maximize radii for this configuration
        for _ in range(150):
            improved = False
            for j in range(n):
                max_radius = calculate_max_radius(trial_circles, j, width, height)
                if max_radius > trial_circles[j][2]:
                    trial_circles[j][2] = max_radius
                    improved = True
            if not improved:
                break

        # Check if this trial is better
        trial_sum = np.sum(trial_circles[:, 2])
        if trial_sum > best_sum:
            best_sum = trial_sum
            best_solution = trial_circles.copy()
    
    circles = best_solution

    # Phase 2: Multi-scale aggressive optimization guided by Voronoi structure
    max_iterations = 1000
    last_improvement_iter = 0
    improvement_count = 0
    prev_sum_radii = 0

    for iteration in range(max_iterations):
        improved = False
        
        # Adaptive parameters with Voronoi-inspired progression
        if iteration < 250:
            step_size = 0.2
            radius_update_factor = 0.9
        elif iteration < 600:
            step_size = 0.1
            radius_update_factor = 0.7
        else:
            step_size = 0.05
            radius_update_factor = 0.5

        # Try to increase each circle's radius considering Voronoi neighbors
        for i in range(n):
            # Find maximum possible radius for circle i
            max_radius = calculate_max_radius(circles, i, width, height)
            
            if max_radius > circles[i][2]:
                circles[i][2] = min(max_radius, circles[i][2] * radius_update_factor)
                improved = True

        # Track improvement for early stopping
        current_sum_radii = np.sum(circles[:, 2])
        if current_sum_radii > prev_sum_radii:
            improvement_count += 1
            last_improvement_iter = iteration
            prev_sum_radii = current_sum_radii
        else:
            improvement_count = 0

        # Early termination conditions - Voronoi-informed adaptive stopping
        if improvement_count > 40 or (iteration - last_improvement_iter > 400 and current_sum_radii < prev_sum_radii * 1.002):
            break

    # Phase 3: Advanced local search with Voronoi-driven neighborhood exploration
    local_search_iterations = 500
    
    for refinement_iteration in range(local_search_iterations):
        # Progressive step size reduction with Voronoi-inspired patterns
        if refinement_iteration < 150:
            step_size = 0.15
        elif refinement_iteration < 350:
            step_size = 0.05
        else:
            step_size = 0.02

        # Try moving each circle in a more Voronoi-aware pattern
        for i in range(n):
            current_x, current_y, current_r = circles[i]

            # Use Voronoi-inspired adaptive grid pattern
            if refinement_iteration < 100:
                # Coarse grid for early exploration
                search_grid = [-step_size*2, -step_size, 0, step_size, step_size*2]
            elif refinement_iteration < 250:
                # Medium grid for middle refinement
                search_grid = [-step_size, -step_size/2, 0, step_size/2, step_size]
            else:
                # Fine grid for final refinement
                search_grid = [-step_size/2, 0, step_size/2]

            # Add diagonal moves for better Voronoi coverage
            if refinement_iteration > 200:
                search_grid.extend([-step_size*1.5, step_size*1.5])

            # Try all combinations in search grid
            best_pos = [current_x, current_y, current_r]
            best_radius = current_r
            
            for dx in search_grid:
                for dy in search_grid:
                    new_x, new_y = current_x + dx, current_y + dy

                    # Check if new position is within bounds
                    if 0 <= new_x <= width and 0 <= new_y <= height:
                        # Calculate max radius at new position
                        max_radius = calculate_max_radius_at_position(
                            circles, i, new_x, new_y, width, height
                        )

                        if max_radius > best_radius:
                            best_radius = max_radius
                            best_pos = [new_x, new_y, max_radius]

            # Move circle to better position if found
            if best_pos[2] > circles[i][2]:
                circles[i] = best_pos

    # Final validation
    for i in range(n):
        circles[i][2] = max(circles[i][2], 0.001)
        circles[i][0] = np.clip(circles[i][0], 0.001, width - 0.001)
        circles[i][1] = np.clip(circles[i][1], 0.001, height - 0.001)
        
    return circles


def calculate_max_radius(circles, index, width, height):
    """Calculate maximum radius for circle at given index without overlapping others."""
    x, y, current_radius = circles[index]

    # Maximum radius based on container boundaries
    max_radius_bound = min(x, y, width - x, height - y)

    # Maximum radius based on other circles
    max_radius_overlap = float('inf')

    for i, (cx, cy, cr) in enumerate(circles):
        if i != index:
            # Distance to other circle center
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            # Max radius that avoids overlap
            max_radius_for_this_circle = dist - cr

            if max_radius_for_this_circle < max_radius_overlap:
                max_radius_overlap = max_radius_for_this_circle

    max_radius = min(max_radius_bound, max_radius_overlap)
    return max(max_radius, 0.001)  # Ensure minimum radius


def calculate_max_radius_at_position(circles, index, x, y, width, height):
    """Calculate maximum radius for circle at given position without overlapping others."""
    # Maximum radius based on container boundaries
    max_radius_bound = min(x, y, width - x, height - y)

    # Maximum radius based on other circles
    max_radius_overlap = float('inf')

    for i, (cx, cy, cr) in enumerate(circles):
        if i != index:
            # Distance to other circle center
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            # Max radius that avoids overlap
            max_radius_for_this_circle = dist - cr

            if max_radius_for_this_circle < max_radius_overlap:
                max_radius_overlap = max_radius_for_this_circle

    max_radius = min(max_radius_bound, max_radius_overlap)
    return max(max_radius, 0.001)  # Ensure minimum radius


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")