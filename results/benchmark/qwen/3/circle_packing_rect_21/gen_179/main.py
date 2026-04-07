# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')


def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    width, height = 1.2, 0.8

    # Initialize circles array
    n = 21
    circles = np.zeros((n, 3))

    # Phase 1: Voronoi-based initialization with boundary-aware seeding
    # Generate initial points using a combination of corner points, edge midpoints and Voronoi-based sampling
    corner_points = [
        [0.1, 0.1],           # Bottom-left
        [width-0.1, 0.1],     # Bottom-right
        [0.1, height-0.1],    # Top-left
        [width-0.1, height-0.1], # Top-right
    ]
    
    edge_midpoints = [
        [width/2, 0.1],       # Bottom-middle
        [width/2, height-0.1], # Top-middle
        [0.1, height/2],      # Left-middle
        [width-0.1, height/2], # Right-middle
    ]

    # Generate additional points using Voronoi sampling in the interior
    interior_points = generate_voronoi_seeds(width, height, n - 8, 1000)
    
    # Combine all initial points
    all_init_points = corner_points + edge_midpoints + interior_points
    
    # Initialize with small radii at these positions
    for i in range(n):
        circles[i] = [all_init_points[i][0], all_init_points[i][1], 0.01]

    # Phase 2: Multi-resolution optimization with adaptive constraints
    # Coarse optimization first
    for iteration in range(300):
        improved = False
        
        # Apply adaptive updates with decreasing step sizes
        step_factor = max(0.05, 1.0 - iteration/500.0)
        
        for i in range(n):
            max_radius = calculate_max_radius(circles, i, width, height, relax_factor=0.8)
            
            if max_radius > circles[i][2]:
                circles[i][2] = min(max_radius, circles[i][2] * (1.0 - 0.1 * step_factor))
                improved = True
                
        if not improved and iteration > 150:
            break

    # Phase 3: Medium resolution refinement 
    for iteration in range(300):
        improved = False
        
        # Apply adaptive updates with medium step sizes
        step_factor = max(0.05, 1.0 - iteration/400.0)
        
        for i in range(n):
            max_radius = calculate_max_radius(circles, i, width, height, relax_factor=1.0)
            
            if max_radius > circles[i][2]:
                circles[i][2] = min(max_radius, circles[i][2] * (1.0 - 0.05 * step_factor))
                improved = True
                
        if not improved and iteration > 150:
            break

    # Phase 4: Fine-grained local search with directional exploration
    for iteration in range(400):
        improved = False

        # Progressive refinement with diminishing step sizes
        step_size = max(0.005, 0.1 * (1.0 - iteration/800.0))

        # Prioritize circles in order of current radius for better convergence
        radii = circles[:, 2]
        sorted_indices = np.argsort(radii)[::-1]  # Sort by radius descending
        
        for i in sorted_indices:
            current_x, current_y, current_r = circles[i]
            
            # Enhanced directional search with adaptive step sizes
            best_pos = [current_x, current_y, current_r]
            best_radius = current_r
            
            # Search in concentric rings to balance exploration and exploitation
            search_distances = [0, step_size, step_size*2, step_size*3]
            
            for dist in search_distances:
                angles = np.linspace(0, 2*np.pi, 12, endpoint=False) 
                for angle in angles:
                    dx = dist * np.cos(angle)
                    dy = dist * np.sin(angle)
                    
                    new_x, new_y = current_x + dx, current_y + dy
                    
                    # Check bounds
                    if 0.001 <= new_x <= width - 0.001 and 0.001 <= new_y <= height - 0.001:
                        max_radius = calculate_max_radius_at_position(
                            circles, i, new_x, new_y, width, height, relax_factor=1.0
                        )
                        
                        if max_radius > best_radius:
                            best_radius = max_radius
                            best_pos = [new_x, new_y, max_radius]

            if best_pos[2] > circles[i][2]:
                circles[i] = best_pos
                improved = True
                
        if not improved and iteration > 200:
            break

    return circles


def generate_voronoi_seeds(width, height, n_points, sample_size):
    """Generate initial points using Voronoi-based sampling in interior region."""
    # Generate random points in the interior
    np.random.seed(42)  # For reproducibility
    points = []
    
    # Sample more points to ensure good distribution
    for _ in range(sample_size):
        x = np.random.uniform(0.01, width - 0.01)
        y = np.random.uniform(0.01, height - 0.01)
        points.append([x, y])
    
    points = np.array(points)
    
    # Create Voronoi diagram
    try:
        vor = Voronoi(points)
        
        # Get the centroids of the finite Voronoi cells
        centroids = []
        for region in vor.regions:
            if len(region) > 0 and -1 not in region:
                # Compute centroid of region
                vertices = vor.vertices[region]
                if len(vertices) > 0:
                    centroid_x = np.mean(vertices[:, 0])
                    centroid_y = np.mean(vertices[:, 1])
                    # Ensure the centroid is within bounds
                    if (0.01 <= centroid_x <= width - 0.01 and 
                        0.01 <= centroid_y <= height - 0.01):
                        centroids.append([centroid_x, centroid_y])
        
        # If we have enough centroids, return subset
        if len(centroids) >= n_points:
            return centroids[:n_points]
        else:
            # Fill with random points if insufficient
            additional_points = n_points - len(centroids)
            extra_points = []
            for _ in range(additional_points):
                x = np.random.uniform(0.01, width - 0.01)
                y = np.random.uniform(0.01, height - 0.01)
                extra_points.append([x, y])
            return centroids + extra_points
            
    except:
        # Fallback to uniform random sampling if Voronoi fails
        return [[np.random.uniform(0.01, width - 0.01),
                 np.random.uniform(0.01, height - 0.01)] for _ in range(n_points)]


def calculate_max_radius(circles, index, width, height, relax_factor=1.0):
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
            max_radius_for_this_circle = dist - cr * relax_factor

            if max_radius_for_this_circle < max_radius_overlap:
                max_radius_overlap = max_radius_for_this_circle

    max_radius = min(max_radius_bound, max_radius_overlap)
    return max(max_radius, 0.001)  # Ensure minimum radius


def calculate_max_radius_at_position(circles, index, x, y, width, height, relax_factor=1.0):
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
            max_radius_for_this_circle = dist - cr * relax_factor

            if max_radius_for_this_circle < max_radius_overlap:
                max_radius_overlap = max_radius_for_this_circle

    max_radius = min(max_radius_bound, max_radius_overlap)
    return max(max_radius, 0.001)  # Ensure minimum radius


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")