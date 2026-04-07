# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance_matrix
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time

# Global constants
RECT_PERIMETER = 4.0
RECT_WIDTH_HEIGHT_RATIO = 3.0  # Better aspect ratio for 21 circles
RECT_WIDTH = RECT_PERIMETER / (2 * (1 + RECT_WIDTH_HEIGHT_RATIO))
RECT_HEIGHT = RECT_PERIMETER / (2 * (1 + RECT_WIDTH_HEIGHT_RATIO))
NUM_CIRCLES = 21

def compute_voronoi_force_field(points: np.ndarray, width: float, height: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Voronoi-based force field for circle centers to guide optimization.
    Returns forces (dx, dy) for each point.
    """
    # Add boundary points to ensure proper Voronoi cells near edges
    boundary_points = [
        [-0.1, -0.1], [-0.1, height + 0.1], [width + 0.1, -0.1], [width + 0.1, height + 0.1],
        [width/2, -0.1], [width/2, height + 0.1], [-0.1, height/2], [width + 0.1, height/2]
    ]
    
    all_points = np.vstack([points, boundary_points])
    
    try:
        vor = Voronoi(all_points)
        
        # Calculate forces for each point (excluding boundary points)
        forces = np.zeros((len(points), 2))
        cell_areas = np.zeros(len(points))
        
        for i in range(len(points)):
            # Find Voronoi cell for this point
            # This is simplified - we'll use proximity to nearby points instead
            # For the purpose of this algorithm, we compute a simpler force field
            
            # Compute forces from nearby points (excluding itself)
            current_point = points[i]
            distances = np.sqrt(np.sum((points - current_point)**2, axis=1))
            
            # Exclude self (distance = 0)
            distances[i] = np.inf
            
            # Compute repulsive forces from nearby points
            nearby_indices = np.argsort(distances)[:min(10, len(points)-1)]
            force_x, force_y = 0.0, 0.0
            
            for j in nearby_indices:
                if i != j:
                    dx = points[i, 0] - points[j, 0]
                    dy = points[i, 1] - points[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist > 0.001:  # Avoid division by zero
                        # Repulsive force inversely proportional to distance squared
                        force_magnitude = 0.001 / (dist * dist + 0.001)
                        force_x += force_magnitude * dx / dist
                        force_y += force_magnitude * dy / dist
            
            # Add boundary forces (push inward if near edges)
            boundary_force_x, boundary_force_y = 0.0, 0.0
            boundary_margin = 0.05
            
            if current_point[0] < boundary_margin:
                boundary_force_x += 0.1 * (boundary_margin - current_point[0])
            elif current_point[0] > width - boundary_margin:
                boundary_force_x -= 0.1 * (current_point[0] - (width - boundary_margin))
                
            if current_point[1] < boundary_margin:
                boundary_force_y += 0.1 * (boundary_margin - current_point[1])
            elif current_point[1] > height - boundary_margin:
                boundary_force_y -= 0.1 * (current_point[1] - (height - boundary_margin))
            
            forces[i] = [force_x + boundary_force_x, force_y + boundary_force_y]
            # Compute effective area (simplified - just inverse of distance to nearest neighbor)
            if len(distances) > 1:
                cell_areas[i] = 1.0 / (np.min(distances) + 0.001)
        
        return forces, cell_areas
        
    except Exception:
        # Fallback to simple repulsion forces
        forces = np.zeros((len(points), 2))
        for i in range(len(points)):
            current_point = points[i]
            force_x, force_y = 0.0, 0.0
            for j in range(len(points)):
                if i != j:
                    dx = points[i, 0] - points[j, 0]
                    dy = points[i, 1] - points[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist > 0.001:
                        force_magnitude = 0.001 / (dist * dist + 0.001)
                        force_x += force_magnitude * dx / dist
                        force_y += force_magnitude * dy / dist
            forces[i] = [force_x, force_y]
        return forces, None

def compute_max_radius_at_position(x: float, y: float, circles: np.ndarray, 
                                  boundary_padding: float = 0.01) -> float:
    """Compute maximum radius for a circle at position (x,y) without overlaps."""
    # Boundary constraints
    max_radius = min(x - boundary_padding, 
                     RECT_WIDTH - x - boundary_padding,
                     y - boundary_padding, 
                     RECT_HEIGHT - y - boundary_padding)
    
    # Overlap constraints with existing circles
    for cx, cy, r in circles:
        if cx != x or cy != y:  # Skip self-comparison
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            max_radius = min(max_radius, dist - r)
    
    return max(max_radius, 0.001)  # Ensure positive radius

def validate_circle_config(circles: np.ndarray) -> bool:
    """Validate that all circles are within bounds and non-overlapping."""
    for i, (x, y, r) in enumerate(circles):
        # Check boundary constraints
        if (x - r < 0 or x + r > RECT_WIDTH or 
            y - r < 0 or y + r > RECT_HEIGHT):
            return False
        
        # Check overlap with other circles
        for j, (cx, cy, cr) in enumerate(circles):
            if i != j:
                dx = x - cx
                dy = y - cy
                distance = np.sqrt(dx*dx + dy*dy)
                if distance < (r + cr):
                    return False
    
    return True

def project_to_bounds(point: np.ndarray) -> np.ndarray:
    """Project point to be within rectangle bounds."""
    x, y = point
    x = np.clip(x, 0.01, RECT_WIDTH - 0.01)
    y = np.clip(y, 0.01, RECT_HEIGHT - 0.01)
    return np.array([x, y])

def generate_voronoi_initialization(n: int) -> np.ndarray:
    """
    Generate initial configuration using a combination of Voronoi sampling and grid-based initialization.
    """
    # Start with structured grid points
    grid_size = max(3, int(np.ceil(np.sqrt(n))))
    x_coords = np.linspace(0.1, RECT_WIDTH - 0.1, grid_size)
    y_coords = np.linspace(0.1, RECT_HEIGHT - 0.1, grid_size)
    
    initial_points = []
    for x in x_coords:
        for y in y_coords:
            initial_points.append([x, y])
    
    # If we have more points than needed, select using maximin property
    if len(initial_points) > n:
        # Use maximin selection to get well spread points
        selected = []
        remaining = list(range(len(initial_points)))
        
        if remaining:
            # Start with a random point
            start_idx = random.randint(0, len(remaining) - 1)
            selected.append(remaining.pop(start_idx))
            
            # Iteratively add points that maximize minimum distance to existing selected points
            while len(selected) < n and remaining:
                # Compute distances from remaining points to selected points
                selected_points = np.array([initial_points[i] for i in selected])
                remaining_points = np.array([initial_points[i] for i in remaining])
                
                if len(selected_points) > 0:
                    dist_matrix = cdist(remaining_points, selected_points)
                    min_distances = np.min(dist_matrix, axis=1)
                    
                    # Select point with maximum minimum distance
                    max_idx = np.argmax(min_distances)
                    selected.append(remaining.pop(max_idx))
                else:
                    break
            
            initial_points = [initial_points[i] for i in selected]
        else:
            initial_points = initial_points[:n]
    else:
        initial_points = initial_points[:n]
    
    # Convert to numpy array
    points_array = np.array(initial_points)
    
    # Ensure exact count
    if len(points_array) < n:
        # Fill missing points randomly
        for _ in range(n - len(points_array)):
            x = random.uniform(0.1, RECT_WIDTH - 0.1)
            y = random.uniform(0.1, RECT_HEIGHT - 0.1)
            points_array = np.vstack([points_array, [x, y]])
    
    # Initialize with small radii
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i] = [points_array[i][0], points_array[i][1], 0.02]
    
    return circles

def voronoi_gradient_optimization() -> np.ndarray:
    """
    Optimize circle packing using Voronoi-guided gradient descent.
    """
    # Initialize
    circles = generate_voronoi_initialization(NUM_CIRCLES)
    
    # Set up optimization parameters
    max_iterations = 1000
    learning_rate = 0.1
    momentum = 0.8
    velocity = np.zeros_like(circles[:, :2])
    
    # Store best solution
    best_circles = circles.copy()
    best_sum = np.sum(circles[:, 2])
    
    # Optimization loop
    for iteration in range(max_iterations):
        # Compute Voronoi-guided forces
        positions = circles[:, :2]
        
        # Compute forces
        forces, _ = compute_voronoi_force_field(positions, RECT_WIDTH, RECT_HEIGHT)
        
        # Apply forces with learning rate
        for i in range(len(circles)):
            # Update velocity with momentum
            velocity[i] = momentum * velocity[i] + learning_rate * forces[i]
            
            # Update position
            circles[i, 0] += velocity[i][0]
            circles[i, 1] += velocity[i][1]
            
            # Project to bounds
            circles[i, 0] = np.clip(circles[i, 0], 0.01, RECT_WIDTH - 0.01)
            circles[i, 1] = np.clip(circles[i, 1], 0.01, RECT_HEIGHT - 0.01)
        
        # Recompute maximum possible radii for each circle 
        # (this step ensures we're truly optimizing the sum of radii)
        for i in range(len(circles)):
            x, y, r = circles[i]
            max_r = compute_max_radius_at_position(x, y, circles)
            circles[i, 2] = max_r
        
        # Check if this is better
        current_sum = np.sum(circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = circles.copy()
        
        # Adaptive learning rate decrease
        if iteration > 0 and iteration % 100 == 0:
            learning_rate *= 0.95
        
        # Early stopping based on improvement rate
        if iteration > 100 and abs(current_sum - best_sum) < 0.0001:
            break
    
    # Final adjustment with local refinement
    refined_circles = best_circles.copy()
    for _ in range(500):
        # Pick random circle
        idx = random.randint(0, NUM_CIRCLES - 1)
        
        # Try small moves in various directions
        best_x, best_y, best_r = refined_circles[idx]
        best_sum = np.sum(refined_circles[:, 2])
        
        # Try different small displacements
        step_size = 0.02
        directions = [(0, 0), (step_size, 0), (-step_size, 0), (0, step_size), (0, -step_size),
                      (step_size, step_size), (-step_size, -step_size), (step_size, -step_size), (-step_size, step_size)]
        
        for dx, dy in directions:
            test_x = np.clip(refined_circles[idx, 0] + dx, 0.01, RECT_WIDTH - 0.01)
            test_y = np.clip(refined_circles[idx, 1] + dy, 0.01, RECT_HEIGHT - 0.01)
            
            # Compute max radius at new position
            temp_circles = refined_circles.copy()
            temp_circles[idx] = [test_x, test_y, 0.01]
            max_r = compute_max_radius_at_position(test_x, test_y, temp_circles)
            test_r = max_r
            
            # Update if better
            temp_circles[idx] = [test_x, test_y, test_r]
            new_sum = np.sum(temp_circles[:, 2])
            if new_sum > best_sum:
                best_sum = new_sum
                best_x, best_y, best_r = test_x, test_y, test_r
        
        # Apply best move
        refined_circles[idx] = [best_x, best_y, best_r]
    
    return refined_circles

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run Voronoi-guided gradient optimization
    circles = voronoi_gradient_optimization()
    
    # Final validation
    if not validate_circle_config(circles):
        # If invalid, reinitialize with better approach
        circles = generate_voronoi_initialization(NUM_CIRCLES)
        
        # Add small random variation to avoid plateaus
        for i in range(len(circles)):
            circles[i, 0] += random.uniform(-0.01, 0.01)
            circles[i, 1] += random.uniform(-0.01, 0.01)
            circles[i, 0] = np.clip(circles[i, 0], 0.01, RECT_WIDTH - 0.01)
            circles[i, 1] = np.clip(circles[i, 1], 0.01, RECT_HEIGHT - 0.01)
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")