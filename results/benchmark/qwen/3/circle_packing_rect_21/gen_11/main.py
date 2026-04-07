# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
import time

def compute_fitness(circles, container_width=1.0, container_height=1.0):
    """Compute fitness as sum of radii with penalty for constraint violations."""
    # Extract radii
    radii = circles[:, 2]
    
    # Compute sum of radii
    total_radius = np.sum(radii)
    
    # Penalty for circles outside container
    penalty = 0
    
    # Check boundary constraints
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x - r < 0 or x + r > container_width or y - r < 0 or y + r > container_height:
            penalty += 1000  # Large penalty for boundary violations
    
    # Check overlap constraints
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if distance < (r1 + r2):
                penalty += 1000 * (r1 + r2 - distance)  # Penalty proportional to overlap
    
    return total_radius - penalty

def validate_circles(circles, container_width=1.0, container_height=1.0):
    """Validate that circles satisfy all constraints."""
    # Check boundary constraints
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x - r < 0 or x + r > container_width or y - r < 0 or y + r > container_height:
            return False
    
    # Check overlap constraints
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if distance < (r1 + r2):
                return False
    
    return True

def generate_initial_grid_placement(n_circles, container_width=1.0, container_height=1.0):
    """Generate initial configuration by placing circles on a grid."""
    # Create a grid of candidate positions
    grid_size = int(np.ceil(np.sqrt(n_circles)))
    spacing_x = container_width / (grid_size + 1)
    spacing_y = container_height / (grid_size + 1)
    
    circles = []
    placed_count = 0
    
    # Place circles on grid
    for i in range(grid_size):
        for j in range(grid_size):
            if placed_count >= n_circles:
                break
            x = (i + 1) * spacing_x
            y = (j + 1) * spacing_y
            
            # Calculate max possible radius at this position
            min_radius = min(x, container_width - x, y, container_height - y)
            
            # Add circle with small random radius (up to max possible)
            radius = min_radius * 0.4 * (0.5 + np.random.random() * 0.5)
            circles.append([x, y, radius])
            placed_count += 1
        
        if placed_count >= n_circles:
            break
    
    # Fill remaining slots with random placements
    while len(circles) < n_circles:
        x = np.random.random() * container_width
        y = np.random.random() * container_height
        min_radius = min(x, container_width - x, y, container_height - y)
        radius = min_radius * (0.1 + np.random.random() * 0.3)
        circles.append([x, y, radius])
        
    return np.array(circles)

def optimize_local(circles, container_width=1.0, container_height=1.0, iterations=50):
    """Perform local optimization on circle positions."""
    def objective(params):
        # Convert params back to circles array
        circles_copy = circles.copy()
        for i in range(len(circles)):
            circles_copy[i, 0] = params[3*i]   # x
            circles_copy[i, 1] = params[3*i+1] # y
            circles_copy[i, 2] = params[3*i+2] # r
        
        # Ensure radii are positive
        circles_copy[:, 2] = np.maximum(circles_copy[:, 2], 0.001)
        
        return -compute_fitness(circles_copy, container_width, container_height)
    
    # Flatten parameters
    initial_params = circles.flatten()
    bounds = []
    for i in range(len(circles)):
        bounds.append((0.001, container_width - 0.001))     # x bounds
        bounds.append((0.001, container_height - 0.001))   # y bounds
        bounds.append((0.001, min(container_width, container_height) / 2))  # r bounds
    
    # Use differential evolution for local optimization
    try:
        result = differential_evolution(objective, bounds, maxiter=iterations, popsize=15, seed=42)
        if result.success:
            optimized_circles = circles.copy()
            for i in range(len(circles)):
                optimized_circles[i, 0] = result.x[3*i]   # x
                optimized_circles[i, 1] = result.x[3*i+1] # y
                optimized_circles[i, 2] = result.x[3*i+2] # r
            return optimized_circles
    except:
        pass
    
    return circles

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set up container dimensions (perimeter = 4, so width + height = 2)
    container_width = 1.0
    container_height = 1.0
    
    best_circles = None
    best_fitness = -np.inf
    
    # Try multiple random initializations
    for attempt in range(10):
        # Generate initial grid placement
        circles = generate_initial_grid_placement(21, container_width, container_height)
        
        # Local optimization
        optimized_circles = optimize_local(circles, container_width, container_height, 100)
        
        # Check if valid solution
        if validate_circles(optimized_circles, container_width, container_height):
            fitness = compute_fitness(optimized_circles, container_width, container_height)
            if fitness > best_fitness:
                best_fitness = fitness
                best_circles = optimized_circles.copy()
    
    # If no valid solution found, use fallback
    if best_circles is None:
        # Fallback to simple uniform distribution
        circles = np.zeros((21, 3))
        for i in range(21):
            circles[i] = [
                0.1 + i * 0.04,  # x coordinate
                0.5,             # y coordinate  
                0.1              # radius
            ]
        return circles
    
    return best_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
