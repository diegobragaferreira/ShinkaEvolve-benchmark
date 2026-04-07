# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
import random
from typing import Tuple, List
import time

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def check_constraints(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> bool:
    """Efficiently check if all circles satisfy the constraints with early termination."""
    n = len(circles)
    
    # Check boundary constraints first
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
            return False
    
    # Use spatial grid for efficient overlap checking
    if n > 1:
        # Create spatial grid
        grid_size = 0.15  # Size of grid cells - adjust based on typical circle size
        grid_width = int(np.ceil(rect_width / grid_size))
        grid_height = int(np.ceil(rect_height / grid_size))

        # Initialize grid
        grid = {}

        # Place circles in grid cells
        for i in range(n):
            x, y, r = circles[i]
            # Get grid coordinates for this circle
            grid_x = int(x / grid_size)
            grid_y = int(y / grid_size)

            if (grid_x, grid_y) not in grid:
                grid[(grid_x, grid_y)] = []
            grid[(grid_x, grid_y)].append(i)

        # Check for overlaps using grid-based approach
        for i in range(n):
            x1, y1, r1 = circles[i]
            # Get grid coordinates for this circle
            grid_x = int(x1 / grid_size)
            grid_y = int(y1 / grid_size)

            # Check this cell and adjacent cells
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor_cell = (grid_x + dx, grid_y + dy)
                    if neighbor_cell in grid:
                        for j in grid[neighbor_cell]:
                            if i != j:  # Don't compare with self
                                x2, y2, r2 = circles[j]
                                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                                if distance < (r1 + r2):
                                    return False

    return True

def evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate fitness as the sum of radii with constraint validation."""
    if not check_constraints(circles):
        return -np.inf
    
    return np.sum(circles[:, 2])

def compute_voronoi_criticality(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """Compute criticality scores for each circle based on Voronoi diagram."""
    if len(circles) < 3:
        return np.ones(len(circles)) * 0.5
    
    try:
        # Get circle centers
        points = circles[:, :2]
        
        # Shift points to avoid boundary issues
        shifted_points = points.copy()
        shifted_points[:, 0] = np.clip(shifted_points[:, 0], 0.01, rect_width - 0.01)
        shifted_points[:, 1] = np.clip(shifted_points[:, 1], 0.01, rect_height - 0.01)
        
        # Compute Voronoi diagram
        vor = Voronoi(shifted_points)
        
        # Compute area of Voronoi cells
        areas = []
        for i in range(len(shifted_points)):
            # Get vertices of Voronoi cell for point i
            region = vor.regions[vor.point_region[i]]
            if -1 in region:
                # Infinite region, skip
                areas.append(1000000)  # Large area for infinite regions
            else:
                # Compute polygon area
                vertices = [vor.vertices[j] for j in region if j >= 0]
                if len(vertices) < 3:
                    areas.append(1000000)
                else:
                    # Simplified area calculation using cross product
                    vertices_array = np.array(vertices)
                    x = vertices_array[:, 0]
                    y = vertices_array[:, 1]
                    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                    areas.append(area)
        
        # Convert to criticality score (lower area = higher criticality)
        areas = np.array(areas)
        # Normalize to [0,1] where 0 = most critical
        normalized_areas = (areas - areas.min()) / (areas.max() - areas.min() + 1e-8)
        return 1.0 - normalized_areas  # Higher criticality = closer to 1
        
    except:
        # Fallback to uniform criticality if Voronoi fails
        return np.ones(len(circles)) * 0.5

def create_hexagonal_initial_solution(rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """Create initial solution using hexagonal lattice pattern for better packing efficiency."""
    circles = np.zeros((21, 3))
    
    # Hexagonal grid parameters
    rows = 5  # Increase rows for better coverage
    cols = 5  # Increase cols for better coverage
    
    # Adjust for rectangular container
    if rect_width >= rect_height:
        # Width is larger, arrange horizontally
        grid_width = rect_width * 0.9
        grid_height = rect_height * 0.9
    else:
        # Height is larger, arrange vertically
        grid_width = rect_width * 0.9
        grid_height = rect_height * 0.9

    # Calculate spacing based on rectangle dimensions
    cell_width = grid_width / cols
    cell_height = grid_height / rows
    min_cell_dim = min(cell_width, cell_height)

    # Hexagon radius (circles should fit comfortably)
    hex_radius = min_cell_dim * 0.4

    # Arrange in hexagonal pattern
    placed = 0
    for row in range(rows):
        if placed >= 21:
            break
        for col in range(cols):
            if placed >= 21:
                break
                
            # Offset every other row for hexagonal pattern
            offset = (row % 2) * (cell_width / 2)
            x = offset + col * cell_width + cell_width / 2
            y = row * cell_height + cell_height / 2
            
            # Ensure we're within bounds
            x = np.clip(x, hex_radius, rect_width - hex_radius)
            y = np.clip(y, hex_radius, rect_height - hex_radius)
            
            # Adjust radius to prevent boundary issues
            max_radius = min(x, y, rect_width - x, rect_height - y)
            r = min(hex_radius, max_radius * 0.8)
            
            circles[placed] = [x, y, r]
            placed += 1
    
    # Fill remaining positions with small random circles
    for i in range(placed, 21):
        # Place remaining circles randomly but within bounds
        x = np.random.uniform(hex_radius, rect_width - hex_radius)
        y = np.random.uniform(hex_radius, rect_height - hex_radius)
        r = np.random.uniform(0.005, hex_radius * 0.5)
        circles[i] = [x, y, r]
        
    return circles

def create_voronoi_initial_solution(rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """Create initial solution inspired by Voronoi diagrams for better spatial distribution."""
    circles = np.zeros((21, 3))
    
    # Start with a few strategically placed circles
    # Corner placements
    corner_positions = [
        (0.2 * rect_width, 0.2 * rect_height),
        (0.8 * rect_width, 0.2 * rect_height),
        (0.2 * rect_width, 0.8 * rect_height),
        (0.8 * rect_width, 0.8 * rect_height),
        (rect_width/2, rect_height/2)
    ]
    
    placed = 0
    for x, y in corner_positions:
        if placed >= 21:
            break
        r = min(x, y, rect_width - x, rect_height - y) * 0.15
        circles[placed] = [x, y, r]
        placed += 1
    
    # Fill remaining positions using a greedy approach inspired by Voronoi
    max_attempts = 10000
    for attempt in range(max_attempts):
        if placed >= 21:
            break
            
        # Pick a random point and see if it's far enough from existing circles
        x = np.random.uniform(0.05 * rect_width, 0.95 * rect_width)
        y = np.random.uniform(0.05 * rect_height, 0.95 * rect_height)
        
        # Find closest existing circle
        min_dist = float('inf')
        for i in range(placed):
            existing_x, existing_y, existing_r = circles[i]
            distance = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
            min_dist = min(min_dist, distance)
        
        # If it's far enough, place it with appropriate radius
        if min_dist > 0.1 * min(rect_width, rect_height):  # Minimum distance threshold
            r = np.random.uniform(0.01, min(x, y, rect_width - x, rect_height - y) * 0.2)
            circles[placed] = [x, y, r]
            placed += 1
    
    # Fill remaining positions with small random circles
    for i in range(placed, 21):
        x = np.random.uniform(0.05 * rect_width, 0.95 * rect_width)
        y = np.random.uniform(0.05 * rect_height, 0.95 * rect_height)
        r = np.random.uniform(0.005, 0.05)
        circles[i] = [x, y, r]
    
    return circles

def create_strategic_initial_solution(rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """Create initial solution with strategic placement near boundaries and center."""
    circles = np.zeros((21, 3))
    
    # Place circles in key locations
    # Center
    circles[0] = [rect_width / 2, rect_height / 2, min(rect_width, rect_height) * 0.1]
    
    # Corners
    corners = [
        (rect_width * 0.1, rect_height * 0.1),
        (rect_width * 0.9, rect_height * 0.1),
        (rect_width * 0.1, rect_height * 0.9),
        (rect_width * 0.9, rect_height * 0.9)
    ]
    
    placed = 1
    for x, y in corners:
        if placed >= 21:
            break
        r = min(x, y, rect_width - x, rect_height - y) * 0.1
        circles[placed] = [x, y, r]
        placed += 1
    
    # Along edges (not corners)
    edges = [
        (rect_width * 0.5, rect_height * 0.1),  # Top edge
        (rect_width * 0.5, rect_height * 0.9),  # Bottom edge
        (rect_width * 0.1, rect_height * 0.5),  # Left edge
        (rect_width * 0.9, rect_height * 0.5),  # Right edge
    ]
    
    for x, y in edges:
        if placed >= 21:
            break
        r = min(x, y, rect_width - x, rect_height - y) * 0.08
        circles[placed] = [x, y, r]
        placed += 1
    
    # Fill remaining with uniform random placement
    for i in range(placed, 21):
        x = np.random.uniform(0.05 * rect_width, 0.95 * rect_width)
        y = np.random.uniform(0.05 * rect_height, 0.95 * rect_height)
        r = np.random.uniform(0.005, min(x, y, rect_width - x, rect_height - y) * 0.15)
        circles[i] = [x, y, r]
    
    return circles

def create_initial_solution(rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """Create a high-quality initial solution using multiple strategies."""
    # Use multiple initialization strategies and pick the best
    strategies = [
        create_hexagonal_initial_solution,
        create_voronoi_initial_solution,
        create_strategic_initial_solution
    ]
    
    best_solution = None
    best_fitness = -np.inf
    
    for strategy in strategies:
        solution = strategy(rect_width, rect_height)
        fitness = evaluate_fitness(solution)
        if fitness > best_fitness:
            best_fitness = fitness
            best_solution = solution.copy()
    
    return best_solution if best_solution is not None else create_hexagonal_initial_solution(rect_width, rect_height)

def optimize_rectangle_dimensions(circles: np.ndarray) -> Tuple[float, float]:
    """Heuristic to determine optimal rectangle dimensions."""
    # Estimate minimum width and height needed based on circle radii
    total_area = np.sum(circles[:, 2]**2) * np.pi
    # Assume 60% packing efficiency for circles
    estimated_width = np.sqrt(total_area / 0.6)
    estimated_height = estimated_width
    
    # Use a reasonable range around the estimate
    if estimated_width + estimated_height > 2.0:
        # Normalize to perimeter 4
        scale = 2.0 / (estimated_width + estimated_height)
        estimated_width *= scale
        estimated_height *= scale
    
    # Prefer slightly wider rectangle (more common in practice)
    optimized_width = min(1.8, max(0.2, estimated_width))
    optimized_height = 2.0 - optimized_width
    
    return optimized_width, optimized_height

def compute_conflict_graph(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> List[List[int]]:
    """Compute a graph of conflicting circles to help with optimization."""
    n = len(circles)
    conflict_graph = [[] for _ in range(n)]
    
    if n > 1:
        # Create spatial grid for efficient conflict checking
        grid_size = 0.2  # Size of grid cells - adjust based on typical circle size
        grid_width = int(np.ceil(rect_width / grid_size))
        grid_height = int(np.ceil(rect_height / grid_size))

        # Initialize grid
        grid = {}

        # Place circles in grid cells
        for i in range(n):
            x, y, r = circles[i]
            # Get grid coordinates for this circle
            grid_x = int(x / grid_size)
            grid_y = int(y / grid_size)

            if (grid_x, grid_y) not in grid:
                grid[(grid_x, grid_y)] = []
            grid[(grid_x, grid_y)].append(i)

        # Check for overlaps using grid-based approach
        for i in range(n):
            x1, y1, r1 = circles[i]
            # Get grid coordinates for this circle
            grid_x = int(x1 / grid_size)
            grid_y = int(y1 / grid_size)

            # Check this cell and adjacent cells
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor_cell = (grid_x + dx, grid_y + dy)
                    if neighbor_cell in grid:
                        for j in grid[neighbor_cell]:
                            if i != j:  # Don't compare with self
                                x2, y2, r2 = circles[j]
                                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                                if distance < (r1 + r2):
                                    conflict_graph[i].append(j)
                                    conflict_graph[j].append(i)
    
    return conflict_graph

def voronoi_guided_local_optimization(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0, 
                                     max_iterations: int = 500) -> np.ndarray:
    """
    Perform local optimization guided by Voronoi criticality analysis
    """
    # Compute criticality scores
    criticality_scores = compute_voronoi_criticality(circles, rect_width, rect_height)
    
    optimized = circles.copy()
    
    for iteration in range(max_iterations):
        # Identify non-conflicting circles to expand (higher criticality = less urgent)
        # We'll prioritize low-criticality circles for expansion
        expandable_circles = []
        for i in range(len(optimized)):
            if criticality_scores[i] < 0.5:  # Focus on medium-low criticality circles
                expandable_circles.append(i)
        
        if not expandable_circles:
            # If no expandable circles, try to resolve conflicts with smallest moves
            break
            
        # Choose a circle to potentially increase radius
        circle_idx = np.random.choice(expandable_circles)
        
        # Try to increase radius while maintaining constraints
        old_radius = optimized[circle_idx, 2]
        old_x, old_y = optimized[circle_idx, :2]
        
        # Calculate maximum possible radius
        max_possible_radius = min(old_x, old_y, rect_width - old_x, rect_height - old_y)
        
        # Try increasing radius with small step
        step_size = 0.001
        new_radius = old_radius + step_size
        
        # Ensure the new radius doesn't violate space constraints
        if new_radius <= max_possible_radius:
            # Test if the new configuration is valid
            test_config = optimized.copy()
            test_config[circle_idx, 2] = new_radius
            
            # Check if new configuration is valid
            if check_constraints(test_config, rect_width, rect_height):
                optimized = test_config
                continue  # Continue optimizing with same circle
                    
        # If we can't increase the radius, try moving the circle
        # Random movement in small steps
        move_step = 0.005
        new_x = old_x + np.random.uniform(-move_step, move_step)
        new_y = old_y + np.random.uniform(-move_step, move_step)
        
        # Keep within bounds
        new_x = np.clip(new_x, old_radius, rect_width - old_radius)
        new_y = np.clip(new_y, old_radius, rect_height - old_radius)
        
        # Try new configuration with movement
        test_config = optimized.copy()
        test_config[circle_idx, 0] = new_x
        test_config[circle_idx, 1] = new_y
        
        if check_constraints(test_config, rect_width, rect_height):
            optimized = test_config
            
    return optimized

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    rect_width = 1.0
    rect_height = 1.0

    # Initialize with good starting configuration
    best_solution = create_initial_solution(rect_width, rect_height)
    best_fitness = evaluate_fitness(best_solution)
    
    # Allow time for better optimization
    start_time = time.time()
    max_time_seconds = 58  # Leave 2 seconds for final processing
    
    # Main optimization loop
    for iteration in range(5000):  # Limited iterations for time control
        if time.time() - start_time > max_time_seconds:
            break
            
        # Create new candidate solution
        candidate = best_solution.copy()
        
        # Apply various optimizations
        # 1. Local Voronoi-guided optimization
        candidate = voronoi_guided_local_optimization(candidate, rect_width, rect_height, max_iterations=50)
        
        # 2. Try to improve with small random changes to several circles
        for _ in range(10):  # 10 random tweaks
            idx = np.random.randint(0, 21)
            # Slight position changes
            candidate[idx, 0] += np.random.normal(0, 0.001)
            candidate[idx, 1] += np.random.normal(0, 0.001)
            # Radius adjustments
            candidate[idx, 2] *= np.exp(np.random.normal(0, 0.01))
            candidate[idx, 2] = max(0.001, candidate[idx, 2])
            
            # Keep within bounds
            r = candidate[idx, 2]
            candidate[idx, 0] = np.clip(candidate[idx, 0], r, rect_width - r)
            candidate[idx, 1] = np.clip(candidate[idx, 1], r, rect_height - r)
        
        # Check if new solution is better
        new_fitness = evaluate_fitness(candidate)
        if new_fitness > best_fitness:
            best_fitness = new_fitness
            best_solution = candidate.copy()
            
        # Occasionally try to optimize rectangle dimensions
        if iteration % 100 == 0 and iteration > 0:
            optimized_width, optimized_height = optimize_rectangle_dimensions(best_solution)
            rect_width = optimized_width
            rect_height = optimized_height

    # Final fine-tuning
    final_solution = voronoi_guided_local_optimization(best_solution, rect_width, rect_height, max_iterations=100)
    
    # One final validation check
    if evaluate_fitness(final_solution) > best_fitness:
        best_solution = final_solution
    
    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")