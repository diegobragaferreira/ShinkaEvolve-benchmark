# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
import random
from typing import Tuple, List

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

    # Check overlap constraints using spatial indexing for O(n) complexity instead of O(n^2)
    if n > 1:
        # Create spatial grid
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
    # Determine grid dimensions
    rows = int(np.ceil(np.sqrt(21)))
    cols = int(np.ceil(21 / rows))

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

def compute_forces(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """Compute net forces on each circle including repulsion, boundary constraints, and attraction to center."""
    n = len(circles)
    forces = np.zeros((n, 2))
    
    # Repulsion forces between overlapping circles
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            
            dx = x2 - x1
            dy = y2 - y1
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance > 0 and distance < (r1 + r2):
                # Repulsive force
                force_magnitude = 1.0 / (distance * distance)
                forces[i, 0] -= force_magnitude * dx / distance
                forces[i, 1] -= force_magnitude * dy / distance
                forces[j, 0] += force_magnitude * dx / distance
                forces[j, 1] += force_magnitude * dy / distance
    
    # Boundary forces
    for i in range(n):
        x, y, r = circles[i]
        # Left boundary
        if x - r < 0:
            forces[i, 0] += 100 * (r - x)
        # Right boundary
        if x + r > rect_width:
            forces[i, 0] -= 100 * (x + r - rect_width)
        # Bottom boundary
        if y - r < 0:
            forces[i, 1] += 100 * (r - y)
        # Top boundary
        if y + r > rect_height:
            forces[i, 1] -= 100 * (y + r - rect_height)
    
    # Attraction to center (to keep circles distributed)
    center_x, center_y = rect_width / 2, rect_height / 2
    for i in range(n):
        x, y, r = circles[i]
        dx = center_x - x
        dy = center_y - y
        distance = np.sqrt(dx*dx + dy*dy)
        if distance > 0:
            # Attractive force to center
            force_magnitude = 0.01 / (distance + 1e-8)
            forces[i, 0] += force_magnitude * dx / distance
            forces[i, 1] += force_magnitude * dy / distance
    
    return forces

def physics_based_optimization(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0, 
                              max_iterations: int = 500, dt: float = 0.01) -> np.ndarray:
    """Optimize circle arrangement using physics simulation with adaptive time stepping."""
    current_circles = circles.copy()
    n = len(current_circles)
    
    # Compute Voronoi criticality to focus optimization on critical regions
    criticality_scores = compute_voronoi_criticality(current_circles, rect_width, rect_height)
    
    # Adaptive time stepping based on criticality
    for iteration in range(max_iterations):
        # Compute forces
        forces = compute_forces(current_circles, rect_width, rect_height)
        
        # Apply forces with adaptive step size based on criticality
        # Circles in high-criticality regions move slower to maintain stability
        for i in range(n):
            # Adjust time step based on criticality (higher criticality = smaller step)
            crit_factor = 1.0 - criticality_scores[i]  # Low criticality = more movement
            step_size = dt * (0.5 + 0.5 * crit_factor)  # Range from 0.5dt to dt
            
            # Apply force to position
            current_circles[i, 0] += forces[i, 0] * step_size
            current_circles[i, 1] += forces[i, 1] * step_size
            
            # Ensure circles stay within bounds
            x, y, r = current_circles[i]
            current_circles[i, 0] = np.clip(x, r, rect_width - r)
            current_circles[i, 1] = np.clip(y, r, rect_height - r)
        
        # Occasionally optimize radii (every 10 iterations)
        if iteration % 10 == 0:
            # Try to increase radii where possible
            for i in range(n):
                x, y, r = current_circles[i]
                # Maximum possible radius
                max_radius = min(x, y, rect_width - x, rect_height - y)
                
                # Try to increase radius slightly if it doesn't violate constraints
                if r < max_radius * 0.99:
                    test_r = min(r * 1.02, max_radius * 0.99)
                    test_circles = current_circles.copy()
                    test_circles[i, 2] = test_r
                    
                    # Check if this would still satisfy constraints
                    if check_constraints(test_circles, rect_width, rect_height):
                        current_circles[i, 2] = test_r
    
    return current_circles

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

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    rect_width = 1.0
    rect_height = 1.0

    # Create initial solution
    initial_solution = create_initial_solution(rect_width, rect_height)
    
    # Apply physics-based optimization
    optimized_solution = physics_based_optimization(initial_solution, rect_width, rect_height)
    
    # Refine with additional local optimization cycles
    for _ in range(3):
        # Run physics optimization again with potentially better starting point
        refined_solution = physics_based_optimization(optimized_solution, rect_width, rect_height)
        
        # Check if improvement occurred
        old_fitness = evaluate_fitness(optimized_solution)
        new_fitness = evaluate_fitness(refined_solution)
        
        if new_fitness > old_fitness:
            optimized_solution = refined_solution
        else:
            break
    
    # Final constraint validation
    if not check_constraints(optimized_solution, rect_width, rect_height):
        # If constraints not satisfied, fall back to original solution
        return initial_solution
    
    return optimized_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")