# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
import time
from typing import Tuple, List

# Global constants for the optimization
BOUNDARY_MARGIN = 0.01
MAX_LOCAL_SEARCH_ITERATIONS = 300

def is_valid_configuration(circles: np.ndarray) -> bool:
    """Check if a configuration of circles is valid (no overlaps, fully contained)."""
    n = len(circles)

    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0 or x < r + BOUNDARY_MARGIN or x > 1-r - BOUNDARY_MARGIN or y < r + BOUNDARY_MARGIN or y > 1-r - BOUNDARY_MARGIN:
            return False

    # Check overlap constraints with early termination
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance_squared = (x1-x2)**2 + (y1-y2)**2
            min_distance_squared = (r1 + r2)**2
            if distance_squared < min_distance_squared:
                return False

    return True

def evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate fitness as the sum of radii."""
    return np.sum(circles[:, 2])

def create_voronoi_initialization(n_circles: int) -> np.ndarray:
    """Create initial configuration using enhanced Voronoi-based spreading."""
    # Generate more diverse initial points using a combination of strategies
    points = []

    # 1. Grid-based points
    grid_size = int(np.ceil(np.sqrt(n_circles)) * 1.3)  # Larger grid for better spread
    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) < n_circles * 3:  # Generate more points than needed
                x = (j + 0.5) / grid_size
                y = (i + 0.5) / grid_size
                points.append([x, y])

    # 2. Random points
    random_points = np.random.rand(n_circles * 3, 2)
    points.extend(random_points.tolist())

    # 3. Boundary points for better edge coverage
    for _ in range(n_circles * 2):
        side = np.random.randint(0, 4)
        if side == 0:  # Top
            points.append([np.random.rand(), 1.0 - BOUNDARY_MARGIN])
        elif side == 1:  # Bottom
            points.append([np.random.rand(), BOUNDARY_MARGIN])
        elif side == 2:  # Left
            points.append([BOUNDARY_MARGIN, np.random.rand()])
        else:  # Right
            points.append([1.0 - BOUNDARY_MARGIN, np.random.rand()])

    points = np.array(points)[:n_circles * 5]  # Take enough points
    points = np.clip(points, BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)

    # Compute Voronoi diagram
    try:
        vor = Voronoi(points)
        # Use Voronoi cell centers as initial circle positions
        centroids = vor.points[vor.point_region[:-1]]  # Exclude infinite region

        # Limit to number of circles needed
        selected_centroids = centroids[:n_circles]

        # Create circles with initial radii based on Voronoi cell sizes
        circles = np.zeros((n_circles, 3))
        for i in range(n_circles):
            x, y = selected_centroids[i]

            # Find nearest neighbor to estimate appropriate radius
            distances = np.sqrt(np.sum((selected_centroids - [x, y])**2, axis=1))
            distances = distances[distances > 0]  # Exclude self-distance
            if len(distances) > 0:
                avg_distance = np.min(distances) * 0.4
                radius = min(avg_distance, 0.35)  # Increased max radius
            else:
                radius = 0.1

            # Ensure it's within bounds
            radius = min(radius, x - BOUNDARY_MARGIN, 1 - x - BOUNDARY_MARGIN,
                        y - BOUNDARY_MARGIN, 1 - y - BOUNDARY_MARGIN)

            circles[i] = [x, y, max(radius, 0.001)]

        return circles
    except:
        # Fallback to grid-based initialization if Voronoi fails
        return generate_grid_initialization(n_circles)

def generate_grid_initialization(n_circles: int) -> np.ndarray:
    """Generate grid-based initial configuration."""
    circles = np.zeros((n_circles, 3))
    grid_size = int(np.ceil(np.sqrt(n_circles)))
    spacing = 1.0 / grid_size
    r = spacing * 0.35  # Slightly larger initial radius

    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if count < n_circles:
                x = (j + 0.5) * spacing
                y = (i + 0.5) * spacing
                # Adjust for boundary constraints
                x = np.clip(x, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                y = np.clip(y, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                circles[count] = [x, y, r]
                count += 1

    return circles

def preprocess_critical_overlaps(circles: np.ndarray) -> np.ndarray:
    """Preprocess to resolve the most critical overlaps before general optimization."""
    processed = circles.copy()
    
    # Identify and resolve critical overlaps
    n = len(processed)
    resolved = True
    while resolved:
        resolved = False
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = processed[i]
                x2, y2, r2 = processed[j]
                
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                min_dist = r1 + r2
                
                if dist < min_dist:
                    # Resolve by moving circles apart
                    dx = x2 - x1
                    dy = y2 - y1
                    if dx == 0 and dy == 0:
                        # Circles at same position, move randomly
                        angle = np.random.uniform(0, 2*np.pi)
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                    
                    dist = np.sqrt(dx*dx + dy*dy) + 1e-8
                    dx /= dist
                    dy /= dist
                    
                    # Move apart by the overlap amount
                    overlap = min_dist - dist
                    move_amount = overlap * 0.5
                    
                    # Apply to both circles
                    processed[i, 0] -= dx * move_amount
                    processed[i, 1] -= dy * move_amount
                    processed[j, 0] += dx * move_amount
                    processed[j, 1] += dy * move_amount
                    
                    # Keep within boundaries
                    processed[i, 0] = np.clip(processed[i, 0], r1 + BOUNDARY_MARGIN, 1 - r1 - BOUNDARY_MARGIN)
                    processed[i, 1] = np.clip(processed[i, 1], r1 + BOUNDARY_MARGIN, 1 - r1 - BOUNDARY_MARGIN)
                    processed[j, 0] = np.clip(processed[j, 0], r2 + BOUNDARY_MARGIN, 1 - r2 - BOUNDARY_MARGIN)
                    processed[j, 1] = np.clip(processed[j, 1], r2 + BOUNDARY_MARGIN, 1 - r2 - BOUNDARY_MARGIN)
                    
                    resolved = True
                    break
            if resolved:
                break
    
    return processed

def constraint_aware_local_search(circles: np.ndarray, max_iterations: int = MAX_LOCAL_SEARCH_ITERATIONS) -> np.ndarray:
    """
    Apply a constrained local search optimization to improve solution quality.
    This is a novel approach that simultaneously optimizes radii and positions.
    """
    # Clone input to avoid modifying original
    current_solution = circles.copy()

    # Preprocess to resolve critical overlaps
    current_solution = preprocess_critical_overlaps(current_solution)

    # Keep track of best solution found during local search
    best_solution = current_solution.copy()
    best_fitness = evaluate_fitness(current_solution)

    for iteration in range(max_iterations):
        improved = False

        # Strategy 1: Try to expand each circle's radius while maintaining constraints
        for i in range(len(current_solution)):
            original_r = current_solution[i, 2]
            x, y, r = current_solution[i]

            # Calculate maximum possible radius at this position
            max_radius = min(x - BOUNDARY_MARGIN, 1 - x - BOUNDARY_MARGIN,
                           y - BOUNDARY_MARGIN, 1 - y - BOUNDARY_MARGIN)

            # Try to expand radius
            if max_radius > r:
                # Binary search for maximum safe expansion with more iterations
                low = 0.0
                high = max_radius - r
                best_expansion = 0.0

                for _ in range(20):  # More iterations for better precision
                    test_expansion = (low + high) / 2
                    test_radius = r + test_expansion

                    # Check if expansion violates any constraints
                    valid = True
                    for j in range(len(current_solution)):
                        if i != j:
                            pos_j = current_solution[j, :2]
                            r_j = current_solution[j, 2]
                            dist = np.sqrt((x - pos_j[0])**2 + (y - pos_j[1])**2)
                            if dist < (test_radius + r_j):
                                valid = False
                                break

                    if valid:
                        best_expansion = test_expansion
                        low = test_expansion
                    else:
                        high = test_expansion

                if best_expansion > 0:
                    current_solution[i, 2] = r + best_expansion
                    improved = True

        # Strategy 2: Try position adjustments to resolve overlaps
        for i in range(len(current_solution)):
            x, y, r = current_solution[i]

            # Collect overlapping neighbors
            overlapping_pairs = []
            for j in range(len(current_solution)):
                if i != j:
                    pos_j = current_solution[j, :2]
                    r_j = current_solution[j, 2]
                    dist = np.sqrt((x - pos_j[0])**2 + (y - pos_j[1])**2)
                    if dist < (r + r_j):
                        overlapping_pairs.append((j, dist))

            # If there are overlaps, try adjusting position
            if overlapping_pairs:
                # Try several small moves
                best_move = [0.0, 0.0]
                best_score = -1000

                # Use a more thorough search around current position
                moves = []
                for dx in [-0.03, -0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02, 0.03]:
                    for dy in [-0.03, -0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02, 0.03]:
                        moves.append((dx, dy))

                for dx, dy in moves:
                    test_x = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, x + dx))
                    test_y = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, y + dy))

                    # Check if move helps
                    score = 0
                    valid = True

                    # Check overlap with others
                    for j in range(len(current_solution)):
                        if i != j:
                            pos_j = current_solution[j, :2]
                            r_j = current_solution[j, 2]
                            dist = np.sqrt((test_x - pos_j[0])**2 + (test_y - pos_j[1])**2)
                            if dist < (r + r_j):
                                valid = False
                                break
                            # Score based on how much we improve overlap (closer to minimum distance)
                            if dist < (r + r_j + 0.005):
                                score -= 10  # Penalty for remaining overlap
                            else:
                                score += (dist - (r + r_j))  # Reward for better separation

                    if valid and score > best_score:
                        best_score = score
                        best_move = [dx, dy]

                # Apply best move if beneficial
                if best_score > -1000 and best_score > 0:
                    current_solution[i, 0] = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, x + best_move[0]))
                    current_solution[i, 1] = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, y + best_move[1]))
                    improved = True

        # Strategy 3: Global improvement by adjusting all positions
        if not improved and iteration % 5 == 0:  # Only do occasionally
            # Try small random adjustments for all circles
            for i in range(len(current_solution)):
                x, y, r = current_solution[i]
                dx = np.random.normal(0, 0.007)  # Slightly larger perturbation
                dy = np.random.normal(0, 0.007)

                test_x = np.clip(x + dx, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                test_y = np.clip(y + dy, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)

                # Check overlap - be more conservative here
                valid = True
                for j in range(len(current_solution)):
                    if i != j:
                        pos_j = current_solution[j, :2]
                        r_j = current_solution[j, 2]
                        dist = np.sqrt((test_x - pos_j[0])**2 + (test_y - pos_j[1])**2)
                        if dist < (r + r_j):
                            valid = False
                            break

                if valid:
                    current_solution[i, 0] = test_x
                    current_solution[i, 1] = test_y
                    improved = True

        # Strategy 4: Specific overlap resolution for problematic pairs
        if not improved and iteration % 3 == 0:
            # Find the most problematic pairs and try to resolve them
            n = len(current_solution)
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = current_solution[i]
                    x2, y2, r2 = current_solution[j]
                    
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    min_dist = r1 + r2
                    
                    if dist < min_dist:
                        # Calculate force vector to separate them
                        dx = x2 - x1
                        dy = y2 - y1
                        if dx == 0 and dy == 0:
                            angle = np.random.uniform(0, 2*np.pi)
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                        
                        dist = np.sqrt(dx*dx + dy*dy) + 1e-8
                        dx /= dist
                        dy /= dist
                        
                        # Move them apart
                        overlap = min_dist - dist
                        move_amount = overlap * 0.3
                        
                        # Apply to both circles
                        current_solution[i, 0] -= dx * move_amount * 0.5
                        current_solution[i, 1] -= dy * move_amount * 0.5
                        current_solution[j, 0] += dx * move_amount * 0.5
                        current_solution[j, 1] += dy * move_amount * 0.5
                        
                        # Keep within boundaries
                        current_solution[i, 0] = np.clip(current_solution[i, 0], r1 + BOUNDARY_MARGIN, 1 - r1 - BOUNDARY_MARGIN)
                        current_solution[i, 1] = np.clip(current_solution[i, 1], r1 + BOUNDARY_MARGIN, 1 - r1 - BOUNDARY_MARGIN)
                        current_solution[j, 0] = np.clip(current_solution[j, 0], r2 + BOUNDARY_MARGIN, 1 - r2 - BOUNDARY_MARGIN)
                        current_solution[j, 1] = np.clip(current_solution[j, 1], r2 + BOUNDARY_MARGIN, 1 - r2 - BOUNDARY_MARGIN)
                        
                        improved = True
                        break
                if improved:
                    break

        # Update best solution if current is better
        current_fitness = evaluate_fitness(current_solution)
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_solution = current_solution.copy()

        if not improved:
            break

    return best_solution

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility

    n = 26
    best_solution = None
    best_fitness = -np.inf

    # Generate initial population with Voronoi-based approach
    initial_candidates = []
    for _ in range(20):  # Generate more candidates for better exploration
        circles = create_voronoi_initialization(n)
        if is_valid_configuration(circles):
            initial_candidates.append(circles)

    # If no valid candidates, fall back to grid
    if not initial_candidates:
        circles = generate_grid_initialization(n)
        return circles

    # Select the best initial candidate
    initial_fitnesses = [evaluate_fitness(c) for c in initial_candidates]
    best_initial_idx = np.argmax(initial_fitnesses)
    current_solution = initial_candidates[best_initial_idx].copy()

    # Apply local search to the best initial configuration
    refined_solution = constraint_aware_local_search(current_solution)
    current_fitness = evaluate_fitness(refined_solution)

    # Track best solution so far
    if current_fitness > best_fitness:
        best_fitness = current_fitness
        best_solution = refined_solution.copy()

    # Do additional local search refinements
    for _ in range(5):  # Multiple rounds of refinement
        # Apply local search again
        local_refined = constraint_aware_local_search(refined_solution, 150)
        local_fitness = evaluate_fitness(local_refined)

        if local_fitness > best_fitness:
            best_fitness = local_fitness
            best_solution = local_refined.copy()
            refined_solution = local_refined.copy()
        else:
            # Try another approach if not improving
            # Perturb slightly and try again
            perturbed = refined_solution.copy()
            for i in range(len(perturbed)):
                # Small random perturbation
                perturbed[i, 0] += np.random.normal(0, 0.007)  # Slightly larger perturbation
                perturbed[i, 1] += np.random.normal(0, 0.007)
                # Keep within bounds
                perturbed[i, 0] = np.clip(perturbed[i, 0],
                                        perturbed[i, 2] + BOUNDARY_MARGIN,
                                        1 - perturbed[i, 2] - BOUNDARY_MARGIN)
                perturbed[i, 1] = np.clip(perturbed[i, 1],
                                        perturbed[i, 2] + BOUNDARY_MARGIN,
                                        1 - perturbed[i, 2] - BOUNDARY_MARGIN)

            # Validate and update
            if is_valid_configuration(perturbed):
                perturbed_fitness = evaluate_fitness(perturbed)
                if perturbed_fitness > best_fitness:
                    best_fitness = perturbed_fitness
                    best_solution = perturbed.copy()
                    refined_solution = perturbed.copy()

    return best_solution if best_solution is not None else generate_grid_initialization(n)


# EVOLVE-BLOCK-END