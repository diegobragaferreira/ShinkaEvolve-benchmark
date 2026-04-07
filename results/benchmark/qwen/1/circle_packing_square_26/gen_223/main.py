# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
import time
from typing import Tuple, List

# Global constants for the optimization
BOUNDARY_MARGIN = 0.01
MAX_LOCAL_SEARCH_ITERATIONS = 200

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
    """Create initial configuration using enhanced Voronoi-based spreading with area-based radius estimation."""
    # Generate more diverse initial points using a combination of strategies
    points = []

    # 1. Grid-based points with better spacing
    grid_size = int(np.ceil(np.sqrt(n_circles)) * 1.2)  # Slightly larger grid
    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) < n_circles * 2:  # Generate more points than needed
                x = (j + 0.5) / grid_size
                y = (i + 0.5) / grid_size
                points.append([x, y])

    # 2. Random points with better distribution
    # Use Latin Hypercube Sampling or similar for better spread
    random_points = np.random.rand(n_circles * 2, 2)
    points.extend(random_points.tolist())

    # 3. Strategic boundary points for better edge coverage
    for _ in range(n_circles):
        side = np.random.randint(0, 4)
        if side == 0:  # Top
            points.append([np.random.rand(), 1.0 - BOUNDARY_MARGIN])
        elif side == 1:  # Bottom
            points.append([np.random.rand(), BOUNDARY_MARGIN])
        elif side == 2:  # Left
            points.append([BOUNDARY_MARGIN, np.random.rand()])
        else:  # Right
            points.append([1.0 - BOUNDARY_MARGIN, np.random.rand()])

    points = np.array(points)[:n_circles * 3]  # Take enough points
    points = np.clip(points, BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)

    # Compute Voronoi diagram
    try:
        vor = Voronoi(points)

        # Get Voronoi vertices and compute cell areas using a more robust method
        # Use the fact that Voronoi cells are convex polygons

        # For efficiency, let's select the most promising points as circle centers
        # We'll use a selection strategy that favors points with good Voronoi cell characteristics

        # First, get valid Voronoi vertices (excluding infinite regions)
        if hasattr(vor, 'point_region') and len(vor.point_region) > 0:
            # Select vertices that correspond to finite regions
            valid_regions = []
            for i, region in enumerate(vor.point_region[:-1]):  # Exclude last infinite region
                if region != -1:  # Valid region
                    valid_regions.append(i)

            # If we have enough valid regions, use them; otherwise fall back to all points
            if len(valid_regions) >= n_circles:
                selected_indices = valid_regions[:n_circles]
            else:
                # Fall back to all points if we don't have enough valid regions
                selected_indices = list(range(min(n_circles, len(vor.points))))
        else:
            # Fallback to all points
            selected_indices = list(range(min(n_circles, len(vor.points))))

        # Select points as circle centers
        if selected_indices:
            selected_centroids = vor.points[selected_indices]
        else:
            # If no valid indices, fall back to selecting the first n_circles points
            selected_centroids = vor.points[:n_circles]

        # Create circles with initial radii based on Voronoi cell characteristics
        circles = np.zeros((n_circles, 3))
        for i in range(n_circles):
            x, y = selected_centroids[i]

            # Find nearest neighbor to estimate appropriate radius
            distances = np.sqrt(np.sum((selected_centroids - [x, y])**2, axis=1))
            distances = distances[distances > 0]  # Exclude self-distance

            # Use a more sophisticated approach for radius calculation
            if len(distances) > 0:
                # Consider both minimum distance and the distribution of neighbors
                min_distance = np.min(distances)
                avg_distance = np.mean(distances)

                # Estimate radius based on a weighted combination of min and avg distances
                # with preference for smaller spacing (better packing)
                radius = min(min_distance * 0.35, avg_distance * 0.25, 0.3)
            else:
                radius = 0.1

            # Ensure it's within bounds with a more conservative margin
            radius = min(radius,
                        x - BOUNDARY_MARGIN,
                        1 - x - BOUNDARY_MARGIN,
                        y - BOUNDARY_MARGIN,
                        1 - y - BOUNDARY_MARGIN)

            # Add small random perturbation to avoid degenerate cases
            radius = max(radius * (0.95 + np.random.rand() * 0.1), 0.001)

            circles[i] = [x, y, radius]

        return circles
    except Exception as e:
        # Fallback to grid-based initialization if Voronoi fails
        return generate_grid_initialization(n_circles)

def generate_grid_initialization(n_circles: int) -> np.ndarray:
    """Generate grid-based initial configuration."""
    circles = np.zeros((n_circles, 3))
    grid_size = int(np.ceil(np.sqrt(n_circles)))
    spacing = 1.0 / grid_size
    r = spacing * 0.3

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

def constraint_aware_local_search(circles: np.ndarray, max_iterations: int = MAX_LOCAL_SEARCH_ITERATIONS) -> np.ndarray:
    """
    Apply a constrained local search optimization to improve solution quality.
    This is a novel approach that simultaneously optimizes radii and positions.
    """
    # Clone input to avoid modifying original
    current_solution = circles.copy()

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
                # Binary search for maximum safe expansion
                low = 0.0
                high = max_radius - r
                best_expansion = 0.0

                for _ in range(15):  # More iterations for better precision
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
                for dx in [-0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02]:
                    for dy in [-0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02]:
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
                dx = np.random.normal(0, 0.005)  # Smaller perturbation
                dy = np.random.normal(0, 0.005)

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
    for _ in range(10):  # Generate several candidates
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
        local_refined = constraint_aware_local_search(refined_solution, 100)
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
                perturbed[i, 0] += np.random.normal(0, 0.005)
                perturbed[i, 1] += np.random.normal(0, 0.005)
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