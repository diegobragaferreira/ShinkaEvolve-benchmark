# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import pdist, squareform
import math
from typing import Tuple

def compute_distance_matrix(points):
    """Compute pairwise distance matrix for given points."""
    return squareform(pdist(points))

def calculate_min_max_ratio(distance_matrix):
    """Calculate the ratio of minimum to maximum distances."""
    # Exclude diagonal (distance to self)
    off_diagonal = distance_matrix[distance_matrix > 0]
    if len(off_diagonal) == 0:
        return 0.0
    d_min = np.min(off_diagonal)
    d_max = np.max(off_diagonal)
    return d_min / d_max if d_max > 0 else 0.0

def compute_voronoi_areas(points):
    """Compute Voronoi cell areas for each point using proper Voronoi diagram computation."""
    try:
        vor = Voronoi(points)
        areas = []
        for i in range(len(points)):
            # Get the region index for point i
            region_index = vor.point_region[i]
            # Get the vertices of the Voronoi cell for point i
            region = vor.regions[region_index]

            # Handle unbounded regions (contains -1)
            if -1 in region:
                areas.append(0.0)  # Skip unbounded regions
                continue

            if len(region) < 3:
                areas.append(0.0)  # Not enough vertices for a polygon
                continue

            # Get actual vertices for this region
            cell_vertices = [vor.vertices[k] for k in region]
            if len(cell_vertices) < 3:
                areas.append(0.0)
                continue

            # Compute polygon area using Shoelace formula
            verts = np.array(cell_vertices)
            x = verts[:, 0]
            y = verts[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            areas.append(area)

        return np.array(areas)
    except Exception:
        return np.zeros(len(points))

def initialize_points_hexagonal_packing():
    """Initialize points using hexagonal packing principles for better distribution."""
    # Create a 4x4 grid with hexagonal offset pattern
    points = []
    rows = 4
    cols = 4

    # Hexagonal offset pattern
    for i in range(rows):
        for j in range(cols):
            x = j + 0.5 * (i % 2)
            y = i * math.sqrt(3)/2
            points.append([x, y])

    # Normalize to [0,1] range
    points = np.array(points)

    # Scale and shift to fit in unit square
    x_range = np.max(points[:, 0]) - np.min(points[:, 0])
    y_range = np.max(points[:, 1]) - np.min(points[:, 1])

    if x_range > 0:
        points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
    if y_range > 0:
        points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range

    # Add some randomization to break symmetry
    np.random.seed(42)
    points += np.random.normal(0, 0.01, points.shape)

    # Keep within bounds
    points[:, 0] = np.clip(points[:, 0], 0, 1)
    points[:, 1] = np.clip(points[:, 1], 0, 1)

    return points

def initialize_points_regular_grid():
    """Initialize points using regular grid with slight perturbations."""
    # Create a 4x4 grid
    grid_size = 4
    points = np.array([[i/(grid_size-1), j/(grid_size-1)] for i in range(grid_size) for j in range(grid_size)])

    # Add slight random perturbations to avoid symmetric solutions
    np.random.seed(42)
    noise = np.random.normal(0, 0.02, points.shape)
    points += noise
    points = np.clip(points, 0, 1)

    return points

def initialize_points_random():
    """Initialize points with random distribution."""
    np.random.seed(42)
    return np.random.rand(16, 2)

def initialize_points_triangular():
    """Initialize points using triangular lattice."""
    points = []
    rows = 4
    cols = 4

    for i in range(rows):
        for j in range(cols):
            x = j + (i % 2) * 0.5
            y = i * math.sqrt(3)/2
            points.append([x, y])

    points = np.array(points)

    # Normalize
    max_x = cols - 0.5
    max_y = (rows - 1) * math.sqrt(3)/2

    points[:, 0] = points[:, 0] / max_x
    points[:, 1] = points[:, 1] / max_y

    # Add noise
    np.random.seed(42)
    points += np.random.normal(0, 0.015, points.shape)
    points = np.clip(points, 0, 1)

    return points

def evaluate_voronoi_quality(points):
    """Evaluate quality based on Voronoi properties."""
    try:
        vor = Voronoi(points)
        # Measure variance in Voronoi cell areas (more uniform cells = better distribution)
        areas = compute_voronoi_areas(points)
        if len(areas) > 0:
            # Avoid zero areas for numerical stability
            areas = np.maximum(areas, 1e-10)
            area_variance = np.var(areas)
            # Prefer more uniform cell areas (lower variance)
            return -area_variance
        return 0
    except:
        return 0

def discrete_evolution_step(points, current_ratio, iteration, max_iter):
    """Perform a single evolution step guided by Voronoi analysis."""
    # Create a copy of points to modify
    new_points = points.copy()

    # Determine adaptive step size based on iteration (decrease over time)
    if max_iter > 0:
        progress = iteration / max_iter
        step_size = 0.05 * (1 - progress * 0.8)  # Start large, decrease
    else:
        step_size = 0.05

    # Choose a random point to modify
    idx = np.random.randint(len(points))

    # Get Voronoi information for this point
    try:
        vor = Voronoi(points)

        # Compute Voronoi area for current point
        current_areas = compute_voronoi_areas(points)
        current_area = current_areas[idx]

        # Get Voronoi cell vertices for this point
        region_index = vor.point_region[idx]
        region = vor.regions[region_index]

        if -1 in region or len(region) < 3:
            # Fallback for problematic Voronoi cells
            new_points[idx, 0] += np.random.normal(0, step_size * 0.5)
            new_points[idx, 1] += np.random.normal(0, step_size * 0.5)
        else:
            # Get the actual vertices of the Voronoi cell
            cell_vertices = [vor.vertices[k] for k in region]
            cell_polygon = np.array(cell_vertices)

            # Determine move direction based on Voronoi geometry
            if current_area < 0.005:  # Very small Voronoi area (point too clustered)
                # Move away from neighboring points to increase minimum distance
                # Find nearest neighbors based on point proximity
                distances_to_other = np.linalg.norm(points[idx] - points, axis=1)
                distances_to_other[idx] = np.inf  # Ignore self-distance
                nearest_indices = np.argsort(distances_to_other)[:4]  # Top 4 nearest

                # Move away from neighbors weighted by distance
                move_direction = np.zeros(2)
                for neighbor_idx in nearest_indices:
                    diff = points[idx] - points[neighbor_idx]
                    distance = np.linalg.norm(diff)
                    if distance > 1e-8:
                        move_direction += diff / distance * (step_size * 0.8) / (distance + 0.01)

                # Apply move
                new_points[idx] += move_direction

            elif current_area > 0.03:  # Large Voronoi area (point too isolated)
                # Move towards neighbors to compact the distribution
                # Find nearest neighbors based on point proximity
                distances_to_other = np.linalg.norm(points[idx] - points, axis=1)
                distances_to_other[idx] = np.inf  # Ignore self-distance
                nearest_indices = np.argsort(distances_to_other)[:4]  # Top 4 nearest

                # Move towards neighbors
                move_direction = np.zeros(2)
                for neighbor_idx in nearest_indices:
                    diff = points[neighbor_idx] - points[idx]
                    distance = np.linalg.norm(diff)
                    if distance > 1e-8:
                        move_direction += diff / distance * (step_size * 0.8) / (distance + 0.01)

                # Apply move
                new_points[idx] += move_direction
            else:
                # Moderate area - move based on Voronoi shape and neighbor distribution
                # Find nearest neighbors
                distances_to_other = np.linalg.norm(points[idx] - points, axis=1)
                distances_to_other[idx] = np.inf  # Ignore self-distance
                nearest_indices = np.argsort(distances_to_other)[:4]  # Top 4 nearest

                # Move intelligently based on neighbor distribution
                move_direction = np.zeros(2)
                for neighbor_idx in nearest_indices:
                    diff = points[neighbor_idx] - points[idx]
                    distance = np.linalg.norm(diff)
                    if distance > 1e-8:
                        # Direction towards neighbor, but with weighted magnitude
                        weight = 1.0 / (distance + 0.01)
                        move_direction += diff / distance * weight * step_size * 0.6

                # Apply move
                new_points[idx] += move_direction

    except Exception:
        # Fallback: classic random walk if Voronoi fails
        new_points[idx, 0] += np.random.normal(0, step_size * 0.5)
        new_points[idx, 1] += np.random.normal(0, step_size * 0.5)

    # Enforce boundary constraints
    new_points[:, 0] = np.clip(new_points[:, 0], 0, 1)
    new_points[:, 1] = np.clip(new_points[:, 1], 0, 1)

    return new_points

def discrete_voronoi_evolve(points, max_iterations=3000):
    """Evolve points using discrete Voronoi-guided optimization."""
    current_points = points.copy()
    current_ratio = calculate_min_max_ratio(compute_distance_matrix(current_points))

    best_points = current_points.copy()
    best_ratio = current_ratio

    # Track convergence
    stable_count = 0
    prev_ratio = current_ratio
    prev_points = current_points.copy()

    # Adaptive temperature-like parameter for step size
    temp = 0.1

    for iteration in range(max_iterations):
        # Occasionally adjust temperature
        if iteration % 100 == 0 and iteration > 0:
            temp *= 0.99  # Gradually cool down

        # Perform evolution step
        new_points = discrete_evolution_step(current_points, current_ratio, iteration, max_iterations)

        # Calculate new ratio
        new_distance_matrix = compute_distance_matrix(new_points)
        new_ratio = calculate_min_max_ratio(new_distance_matrix)

        # Accept or reject based on ratio improvement
        if new_ratio > current_ratio:
            current_points = new_points.copy()
            current_ratio = new_ratio

            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_points = new_points.copy()
                stable_count = 0  # Reset stability counter
        else:
            # Sometimes accept worse solutions to escape local minima
            if np.random.rand() < np.exp((new_ratio - current_ratio) / temp):
                current_points = new_points.copy()
                current_ratio = new_ratio

        # Convergence check
        if abs(current_ratio - prev_ratio) < 1e-6:
            stable_count += 1
            if stable_count > 50:
                break
        else:
            stable_count = 0

        prev_ratio = current_ratio
        prev_points = current_points.copy()

    return best_points, best_ratio

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    # Try multiple initialization strategies and pick the best
    init_functions = [
        initialize_points_hexagonal_packing,
        initialize_points_regular_grid,
        initialize_points_random,
        initialize_points_triangular
    ]

    best_points = None
    best_ratio = 0

    for init_func in init_functions:
        try:
            initial_points = init_func()

            # Evolve using discrete Voronoi-based approach
            evolved_points, ratio = discrete_voronoi_evolve(initial_points, max_iterations=2000)

            if ratio > best_ratio:
                best_ratio = ratio
                best_points = evolved_points.copy()

        except Exception as e:
            continue  # Skip this strategy if it fails

    # If no valid configuration was found, use a fallback
    if best_points is None:
        initial_points = initialize_points_hexagonal_packing()
        best_points, _ = discrete_voronoi_evolve(initial_points, max_iterations=2000)

    return best_points

# EVOLVE-BLOCK-END