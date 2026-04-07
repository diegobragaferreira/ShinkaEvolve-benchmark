# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

def compute_local_density(circles, point, k=5):
    """Compute local density around a point using k-nearest neighbors"""
    if len(circles) <= 1:
        return 0.0

    # Calculate distances to all other circles
    distances = np.sqrt(np.sum((circles[:, :2] - point)**2, axis=1))

    # Get indices of k nearest neighbors (excluding self)
    sorted_indices = np.argsort(distances)
    nearest_indices = sorted_indices[1:k+1] if len(sorted_indices) > 1 else sorted_indices

    # Return average distance to neighbors
    if len(nearest_indices) == 0:
        return 0.0
    return np.mean(distances[nearest_indices])

def initialize_hexagonal_grid(n_circles, padding=0.05):
    """Initialize circles in a hexagonal grid pattern with density awareness"""
    # Determine grid dimensions
    rows = int(np.ceil(np.sqrt(n_circles)))
    cols = int(np.ceil(n_circles / rows))

    # Create hexagonal pattern
    spacing_x = (1 - 2*padding) / cols
    spacing_y = (1 - 2*padding) / rows

    # Adjust spacing for hexagonal arrangement
    hex_spacing_x = spacing_x
    hex_spacing_y = spacing_y * np.sqrt(3)/2

    circles = []
    circle_count = 0

    for i in range(rows):
        for j in range(cols):
            if circle_count >= n_circles:
                break

            # Hexagonal offset
            x_offset = (j if i % 2 == 0 else j + 0.5) * hex_spacing_x + padding
            y_offset = i * hex_spacing_y + padding

            # Add some randomness to avoid perfect grid
            x = max(padding, min(1-padding, x_offset + np.random.normal(0, 0.01*hex_spacing_x)))
            y = max(padding, min(1-padding, y_offset + np.random.normal(0, 0.01*hex_spacing_y)))

            # Set initial radius based on spacing and density
            base_radius = min(hex_spacing_x, hex_spacing_y) * 0.4
            circles.append([x, y, base_radius])
            circle_count += 1

        if circle_count >= n_circles:
            break

    # Ensure exactly n_circles
    while len(circles) < n_circles:
        # Add random circles in valid positions
        x = np.random.uniform(padding, 1-padding)
        y = np.random.uniform(padding, 1-padding)
        # Radius based on proximity to existing circles
        r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
        circles.append([x, y, r])

    return np.array(circles[:n_circles])

def is_valid_placement(circles, threshold=1e-6):
    """Check if circle configuration is valid with improved tolerance"""
    n = len(circles)
    if n == 0:
        return False

    # Check boundary constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False

    # Check overlap constraints using KDTree for efficiency
    if n > 1:
        positions = circles[:, :2]
        tree = cKDTree(positions)

        # Query pairs within 2*(max_radius) distance
        max_radius = np.max(circles[:, 2])
        pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')

        for i, j in pairs:
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if distance < r1 + r2 - threshold:
                return False

    return True

def calculate_penalty(circle_config, weight_boundary=1000, weight_overlap=1000):
    """Calculate penalty for constraint violations with adaptive weights"""
    penalty = 0.0
    n = len(circle_config)

    # Boundary penalties (smooth exponential)
    for i in range(n):
        x, y, r = circle_config[i]
        # Penalties for going outside boundaries
        if x - r < 0:
            penalty += weight_boundary * np.exp(10 * (x - r))
        if x + r > 1:
            penalty += weight_boundary * np.exp(10 * (x + r - 1))
        if y - r < 0:
            penalty += weight_boundary * np.exp(10 * (y - r))
        if y + r > 1:
            penalty += weight_boundary * np.exp(10 * (y + r - 1))

    # Overlap penalties (smooth exponential)
    if n > 1:
        positions = circle_config[:, :2]
        radii = circle_config[:, 2]
        tree = cKDTree(positions)

        # Query pairs within 2*(max_radius) distance
        max_radius = np.max(radii)
        pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')

        for i, j in pairs:
            x1, y1, r1 = circle_config[i]
            x2, y2, r2 = circle_config[j]
            distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if distance < r1 + r2:
                # Adaptive penalty based on violation magnitude
                violation = r1 + r2 - distance
                penalty += weight_overlap * np.exp(10 * violation)

    return penalty

def objective_function(circle_config):
    """Objective function to maximize sum of radii"""
    # Convert to array if needed
    if not isinstance(circle_config, np.ndarray):
        circle_config = np.array(circle_config)

    # We want to maximize sum of radii, so return negative sum plus penalty
    return -np.sum(circle_config[:, 2]) + calculate_penalty(circle_config)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    max_iter = 1000
    tolerance = 1e-6

    best_sum_radii = 0.0
    best_circles = None

    # Multi-start optimization with different initial configurations
    num_starts = 10
    for start_idx in range(num_starts):
        # Initialize with hexagonal grid
        np.random.seed(start_idx * 12345)  # Fixed seed for reproducibility
        circles = initialize_hexagonal_grid(n)

        # Prepare flattened parameter vector (x, y, r for each circle)
        initial_params = circles.flatten()

        # Define bounds for optimization (x,y in [0.05, 0.95], r in [0.01, 0.4])
        bounds = []
        for _ in range(n):
            bounds.extend([(0.05, 0.95), (0.05, 0.95), (0.01, 0.4)])

        # Optimization with L-BFGS-B
        try:
            result = minimize(
                objective_function,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': tolerance, 'gtol': tolerance}
            )

            # Extract optimized parameters
            optimized_params = result.x
            optimized_circles = optimized_params.reshape(-1, 3)

            # Validate the solution
            if is_valid_placement(optimized_circles):
                current_sum = np.sum(optimized_circles[:, 2])
                if current_sum > best_sum_radii:
                    best_sum_radii = current_sum
                    best_circles = optimized_circles.copy()

        except Exception as e:
            # Continue with other starts if one fails
            continue

    # If no valid solution found, fall back to initial configuration
    if best_circles is None:
        np.random.seed(42)  # Fixed seed
        best_circles = initialize_hexagonal_grid(n)

    # Final validation and correction
    if not is_valid_placement(best_circles):
        # Reset to hexagonal grid if final configuration is invalid
        np.random.seed(42)  # Fixed seed
        best_circles = initialize_hexagonal_grid(n)

    # Ensure all circles are within bounds
    for i in range(len(best_circles)):
        x, y, r = best_circles[i]
        # Constrain positions to valid range
        best_circles[i][0] = max(r, min(1-r, x))
        best_circles[i][1] = max(r, min(1-r, y))
        # Constrain radii to valid range
        best_circles[i][2] = max(0.01, min(0.4, r))

    return best_circles

# EVOLVE-BLOCK-END