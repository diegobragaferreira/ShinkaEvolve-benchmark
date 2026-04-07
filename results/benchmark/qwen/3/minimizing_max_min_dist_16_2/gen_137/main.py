# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import math
from scipy.optimize import differential_evolution
import time

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

def initialize_points_hexagonal_lattice():
    """Initialize points using a true hexagonal lattice with proper spacing."""
    # Create a hexagonal lattice with 16 points arranged in a 4x4 pattern
    # This creates a more uniform distribution than simple grids
    points = []

    # Hexagonal lattice parameters - using precise values
    sqrt3 = math.sqrt(3)
    row_spacing = sqrt3 / 2
    col_spacing = 1.0

    rows = 4
    cols = 4

    for i in range(rows):
        for j in range(cols):
            # Offset every other row with proper hexagonal spacing
            x = j * col_spacing + (i % 2) * col_spacing / 2
            y = i * row_spacing

            points.append([x, y])

    # Convert to numpy array
    points = np.array(points)

    # Normalize to fit in [0,1] square with better scaling
    x_range = np.max(points[:, 0]) - np.min(points[:, 0])
    y_range = np.max(points[:, 1]) - np.min(points[:, 1])

    # Avoid division by zero
    if x_range > 0:
        points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
    if y_range > 0:
        points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range

    # Apply better boundary-aware scaling with centering
    scale_factor = 0.9
    center_x = np.mean(points[:, 0])
    center_y = np.mean(points[:, 1])

    points[:, 0] = 0.05 + scale_factor * (points[:, 0] - center_x) + 0.5
    points[:, 1] = 0.05 + scale_factor * (points[:, 1] - center_y) + 0.5

    # Add more aggressive symmetry-breaking perturbations with varied magnitudes
    np.random.seed(42)
    # Apply different perturbation magnitudes to break symmetry more effectively
    for i in range(len(points)):
        # Use a pattern that varies perturbation magnitude based on position and index
        base_magnitude = 0.01
        # Vary magnitude based on position and iteration index for better asymmetry
        magnitude_variation = 0.005 * ((i % 7) / 7.0) * (1 + (i // 4) * 0.1)
        perturbation_magnitude = base_magnitude + magnitude_variation
        points[i] += np.random.normal(0, perturbation_magnitude, 2)

    # Clamp to [0,1] bounds
    points = np.clip(points, 0, 1)

    return points

def initialize_points_random():
    """Initialize points randomly with better distribution properties."""
    np.random.seed(42)
    points = np.random.uniform(0, 1, (16, 2))
    return points

def initialize_points_voronoi_based():
    """Initialize points using a Voronoi-inspired approach with better distribution."""
    # Create a 4x4 grid with more sophisticated perturbations
    points = []
    rows = 4
    cols = 4

    # Create more structured grid with better spacing
    for i in range(rows):
        for j in range(cols):
            # Add structured perturbations to avoid perfect symmetry
            x = j + np.random.normal(0, 0.03) * (1 + (i % 2) * 0.2)
            y = i + np.random.normal(0, 0.03) * (1 + (j % 2) * 0.2)
            points.append([x, y])

    points = np.array(points)

    # Normalize to [0,1] range with better control
    x_range = np.max(points[:, 0]) - np.min(points[:, 0])
    y_range = np.max(points[:, 1]) - np.min(points[:, 1])

    if x_range > 0:
        points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
    if y_range > 0:
        points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range

    # Apply boundary constraints with better scaling
    points[:, 0] = np.clip(points[:, 0], 0.02, 0.98)
    points[:, 1] = np.clip(points[:, 1], 0.02, 0.98)

    return points

def optimize_with_simulated_annealing(initial_points, max_iter=1000):
    """Optimize using a modified simulated annealing approach for better exploration."""

    def objective_function(points_flat):
        points = points_flat.reshape((16, 2))
        points = np.clip(points, 0, 1)
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return float('inf')
        return -d_min / d_max

    # Convert to flat array
    x0 = initial_points.flatten()

    # Simulated Annealing with better cooling schedule
    current_points = x0.copy()
    current_energy = objective_function(current_points)

    best_points = current_points.copy()
    best_energy = current_energy

    # Cooling schedule - more adaptive approach
    temp = 0.1
    cooling_rate = 0.9995
    min_temp = 1e-6

    for iteration in range(max_iter):
        # Generate neighbor solution with adaptive step size
        step_size = temp * 0.5
        neighbor_points = current_points + np.random.normal(0, step_size, current_points.shape)
        neighbor_points = np.clip(neighbor_points, 0, 1)

        # Evaluate neighbor
        neighbor_energy = objective_function(neighbor_points)

        # Accept or reject based on Metropolis criterion
        if neighbor_energy < current_energy or np.random.rand() < np.exp((current_energy - neighbor_energy) / temp):
            current_points = neighbor_points
            current_energy = neighbor_energy

            # Update best solution
            if current_energy < best_energy:
                best_points = current_points.copy()
                best_energy = current_energy

        # Cool down
        temp = max(min_temp, temp * cooling_rate)

        # Early stopping if improvement is minimal
        if iteration > 100 and abs(current_energy - best_energy) < 1e-8:
            break

    return best_points.reshape((16, 2))

def multi_start_optimization(initial_points_list, max_iterations=500):
    """Run optimization from multiple starting points and return the best result."""
    best_points = None
    best_ratio = 0.0

    for i, initial_points in enumerate(initial_points_list):
        try:
            # Run optimization from this starting point
            final_points = optimize_point_placement(initial_points, max_iterations)

            # Calculate ratio
            dist_matrix = compute_distance_matrix(final_points)
            ratio = calculate_min_max_ratio(dist_matrix)

            if ratio > best_ratio:
                best_ratio = ratio
                best_points = final_points.copy()

        except Exception:
            continue  # Skip this optimization attempt if it fails

    return best_points if best_points is not None else initial_points_list[0]

def optimize_point_placement(initial_points, max_iterations=500):
    """Optimize point placement using hybrid approach."""
    def objective(x_flat):
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)

        # Ensure points are within bounds
        points = np.clip(points, 0, 1)

        try:
            dist_matrix = compute_distance_matrix(points)
            ratio = calculate_min_max_ratio(dist_matrix)

            # Return negative ratio since we want to maximize
            # We also penalize configurations that are too close to boundary
            penalty = 0

            # Additional penalty for points very close to boundaries
            boundary_penalty = 0
            if np.any(points < 0.02) or np.any(points > 0.98):
                boundary_penalty = -0.02

            # Penalty for extreme ratios (to avoid numerical issues)
            if ratio < 1e-8:
                penalty = -1.0

            return -(ratio + boundary_penalty + penalty)
        except Exception:
            return 1e6  # Return large value for invalid configurations

    # Flatten initial points for optimization
    x0 = initial_points.flatten()

    # Define bounds for each coordinate (0 to 1) but slightly inside to avoid boundary issues
    bounds = [(0.01, 0.99) for _ in range(len(x0))]

    # First try with L-BFGS-B
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iterations//2, 'ftol': 1e-8, 'gtol': 1e-5},
            callback=None
        )

        # Refine with simulated annealing
        refined_points = result.x.reshape(-1, 2)
        refined_points = np.clip(refined_points, 0, 1)

        # Run simulated annealing refinement
        final_points = optimize_with_simulated_annealing(refined_points, max_iter=max_iterations//2)
        final_points = np.clip(final_points, 0, 1)

        return final_points
    except Exception:
        # Fallback to direct simulated annealing
        return optimize_with_simulated_annealing(initial_points, max_iter=max_iterations)

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    # Create multiple diverse initial configurations
    initial_configs = []

    # 1. Multiple hexagonal lattice variations with different perturbations
    for seed_val in [42, 123, 456, 789]:
        np.random.seed(seed_val)
        initial_configs.append(initialize_points_hexagonal_lattice())

    # 2. Random initializations with different seeds
    for seed_val in [999, 888, 777]:
        np.random.seed(seed_val)
        initial_configs.append(initialize_points_random())

    # 3. Voronoi-inspired initialization
    initial_configs.append(initialize_points_voronoi_based())

    # Run multi-start optimization
    best_points = multi_start_optimization(initial_configs, max_iterations=500)

    return best_points

# EVOLVE-BLOCK-END