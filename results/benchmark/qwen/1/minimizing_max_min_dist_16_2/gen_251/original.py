# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    # Set seed for reproducibility
    np.random.seed(42)

    def compute_energy_and_ratio(points):
        # Compute pairwise distances efficiently
        distances = squareform(pdist(points))

        # Mask diagonal elements (distance to self is 0)
        np.fill_diagonal(distances, np.inf)

        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            ratio = 0
        else:
            ratio = min_dist / max_dist

        # Energy is negative ratio (we want to maximize ratio, so minimize negative ratio)
        # Add penalty for points outside bounds with epsilon padding
        penalty = 0
        epsilon = 1e-8
        for pt in points:
            if pt[0] < 0+epsilon or pt[0] > 1-epsilon or pt[1] < 0+epsilon or pt[1] > 1-epsilon:
                penalty += 1000

        return -ratio + penalty, ratio

    def initialize_hexagonal_grid():
        """Initialize points using a better hexagonal grid pattern."""
        n = 16
        points = np.zeros((n, 2))

        # Create hexagonal grid pattern with better distribution
        rows = 4
        cols = 4
        spacing = 0.25

        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx < n:
                    # Offset every other row for hexagonal packing
                    x = col * spacing + (row % 2) * spacing * 0.5
                    y = row * spacing * math.sqrt(3) / 2
                    points[idx] = [x, y]
                    idx += 1

        # Adjust points to fit within [0.1,0.9]x[0.1,0.9] with some randomness
        points[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min()) * 0.8 + 0.1
        points[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min()) * 0.8 + 0.1

        # Add small random perturbation
        points += np.random.normal(0, 0.01, points.shape)

        # Clamp to bounds
        points = np.clip(points, 0.01, 0.99)

        return points

    def initialize_spiral_pattern():
        """Initialize points using a spiral pattern."""
        n = 16
        points = np.zeros((n, 2))

        # Create spiral pattern
        angles = np.linspace(0, 4*np.pi, n)
        radii = np.linspace(0.1, 0.4, n)

        for i in range(n):
            points[i, 0] = 0.5 + radii[i] * np.cos(angles[i])
            points[i, 1] = 0.5 + radii[i] * np.sin(angles[i])

        return points

    def initialize_random():
        """Initialize points using random distribution."""
        return np.random.uniform(0.1, 0.9, (16, 2))

    def initialize_golden_spiral():
        """Initialize points using golden spiral pattern."""
        n = 16
        points = np.zeros((n, 2))

        # Golden spiral with logarithmic growth
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        for i in range(n):
            angle = i * 2 * np.pi / (phi * phi)
            radius = i * 0.3 / n
            points[i, 0] = 0.5 + radius * np.cos(angle)
            points[i, 1] = 0.5 + radius * np.sin(angle)

        return points

    def initialize_circle_packing():
        """Initialize points in a circular arrangement."""
        n = 16
        points = np.zeros((n, 2))

        # Arrange points evenly on a circle with some randomization
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radii = 0.35 + np.random.normal(0, 0.05, n)  # Slight variations in radii

        for i in range(n):
            points[i, 0] = 0.5 + radii[i] * np.cos(angles[i])
            points[i, 1] = 0.5 + radii[i] * np.sin(angles[i])

        return points

    def initialize_regular_grid():
        """Initialize points in a regular 4x4 grid."""
        n = 16
        points = np.zeros((n, 2))

        # Create regular grid
        grid_size = 4
        spacing = 1.0 / (grid_size - 1)

        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx < n:
                    points[idx, 0] = j * spacing
                    points[idx, 1] = i * spacing
                    idx += 1

        # Add some randomness
        points += np.random.normal(0, 0.02, points.shape)
        points = np.clip(points, 0.01, 0.99)

        return points

    # Try multiple initialization strategies
    initial_configs = [
        initialize_hexagonal_grid(),
        initialize_spiral_pattern(),
        initialize_random(),
        initialize_golden_spiral(),
        initialize_circle_packing(),
        initialize_regular_grid()
    ]

    best_points = None
    best_ratio = -np.inf

    # Run optimization from each initialization
    for init_config in initial_configs:
        points = init_config.copy()

        # Simulated Annealing parameters with adaptive cooling
        temp = 1.0
        min_temp = 1e-8
        max_iter = 50000

        # Track convergence for adaptive cooling
        current_energy, current_ratio = compute_energy_and_ratio(points)
        best_points_local = points.copy()
        best_ratio_local = current_ratio

        # Track recent improvements
        recent_improvements = []
        last_improvement = 0
        patience = 2000

        # Main optimization loop with adaptive cooling
        for iteration in range(max_iter):
            # Generate neighbor solution (random perturbation)
            neighbor_points = points.copy()
            # Pick a random point to move
            move_idx = np.random.randint(0, 16)

            # Adaptive displacement magnitude based on temperature
            displacement_magnitude = max(0.0001, 0.01 * temp)
            displacement = np.random.normal(0, displacement_magnitude, 2)
            neighbor_points[move_idx] += displacement

            # Apply boundary constraints with epsilon padding
            epsilon = 1e-8
            neighbor_points[move_idx, 0] = np.clip(neighbor_points[move_idx, 0], 0+epsilon, 1-epsilon)
            neighbor_points[move_idx, 1] = np.clip(neighbor_points[move_idx, 1], 0+epsilon, 1-epsilon)

            # Compute energy of neighbor
            neighbor_energy, neighbor_ratio = compute_energy_and_ratio(neighbor_points)

            # Accept or reject move
            if neighbor_energy < current_energy:
                # Always accept better solutions
                points = neighbor_points
                current_energy = neighbor_energy
                current_ratio = neighbor_ratio
            else:
                # Accept worse solutions with probability based on temperature
                delta = neighbor_energy - current_energy
                if np.random.rand() < np.exp(-delta / temp):
                    points = neighbor_points
                    current_energy = neighbor_energy
                    current_ratio = neighbor_ratio

            # Update best solution
            if current_ratio > best_ratio_local:
                best_ratio_local = current_ratio
                best_points_local = points.copy()
                last_improvement = iteration
                recent_improvements.append(iteration)
                if len(recent_improvements) > 10:
                    recent_improvements.pop(0)

            # Adaptive cooling schedule
            # Start with fast cooling, slow down as we get closer to optimum
            if iteration < 10000:
                cooling_rate = 0.9995
            elif iteration < 25000:
                cooling_rate = 0.9998
            else:
                cooling_rate = 0.9999

            temp *= cooling_rate

            # More aggressive cooling if we haven't improved recently
            if iteration - last_improvement > patience // 2 and temp > min_temp:
                temp *= 0.98

            # Early stopping based on patience and convergence
            if iteration - last_improvement > patience:
                # Check if recent improvements have stalled
                if len(recent_improvements) >= 5:
                    if recent_improvements[-1] - recent_improvements[0] < 1000:
                        break
                else:
                    break

            # Stop if temperature gets too low
            if temp < min_temp:
                break

        # Update global best if this run was better
        if best_ratio_local > best_ratio:
            best_ratio = best_ratio_local
            best_points = best_points_local.copy()

    return best_points

# EVOLVE-BLOCK-END