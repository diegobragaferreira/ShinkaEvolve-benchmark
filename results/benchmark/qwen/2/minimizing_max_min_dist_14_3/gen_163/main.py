# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform


def fibonacci_sphere(n):
    """Generate n points on a sphere using Fibonacci spiral method"""
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle

    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)


def compute_min_max_ratio(points):
    """Compute the ratio of minimum to maximum pairwise distances"""
    # Compute pairwise distances
    distances = pdist(points)

    # Get minimum and maximum distances
    d_min = np.min(distances)
    d_max = np.max(distances)

    # Return ratio (avoid division by zero)
    if d_max == 0:
        return 0
    return d_min / d_max


def perturb_points(points, temperature, strategy='random', perturbation_scale=0.01):
    """Perturb points with temperature-dependent magnitude using different strategies"""
    n_points = points.shape[0]

    if strategy == 'density_based':
        # Choose point based on local density analysis
        distances = pdist(points)
        distance_matrix = squareform(distances)

        # Calculate average distance for each point to its neighbors
        # Exclude self-distances by setting them to infinity
        masked_distances = distance_matrix + np.eye(n_points) * np.inf
        avg_distances = np.mean(masked_distances, axis=1)

        # Points with lower average distances are more crowded
        # We prefer perturbing less crowded points to encourage expansion
        # But also occasionally perturb crowded points for contraction
        weights = 1.0 / (avg_distances + 1e-8)  # Avoid division by zero
        weights = weights / np.sum(weights)  # Normalize probabilities

        # Sample point index with probability proportional to weights
        idx = np.random.choice(n_points, p=weights)

    elif strategy == 'farthest':
        # Choose point that is farthest from its nearest neighbor to perturb
        distances = pdist(points)
        distance_matrix = squareform(distances)
        # Find the point with minimum distance to its nearest neighbor
        min_distances = np.min(distance_matrix + np.eye(n_points) * np.inf, axis=1)
        idx = np.argmax(min_distances)  # Perturb the farthest from nearest neighbor
    else:  # random strategy
        # Select random point to perturb
        idx = np.random.randint(0, n_points)

    # Generate perturbation vector with adaptive scaling
    # Base on current temperature and point's local characteristics
    if strategy == 'density_based':
        # For density-based, we might want to adjust magnitude based on how crowded the point is
        # More crowded points get smaller perturbations to avoid destabilizing
        local_crowding = 1.0 / (np.min(masked_distances[idx]) + 1e-8)
        adaptive_scale = perturbation_scale * temperature * (0.5 + 0.5 * local_crowding)
    else:
        adaptive_scale = perturbation_scale * temperature

    perturbation = np.random.randn(3) * adaptive_scale

    # Apply perturbation
    new_points = points.copy()
    new_points[idx] += perturbation

    # Project back to unit sphere
    norm = np.linalg.norm(new_points[idx])
    if norm > 0:
        new_points[idx] = new_points[idx] / norm

    return new_points


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """

    np.random.seed(42)

    n = 14
    d = 3

    # Multi-start approach - try multiple initializations and keep the best
    best_overall_ratio = 0.0
    best_overall_points = None

    # Try different initialization strategies
    initial_configurations = []

    # Strategy 1: Fibonacci sphere
    points1 = fibonacci_sphere(n)
    initial_configurations.append(('fibonacci', points1))

    # Strategy 2: Random with normalization
    points2 = np.random.uniform(-1, 1, (n, 3))
    for i in range(len(points2)):
        norm = np.linalg.norm(points2[i])
        if norm > 0:
            points2[i] = points2[i] / norm
    initial_configurations.append(('random', points2))

    # Strategy 3: Modified icosahedron-based (if needed)
    # We'll use the fibonacci approach for now, but this gives us flexibility

    for init_name, initial_points in initial_configurations:
        # Initialize points using Fibonacci sphere distribution for better spread
        points = initial_points.copy()

        # Normalize to unit sphere (already done in our strategies)
        norms = np.linalg.norm(points, axis=1)
        if np.max(norms) > 0:
            points = points / np.max(norms)  # Scale to unit sphere

        # Enhanced Simulated Annealing parameters
        initial_temperature = 0.1
        final_temperature = 1e-6
        max_iterations = 100000
        stagnation_threshold = 5000  # Number of iterations without improvement before cooling

        # Track best solution for this initialization
        current_points = points.copy()
        best_points = points.copy()
        best_ratio = compute_min_max_ratio(current_points)

        # Checkpoint variables
        best_checkpoint = best_points.copy()
        best_checkpoint_ratio = best_ratio
        last_improvement_iteration = 0

        temperature = initial_temperature
        iteration = 0

        # Optimization loop
        while iteration < max_iterations and temperature > final_temperature:
            # Adaptive perturbation strategy - more sophisticated selection
            # In first phase: mostly random, then shift towards density-based
            if iteration < max_iterations // 4:
                # Early phase: random perturbations
                strategy = 'random'
            elif iteration < max_iterations // 2:
                # Mid phase: mixed strategy
                strategy = np.random.choice(['random', 'density_based'], p=[0.7, 0.3])
            else:
                # Late phase: more density-based for local refinement
                strategy = np.random.choice(['density_based', 'farthest'], p=[0.6, 0.4])

            # Perturb points with adaptive strategy
            new_points = perturb_points(current_points, temperature, strategy=strategy)

            # Compute new ratio
            new_ratio = compute_min_max_ratio(new_points)

            # Accept or reject the new configuration using Metropolis criterion
            if new_ratio > best_ratio or np.random.rand() < np.exp((new_ratio - best_ratio) / temperature):
                current_points = new_points

                # Update best solution if improved
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = new_points.copy()
                    last_improvement_iteration = iteration
                    # Update checkpoint
                    if new_ratio > best_checkpoint_ratio:
                        best_checkpoint = new_points.copy()
                        best_checkpoint_ratio = new_ratio

            # Adaptive cooling schedule with more dynamic behavior
            if iteration % 1000 == 0:
                # More aggressive cooling when stuck
                if iteration - last_improvement_iteration > stagnation_threshold:
                    # If we haven't improved for a while, cool faster
                    # But maintain some minimum cooling rate
                    cooling_factor = max(0.9, 0.98 + 0.01 * np.random.random())
                    temperature *= cooling_factor
                else:
                    # Normal cooling, but adapt based on progress
                    if iteration < max_iterations // 2:
                        # Early stage - more aggressive cooling
                        temperature *= 0.998
                    else:
                        # Later stage - slower cooling for fine-tuning
                        temperature *= 0.9995
            else:
                # Standard cooling for non-checkpoint iterations
                temperature *= 0.9995

            iteration += 1

        # Update overall best if this initialization was better
        if best_checkpoint_ratio > best_overall_ratio:
            best_overall_ratio = best_checkpoint_ratio
            best_overall_points = best_checkpoint.copy()

    return best_overall_points


# EVOLVE-BLOCK-END