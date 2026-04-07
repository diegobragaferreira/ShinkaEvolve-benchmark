# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans

def icosahedron_points(n=14):
    """Generate points using icosahedron-based construction"""
    # Vertices of a regular icosahedron
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])

    # Normalize to unit sphere
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

    # If we need more than 12 points, distribute additional points
    if n <= 12:
        # Just return subset of vertices
        return vertices[:n]
    else:
        # For 14 points, we'll start with icosahedron vertices and add two more
        points = vertices.copy()

        # Add two more points that are well-distributed
        # Add points along major axes
        points = np.vstack([points, [[0, 0, 1], [0, 0, -1]]])

        # Apply slight random perturbation to ensure good distribution
        np.random.seed(42)
        points += np.random.normal(0, 0.05, (points.shape[0], 3))

        # Normalize again to maintain unit sphere
        norms = np.linalg.norm(points, axis=1)
        points = points / np.maximum(norms[:, np.newaxis], 1e-12)

        return points[:n]

def fibonacci_spiral_sphere(n_points):
    """Generate points on a sphere using Fibonacci spiral method."""
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle

    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

def min_max_dist_ratio(points):
    """Calculate the ratio of minimum to maximum distance."""
    if len(points) < 2:
        return 0.0
    distances = pdist(points)
    if len(distances) == 0:
        return 0.0
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    if max_dist < 1e-12:
        return 0.0
    return min_dist / max_dist

def normalize_points(points):
    """Normalize points to unit sphere with improved numerical stability."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Use maximum to avoid division by very small numbers
    normalized = points / np.maximum(norms, 1e-12)
    return normalized

def smart_optimization(initial_points, max_iter=800):
    """Smart optimization pipeline with multiple stages"""
    n, d = initial_points.shape

    def objective(x_flat):
        points_reshaped = x_flat.reshape(n, d)
        # Ensure points are on unit sphere
        normalized_points = normalize_points(points_reshaped)
        return -min_max_dist_ratio(normalized_points)

    def constraint_sphere(x_flat):
        points_reshaped = x_flat.reshape(n, d)
        norms = np.linalg.norm(points_reshaped, axis=1)
        return norms - 1.0

    constraints = {'type': 'eq', 'fun': constraint_sphere}
    bounds = [(-2, 2) for _ in range(n * d)]

    # Stage 1: Quick coarse optimization with L-BFGS-B
    try:
        result1 = minimize(
            objective,
            initial_points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter//3, 'ftol': 1e-8, 'gtol': 1e-8}
        )

        if result1.success:
            # Stage 2: Medium refinement with Trust-Constr
            try:
                result2 = minimize(
                    objective,
                    result1.x,
                    method='trust-constr',
                    constraints=constraints,
                    options={'maxiter': max_iter//3, 'xtol': 1e-10, 'gtol': 1e-10}
                )

                if result2.success:
                    optimized_points = result2.x.reshape(n, d)
                    return normalize_points(optimized_points)
            except Exception:
                pass

            # Fallback to L-BFGS-B result
            optimized_points = result1.x.reshape(n, d)
            return normalize_points(optimized_points)
    except Exception:
        pass

    # Stage 3: Final fine-tuning with SLSQP if needed
    try:
        result3 = minimize(
            objective,
            initial_points.flatten(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter//2, 'ftol': 1e-9, 'gtol': 1e-9}
        )

        if result3.success:
            optimized_points = result3.x.reshape(n, d)
            return normalize_points(optimized_points)
    except Exception:
        pass

    # Return original if all optimizations fail
    return initial_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    d = 3

    # Prioritize initialization strategies based on performance expectation
    initial_strategies = [
        ("icosahedron", icosahedron_points(n)),
        ("fibonacci", fibonacci_spiral_sphere(n)),
        ("random_clustered", None)  # Will generate dynamically
    ]

    best_ratio = -np.inf
    best_points = None

    # Try multiple initialization strategies with different restarts
    for strategy_name, base_points in initial_strategies:
        # Handle special case for random clustered points
        if strategy_name == "random_clustered":
            # Initialize with k-means clustering approach
            np.random.seed(42)
            points = np.random.rand(n, d) * 2 - 1  # [-1, 1]

            # Normalize to unit sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / np.maximum(norms, 1e-12)

            # Apply k-means clustering to identify natural groupings
            # and adjust points to spread them out
            if n > 1:
                kmeans = KMeans(n_clusters=min(n, 8), random_state=42, n_init=10)
                labels = kmeans.fit_predict(points)

                # Adjust points to increase separation
                centers = kmeans.cluster_centers_
                for i in range(n):
                    if n > 1:
                        # Move points away from cluster centers
                        center_idx = labels[i]
                        direction = points[i] - centers[center_idx]
                        distance_from_center = np.linalg.norm(direction)
                        if distance_from_center > 1e-12:
                            # Push points outward slightly
                            push_strength = 0.02 / (distance_from_center + 1e-12)
                            points[i] += direction * push_strength

            base_points = points

        # Multiple restarts for each strategy
        for restart in range(2):  # Reduced restarts for efficiency
            # Set deterministic seed for reproducibility
            np.random.seed(42 + restart * 100 + hash(strategy_name) % 1000)

            # Add controlled noise to break symmetries
            noise_magnitude = 0.05 if strategy_name == "icosahedron" else 0.03
            noisy_points = base_points + np.random.normal(0, noise_magnitude, (n, d))

            # Normalize to unit sphere
            normalized_points = normalize_points(noisy_points)

            # Smart optimization pipeline
            optimized_points = smart_optimization(normalized_points, max_iter=600)

            # Evaluate result
            ratio = min_max_dist_ratio(optimized_points)

            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()

    # If no good solution found, try random initialization with smart optimization
    if best_points is None:
        np.random.seed(42)
        random_points = np.random.randn(n, d)
        normalized_points = normalize_points(random_points)
        optimized_points = smart_optimization(normalized_points, max_iter=500)
        ratio = min_max_dist_ratio(optimized_points)

        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()

    # Final fallback to random points
    if best_points is None:
        np.random.seed(42)
        points = np.random.rand(n, d) * 2 - 1
        best_points = normalize_points(points)

    return best_points

# EVOLVE-BLOCK-END