# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi
import time
from sklearn.cluster import KMeans

def torus_points(n_points, R=1.0, r=0.5):
    """Generate points on a torus surface"""
    points = []
    for i in range(n_points):
        # Parameterize torus using angles
        theta = 2 * np.pi * i / n_points  # Major circle angle
        phi = 2 * np.pi * np.random.random()  # Minor circle angle

        # Torus coordinates
        x = (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)

        points.append([x, y, z])

    return np.array(points)

def torus_to_sphere_mapping(torus_points):
    """Map torus points to sphere while preserving relative distances as much as possible"""
    # Normalize to unit sphere
    norms = np.linalg.norm(torus_points, axis=1, keepdims=True)
    sphere_points = torus_points / np.maximum(norms, 1e-12)

    # Apply slight adjustment to spread points more evenly
    # This helps reduce clustering that might occur from direct mapping
    adjustment_factor = 0.1
    for i in range(len(sphere_points)):
        # Perturb based on normalized coordinates
        perturbation = adjustment_factor * sphere_points[i] * np.random.random()
        sphere_points[i] += perturbation

    # Renormalize
    norms = np.linalg.norm(sphere_points, axis=1, keepdims=True)
    sphere_points = sphere_points / np.maximum(norms, 1e-12)

    return sphere_points

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

def initialize_with_clustering(n_points):
    """Initialize points using k-means clustering approach for even distribution"""
    # Start with random points
    np.random.seed(42)
    points = np.random.rand(n_points, 3) * 2 - 1  # [-1, 1]

    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / np.maximum(norms, 1e-12)

    # Apply k-means clustering to identify natural groupings
    # and adjust points to spread them out
    if n_points > 1:
        kmeans = KMeans(n_clusters=min(n_points, 8), random_state=42, n_init=10)
        labels = kmeans.fit_predict(points)

        # Adjust points to increase separation
        centers = kmeans.cluster_centers_
        for i in range(n_points):
            if n_points > 1:
                # Move points away from cluster centers
                center_idx = labels[i]
                direction = points[i] - centers[center_idx]
                distance_from_center = np.linalg.norm(direction)
                if distance_from_center > 1e-12:
                    # Push points outward slightly
                    push_strength = 0.02 / (distance_from_center + 1e-12)
                    points[i] += direction * push_strength

    return points

def adaptive_constraint_tightening(iteration, max_iter):
    """Adaptively tighten constraints during optimization"""
    # Start with looser constraints and tighten over time
    base_radius = 1.0
    tightness = 0.1 + 0.9 * (iteration / max_iter)  # Gradually tighten
    return base_radius * tightness

def hybrid_optimization(points, max_iter=1000):
    """Perform hybrid optimization combining multiple strategies"""

    def objective(x_flat):
        points_reshaped = x_flat.reshape(-1, 3)

        # Ensure points are on unit sphere
        norms = np.linalg.norm(points_reshaped, axis=1, keepdims=True)
        normalized_points = points_reshaped / np.maximum(norms, 1e-12)

        # Calculate ratio
        ratio = min_max_dist_ratio(normalized_points)

        # Add penalty for non-uniformity (high variation in distances)
        distances = pdist(normalized_points)
        if len(distances) > 0:
            dist_std = np.std(distances)
            dist_mean = np.mean(distances)
            if dist_mean > 1e-12:
                uniformity_penalty = dist_std / dist_mean
                # We want to maximize ratio AND minimize uniformity penalty
                return -(ratio - 0.1 * uniformity_penalty)
        return -ratio if ratio > 0 else 1e10

    def constraint_sphere(x_flat):
        points_reshaped = x_flat.reshape(-1, 3)
        norms = np.linalg.norm(points_reshaped, axis=1)
        return norms - 1.0

    constraints = {'type': 'eq', 'fun': constraint_sphere}
    bounds = [(-2, 2) for _ in range(len(points) * 3)]

    # Multi-stage optimization with adaptive tightening
    best_points = points.copy()
    best_ratio = min_max_dist_ratio(points)

    # Stage 1: Coarse optimization with relaxed constraints
    try:
        result = minimize(
            objective,
            points.flatten(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter//3, 'ftol': 1e-6, 'gtol': 1e-6}
        )
        if result.success:
            refined_points = result.x.reshape(-1, 3)
            norms = np.linalg.norm(refined_points, axis=1, keepdims=True)
            refined_points = refined_points / np.maximum(norms, 1e-12)

            ratio = min_max_dist_ratio(refined_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined_points.copy()
    except Exception:
        pass

    # Stage 2: Fine optimization with stricter criteria
    try:
        result = minimize(
            objective,
            best_points.flatten(),
            method='trust-constr',
            constraints=constraints,
            options={'maxiter': max_iter//3, 'xtol': 1e-9, 'gtol': 1e-9}
        )
        if result.success:
            refined_points = result.x.reshape(-1, 3)
            norms = np.linalg.norm(refined_points, axis=1, keepdims=True)
            refined_points = refined_points / np.maximum(norms, 1e-12)

            ratio = min_max_dist_ratio(refined_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined_points.copy()
    except Exception:
        pass

    # Stage 3: Local refinement with L-BFGS-B
    try:
        result = minimize(
            objective,
            best_points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter//3, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        if result.success:
            refined_points = result.x.reshape(-1, 3)
            norms = np.linalg.norm(refined_points, axis=1, keepdims=True)
            refined_points = refined_points / np.maximum(norms, 1e-12)

            ratio = min_max_dist_ratio(refined_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined_points.copy()
    except Exception:
        pass

    return best_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses torus-based initialization and hybrid optimization approach.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    d = 3

    # Multi-strategy initialization
    initial_strategies = []

    # Strategy 1: Torus-based initialization
    torus_points_arr = torus_points(n)
    torus_mapped = torus_to_sphere_mapping(torus_points_arr)
    initial_strategies.append(("torus", torus_mapped))

    # Strategy 2: Clustering-based initialization
    clustered_points = initialize_with_clustering(n)
    initial_strategies.append(("clustering", clustered_points))

    # Strategy 3: Fibonacci-like initialization with perturbations
    golden_angle = np.pi * (3 - np.sqrt(5))
    fib_points = np.zeros((n, 3))
    for i in range(n):
        y = 1 - (i / max(1, n - 1)) * 2
        radius = np.sqrt(max(0, 1 - y * y))
        theta = golden_angle * i + np.sin(i * 0.3) * 0.1
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        fib_points[i] = [x, y, z]

    # Normalize
    norms = np.linalg.norm(fib_points, axis=1, keepdims=True)
    fib_points = fib_points / np.maximum(norms, 1e-12)
    initial_strategies.append(("fibonacci_perturbed", fib_points))

    best_ratio = -np.inf
    best_points = None

    # Try each initialization strategy
    for strategy_name, initial_points in initial_strategies:
        # Add slight random perturbation to break symmetry
        np.random.seed(42)
        noisy_points = initial_points + np.random.normal(0, 0.01, (n, d))

        # Ensure all points are on unit sphere
        norms = np.linalg.norm(noisy_points, axis=1, keepdims=True)
        normalized_points = noisy_points / np.maximum(norms, 1e-12)

        # Apply hybrid optimization
        optimized_points = hybrid_optimization(normalized_points, max_iter=800)

        # Evaluate the result
        ratio = min_max_dist_ratio(optimized_points)

        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()

    # If no good solution was found, return a fallback
    if best_points is None:
        np.random.seed(42)
        points = np.random.rand(n, d) * 2 - 1
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        best_points = points / np.maximum(norms, 1e-12)

    return best_points

# EVOLVE-BLOCK-END