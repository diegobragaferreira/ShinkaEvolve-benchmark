# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

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

def sobol_points_sphere(n_points):
    """Generate points on sphere using 3D Sobol sequence"""
    try:
        # Try to import sobol sequence generator
        from sobol_seq import i4_sobol_generate

        # Generate Sobol points in [0,1]^3
        sobol_points = i4_sobol_generate(3, n_points)

        # Convert to sphere using spherical coordinates
        points = np.zeros((n_points, 3))

        # Use the Sobol points to create well-distributed points on sphere
        for i in range(n_points):
            # Map to sphere using similar approach as Fibonacci
            u = sobol_points[i, 0]  # Uniform random in [0,1]
            v = sobol_points[i, 1]  # Uniform random in [0,1]

            # Use these as parameters for spherical coordinates
            theta = 2 * np.pi * u  # azimuthal angle
            phi = np.arccos(2 * v - 1)  # polar angle

            # Convert to Cartesian
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

            points[i] = [x, y, z]

        return points

    except ImportError:
        # Fallback to fibonacci if sobol not available
        return fibonacci_spiral_sphere(n_points)

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

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    d = 3

    def objective(x_flat):
        points_reshaped = x_flat.reshape(n, d)
        # Ensure points are on unit sphere
        norms = np.linalg.norm(points_reshaped, axis=1, keepdims=True)
        normalized_points = points_reshaped / np.maximum(norms, 1e-12)
        return -min_max_dist_ratio(normalized_points)

    def constraint_sphere(x_flat):
        points_reshaped = x_flat.reshape(n, d)
        norms = np.linalg.norm(points_reshaped, axis=1)
        return norms - 1.0

    def adaptive_bounds(iteration, max_iterations):
        """Dynamically adjust bounds based on optimization progress"""
        # Start with wider bounds for exploration
        if iteration < max_iterations * 0.3:
            return [(-2.0, 2.0) for _ in range(n * d)]
        # Middle phase - moderate bounds
        elif iteration < max_iterations * 0.7:
            return [(-1.5, 1.5) for _ in range(n * d)]
        # Final phase - tight bounds for precision
        else:
            return [(-1.1, 1.1) for _ in range(n * d)]

    # Define constraints and bounds
    constraints = {'type': 'eq', 'fun': constraint_sphere}
    bounds = [(-2, 2) for _ in range(n * d)]

    # Try multiple initial configurations
    initial_configs = []

    # 1. Fibonacci spiral initialization
    fib_points = fibonacci_spiral_sphere(n)
    initial_configs.append(("fibonacci", fib_points))

    # 2. Icosahedron-based initialization
    ico_points = icosahedron_points(n)
    initial_configs.append(("icosahedron", ico_points))

    # 3. Sobol sequence initialization
    sobol_points = sobol_points_sphere(n)
    initial_configs.append(("sobol", sobol_points))

    # 4. Random uniform points on sphere
    np.random.seed(42)
    random_points = np.random.randn(n, d)
    norms = np.linalg.norm(random_points, axis=1, keepdims=True)
    random_points = random_points / np.maximum(norms, 1e-12)
    initial_configs.append(("random", random_points))

    best_ratio = -np.inf
    best_points = None

    # Try each initial configuration with multiple restarts
    for config_name, initial_points in initial_configs:
        # Add slight random noise to break symmetry
        np.random.seed(42)
        noisy_points = initial_points + np.random.normal(0, 0.02, (n, d))

        # Ensure all points are on unit sphere
        norms = np.linalg.norm(noisy_points, axis=1, keepdims=True)
        normalized_points = noisy_points / np.maximum(norms, 1e-12)

        # Flatten for optimization
        x0 = normalized_points.flatten()

        # Progressive optimization with adaptive parameters
        current_x = x0.copy()
        current_best_ratio = -np.inf
        current_best_points = None

        # Stage 1: Coarse optimization with relaxed tolerances
        try:
            result = minimize(
                objective,
                current_x,
                method='SLSQP',
                bounds=[(-2.0, 2.0) for _ in range(n * d)],
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-4, 'gtol': 1e-4},
                tol=1e-4
            )

            if result.success:
                current_x = result.x
                points = current_x.reshape(n, d)
                norms = np.linalg.norm(points, axis=1, keepdims=True)
                normalized = points / np.maximum(norms, 1e-12)
                ratio = min_max_dist_ratio(normalized)
                if ratio > current_best_ratio:
                    current_best_ratio = ratio
                    current_best_points = normalized.copy()
        except Exception:
            pass

        # Stage 2: Medium optimization with moderate tolerances
        try:
            result = minimize(
                objective,
                current_x,
                method='SLSQP',
                bounds=[(-1.5, 1.5) for _ in range(n * d)],
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-6, 'gtol': 1e-6},
                tol=1e-6
            )

            if result.success:
                current_x = result.x
                points = current_x.reshape(n, d)
                norms = np.linalg.norm(points, axis=1, keepdims=True)
                normalized = points / np.maximum(norms, 1e-12)
                ratio = min_max_dist_ratio(normalized)
                if ratio > current_best_ratio:
                    current_best_ratio = ratio
                    current_best_points = normalized.copy()
        except Exception:
            pass

        # Stage 3: Fine optimization with tight tolerances
        try:
            result = minimize(
                objective,
                current_x,
                method='L-BFGS-B',
                bounds=[(-1.1, 1.1) for _ in range(n * d)],
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10},
                tol=1e-10
            )

            if result.success:
                points = result.x.reshape(n, d)
                norms = np.linalg.norm(points, axis=1, keepdims=True)
                normalized = points / np.maximum(norms, 1e-12)
                ratio = min_max_dist_ratio(normalized)
                if ratio > current_best_ratio:
                    current_best_ratio = ratio
                    current_best_points = normalized.copy()
        except Exception:
            pass

        # Update global best
        if current_best_points is not None and current_best_ratio > best_ratio:
            best_ratio = current_best_ratio
            best_points = current_best_points.copy()

    # If no good solution was found, return a random configuration
    if best_points is None:
        np.random.seed(42)
        random_points = np.random.randn(n, d)
        norms = np.linalg.norm(random_points, axis=1, keepdims=True)
        best_points = random_points / np.maximum(norms, 1e-12)

    return best_points

# EVOLVE-BLOCK-END