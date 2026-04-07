# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Add Sobol sequence import
try:
    from sobol_seq import i4_sobol_generate
except ImportError:
    # Fallback to basic implementation if sobol_seq not available
    def i4_sobol_generate(dim, n):
        # Simple quasi-random sequence generator
        np.random.seed(42)
        return np.random.rand(n, dim)

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def objective(x):
        # Reshape x back to points array
        points = x.reshape(-1, 3)
        # Compute pairwise distances
        distances = cdist(points, points)
        # Set diagonal to large value to ignore self-distances
        np.fill_diagonal(distances, np.inf)
        # Minimize negative of minimum distance (maximize minimum distance)
        return -np.min(distances)

    def constraint_sphere(x):
        # Ensure points stay within unit sphere
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return 1 - norms  # Should be >= 0

    def constraint_max_distance(x):
        # Ensure maximum distance doesn't exceed some reasonable bound
        points = x.reshape(-1, 3)
        distances = cdist(points, points)
        np.fill_diagonal(distances, 0)
        max_dist = np.max(distances)
        return 2 - max_dist  # Should be >= 0 (allowing up to diameter 2)

    def normalize_points(points):
        """Normalize points to lie on unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return points / norms

    def generate_fibonacci_points(n):
        """Generate points using Fibonacci spiral on sphere"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            theta = np.arccos(1 - 2*(i/(n-1)))
            phi = i * 2 * np.pi / golden_ratio
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            points.append([x, y, z])
        return np.array(points)

    def generate_sobol_points(n):
        """Generate points using Sobol sequence on unit sphere"""
        # Generate Sobol points in [0,1]^3
        sobol_points = i4_sobol_generate(3, n)

        # Map to unit sphere using Fibonacci-like approach
        points = []
        for i in range(n):
            # Convert to spherical coordinates
            x, y, z = sobol_points[i]
            # Scale and center to [0,1] range
            x = x * 2 - 1
            y = y * 2 - 1
            z = z * 2 - 1

            # Normalize to unit sphere
            norm = np.sqrt(x**2 + y**2 + z**2)
            if norm > 0:
                x, y, z = x/norm, y/norm, z/norm
            points.append([x, y, z])
        return np.array(points)

    def generate_perturbed_fibonacci_points(n, perturbation_strength=0.05):
        """Generate fibonacci points with small random perturbations"""
        base_points = generate_fibonacci_points(n)
        perturbations = np.random.normal(0, perturbation_strength, (n, 3))
        perturbed_points = base_points + perturbations
        # Normalize back to unit sphere
        perturbed_points = perturbed_points / np.linalg.norm(perturbed_points, axis=1, keepdims=True)
        return perturbed_points

    # Use a more diverse set of starting configurations
    configs = []

    # Configuration 1: Pure Fibonacci spiral on sphere
    configs.append(generate_fibonacci_points(14))

    # Configuration 2: Sobol sequence points on sphere
    configs.append(generate_sobol_points(14))

    # Configuration 3: Perturbed Fibonacci with small perturbation
    np.random.seed(100)
    configs.append(generate_perturbed_fibonacci_points(14, 0.02))

    # Configuration 4: Perturbed Fibonacci with medium perturbation
    np.random.seed(200)
    configs.append(generate_perturbed_fibonacci_points(14, 0.05))

    # Configuration 5: Another deterministic Fibonacci
    configs.append(generate_fibonacci_points(14))

    # Configuration 6: Small random perturbation of Fibonacci
    np.random.seed(300)
    fib_points = generate_fibonacci_points(14)
    perturbations = np.random.normal(0, 0.01, fib_points.shape)
    random_perturbed = fib_points + perturbations
    configs.append(normalize_points(random_perturbed))

    # Configuration 7: Alternative Sobol points with different seed
    np.random.seed(500)
    configs.append(generate_sobol_points(14))

    # Main optimization loop with multiple restarts
    best_final_points = None
    best_ratio = 0

    # Try local optimization from each configuration
    for i, initial_config in enumerate(configs):
        try:
            # Local optimization around initial point
            x0 = initial_config.flatten()
            cons = [
                {'type': 'ineq', 'fun': constraint_sphere},
                {'type': 'ineq', 'fun': constraint_max_distance}
            ]

            # Use SLSQP for local optimization with high precision
            result = minimize(objective, x0, method='SLSQP', constraints=cons,
                            options={'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 1000})

            optimized_points = result.x.reshape(-1, 3)
            optimized_points = normalize_points(optimized_points)

            # Evaluate this solution
            distances = cdist(optimized_points, optimized_points)
            np.fill_diagonal(distances, np.inf)
            min_dist = np.min(distances)
            max_dist = np.max(distances)

            if max_dist > 0:
                ratio = min_dist / max_dist
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_final_points = optimized_points.copy()

        except Exception as e:
            continue

    # If no solution found, fall back to Fibonacci points
    if best_final_points is None:
        return generate_fibonacci_points(14)

    return best_final_points

# EVOLVE-BLOCK-END