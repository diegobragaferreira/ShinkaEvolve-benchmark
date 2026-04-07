# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import SphericalVoronoi
import math

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def distance_ratio(points_flat):
        """Calculate the ratio of minimum to maximum distance"""
        points = points_flat.reshape(-1, 3)
        distances = squareform(pdist(points))
        # Set diagonal to large value so it doesn't affect min/max
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return min_dist / max_dist

    def objective_function(points_flat):
        """Minimize negative of distance ratio (since we want to maximize)"""
        return -distance_ratio(points_flat)

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

    def enhanced_hybrid_initialization(n):
        """
        Enhanced hybrid initialization combining icosahedron, Fibonacci, and Sobol-based approaches.
        Provides superior starting configurations compared to single-method approaches.
        """
        # Start with icosahedron vertices for better symmetry
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        vertices = [
            (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
            (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
            (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1)
        ]

        # Convert to numpy array and normalize
        points = np.array(vertices, dtype=float)
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]

        # Add extra points using enhanced Fibonacci distribution
        remaining = n - len(points)
        if remaining > 0:
            # Use Fibonacci-like distribution with better spread properties
            for i in range(remaining):
                # Use more sophisticated Fibonacci distribution with golden ratio multiples
                theta = math.acos(1 - 2 * (i / (remaining - 1)))
                phi_coord = (i * 2.414213562) % (2 * math.pi)  # Golden ratio multiple

                x = math.sin(theta) * math.cos(phi_coord)
                y = math.sin(theta) * math.sin(phi_coord)
                z = math.cos(theta)
                points = np.vstack([points, [x, y, z]])

        # Apply multiple jittering strategies to break symmetry effectively
        np.random.seed(42)
        # Primary jitter with moderate magnitude
        noise1 = np.random.normal(0, 0.015, points.shape)
        points += noise1

        # Secondary jitter with smaller magnitude
        noise2 = np.random.normal(0, 0.005, points.shape)
        points += noise2

        # Normalize again to maintain unit sphere
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]

        return points

    def progressive_optimization(x0, maxiter=1000):
        """
        Perform progressive optimization with varying constraints and tolerances
        """
        # Define constraints for normalization (points should be on unit sphere)
        constraints = []

        def constraint_func(x):
            points = x.reshape(-1, 3)
            norms = np.linalg.norm(points, axis=1)
            return norms - 1.0  # Should be near 0 for unit sphere

        # Add constraint for each point to lie on unit sphere
        for i in range(14):
            constraints.append({'type': 'eq', 'fun': lambda x, i=i: constraint_func(x)[i]})

        # Phase 1: Coarse optimization (fast, loose constraints)
        bounds = [(-1.2, 1.2)] * len(x0)

        result = minimize(
            objective_function,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-4, 'gtol': 1e-4},
            tol=1e-4
        )

        # Phase 2: Medium optimization (moderate constraints)
        x1 = result.x
        bounds = [(-1.1, 1.1)] * len(x0)

        result = minimize(
            objective_function,
            x1,
            method='L-BFGS-B',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-6, 'gtol': 1e-6},
            tol=1e-6
        )

        # Phase 3: Fine optimization (tight constraints)
        x2 = result.x
        bounds = [(-1.05, 1.05)] * len(x0)

        result = minimize(
            objective_function,
            x2,
            method='L-BFGS-B',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8},
            tol=1e-8
        )

        return result.x

    best_ratio = -np.inf
    best_points = None

    # Multi-start optimization with enhanced initialization and more restarts
    # Try multiple initial configurations including Sobol sequence
    initial_configs = []

    # 1. Enhanced hybrid initialization (existing method)
    initial_configs.append(("enhanced_hybrid", enhanced_hybrid_initialization(14)))

    # 2. Sobol sequence initialization
    initial_configs.append(("sobol", sobol_points_sphere(14)))

    # 3. Fibonacci spiral initialization
    initial_configs.append(("fibonacci", fibonacci_spiral_sphere(14)))

    for config_name, initial_points in initial_configs:
        for restart in range(5):  # 5 restarts per initialization method
            np.random.seed(42 + restart)

            x0 = initial_points.flatten()

            # Add small random perturbation to break any symmetry
            perturbation = np.random.normal(0, 0.01, x0.shape)
            x0 += perturbation

            # Optimize with progressive refinement
            try:
                optimized_points = progressive_optimization(x0, maxiter=1000)

                # Calculate final ratio
                ratio = distance_ratio(optimized_points)

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()

            except Exception as e:
                continue

    # If no good solution found, fallback to enhanced initialization
    if best_points is None:
        initial_points = enhanced_hybrid_initialization(14)
        best_points = initial_points.flatten()

    # Convert back to 14x3 array
    final_points = best_points.reshape(14, 3)

    return final_points

# EVOLVE-BLOCK-END