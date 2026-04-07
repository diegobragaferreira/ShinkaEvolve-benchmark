# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.stats import qmc
import warnings
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def fibonacci_spiral_on_sphere(n_points: int) -> np.ndarray:
        """Generate points on sphere using Fibonacci spiral with golden angle."""
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle

        for i in range(n_points):
            y = 1 - (i / (n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def sobol_initialization(n: int, seed: int = 42) -> np.ndarray:
        """Generate points using Sobol sequence for better space-filling properties"""
        try:
            # Create Sobol sequence sampler
            sampler = qmc.Sobol(d=3, seed=seed)
            # Generate points
            points = sampler.random(n)
            # Scale to [-1, 1]^3
            points = points * 2 - 1
            return points
        except:
            # Fallback to random initialization if Sobol fails
            return np.random.uniform(-1, 1, (n, 3))

    def icosahedron_initialization(n: int) -> np.ndarray:
        """Initialize points using icosahedron vertices and refinement"""
        # Regular icosahedron vertices
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
        ])

        # Normalize vertices to unit sphere
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        vertices = vertices / np.where(norms > 0, norms, 1)

        # If we need more than 12 points, generate additional points
        if n <= 12:
            return vertices[:n]
        else:
            # For more points, distribute them more evenly by subdividing faces
            points = vertices.copy()
            for i in range(12, n):
                # Add point near one of existing vertices with small perturbation
                idx = i % 12
                base_point = vertices[idx]
                # Add small random perturbation
                perturbation = np.random.normal(0, 0.05, 3)
                new_point = base_point + perturbation
                # Normalize back to sphere
                norm = np.linalg.norm(new_point)
                if norm > 0:
                    new_point = new_point / norm
                points = np.vstack([points, new_point])

            return points

    def normalize_to_unit_sphere(points: np.ndarray) -> np.ndarray:
        """Normalize points to lie on unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        safe_norms = np.where(norms == 0, 1, norms)
        return points / safe_norms

    def calculate_min_max_ratio(points: np.ndarray) -> float:
        """Calculate the minimum-to-maximum distance ratio"""
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist <= 0:
            return 0.0
        return min_dist / max_dist

    def objective_ratio_with_variance_regularization(points_flat: np.ndarray, lambda_reg: float = 0.15) -> float:
        """Objective function that maximizes min/max distance ratio with distance variance regularization"""
        points = points_flat.reshape(-1, 3)
        # Normalize points to unit sphere
        points = normalize_to_unit_sphere(points)

        # Compute distance matrix
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)

        min_dist = np.min(distances)
        max_dist = np.max(distances)
        distance_variance = np.max(distances) - np.min(distances)

        if max_dist <= 0:
            return -1.0

        # Combine ratio with variance regularization
        # We want to maximize min_dist / max_dist while minimizing distance variance
        ratio = min_dist / max_dist
        regularized_objective = -ratio + lambda_reg * distance_variance
        return regularized_objective

    def objective_energy(x):
        """Objective function that minimizes potential energy (inverse distance)"""
        points = x.reshape(-1, 3)
        # Normalize points to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / np.where(norms > 0, norms, 1)
        # Compute distance matrix
        dist_matrix = cdist(points, points)
        # Avoid division by zero
        np.fill_diagonal(dist_matrix, 1e-10)
        # Energy is sum of inverse distances
        return np.sum(1.0 / dist_matrix)

    def constraint_sphere(x):
        """Constraint function ensuring all points lie on unit sphere"""
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        # Return difference from unit radius (should be close to 0)
        return norms - 1.0

    def adaptive_constraint_tightening(iteration: int, max_iterations: int, stage: str = 'global') -> float:
        """Adaptively tighten constraints during optimization with stage-specific behavior"""
        # Start with relaxed constraints, tighten towards the end
        if stage == 'global':
            relaxation_factor = 1.0 - (iteration / max_iterations) * 0.3
            return max(0.05, relaxation_factor)
        else:  # local refinement
            relaxation_factor = 1.0 - (iteration / max_iterations) * 0.7
            return max(0.01, relaxation_factor)

    # Multi-start optimization with diverse strategies
    best_ratio = -np.inf
    best_points = None

    # Strategy 1: Standard Fibonacci spiral
    fib_points = fibonacci_spiral_on_sphere(14)

    # Strategy 2: Sobol sequence initialization
    sobol_points = sobol_initialization(14, seed=42)

    # Strategy 3: Icosahedron-based initialization
    ico_points = icosahedron_initialization(14)

    # Strategy 4: Perturbed Fibonacci points
    np.random.seed(42)
    perturbed_fib = fib_points + np.random.normal(0, 0.03, fib_points.shape)

    # Strategy 5: Perturbed Sobol points
    perturbed_sobol = sobol_points + np.random.normal(0, 0.03, sobol_points.shape)

    # Strategy 6: Perturbed icosahedron points
    perturbed_ico = ico_points + np.random.normal(0, 0.03, ico_points.shape)

    # Strategy 7: Enhanced Sobol sequence with different seed
    sobol_points_2 = sobol_initialization(14, seed=123)

    # Strategy 8: Random initialization with fixed seed
    np.random.seed(999)
    random_points = np.random.uniform(-1, 1, (14, 3))

    # Strategy 9: More diverse Sobol initialization with yet another seed
    sobol_points_3 = sobol_initialization(14, seed=456)

    # Strategy 10: Perturbed version of the golden spiral points with higher variance
    perturbed_fib_high = fib_points + np.random.normal(0, 0.08, fib_points.shape)

    initializations = [
        fib_points,
        sobol_points,
        ico_points,
        perturbed_fib,
        perturbed_sobol,
        perturbed_ico,
        sobol_points_2,
        random_points,
        sobol_points_3,
        perturbed_fib_high
    ]

    # Try each initialization with optimization
    for i, initial_points in enumerate(initializations):
        try:
            # Ensure points are on unit sphere
            initial_points = normalize_to_unit_sphere(initial_points)

            # Phase 1: Global optimization using L-BFGS-B
            x0 = initial_points.flatten()

            # Use L-BFGS-B for initial coarse optimization
            result_coarse = minimize(
                objective_energy,
                x0,
                method='L-BFGS-B',
                options={'ftol': 1e-8, 'gtol': 1e-8, 'maxiter': 300},
                tol=1e-8
            )

            if result_coarse.success:
                # Phase 2: Hybrid refinement with L-BFGS-B and SLSQP
                refined_points = result_coarse.x.reshape(-1, 3)
                refined_points = normalize_to_unit_sphere(refined_points)

                # Define constraint for SLSQP
                cons = {'type': 'eq', 'fun': constraint_sphere}

                # First, try L-BFGS-B for local refinement (faster but less precise)
                lbfgs_result = minimize(
                    objective_ratio_with_variance_regularization,
                    refined_points.flatten(),
                    method='L-BFGS-B',
                    options={'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 200},
                    tol=1e-10
                )

                # Then, if successful, do final SLSQP refinement for highest precision
                if lbfgs_result.success:
                    # Final refinement with SLSQP for maximum precision
                    fine_result = minimize(
                        objective_ratio_with_variance_regularization,
                        lbfgs_result.x,
                        method='SLSQP',
                        constraints=cons,
                        options={'ftol': 1e-12, 'maxiter': 500},
                        tol=1e-12
                    )

                    if fine_result.success:
                        final_points = fine_result.x.reshape(-1, 3)
                        final_points = normalize_to_unit_sphere(final_points)
                        ratio = calculate_min_max_ratio(final_points)

                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()
                    else:
                        # If SLSQP fails, use L-BFGS-B result
                        final_points = lbfgs_result.x.reshape(-1, 3)
                        final_points = normalize_to_unit_sphere(final_points)
                        ratio = calculate_min_max_ratio(final_points)

                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()
                else:
                    # If L-BFGS-B fails, just use the coarse result
                    coarse_points = result_coarse.x.reshape(-1, 3)
                    coarse_points = normalize_to_unit_sphere(coarse_points)
                    ratio = calculate_min_max_ratio(coarse_points)

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = coarse_points.copy()
            else:
                # If coarse optimization fails, just use the initial points
                ratio = calculate_min_max_ratio(initial_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = initial_points.copy()

        except Exception as e:
            warnings.warn(f"Initialization {i} failed: {str(e)}")
            continue

    # Fallback to standard Fibonacci spiral if everything else fails
    if best_points is None:
        try:
            initial_points = fibonacci_spiral_on_sphere(14)
            initial_points = normalize_to_unit_sphere(initial_points)

            # Direct optimization with SLSQP
            x0 = initial_points.flatten()
            cons = {'type': 'eq', 'fun': constraint_sphere}

            result = minimize(
                objective_ratio_with_variance_regularization,
                x0,
                method='SLSQP',
                constraints=cons,
                options={'ftol': 1e-12, 'maxiter': 500},
                tol=1e-12
            )

            if result.success:
                best_points = result.x.reshape(-1, 3)
                best_points = normalize_to_unit_sphere(best_points)
            else:
                best_points = initial_points

        except Exception:
            best_points = initial_points

    return best_points

# EVOLVE-BLOCK-END