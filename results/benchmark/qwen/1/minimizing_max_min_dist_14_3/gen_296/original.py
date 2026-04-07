# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from scipy.stats import qmc
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses hierarchical evolutionary optimization with enhanced initialization and constraint handling.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def objective_ratio(x):
        """Objective function to maximize min/max distance ratio"""
        points = x.reshape(-1, 3)
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        if max_dist == 0 or min_dist == 0:
            return 0.0
        return min_dist / max_dist

    def objective_with_penalty(x, penalty_weight=1000.0):
        """Objective with penalty for constraint violations"""
        points = x.reshape(-1, 3)
        ratio = objective_ratio(x)

        # Penalty for points outside unit sphere
        norms = np.linalg.norm(points, axis=1)
        penalty = penalty_weight * np.sum(np.maximum(0, norms - 1.0)**2)

        # Return negative ratio plus penalty for minimization
        return -ratio + penalty

    def constraint_sphere(x):
        """Ensure points are on or inside unit sphere"""
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        # Return positive values when constraints are satisfied (<= 0 for violation)
        return 1.0 - norms

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

    def generate_sobol_like_points(n):
        """Generate points using a Sobol-like sequence for better space filling"""
        points = []
        # Use prime numbers for better distribution
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
        phi = np.pi * (3 - np.sqrt(5))  # golden angle

        for i in range(n):
            # Use multiple prime-based sequences for better distribution
            y = 1 - (i / float(n - 1)) * 2
            radius = np.sqrt(1 - y * y)

            # Mix with prime number sequences for better spacing
            theta = phi * i + (primes[i % len(primes)] * 0.1)

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])
        return np.array(points)

    def generate_diverse_initial_points(n, seed=42):
        """Generate diverse initial configurations"""
        np.random.seed(seed)
        configs = []

        # Configuration 1: Fibonacci spiral
        configs.append(generate_fibonacci_points(n))

        # Configuration 2: Sobol-like sequence
        configs.append(generate_sobol_like_points(n))

        # Configuration 3: Perturbed Fibonacci with small noise
        fib_points = generate_fibonacci_points(n)
        noise = np.random.normal(0, 0.03, (n, 3))
        perturbed = fib_points + noise
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        configs.append(perturbed / norms)

        # Configuration 4: Perturbed Sobol-like with moderate noise
        sobol_points = generate_sobol_like_points(n)
        noise = np.random.normal(0, 0.05, (n, 3))
        perturbed = sobol_points + noise
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        configs.append(perturbed / norms)

        # Configuration 5: Random points with sphere normalization
        random_points = np.random.randn(n, 3)
        norms = np.linalg.norm(random_points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        configs.append(random_points / norms)

        # Configuration 6: Sobol sequence in [0,1]^3 mapped to sphere
        try:
            sobol_engine = qmc.Sobol(d=3, seed=seed)
            sobol_points = sobol_engine.random(n)
            # Map to sphere using similar approach as above
            theta = np.arccos(1 - 2 * sobol_points[:, 0])
            phi = 2 * np.pi * sobol_points[:, 1]
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            sobol_sphere = np.column_stack([x, y, z])
            configs.append(sobol_sphere)
        except:
            # Fallback to random if Sobol fails
            configs.append(np.random.randn(n, 3) / np.sqrt(np.sum(np.random.randn(n, 3)**2, axis=1, keepdims=True)))

        return configs

    def hierarchical_optimization(initial_points, max_iter=300):
        """Hierarchical optimization with coarse-to-fine approach"""
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_ratio = 0.0

        # Stage 1: Coarse global search with Differential Evolution
        try:
            x0 = current_points.flatten()
            bounds = [(-2.0, 2.0) for _ in range(len(x0))]

            # Use differential evolution for global search
            de_result = differential_evolution(
                objective_with_penalty,
                bounds,
                args=(1000.0,),
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7
            )

            if de_result.success:
                de_points = de_result.x.reshape(-1, 3)
                norms = np.linalg.norm(de_points, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                de_points = de_points / norms

                ratio = objective_ratio(de_result.x)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = de_points.copy()
                    current_points = de_points.copy()
        except Exception:
            pass

        # Stage 2: Local refinement with L-BFGS-B
        try:
            x0 = current_points.flatten()
            result = minimize(
                objective_with_penalty,
                x0,
                method='L-BFGS-B',
                args=(1000.0,),
                bounds=[(-2.0, 2.0) for _ in range(len(x0))],
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10}
            )

            if result.success:
                refined_points = result.x.reshape(-1, 3)
                norms = np.linalg.norm(refined_points, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                refined_points = refined_points / norms

                ratio = objective_ratio(result.x)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = refined_points.copy()
        except Exception:
            pass

        # Stage 3: Additional refinement with SLSQP
        try:
            x0 = best_points.flatten()
            cons = {'type': 'ineq', 'fun': constraint_sphere}

            result = minimize(
                lambda x: -objective_ratio(x),
                x0,
                method='SLSQP',
                constraints=cons,
                options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 150}
            )

            if result.success:
                slsqp_points = result.x.reshape(-1, 3)
                norms = np.linalg.norm(slsqp_points, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                slsqp_points = slsqp_points / norms

                ratio = objective_ratio(result.x)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = slsqp_points.copy()
        except Exception:
            pass

        return best_points

    # Generate diverse initial configurations
    initial_configs = generate_diverse_initial_points(14, seed=42)

    # Main optimization loop
    best_final_points = None
    best_ratio = 0.0

    # Test each initial configuration
    for i, initial_config in enumerate(initial_configs):
        try:
            # Apply hierarchical optimization
            optimized_points = hierarchical_optimization(initial_config, max_iter=200)

            # Evaluate solution
            ratio = objective_ratio(optimized_points.flatten())

            if ratio > best_ratio:
                best_ratio = ratio
                best_final_points = optimized_points.copy()

        except Exception:
            continue

    # Final verification and fallback
    if best_final_points is None:
        # Fallback to best Fibonacci configuration
        fib_points = generate_fibonacci_points(14)
        return fib_points

    # Final refinement step with higher precision
    try:
        x0 = best_final_points.flatten()
        cons = {'type': 'ineq', 'fun': constraint_sphere}

        result = minimize(
            lambda x: -objective_ratio(x),
            x0,
            method='L-BFGS-B',
            bounds=[(-2.0, 2.0) for _ in range(len(x0))],
            constraints=cons,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12}
        )

        if result.success:
            final_points = result.x.reshape(-1, 3)
            norms = np.linalg.norm(final_points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            final_points = final_points / norms

            final_ratio = objective_ratio(result.x)
            if final_ratio > best_ratio:
                best_final_points = final_points.copy()

    except Exception:
        pass

    return best_final_points

# EVOLVE-BLOCK-END