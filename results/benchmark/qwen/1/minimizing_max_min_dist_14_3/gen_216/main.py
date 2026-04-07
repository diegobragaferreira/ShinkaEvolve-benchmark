# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist, pdist
from scipy.stats import qmc
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

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

    def constraint_sphere(x):
        # Ensure points stay within unit sphere
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return 1 - norms  # Should be >= 0

    def constraint_bounds(x):
        # Ensure all points are within [0,1]^3 bounds
        points = x.reshape(-1, 3)
        return np.concatenate([
            points.flatten() - 0.0,      # lower bound
            1.0 - points.flatten()       # upper bound
        ])

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
        """Generate points using Sobol sequence on sphere"""
        # Generate Sobol sequence in 3D space
        sobol_engine = qmc.Sobol(d=3, seed=42)
        points = sobol_engine.random(n)

        # Transform to sphere using spherical coordinates
        # Map [0,1]^3 to sphere coordinates
        theta = np.arccos(1 - 2 * points[:, 0])  # Polar angle
        phi = 2 * np.pi * points[:, 1]          # Azimuthal angle
        # Scale the radial component to be 1 (unit sphere)

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        return np.column_stack([x, y, z])

    def generate_perturbed_fibonacci_points(n, perturbation_strength=0.05):
        """Generate fibonacci points with small random perturbations"""
        base_points = generate_fibonacci_points(n)
        perturbations = np.random.normal(0, perturbation_strength, (n, 3))
        perturbed_points = base_points + perturbations
        # Normalize back to unit sphere
        perturbed_points = perturbed_points / np.linalg.norm(perturbed_points, axis=1, keepdims=True)
        return perturbed_points

    # Multiple starting configurations with enhanced diversity
    configs = []

    # Configuration 1: Fibonacci spiral on sphere (primary)
    configs.append(generate_fibonacci_points(14))

    # Configuration 2: Perturbed Fibonacci with small perturbation
    np.random.seed(100)
    fib_points = generate_fibonacci_points(14)
    perturbed_fib = fib_points + np.random.normal(0, 0.02, fib_points.shape)
    perturbed_fib = normalize_points(perturbed_fib)
    configs.append(perturbed_fib)

    # Configuration 3: Sobol sequence points
    configs.append(generate_sobol_points(14))

    # Configuration 4: Another perturbed version with different seed
    np.random.seed(456)
    configs.append(generate_perturbed_fibonacci_points(14, 0.03))

    # Configuration 5: Fibonacci again for redundancy
    configs.append(generate_fibonacci_points(14))

    # Configuration 6: Another deterministic seed
    np.random.seed(123)
    random_points = np.random.randn(14, 3)
    random_points = normalize_points(random_points)
    configs.append(random_points)

    # Configuration 7: Another random approach
    np.random.seed(789)
    random_points2 = np.random.randn(14, 3)
    random_points2 = normalize_points(random_points2)
    configs.append(random_points2)

    # Configuration 8: Perturbed Fibonacci with larger perturbation
    np.random.seed(300)
    perturbed_fib_large = generate_fibonacci_points(14) + np.random.normal(0, 0.05, (14, 3))
    perturbed_fib_large = normalize_points(perturbed_fib_large)
    configs.append(perturbed_fib_large)

    # Main optimization loop with multiple restarts
    best_final_points = None
    best_ratio = 0
    best_objective_value = float('inf')

    # First try differential evolution on each configuration with better parameters
    for i, initial_config in enumerate(configs):
        try:
            # Use differential evolution for global search first
            n_points = 14
            n_vars = n_points * 3  # 14 points * 3 coordinates each

            # Bounds for each coordinate: [0, 1] to keep within unit cube
            bounds = [(0, 1) for _ in range(n_vars)]

            # Run differential evolution with better settings
            result = differential_evolution(
                lambda x: -objective_ratio(x),  # Negative because we want to maximize
                bounds,
                seed=42 + i,
                maxiter=1000,  # Increased iterations for better convergence
                popsize=25,   # Larger population size
                tol=1e-9,     # Tighter tolerance
                mutation=(0.5, 1),
                recombination=0.8,  # Higher recombination rate
                disp=False
            )

            # Extract optimized points 
            optimized_points = result.x.reshape(-1, 3)

            # Evaluate this solution with ratio calculation
            ratio = objective_ratio(result.x)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_objective_value = -result.fun
                best_final_points = optimized_points.copy()

        except Exception as e:
            continue

    # If no good solution from DE, try local optimization from best configurations
    if best_final_points is None:
        # Try optimizing from the best initial configurations using local method
        for i, initial_config in enumerate(configs[:5]):  # Try first 5 configs
            try:
                # Local optimization around initial point
                x0 = initial_config.flatten()
                cons = [
                    {'type': 'ineq', 'fun': constraint_bounds}
                ]

                # Try multiple local optimization methods for robustness
                methods = ['SLSQP', 'L-BFGS-B']
                for method in methods:
                    try:
                        result = minimize(
                            lambda x: -objective_ratio(x),  # Negative because we want to maximize
                            x0, 
                            method=method, 
                            constraints=cons,
                            options={'ftol': 1e-12, 'maxiter': 1000}
                        )
                        
                        if result.success:
                            optimized_points = result.x.reshape(-1, 3)

                            # Evaluate this solution
                            ratio = objective_ratio(result.x)
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_objective_value = -result.fun
                                best_final_points = optimized_points.copy()
                            break  # Break if successful optimization
                    except:
                        continue
                        
            except Exception as e:
                continue

    # Final refinement step with enhanced optimization
    if best_final_points is not None:
        try:
            # Try several refinement approaches
            refinements = [
                ('SLSQP', {'ftol': 1e-12, 'maxiter': 500}),
                ('L-BFGS-B', {'ftol': 1e-12, 'maxiter': 500})
            ]
            
            for method, options in refinements:
                try:
                    x0 = best_final_points.flatten()
                    cons = [
                        {'type': 'ineq', 'fun': constraint_bounds}
                    ]

                    refined_result = minimize(
                        lambda x: -objective_ratio(x),  # Negative because we want to maximize
                        x0, 
                        method=method, 
                        constraints=cons,
                        options=options
                    )

                    if refined_result.success:
                        refined_points = refined_result.x.reshape(-1, 3)

                        # Re-evaluate final solution
                        refined_ratio = objective_ratio(refined_result.x)
                        if refined_ratio > best_ratio:
                            best_ratio = refined_ratio
                            best_final_points = refined_points.copy()
                        break
                        
                except Exception as e:
                    continue
                    
        except Exception as e:
            pass

    # If still no solution, return the last attempt or fallback to Fibonacci
    if best_final_points is None:
        return generate_fibonacci_points(14)

    return best_final_points

# EVOLVE-BLOCK-END