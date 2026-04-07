# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """

    np.random.seed(42)

    def compute_min_max_ratio(points):
        """Compute the min/max distance ratio for given points."""
        if len(points) < 2:
            return 0.0

        # Compute pairwise distances efficiently
        distances = pdist(points)

        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Avoid division by zero
        if d_max == 0:
            return 0.0

        return d_min / d_max

    def fibonacci_sphere(n):
        """Generate points on sphere using Fibonacci spiral."""
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))

        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = golden_angle * i  # Golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def spherical_constraint(points):
        """Normalize points to lie on the unit sphere."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return points / norms

    def objective_function(points_flat):
        """Objective function to maximize - negative of min/max ratio."""
        # Reshape flat array to 2D points array
        points = points_flat.reshape(-1, 3)

        # Apply spherical constraint to keep points on unit sphere
        points = spherical_constraint(points)

        # Compute ratio
        ratio = compute_min_max_ratio(points)

        # Return negative because we want to maximize ratio, but optimizers minimize
        return -ratio

    def generate_diverse_initializations():
        """Generate multiple diverse initial point sets."""
        initial_sets = []

        # Strategy 1: Fibonacci sphere distribution
        fib_points = fibonacci_sphere(14)
        # Add small perturbations
        perturbed = fib_points + np.random.normal(0, 0.03, fib_points.shape)
        initial_sets.append(spherical_constraint(perturbed))

        # Strategy 2: Random points on sphere
        random_points = np.random.randn(14, 3)
        initial_sets.append(spherical_constraint(random_points))

        # Strategy 3: Pseudo-random distribution with some structure
        # Place points along axes and in between
        struct_points = np.zeros((14, 3))
        for i in range(14):
            if i < 3:
                # Along axes
                struct_points[i] = [1 if j==i else 0 for j in range(3)]
            elif i < 6:
                # Opposite axes
                struct_points[i] = [-1 if j==i-3 else 0 for j in range(3)]
            elif i < 9:
                # Diagonal combinations
                j = i - 6
                struct_points[i] = [1 if k==j else -1 if k==(j+1)%3 else 0 for k in range(3)]
            else:
                # Random points on sphere
                struct_points[i] = np.random.randn(3)
        initial_sets.append(spherical_constraint(struct_points))

        # Strategy 4: Slightly perturbed Fibonacci with larger variance
        fib_perturbed = fib_points + np.random.normal(0, 0.07, fib_points.shape)
        initial_sets.append(spherical_constraint(fib_perturbed))

        return initial_sets

    def hybrid_optimization(initial_points, maxiter=100):
        """Perform hybrid optimization combining global and local methods."""
        points = initial_points.copy()

        # Global optimization using differential evolution
        bounds = [(-1, 1)] * (14 * 3)

        try:
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=maxiter,
                popsize=25,
                seed=42,
                disp=False,
                polish=True,
                strategy='best1bin'
            )

            if result.success:
                points = result.x.reshape(-1, 3)
                points = spherical_constraint(points)
        except:
            pass

        # Local refinement with L-BFGS-B
        def local_obj(x_flat):
            points = x_flat.reshape(-1, 3)
            points = spherical_constraint(points)
            ratio = compute_min_max_ratio(points)
            return -ratio  # Negative for minimization

        try:
            x0 = points.flatten()
            bounds = [(-1, 1)] * (14 * 3)
            result = minimize(
                local_obj,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-9, 'gtol': 1e-9, 'disp': False}
            )
            if result.success:
                points = result.x.reshape(-1, 3)
                points = spherical_constraint(points)
        except:
            pass

        return points

    def adaptive_refinement(initial_points, max_iterations=100):
        """Apply adaptive refinement to improve solution quality."""
        points = initial_points.copy()
        best_ratio = compute_min_max_ratio(points)
        best_points = points.copy()
        
        # Adaptive step size control
        step_size = 0.01
        patience = 0
        max_patience = 10
        
        for iteration in range(max_iterations):
            current_ratio = compute_min_max_ratio(points)
            
            # Check for improvement
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = points.copy()
                patience = 0
                step_size = min(0.01, step_size * 1.1)  # Increase step size
            else:
                patience += 1
                if patience > max_patience:
                    step_size = max(0.0001, step_size * 0.8)  # Decrease step size
                    
            # Try small perturbations
            improved = False
            for i in range(14):
                for dim in range(3):
                    # Try perturbing in both directions
                    for direction in [-1, 1]:
                        test_points = points.copy()
                        test_points[i, dim] += direction * step_size
                        
                        # Project back to unit sphere
                        norm = np.linalg.norm(test_points[i])
                        if norm > 0:
                            test_points[i] = test_points[i] / norm
                        
                        test_ratio = compute_min_max_ratio(test_points)
                        
                        if test_ratio > best_ratio:
                            best_ratio = test_ratio
                            best_points = test_points.copy()
                            points = test_points.copy()
                            improved = True
                            patience = 0
                            break
                if improved:
                    break
                    
            if not improved and patience > max_patience:
                break
                
        return best_points

    # Multi-start optimization with diverse initializations
    best_solution = None
    best_ratio = 0.0

    # Generate multiple initial sets
    initial_sets = generate_diverse_initializations()

    # Try each initialization
    for i, initial_points in enumerate(initial_sets):
        # Perform hybrid optimization
        optimized_points = hybrid_optimization(initial_points, maxiter=50)
        ratio = compute_min_max_ratio(optimized_points)

        if ratio > best_ratio:
            best_ratio = ratio
            best_solution = optimized_points.copy()

    # Additional refinement with adaptive optimization
    if best_solution is not None:
        refined_points = adaptive_refinement(best_solution, max_iterations=50)
        ratio = compute_min_max_ratio(refined_points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_solution = refined_points.copy()

    # Final L-BFGS optimization for maximum precision
    if best_solution is not None:
        def final_obj(x_flat):
            points = x_flat.reshape(-1, 3)
            points = spherical_constraint(points)
            ratio = compute_min_max_ratio(points)
            return -ratio  # Negative because we minimize

        try:
            x0 = best_solution.flatten()
            bounds = [(-1, 1)] * (14 * 3)
            result = minimize(
                final_obj,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
            )
            if result.success:
                final_points = result.x.reshape(-1, 3)
                final_points = spherical_constraint(final_points)
                return final_points
        except:
            pass

    # If nothing worked, return the best initialization
    if best_solution is not None:
        return best_solution

    # Fallback to Fibonacci with small perturbation
    fib_points = fibonacci_sphere(14)
    fib_points = fib_points + np.random.normal(0, 0.05, fib_points.shape)
    return spherical_constraint(fib_points)

# EVOLVE-BLOCK-END