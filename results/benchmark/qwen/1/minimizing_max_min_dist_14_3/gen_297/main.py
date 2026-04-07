# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
from scipy.stats import qmc
import time


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def objective(x):
        # Reshape x into 14 points in 3D
        points = x.reshape(-1, 3)

        # Calculate pairwise distances
        distances = pdist(points)

        # Remove near-zero distances that might occur due to numerical errors
        distances = distances[distances > 1e-12]
        
        if len(distances) == 0:
            return -np.inf

        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Return negative ratio since we want to maximize
        # We add a small epsilon to avoid division by zero
        if d_max < 1e-10:
            return -1e10
        return -d_min / d_max

    def constraint_sphere(x):
        # Ensure points stay on unit sphere
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0  # Should equal 0 for points on unit sphere

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

    def sobol_initialization(n_points, seed=42):
        """Initialize points using 3D Sobol sequence for better space-filling properties."""
        sampler = qmc.Sobol(d=3, seed=seed)
        points = sampler.random(n=n_points)
        # Scale to unit sphere
        points = points * 2 - 1  # Map to [-1, 1]^3
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        points = points / norms
        return points

    def generate_diverse_initializations():
        """Create diverse initial point configurations."""
        initial_configs = []
        
        # 1. Fibonacci spiral on sphere
        fib_points = fibonacci_spiral_sphere(14)
        initial_configs.append(fib_points.copy())
        
        # 2. Sobol sequence initialization  
        sobol_points = sobol_initialization(14, seed=42)
        initial_configs.append(sobol_points.copy())
        
        # 3. Perturbed Fibonacci
        np.random.seed(100)
        perturbation = np.random.normal(0, 0.02, (14, 3))
        perturbed_fib = fib_points + perturbation
        norms = np.linalg.norm(perturbed_fib, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        perturbed_fib = perturbed_fib / norms
        initial_configs.append(perturbed_fib.copy())
        
        # 4. Another Fibonacci variant
        np.random.seed(200)
        perturbation2 = np.random.normal(0, 0.015, (14, 3))
        fib_variant = fib_points + perturbation2
        norms = np.linalg.norm(fib_variant, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        fib_variant = fib_variant / norms
        initial_configs.append(fib_variant.copy())
        
        # 5. Random points on sphere
        np.random.seed(300)
        random_points = np.random.randn(14, 3)
        norms = np.linalg.norm(random_points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        random_points = random_points / norms
        initial_configs.append(random_points.copy())
        
        return initial_configs

    def optimize_from_initial(initial_points):
        """Optimize from given initial points using both DE and local methods."""
        x0 = initial_points.flatten()
        
        # First try differential evolution for global search
        bounds = [(-2, 2) for _ in range(14 * 3)]
        
        try:
            de_result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=300,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                disp=False
            )
            
            if de_result.success:
                optimized_points = de_result.x.reshape(-1, 3)
                # Normalize to unit sphere
                norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                optimized_points = optimized_points / norms
                return optimized_points
        except Exception:
            pass
            
        # Fallback to local optimization with constraints
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=[(-2, 2) for _ in range(14 * 3)],
                constraints={'type': 'eq', 'fun': constraint_sphere},
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 3)
                # Normalize to unit sphere
                norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                optimized_points = optimized_points / norms
                return optimized_points
        except Exception:
            pass
            
        return initial_points.copy()

    # Generate diverse initial configurations
    initial_configs = generate_diverse_initializations()
    
    # Multi-start optimization with best solution tracking
    best_ratio = -np.inf
    best_points = None
    
    # Try optimization from each initial configuration
    for i, initial_config in enumerate(initial_configs):
        try:
            optimized_points = optimize_from_initial(initial_config)
            
            # Evaluate the solution
            distances = pdist(optimized_points)
            distances = distances[distances > 1e-12]
            
            if len(distances) > 0:
                d_min = np.min(distances)
                d_max = np.max(distances)
                
                if d_max > 0:
                    ratio = d_min / d_max
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
        except Exception:
            continue

    # If no successful optimization, use the best initial configuration
    if best_points is None:
        # Return the Fibonacci spiral configuration as fallback
        return fibonacci_spiral_sphere(14)

    return best_points


# EVOLVE-BLOCK-END