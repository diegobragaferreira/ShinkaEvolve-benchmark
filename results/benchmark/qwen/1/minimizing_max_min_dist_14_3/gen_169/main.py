# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi
import time

def sobol_points_sphere(n_points):
    """Generate points on sphere using 3D Sobol sequence for better space-filling properties"""
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
        return fibonacci_sphere(n_points)

def fibonacci_sphere(n):
    """Generate n points on a sphere using Fibonacci spiral method"""
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle

    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

    def voronoi_uniformity_penalty(points):
        """Calculate penalty based on Voronoi cell area uniformity"""
        try:
            # Project points to unit sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            normalized_points = points / norms

            sv = SphericalVoronoi(normalized_points)
            areas = sv.voronoi_cell_areas()

            # Return variance of areas (lower is better for uniformity)
            return np.var(areas)
        except:
            return 1000.0

    def objective(x):
        # Reshape x into 14 points in 3D
        points = x.reshape(-1, 3)

        # Calculate pairwise distances
        distances = pdist(points)

        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Return negative ratio since we want to maximize
        if d_max < 1e-10:
            return -1e10

        # Base ratio
        ratio = d_min / d_max

        # Add penalty for non-uniform Voronoi cells (higher variance = worse uniformity)
        voronoi_penalty = voronoi_uniformity_penalty(points)

        # Weighted combination: prioritize ratio but penalize poor uniformity
        # Use a moderate penalty coefficient (0.15) to better balance both objectives
        penalty_factor = 0.15 * voronoi_penalty

        # Also add a term to encourage larger minimum distances
        # This helps prevent points from clustering too closely
        min_distance_term = 0.05 * d_min

        return -(ratio - penalty_factor - min_distance_term)

    def normalize_to_unit_sphere(points):
        """Normalize points to lie exactly on unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return points / norms

    def mixed_initialization(n_points):
        """Generate high-quality initial points using mixed strategies"""
        # Strategy 1: Sobol points for excellent space-filling
        sobol_points = sobol_points_sphere(n_points)

        # Strategy 2: Fibonacci points as baseline
        fib_points = fibonacci_sphere(n_points)

        # Strategy 3: Random points for diversity
        random_points = np.random.rand(n_points, 3) * 2 - 1  # [-1, 1]

        # Normalize all to unit sphere
        sobol_points = normalize_to_unit_sphere(sobol_points)
        fib_points = normalize_to_unit_sphere(fib_points)
        random_points = normalize_to_unit_sphere(random_points)

        # Combine strategies with weighted selection
        # Use Sobol points as primary strategy, but mix in others
        strategy_weights = [0.5, 0.3, 0.2]  # weights for each strategy
        strategies = [sobol_points, fib_points, random_points]

        # Select the best strategy or combine multiple
        np.random.seed(42)  # For reproducibility
        selected_strategy = np.random.choice(strategies, p=strategy_weights)

        # Add controlled noise to break symmetries
        noise_magnitude = 0.03
        noise = np.random.normal(0, noise_magnitude, selected_strategy.shape)
        initial_points = selected_strategy + noise

        # Re-normalize to sphere
        initial_points = normalize_to_unit_sphere(initial_points)

        # Scale to fit nicely in [0,1]^3
        # Center and scale to avoid extreme boundaries
        center = np.mean(initial_points, axis=0)
        centered = initial_points - center

        # Scale to fit in [-0.4, 0.4] range, then translate to [0.1, 0.9]
        max_extent = np.max(np.abs(centered))
        if max_extent > 0:
            scaled = centered / max_extent * 0.4
        else:
            scaled = centered

        final_points = scaled + 0.5

        return final_points

    # Multi-stage optimization with adaptive parameters
    best_result = None
    best_ratio = -np.inf

    # Try multiple independent optimization runs with enhanced diversity
    # Using 8 different random seeds for better exploration
    seeds = [42, 123, 456, 789, 1001, 2002, 3003, 4004]

    # Different initialization strategies to use
    init_strategies = ['sobol', 'fibonacci', 'random']

    # Try multiple independent optimization runs
    for i, seed in enumerate(seeds):
        np.random.seed(seed)

        # Select initialization strategy based on restart index for better diversity
        strategy_idx = i % len(init_strategies)
        if init_strategies[strategy_idx] == 'sobol':
            initial_points = sobol_points_sphere(14)
        elif init_strategies[strategy_idx] == 'fibonacci':
            initial_points = fibonacci_sphere(14)
        else:  # random
            initial_points = np.random.rand(14, 3) * 2 - 1  # [-1, 1]

        # Normalize to unit sphere first
        initial_points = normalize_to_unit_sphere(initial_points)

        # Add controlled noise for diversity
        noise_magnitude = 0.03 + (i * 0.005)  # Increasing noise for later runs
        noise = np.random.normal(0, noise_magnitude, initial_points.shape)
        initial_points = initial_points + noise

        # Re-normalize to sphere
        initial_points = normalize_to_unit_sphere(initial_points)

        # Flatten for optimization
        x0 = initial_points.flatten()

        # Set up bounds for optimization (0 to 1 for all coordinates)
        bounds = [(0.0, 1.0)] * 14 * 3

        # Stage 1: Differential Evolution for global search with adaptive parameters
        try:
            # Adaptive parameter selection based on problem complexity
            de_params = {
                'maxiter': 250,
                'popsize': 12,
                'tol': 1e-8,
                'mutation': (0.5, 1.0),
                'recombination': 0.7,
                'disp': False
            }

            de_result = differential_evolution(
                objective,
                bounds,
                seed=seed,
                **de_params
            )

            # Stage 2: Local refinement with L-BFGS-B for precise optimization
            refined_result = minimize(
                objective,
                de_result.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-14, 'gtol': 1e-14, 'maxiter': 500},
                callback=None
            )

            # Evaluate final result
            final_points = refined_result.x.reshape(-1, 3)

            # Validate and calculate true ratio
            distances = pdist(final_points)
            d_min = np.min(distances)
            d_max = np.max(distances)

            if d_max > 1e-10:
                ratio = d_min / d_max
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_result = refined_result.x.copy()

        except Exception as e:
            continue

    # If no good result was found, fallback to direct optimization
    if best_result is None:
        np.random.seed(42)
        initial_points = adaptive_initialization(14)
        x0 = initial_points.flatten()
        bounds = [(0.0, 1.0)] * 14 * 3

        # Direct optimization with simpler parameters for robustness
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=200,
            popsize=10,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            disp=False
        )
        return result.x.reshape(-1, 3)

    return best_result.reshape(-1, 3)


# EVOLVE-BLOCK-END