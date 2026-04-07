# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    n = 16
    d = 2
    best_ratio = -np.inf
    best_points = None

    def compute_ratio(points):
        """Compute the min/max distance ratio for given points."""
        distances = cdist(points, points, metric='euclidean')
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist <= 0:
            return 0
        return min_dist / max_dist

    def compute_distance_stats(points):
        """Compute statistics of distance distribution."""
        distances = cdist(points, points, metric='euclidean')
        np.fill_diagonal(distances, np.inf)
        if len(distances[distances < np.inf]) == 0:
            return 0, 0, 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        mean_dist = np.mean(distances[distances < np.inf])
        std_dist = np.std(distances[distances < np.inf])
        return min_dist, max_dist, mean_dist, std_dist

    # Multiple restart strategies with enhanced geometric understanding
    def _structured_grid_init():
        """Initialize with structured grid pattern."""
        grid_points = np.array([[i, j] for i in range(4) for j in range(4)])
        points = grid_points.astype(float) / 3.0  # Normalize to [0,1] range
        # Add slight perturbations to break symmetry
        np.random.seed(42)
        points += np.random.uniform(-0.015, 0.015, points.shape)
        points = np.clip(points, 0, 1)
        return points

    def _hexagonal_packing_init():
        """Initialize with hexagonal packing pattern."""
        points = []
        rows, cols = 4, 4
        for i in range(rows):
            for j in range(cols):
                x_offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + x_offset) * 0.25 + 0.125
                y = i * 0.25 + 0.125
                points.append([x, y])
        return np.array(points)

    def _golden_spiral_init():
        """Initialize with golden spiral pattern."""
        points = []
        phi = (1 + np.sqrt(5)) / 2
        for i in range(n):
            angle = 2 * np.pi * i / phi
            radius = np.sqrt(i / (n - 1)) if n > 1 else 0
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            # Scale and center
            x = 0.4 * x + 0.5
            y = 0.4 * y + 0.5
            points.append([x, y])
        return np.array(points)

    def _adaptive_clustering_init():
        """Initialize with adaptive clustering prevention."""
        np.random.seed(42)
        points = np.random.rand(n, d)
        
        # Apply clustering avoidance
        for iter in range(10):
            for i in range(n):
                # Move away from cluster centers
                distances = cdist([points[i]], points)[0]
                distances[i] = np.inf  # Exclude self
                nearest = np.argmin(distances)
                if distances[nearest] < 0.1:  # If too close
                    direction = points[i] - points[nearest]
                    if np.linalg.norm(direction) > 0:
                        points[i] += direction * 0.02
            points = np.clip(points, 0, 1)
        return points

    restart_strategies = [
        _structured_grid_init,
        _hexagonal_packing_init,
        _golden_spiral_init,
        _adaptive_clustering_init
    ]

    # Dual-space optimization approach
    class DualSpaceOptimizer:
        def __init__(self):
            self.optimization_history = []
            
        def dual_objective(self, x, weights=(0.7, 0.3)):
            """
            Dual objective function that considers both distance distribution and ratio.
            Combines:
            1. Negative minimum distance (to maximize it)
            2. Positive distance variance (to avoid clustering)
            """
            # Reshape
            points = x.reshape(n, d)
            
            # Compute distances
            distances = cdist(points, points, metric='euclidean')
            np.fill_diagonal(distances, np.inf)
            
            if len(distances[distances < np.inf]) == 0:
                return 1e10
                
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            std_dist = np.std(distances[distances < np.inf])
            mean_dist = np.mean(distances[distances < np.inf])
            
            # Avoid division by zero
            if max_dist <= 0:
                return 1e10
                
            ratio = min_dist / max_dist
            
            # Weighted combination of objectives
            # Maximize ratio, minimize distance variance
            obj_value = -ratio + weights[1] * std_dist / (mean_dist + 1e-8)
            return obj_value

        def symmetry_breaking_constraints(self, x):
            """Enforce symmetry breaking through fixed reference points."""
            points = x.reshape(n, d)
            # Fix first point to be near origin (0.1, 0.1) to break translation symmetry
            con1 = points[0, 0] - 0.1
            con2 = points[0, 1] - 0.1
            # Fix second point to ensure orientation
            con3 = points[1, 0] - 0.3
            con4 = points[1, 1] - 0.1
            return np.array([con1, con2, con3, con4])

        def lexicographic_ordering(self, x):
            """Ensure lexico-graphic ordering to break permutation symmetry."""
            points = x.reshape(n, d)
            constraints = []
            for i in range(1, n):
                # X coordinate should be non-decreasing
                constraints.append(points[i, 0] - points[i-1, 0])
                # If x-coordinates are equal, y should be non-decreasing
                if abs(points[i, 0] - points[i-1, 0]) < 1e-8:
                    constraints.append(points[i, 1] - points[i-1, 1])
            return np.array(constraints)

        def optimize_with_hierarchical_approach(self, initial_points, max_time=160):
            """Optimize using hierarchical approach: global → medium → local"""
            start_time = time.time()
            
            # Level 1: Global search with differential evolution (coarse)
            try:
                bounds = [(0, 1) for _ in range(n * d)]
                
                # Simple DE for global search
                de_result = differential_evolution(
                    lambda x: self.dual_objective(x, weights=(0.8, 0.2)),
                    bounds,
                    maxiter=20,
                    popsize=12,
                    seed=42,
                    tol=1e-5
                )
                
                if de_result.success:
                    initial_points = de_result.x.reshape(n, d)
                    self.optimization_history.append(("DE", compute_ratio(initial_points)))
                    
            except Exception as e:
                pass

            # Level 2: Medium refinement with L-BFGS-B
            try:
                bounds = [(0, 1) for _ in range(n * d)]
                result = minimize(
                    lambda x: self.dual_objective(x, weights=(0.7, 0.3)),
                    initial_points.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'ftol': 1e-10, 'gtol': 1e-10}
                )
                
                if result.success:
                    refined_points = result.x.reshape(n, d)
                    self.optimization_history.append(("L-BFGS", compute_ratio(refined_points)))
                    initial_points = refined_points
            except Exception as e:
                pass

            # Level 3: Fine refinement with SLSQP
            try:
                bounds = [(0, 1) for _ in range(n * d)]
                result = minimize(
                    lambda x: self.dual_objective(x, weights=(0.6, 0.4)),
                    initial_points.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'ftol': 1e-12, 'gtol': 1e-12}
                )
                
                if result.success:
                    final_points = result.x.reshape(n, d)
                    self.optimization_history.append(("SLSQP", compute_ratio(final_points)))
                    return final_points
            except Exception as e:
                pass

            return initial_points

    # Main optimization loop
    optimizer = DualSpaceOptimizer()
    
    # Try each initialization strategy
    for strategy_idx, init_func in enumerate(restart_strategies):
        for restart in range(2):  # Reduced restarts for efficiency
            np.random.seed(strategy_idx * 1000 + restart)

            # Get initial points
            points = init_func()
            
            # Apply dual-space optimization
            optimized_points = optimizer.optimize_with_hierarchical_approach(points)
            
            # Calculate ratio for this optimization run
            ratio = compute_ratio(optimized_points)

            # Keep track of best solution
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()

    # Additional evolutionary restart with enhanced constraints
    try:
        # Enhanced DE with custom constraints
        def enhanced_de_objective(x):
            points = x.reshape(-1, 2)
            distances = cdist(points, points, metric='euclidean')
            np.fill_diagonal(distances, np.inf)
            
            if len(distances[distances < np.inf]) == 0:
                return 1e10
                
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist <= 0:
                return 1e10
                
            # Use ratio maximization with penalty for bad configurations
            ratio = min_dist / max_dist
            
            # Add penalty for points too close to boundaries (to avoid edge clustering)
            boundary_penalty = 0
            for pt in points:
                boundary_dist = min(pt[0], 1-pt[0], pt[1], 1-pt[1])
                if boundary_dist < 0.05:
                    boundary_penalty += (0.05 - boundary_dist) * 1000
            
            return -(ratio - boundary_penalty/10000)

        bounds = [(0, 1) for _ in range(n * d)]
        
        de_result = differential_evolution(
            enhanced_de_objective,
            bounds,
            maxiter=30,
            popsize=15,
            seed=42,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7
        )

        if de_result.success:
            de_points = de_result.x.reshape(-1, 2)
            # Refine the DE result
            refined_de_points = optimizer.optimize_with_hierarchical_approach(de_points)
            ratio = compute_ratio(refined_de_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined_de_points.copy()

    except Exception as e:
        pass

    # Final fallback check
    if best_points is None:
        # Use the best initial configuration
        best_initial = max(restart_strategies, key=lambda f: compute_ratio(f())) 
        best_points = best_initial()

    return best_points

# EVOLVE-BLOCK-END