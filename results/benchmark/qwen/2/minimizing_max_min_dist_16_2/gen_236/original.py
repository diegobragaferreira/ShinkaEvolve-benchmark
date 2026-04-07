# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a geometry-inspired packing evolution algorithm with hierarchical refinement.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def compute_distance_ratios(points):
        """Helper function to compute the ratio metrics for evaluation."""
        distances = pdist(points)
        if len(distances) == 0:
            return 0, 0, 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0, 0, 0
        ratio = min_dist / max_dist
        return ratio, min_dist, max_dist

    def packing_energy_objective(x):
        """Energy-based objective that encourages good point distribution."""
        points = x.reshape(-1, 2)
        distances = pdist(points)

        if len(distances) == 0:
            return 0

        # Use inverse distance squared as energy term (repulsion effect)
        # But also consider the ratio structure
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        if max_dist == 0:
            return np.inf

        # Energy function that penalizes both very small and very large distances
        # This encourages a balanced distribution
        energy = 0
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                dist_sq = dx*dx + dy*dy

                if dist_sq > 0:
                    energy += 1.0 / dist_sq

        return energy

    def create_golden_ratio_pattern():
        """Create pattern based on golden ratio properties for even distribution."""
        np.random.seed(42)
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio

        # Distribute points using golden angle spiral approach
        for i in range(16):
            angle = i * 2 * np.pi / phi
            radius = np.sqrt(i / 15.0)  # Scale to keep within unit square
            x = 0.5 + radius * np.cos(angle) * 0.4
            y = 0.5 + radius * np.sin(angle) * 0.4
            points.append([x, y])

        # Add small perturbations to break perfect symmetry
        points = np.array(points)
        noise = np.random.normal(0, 0.01, points.shape)
        points = points + noise
        points = np.clip(points, 0, 1)
        return points

    def create_hierarchical_grid():
        """Create a hierarchy of grid patterns for multi-scale optimization."""
        # Start with coarse grid, then refine
        base_points = []

        # Base 2x2 grid with some offset for diversity
        offsets = [[0,0], [0.5,0], [0,0.5], [0.5,0.5]]
        for i in range(4):
            for j in range(4):
                x = 0.1 + j * 0.225 + offsets[i%4][0] * 0.05
                y = 0.1 + i * 0.225 + offsets[i%4][1] * 0.05
                base_points.append([x, y])

        return np.array(base_points)

    def create_refined_pattern():
        """Create a refined pattern using geometric construction."""
        # Use a combination of spiral and grid elements
        np.random.seed(42)

        # Create a base spiral
        spiral_points = []
        for i in range(16):
            t = i / 15.0 * 4 * np.pi
            r = 0.4 * (i / 15.0)
            x = 0.5 + r * np.cos(t) * 0.8
            y = 0.5 + r * np.sin(t) * 0.8
            spiral_points.append([x, y])

        # Add some randomization to break symmetry
        points = np.array(spiral_points)
        noise = np.random.normal(0, 0.01, points.shape)
        points = points + noise
        points = np.clip(points, 0, 1)
        return points

    def create_symmetry_breaking_pattern():
        """Create pattern that breaks common symmetries."""
        # Start with a regular grid but apply non-uniform transformation
        base_points = []
        for i in range(4):
            for j in range(4):
                x = 0.1 + j * 0.225
                y = 0.1 + i * 0.225
                base_points.append([x, y])

        points = np.array(base_points)

        # Apply a non-linear transformation to break symmetry
        transformed = points.copy()
        for i in range(len(transformed)):
            # Apply small non-linear deformations per point
            rx = np.random.random() * 0.01
            ry = np.random.random() * 0.01
            transformed[i][0] += rx * (points[i][0] - 0.5)
            transformed[i][1] += ry * (points[i][1] - 0.5)

        # Add noise and clip
        noise = np.random.normal(0, 0.005, transformed.shape)
        transformed = transformed + noise
        transformed = np.clip(transformed, 0, 1)
        return transformed

    def optimize_with_constraints(points, max_iter=1000):
        """Optimize points with explicit bound constraints."""
        x0 = points.flatten()
        bounds = [(0, 1) for _ in range(32)]

        # Use a hybrid approach: try different optimization methods
        methods = ['L-BFGS-B', 'SLSQP']
        best_points = points.copy()
        best_ratio = 0

        for method in methods:
            try:
                result = minimize(
                    packing_energy_objective,
                    x0,
                    method=method,
                    bounds=bounds,
                    options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-10}
                )

                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    ratio, _, _ = compute_distance_ratios(final_points)

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()

            except:
                continue

        return best_points

    def hierarchical_evolution():
        """Hierarchical optimization with progressive refinement."""
        best_ratio = 0
        best_points = None

        # Strategy 1: Golden ratio pattern
        try:
            points = create_golden_ratio_pattern()
            refined = optimize_with_constraints(points, 500)
            ratio, _, _ = compute_distance_ratios(refined)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined.copy()
        except:
            pass

        # Strategy 2: Hierarchical grid
        try:
            points = create_hierarchical_grid()
            refined = optimize_with_constraints(points, 500)
            ratio, _, _ = compute_distance_ratios(refined)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined.copy()
        except:
            pass

        # Strategy 3: Refined spiral pattern
        try:
            points = create_refined_pattern()
            refined = optimize_with_constraints(points, 500)
            ratio, _, _ = compute_distance_ratios(refined)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined.copy()
        except:
            pass

        # Strategy 4: Symmetry breaking pattern
        try:
            points = create_symmetry_breaking_pattern()
            refined = optimize_with_constraints(points, 500)
            ratio, _, _ = compute_distance_ratios(refined)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined.copy()
        except:
            pass

        # Strategy 5: Multi-start with random seeds
        if best_points is None:
            for seed_val in [42, 123, 456, 789]:
                try:
                    np.random.seed(seed_val)
                    points = np.random.rand(16, 2)
                    refined = optimize_with_constraints(points, 300)
                    ratio, _, _ = compute_distance_ratios(refined)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = refined.copy()
                except:
                    continue

        # Final refinement if we have a reasonable solution
        if best_points is not None and best_ratio > 0.15:
            try:
                # Apply one final high-precision optimization
                refined_final = optimize_with_constraints(best_points, 800)
                final_ratio, _, _ = compute_distance_ratios(refined_final)
                if final_ratio > best_ratio:
                    best_points = refined_final
            except:
                pass

        return best_points if best_points is not None else create_hierarchical_grid()

    # Main execution
    try:
        result = hierarchical_evolution()
        return result
    except:
        # Fallback to simple grid
        return create_hierarchical_grid()

# EVOLVE-BLOCK-END