# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import SphericalVoronoi
import warnings
warnings.filterwarnings('ignore')


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def generate_fibonacci_points_on_sphere(n):
        """Generate points using Fibonacci-like distribution on unit sphere"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            # Modified Fibonacci method for better spread
            theta = np.arccos(1 - 2*(i/(n-1)))
            phi = i * 2 * np.pi / golden_ratio
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            points.append([x, y, z])
        return np.array(points)

    def normalize_to_unit_sphere(points):
        """Normalize points to lie exactly on unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return points / norms

    def compute_voronoi_uniformity(points):
        """Compute a measure of how uniform the spherical Voronoi cells are"""
        try:
            sv = SphericalVoronoi(points)
            areas = sv.voronoi_cell_areas()
            # Return coefficient of variation of cell areas (lower is more uniform)
            if len(areas) > 0:
                mean_area = np.mean(areas)
                if mean_area > 0:
                    cv = np.std(areas) / mean_area
                    return cv
            return 1.0
        except:
            return 1.0

    def energy_objective(points):
        """Energy-based objective that encourages uniform distribution"""
        # Convert to unit sphere if needed
        points = normalize_to_unit_sphere(points)

        # Compute pairwise distances
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)

        # Use inverse distance squared as energy (repulsion-like)
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            energy_matrix = 1.0 / (distances**2 + 1e-12)

        # Set diagonal to zero
        np.fill_diagonal(energy_matrix, 0)

        # Total energy (sum of all repulsive interactions)
        total_energy = np.sum(energy_matrix)

        # Also consider the minimum distance
        min_dist = np.min(distances)

        # Combine energy and minimum distance (both should be maximized)
        # The energy function pushes points apart, minimizing overlap
        # The minimum distance gives us the actual metric we want to optimize

        return -min_dist  # We want to maximize min_dist, so minimize -min_dist

    def voronoi_based_objective(points):
        """Objective that uses Voronoi uniformity along with distance metrics"""
        points = normalize_to_unit_sphere(points)

        # Compute pairwise distances
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Add Voronoi-based uniformity measure
        uniformity = compute_voronoi_uniformity(points)

        # Prefer configurations that are both well-separated and uniform
        if max_dist > 0:
            ratio = min_dist / max_dist
        else:
            ratio = 0

        # Weight the ratio by uniformity measure
        # Uniform distributions tend to have better ratios
        weighted_ratio = ratio * (1.0 - uniformity * 0.5)

        return -weighted_ratio  # Minimize negative to maximize ratio

    def local_improvement_step(points, max_iter=50):
        """Apply local optimization to refine the configuration"""
        # Use scipy minimize with L-BFGS-B for local refinement
        points_flat = points.flatten()

        def obj_func(x):
            pts = x.reshape(-1, 3)
            # Normalize to unit sphere
            pts = normalize_to_unit_sphere(pts)

            # Compute distances
            distances = cdist(pts, pts)
            np.fill_diagonal(distances, np.inf)

            min_dist = np.min(distances)
            return -min_dist  # Maximize minimum distance

        def constraint_sphere(x):
            points_temp = x.reshape(-1, 3)
            norms = np.linalg.norm(points_temp, axis=1)
            return 1 - norms  # Should be >= 0

        try:
            cons = [{'type': 'ineq', 'fun': constraint_sphere}]
            result = minimize(obj_func, points_flat, method='L-BFGS-B',
                            constraints=cons, options={'maxiter': max_iter, 'ftol': 1e-10})
            if result.success:
                return result.x.reshape(-1, 3)
        except:
            pass
        return points

    def run_multiple_starts():
        """Run optimization with multiple starting configurations"""
        best_points = None
        best_ratio = 0

        # Different starting configurations
        configs = []

        # 1. Fibonacci-like spiral
        configs.append(generate_fibonacci_points_on_sphere(14))

        # 2. Perturbed Fibonacci
        fib_points = generate_fibonacci_points_on_sphere(14)
        np.random.seed(42)
        perturbation = np.random.normal(0, 0.03, fib_points.shape)
        configs.append(normalize_to_unit_sphere(fib_points + perturbation))

        # 3. Another Fibonacci variant with different seed
        np.random.seed(100)
        fib_points2 = generate_fibonacci_points_on_sphere(14)
        configs.append(fib_points2)

        # 4. Random distribution on sphere
        np.random.seed(200)
        random_points = np.random.randn(14, 3)
        configs.append(normalize_to_unit_sphere(random_points))

        # 5. Another random distribution
        np.random.seed(300)
        random_points2 = np.random.randn(14, 3)
        configs.append(normalize_to_unit_sphere(random_points2))

        # 6. Slightly perturbed Fibonacci again
        np.random.seed(400)
        perturbed_fib = fib_points + np.random.normal(0, 0.02, fib_points.shape)
        configs.append(normalize_to_unit_sphere(perturbed_fib))

        for i, initial_points in enumerate(configs):
            try:
                # Apply multiple rounds of optimization
                current_points = initial_points.copy()

                # Apply local improvements
                current_points = local_improvement_step(current_points, max_iter=100)

                # Do one more round with improved objective
                try:
                    # Try the Voronoi-based objective for additional refinement
                    def obj_func_voronoi(x):
                        pts = x.reshape(-1, 3)
                        pts = normalize_to_unit_sphere(pts)
                        distances = cdist(pts, pts)
                        np.fill_diagonal(distances, np.inf)
                        min_dist = np.min(distances)
                        max_dist = np.max(distances)
                        if max_dist > 0:
                            ratio = min_dist / max_dist
                            return -ratio  # Maximize ratio
                        return 0

                    def constraint_sphere(x):
                        points_temp = x.reshape(-1, 3)
                        norms = np.linalg.norm(points_temp, axis=1)
                        return 1 - norms

                    cons = [{'type': 'ineq', 'fun': constraint_sphere}]
                    result = minimize(obj_func_voronoi, current_points.flatten(),
                                    method='L-BFGS-B', constraints=cons,
                                    options={'maxiter': 50, 'ftol': 1e-10})

                    if result.success:
                        refined_points = result.x.reshape(-1, 3)
                        refined_points = normalize_to_unit_sphere(refined_points)
                        current_points = refined_points
                except:
                    pass

                # Final evaluation
                distances = cdist(current_points, current_points)
                np.fill_diagonal(distances, np.inf)
                min_dist = np.min(distances)
                max_dist = np.max(distances)

                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = current_points.copy()

            except Exception as e:
                continue

        return best_points if best_points is not None else generate_fibonacci_points_on_sphere(14)

    # Execute the main optimization procedure
    return run_multiple_starts()

# EVOLVE-BLOCK-END