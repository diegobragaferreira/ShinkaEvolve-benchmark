# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import SphericalVoronoi
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def objective_function(x):
        # Reshape flat array back to 14x3 points
        points = x.reshape((14, 3))

        # Compute pairwise distances efficiently
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return -np.inf

        # Return negative because we want to maximize the ratio
        return -min_dist / max_dist

    def penalty_objective(x, penalty_weight=1e6):
        """Objective function with penalty for boundary violations"""
        points = x.reshape((14, 3))

        # Calculate base objective
        ratio = -objective_function(x)
        base_obj = ratio

        # Add penalty for points outside [0,1]^3 bounds
        penalty = 0
        for i in range(14):
            for j in range(3):
                coord = points[i, j]
                if coord < 0:
                    penalty += penalty_weight * (0 - coord) ** 2
                elif coord > 1:
                    penalty += penalty_weight * (coord - 1) ** 2

        return base_obj + penalty

    def fibonacci_sphere_sampling(n):
        """Generate points on a unit sphere using Fibonacci spiral method with better distribution"""
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle

        for i in range(n):
            y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def spherical_voronoi_initialization():
        """Initialize points using spherical Voronoi diagram approach for better spreading"""
        # Start with random points on sphere
        np.random.seed(42)
        points = np.random.randn(14, 3)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)

        # Use spherical Voronoi to get more uniform distribution
        try:
            sv = SphericalVoronoi(points)
            # Get the centers of the Voronoi cells as new candidates
            voronoi_centers = sv.vertices
            # Normalize to unit sphere again
            voronoi_centers = voronoi_centers / np.linalg.norm(voronoi_centers, axis=1, keepdims=True)

            # Take first 14 points, or generate more if needed
            if len(voronoi_centers) >= 14:
                selected = voronoi_centers[:14]
            else:
                # If not enough, use a combination of original and Voronoi points
                selected = np.vstack([voronoi_centers, points[:14-len(voronoi_centers)]])

            points = selected
        except:
            # Fallback to fibonacci if spherical voronoi fails
            points = fibonacci_sphere_sampling(14)

        # Add slight perturbations to break symmetries
        np.random.seed(42)
        noise = np.random.normal(0, 0.05, points.shape)
        points += noise

        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms

        # Scale and transform to [0,1]^3
        points *= 0.8
        points = (points + 1) / 2

        return points

    def enhanced_spherical_initialization():
        """Enhanced spherical initialization using multiple approaches"""
        # Strategy 1: True Fibonacci sphere sampling
        points1 = fibonacci_sphere_sampling(14)

        # Strategy 2: Spherical Voronoi based initialization
        points2 = spherical_voronoi_initialization()

        # Strategy 3: Random points on sphere
        np.random.seed(42)
        points3 = np.random.randn(14, 3)
        points3 = points3 / np.linalg.norm(points3, axis=1, keepdims=True)
        points3 *= 0.8
        points3 = (points3 + 1) / 2

        # Strategy 4: Hybrid approach - perturbed Fibonacci with better spread
        np.random.seed(42)
        points4 = points1 + np.random.normal(0, 0.1, points1.shape)
        norms = np.linalg.norm(points4, axis=1, keepdims=True)
        points4 = points4 / norms
        points4 *= 0.8
        points4 = (points4 + 1) / 2

        # Strategy 5: Another Voronoi approach with different seed
        np.random.seed(123)
        points5 = np.random.randn(14, 3)
        points5 = points5 / np.linalg.norm(points5, axis=1, keepdims=True)
        try:
            sv = SphericalVoronoi(points5)
            voronoi_centers = sv.vertices
            voronoi_centers = voronoi_centers / np.linalg.norm(voronoi_centers, axis=1, keepdims=True)

            if len(voronoi_centers) >= 14:
                points5 = voronoi_centers[:14]
            else:
                points5 = np.vstack([voronoi_centers, points5[:14-len(voronoi_centers)]])
        except:
            points5 = points1

        points5 *= 0.8
        points5 = (points5 + 1) / 2

        # Normalize all strategies to unit sphere
        for points in [points1, points4, points5]:
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points /= norms

        # Scale appropriately
        points1 *= 0.8
        points4 *= 0.8
        points5 *= 0.8

        # Transform to [0,1]^3 space
        points1 = (points1 + 1) / 2
        points4 = (points4 + 1) / 2
        points5 = (points5 + 1) / 2

        # Evaluate all initializations
        eval_points = [points1, points2, points3, points4, points5]
        ratios = []

        for points in eval_points:
            distances = pdist(points)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist > 0:
                ratio = min_dist / max_dist
            else:
                ratio = 0
            ratios.append(ratio)

        # Select the best initialization
        best_idx = np.argmax(ratios)
        return eval_points[best_idx].flatten()

    def adaptive_differential_evolution(initial_points):
        """Perform differential evolution with adaptive population sizing and multiple strategies"""
        bounds = [(0, 1)] * 14 * 3

        # Start with larger population for better exploration
        popsize = 20
        maxiter = 200
        mutation = (0.5, 1.0)
        recombination = 0.7
        tol = 1e-8

        try:
            result = differential_evolution(
                penalty_objective,
                bounds,
                seed=42,
                maxiter=maxiter,
                popsize=popsize,
                mutation=mutation,
                recombination=recombination,
                tol=tol,
                callback=None
            )

            # Evaluate result
            points = result.x.reshape((14, 3))
            distances = pdist(points)
            min_dist = np.min(distances)
            max_dist = np.max(distances)

            if max_dist > 0:
                ratio = min_dist / max_dist
            else:
                ratio = 0

            return points

        except Exception:
            # Fallback to simpler approach
            return initial_points.reshape((14, 3))

    def improved_local_refinement(points):
        """Apply improved local optimization refinement using multiple techniques"""
        best_points = points.copy()
        best_ratio = -np.inf

        # Try multiple refinement strategies
        strategies = [
            {'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 1000},
            {'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1500},
            {'ftol': 1e-8, 'gtol': 1e-8, 'maxiter': 800}
        ]

        for i, strategy in enumerate(strategies):
            try:
                # Apply random perturbation for diversity
                np.random.seed(42 + i)
                perturbed = points + np.random.normal(0, 0.005, points.shape)
                perturbed = np.clip(perturbed, 0, 1)

                result = minimize(
                    objective_function,
                    perturbed.flatten(),
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * 14 * 3,
                    options=strategy,
                    tol=strategy['ftol']
                )

                refined_points = result.x.reshape((14, 3))
                refined_points = np.clip(refined_points, 0, 1)

                # Evaluate refined points
                distances = pdist(refined_points)
                min_dist = np.min(distances)
                max_dist = np.max(distances)

                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = refined_points

            except Exception:
                continue

        return best_points

    def validate_and_correct_bounds(points):
        """Ensure all points are within [0,1]^3 bounds"""
        corrected_points = np.clip(points, 0, 1)
        return corrected_points

    # Initialize with enhanced spherical configuration
    initial_points = initialize_points()

    # Phase 1: Global optimization with adaptive differential evolution
    global_optimized = adaptive_differential_evolution(initial_points)

    # Phase 2: Local refinement with improved technique
    local_optimized = improved_local_refinement(global_optimized)

    # Phase 3: Additional local refinement with different settings
    final_local_optimized = improved_local_refinement(local_optimized)

    # Final validation
    final_points = validate_and_correct_bounds(final_local_optimized)

    return final_points

# EVOLVE-BLOCK-END