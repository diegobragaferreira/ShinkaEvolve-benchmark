# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize, basinhopping
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

    def golden_ratio_sphere_sampling(n):
        """Generate points on a unit sphere using golden ratio method for optimal distribution"""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # golden ratio

        # For 14 points, we'll use a modified approach
        # Generate points using Fibonacci-like distribution with golden ratio
        for i in range(n):
            # Use golden angle in spherical coordinates
            theta = np.arccos(1 - 2*(i/(n-1)))  # Polar angle
            phi_angle = i * 2.399963229728653  # Golden angle in radians (2π/φ)

            # Convert to Cartesian coordinates
            x = np.sin(theta) * np.cos(phi_angle)
            y = np.sin(theta) * np.sin(phi_angle)
            z = np.cos(theta)

            points.append([x, y, z])

        return np.array(points)

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

    def initialize_points():
        """Initialize points using enhanced spherical sampling with multiple strategies"""
        # Strategy 1: Golden ratio sphere sampling (new method)
        points1 = golden_ratio_sphere_sampling(14)

        # Strategy 2: Fibonacci sphere sampling
        points2 = fibonacci_sphere_sampling(14)

        # Strategy 3: Random uniform initialization
        np.random.seed(42)
        points3 = np.random.rand(14, 3)

        # Strategy 4: Perturbed Fibonacci
        np.random.seed(42)
        points4 = points2 + np.random.normal(0, 0.05, points2.shape)

        # Strategy 5: Perturbed golden ratio
        np.random.seed(43)
        points5 = points1 + np.random.normal(0, 0.05, points1.shape)

        # Normalize all spherical strategies to unit sphere for consistency
        for points in [points1, points2, points4, points5]:
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points /= norms

        # Scale appropriately
        points1 *= 0.8
        points2 *= 0.8
        points4 *= 0.8
        points5 *= 0.8

        # Transform to [0,1]^3 space
        points1 = (points1 + 1) / 2
        points2 = (points2 + 1) / 2
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

        # Try different population sizes and mutation strategies for robustness
        strategies = [
            {'popsize': 15, 'mutation': (0.5, 1.0), 'recombination': 0.7, 'maxiter': 300},
            {'popsize': 20, 'mutation': (0.7, 1.0), 'recombination': 0.8, 'maxiter': 250},
            {'popsize': 25, 'mutation': (0.8, 1.0), 'recombination': 0.9, 'maxiter': 200}
        ]

        best_points = initial_points.reshape((14, 3))
        best_ratio = -np.inf

        for strategy in strategies:
            try:
                result = differential_evolution(
                    penalty_objective,
                    bounds,
                    seed=42,
                    maxiter=strategy['maxiter'],
                    popsize=strategy['popsize'],
                    mutation=strategy['mutation'],
                    recombination=strategy['recombination'],
                    tol=1e-8,
                    callback=None
                )

                # Evaluate result
                points = result.x.reshape((14, 3))
                distances = pdist(points)
                min_dist = np.min(distances)
                max_dist = np.max(distances)

                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = points

            except Exception:
                continue

        # Additional adaptive optimization: run with increasing population size if necessary
        if best_ratio < 0.3:  # If we haven't achieved good results yet
            try:
                # Try with larger population size
                result = differential_evolution(
                    penalty_objective,
                    bounds,
                    seed=43,
                    maxiter=200,
                    popsize=30,
                    mutation=(0.8, 1.0),
                    recombination=0.9,
                    tol=1e-8,
                    callback=None
                )

                # Evaluate result
                points = result.x.reshape((14, 3))
                distances = pdist(points)
                min_dist = np.min(distances)
                max_dist = np.max(distances)

                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = points
            except Exception:
                pass

        return best_points

    def advanced_local_refinement(points, maxiter=1000):
        """Apply advanced local optimization refinement with multiple techniques"""
        best_points = points.copy()
        best_ratio = -np.inf
        
        # Try Basin-hopping first - very effective for this kind of problem
        try:
            minimizer_kwargs = {"method": "L-BFGS-B", "bounds": [(0, 1) for _ in range(42)]}
            result_bh = basinhopping(
                objective_function,
                points.flatten(),
                niter=10,
                T=1.0,
                stepsize=0.1,
                minimizer_kwargs=minimizer_kwargs,
                seed=42
            )

            if result_bh.success:
                refined_points = result_bh.x.reshape(-1, 3)
                distances = pdist(refined_points)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = refined_points.copy()
        except Exception as e:
            pass

        # Apply L-BFGS-B refinement with multiple tolerance levels
        try:
            # Initial coarse refinement
            result_coarse = minimize(
                objective_function,
                best_points.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(42)],
                options={'ftol': 1e-6, 'gtol': 1e-6, 'maxiter': maxiter//3}
            )
            
            if result_coarse.success:
                refined_points_coarse = result_coarse.x.reshape(-1, 3)
                
                # Fine refinement with tighter tolerances
                result_fine = minimize(
                    objective_function,
                    refined_points_coarse.flatten(),
                    method='L-BFGS-B',
                    bounds=[(0, 1) for _ in range(42)],
                    options={'ftol': 1e-9, 'gtol': 1e-9, 'maxiter': maxiter//3}
                )
                
                if result_fine.success:
                    refined_points_fine = result_fine.x.reshape(-1, 3)
                    
                    # Evaluate fine refinement
                    distances = pdist(refined_points_fine)
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    if max_dist > 0:
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = refined_points_fine.copy()
                            
        except Exception as e:
            pass
            
        # Final refinement with even stricter tolerances if we haven't found a better solution
        if best_ratio < 0.3:  # Only proceed if needed
            try:
                result_final = minimize(
                    objective_function,
                    best_points.flatten(),
                    method='L-BFGS-B',
                    bounds=[(0, 1) for _ in range(42)],
                    options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': maxiter//3}
                )
                
                if result_final.success:
                    refined_points_final = result_final.x.reshape(-1, 3)
                    
                    # Evaluate final refinement
                    distances = pdist(refined_points_final)
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    if max_dist > 0:
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = refined_points_final.copy()
            except Exception as e:
                pass

        return best_points

    def validate_and_correct_bounds(points):
        """Ensure all points are within [0,1]^3 bounds"""
        corrected_points = np.clip(points, 0, 1)
        return corrected_points

    # Initialize with enhanced spherical configuration
    initial_points = initialize_points()

    # Phase 1: Global optimization with adaptive differential evolution
    global_optimized = adaptive_differential_evolution(initial_points)

    # Phase 2: Advanced local refinement
    local_optimized = advanced_local_refinement(global_optimized, maxiter=500)

    # Final validation
    final_points = validate_and_correct_bounds(local_optimized)

    return final_points

# EVOLVE-BLOCK-END