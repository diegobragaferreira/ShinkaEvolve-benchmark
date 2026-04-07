# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

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

        # Handle edge cases
        if len(distances) == 0:
            return -np.inf

        # Remove any NaN or infinite values
        distances = distances[np.isfinite(distances)]

        if len(distances) == 0:
            return -np.inf

        # Calculate min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Return negative ratio to maximize (since we're minimizing)
        if d_max <= 0:
            return -np.inf
        return -(d_min / d_max)

    def penalty_objective(x, penalty_weight=1e6):
        """Objective with penalty for boundary violations - vectorized version"""
        points = x.reshape(-1, 3)

        # Vectorized penalty calculation
        below_penalty = np.sum(np.maximum(0, -points)**2) * penalty_weight
        above_penalty = np.sum(np.maximum(0, points - 1)**2) * penalty_weight

        # Original objective
        original_obj = objective(x)

        return original_obj + below_penalty + above_penalty

    def spherical_voronoi_points(n):
        """Generate points using spherical Voronoi diagram for even distribution"""
        # Start with random points on sphere
        np.random.seed(42)
        points = np.random.randn(n, 3)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)

        # Use spherical Voronoi to get more uniform distribution
        try:
            sv = SphericalVoronoi(points)
            # Get the centers of the Voronoi cells as new candidates
            voronoi_centers = sv.vertices
            # Normalize to unit sphere again
            voronoi_centers = voronoi_centers / np.linalg.norm(voronoi_centers, axis=1, keepdims=True)

            # Take first n points, or generate more if needed
            if len(voronoi_centers) >= n:
                selected = voronoi_centers[:n]
            else:
                # If not enough, use a combination of original and Voronoi points
                selected = np.vstack([voronoi_centers, points[:n-len(voronoi_centers)]])

            return selected
        except:
            # Fallback to fibonacci if spherical voronoi fails
            return spherical_fibonacci_points(n)

    def spherical_fibonacci_points(n):
        """Generate points on sphere using Fibonacci spiral method"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2

        for i in range(n):
            # Latitude
            phi = np.arccos(1 - 2*i/(n-1))
            # Longitude
            theta = 2 * np.pi * i / golden_ratio

            # Convert to Cartesian coordinates
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

            points.append([x, y, z])

        return np.array(points)

    def _spherical_code_points(n):
        """Generate points using known spherical code constructions for 14 points"""
        # Use the vertices of a specific polyhedron that provides good distribution
        # For 14 points, we can use a construction based on the snub cube or similar
        # This is a mathematically-derived configuration that tends to work well

        # Known good configuration based on mathematical constructions for 14 points
        # These coordinates are normalized to unit sphere
        points = np.array([
            # 8 vertices of a cube (scaled appropriately)
            [ 1,  1,  1], [ 1,  1, -1], [ 1, -1,  1], [ 1, -1, -1],
            [-1,  1,  1], [-1,  1, -1], [-1, -1,  1], [-1, -1, -1],
            # 6 additional points placed at strategic locations
            [ 0,  0,  1], [ 0,  0, -1]
        ])

        # Normalize to unit sphere
        points = points / np.linalg.norm(points, axis=1, keepdims=True)

        # If we need fewer points, take first n
        if n < 14:
            return points[:n]
        elif n > 14:
            # For more than 14 points, we'd need a more complex construction
            # But since we only need 14, we'll stick to this configuration
            return points
        else:
            return points

    def _icosahedron_points(n):
        """Generate points using icosahedron vertices for better spherical distribution"""
        # Vertices of a regular icosahedron (normalized)
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = np.array([
            [-1,  phi,  0],
            [ 1,  phi,  0],
            [-1, -phi,  0],
            [ 1, -phi,  0],
            [ 0, -1,  phi],
            [ 0,  1,  phi],
            [ 0, -1, -phi],
            [ 0,  1, -phi],
            [ phi,  0, -1],
            [ phi,  0,  1],
            [-phi,  0, -1],
            [-phi,  0,  1]
        ])

        # Normalize vertices to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

        # For 14 points, we can use the 12 vertices plus 2 more strategically placed
        if n <= 12:
            return vertices[:n]
        else:
            # Use existing vertices and add extra points
            points = vertices.copy()
            # Add 2 more points for a total of 14 - place them at poles
            extra_points = np.array([[0, 0, 1], [0, 0, -1]])
            points = np.vstack([points, extra_points[:n-12]])
            return points

    def initialize_cube_grid_points(n_points):
        """Initialize points in a 3D cube grid"""
        # Find appropriate grid size
        grid_size = int(np.ceil(n_points**(1/3)))
        coords = np.linspace(0, 1, grid_size)
        grid_points = []

        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if len(grid_points) < n_points:
                        grid_points.append([coords[i], coords[j], coords[k]])

        return np.array(grid_points[:n_points])

    def evaluate_initialization(points):
        """Fast evaluation of initialization quality"""
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max > 1e-12:
            return d_min / d_max
        return 0

    def adaptive_differential_evolution(objective_func, bounds, initial_popsize=25, maxiter=300):
        """Enhanced differential evolution with adaptive population sizing and early stopping"""
        current_popsize = initial_popsize
        prev_best = -np.inf
        stagnation_count = 0
        improvement_threshold = 1e-8
        min_improvement = 1e-12

        # Track improvement for early stopping
        recent_improvements = []

        for iteration in range(maxiter // 10):  # Reduced iterations per batch
            # Adjust population size based on convergence
            if stagnation_count > 3 and current_popsize < 35:
                current_popsize = min(current_popsize + 5, 35)

            # Run differential evolution with current parameters
            try:
                result = differential_evolution(
                    objective_func,
                    bounds,
                    seed=42 + iteration,
                    maxiter=10,  # Fewer iterations per batch
                    popsize=current_popsize,
                    tol=1e-9,   # Tighter tolerance
                    mutation=(0.7, 1.0),  # More aggressive exploration
                    recombination=0.85,   # Higher recombination for better exploration
                    disp=False
                )
            except:
                # Fall back to smaller population if needed
                try:
                    result = differential_evolution(
                        objective_func,
                        bounds,
                        seed=42 + iteration,
                        maxiter=10,
                        popsize=max(5, current_popsize - 5),
                        tol=1e-9,
                        mutation=(0.7, 1.0),
                        recombination=0.85,
                        disp=False
                    )
                except:
                    # Last resort - use basic differential evolution
                    result = differential_evolution(
                        objective_func,
                        bounds,
                        seed=42 + iteration,
                        maxiter=10,
                        popsize=10,
                        tol=1e-9,
                        mutation=(0.7, 1.0),
                        recombination=0.7,
                        disp=False
                    )

            # Check for improvement
            current_best = -result.fun
            improvement = current_best - prev_best

            recent_improvements.append(improvement)
            if len(recent_improvements) > 5:
                recent_improvements.pop(0)

            # Early stopping if improvement is minimal
            if len(recent_improvements) == 5 and all(abs(impr) < min_improvement for impr in recent_improvements):
                break

            if improvement > improvement_threshold:
                stagnation_count = 0
            else:
                stagnation_count += 1

            prev_best = current_best

        return result

    # Generate comprehensive initialization strategies
    strategies = []

    # Strategy 1: Spherical Voronoi points (improved)
    voronoi_points = spherical_voronoi_points(14)
    voronoi_points = (voronoi_points + 1) / 2  # Normalize to [0,1]^3
    strategies.append(("voronoi", voronoi_points))

    # Strategy 2: Spherical Fibonacci points
    fib_points = spherical_fibonacci_points(14)
    fib_points = (fib_points + 1) / 2  # Normalize to [0,1]^3
    strategies.append(("fibonacci", fib_points))

    # Strategy 3: Spherical code points (new approach)
    sc_points = _spherical_code_points(14)
    sc_points = (sc_points + 1) / 2  # Normalize to [0,1]^3
    strategies.append(("spherical_code", sc_points))

    # Strategy 4: Icosahedron-based points
    ico_points = _icosahedron_points(14)
    ico_points = (ico_points + 1) / 2  # Normalize to [0,1]^3
    strategies.append(("icosahedron", ico_points))

    # Strategy 5: Cube grid points
    cube_points = initialize_cube_grid_points(14)
    strategies.append(("cube_grid", cube_points))

    # Strategy 6: Random points
    np.random.seed(42)
    random_points = np.random.rand(14, 3)
    strategies.append(("random", random_points))

    # Strategy 7: Perturbed spherical points
    np.random.seed(42)
    perturbed_points = fib_points + np.random.normal(0, 0.03, (14, 3))
    perturbed_points = np.clip(perturbed_points, 0, 1)
    strategies.append(("perturbed", perturbed_points))

    # Strategy 8: KMeans clustering approach with more samples
    np.random.seed(42)
    kmeans_points = np.random.rand(50, 3)  # More samples for better clustering
    kmeans = KMeans(n_clusters=14, random_state=42, n_init=20)
    kmeans.fit(kmeans_points)
    kmeans_centers = kmeans.cluster_centers_
    strategies.append(("kmeans", kmeans_centers))

    # Strategy 9: Perturbed icosahedron points
    np.random.seed(42)
    perturbed_ico_points = (ico_points + 1) / 2 + np.random.normal(0, 0.02, (14, 3))
    perturbed_ico_points = np.clip(perturbed_ico_points, 0, 1)
    strategies.append(("perturbed_ico", perturbed_ico_points))

    # Strategy 10: Perturbed spherical code points
    np.random.seed(42)
    perturbed_sc_points = (sc_points + 1) / 2 + np.random.normal(0, 0.03, (14, 3))
    perturbed_sc_points = np.clip(perturbed_sc_points, 0, 1)
    strategies.append(("perturbed_sc", perturbed_sc_points))

    # Evaluate all strategies and select the best
    best_initialization = None
    best_ratio = -np.inf

    for name, points in strategies:
        ratio = evaluate_initialization(points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_initialization = points.copy()

    # Use the best initialization as starting point
    x0 = best_initialization.flatten()

    # Bounds for each coordinate: [0, 1] for all 14 points × 3 coordinates
    bounds = [(0, 1)] * 14 * 3

    # Run adaptive differential evolution optimization
    best_result = None
    best_ratio = -np.inf

    # Try 5 different random seeds for better exploration with exponential backoff
    seeds = [42, 123, 456, 789, 999]
    for i, seed_val in enumerate(seeds):
        np.random.seed(seed_val)

        # Use adaptive differential evolution
        result = adaptive_differential_evolution(
            penalty_objective,
            bounds,
            initial_popsize=25,  # Increased from 20
            maxiter=300  # Increased from 200
        )

        # Check if this result is better
        if -result.fun > best_ratio:
            best_ratio = -result.fun
            best_result = result

    # Extract optimized points
    optimized_points = best_result.x.reshape(-1, 3)

    # Apply adaptive tolerance refinement with improvement-based tightening
    def adaptive_tolerance_refinement(points):
        """Apply local refinement using L-BFGS-B with adaptive tolerance adjustment"""
        refined_points = points.copy()
        previous_ratio = -np.inf  # Negative because we minimize negative ratio
        improvement_threshold = 1e-10
        max_iterations = 5

        # Start with moderate tolerances for faster initial convergence
        ftol = 1e-6
        gtol = 1e-6
        current_tol = 1e-6

        for iteration in range(max_iterations):
            try:
                x0_refine = refined_points.flatten()

                def obj_for_adaptive(x):
                    points_refined = x.reshape(-1, 3)
                    distances = pdist(points_refined)

                    if len(distances) == 0:
                        return -np.inf

                    d_min = np.min(distances)
                    d_max = np.max(distances)

                    if d_max > 1e-12:
                        return -(d_min / d_max)
                    else:
                        return -np.inf

                # Apply refinement with current tolerances
                result_refine = minimize(
                    obj_for_adaptive,
                    x0_refine,
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * 42,
                    options={'ftol': ftol, 'gtol': gtol},
                    tol=current_tol
                )

                if result_refine.success:
                    new_points = result_refine.x.reshape(-1, 3)
                    new_points = np.clip(new_points, 0, 1)

                    # Calculate new ratio
                    distances_new = pdist(new_points)
                    if len(distances_new) > 0:
                        d_min_new = np.min(distances_new)
                        d_max_new = np.max(distances_new)
                        if d_max_new > 1e-12:
                            new_ratio = d_min_new / d_max_new

                            # Check if there was significant improvement
                            improvement = new_ratio - previous_ratio
                            if improvement > improvement_threshold:
                                # Tighten tolerances for next iteration
                                ftol = max(ftol / 10, 1e-12)
                                gtol = max(gtol / 10, 1e-12)
                                current_tol = max(current_tol / 10, 1e-12)
                                previous_ratio = new_ratio
                            else:
                                # If no significant improvement, keep current tolerances
                                pass

                            refined_points = new_points
                else:
                    # If optimization failed, continue with existing points
                    break

            except Exception:
                # If anything goes wrong, break to avoid infinite loops
                break

        return refined_points

    # Apply adaptive tolerance refinement
    optimized_points = adaptive_tolerance_refinement(optimized_points)

    # Apply final multi-stage refinement for maximum quality
    def multi_stage_refinement(points):
        """Apply multiple rounds of refinement with different approaches"""
        refined_points = points.copy()
        
        # Stage 1: Coarse local refinement with moderate tolerances
        try:
            x0_refine = refined_points.flatten()
            def obj_for_lbfgs(x):
                points_refined = x.reshape(-1, 3)
                distances = pdist(points_refined)
                
                if len(distances) == 0:
                    return -np.inf
                    
                d_min = np.min(distances)
                d_max = np.max(distances)
                
                if d_max > 1e-12:
                    return -(d_min / d_max)
                else:
                    return -np.inf
                    
            result_refine = minimize(
                obj_for_lbfgs,
                x0_refine,
                method='L-BFGS-B',
                bounds=[(0, 1)] * 42,
                options={'ftol': 1e-9, 'gtol': 1e-9},  # Tighter tolerances
                tol=1e-9
            )
            
            refined_points = result_refine.x.reshape(-1, 3)
            refined_points = np.clip(refined_points, 0, 1)
        except:
            pass
            
        # Stage 2: Fine local refinement with very tight tolerances
        try:
            x0_refine = refined_points.flatten()
            def obj_for_lbfgs_fine(x):
                points_refined = x.reshape(-1, 3)
                distances = pdist(points_refined)
                
                if len(distances) == 0:
                    return -np.inf
                    
                d_min = np.min(distances)
                d_max = np.max(distances)
                
                if d_max > 1e-12:
                    return -(d_min / d_max)
                else:
                    return -np.inf
                    
            result_refine = minimize(
                obj_for_lbfgs_fine,
                x0_refine,
                method='L-BFGS-B',
                bounds=[(0, 1)] * 42,
                options={'ftol': 1e-12, 'gtol': 1e-12},  # Very tight tolerances
                tol=1e-12
            )
            
            refined_points = result_refine.x.reshape(-1, 3)
            refined_points = np.clip(refined_points, 0, 1)
        except:
            pass
            
        return refined_points
    
    # Apply final multi-stage refinement
    optimized_points = multi_stage_refinement(optimized_points)

    # Final clipping to ensure bounds are respected
    optimized_points = np.clip(optimized_points, 0, 1)

    return optimized_points

# EVOLVE-BLOCK-END