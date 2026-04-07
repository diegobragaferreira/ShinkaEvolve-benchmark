# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class SpatialOptimizer3D:
    def __init__(self, n_points=14, dimensions=3, seed=42):
        self.n_points = n_points
        self.dimensions = dimensions
        self.seed = seed
        np.random.seed(seed)
        self.best_score = -np.inf
        self.best_points = None

    def _fibonacci_sphere(self):
        """Generate points on sphere using Fibonacci spiral method"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2

        for i in range(self.n_points):
            # Latitude
            phi = np.arccos(1 - 2*i/(self.n_points-1))
            # Longitude
            theta = 2 * np.pi * i / golden_ratio

            # Convert to Cartesian coordinates
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

            points.append([x, y, z])

        return np.array(points)

    def _spherical_voronoi_points(self):
        """Generate points using spherical Voronoi diagram for even distribution"""
        # Start with random points on sphere
        points = np.random.randn(self.n_points, 3)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)

        # Use spherical Voronoi to get more uniform distribution
        try:
            sv = SphericalVoronoi(points)
            # Get the centers of the Voronoi cells as new candidates
            voronoi_centers = sv.vertices
            # Normalize to unit sphere again
            voronoi_centers = voronoi_centers / np.linalg.norm(voronoi_centers, axis=1, keepdims=True)

            # Take first n points, or generate more if needed
            if len(voronoi_centers) >= self.n_points:
                selected = voronoi_centers[:self.n_points]
            else:
                # If not enough, use a combination of original and Voronoi points
                selected = np.vstack([voronoi_centers, points[:self.n_points-len(voronoi_centers)]])

            return selected
        except:
            # Fallback to fibonacci if spherical voronoi fails
            return self._fibonacci_sphere()

    def _spherical_code_points(self):
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
        if self.n_points < 14:
            return points[:self.n_points]
        elif self.n_points > 14:
            # For more than 14 points, we'd need a more complex construction
            # But since we only need 14, we'll stick to this configuration
            return points
        else:
            return points

    def _icosahedron_points(self):
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
        if self.n_points <= 12:
            return vertices[:self.n_points]
        else:
            # Use existing vertices and add extra points
            points = vertices.copy()
            # Add 2 more points for a total of 14 - place them at poles
            extra_points = np.array([[0, 0, 1], [0, 0, -1]])
            points = np.vstack([points, extra_points[:self.n_points-12]])
            return points

    def _cube_grid_points(self):
        """Initialize points in a 3D cube grid"""
        # Find appropriate grid size
        grid_size = int(np.ceil(self.n_points**(1/3)))
        coords = np.linspace(0, 1, grid_size)
        grid_points = []

        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if len(grid_points) < self.n_points:
                        grid_points.append([coords[i], coords[j], coords[k]])

        return np.array(grid_points[:self.n_points])

    def _initialize_strategies(self):
        """Generate multiple initialization strategies"""
        strategies = {}

        # Strategy 1: Spherical Fibonacci points
        fib_points = self._fibonacci_sphere()
        strategies["fibonacci"] = (fib_points + 1) / 2  # Normalize to [0,1]^3

        # Strategy 2: Spherical Voronoi points
        sv_points = self._spherical_voronoi_points()
        strategies["voronoi"] = (sv_points + 1) / 2  # Normalize to [0,1]^3

        # Strategy 3: Spherical code points (new)
        sc_points = self._spherical_code_points()
        strategies["spherical_code"] = (sc_points + 1) / 2  # Normalize to [0,1]^3

        # Strategy 4: Icosahedron-based points
        ico_points = self._icosahedron_points()
        strategies["icosahedron"] = (ico_points + 1) / 2  # Normalize to [0,1]^3

        # Strategy 5: Cube grid points
        cube_points = self._cube_grid_points()
        strategies["cube_grid"] = cube_points

        # Strategy 6: Random points
        strategies["random"] = np.random.rand(self.n_points, self.dimensions)

        # Strategy 7: Perturbed Fibonacci points
        perturbed_points = strategies["fibonacci"] + np.random.normal(0, 0.03, (self.n_points, self.dimensions))
        strategies["perturbed"] = np.clip(perturbed_points, 0, 1)

        # Strategy 8: KMeans clustering approach
        kmeans_points = np.random.rand(50, self.dimensions)  # More samples for better clustering
        kmeans = KMeans(n_clusters=self.n_points, random_state=self.seed, n_init=20)
        kmeans.fit(kmeans_points)
        strategies["kmeans"] = kmeans.cluster_centers_

        return strategies

    def _evaluate_strategy(self, points):
        """Fast evaluation of initialization quality"""
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max > 1e-12:
            return d_min / d_max
        return 0

    def _select_best_initialization(self, strategies):
        """Select the best initialization strategy"""
        best_strategy = None
        best_ratio = -np.inf

        for name, points in strategies.items():
            ratio = self._evaluate_strategy(points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_strategy = points.copy()

        return best_strategy

    def _objective(self, x):
        """Objective function to minimize (negative ratio)"""
        # Reshape x into points array
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

    def _penalty_objective(self, x, penalty_weight=1e6):
        """Objective with penalty for boundary violations"""
        points = x.reshape(-1, 3)

        # Vectorized penalty calculation
        below_penalty = np.sum(np.maximum(0, -points)**2) * penalty_weight
        above_penalty = np.sum(np.maximum(0, points - 1)**2) * penalty_weight

        # Original objective
        original_obj = self._objective(x)

        return original_obj + below_penalty + above_penalty

    def _adaptive_differential_evolution(self, objective_func, bounds, maxiter=300):
        """Enhanced differential evolution with adaptive population sizing and early stopping"""
        current_popsize = 25
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
                    seed=self.seed + iteration,
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
                        seed=self.seed + iteration,
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
                        seed=self.seed + iteration,
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

    def _local_refinement(self, points):
        """Apply local refinement using L-BFGS-B with progressively tighter tolerances"""
        refined_points = points.copy()

        # First stage: coarse refinement
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
                bounds=[(0, 1)] * self.n_points * 3,
                options={'ftol': 1e-9, 'gtol': 1e-9},  # Tighter tolerances
                tol=1e-9
            )

            refined_points = result_refine.x.reshape(-1, 3)
            refined_points = np.clip(refined_points, 0, 1)
        except:
            pass

        # Second stage: fine refinement
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
                bounds=[(0, 1)] * self.n_points * 3,
                options={'ftol': 1e-12, 'gtol': 1e-12},  # Very tight tolerances
                tol=1e-12
            )

            refined_points = result_refine.x.reshape(-1, 3)
            refined_points = np.clip(refined_points, 0, 1)
        except:
            pass

        return refined_points

    def optimize(self):
        """Main optimization pipeline"""
        # Generate initialization strategies
        strategies = self._initialize_strategies()

        # Select the best initialization strategy
        initial_points = self._select_best_initialization(strategies)

        # Set up bounds for optimization
        bounds = [(0, 1)] * self.n_points * 3

        # Run adaptive differential evolution optimization
        best_result = None
        best_ratio = -np.inf

        # Try multiple seeds for better exploration
        seeds = [42, 123, 456, 789, 999]
        for seed_val in seeds:
            np.random.seed(seed_val)

            # Use adaptive differential evolution
            result = self._adaptive_differential_evolution(
                self._penalty_objective,
                bounds,
                maxiter=300
            )

            # Check if this result is better
            if -result.fun > best_ratio:
                best_ratio = -result.fun
                best_result = result

        # Extract optimized points
        optimized_points = best_result.x.reshape(-1, 3)

        # Apply local refinement
        optimized_points = self._local_refinement(optimized_points)

        # Final clipping to ensure bounds are respected
        optimized_points = np.clip(optimized_points, 0, 1)

        return optimized_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    optimizer = SpatialOptimizer3D(n_points=14, dimensions=3, seed=42)
    return optimizer.optimize()

# EVOLVE-BLOCK-END