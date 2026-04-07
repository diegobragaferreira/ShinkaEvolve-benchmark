# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class AdaptiveSpatialOptimizer:
    def __init__(self, n_points=14, dimensions=3, seed=42):
        self.n_points = n_points
        self.dimensions = dimensions
        self.seed = seed
        np.random.seed(seed)

    def _initialize_spherical_points(self):
        """Initialize points on a unit sphere using Fibonacci spiral method"""
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # Golden angle

        for i in range(self.n_points):
            y = 1 - (i / float(self.n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def _initialize_icosahedron_points(self):
        """Initialize points using icosahedron vertices for better spherical distribution"""
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

        # For 14 points, we can use the 12 vertices plus 2 additional points
        if self.n_points <= 12:
            return vertices[:self.n_points]
        else:
            # Use existing vertices and add 2 more points
            points = vertices.copy()
            extra_points = np.array([[0, 0, 1], [0, 0, -1]])
            points = np.vstack([points, extra_points[:self.n_points-12]])
            return points

    def _initialize_cube_grid_points(self):
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

    def _initialize_spherical_voronoi_points(self):
        """Initialize points using spherical Voronoi diagram for even distribution"""
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
            return self._initialize_spherical_points()

    def _evaluate_initialization(self, points):
        """Fast evaluation of initialization quality"""
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max > 1e-12:
            return d_min / d_max
        return 0

    def _generate_initial_strategies(self):
        """Generate multiple initialization strategies"""
        strategies = []

        # Strategy 1: Spherical Fibonacci points
        fib_points = self._initialize_spherical_points()
        fib_points = (fib_points + 1) / 2  # Normalize to [0,1]^3
        strategies.append(("fibonacci", fib_points.copy()))

        # Strategy 2: Icosahedron-based points
        ico_points = self._initialize_icosahedron_points()
        ico_points = (ico_points + 1) / 2  # Normalize to [0,1]^3
        strategies.append(("icosahedron", ico_points.copy()))

        # Strategy 3: Spherical Voronoi points
        sv_points = self._initialize_spherical_voronoi_points()
        sv_points = (sv_points + 1) / 2  # Normalize to [0,1]^3
        strategies.append(("spherical_voronoi", sv_points.copy()))

        # Strategy 4: Cube grid points
        cube_points = self._initialize_cube_grid_points()
        strategies.append(("cube_grid", cube_points.copy()))

        # Strategy 5: Random points
        random_points = np.random.rand(self.n_points, self.dimensions)
        strategies.append(("random", random_points.copy()))

        # Strategy 6: KMeans clustering approach with more samples
        kmeans_points = np.random.rand(50, self.dimensions)  # More samples for better clustering
        kmeans = KMeans(n_clusters=self.n_points, random_state=self.seed, n_init=20)
        kmeans.fit(kmeans_points)
        kmeans_centers = kmeans.cluster_centers_
        strategies.append(("kmeans", kmeans_centers.copy()))

        # Strategy 7: Perturbed spherical points
        perturbed_points = fib_points + np.random.normal(0, 0.05, (self.n_points, self.dimensions))
        perturbed_points = np.clip(perturbed_points, 0, 1)
        strategies.append(("perturbed", perturbed_points.copy()))

        # Strategy 8: Alternative perturbed spherical points
        alt_perturbed_points = fib_points + np.random.normal(0, 0.03, (self.n_points, self.dimensions))
        alt_perturbed_points = np.clip(alt_perturbed_points, 0, 1)
        strategies.append(("alt_perturbed", alt_perturbed_points.copy()))

        return strategies

    def _adaptive_differential_evolution(self, objective_func, bounds, maxiter=300):
        """Enhanced differential evolution with sophisticated adaptive parameters"""
        # Initialize parameters
        current_popsize = 20  # Starting with a more conservative population
        current_tol = 1e-6
        current_mutation = (0.5, 1.0)  # Standard range
        current_recombination = 0.8  # Higher recombination for better exploration
        
        # Performance tracking
        prev_obj_values = []
        improvement_window = []
        stagnation_counter = 0
        max_stagnation = 10
        
        # Phase-based parameter adjustment
        phases = [
            (0, 50, 30, 0.8, 0.5, 1.0),   # Phase 1: High exploration
            (50, 100, 20, 0.7, 0.3, 0.7),  # Phase 2: Balanced
            (100, float('inf'), 15, 0.9, 0.1, 0.5)  # Phase 3: Exploitation
        ]
        
        for iteration in range(maxiter // 10):
            # Determine phase and set parameters accordingly
            for start_iter, end_iter, pop_size, mut_min, mut_max, rec_rate in phases:
                if start_iter <= iteration < end_iter:
                    current_popsize = pop_size
                    current_mutation = (mut_min, mut_max)
                    current_recombination = rec_rate
                    break

            try:
                # Perform differential evolution with current parameters
                result = differential_evolution(
                    objective_func,
                    bounds,
                    seed=self.seed + iteration,
                    maxiter=10,
                    popsize=current_popsize,
                    tol=current_tol,
                    mutation=current_mutation,
                    recombination=current_recombination,
                    disp=False
                )

                # Track objective values for convergence analysis
                current_obj = -result.fun
                prev_obj_values.append(current_obj)
                
                if len(prev_obj_values) > 3:
                    prev_obj_values.pop(0)
                    
                improvement_window.append(current_obj)
                if len(improvement_window) > 5:
                    improvement_window.pop(0)

                # Adaptive parameter tuning based on recent progress
                if len(improvement_window) >= 2:
                    recent_improvement = improvement_window[-1] - improvement_window[-2]
                    
                    # Adjust tolerance based on improvement rate
                    if recent_improvement < 1e-8:
                        current_tol = max(1e-12, current_tol * 0.7)
                        stagnation_counter += 1
                    elif recent_improvement > 1e-5:
                        current_tol = max(1e-12, current_tol * 0.9)
                        stagnation_counter = max(0, stagnation_counter - 1)
                    else:
                        stagnation_counter = max(0, stagnation_counter - 1)
                        
                    # Adjust mutation based on improvement rate
                    if recent_improvement < 1e-8 and stagnation_counter > 3:
                        # Slow progress - increase mutation for more exploration
                        current_mutation = (0.7, 1.0)
                    elif recent_improvement > 1e-5:
                        # Fast progress - decrease mutation for exploitation
                        current_mutation = (0.3, 0.7)
                    else:
                        # Moderate progress - keep standard mutation
                        current_mutation = (0.5, 1.0)
                
                # Early stopping if stagnation occurs
                if stagnation_counter >= max_stagnation:
                    break

            except Exception as e:
                # Fallback to simpler configuration if optimization fails
                current_popsize = max(10, current_popsize - 5)
                current_tol = max(1e-10, current_tol * 2)
                if current_popsize < 10:
                    break
                continue

        return result

    def _multi_stage_local_refinement(self, points):
        """Apply multi-stage local refinement with progressive tightening"""
        refined_points = points.copy()
        
        # Stage 1: Coarse refinement with loose tolerances
        try:
            def objective_local(x):
                points_local = x.reshape(-1, 3)
                distances = pdist(points_local)

                if len(distances) == 0:
                    return -np.inf

                d_min = np.min(distances)
                d_max = np.max(distances)

                if d_max > 1e-12:
                    return -(d_min / d_max)
                else:
                    return -np.inf

            x0_refine = refined_points.flatten()
            bounds = [(0, 1)] * self.n_points * 3

            result_refine = minimize(
                objective_local,
                x0_refine,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-7, 'gtol': 1e-7},
                tol=1e-7
            )

            if result_refine.success:
                refined_points = result_refine.x.reshape(-1, 3)
                refined_points = np.clip(refined_points, 0, 1)
        except:
            pass

        # Stage 2: Fine refinement with tighter tolerances
        try:
            def objective_local_tight(x):
                points_local = x.reshape(-1, 3)
                distances = pdist(points_local)

                if len(distances) == 0:
                    return -np.inf

                d_min = np.min(distances)
                d_max = np.max(distances)

                if d_max > 1e-12:
                    return -(d_min / d_max)
                else:
                    return -np.inf

            x0_refine = refined_points.flatten()
            bounds = [(0, 1)] * self.n_points * 3

            result_refine = minimize(
                objective_local_tight,
                x0_refine,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-10, 'gtol': 1e-10},
                tol=1e-10
            )

            if result_refine.success:
                refined_points = result_refine.x.reshape(-1, 3)
                refined_points = np.clip(refined_points, 0, 1)
        except:
            pass

        return refined_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    optimizer = AdaptiveSpatialOptimizer(n_points=14, dimensions=3, seed=42)

    # Generate all initialization strategies
    strategies = optimizer._generate_initial_strategies()

    # Evaluate all strategies and select the best
    best_initialization = None
    best_ratio = -np.inf

    for name, points in strategies:
        ratio = optimizer._evaluate_initialization(points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_initialization = points.copy()

    # Use the best initialization as starting point
    x0 = best_initialization.flatten()

    # Bounds for each coordinate: [0, 1] for all 14 points × 3 coordinates
    bounds = [(0, 1)] * 14 * 3

    # Objective function for optimization
    def objective(x):
        points = x.reshape(-1, 3)
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)

        if d_max == 0:
            return -np.inf

        return -(d_min / d_max)

    # Run adaptive differential evolution optimization
    result = optimizer._adaptive_differential_evolution(objective, bounds, maxiter=250)

    # Extract optimized points
    optimized_points = result.x.reshape(-1, 3)

    # Apply multi-stage local refinement
    final_points = optimizer._multi_stage_local_refinement(optimized_points)

    # Final clipping to ensure bounds are respected
    final_points = np.clip(final_points, 0, 1)

    return final_points

# EVOLVE-BLOCK-END