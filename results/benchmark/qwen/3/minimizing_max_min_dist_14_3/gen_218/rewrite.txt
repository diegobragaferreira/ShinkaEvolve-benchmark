# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import SphericalVoronoi
import time
import warnings
warnings.filterwarnings('ignore')

class HybridPointOptimizer3D:
    def __init__(self, n_points=14, dimension=3, max_time_seconds=360):
        self.n_points = n_points
        self.dimension = dimension
        self.max_time_seconds = max_time_seconds
        self.best_score = -np.inf
        self.best_points = None

    def _compute_distances(self, points):
        """Compute pairwise distances efficiently"""
        return cdist(points, points, 'euclidean')

    def _calculate_ratio(self, points):
        """Calculate min/max distance ratio"""
        distances = self._compute_distances(points)
        np.fill_diagonal(distances, np.inf)
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        if max_dist == 0:
            return 0.0

        return min_dist / max_dist

    def _objective_function(self, x):
        """Objective function returning negative ratio for maximization"""
        points = x.reshape((self.n_points, self.dimension))
        ratio = self._calculate_ratio(points)
        return -ratio

    def _penalty_objective(self, x, penalty_weight=1e6):
        """Objective with penalty for boundary violations"""
        points = x.reshape((self.n_points, self.dimension))

        # Calculate base objective
        ratio = self._calculate_ratio(points)
        base_obj = -ratio

        # Add penalty for points outside [0,1]^3 bounds
        penalty = 0
        for i in range(self.n_points):
            for j in range(self.dimension):
                coord = points[i, j]
                if coord < 0:
                    penalty += penalty_weight * (0 - coord) ** 2
                elif coord > 1:
                    penalty += penalty_weight * (coord - 1) ** 2

        return base_obj + penalty

    def _fibonacci_sphere_sampling(self, n):
        """Generate points on a unit sphere using Fibonacci spiral method"""
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

    def _electrostatic_repulsion(self, points, iterations=5):
        """Apply simple electrostatic repulsion to improve point distribution"""
        for _ in range(iterations):
            # Calculate pairwise distances and forces
            forces = np.zeros_like(points)
            distances = cdist(points, points, 'euclidean')
            np.fill_diagonal(distances, 1000)  # Avoid self-interaction

            for i in range(len(points)):
                for j in range(len(points)):
                    if i != j:
                        diff = points[i] - points[j]
                        dist_sq = np.sum(diff**2)
                        if dist_sq > 1e-10:
                            force_magnitude = 1.0 / (dist_sq * np.sqrt(dist_sq))
                            forces[i] += force_magnitude * diff

            # Update positions with forces
            points += 0.01 * forces

            # Keep points on unit sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            points = points / norms
            
        return points

    def _initialize_spherical_points(self):
        """Initialize points using improved spherical sampling with better distribution"""
        # Generate points using Fibonacci-like method
        points = self._fibonacci_sphere_sampling(self.n_points)

        # Apply electrostatic repulsion for better distribution
        points = self._electrostatic_repulsion(points, iterations=5)

        # Add small random perturbations to escape local minima
        np.random.seed(42)
        perturbations = np.random.normal(0, 0.005, points.shape)
        points += perturbations

        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        points = points / norms

        # Scale appropriately to have reasonable distances
        points *= 0.8

        # Project to [0,1]^3 space
        points_cube = (points + 1) / 2
        return points_cube

    def _initialize_random_points(self):
        """Alternative random initialization"""
        np.random.seed(42)
        points = np.random.rand(self.n_points, self.dimension)
        return points

    def _initialize_points(self):
        """Multi-strategy initialization with diversity"""
        # Try spherical initialization
        spherical_points = self._initialize_spherical_points()
        
        # Try random initialization
        random_points = self._initialize_random_points()

        # Evaluate both initializations
        spherical_ratio = self._calculate_ratio(spherical_points)
        random_ratio = self._calculate_ratio(random_points)

        # Choose the better initialization
        if spherical_ratio > random_ratio:
            return spherical_points
        else:
            return random_points

    def _adaptive_differential_evolution(self, initial_points, max_time):
        """Perform differential evolution with adaptive population sizing and early stopping"""
        bounds = [(0, 1)] * self.n_points * self.dimension
        
        # Adaptive parameters
        popsize = 15
        maxiter = 500
        stagnant_generations = 0
        max_stagnant = 10
        previous_best = float('inf')
        improvement_threshold = 1e-8

        def adaptive_callback(x, convergence):
            nonlocal popsize, stagnant_generations, previous_best
            
            current_best = -convergence  # Convert back to ratio
            
            if abs(previous_best - current_best) < improvement_threshold:
                stagnant_generations += 1
                if stagnant_generations > max_stagnant and popsize < 40:
                    popsize = min(popsize + 3, 40)
                    stagnant_generations = 0
            else:
                stagnant_generations = 0
                previous_best = current_best
                
            return time.time() - start_time > max_time - 10

        try:
            start_time = time.time()
            result = differential_evolution(
                self._penalty_objective,
                bounds,
                seed=42,
                maxiter=maxiter,
                popsize=popsize,
                mutation=(0.5, 1.0),
                recombination=0.7,
                tol=1e-8,
                callback=adaptive_callback,
                disp=False
            )

            return result.x.reshape((self.n_points, self.dimension))

        except Exception:
            # Fallback to basic optimization with local search
            return self._local_refinement(initial_points)

    def _enhanced_local_refinement(self, points):
        """Apply multiple local optimization approaches with smart fallback"""
        def obj(x):
            points_temp = x.reshape(self.n_points, self.dimension)
            distances = self._compute_distances(points_temp)
            np.fill_diagonal(distances, np.inf)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist == 0:
                return 1e10
            return -min_dist / max_dist  # Negative for maximization

        # Strategy 1: L-BFGS-B with bounds (primary method)
        bounds = [(0, 1) for _ in range(self.n_points * self.dimension)]
        try:
            res = minimize(obj, points.flatten(), method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10})
            if res.success:
                return res.x.reshape(self.n_points, self.dimension)
        except:
            pass
            
        # Strategy 2: SLSQP as secondary method
        try:
            res = minimize(obj, points.flatten(), method='SLSQP', bounds=bounds,
                         options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10})
            if res.success:
                return res.x.reshape(self.n_points, self.dimension)
        except:
            pass
            
        # Strategy 3: Nelder-Mead as fallback
        try:
            res = minimize(obj, points.flatten(), method='Nelder-Mead',
                         options={'maxiter': 300, 'disp': False})
            if res.success:
                return res.x.reshape(self.n_points, self.dimension)
        except:
            pass
            
        return points

    def _validate_and_correct_bounds(self, points):
        """Ensure all points are within [0,1]^3 bounds"""
        corrected_points = np.clip(points, 0, 1)
        return corrected_points

    def _multiple_restarts(self, initial_points):
        """Try multiple restarts with different perturbation schemes"""
        best_points = initial_points.copy()
        best_ratio = self._calculate_ratio(initial_points)
        
        # Different perturbation magnitudes for restarts
        perturbation_magnitudes = [0.005, 0.01, 0.015]
        
        for restart in range(5):
            # Select perturbation magnitude with cycling
            mag_idx = restart % len(perturbation_magnitudes)
            mag = perturbation_magnitudes[mag_idx]
            
            np.random.seed(restart * 1000 + 42)
            
            # Create slightly perturbed starting point
            perturbed = initial_points + np.random.normal(0, mag, initial_points.shape)
            perturbed = np.clip(perturbed, 0, 1)
            
            # Apply refinement to perturbed point
            restarted_points = self._enhanced_local_refinement(perturbed)
            restarted_ratio = self._calculate_ratio(restarted_points)
            
            if restarted_ratio > best_ratio:
                best_ratio = restarted_ratio
                best_points = restarted_points.copy()

        return best_points

    def optimize(self):
        """Main optimization routine with hierarchical approach"""
        start_time = time.time()
        
        # Phase 1: Initialization
        initial_points = self._initialize_points()

        # Phase 2: Global optimization with adaptive DE
        de_optimized = self._adaptive_differential_evolution(initial_points, self.max_time_seconds - 60)

        # Phase 3: Local refinement with enhanced strategies
        local_optimized = self._enhanced_local_refinement(de_optimized)

        # Phase 4: Multiple restarts for exploration
        final_points = self._multiple_restarts(local_optimized)

        # Final validation
        final_points = self._validate_and_correct_bounds(final_points)

        # Final check of quality
        final_ratio = self._calculate_ratio(final_points)

        # Store best solution found
        if final_ratio > self.best_score:
            self.best_score = final_ratio
            self.best_points = final_points.copy()

        return self.best_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    optimizer = HybridPointOptimizer3D(n_points=14, dimension=3)
    return optimizer.optimize()

# EVOLVE-BLOCK-END