# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class InitializationManager:
    """Manages various point initialization strategies"""
    
    def __init__(self, n_points=14, seed=42):
        self.n_points = n_points
        self.seed = seed
        np.random.seed(seed)
    
    def fibonacci_points(self):
        """Initialize points on a unit sphere using Fibonacci spiral method"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2

        for i in range(self.n_points):
            phi = np.arccos(1 - 2*i/(self.n_points-1))
            theta = 2 * np.pi * i / golden_ratio

            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

            points.append([x, y, z])

        return np.array(points)
    
    def cube_grid_points(self):
        """Initialize points in a 3D cube grid"""
        grid_size = int(np.ceil(self.n_points**(1/3)))
        coords = np.linspace(0, 1, grid_size)
        grid_points = []

        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if len(grid_points) < self.n_points:
                        grid_points.append([coords[i], coords[j], coords[k]])

        return np.array(grid_points[:self.n_points])
    
    def random_points(self):
        """Initialize random points in 3D space"""
        return np.random.rand(self.n_points, 3)
    
    def voronoi_uniform_points(self):
        """Initialize points with good Voronoi uniformity using iterative approach"""
        points = np.random.rand(self.n_points, 3)

        for _ in range(50):
            distances = pdist(points)
            if len(distances) > 0:
                for i in range(self.n_points):
                    dist_row = distances[i*(self.n_points-1):(i+1)*(self.n_points-1)]
                    if len(dist_row) > 0:
                        closest_idx = np.argmin(dist_row)
                        if closest_idx < i:
                            neighbor = points[closest_idx]
                        else:
                            neighbor = points[closest_idx + 1]

                        direction = points[i] - neighbor
                        norm_dir = np.linalg.norm(direction)
                        if norm_dir > 1e-12:
                            points[i] += 0.01 * direction / norm_dir
                points = np.clip(points, 0, 1)

        return points
    
    def spherical_cap_points(self):
        """Initialize points on a spherical cap for better distribution"""
        points = []
        for i in range(self.n_points):
            phi = np.arccos(1 - 2 * (i / (self.n_points - 1)))
            theta = 2 * np.pi * i * (1 + np.sqrt(5)) / 2
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points.append([x, y, z])
        return np.array(points)
    
    def golden_spiral_points(self):
        """Initialize points using golden spiral method for better uniformity"""
        points = []
        phi = np.pi * (3 - np.sqrt(5))
        for i in range(self.n_points):
            y = 1 - (i / float(self.n_points - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = phi * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)
    
    def hybrid_points(self):
        """Initialize hybrid points combining multiple strategies"""
        fib_points = self.fibonacci_points()
        fib_points = (fib_points + 1) / 2
        
        noise = np.random.normal(0, 0.03, fib_points.shape)
        perturbed = fib_points + noise
        perturbed = np.clip(perturbed, 0, 1)
        
        try:
            kmeans = KMeans(n_clusters=self.n_points, random_state=self.seed, n_init=5)
            kmeans.fit(perturbed)
            return kmeans.cluster_centers_
        except:
            return perturbed
    
    def perturb_points(self, points, magnitude=0.05):
        """Add small random perturbations to break symmetry"""
        noise = np.random.normal(0, magnitude, points.shape)
        perturbed = points + noise
        return np.clip(perturbed, 0, 1)
    
    def generate_all_strategies(self):
        """Generate all initialization strategies"""
        strategies = {}
        
        # Basic strategies
        strategies['fibonacci'] = self.fibonacci_points()
        strategies['cube_grid'] = self.cube_grid_points()
        strategies['random'] = self.random_points()
        strategies['voronoi_uniform'] = self.voronoi_uniform_points()
        strategies['spherical_cap'] = self.spherical_cap_points()
        strategies['golden_spiral'] = self.golden_spiral_points()
        strategies['hybrid'] = self.hybrid_points()
        
        # Derived strategies
        fib_points = strategies['fibonacci']
        strategies['fibonacci_normalized'] = (fib_points + 1) / 2
        strategies['perturbed_fibonacci'] = self.perturb_points(fib_points, 0.05)
        strategies['optimized_fibonacci'] = self.perturb_points(fib_points, 0.02)
        
        # Normalize spherical points to unit cube
        for key in ['fibonacci', 'spherical_cap', 'golden_spiral']:
            if key in strategies:
                strategies[key] = (strategies[key] + 1) / 2
                
        return strategies

class EvaluationEngine:
    """Handles point evaluation and distance calculations"""
    
    @staticmethod
    def compute_distance_ratio(points):
        """Compute the minimum/maximum distance ratio"""
        if len(points) < 2:
            return 0

        distances = pdist(points)
        if len(distances) == 0:
            return 0

        d_min = np.min(distances)
        d_max = np.max(distances)

        if d_max <= 1e-12:
            return 0

        return d_min / d_max
    
    @staticmethod
    def evaluate_initialization(points):
        """Fast evaluation of initialization quality with uniformity consideration"""
        ratio = EvaluationEngine.compute_distance_ratio(points)

        try:
            distances = pdist(points)
            if len(distances) > 0:
                distance_mean = np.mean(distances)
                if distance_mean > 1e-12:
                    distance_variance = np.var(distances)
                    uniformity_score = 1.0 / (distance_variance + 1e-12) * distance_mean
                    return 0.8 * ratio + 0.2 * uniformity_score
                else:
                    return ratio
            else:
                return ratio
        except:
            return ratio

class OptimizationEngine:
    """Handles the optimization processes"""
    
    def __init__(self, n_points=14, seed=42):
        self.n_points = n_points
        self.seed = seed
    
    def _objective_function(self, x):
        """Objective function that returns negative ratio to minimize"""
        points = x.reshape(-1, 3)
        points = np.clip(points, 0, 1)
        
        distances = pdist(points)
        if len(distances) == 0:
            return -np.inf

        d_min = np.min(distances)
        d_max = np.max(distances)

        if d_max <= 1e-12:
            return -np.inf

        return -(d_min / d_max)
    
    def _penalty_objective(self, x, penalty_weight=1e7):
        """Objective with penalty for boundary violations"""
        points = x.reshape(-1, 3)
        
        penalty = 0
        penalty += np.sum(np.maximum(0, -points)**2) * penalty_weight
        penalty += np.sum(np.maximum(0, points - 1)**2) * penalty_weight

        original_obj = self._objective_function(x)
        return original_obj + penalty
    
    def adaptive_differential_evolution(self, objective_func, bounds, maxiter=300):
        """Enhanced differential evolution with adaptive parameters"""
        current_popsize = 25
        prev_best = -np.inf
        stagnation_count = 0
        improvement_threshold = 1e-8
        min_improvement = 1e-10
        recent_improvements = []
        improvement_window = 10
        
        for iteration in range(maxiter // 15):
            if stagnation_count > 2 and current_popsize < 40:
                current_popsize = min(current_popsize + 5, 40)
            
            try:
                result = differential_evolution(
                    objective_func,
                    bounds,
                    seed=self.seed + iteration,
                    maxiter=15,
                    popsize=current_popsize,
                    tol=1e-15,
                    mutation=(0.7, 1.0),
                    recombination=0.8,
                    disp=False
                )
            except Exception:
                try:
                    result = differential_evolution(
                        objective_func,
                        bounds,
                        seed=self.seed + iteration,
                        maxiter=15,
                        popsize=max(5, current_popsize - 5),
                        tol=1e-15,
                        mutation=(0.7, 1.0),
                        recombination=0.8,
                        disp=False
                    )
                except Exception:
                    result = differential_evolution(
                        objective_func,
                        bounds,
                        seed=self.seed + iteration,
                        maxiter=15,
                        popsize=15,
                        tol=1e-15,
                        mutation=(0.7, 1.0),
                        recombination=0.7,
                        disp=False
                    )
            
            current_best = -result.fun
            improvement = current_best - prev_best
            
            recent_improvements.append(improvement)
            if len(recent_improvements) > improvement_window:
                recent_improvements.pop(0)
            
            if len(recent_improvements) == improvement_window:
                avg_improvement = np.mean(recent_improvements)
                if abs(avg_improvement) < min_improvement:
                    break
                    
            if improvement > improvement_threshold:
                stagnation_count = 0
            else:
                stagnation_count += 1
                
            prev_best = current_best
            
        return result
    
    def local_refinement(self, points):
        """Apply local refinement to improve final solution"""
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

        try:
            x0_refine = points.flatten()
            bounds = [(0, 1)] * self.n_points * 3
            
            result_refine = minimize(
                objective_local,
                x0_refine,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-15, 'gtol': 1e-15},
                tol=1e-15
            )
            
            refined_points = result_refine.x.reshape(-1, 3)
            refined_points = np.clip(refined_points, 0, 1)
            
            final_ratio = EvaluationEngine.compute_distance_ratio(refined_points)
            if final_ratio > 0.95 * EvaluationEngine.compute_distance_ratio(points):
                result_refine2 = minimize(
                    objective_local,
                    refined_points.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'ftol': 1e-16, 'gtol': 1e-16},
                    tol=1e-16
                )
                refined_points = result_refine2.x.reshape(-1, 3)
                refined_points = np.clip(refined_points, 0, 1)
            
            return refined_points
        except Exception:
            return points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Initialize components
    init_manager = InitializationManager(n_points=14, seed=42)
    eval_engine = EvaluationEngine()
    opt_engine = OptimizationEngine(n_points=14, seed=42)
    
    # Generate all initialization strategies
    strategies = init_manager.generate_all_strategies()
    
    # Select best initialization
    best_initialization = None
    best_ratio = -np.inf

    for name, points in strategies.items():
        ratio = eval_engine.evaluate_initialization(points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_initialization = points.copy()

    # Bounds definition
    bounds = [(0, 1)] * 14 * 3

    # Run adaptive differential evolution optimization
    best_result = None
    best_ratio = -np.inf

    # Try multiple seeds for better exploration
    for seed_val in [42, 123, 456, 789, 101]:
        np.random.seed(seed_val)

        result = opt_engine.adaptive_differential_evolution(
            opt_engine._penalty_objective,
            bounds,
            maxiter=300
        )

        if -result.fun > best_ratio:
            best_ratio = -result.fun
            best_result = result

    # Extract optimized points
    optimized_points = best_result.x.reshape(-1, 3)

    # Apply local refinement
    final_points = opt_engine.local_refinement(optimized_points)

    # Final clipping
    final_points = np.clip(final_points, 0, 1)

    return final_points

# EVOLVE-BLOCK-END