# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize, basinhopping
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class PointDispersionEvolver:
    """Evolver for optimizing 14 points in 3D to maximize min/max distance ratio."""
    
    def __init__(self, num_points=14, dimension=3):
        self.num_points = num_points
        self.dimension = dimension
        self.bounds = [(0, 1)] * num_points * dimension
        self.config = {
            'initialization_strategies': 6,
            'de_popsize': 25,
            'de_maxiter': 300,
            'lbfgs_tolerance': 1e-12,
            'penalty_weight': 1e6,
            'refinement_iterations': 1000
        }
    
    def _compute_distances(self, points):
        """Compute pairwise distances efficiently."""
        return pdist(points)
    
    def _calculate_ratio(self, points):
        """Calculate the ratio of minimum to maximum distances."""
        distances = self._compute_distances(points)
        if len(distances) == 0:
            return 0.0
            
        distances = distances[np.isfinite(distances)]
        if len(distances) == 0:
            return 0.0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max <= 1e-12:
            return 0.0
            
        return d_min / d_max
    
    def _objective_function(self, x):
        """Objective function returning negative ratio for minimization."""
        points = x.reshape((self.num_points, self.dimension))
        ratio = self._calculate_ratio(points)
        return -ratio
    
    def _penalized_objective(self, x):
        """Objective function with boundary penalty."""
        points = x.reshape((self.num_points, self.dimension))
        
        # Calculate base objective
        ratio = -self._objective_function(x)
        
        # Add boundary penalty
        penalty = 0
        penalty_weight = self.config['penalty_weight']
        
        # Penalty for points below 0
        below_penalty = np.sum(np.maximum(0, -points)**2) * penalty_weight
        # Penalty for points above 1
        above_penalty = np.sum(np.maximum(0, points - 1)**2) * penalty_weight
        
        return ratio + below_penalty + above_penalty
    
    def _fibonacci_sphere_points(self, n):
        """Generate well-distributed points on a unit sphere using Fibonacci method."""
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
    
    def _spherical_voronoi_points(self, n):
        """Generate points using spherical Voronoi diagram approach."""
        # Generate initial points randomly on sphere
        np.random.seed(42)
        points = np.random.randn(n, 3)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Simple repulsion simulation for better distribution
        for _ in range(20):
            forces = np.zeros_like(points)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        diff = points[i] - points[j]
                        dist_sq = np.sum(diff**2)
                        if dist_sq > 1e-10:
                            force_magnitude = 1.0 / dist_sq
                            forces[i] += force_magnitude * diff
            
            points += 0.01 * forces
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            points = points / norms
            
        return points
    
    def _create_initialization_strategies(self):
        """Factory method to create multiple initialization strategies."""
        strategies = []
        
        # Strategy 1: Fibonacci sphere with jitter
        np.random.seed(42)
        fib_points = self._fibonacci_sphere_points(self.num_points)
        jitter = np.random.normal(0, 0.02, fib_points.shape)
        strat1 = fib_points + jitter
        norms = np.linalg.norm(strat1, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        strat1 = strat1 / norms
        # Scale to [0,1]^3
        strat1 = strat1 - np.mean(strat1, axis=0)
        max_coord = np.max(np.abs(strat1))
        if max_coord > 0:
            strat1 = strat1 / (2 * max_coord) + 0.5
        strategies.append(("fibonacci_jittered", strat1))
        
        # Strategy 2: Simple random uniform
        np.random.seed(123)
        strat2 = np.random.rand(self.num_points, self.dimension)
        strategies.append(("random_uniform", strat2))
        
        # Strategy 3: Fibonacci with different scaling
        np.random.seed(456)
        strat3 = self._fibonacci_sphere_points(self.num_points)
        strat3 = strat3 * 0.8 + 0.1
        strategies.append(("fibonacci_scaled", strat3))
        
        # Strategy 4: Grid-based approach
        np.random.seed(789)
        grid_size = int(np.ceil(self.num_points**(1/3)))
        coords = np.linspace(0, 1, grid_size)
        grid_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if len(grid_points) < self.num_points:
                        grid_points.append([coords[i], coords[j], coords[k]])
        strat4 = np.array(grid_points[:self.num_points])
        # Add jitter
        jitter = np.random.normal(0, 0.03, strat4.shape)
        strat4 += jitter
        strat4 = np.clip(strat4, 0, 1)
        strategies.append(("grid_jittered", strat4))
        
        # Strategy 5: Spherical Voronoi
        voronoi_points = self._spherical_voronoi_points(self.num_points)
        strat5 = (voronoi_points + 1) / 2  # Normalize to [0,1]^3
        strategies.append(("voronoi", strat5))
        
        # Strategy 6: KMeans clustering
        np.random.seed(999)
        kmeans_points = np.random.rand(50, self.dimension)
        kmeans = KMeans(n_clusters=self.num_points, random_state=42, n_init=20)
        kmeans.fit(kmeans_points)
        strat6 = kmeans.cluster_centers_
        strategies.append(("kmeans", strat6))
        
        return strategies
    
    def _select_best_initialization(self):
        """Select the best initialization strategy from all available."""
        strategies = self._create_initialization_strategies()
        best_strategy = None
        best_ratio = -np.inf
        
        for name, points in strategies:
            try:
                ratio = self._calculate_ratio(points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_strategy = points.copy()
            except Exception:
                continue
                
        # Fallback to random if all fail
        if best_strategy is None:
            np.random.seed(42)
            best_strategy = np.random.rand(self.num_points, self.dimension)
            
        return best_strategy.flatten()
    
    def _adaptive_differential_evolution(self, initial_points):
        """Perform adaptive differential evolution with multiple retries."""
        bounds = self.bounds
        
        # Multiple DE configurations to try
        configs = [
            {'popsize': 20, 'mutation': (0.5, 1.0), 'recombination': 0.7, 'maxiter': 200},
            {'popsize': 25, 'mutation': (0.7, 1.0), 'recombination': 0.8, 'maxiter': 250},
            {'popsize': 30, 'mutation': (0.8, 1.0), 'recombination': 0.9, 'maxiter': 200} 
        ]
        
        best_points = initial_points.reshape((self.num_points, self.dimension))
        best_ratio = -np.inf
        
        # Try multiple configurations
        for config in configs:
            try:
                result = differential_evolution(
                    self._penalized_objective,
                    bounds,
                    seed=42,
                    maxiter=config['maxiter'],
                    popsize=config['popsize'],
                    mutation=config['mutation'],
                    recombination=config['recombination'],
                    tol=1e-9,
                    disp=False
                )
                
                points = result.x.reshape((self.num_points, self.dimension))
                ratio = self._calculate_ratio(points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = points
                    
            except Exception:
                continue
                
        return best_points
    
    def _basin_hopping_refinement(self, points):
        """Apply basin hopping for local refinement."""
        try:
            minimizer_kwargs = {"method": "L-BFGS-B", "bounds": self.bounds}
            result_bh = basinhopping(
                self._objective_function,
                points.flatten(),
                niter=15,
                T=1.0,
                stepsize=0.1,
                minimizer_kwargs=minimizer_kwargs,
                seed=42
            )
            
            if result_bh.success:
                refined_points = result_bh.x.reshape(-1, self.dimension)
                return refined_points
        except Exception:
            pass
            
        return points
    
    def _multi_stage_lbfgs(self, points):
        """Apply multi-stage L-BFGS refinement."""
        # Stage 1: Coarse refinement
        try:
            result_coarse = minimize(
                self._objective_function,
                points.flatten(),
                method='L-BFGS-B',
                bounds=self.bounds,
                options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if result_coarse.success:
                refined_coarse = result_coarse.x.reshape(-1, self.dimension)
                
                # Stage 2: Fine refinement  
                result_fine = minimize(
                    self._objective_function,
                    refined_coarse.flatten(),
                    method='L-BFGS-B',
                    bounds=self.bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                
                if result_fine.success:
                    refined_fine = result_fine.x.reshape(-1, self.dimension)
                    return refined_fine
                else:
                    return refined_coarse
            else:
                return points
        except Exception:
            return points
    
    def _validate_and_correct_bounds(self, points):
        """Ensure all points are within valid bounds."""
        return np.clip(points, 0, 1)
    
    def evolve(self):
        """Main evolution process."""
        # Phase 1: Initialization
        initial_points = self._select_best_initialization()
        
        # Phase 2: Global optimization
        global_optimized = self._adaptive_differential_evolution(initial_points)
        
        # Phase 3: Local refinement with basin hopping
        local_refined = self._basin_hopping_refinement(global_optimized)
        
        # Phase 4: Multi-stage L-BFGS refinement
        final_refined = self._multi_stage_lbfgs(local_refined)
        
        # Phase 5: Final validation
        final_points = self._validate_and_correct_bounds(final_refined)
        
        return final_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    evolver = PointDispersionEvolver(num_points=14, dimension=3)
    return evolver.evolve()

# EVOLVE-BLOCK-END