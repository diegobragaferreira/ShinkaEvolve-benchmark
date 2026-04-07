# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import warnings
warnings.filterwarnings('ignore')

class PointDispersionOptimizer:
    """Optimizes placement of 14 points in 3D space to maximize min/max distance ratio."""
    
    def __init__(self, num_points=14, dimension=3):
        self.num_points = num_points
        self.dimension = dimension
        self.best_solution = None
        self.best_ratio = -np.inf
        
    def _compute_distance_matrix(self, points):
        """Efficiently compute pairwise distances."""
        return pdist(points)
    
    def _calculate_min_max_ratio(self, points):
        """Calculate the ratio of minimum to maximum pairwise distances."""
        distances = self._compute_distance_matrix(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0.0
            
        return min_dist / max_dist
    
    def _objective_function(self, x):
        """Objective function for maximization (returns negative ratio)."""
        points = x.reshape((self.num_points, self.dimension))
        ratio = self._calculate_min_max_ratio(points)
        return -ratio
    
    def _penalized_objective(self, x, penalty_weight=1e6):
        """Objective function with penalty for boundary violations."""
        points = x.reshape((self.num_points, self.dimension))
        
        # Base objective (negative ratio for maximization)
        base_ratio = -self._objective_function(x)
        
        # Penalty for out-of-bounds points
        penalty = 0
        for i in range(self.num_points):
            for j in range(self.dimension):
                coord = points[i, j]
                if coord < 0:
                    penalty += penalty_weight * (0 - coord) ** 2
                elif coord > 1:
                    penalty += penalty_weight * (coord - 1) ** 2
                    
        return base_ratio + penalty
    
    def _fibonacci_sphere_points(self, n):
        """Generate well-distributed points on a unit sphere."""
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
    
    def _enhanced_spherical_initialization(self):
        """Create initial points using enhanced spherical sampling."""
        # Generate Fibonacci sphere points
        sphere_points = self._fibonacci_sphere_points(self.num_points)
        
        # Apply iterative improvement (simple repulsion)
        for _ in range(10):
            # Compute pairwise distances
            distances = pdist(sphere_points)
            distance_matrix = np.zeros((len(sphere_points), len(sphere_points)))
            distance_matrix[np.triu_indices_from(distance_matrix, k=1)] = distances
            distance_matrix += distance_matrix.T
            
            # Calculate repulsion forces
            forces = np.zeros_like(sphere_points)
            for i in range(len(sphere_points)):
                for j in range(len(sphere_points)):
                    if i != j:
                        diff = sphere_points[i] - sphere_points[j]
                        dist_sq = np.sum(diff**2)
                        if dist_sq > 1e-10:
                            force_magnitude = 1.0 / dist_sq
                            forces[i] += force_magnitude * diff
            
            # Apply forces and project back to sphere
            sphere_points += 0.01 * forces
            norms = np.linalg.norm(sphere_points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            sphere_points = sphere_points / norms
        
        # Add small random perturbations
        np.random.seed(42)
        perturbations = np.random.normal(0, 0.005, sphere_points.shape)
        sphere_points += perturbations
        
        # Normalize and scale
        norms = np.linalg.norm(sphere_points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        sphere_points = sphere_points / norms
        sphere_points *= 0.8
        
        # Transform to [0,1]^3 space
        sphere_points = (sphere_points + 1) / 2
        
        return sphere_points
    
    def _random_initialization(self):
        """Generate random points in [0,1]^3."""
        np.random.seed(42)
        return np.random.rand(self.num_points, self.dimension)
    
    def _initialize_population(self):
        """Initialize multiple strategies and return the best one."""
        strategies = []
        
        # Strategy 1: Enhanced spherical initialization
        spherical_points = self._enhanced_spherical_initialization()
        strategies.append(("spherical", spherical_points))
        
        # Strategy 2: Random initialization
        random_points = self._random_initialization()
        strategies.append(("random", random_points))
        
        # Strategy 3: Perturbed spherical with medium variation
        np.random.seed(42)
        perturbed_points = spherical_points + np.random.normal(0, 0.05, spherical_points.shape)
        strategies.append(("perturbed_spherical", perturbed_points))
        
        # Evaluate all strategies and select best
        best_strategy = None
        best_ratio = -np.inf
        
        for name, points in strategies:
            ratio = self._calculate_min_max_ratio(points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_strategy = points
                
        return best_strategy.flatten()
    
    def _global_optimization(self, initial_points):
        """Perform global optimization using differential evolution."""
        bounds = [(0, 1)] * self.num_points * self.dimension
        
        # Try multiple DE configurations
        configs = [
            {'popsize': 20, 'mutation': (0.5, 1.0), 'recombination': 0.7, 'maxiter': 300},
            {'popsize': 25, 'mutation': (0.7, 1.0), 'recombination': 0.8, 'maxiter': 250},
            {'popsize': 30, 'mutation': (0.8, 1.0), 'recombination': 0.9, 'maxiter': 200}
        ]
        
        best_points = initial_points.reshape((self.num_points, self.dimension))
        best_ratio = -np.inf
        
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
                    tol=1e-8,
                    callback=None
                )
                
                # Evaluate result
                points = result.x.reshape((self.num_points, self.dimension))
                ratio = self._calculate_min_max_ratio(points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = points
                    
            except Exception:
                continue
                
        return best_points
    
    def _local_refinement(self, points):
        """Apply local refinement using L-BFGS-B."""
        try:
            # First stage: moderate tolerances
            result1 = minimize(
                self._objective_function,
                points.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1)] * self.num_points * self.dimension,
                options={'ftol': 1e-6, 'gtol': 1e-6, 'maxiter': 500}
            )
            
            refined_points = result1.x.reshape((self.num_points, self.dimension))
            
            # Second stage: tight tolerances for final refinement
            result2 = minimize(
                self._objective_function,
                refined_points.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1)] * self.num_points * self.dimension,
                options={'ftol': 1e-9, 'gtol': 1e-9, 'maxiter': 500}
            )
            
            final_points = result2.x.reshape((self.num_points, self.dimension))
            return final_points
            
        except Exception:
            return points
    
    def _validate_bounds(self, points):
        """Ensure all points remain within [0,1]^3 bounds."""
        return np.clip(points, 0, 1)
    
    def optimize(self):
        """Execute the complete optimization process."""
        # Phase 1: Initialize points
        initial_points = self._initialize_population()
        
        # Phase 2: Global optimization
        global_optimized = self._global_optimization(initial_points)
        
        # Phase 3: Local refinement
        local_optimized = self._local_refinement(global_optimized)
        
        # Phase 4: Final validation
        final_points = self._validate_bounds(local_optimized)
        
        # Store best solution
        final_ratio = self._calculate_min_max_ratio(final_points)
        if final_ratio > self.best_ratio:
            self.best_ratio = final_ratio
            self.best_solution = final_points.copy()
            
        return self.best_solution

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    optimizer = PointDispersionOptimizer(num_points=14, dimension=3)
    return optimizer.optimize()

# EVOLVE-BLOCK-END