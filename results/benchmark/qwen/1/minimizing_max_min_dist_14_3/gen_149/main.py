# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

class PointInitializer:
    """Handles various point initialization strategies"""
    
    @staticmethod
    def fibonacci_spiral(n):
        """Generate points using Fibonacci spiral on sphere"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            theta = np.arccos(1 - 2*(i/(n-1)))
            phi = i * 2 * np.pi / golden_ratio
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            points.append([x, y, z])
        return np.array(points)
    
    @staticmethod
    def perturbed_fibonacci(n, perturbation_strength=0.05):
        """Generate fibonacci points with small random perturbations"""
        base_points = PointInitializer.fibonacci_spiral(n)
        perturbations = np.random.normal(0, perturbation_strength, (n, 3))
        perturbed_points = base_points + perturbations
        # Normalize back to unit sphere
        norms = np.linalg.norm(perturbed_points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return perturbed_points / norms
    
    @staticmethod
    def random_on_sphere(n):
        """Generate random points on unit sphere"""
        points = np.random.randn(n, 3)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return points / norms

class OptimizationStrategy:
    """Manages different optimization approaches"""
    
    @staticmethod
    def global_search(objective, bounds, initial_points, seed_offset=0):
        """Perform global search using differential evolution"""
        try:
            result = differential_evolution(
                objective,
                bounds,
                seed=42 + seed_offset,
                maxiter=1000,
                popsize=20,
                tol=1e-8,
                mutation=(0.5, 1),
                recombination=0.7
            )
            return result.x.reshape(-1, 3)
        except Exception:
            return None
    
    @staticmethod
    def local_refinement(objective, initial_points, constraints):
        """Perform local refinement using SLSQP"""
        try:
            x0 = initial_points.flatten()
            result = minimize(objective, x0, method='SLSQP', constraints=constraints,
                            options={'ftol': 1e-10, 'maxiter': 1000})
            return result.x.reshape(-1, 3)
        except Exception:
            return None
    
    @staticmethod
    def final_polish(objective, initial_points, constraints):
        """Final polishing using L-BFGS-B"""
        try:
            x0 = initial_points.flatten()
            result = minimize(objective, x0, method='L-BFGS-B', constraints=constraints,
                            options={'ftol': 1e-14, 'maxiter': 500})
            return result.x.reshape(-1, 3)
        except Exception:
            return None

class SolutionEvaluator:
    """Evaluates optimization solutions"""
    
    @staticmethod
    def calculate_ratio(points):
        """Calculate min/max distance ratio"""
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist <= 0:
            return 0.0
        return min_dist / max_dist

class PointOptimizer:
    """Main optimization controller"""
    
    def __init__(self):
        self.initializer = PointInitializer()
        self.optimizer = OptimizationStrategy()
        self.evaluator = SolutionEvaluator()
        
    def _normalize_points(self, points):
        """Normalize points to lie on unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return points / norms
    
    def _create_constraints(self):
        """Create constraint functions for optimization"""
        def constraint_sphere(x):
            points = x.reshape(-1, 3)
            norms = np.linalg.norm(points, axis=1)
            return 1 - norms  # Should be >= 0
        
        def constraint_max_distance(x):
            points = x.reshape(-1, 3)
            distances = cdist(points, points)
            np.fill_diagonal(distances, 0)
            max_dist = np.max(distances)
            return 2 - max_dist  # Should be >= 0 (allowing up to diameter 2)
        
        return [
            {'type': 'ineq', 'fun': constraint_sphere},
            {'type': 'ineq', 'fun': constraint_max_distance}
        ]
    
    def _objective_function(self, x):
        """Objective function to maximize min/max distance ratio"""
        points = x.reshape(-1, 3)
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        return -np.min(distances)
    
    def _get_initial_configurations(self):
        """Generate diverse initial configurations"""
        configs = []
        
        # Configuration 1: Standard Fibonacci spiral
        configs.append(self.initializer.fibonacci_spiral(14))
        
        # Configuration 2: Perturbed Fibonacci with small noise
        configs.append(self.initializer.perturbed_fibonacci(14, 0.02))
        
        # Configuration 3: Perturbed Fibonacci with medium noise  
        configs.append(self.initializer.perturbed_fibonacci(14, 0.05))
        
        # Configuration 4: Perturbed Fibonacci with larger noise
        configs.append(self.initializer.perturbed_fibonacci(14, 0.1))
        
        # Configuration 5: Random points on sphere
        configs.append(self.initializer.random_on_sphere(14))
        
        # Configuration 6: Another set of random points
        np.random.seed(200)
        configs.append(self.initializer.random_on_sphere(14))
        
        # Configuration 7: Different Fibonacci variant with offset
        np.random.seed(300)
        base_fib = self.initializer.fibonacci_spiral(14)
        offset_noise = np.random.normal(0, 0.03, (14, 3))
        configs.append(self._normalize_points(base_fib + offset_noise))
        
        # Configuration 8: Another Fibonacci with different seed
        np.random.seed(400)
        configs.append(self.initializer.perturbed_fibonacci(14, 0.04))
        
        return configs
    
    def optimize(self):
        """Main optimization routine with hierarchical approach"""
        # Setup
        bounds = [(-1, 1) for _ in range(14 * 3)]
        constraints = self._create_constraints()
        configs = self._get_initial_configurations()
        
        best_points = None
        best_ratio = 0.0
        
        # Stage 1: Global Search with Differential Evolution
        for i, initial_config in enumerate(configs):
            optimized_points = self.optimizer.global_search(
                self._objective_function, bounds, initial_config, i
            )
            
            if optimized_points is not None:
                # Normalize and evaluate
                normalized_points = self._normalize_points(optimized_points)
                ratio = self.evaluator.calculate_ratio(normalized_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = normalized_points.copy()
        
        # Stage 2: Local Refinement if needed
        if best_points is not None:
            refined_points = self.optimizer.local_refinement(
                self._objective_function, best_points, constraints
            )
            
            if refined_points is not None:
                normalized_refined = self._normalize_points(refined_points)
                refined_ratio = self.evaluator.calculate_ratio(normalized_refined)
                
                if refined_ratio > best_ratio:
                    best_ratio = refined_ratio
                    best_points = normalized_refined.copy()
        
        # Stage 3: Final Polishing
        if best_points is not None:
            final_points = self.optimizer.final_polish(
                self._objective_function, best_points, constraints
            )
            
            if final_points is not None:
                normalized_final = self._normalize_points(final_points)
                final_ratio = self.evaluator.calculate_ratio(normalized_final)
                
                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_points = normalized_final.copy()
        
        # Fallback mechanism
        if best_points is None:
            # Try a few more targeted attempts
            np.random.seed(42)
            fallback_points = self.initializer.random_on_sphere(14)
            ratio = self.evaluator.calculate_ratio(fallback_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = fallback_points.copy()
        
        # Default fallback
        if best_points is None:
            best_points = self.initializer.fibonacci_spiral(14)
        
        return best_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    optimizer = PointOptimizer()
    return optimizer.optimize()

# EVOLVE-BLOCK-END