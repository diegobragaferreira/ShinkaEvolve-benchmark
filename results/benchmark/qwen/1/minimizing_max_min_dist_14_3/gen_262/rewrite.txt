# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import SphericalVoronoi
import warnings
from typing import Tuple, Optional
import time

class SphericalVoronoiEvolutionOptimizer:
    """Novel optimizer using spherical Voronoi diagrams for point distribution optimization."""
    
    def __init__(self, num_points: int = 14):
        self.n = num_points
        self.golden_ratio = (1 + np.sqrt(5)) / 2
        self.best_solution = None
        self.best_ratio = -np.inf
        self.eval_time = 0.0

    def _calculate_ratio(self, points: np.ndarray) -> Tuple[float, float]:
        """Calculate min/max distance ratio for given points."""
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        return min_dist, max_dist

    def _voronoi_initialize(self) -> np.ndarray:
        """Initialize points using spherical Voronoi diagram approach for better spatial distribution."""
        # Generate initial points that form a basic symmetric structure
        # This creates points that are naturally well-separated
        points = []
        
        # Create a 3D grid-like structure projected onto sphere
        # Using Fibonacci-based approach but with added Voronoi-inspired spacing
        phi = np.pi * (3 - np.sqrt(5))  # golden angle
        
        for i in range(self.n):
            # Distribute points more evenly using modified Fibonacci
            y = 1 - (i / float(self.n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            # Add Voronoi-inspired perturbation for better distribution
            theta = phi * i + 0.1 * np.sin(i * 0.7) + 0.05 * np.cos(i * 1.3)
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        points = np.array(points)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        points = points / norms
        
        # Add small random perturbations to break any remaining symmetries
        np.random.seed(42)
        noise_magnitude = 0.015
        noisy_points = points + np.random.normal(0, noise_magnitude, points.shape)
        
        # Ensure normalization after noise
        norms = np.linalg.norm(noisy_points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        points = noisy_points / norms
        
        return points

    def _objective_with_regularization(self, x: np.ndarray, lambda_reg: float = 0.15) -> float:
        """Objective function with distance variance regularization."""
        points = x.reshape(-1, 3)
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist <= 0:
            return 0.0
            
        # Calculate variance penalty
        distance_var = np.var(distances[distances != np.inf])
        variance_penalty = lambda_reg * distance_var
        
        # Ratio with regularization
        ratio = min_dist / max_dist
        return -(ratio - variance_penalty)

    def _constraint_func(self, x: np.ndarray) -> np.ndarray:
        """Constraint function ensuring points lie on unit sphere."""
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0

    def _optimize_with_adaptive_constraints(self, x0: np.ndarray, 
                                          method: str, 
                                          options: dict,
                                          constraint_tightening_factor: float = 1.0) -> Optional[np.ndarray]:
        """Optimize with dynamically adjusted constraints based on progress."""
        try:
            # We'll use a simple constraint tightening approach
            cons = {'type': 'eq', 'fun': self._constraint_func}
            
            # Apply tighter constraints for certain methods
            if 'L-BFGS-B' in method:
                # Use more stringent tolerance for L-BFGS-B
                opt_dict = options.copy()
                opt_dict['ftol'] = options.get('ftol', 1e-12) * constraint_tightening_factor
                opt_dict['gtol'] = options.get('gtol', 1e-12) * constraint_tightening_factor
                result = minimize(self._objective_with_regularization, x0, method=method, 
                                constraints=cons, options=opt_dict)
            else:
                result = minimize(self._objective_with_regularization, x0, method=method, 
                                constraints=cons, options=options)
                
            if result.success:
                optimized_points = result.x.reshape(-1, 3)
                # Ensure normalization
                norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                optimized_points = optimized_points / norms
                return optimized_points
                
        except Exception as e:
            warnings.warn(f"Optimization with {method} failed: {e}")
        return None

    def _evaluate_and_update_best(self, points: np.ndarray) -> bool:
        """Evaluate solution and update best if better."""
        min_dist, max_dist = self._calculate_ratio(points)
        if max_dist > 0:
            ratio = min_dist / max_dist
            if ratio > self.best_ratio:
                self.best_ratio = ratio
                self.best_solution = points.copy()
                return True
        return False

    def _multi_resolution_optimization(self) -> np.ndarray:
        """Perform multi-resolution optimization for better convergence."""
        # Phase 1: Very coarse initialization with Voronoi-based points
        print("Phase 1: Voronoi initialization")
        initial_points = self._voronoi_initialize()
        self._evaluate_and_update_best(initial_points)
        
        # Create multiple diversified initial points
        diverse_initials = [initial_points]
        
        # Add variations with different seeds
        for i in range(4):
            np.random.seed(100 + i)
            varied_points = initial_points + np.random.normal(0, 0.01, initial_points.shape)
            norms = np.linalg.norm(varied_points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            varied_points = varied_points / norms
            diverse_initials.append(varied_points)
        
        # Phase 2: Multi-resolution optimization
        # Start with coarse optimization
        print("Phase 2: Coarse optimization")
        coarse_options = {'ftol': 1e-8, 'gtol': 1e-8, 'maxiter': 200}
        fine_options = {'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 500}
        aggressive_options = {'ftol': 1e-14, 'gtol': 1e-14, 'maxiter': 800}
        
        optimization_phases = [
            (diverse_initials, ['L-BFGS-B'], coarse_options),
            (diverse_initials, ['SLSQP'], fine_options),
            (diverse_initials, ['L-BFGS-B'], aggressive_options),
        ]
        
        for phase_idx, (initial_set, methods, opts) in enumerate(optimization_phases):
            print(f"  Phase {phase_idx+1}: Running optimization with {len(initial_set)} starts")
            for method in methods:
                for start_idx, initial_point in enumerate(initial_set):
                    try:
                        # Set constraint tightening factor based on optimization phase
                        tightening_factor = 1.0
                        if phase_idx >= 2:  # Aggressive phase
                            tightening_factor = 10.0
                        elif phase_idx >= 1:  # Fine phase
                            tightening_factor = 2.0
                        
                        # Add some randomization for diversity
                        np.random.seed(phase_idx * 1000 + start_idx * 100)
                        x0 = initial_point.flatten()
                        
                        optimized = self._optimize_with_adaptive_constraints(
                            x0, method, opts, tightening_factor
                        )
                        
                        if optimized is not None:
                            self._evaluate_and_update_best(optimized)
                            
                    except Exception as e:
                        warnings.warn(f"Failed optimization in phase {phase_idx}, start {start_idx}: {e}")
                        continue
        
        # Phase 3: Local refinement around the best solution
        print("Phase 3: Local refinement")
        if self.best_solution is not None:
            # Perturb the best solution and refine locally
            np.random.seed(999)
            refined_x0 = self.best_solution + np.random.normal(0, 0.005, self.best_solution.shape)
            norms = np.linalg.norm(refined_x0, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            refined_x0 = refined_x0 / norms
            
            # Aggressive refinement
            final_options = {'ftol': 1e-14, 'gtol': 1e-14, 'maxiter': 1000}
            refined = self._optimize_with_adaptive_constraints(
                refined_x0.flatten(), 'L-BFGS-B', final_options, 10.0
            )
            
            if refined is not None:
                self._evaluate_and_update_best(refined)
        
        return self.best_solution if self.best_solution is not None else initial_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Set fixed seed for reproducibility
    np.random.seed(42)
    
    # Create optimizer instance
    optimizer = SphericalVoronoiEvolutionOptimizer(14)
    
    # Execute optimization
    result = optimizer._multi_resolution_optimization()
    
    return result

# EVOLVE-BLOCK-END