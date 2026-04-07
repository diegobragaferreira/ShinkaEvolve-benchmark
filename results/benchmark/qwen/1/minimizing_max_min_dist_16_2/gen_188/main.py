# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution
from typing import Tuple, Optional, List
import time

class OptimizationConfig:
    """Configuration class for optimization parameters"""
    def __init__(self):
        self.lbfgs_maxiter = 1000
        self.lbfgs_ftol = 1e-10
        self.lbfgs_gtol = 1e-10
        self.de_maxiter = 150
        self.de_popsize = 20
        self.de_mutation = (0.5, 1)
        self.de_recombination = 0.7
        self.bounds_epsilon = 1e-6
        self.initial_perturbation = 0.15
        self.hexagonal_padding = 0.02

class PointInitializationStrategy:
    """Handles different point initialization strategies"""
    
    @staticmethod
    def hexagonal_grid_with_perturbation(n_points: int = 16, 
                                       seed: int = 42,
                                       perturbation_scale: float = 0.15,
                                       padding: float = 0.02) -> np.ndarray:
        """Create hexagonal grid with strategic perturbations"""
        np.random.seed(seed)
        
        points = []
        rows, cols = 4, 4
        
        for i in range(rows):
            for j in range(cols):
                # Hexagonal grid with proper spacing
                x = (j + 0.5 * (i % 2)) / (cols - 1) if cols > 1 else 0.5
                y = i / (rows - 1) if rows > 1 else 0.5
                
                # Add controlled random perturbation
                x += (np.random.rand() - 0.5) * perturbation_scale
                y += (np.random.rand() - 0.5) * perturbation_scale
                
                # Ensure points stay within safe boundaries with padding
                x = np.clip(x, padding, 1 - padding)
                y = np.clip(y, padding, 1 - padding)
                
                points.append([x, y])
        
        return np.array(points[:n_points])

    @staticmethod
    def random_centered(n_points: int = 16, 
                      seed: int = 42,
                      padding: float = 0.1) -> np.ndarray:
        """Create random points centered away from edges"""
        np.random.seed(seed)
        points = np.random.rand(n_points, 2) * (1 - 2*padding) + padding
        return points

class OptimizationStrategy:
    """Handles different optimization strategies"""
    
    @staticmethod
    def lbfgs_optimize(objective_func, x0: np.ndarray, 
                      bounds: List[Tuple[float, float]],
                      constraints: dict,
                      config: OptimizationConfig) -> np.ndarray:
        """Perform L-BFGS-B optimization with error handling"""
        try:
            result = minimize(
                objective_func,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                constraints=constraints,
                options={
                    'maxiter': config.lbfgs_maxiter,
                    'ftol': config.lbfgs_ftol,
                    'gtol': config.lbfgs_gtol
                }
            )
            return result.x if result.success else x0
        except Exception:
            return x0
    
    @staticmethod
    def differential_evolution_optimize(objective_func, 
                                      bounds: List[Tuple[float, float]],
                                      config: OptimizationConfig) -> np.ndarray:
        """Perform Differential Evolution optimization"""
        try:
            result = differential_evolution(
                objective_func,
                bounds,
                maxiter=config.de_maxiter,
                popsize=config.de_popsize,
                mutation=config.de_mutation,
                recombination=config.de_recombination,
                seed=42,
                disp=False
            )
            return result.x
        except Exception:
            return None

class SolutionValidator:
    """Handles solution validation and final formatting"""
    
    @staticmethod
    def validate_and_finalize(points_flat: np.ndarray, 
                            config: OptimizationConfig,
                            dimension: int = 2) -> np.ndarray:
        """Ensure solution respects constraints and is numerically stable"""
        points = points_flat.reshape(-1, dimension)
        
        # Clamp points to valid range with epsilon to avoid boundary issues
        points = np.clip(points, config.bounds_epsilon, 1 - config.bounds_epsilon)
        
        return points.flatten()

class PointDispersionOptimizer:
    """Main optimizer class implementing multi-strategy approach"""
    
    def __init__(self, n_points: int = 16, dimension: int = 2):
        self.n_points = n_points
        self.dimension = dimension
        self.total_vars = n_points * dimension
        self.config = OptimizationConfig()
        
    def calculate_objective(self, x_flat: np.ndarray) -> float:
        """Calculate the negative min/max distance ratio with robust error handling"""
        points = x_flat.reshape(-1, self.dimension)
        
        # Calculate pairwise distances using squareform for numerical stability
        distances = squareform(pdist(points))
        
        # Mask diagonal elements (distance to itself) with infinity
        np.fill_diagonal(distances, np.inf)
        
        # Handle edge case where there are no distances or all infinities
        if distances.size == 0 or np.all(distances == np.inf):
            return -1.0
            
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Return negative ratio to maximize (since we're minimizing)
        # Handle edge case to avoid division by zero or invalid distances
        if d_max <= 0:
            return -1.0
        return -d_min / d_max
    
    def create_bounds_constraints(self, x_flat: np.ndarray) -> np.ndarray:
        """Create bound constraints ensuring all points stay within [0,1] bounds"""
        points = x_flat.reshape(-1, self.dimension)
        
        # Create constraint vectors for lower and upper bounds
        lower_bounds = -points.flatten()  # For points to be >= 0  
        upper_bounds = points.flatten() - 1.0  # For points to be <= 1
        
        return np.concatenate([lower_bounds, upper_bounds])
    
    def initialize_points(self) -> np.ndarray:
        """Initialize points using enhanced hexagonal arrangement"""
        return PointInitializationStrategy.hexagonal_grid_with_perturbation(
            n_points=self.n_points,
            seed=42,
            perturbation_scale=self.config.initial_perturbation,
            padding=self.config.hexagonal_padding
        )
    
    def optimize(self) -> np.ndarray:
        """Main optimization routine with multi-strategy approach"""
        # Phase 1: Initialize points
        initial_points = self.initialize_points()
        x0 = initial_points.flatten()
        
        # Phase 2: Setup optimization parameters
        bounds = [(0, 1) for _ in range(self.total_vars)]
        constraints = {'type': 'ineq', 'fun': self.create_bounds_constraints}
        
        # Phase 3: Multi-stage optimization with fallbacks
        best_solution = None
        best_objective_value = float('inf')
        
        # Strategy 1: Local optimization with L-BFGS-B
        try:
            x_lbfgs = OptimizationStrategy.lbfgs_optimize(
                self.calculate_objective, 
                x0, 
                bounds, 
                constraints, 
                self.config
            )
            obj_lbfgs = self.calculate_objective(x_lbfgs)
            
            if obj_lbfgs < best_objective_value:
                best_objective_value = obj_lbfgs
                best_solution = x_lbfgs
                
        except Exception:
            pass
        
        # Strategy 2: Global optimization with Differential Evolution
        try:
            x_de = OptimizationStrategy.differential_evolution_optimize(
                self.calculate_objective,
                bounds,
                self.config
            )
            
            if x_de is not None:
                obj_de = self.calculate_objective(x_de)
                if obj_de < best_objective_value:
                    best_objective_value = obj_de
                    best_solution = x_de
                    
        except Exception:
            pass
        
        # Strategy 3: Fallback to initial solution if no optimization succeeded
        if best_solution is None:
            best_solution = x0
        
        # Phase 4: Final validation and cleanup
        validated_solution = SolutionValidator.validate_and_finalize(
            best_solution, 
            self.config, 
            self.dimension
        )
        
        # Convert to final point array format
        final_points = validated_solution.reshape(-1, self.dimension)
        
        return final_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = PointDispersionOptimizer(n_points=16, dimension=2)
    return optimizer.optimize()

# EVOLVE-BLOCK-END