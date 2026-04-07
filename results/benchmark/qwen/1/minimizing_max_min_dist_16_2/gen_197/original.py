# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
import warnings
from typing import List, Tuple, Optional, Any

class PointDistributionOptimizer:
    """A structured optimizer for maximizing the min/max distance ratio of point distributions."""
    
    def __init__(self, num_points: int = 16, bounds: Tuple[float, float] = (0.01, 0.99)):
        self.num_points = num_points
        self.bounds = bounds
        self.dimension = 2
        
    def _compute_distance_matrix(self, points: np.ndarray) -> np.ndarray:
        """Compute full pairwise distance matrix with proper numerical handling."""
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        return distances
    
    def _objective_function(self, x: np.ndarray) -> float:
        """Core objective function that computes the negative min/max distance ratio."""
        points = x.reshape(-1, self.dimension)
        points = np.clip(points, self.bounds[0], self.bounds[1])
        
        try:
            distance_matrix = self._compute_distance_matrix(points)
            d_min = np.min(distance_matrix)
            d_max = np.max(distance_matrix)
            
            if d_max <= 1e-12:
                return -np.inf
                
            return -d_min / d_max
        except Exception as e:
            warnings.warn(f"Error in objective computation: {e}")
            return -np.inf
    
    def _create_geometric_initializations(self) -> List[np.ndarray]:
        """Generate diverse geometric initial configurations."""
        initial_guesses = []
        
        # 1. Hexagonal grid initialization
        hex_points = self._generate_hexagonal_grid()
        initial_guesses.append(hex_points.flatten())
        
        # 2. Golden spiral initialization
        spiral_points = self._generate_golden_spiral()
        initial_guesses.append(spiral_points.flatten())
        
        # 3. Regular grid with perturbation
        grid_points = self._generate_perturbed_grid()
        initial_guesses.append(grid_points.flatten())
        
        # 4. Random initialization with fixed seed
        np.random.seed(42)
        random_points = np.random.rand(self.num_points, self.dimension)
        random_points = np.clip(random_points, self.bounds[0], self.bounds[1])
        initial_guesses.append(random_points.flatten())
        
        # 5. Another random initialization with different seed
        np.random.seed(246)
        random_points_2 = np.random.rand(self.num_points, self.dimension)
        random_points_2 = np.clip(random_points_2, self.bounds[0], self.bounds[1])
        initial_guesses.append(random_points_2.flatten())
        
        return initial_guesses
    
    def _generate_hexagonal_grid(self) -> np.ndarray:
        """Generate points in a hexagonal grid pattern."""
        rows, cols = 4, 4
        sqrt3 = np.sqrt(3)
        spacing = 0.8
        
        points = []
        for i in range(rows):
            for j in range(cols):
                if len(points) >= self.num_points:
                    break
                    
                x = j * spacing + (i % 2) * spacing * 0.5
                y = i * spacing * sqrt3 / 2
                
                x_scaled = 0.05 + (x / (spacing * cols)) * 0.9
                y_scaled = 0.05 + (y / (spacing * rows * sqrt3 / 2)) * 0.9
                
                # Add small random perturbation
                x_scaled += np.random.normal(0, 0.01)
                y_scaled += np.random.normal(0, 0.01)
                
                points.append([x_scaled, y_scaled])
                
        return np.array(points[:self.num_points])
    
    def _generate_golden_spiral(self) -> np.ndarray:
        """Generate points in a golden spiral pattern."""
        golden_angle = 2.399963229728653  # ~4π/(3+√5)
        points = []
        
        for i in range(self.num_points):
            radius = np.sqrt(i/(self.num_points-1)) if self.num_points > 1 else 0
            angle = i * golden_angle
            x = 0.5 + radius * np.cos(angle) * 0.45
            y = 0.5 + radius * np.sin(angle) * 0.45
            points.append([x, y])
            
        return np.array(points)
    
    def _generate_perturbed_grid(self) -> np.ndarray:
        """Generate a perturbed regular grid."""
        grid_points = np.array([[i/3, j/3] for i in range(4) for j in range(4) 
                               if i*4+j < self.num_points]).reshape(-1, 2)
        perturbed_grid = grid_points + np.random.normal(0, 0.05, (self.num_points, 2))
        perturbed_grid = np.clip(perturbed_grid, self.bounds[0], self.bounds[1])
        return perturbed_grid
    
    def _optimize_stage(self, objective_func, x0: np.ndarray, 
                       method: str = 'L-BFGS-B') -> Tuple[bool, np.ndarray, float]:
        """Execute a single optimization stage."""
        bounds = [(self.bounds[0], self.bounds[1]) for _ in range(self.num_points * self.dimension)]
        
        try:
            result = minimize(
                objective_func,
                x0,
                method=method,
                bounds=bounds,
                options={'ftol': 1e-13, 'gtol': 1e-13, 'maxiter': 1000}
            )
            
            return result.success, result.x, result.fun
        except Exception as e:
            warnings.warn(f"Optimization failed with {method}: {e}")
            return False, x0, np.inf
    
    def _multi_start_optimization(self) -> np.ndarray:
        """Perform multi-start optimization with structured approach."""
        initial_guesses = self._create_geometric_initializations()
        bounds = [(self.bounds[0], self.bounds[1]) for _ in range(self.num_points * self.dimension)]
        
        best_ratio = -np.inf
        best_solution = None
        
        for i, initial_guess in enumerate(initial_guesses):
            try:
                # Global optimization with differential evolution
                de_result = differential_evolution(
                    self._objective_function,
                    bounds,
                    seed=42 + i,
                    maxiter=200,
                    popsize=30,
                    tol=1e-9,
                    recombination=0.9,
                    mutation=(0.8, 1.0),
                    disp=False
                )
                
                # Local refinement with L-BFGS-B
                success, refined_x, refined_fun = self._optimize_stage(
                    self._objective_function, 
                    de_result.x, 
                    'L-BFGS-B'
                )
                
                # If L-BFGS-B fails, try SLSQP
                if not success:
                    success, refined_x, refined_fun = self._optimize_stage(
                        self._objective_function, 
                        de_result.x, 
                        'SLSQP'
                    )
                
                # Track best solution
                if success and -refined_fun > best_ratio:
                    best_ratio = -refined_fun
                    best_solution = refined_x
                    
            except Exception as e:
                warnings.warn(f"Multi-start attempt {i} failed: {e}")
                continue
        
        # Final refinement if we found a good starting point
        if best_solution is not None:
            final_success, final_x, final_fun = self._optimize_stage(
                self._objective_function, 
                best_solution, 
                'L-BFGS-B'
            )
            
            if final_success:
                return final_x.reshape(-1, self.dimension)
        
        # Fallback to first initial guess if nothing worked well
        return initial_guesses[0].reshape(-1, self.dimension)
    
    def optimize(self) -> np.ndarray:
        """Main optimization entry point."""
        points = self._multi_start_optimization()
        return np.clip(points, 0, 1)

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = PointDistributionOptimizer(num_points=16, bounds=(0.01, 0.99))
    return optimizer.optimize()

# EVOLVE-BLOCK-END