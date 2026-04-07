# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution
import time

class AdaptiveHexSpiralOptimizer:
    """Optimizes point placement to maximize min/max distance ratio using adaptive multi-strategy initialization and refinement."""
    
    def __init__(self, n_points=16, dimension=2):
        self.n_points = n_points
        self.dimension = dimension
        self.total_vars = n_points * dimension
        np.random.seed(42)
    
    def generate_hexagonal_grid_initialization(self):
        """Generate hexagonal grid initialization."""
        points = []
        rows = 4
        cols = 4
        
        for i in range(rows):
            for j in range(cols):
                if len(points) < self.n_points:
                    x = (j + 0.5 * (i % 2)) / (cols - 1) if cols > 1 else 0.5
                    y = i / (rows - 1) if rows > 1 else 0.5
                    
                    # Add small random perturbation
                    x += (np.random.rand() - 0.5) * 0.1
                    y += (np.random.rand() - 0.5) * 0.1
                    
                    # Ensure within bounds
                    x = np.clip(x, 0.02, 0.98)
                    y = np.clip(y, 0.02, 0.98)
                    
                    points.append([x, y])
        
        return np.array(points[:self.n_points])
    
    def generate_golden_spiral_initialization(self):
        """Generate golden spiral initialization for better point distribution."""
        points = []
        
        # Golden ratio spiral
        phi = (1 + np.sqrt(5)) / 2
        for i in range(self.n_points):
            angle = i * 2 * np.pi / phi
            radius = i / (self.n_points - 1) * 0.4 + 0.1
            
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            
            # Add slight random perturbation
            x += (np.random.rand() - 0.5) * 0.05
            y += (np.random.rand() - 0.5) * 0.05
            
            # Ensure within bounds
            x = np.clip(x, 0.02, 0.98)
            y = np.clip(y, 0.02, 0.98)
            
            points.append([x, y])
        
        return np.array(points)
    
    def generate_random_initialization(self):
        """Generate random initialization with good spread."""
        points = []
        for i in range(self.n_points):
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            points.append([x, y])
        return np.array(points)
    
    def generate_multi_strategy_initialization(self):
        """Generate multiple initializations and select the best."""
        initializations = []
        initializations.append(self.generate_hexagonal_grid_initialization())
        initializations.append(self.generate_golden_spiral_initialization())
        initializations.append(self.generate_random_initialization())
        
        best_init = None
        best_ratio = -np.inf
        
        for init in initializations:
            ratio = self.compute_min_max_ratio(init)
            if ratio > best_ratio:
                best_ratio = ratio
                best_init = init
        
        return best_init
    
    def compute_min_max_ratio(self, points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0
        
        # Compute pairwise distances using squareform for numerical stability
        distances = squareform(pdist(points))
        
        # Mask diagonal elements (distance to itself)
        np.fill_diagonal(distances, np.inf)
        
        # Get min and max distances
        dmin = np.min(distances)
        dmax = np.max(distances)
        
        # Avoid division by zero
        if dmax == 0:
            return 0
            
        return dmin / dmax
    
    def adaptive_objective(self, x_flat):
        """Adaptive objective function with improved numerical handling."""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        
        # Ensure points are within bounds with epsilon padding
        points = np.clip(points, 1e-6, 1-1e-6)
        
        # Calculate pairwise distances using squareform for numerical stability
        distances = squareform(pdist(points))
        
        # Mask diagonal elements (distance to itself)
        np.fill_diagonal(distances, np.inf)
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Return negative ratio to maximize (since we're minimizing)
        if d_max <= 0:
            return -1.0
            
        # Add regularization to avoid degenerate solutions
        ratio = d_min / d_max
        if ratio < 1e-12:  # Very small ratios penalize heavily
            return -1.0
            
        return -ratio
    
    def multi_stage_optimization(self, x_start):
        """Perform multi-stage optimization with adaptive parameters."""
        current_solution = x_start.copy()
        
        # Stage 1: Coarse optimization with loose tolerances
        try:
            result = minimize(
                self.adaptive_objective,
                current_solution,
                method='L-BFGS-B',
                bounds=[(1e-6, 1-1e-6) for _ in range(self.total_vars)],
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            if result.success:
                current_solution = result.x
        except Exception:
            pass
        
        # Stage 2: Medium optimization with moderate tolerances
        try:
            result = minimize(
                self.adaptive_objective,
                current_solution,
                method='L-BFGS-B',
                bounds=[(1e-6, 1-1e-6) for _ in range(self.total_vars)],
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            if result.success:
                current_solution = result.x
        except Exception:
            pass
        
        # Stage 3: Fine optimization with strict tolerances
        try:
            result = minimize(
                self.adaptive_objective,
                current_solution,
                method='L-BFGS-B',
                bounds=[(1e-6, 1-1e-6) for _ in range(self.total_vars)],
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if result.success:
                current_solution = result.x
        except Exception:
            pass
            
        return current_solution
    
    def global_search_optimization(self, x_start):
        """Perform global search using Differential Evolution."""
        bounds = [(1e-6, 1-1e-6) for _ in range(self.total_vars)]
        
        try:
            result = differential_evolution(
                self.adaptive_objective,
                bounds,
                maxiter=50,
                popsize=10,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False
            )
            if result.success:
                return result.x
        except Exception:
            pass
            
        return x_start
    
    def validate_solution(self, points_flat):
        """Validate and finalize the solution."""
        points = points_flat.reshape(-1, 2)
        
        # Clamp points to [0,1] range with small epsilon to avoid boundary issues
        points = np.clip(points, 1e-6, 1 - 1e-6)
        
        return points.flatten()
    
    def optimize(self):
        """Main optimization routine with multi-strategy approach."""
        # Phase 1: Generate high-quality initial configuration using multiple strategies
        initial_points = self.generate_multi_strategy_initialization()
        
        # Phase 2: Multi-stage refinement
        current_solution = initial_points.flatten()
        
        # Try global search first to escape local optima
        global_result = self.global_search_optimization(current_solution)
        global_ratio = self.compute_min_max_ratio(global_result.reshape(-1, 2))
        
        # Then apply multi-stage local optimization
        local_result = self.multi_stage_optimization(current_solution)
        local_ratio = self.compute_min_max_ratio(local_result.reshape(-1, 2))
        
        # Compare results and choose the better one
        if global_ratio > local_ratio:
            final_solution = global_result
        else:
            final_solution = local_result
        
        # Phase 3: Final validation and boundary correction
        validated_solution = self.validate_solution(final_solution)
        
        # Convert to final point array format
        final_points = validated_solution.reshape(-1, 2)
        
        # Double-check final ratio
        final_ratio = self.compute_min_max_ratio(final_points)
        
        # If we got a very poor result, fallback to initial configuration
        if final_ratio < 1e-10:
            final_points = initial_points
            
        return final_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = AdaptiveHexSpiralOptimizer(n_points=16, dimension=2)
    return optimizer.optimize()

# EVOLVE-BLOCK-END