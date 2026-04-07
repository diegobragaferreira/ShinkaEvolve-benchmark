# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
import warnings
import time
import random


class PointEvolutionOptimizer:
    """Optimized point distribution optimizer for maximizing min/max distance ratio."""
    
    def __init__(self, n_points: int = 16, dimensions: int = 2, benchmark_ratio: float = 0.2786):
        self.n_points = n_points
        self.dimensions = dimensions
        self.benchmark_ratio = benchmark_ratio
        self.max_time = 180.0
        
    def calculate_min_max_ratio(self, points: np.ndarray) -> float:
        """Calculate the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0.0
            
        distances = pdist(points)
        
        # Handle edge cases
        if len(distances) == 0 or np.max(distances) <= 0:
            return 0.0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max <= 0:
            return 0.0
            
        return d_min / d_max
    
    def initialize_hexagonal_pattern(self) -> np.ndarray:
        """Initialize points using a hexagonal grid pattern."""
        np.random.seed(42)
        points = []
        
        # Create points in a hexagonal lattice pattern
        rows = 4
        cols = 4
        for i in range(rows):
            for j in range(cols):
                if len(points) >= self.n_points:
                    break
                # Hexagonal offset pattern with better spacing
                x = 0.1 + 0.8 * j / (cols - 1) if cols > 1 else 0.5
                y = 0.1 + 0.8 * i / (rows - 1) if rows > 1 else 0.5
                if i % 2 == 1:  # Offset odd rows
                    x += 0.4 / (cols - 1) if cols > 1 else 0.2
                points.append([x, y])
        
        # Normalize and add slight randomness
        points = np.array(points[:self.n_points])
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        return points
    
    def initialize_spiral_pattern(self) -> np.ndarray:
        """Initialize points using a spiral pattern."""
        np.random.seed(42)
        points = []
        
        # Create a spiral pattern that spreads points well
        for i in range(self.n_points):
            if i == 0:
                points.append([0.5, 0.5])  # Center point
            else:
                angle = i * 2.5  # Angle in radians
                radius = min(0.4, i * 0.05)  # Radius increases gradually
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                points.append([x, y])
        
        points = np.array(points)
        points = np.clip(points, 0, 1)
        return points
    
    def initialize_grid_pattern(self) -> np.ndarray:
        """Initialize points using a regular grid pattern with jitter."""
        np.random.seed(42)
        points = []
        
        # Regular 4x4 grid
        x_vals = np.linspace(0.1, 0.9, 4)
        y_vals = np.linspace(0.1, 0.9, 4)
        
        for i in range(4):
            for j in range(4):
                if len(points) >= self.n_points:
                    break
                points.append([x_vals[i], y_vals[j]])
        
        points = np.array(points[:self.n_points])
        points += np.random.normal(0, 0.015, points.shape)
        points = np.clip(points, 0, 1)
        return points
    
    def initialize_random_pattern(self) -> np.ndarray:
        """Initialize points using random uniform distribution."""
        np.random.seed(42)
        return np.random.rand(self.n_points, self.dimensions)
    
    def initialize_corner_pattern(self) -> np.ndarray:
        """Initialize points using corner-based pattern."""
        np.random.seed(42)
        points = np.array([
            [0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9],
            [0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5],
            [0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75],
            [0.33, 0.33], [0.33, 0.67], [0.67, 0.33], [0.67, 0.67]
        ])
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        return points
    
    def generate_initial_configurations(self) -> list:
        """Generate multiple diverse initial configurations."""
        configs = [
            self.initialize_hexagonal_pattern(),
            self.initialize_spiral_pattern(),
            self.initialize_grid_pattern(),
            self.initialize_random_pattern(),
            self.initialize_corner_pattern()
        ]
        return configs
    
    def objective_function(self, x: np.ndarray) -> float:
        """Objective function that minimizes negative ratio."""
        points = x.reshape(-1, self.dimensions)
        
        # Calculate pairwise distances using squareform for numerical stability
        distances = squareform(pdist(points))
        
        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(distances, np.inf)
        
        # Return negative ratio (since we want to maximize ratio, we minimize its negative)
        valid_distances = distances[distances != np.inf]
        if len(valid_distances) == 0:
            return -1.0  # Worst possible case
            
        min_dist = np.min(valid_distances)
        max_dist = np.max(valid_distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return -1.0
            
        return -min_dist / max_dist
    
    def constraint_function(self, x: np.ndarray) -> np.ndarray:
        """Constraint function ensuring points stay within bounds."""
        # Ensure points are within [0+eps,1-eps] x [0+eps,1-eps] for numerical stability
        eps = 1e-8
        points = x.reshape(-1, self.dimensions)
        constraints = []
        
        # x coordinates in [eps, 1-eps]
        constraints.append(points[:, 0].min() - eps)  # x_min >= eps
        constraints.append(1 - eps - points[:, 0].max())  # x_max <= 1-eps
        
        # y coordinates in [eps, 1-eps]
        constraints.append(points[:, 1].min() - eps)  # y_min >= eps
        constraints.append(1 - eps - points[:, 1].max())  # y_max <= 1-eps
        
        return np.array(constraints)
    
    def bounded_objective(self, x: np.ndarray) -> float:
        """Objective function with boundary checking and clamping."""
        eps = 1e-8
        points = np.clip(x.reshape(-1, self.dimensions), eps, 1-eps).flatten()
        return self.objective_function(points)
    
    def global_optimization(self, initial_points: np.ndarray) -> tuple:
        """Perform global optimization using differential evolution."""
        bounds = [(1e-8, 1-1e-8) for _ in range(len(initial_points.flatten()))]
        
        try:
            result = differential_evolution(
                self.bounded_objective,
                bounds,
                seed=42,
                maxiter=100,
                popsize=20,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                disp=False
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, self.dimensions)
                optimized_points = np.clip(optimized_points, 1e-8, 1-1e-8)
                ratio = self.calculate_min_max_ratio(optimized_points)
                return optimized_points, ratio
        except Exception:
            pass
            
        return initial_points, self.calculate_min_max_ratio(initial_points)
    
    def local_optimization(self, points: np.ndarray, method: str = 'SLSQP') -> tuple:
        """Perform local optimization with specified method."""
        bounds = [(1e-8, 1-1e-8) for _ in range(len(points.flatten()))]
        
        try:
            if method == 'SLSQP':
                result = minimize(
                    self.bounded_objective,
                    points.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    constraints={'type': 'ineq', 'fun': self.constraint_function},
                    options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-9}
                )
            elif method == 'L-BFGS-B':
                result = minimize(
                    self.bounded_objective,
                    points.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200}
                )
            
            if result.success:
                optimized_points = result.x.reshape(-1, self.dimensions)
                optimized_points = np.clip(optimized_points, 1e-8, 1-1e-8)
                ratio = self.calculate_min_max_ratio(optimized_points)
                return optimized_points, ratio
        except Exception:
            pass
            
        return points, self.calculate_min_max_ratio(points)
    
    def simulated_annealing_optimization(self, initial_points: np.ndarray, max_iter: int = 500) -> tuple:
        """Use simulated annealing as fallback optimization method."""
        current_points = initial_points.copy()
        current_ratio = -self.objective_function(current_points.flatten())  # Negative since we're minimizing
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Initial temperature and cooling schedule
        temperature = 0.1
        min_temperature = 1e-6
        
        for iteration in range(max_iter):
            # Generate neighbor solution by perturbing a random point
            neighbor_points = current_points.copy()
            rand_idx = random.randint(0, self.n_points - 1)
            # Add small random perturbation
            neighbor_points[rand_idx] += np.random.normal(0, 0.01, 2)
            # Keep within bounds
            neighbor_points = np.clip(neighbor_points, 0, 1)
            
            # Calculate ratio for neighbor
            neighbor_ratio = -self.objective_function(neighbor_points.flatten())  # Negative since we're minimizing
            
            # Accept or reject based on Metropolis criterion
            if neighbor_ratio > current_ratio:
                current_points = neighbor_points.copy()
                current_ratio = neighbor_ratio
                if neighbor_ratio > best_ratio:
                    best_points = neighbor_points.copy()
                    best_ratio = neighbor_ratio
            else:
                # Accept with probability based on temperature
                delta = neighbor_ratio - current_ratio
                acceptance_prob = np.exp(delta / temperature)
                if random.random() < acceptance_prob:
                    current_points = neighbor_points.copy()
                    current_ratio = neighbor_ratio
            
            # Cool down temperature
            temperature *= 0.95
            if temperature < min_temperature:
                break
        
        return best_points, best_ratio
    
    def optimize(self) -> np.ndarray:
        """Main optimization process."""
        # Generate diverse initial configurations
        initial_configs = self.generate_initial_configurations()
        
        best_ratio = -np.inf
        best_points = None
        
        # Try multiple initial configurations with hybrid optimization
        for i, initial_config in enumerate(initial_configs):
            try:
                # Phase 1: Global optimization with Differential Evolution
                global_points, global_ratio = self.global_optimization(initial_config)
                
                # Phase 2: Local optimization with SLSQP for fine-tuning
                local_points, local_ratio = self.local_optimization(global_points, 'SLSQP')
                
                # If SLSQP fails, try L-BFGS-B as fallback
                if local_ratio <= global_ratio:
                    fallback_points, fallback_ratio = self.local_optimization(global_points, 'L-BFGS-B')
                    if fallback_ratio > local_ratio:
                        local_points, local_ratio = fallback_points, fallback_ratio
                
                # Track the best solution found
                if local_ratio > best_ratio:
                    best_ratio = local_ratio
                    best_points = local_points.copy()
                    
            except Exception as e:
                continue
        
        # Fallback to simulated annealing if optimization didn't work well
        if best_points is None or best_ratio < 0.1:
            try:
                initial_config = self.generate_initial_configurations()[0]  # Use first config as fallback
                sa_points, sa_ratio = self.simulated_annealing_optimization(initial_config, max_iter=500)
                if sa_ratio > best_ratio:
                    best_ratio = sa_ratio
                    best_points = sa_points.copy()
            except Exception as e:
                pass
        
        # If we still don't have a good solution, use a fallback approach
        if best_points is None:
            # Use the most promising initial configuration with more aggressive local optimization
            fallback_config = self.generate_initial_configurations()[0]
            final_points, final_ratio = self.local_optimization(fallback_config, 'SLSQP')
            
            if final_ratio > best_ratio:
                best_points = final_points
            else:
                best_points = fallback_config
        
        # Final safety check - ensure points are within bounds
        if best_points is not None:
            best_points = np.clip(best_points, 1e-8, 1-1e-8)
        else:
            # Last resort: return a default configuration
            best_points = self.generate_initial_configurations()[0]
        
        return best_points


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    optimizer = PointEvolutionOptimizer(n_points=16, dimensions=2)
    return optimizer.optimize()


# EVOLVE-BLOCK-END