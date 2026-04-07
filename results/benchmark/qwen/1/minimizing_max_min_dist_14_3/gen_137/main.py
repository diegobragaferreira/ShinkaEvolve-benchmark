# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time
from typing import List, Tuple, Callable, Any
import warnings

class PointArrangementOptimizer:
    """Structured optimizer for maximizing min/max distance ratio of 14 3D points."""
    
    def __init__(self, num_points: int = 14, dimension: int = 3):
        self.num_points = num_points
        self.dimension = dimension
        self.best_solution = None
        self.best_ratio = -np.inf
        
    def initialize_points(self, method: str = 'sobol') -> np.ndarray:
        """Generate initial point configurations using various methods."""
        if method == 'icosahedron':
            return self._icosahedron_initialization()
        elif method == 'fibonacci':
            return self._fibonacci_initialization()
        elif method == 'sobol':
            return self._sobol_initialization()
        elif method == 'random':
            return self._random_initialization()
        else:
            raise ValueError(f"Unknown initialization method: {method}")
    
    def _icosahedron_initialization(self) -> np.ndarray:
        """Generate points using icosahedron-based construction."""
        # Vertices of a regular icosahedron
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])

        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

        # If we need more than 12 points, distribute additional points
        if self.num_points <= 12:
            # Just return subset of vertices
            return vertices[:self.num_points]
        else:
            # For more points, we'll start with icosahedron vertices and add more
            points = vertices.copy()

            # Add more points for better distribution
            additional_points = [
                [0, 0, 1], [0, 0, -1],  # poles
                [1, 0, 0], [-1, 0, 0],  # x-axis
                [0, 1, 0], [0, -1, 0]   # y-axis
            ]
            
            points = np.vstack([points, additional_points[:self.num_points - 12]])
            
            # Apply perturbation to ensure good distribution
            np.random.seed(42)
            points += np.random.normal(0, 0.05, (points.shape[0], 3))
            
            # Normalize again to maintain unit sphere
            norms = np.linalg.norm(points, axis=1)
            points = points / np.maximum(norms[:, np.newaxis], 1e-12)
            
            return points[:self.num_points]
    
    def _fibonacci_initialization(self) -> np.ndarray:
        """Generate points on a sphere using Fibonacci spiral method."""
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle

        for i in range(self.num_points):
            y = 1 - (i / float(self.num_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(max(0, 1 - y * y))  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)
    
    def _sobol_initialization(self) -> np.ndarray:
        """Generate points on sphere using 3D Sobol sequence or fallback."""
        try:
            # Try to import sobol sequence generator
            from sobol_seq import i4_sobol_generate

            # Generate Sobol points in [0,1]^3
            sobol_points = i4_sobol_generate(3, self.num_points)

            # Convert to sphere using spherical coordinates
            points = np.zeros((self.num_points, 3))

            # Use the Sobol points to create well-distributed points on sphere
            for i in range(self.num_points):
                # Map to sphere using similar approach as Fibonacci
                u = sobol_points[i, 0]  # Uniform random in [0,1]
                v = sobol_points[i, 1]  # Uniform random in [0,1]

                # Use these as parameters for spherical coordinates
                theta = 2 * np.pi * u  # azimuthal angle
                phi = np.arccos(2 * v - 1)  # polar angle

                # Convert to Cartesian
                x = np.sin(phi) * np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi)

                points[i] = [x, y, z]

            return points

        except ImportError:
            # Fallback to fibonacci if sobol not available
            return self._fibonacci_initialization()
    
    def _random_initialization(self) -> np.ndarray:
        """Generate random points on unit sphere."""
        np.random.seed(42)
        points = np.random.randn(self.num_points, self.dimension)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / np.maximum(norms, 1e-12)
    
    def calculate_min_max_ratio(self, points: np.ndarray) -> float:
        """Calculate the ratio of minimum to maximum distance."""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist < 1e-12:
            return 0.0
        return min_dist / max_dist
    
    def _constraint_sphere(self, x_flat: np.ndarray) -> float:
        """Constraint function to keep points on unit sphere."""
        points = x_flat.reshape(-1, self.dimension)
        norms = np.linalg.norm(points, axis=1)
        # Return mean squared difference from unit sphere
        return np.mean((norms - 1.0)**2)
    
    def _objective_function(self, x_flat: np.ndarray) -> float:
        """Objective function to maximize min/max distance ratio."""
        points = x_flat.reshape(-1, self.dimension)
        
        # Ensure points are on unit sphere with better numerical stability
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        safe_norms = np.where(norms == 0, 1, norms)
        normalized_points = points / safe_norms
        
        ratio = self.calculate_min_max_ratio(normalized_points)
        return -ratio  # Negative because we want to maximize
    
    def _optimize_single_run(self, x0: np.ndarray) -> Tuple[np.ndarray, float]:
        """Perform optimization with multiple algorithms on given initial points."""
        best_points = x0.copy()
        best_ratio = self.calculate_min_max_ratio(x0.reshape(-1, self.dimension))
        
        # Define bounds and constraints
        bounds = [(-2, 2) for _ in range(self.num_points * self.dimension)]
        constraints = {'type': 'eq', 'fun': self._constraint_sphere}
        
        # Try SLSQP for global convergence
        try:
            result_slsqp = minimize(
                self._objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9},
                tol=1e-9
            )
            
            if result_slsqp.success:
                optimized_points = result_slsqp.x.reshape(-1, self.dimension)
                # Ensure normalization
                norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
                safe_norms = np.where(norms == 0, 1, norms)
                normalized_points = optimized_points / safe_norms
                
                ratio = self.calculate_min_max_ratio(normalized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = normalized_points.copy()
                    
        except Exception as e:
            warnings.warn(f"SLSQP optimization failed: {e}")
        
        # Try L-BFGS-B for local refinement
        try:
            result_lbfgsb = minimize(
                self._objective_function,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10},
                tol=1e-10
            )
            
            if result_lbfgsb.success:
                optimized_points = result_lbfgsb.x.reshape(-1, self.dimension)
                # Ensure normalization
                norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
                safe_norms = np.where(norms == 0, 1, norms)
                normalized_points = optimized_points / safe_norms
                
                ratio = self.calculate_min_max_ratio(normalized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = normalized_points.copy()
                    
        except Exception as e:
            warnings.warn(f"L-BFGS-B optimization failed: {e}")
        
        return best_points, best_ratio
    
    def optimize(self) -> Tuple[np.ndarray, dict]:
        """Main optimization procedure with multiple strategies."""
        # Strategy 1: Multiple initialization methods
        init_strategies = ['icosahedron', 'fibonacci', 'sobol', 'random']
        results = []
        
        for strategy in init_strategies:
            try:
                # Generate initial points
                initial_points = self.initialize_points(strategy)
                
                # Add noise to break symmetry
                np.random.seed(42)
                noisy_points = initial_points + np.random.normal(0, 0.02, initial_points.shape)
                
                # Ensure they are on unit sphere
                norms = np.linalg.norm(noisy_points, axis=1, keepdims=True)
                normalized_points = noisy_points / np.maximum(norms, 1e-12)
                
                # Flatten for optimization
                x0 = normalized_points.flatten()
                
                # Single optimization run
                optimized_points, ratio = self._optimize_single_run(x0)
                
                results.append((strategy, optimized_points, ratio))
                
                # Update global best
                if ratio > self.best_ratio:
                    self.best_ratio = ratio
                    self.best_solution = optimized_points.copy()
                    
            except Exception as e:
                warnings.warn(f"Strategy {strategy} failed: {e}")
                continue
        
        # Strategy 2: Additional restarts with different random seeds
        for seed in [123, 456, 789]:
            try:
                # Random initialization with different seed
                np.random.seed(seed)
                random_points = np.random.randn(self.num_points, self.dimension)
                norms = np.linalg.norm(random_points, axis=1, keepdims=True)
                normalized_random = random_points / np.maximum(norms, 1e-12)
                
                x0 = normalized_random.flatten()
                
                # Single optimization run
                optimized_points, ratio = self._optimize_single_run(x0)
                
                results.append((f"restart_{seed}", optimized_points, ratio))
                
                # Update global best
                if ratio > self.best_ratio:
                    self.best_ratio = ratio
                    self.best_solution = optimized_points.copy()
                    
            except Exception as e:
                warnings.warn(f"Random restart seed {seed} failed: {e}")
                continue
        
        # If we still have no good solution, fallback to simple random points
        if self.best_solution is None:
            np.random.seed(42)
            random_points = np.random.randn(self.num_points, self.dimension)
            norms = np.linalg.norm(random_points, axis=1, keepdims=True)
            self.best_solution = random_points / np.maximum(norms, 1e-12)
            self.best_ratio = self.calculate_min_max_ratio(self.best_solution)
        
        # Prepare statistics
        stats = {
            'min_max_ratio': self.best_ratio,
            'benchmark_ratio': self.best_ratio / 0.4898,
            'max_distance': 0.0,  # Will be calculated from actual points
            'eval_time': 0.0
        }
        
        # Calculate max distance for stats
        distances = pdist(self.best_solution)
        if len(distances) > 0:
            stats['max_distance'] = np.max(distances)
        
        return self.best_solution, stats

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create optimizer instance
    optimizer = PointArrangementOptimizer(num_points=14, dimension=3)
    
    # Perform optimization
    points, _ = optimizer.optimize()
    
    return points

# EVOLVE-BLOCK-END