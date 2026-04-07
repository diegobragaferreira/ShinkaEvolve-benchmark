# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.spatial import SphericalVoronoi
import time

class SphericalVoronoiOptimizer:
    """Multi-phase optimizer using spherical Voronoi diagrams for point arrangement."""
    
    def __init__(self, num_points: int = 14, dimension: int = 3):
        self.num_points = num_points
        self.dimension = dimension
        self.best_score = -np.inf
        self.best_points = None
        
    def fibonacci_spiral_on_sphere(self, n: int) -> np.ndarray:
        """Generate points on sphere using Fibonacci spiral method."""
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            theta = golden_angle * i  # golden angle increment
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)
    
    def generate_voronoi_initialization(self, n: int, iterations: int = 50) -> np.ndarray:
        """
        Generate initial points using spherical Voronoi iteration method.
        This creates a more uniform distribution than basic Fibonacci spiral.
        """
        # Start with Fibonacci spiral
        points = self.fibonacci_spiral_on_sphere(n)
        
        # Iteratively improve using Voronoi-based relaxation
        for _ in range(iterations):
            # Normalize to unit sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            safe_norms = np.where(norms == 0, 1, norms)
            points = points / safe_norms
            
            try:
                # Create spherical Voronoi diagram
                sv = SphericalVoronoi(points, radius=1.0, center=np.zeros(3))
                
                # Get Voronoi cell centers (which represent optimal positions)
                cell_centers = sv.vertices
                
                # Project cell centers back to sphere
                cell_norms = np.linalg.norm(cell_centers, axis=1, keepdims=True)
                safe_cell_norms = np.where(cell_norms == 0, 1, cell_norms)
                points = cell_centers / safe_cell_norms
                
                # Add some randomness to avoid local minima
                noise_magnitude = 0.01
                noise = np.random.normal(0, noise_magnitude, points.shape)
                points = points + noise
                
            except:
                # If Voronoi fails, fall back to Fibonacci
                points = self.fibonacci_spiral_on_sphere(n)
                
        return points
    
    def calculate_distances(self, points: np.ndarray) -> tuple:
        """Calculate distance matrix and extract min/max distances."""
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        return distances, min_dist, max_dist
    
    def voronoi_quality(self, points: np.ndarray) -> float:
        """Calculate quality metric based on Voronoi cell area variance."""
        try:
            # Normalize points
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            safe_norms = np.where(norms == 0, 1, norms)
            normalized_points = points / safe_norms
            
            # Create spherical Voronoi diagram
            sv = SphericalVoronoi(normalized_points, radius=1.0, center=np.zeros(3))
            
            # Calculate Voronoi cell areas
            cell_areas = sv.voronoi_regions_area()
            
            # Return variance of cell areas (lower variance = more uniform distribution)
            return np.var(cell_areas)
        except:
            # Fallback if Voronoi computation fails
            return np.inf
    
    def objective_function(self, points_flat: np.ndarray, include_voronoi: bool = True) -> float:
        """Objective function to maximize min/max distance ratio with optional Voronoi penalty."""
        points = points_flat.reshape(-1, 3)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0, 1, norms)
        normalized_points = points / safe_norms
        
        _, min_dist, max_dist = self.calculate_distances(normalized_points)
        
        if max_dist == 0:
            return -np.inf
            
        ratio = min_dist / max_dist
        
        # Add Voronoi-based regularization if requested
        if include_voronoi:
            voronoi_penalty = self.voronoi_quality(normalized_points)
            # Scale penalty appropriately
            penalty_weight = 0.01
            ratio = ratio - penalty_weight * voronoi_penalty
            
        return ratio
    
    def constraint_sphere(self, points_flat: np.ndarray) -> float:
        """Constraint function to keep points on unit sphere."""
        points = points_flat.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return np.mean((norms - 1.0)**2)  # Mean squared deviation from unit sphere
    
    def optimize_coarse(self, initial_points: np.ndarray) -> np.ndarray:
        """Coarse global optimization using differential evolution."""
        from scipy.optimize import differential_evolution
        
        def obj_func(x):
            return -self.objective_function(x, include_voronoi=False)
        
        bounds = [(-1, 1) for _ in range(self.num_points * self.dimension)]
        
        # Coarse optimization parameters
        result = differential_evolution(
            obj_func,
            bounds,
            maxiter=30,
            popsize=10,
            seed=42,
            disp=False,
            tol=1e-4
        )
        
        return result.x.reshape(-1, 3)
    
    def optimize_fine(self, initial_points: np.ndarray) -> np.ndarray:
        """Fine-grained local optimization using L-BFGS-B."""
        initial_flat = initial_points.flatten()
        
        # Define bounds for coordinates (-1, 1) for each coordinate
        bounds = [(-1, 1) for _ in range(self.num_points * self.dimension)]
        
        # Optimization parameters with adaptive tolerance
        options = {'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}
        
        # Run optimization
        result = minimize(
            lambda x: -self.objective_function(x),
            initial_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options=options,
            tol=1e-12
        )
        
        # Extract optimized points
        final_points = result.x.reshape(-1, 3)
        
        # Ensure points are normalized to unit sphere
        norms = np.linalg.norm(final_points, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0, 1, norms)
        final_points = final_points / safe_norms
        
        return final_points
    
    def validate_and_score(self, points: np.ndarray) -> tuple:
        """Validate solution and compute performance metrics."""
        distances, min_dist, max_dist = self.calculate_distances(points)
        
        if max_dist == 0:
            ratio = 0.0
        else:
            ratio = min_dist / max_dist
            
        benchmark_ratio = ratio / 0.4898
            
        return ratio, benchmark_ratio, max_dist
    
    def optimize(self) -> tuple:
        """Main multi-phase optimization loop."""
        # Phase 1: Better initialization using Voronoi relaxation
        start_time = time.time()
        initial_points = self.generate_voronoi_initialization(self.num_points, iterations=30)
        
        # Phase 2: Coarse global optimization
        coarse_points = self.optimize_coarse(initial_points)
        
        # Phase 3: Fine local optimization
        optimized_points = self.optimize_fine(coarse_points)
        
        # Phase 4: Validation and Scoring
        min_max_ratio, benchmark_ratio, max_dist = self.validate_and_score(optimized_points)
        
        eval_time = time.time() - start_time
        
        stats = {
            'min_max_ratio': min_max_ratio,
            'benchmark_ratio': benchmark_ratio,
            'max_distance': max_dist,
            'eval_time': eval_time
        }
        
        return optimized_points, stats

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create optimizer instance
    optimizer = SphericalVoronoiOptimizer(num_points=14, dimension=3)
    
    # Perform optimization
    points, _ = optimizer.optimize()
    
    return points

# EVOLVE-BLOCK-END