# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import Voronoi
import time
import math
from typing import Tuple, List, Callable, Optional
import copy

class PointDistributionOptimizer:
    """Modular optimizer for point distribution maximizing min/max distance ratio."""
    
    def __init__(self, n_points: int = 16, dimensions: int = 2, max_time: float = 180.0):
        self.n_points = n_points
        self.dimensions = dimensions
        self.benchmark_ratio = 1 / np.sqrt(12.889266112)  # 0.2786
        self.max_time = max_time
        self._setup_optimization_parameters()
    
    def _setup_optimization_parameters(self):
        """Setup optimization parameters for different stages."""
        self.max_global_iter = 200
        self.max_local_iter_stage1 = 200
        self.max_local_iter_stage2 = 300
        self.max_local_iter_stage3 = 400
        self.initialization_attempts = 10
    
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
    
    def _fibonacci_sphere_points(self) -> np.ndarray:
        """Generate points on a sphere using Fibonacci algorithm."""
        points = []
        phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
        
        for i in range(self.n_points):
            y = 1 - (i / float(self.n_points - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i
            
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def _stereographic_project(self, points_3d: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D using stereographic projection from south pole."""
        points_2d = []
        for x, y, z in points_3d:
            # Stereographic projection from south pole (0,0,-1)
            w = 1 / (1 + z)
            proj_x = x * w
            proj_y = y * w
            points_2d.append([proj_x, proj_y])
        
        points_2d = np.array(points_2d)
        
        # Normalize to unit square
        x_min, y_min = np.min(points_2d, axis=0)
        x_max, y_max = np.max(points_2d, axis=0)
        
        if x_max > x_min and y_max > y_min:
            points_2d[:, 0] = (points_2d[:, 0] - x_min) / (x_max - x_min) * 0.9 + 0.05
            points_2d[:, 1] = (points_2d[:, 1] - y_min) / (y_max - y_min) * 0.9 + 0.05
        
        return points_2d
    
    def _initialize_spherical_projection(self) -> np.ndarray:
        """Initialize points using spherical arrangement projected to 2D."""
        points_3d = self._fibonacci_sphere_points()
        return self._stereographic_project(points_3d)
    
    def _initialize_voronoi_distribution(self) -> np.ndarray:
        """Initialize points using Voronoi-based distribution."""
        # Start with random points
        points = np.random.rand(self.n_points, self.dimensions)
        
        # Iteratively improve using a Voronoi-based approach
        for _ in range(10):  # Reduced iterations for speed
            try:
                vor = Voronoi(points)
                new_points = []
                
                for i in range(len(points)):
                    # For each point, find its Voronoi cell centroid
                    region = vor.regions[vor.point_region[i]]
                    if not region or -1 in region:
                        # Skip infinite regions
                        new_points.append(points[i])
                        continue
                    
                    # Compute centroid of the Voronoi cell
                    vertices = [vor.vertices[j] for j in region if j >= 0]
                    if len(vertices) > 0:
                        vertices = np.array(vertices)
                        centroid = np.mean(vertices, axis=0)
                        new_points.append(centroid)
                    else:
                        new_points.append(points[i])
                
                points = np.array(new_points)
                # Keep within bounds
                points = np.clip(points, 0, 1)
            except Exception:
                break  # If Voronoi computation fails, stop refinement
        
        return points
    
    def _initialize_hexagonal_grid(self) -> np.ndarray:
        """Initialize points using hexagonal grid pattern."""
        # Create a grid that approximates hexagonal packing
        rows = int(np.ceil(np.sqrt(self.n_points)))
        cols = int(np.ceil(self.n_points / rows))
        
        points = []
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= self.n_points:
                    break
                    
                # Offset odd rows for hexagonal arrangement
                offset = 0.5 * (i % 2)
                x = (j + offset) * spacing_x
                y = i * spacing_y
                
                # Ensure we don't exceed bounds
                x = min(x, 0.99)
                y = min(y, 0.99)
                
                points.append([x, y])
        
        # Trim to exact number of points
        points = np.array(points[:self.n_points])
        
        # Normalize to fit properly in [0,1] box
        if len(points) > 0:
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            
            if x_max > x_min and y_max > y_min:
                points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min) * 0.9 + 0.05
                points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min) * 0.9 + 0.05
        
        return points
    
    def _initialize_spiral_pattern(self) -> np.ndarray:
        """Initialize points using spiral pattern."""
        points = []
        angle_step = 2 * np.pi / 10
        radius_step = 1.0 / 10
        
        for i in range(self.n_points):
            if i == 0:
                points.append([0.5, 0.5])  # Center point
            else:
                angle = i * angle_step
                radius = min(0.45, i * radius_step)
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                points.append([x, y])
        
        # Fill remaining points with random if needed
        while len(points) < self.n_points:
            points.append([np.random.rand(), np.random.rand()])
            
        return np.array(points[:self.n_points])
    
    def _initialize_grid_with_perturbation(self) -> np.ndarray:
        """Initialize points using grid pattern with random perturbation."""
        grid_points = []
        grid_size = int(np.ceil(np.sqrt(self.n_points)))
        for i in range(grid_size):
            for j in range(grid_size):
                if len(grid_points) >= self.n_points:
                    break
                x = i / (grid_size - 1) if grid_size > 1 else 0.5
                y = j / (grid_size - 1) if grid_size > 1 else 0.5
                # Add slight randomness to avoid perfect grid
                x += (np.random.rand() - 0.5) * 0.1
                y += (np.random.rand() - 0.5) * 0.1
                grid_points.append([x, y])
        return np.clip(np.array(grid_points[:self.n_points]), 0, 1)
    
    def _initialize_random_points(self) -> np.ndarray:
        """Initialize points using random uniform distribution."""
        return np.random.rand(self.n_points, self.dimensions)
    
    def _generate_initial_strategies(self) -> List[np.ndarray]:
        """Generate diverse initial configurations using different methods."""
        strategies = []
        np.random.seed(42)
        
        # Generate multiple initialization strategies
        init_methods = [
            self._initialize_spherical_projection,
            self._initialize_voronoi_distribution,
            self._initialize_random_points,
            self._initialize_hexagonal_grid,
            self._initialize_spiral_pattern,
            self._initialize_grid_with_perturbation
        ]
        
        # Apply each method with noise
        for i, method in enumerate(init_methods):
            try:
                initial_points = method()
                # Add controlled noise for diversity
                noise_level = 0.05 / (i + 1)
                noisy_points = initial_points + np.random.normal(0, noise_level, initial_points.shape)
                strategies.append(np.clip(noisy_points, 0, 1))
            except Exception:
                # Fallback to random if method fails
                strategies.append(np.random.rand(self.n_points, self.dimensions))
        
        return strategies
    
    def _select_best_initial(self, strategies: List[np.ndarray]) -> np.ndarray:
        """Select the best initial configuration based on min-max ratio."""
        best_ratio = -float('inf')
        best_points = None
        
        for points in strategies:
            try:
                ratio = self.calculate_min_max_ratio(points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = points.copy()
            except Exception:
                continue
                
        return best_points if best_points is not None else np.random.rand(self.n_points, self.dimensions)
    
    def _global_search(self, initial_points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Use global optimization to explore promising regions."""
        def objective(x_flat):
            points = x_flat.reshape(-1, self.dimensions)
            return -self.calculate_min_max_ratio(points)
        
        # Use differential evolution for broad exploration
        bounds = [(0, 1) for _ in range(len(initial_points.flatten()))]
        
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=self.max_global_iter,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, self.dimensions)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = self.calculate_min_max_ratio(optimized_points)
                return optimized_points, ratio
        except Exception:
            pass
            
        return initial_points, self.calculate_min_max_ratio(initial_points)
    
    def _local_refinement_stage1(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Stage 1: Coarse local optimization with L-BFGS-B."""
        def objective(x_flat):
            points_candidate = x_flat.reshape(-1, self.dimensions)
            return -self.calculate_min_max_ratio(points_candidate)
        
        try:
            result = minimize(
                objective, 
                points.flatten(), 
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(len(points.flatten()))],
                options={'maxiter': self.max_local_iter_stage1},
                tol=1e-6
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, self.dimensions)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = self.calculate_min_max_ratio(optimized_points)
                return optimized_points, ratio
        except Exception:
            pass
            
        return points, self.calculate_min_max_ratio(points)
    
    def _local_refinement_stage2(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Stage 2: Medium local optimization with hybrid approach."""
        def objective(x_flat):
            points_candidate = x_flat.reshape(-1, self.dimensions)
            return -self.calculate_min_max_ratio(points_candidate)
        
        best_points = points.copy()
        best_ratio = self.calculate_min_max_ratio(best_points)
        
        # Method 1: L-BFGS-B
        try:
            result = minimize(
                objective, 
                points.flatten(), 
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(len(points.flatten()))],
                options={'maxiter': self.max_local_iter_stage2 // 2},
                tol=1e-6
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, self.dimensions)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = self.calculate_min_max_ratio(optimized_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except Exception:
            pass
        
        # Method 2: Nelder-Mead  
        try:
            result = minimize(
                objective, 
                points.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': self.max_local_iter_stage2 // 2, 'adaptive': True}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, self.dimensions)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = self.calculate_min_max_ratio(optimized_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except Exception:
            pass
        
        return best_points, best_ratio
    
    def _local_refinement_stage3(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Stage 3: Fine local optimization."""
        def objective(x_flat):
            points_candidate = x_flat.reshape(-1, self.dimensions)
            return -self.calculate_min_max_ratio(points_candidate)
        
        try:
            result = minimize(
                objective, 
                points.flatten(), 
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(len(points.flatten()))],
                options={'maxiter': self.max_local_iter_stage3},
                tol=1e-6
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, self.dimensions)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = self.calculate_min_max_ratio(optimized_points)
                return optimized_points, ratio
        except Exception:
            pass
            
        return points, self.calculate_min_max_ratio(points)
    
    def _pipeline_run(self, start_time: float) -> np.ndarray:
        """Execute the complete optimization pipeline."""
        # Phase 1: Initialization with diverse strategies
        strategies = self._generate_initial_strategies()
        initial_points = self._select_best_initial(strategies)
        
        # Phase 2: Global search to find promising regions
        global_points, global_ratio = self._global_search(initial_points)
        
        # Phase 3: Progressive local refinement
        # Stage 1: Coarse refinement
        coarse_points, coarse_ratio = self._local_refinement_stage1(global_points)
        current_points = coarse_points if coarse_ratio > global_ratio else global_points
        current_ratio = max(coarse_ratio, global_ratio)
        
        # Stage 2: Medium refinement
        medium_points, medium_ratio = self._local_refinement_stage2(current_points)
        current_points = medium_points if medium_ratio > current_ratio else current_points
        current_ratio = max(medium_ratio, current_ratio)
        
        # Stage 3: Fine refinement
        fine_points, fine_ratio = self._local_refinement_stage3(current_points)
        current_points = fine_points if fine_ratio > current_ratio else current_points
        current_ratio = max(fine_ratio, current_ratio)
        
        return current_points
    
    def evolve(self) -> np.ndarray:
        """Main evolutionary optimization loop with time management."""
        start_time = time.time()
        best_points = None
        best_ratio = -float('inf')
        
        # Multiple attempts to improve results
        max_attempts = 3
        for attempt in range(max_attempts):
            if time.time() - start_time > self.max_time - 10:
                break
                
            try:
                current_points = self._pipeline_run(start_time)
                current_ratio = self.calculate_min_max_ratio(current_points)
                
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = current_points.copy()
                    
            except Exception:
                continue
        
        # Final fallback to random points if nothing works
        if best_points is None:
            best_points = np.random.rand(self.n_points, self.dimensions)
            
        return best_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = PointDistributionOptimizer(n_points=16, dimensions=2)
    return optimizer.evolve()

# EVOLVE-BLOCK-END