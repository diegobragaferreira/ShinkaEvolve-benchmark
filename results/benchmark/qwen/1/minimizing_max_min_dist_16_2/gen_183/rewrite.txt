# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import Voronoi
import time
from typing import Tuple, List
import math

class PointDispersionOptimizer:
    """Enhanced optimizer for maximizing min/max distance ratio in 2D point placement."""
    
    def __init__(self, n_points: int = 16, max_time: float = 180.0):
        self.n_points = n_points
        self.max_time = max_time
        self.start_time = time.time()
        self.benchmark_ratio = 1 / np.sqrt(12.889266112)  # ~0.2786
        
    def _calculate_min_max_ratio(self, points: np.ndarray) -> float:
        """Calculate the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0.0
            
        try:
            distances = squareform(pdist(points))
            
            # Set diagonal to infinity to exclude self-distances
            np.fill_diagonal(distances, np.inf)
            
            # Get min and max distances
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            # Handle edge cases
            if max_dist == 0 or np.isinf(min_dist) or np.isnan(min_dist) or np.isnan(max_dist):
                return 0.0
                
            return min_dist / max_dist
        except Exception:
            return 0.0
    
    def _initialize_spherical_projection(self) -> np.ndarray:
        """Initialize points using a spherical arrangement projected to 2D."""
        points_sphere = []
        phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
        
        for i in range(self.n_points):
            y = 1 - (i / float(self.n_points - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i
            
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            
            points_sphere.append([x, y, z])
        
        # Project 3D sphere points to 2D using stereographic projection
        points_2d = []
        for x, y, z in points_sphere:
            # Stereographic projection from south pole
            w = 1 / (1 + z)
            proj_x = x * w
            proj_y = y * w
            points_2d.append([proj_x, proj_y])
        
        # Normalize to unit square
        points_2d = np.array(points_2d)
        
        # Scale and center the points
        x_min, y_min = np.min(points_2d, axis=0)
        x_max, y_max = np.max(points_2d, axis=0)
        
        if x_max > x_min and y_max > y_min:
            points_2d[:, 0] = (points_2d[:, 0] - x_min) / (x_max - x_min) * 0.9 + 0.05
            points_2d[:, 1] = (points_2d[:, 1] - y_min) / (y_max - y_min) * 0.9 + 0.05
        
        return points_2d
    
    def _initialize_voronoi_distribution(self) -> np.ndarray:
        """Initialize points using Voronoi-based distribution."""
        # Start with random points
        points = np.random.rand(self.n_points, 2)
        
        # Iteratively improve using a Voronoi-based approach
        for _ in range(20):
            if time.time() - self.start_time > self.max_time - 10:
                break
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
                break
        
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
    
    def _initialize_population(self) -> List[np.ndarray]:
        """Create diverse initial population using multiple strategies."""
        np.random.seed(42)
        
        # Strategy 1: Spherical projection
        try:
            init1 = self._initialize_spherical_projection()
        except Exception:
            init1 = np.random.rand(self.n_points, 2)
        
        # Strategy 2: Voronoi-based
        try:
            init2 = self._initialize_voronoi_distribution()
        except Exception:
            init2 = np.random.rand(self.n_points, 2)
        
        # Strategy 3: Hexagonal grid
        try:
            init3 = self._initialize_hexagonal_grid()
        except Exception:
            init3 = np.random.rand(self.n_points, 2)
        
        # Strategy 4: Spiral pattern
        try:
            init4 = self._initialize_spiral_pattern()
        except Exception:
            init4 = np.random.rand(self.n_points, 2)
        
        # Strategy 5: Random distribution
        init5 = np.random.rand(self.n_points, 2)
        
        # Strategy 6: Grid pattern with perturbation
        try:
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
            init6 = np.clip(np.array(grid_points[:self.n_points]), 0, 1)
        except Exception:
            init6 = np.random.rand(self.n_points, 2)
        
        # Combine strategies
        init_points = [init1, init2, init3, init4, init5, init6]
        
        # Add noise to diversify
        diversified = []
        for i, base_init in enumerate(init_points):
            try:
                # Add some noise to diversify
                noisy = base_init + np.random.normal(0, 0.02, base_init.shape)
                diversified.append(np.clip(noisy, 0, 1))
            except Exception:
                diversified.append(base_init)
        
        return diversified
    
    def _objective_function(self, points_flat: np.ndarray) -> float:
        """Objective function to minimize (negative ratio)"""
        points = points_flat.reshape(-1, 2)
        return -self._calculate_min_max_ratio(points)
    
    def _global_optimization(self, initial_points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Use global optimization to explore promising regions."""
        if time.time() - self.start_time > self.max_time - 10:
            return initial_points, self._calculate_min_max_ratio(initial_points)
            
        bounds = [(0, 1) for _ in range(len(initial_points.flatten()))]
        
        try:
            # Use differential evolution for broad exploration
            result = differential_evolution(
                self._objective_function,
                bounds,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = self._calculate_min_max_ratio(optimized_points)
                return optimized_points, ratio
        except Exception:
            pass
            
        return initial_points, self._calculate_min_max_ratio(initial_points)
    
    def _local_refinement(self, points: np.ndarray, 
                         method: str = 'hybrid') -> Tuple[np.ndarray, float]:
        """Apply local refinement with multiple methods."""
        if time.time() - self.start_time > self.max_time - 10:
            return points, self._calculate_min_max_ratio(points)
            
        def objective(x_flat):
            points_candidate = x_flat.reshape(-1, 2)
            return -self._calculate_min_max_ratio(points_candidate)
        
        best_points = points.copy()
        best_ratio = self._calculate_min_max_ratio(best_points)
        
        # Method 1: L-BFGS-B - most reliable
        try:
            result = minimize(
                objective, 
                points.flatten(), 
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(len(points.flatten()))],
                options={'maxiter': 100},
                tol=1e-6
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = self._calculate_min_max_ratio(optimized_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except Exception:
            pass
        
        # Method 2: Nelder-Mead for additional exploration
        try:
            result = minimize(
                objective, 
                points.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 50, 'adaptive': True}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = self._calculate_min_max_ratio(optimized_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except Exception:
            pass
        
        return best_points, best_ratio
    
    def _progressive_refinement(self, initial_points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Apply progressive refinement with increasing precision."""
        if time.time() - self.start_time > self.max_time - 10:
            return initial_points, self._calculate_min_max_ratio(initial_points)
            
        current_points = initial_points.copy()
        current_ratio = self._calculate_min_max_ratio(current_points)
        
        # Stage 1: Coarse optimization
        try:
            coarse_points, coarse_ratio = self._local_refinement(current_points, 'L-BFGS-B')
            if coarse_ratio > current_ratio:
                current_points = coarse_points
                current_ratio = coarse_ratio
        except Exception:
            pass
        
        # Stage 2: Medium optimization  
        try:
            medium_points, medium_ratio = self._local_refinement(current_points, 'hybrid')
            if medium_ratio > current_ratio:
                current_points = medium_points
                current_ratio = medium_ratio
        except Exception:
            pass
            
        # Stage 3: Fine optimization
        try:
            fine_points, fine_ratio = self._local_refinement(current_points, 'hybrid')
            if fine_ratio > current_ratio:
                current_points = fine_points
                current_ratio = fine_ratio
        except Exception:
            pass
        
        return current_points, current_ratio
    
    def optimize(self) -> np.ndarray:
        """Main optimization process."""
        if time.time() - self.start_time > self.max_time - 10:
            # Fallback to simple random initialization
            return np.random.rand(self.n_points, 2)
        
        # Initialize using diverse strategies
        initial_pop = self._initialize_population()
        
        best_ratio = -np.inf
        best_points = None
        
        # Try each initial configuration with optimization
        for idx, initial_points in enumerate(initial_pop):
            if time.time() - self.start_time > self.max_time - 10:
                break
                
            try:
                # Global optimization to find promising regions
                global_points, global_ratio = self._global_optimization(initial_points)
                
                # Progressive refinement
                refined_points, refined_ratio = self._progressive_refinement(global_points)
                
                # Additional local refinement 
                final_points, final_ratio = self._local_refinement(refined_points, 'hybrid')
                
                # Select best solution
                current_best_ratio = final_ratio
                current_best_points = final_points
                
                if current_best_ratio > best_ratio:
                    best_ratio = current_best_ratio
                    best_points = current_best_points.copy()
                    
            except Exception as e:
                continue
        
        # If no good solution found, return a random configuration
        if best_points is None:
            best_points = np.random.rand(self.n_points, 2)
        
        # Final verification and possible refinement
        if time.time() - self.start_time < self.max_time - 10:
            try:
                # One final optimization pass
                final_points, _ = self._local_refinement(best_points, 'hybrid')
                best_points = final_points
            except Exception:
                pass
        
        return best_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = PointDispersionOptimizer(n_points=16)
    return optimizer.optimize()

# EVOLVE-BLOCK-END