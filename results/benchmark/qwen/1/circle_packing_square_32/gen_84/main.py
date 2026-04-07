# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
import random
from typing import Tuple, List, Optional
import math

class SpatialGrid:
    """Efficient spatial grid for neighbor lookups"""
    def __init__(self, cell_size: float = None):
        self.cell_size = cell_size
        self.grid = {}
        
    def build(self, circles: np.ndarray) -> None:
        """Build spatial grid from circle array"""
        self.grid.clear()
        if self.cell_size is None:
            avg_radius = np.mean(circles[:, 2]) if len(circles) > 0 else 0.1
            self.cell_size = max(0.01, 2 * avg_radius)
            
        for i, (x, y, r) in enumerate(circles):
            grid_x = int(x / self.cell_size)
            grid_y = int(y / self.cell_size)
            
            if (grid_x, grid_y) not in self.grid:
                self.grid[(grid_x, grid_y)] = []
            self.grid[(grid_x, grid_y)].append(i)
    
    def get_neighbors(self, x: float, y: float) -> List[int]:
        """Get all circle indices that could potentially collide with a point"""
        grid_x = int(x / self.cell_size)
        grid_y = int(y / self.cell_size)
        neighbors = []
        
        # Check the cell itself and its 8 neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell_key = (grid_x + dx, grid_y + dy)
                if cell_key in self.grid:
                    neighbors.extend(self.grid[cell_key])
                    
        return neighbors

class CircleValidator:
    """Validates circle packing constraints efficiently"""
    def __init__(self):
        self.spatial_grid = SpatialGrid()
        
    def validate(self, circles: np.ndarray) -> bool:
        """Validate all constraints using spatial optimization"""
        n = len(circles)
        if n == 0:
            return True
            
        # Vectorized containment check
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        radii = circles[:, 2]
        
        containment_ok = (
            (radii <= x_coords) & 
            (x_coords <= 1 - radii) & 
            (radii <= y_coords) & 
            (y_coords <= 1 - radii)
        )
        
        if not np.all(containment_ok):
            return False
            
        # Use spatial grid for efficient overlap checking
        self.spatial_grid.build(circles)
        
        for i in range(n):
            x1, y1, r1 = circles[i]
            neighbor_indices = self.spatial_grid.get_neighbors(x1, y1)
            
            for j in neighbor_indices:
                if i != j:
                    x2, y2, r2 = circles[j]
                    dx = x1 - x2
                    dy = y1 - y2
                    dist_sq = dx*dx + dy*dy
                    min_dist_sq = (r1 + r2) * (r1 + r2)
                    
                    if dist_sq < min_dist_sq:
                        return False
                        
        return True

class CircleOptimizer:
    """Local optimization of circle positions"""
    def __init__(self):
        pass
        
    def optimize(self, circles: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Fine-tune circle positions using local optimization"""
        if len(circles) == 0:
            return circles
            
        # Convert to flattened parameter array for optimization
        params = circles.flatten()
        
        def objective(params_flat: np.ndarray) -> float:
            circles_new = params_flat.reshape(-1, 3)
            return -np.sum(circles_new[:, 2])
        
        try:
            # Define bounds for optimization
            bounds = []
            for i in range(len(params)//3):
                bounds.extend([(0.001, 1-0.001), (0.001, 1-0.001), (0.001, 0.5)])
            
            result = minimize(objective, params, method='L-BFGS-B', 
                            bounds=bounds,
                            options={'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-6})
            
            if result.success:
                return result.x.reshape(-1, 3)
        except Exception:
            pass
            
        return circles

class CircleMutator:
    """Mutation operations for evolutionary algorithm"""
    def __init__(self):
        pass
        
    def mutate(self, circles: np.ndarray, mutation_rate: float = 0.2) -> np.ndarray:
        """Create a mutated version of the circle configuration"""
        new_circles = circles.copy()
        n_mutations = max(1, int(len(circles) * mutation_rate))
        indices = np.random.choice(len(circles), size=n_mutations, replace=False)
        
        for idx in indices:
            x, y, r = new_circles[idx]
            
            # Mutate position slightly
            x += np.random.normal(0, 0.03)
            y += np.random.normal(0, 0.03)
            
            # Bound position to unit square
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            
            # Mutate radius
            r *= (1 + np.random.normal(0, 0.1))
            r = max(0.001, r)
            r = min(r, x, 1-x, y, 1-y)
            
            new_circles[idx] = [x, y, r]
            
        return new_circles

class VoronoiInitializer:
    """Advanced Voronoi-based circle initialization"""
    def __init__(self):
        pass
        
    def initialize(self, n_circles: int) -> np.ndarray:
        """Initialize circles using enhanced Voronoi diagram approach"""
        # Generate random points
        sample_points = np.random.rand(n_circles*10, 2)
        vor = Voronoi(sample_points)
        
        # Select valid Voronoi vertices inside unit square
        valid_vertices = []
        for vertex in vor.vertices:
            if 0 <= vertex[0] <= 1 and 0 <= vertex[1] <= 1:
                valid_vertices.append(vertex)
        
        # Add random points if needed
        while len(valid_vertices) < n_circles:
            valid_vertices.append([np.random.rand(), np.random.rand()])
            
        # Take first n_circles vertices
        selected_vertices = np.array(valid_vertices[:n_circles])
        
        # Create circles with optimized radius calculation
        circles = []
        for i, (x, y) in enumerate(selected_vertices):
            # Calculate minimum distance to neighbors (vectorized)
            if len(selected_vertices) > 1:
                distances = np.sqrt(np.sum((selected_vertices - [x, y])**2, axis=1))
                distances[distances == 0] = np.inf  # Ignore self-distance
                min_dist = np.min(distances)
            else:
                min_dist = 1.0
                
            # Set radius to half the minimum distance or bounded by unit square
            r = min(min_dist/2, x, 1-x, y, 1-y)
            r = max(r, 0.001)
            circles.append([x, y, r])
            
        return np.array(circles)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    # Initialize components
    validator = CircleValidator()
    optimizer = CircleOptimizer()
    mutator = CircleMutator()
    initializer = VoronoiInitializer()
    
    best_circles = None
    best_sum_radii = 0
    
    # Multi-strategy optimization approach
    strategies = [
        ("voronoi_heuristic", 10),
        ("corner_placement", 5),
        ("hexagonal_grid", 5)
    ]
    
    for strategy_name, attempts in strategies:
        for attempt in range(attempts):
            # Initialize based on strategy
            if strategy_name == "voronoi_heuristic":
                circles = initializer.initialize(32)
            elif strategy_name == "corner_placement":
                # Place 4 circles in corners
                circles = np.zeros((32, 3))
                corner_positions = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
                for i in range(4):
                    x, y = corner_positions[i]
                    r = min(x, y, 1-x, 1-y)
                    circles[i] = [x, y, r]
                
                # Fill remaining with Voronoi
                remaining = 28
                remaining_circles = initializer.initialize(remaining)
                circles[4:] = remaining_circles
            else:  # hexagonal_grid
                # Simple hexagonal grid initialization
                circles = np.zeros((32, 3))
                placed = 0
                row = 0
                while placed < 32:
                    col = 0
                    while placed < 32 and col < 8:
                        x = 0.1 + col * 0.12
                        y = 0.1 + row * 0.12
                        r = max(0.02, 0.05 - (placed % 5) * 0.005)
                        circles[placed] = [x, y, r]
                        placed += 1
                        col += 1
                    row += 1
                    if row >= 8: break
            
            # Local optimization
            circles = optimizer.optimize(circles, max_iter=50)
            
            # Validate and track best
            if validator.validate(circles):
                sum_radii = np.sum(circles[:, 2])
                if sum_radii > best_sum_radii:
                    best_sum_radii = sum_radii
                    best_circles = circles.copy()
            
            # Evolutionary improvement with enhanced mutation
            for gen in range(100):
                mutated = mutator.mutate(circles, 0.25)
                mutated = optimizer.optimize(mutated, max_iter=30)
                
                if validator.validate(mutated):
                    sum_radii = np.sum(mutated[:, 2])
                    if sum_radii > best_sum_radii:
                        best_sum_radii = sum_radii
                        best_circles = mutated.copy()
                        circles = mutated.copy()
    
    # Fallback to simple initialization if no good solution found
    if best_circles is None:
        circles = np.zeros((32, 3))
        placed = 0
        for i in range(6):
            for j in range(6):
                if placed >= 32:
                    break
                x = 0.1 + i * 0.15
                y = 0.1 + j * 0.15
                r = 0.05
                circles[placed] = [x, y, r]
                placed += 1
            if placed >= 32:
                break
        best_circles = circles
    
    return best_circles

# EVOLVE-BLOCK-END
