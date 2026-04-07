# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
import math
from typing import Tuple, List

# Global constants
GRID_RESOLUTIONS = [20, 10, 5]  # Coarse to fine grid resolutions
MAX_LOCAL_ITERATIONS = 20
LOCAL_SEARCH_RADIUS = 0.1
VALIDITY_THRESHOLD = 1e-6

class GriddedCirclePacker:
    def __init__(self, n_circles: int = 26):
        self.n_circles = n_circles
        self.grid_size = 1.0
        self.current_best = None
        self.best_sum_radii = 0.0
        
    def get_grid_cells(self, circles: np.ndarray, grid_size: int) -> dict:
        """Create a spatial grid for fast neighbor lookups."""
        grid = {}
        cell_size = self.grid_size / grid_size
        
        for i, (x, y, r) in enumerate(circles):
            # Determine which grid cells this circle might occupy
            min_x_cell = max(0, int((x - r) / cell_size))
            max_x_cell = min(grid_size - 1, int((x + r) / cell_size))
            min_y_cell = max(0, int((y - r) / cell_size))
            max_y_cell = min(grid_size - 1, int((y + r) / cell_size))
            
            for gx in range(min_x_cell, max_x_cell + 1):
                for gy in range(min_y_cell, max_y_cell + 1):
                    if (gx, gy) not in grid:
                        grid[(gx, gy)] = []
                    grid[(gx, gy)].append(i)
        
        return grid

    def check_overlap(self, circles: np.ndarray, grid: dict = None) -> bool:
        """Check if any circles overlap using spatial grid indexing."""
        if len(circles) <= 1:
            return False
        
        if grid is None:
            grid = self.get_grid_cells(circles)
        
        # For each cell, check pairs of circles
        for (gx, gy), indices in grid.items():
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    idx1, idx2 = indices[i], indices[j]
                    x1, y1, r1 = circles[idx1]
                    x2, y2, r2 = circles[idx2]
                    
                    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < (r1 + r2 - VALIDITY_THRESHOLD):
                        return True
        
        return False

    def check_containment(self, circles: np.ndarray) -> bool:
        """Check if all circles are fully contained in the unit square."""
        for x, y, r in circles:
            if x - r < 0 or x + r > self.grid_size or y - r < 0 or y + r > self.grid_size:
                return False
        return True

    def compute_total_radius(self, circles: np.ndarray) -> float:
        """Calculate sum of all radii."""
        return np.sum(circles[:, 2])

    def enforce_bounds(self, circles: np.ndarray) -> np.ndarray:
        """Ensure all circles are within bounds."""
        result = circles.copy()
        for i in range(len(result)):
            x, y, r = result[i]
            # Ensure circle fits in the unit square
            r = min(r, x, self.grid_size - x, y, self.grid_size - y)
            r = max(0.001, min(0.49, r))
            # Clamp coordinates
            x = max(r, min(self.grid_size - r, x))
            y = max(r, min(self.grid_size - r, y))
            result[i] = [x, y, r]
        return result

    def generate_initial_grid_placement(self, grid_resolution: int) -> np.ndarray:
        """Generate initial circle positions on a grid."""
        circles = np.zeros((self.n_circles, 3))
        
        # Create grid points
        spacing = self.grid_size / (grid_resolution + 1)
        points = []
        for i in range(1, grid_resolution + 1):
            for j in range(1, grid_resolution + 1):
                points.append((i * spacing, j * spacing))
        
        # Distribute circles
        for i in range(self.n_circles):
            if i < len(points):
                x, y = points[i]
                # Add small jitter
                x += random.uniform(-spacing/4, spacing/4)
                y += random.uniform(-spacing/4, spacing/4)
                # Initial radius
                circles[i] = [x, y, 0.03]
            else:
                # Place remaining circles randomly
                x = random.uniform(0.05, self.grid_size - 0.05)
                y = random.uniform(0.05, self.grid_size - 0.05)
                circles[i] = [x, y, 0.02]
        
        return circles

    def optimize_local(self, circles: np.ndarray, max_iterations: int = MAX_LOCAL_ITERATIONS) -> np.ndarray:
        """Perform local optimization to improve circle placement."""
        current = circles.copy()
        
        for iteration in range(max_iterations):
            improved = False
            grid = self.get_grid_cells(current)
            
            # For each circle, try to improve its position
            for i in range(len(current)):
                best_pos = current[i].copy()
                best_radius = current[i][2]
                best_value = self.compute_total_radius(current)
                
                # Try small moves in various directions
                x_orig, y_orig, r_orig = current[i]
                moves = [
                    (0, 0, 0),  # No change
                    (-0.005, -0.005, 0),
                    (-0.005, 0.005, 0),
                    (0.005, -0.005, 0),
                    (0.005, 0.005, 0),
                    (0, 0.005, 0),
                    (0, -0.005, 0),
                    (0.005, 0, 0),
                    (-0.005, 0, 0),
                    (0, 0, 0.001),
                    (0, 0, -0.001)
                ]
                
                for dx, dy, dr in moves:
                    test_x = x_orig + dx
                    test_y = y_orig + dy
                    test_r = r_orig + dr
                    
                    # Check bounds
                    if test_x - test_r < 0 or test_x + test_r > self.grid_size:
                        continue
                    if test_y - test_r < 0 or test_y + test_r > self.grid_size:
                        continue
                    if test_r <= 0 or test_r >= 0.5:
                        continue
                        
                    # Test this configuration
                    temp_circles = current.copy()
                    temp_circles[i] = [test_x, test_y, test_r]
                    
                    # Check overlap with others
                    temp_grid = self.get_grid_cells(temp_circles)
                    if not self.check_overlap(temp_circles, temp_grid):
                        # Check if this improves total radius
                        temp_total = self.compute_total_radius(temp_circles)
                        if temp_total > best_value:
                            best_value = temp_total
                            best_pos = [test_x, test_y, test_r]
                            improved = True
                
                # Update if improvement was found
                if improved:
                    current[i] = best_pos
                    grid = self.get_grid_cells(current)  # Refresh grid
                    
            # Early termination if no improvement
            if not improved:
                break
                
        return self.enforce_bounds(current)

    def find_optimal_placement(self) -> np.ndarray:
        """Main optimization routine."""
        best_solution = None
        best_sum = 0.0
        
        # Try multiple grid resolutions from coarse to fine
        for grid_res in GRID_RESOLUTIONS:
            # Generate initial placement
            initial = self.generate_initial_grid_placement(grid_res)
            
            # Refine with local optimization
            refined = self.optimize_local(initial, MAX_LOCAL_ITERATIONS)
            
            # Check if this is valid and better
            if self.check_containment(refined) and not self.check_overlap(refined):
                total_radius = self.compute_total_radius(refined)
                if total_radius > best_sum:
                    best_sum = total_radius
                    best_solution = refined.copy()
            
            # Continue with finer grid if we have room
            if grid_res > 5:
                # Try to further optimize current best
                if best_solution is not None:
                    refined = self.optimize_local(best_solution, MAX_LOCAL_ITERATIONS)
                    if self.check_containment(refined) and not self.check_overlap(refined):
                        total_radius = self.compute_total_radius(refined)
                        if total_radius > best_sum:
                            best_sum = total_radius
                            best_solution = refined.copy()

        # Final thorough local optimization on best found
        if best_solution is not None:
            final_solution = self.optimize_local(best_solution, MAX_LOCAL_ITERATIONS * 2)
            if self.check_containment(final_solution) and not self.check_overlap(final_solution):
                return final_solution
                
        return best_solution if best_solution is not None else self.generate_initial_grid_placement(10)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    packer = GriddedCirclePacker(26)
    circles = packer.find_optimal_placement()
    
    return circles

# EVOLVE-BLOCK-END
