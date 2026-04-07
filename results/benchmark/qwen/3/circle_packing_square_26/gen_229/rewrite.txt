# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Grid-based optimization parameters
GRID_SIZE = 30
SMALL_GRID_SIZE = 10
MAX_LOCAL_ITERATIONS = 200

class GridOptimizedCirclePacker:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.small_grid_size = SMALL_GRID_SIZE
        self.cell_size = 1.0 / GRID_SIZE
        self.small_cell_size = 1.0 / SMALL_GRID_SIZE
        
    def create_grid(self, circles: np.ndarray) -> dict:
        """Create spatial grid for efficient overlap checking."""
        grid = {}
        
        for i, (x, y, r) in enumerate(circles):
            # Determine grid cells touched by circle
            min_x_cell = max(0, int((x - r) * self.grid_size))
            max_x_cell = min(self.grid_size - 1, int((x + r) * self.grid_size))
            min_y_cell = max(0, int((y - r) * self.grid_size))
            max_y_cell = min(self.grid_size - 1, int((y + r) * self.grid_size))
            
            for gx in range(min_x_cell, max_x_cell + 1):
                for gy in range(min_y_cell, max_y_cell + 1):
                    if (gx, gy) not in grid:
                        grid[(gx, gy)] = []
                    grid[(gx, gy)].append(i)
                    
        return grid

    def check_overlap(self, circles: np.ndarray, grid: dict = None) -> bool:
        """Check collision between circles using spatial grid."""
        if grid is None:
            grid = self.create_grid(circles)
            
        for cell, indices in grid.items():
            for i in range(len(indices)):
                idx1 = indices[i]
                x1, y1, r1 = circles[idx1]
                
                for j in range(i + 1, len(indices)):
                    idx2 = indices[j]
                    x2, y2, r2 = circles[idx2]
                    
                    dx = x1 - x2
                    dy = y1 - y2
                    distance_sq = dx*dx + dy*dy
                    
                    if distance_sq < (r1 + r2)**2:
                        return False
                        
        return True

    def check_containment(self, circles: np.ndarray) -> bool:
        """Check if all circles are fully contained."""
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        return True

    def is_valid(self, circles: np.ndarray) -> bool:
        """Check if circles are valid."""
        return self.check_containment(circles) and self.check_overlap(circles)

    def calculate_penalty(self, circles: np.ndarray) -> float:
        """Calculate penalty for constraint violations."""
        penalty = 0.0
        
        # Boundary penalties
        for x, y, r in circles:
            if x - r < 0:
                penalty += (r - x)**2
            if x + r > 1:
                penalty += (x + r - 1)**2
            if y - r < 0:
                penalty += (r - y)**2
            if y + r > 1:
                penalty += (y + r - 1)**2
                
        # Overlap penalties
        if len(circles) > 1:
            positions = circles[:, :2]
            radii = circles[:, 2]
            
            # Use KDTree for efficient overlap checking
            tree = cKDTree(positions)
            pairs = tree.query_pairs(2.0, output_type='ndarray')
            
            for i, j in pairs:
                if i < j:
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < r1 + r2:
                        overlap = (r1 + r2 - distance)
                        penalty += overlap**2
                        
        return penalty

    def objective_function(self, circles: np.ndarray) -> float:
        """Objective function to maximize (negative sum of radii with penalty)."""
        if not self.is_valid(circles):
            penalty = self.calculate_penalty(circles)
            return -np.sum(circles[:, 2]) + penalty * 10000.0
        return -np.sum(circles[:, 2])

    def generate_initial_voronoi_points(self, n_points: int) -> np.ndarray:
        """Generate Voronoi-like initial point distribution."""
        points = []
        
        # Use hexagonal grid for better distribution
        grid_size = max(6, int(np.ceil(np.sqrt(n_points))))
        hex_radius = 1.0 / (grid_size + 2)
        
        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) >= n_points:
                    break
                # Hexagonal offset
                x_offset = (j + 0.5 * (i % 2)) * hex_radius * 2
                y_offset = i * hex_radius * np.sqrt(3)
                
                x = x_offset + np.random.uniform(-hex_radius*0.3, hex_radius*0.3)
                y = y_offset + np.random.uniform(-hex_radius*0.3, hex_radius*0.3)
                
                x = np.clip(x, hex_radius, 1 - hex_radius)
                y = np.clip(y, hex_radius, 1 - hex_radius)
                
                points.append([x, y])
                
        # Fill remaining points
        while len(points) < n_points:
            points.append([np.random.random(), np.random.random()])
            
        return np.array(points[:n_points])

    def initialize_circles(self, n_circles: int) -> np.ndarray:
        """Initialize circle configuration with better distribution."""
        circles = np.zeros((n_circles, 3))
        
        # Generate initial Voronoi-like points
        points = self.generate_initial_voronoi_points(n_circles)
        
        # Assign radii and positions
        for i in range(n_circles):
            x, y = points[i]
            circles[i] = [x, y, 0.05]  # Initial radius
            
        # Optimize radii to fit spatial constraints
        for i in range(n_circles):
            x, y, r = circles[i]
            # Find closest neighbor
            distances = np.sqrt(np.sum((points - points[i])**2, axis=1))
            distances[i] = np.inf
            min_distance = np.min(distances)
            
            # Maximum allowable radius based on neighbors and boundaries
            max_allowable_radius = min(x, y, 1 - x, 1 - y)
            if min_distance > 0:
                proposed_radius = min(min_distance / 3.0, max_allowable_radius * 0.7)
                circles[i, 2] = max(0.01, min(proposed_radius, 0.4))
            else:
                circles[i, 2] = max(0.01, max_allowable_radius * 0.5)
                
        # Ensure validity
        if self.is_valid(circles):
            return circles
            
        # Fall back to grid initialization
        circles = np.zeros((n_circles, 3))
        rows = int(np.ceil(np.sqrt(n_circles)))
        cols = rows
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        radius = min(spacing_x, spacing_y) * 0.3
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n_circles:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                circles[idx] = [x, y, radius]
                idx += 1
                
        # Final refinement
        if not self.is_valid(circles):
            for i in range(n_circles):
                circles[i] = [0.5, 0.5, 0.01]
                
        return circles

    def project_to_feasible_region(self, circles: np.ndarray) -> np.ndarray:
        """Project circles to feasible region to satisfy boundary constraints."""
        projected = circles.copy()
        
        for i in range(len(projected)):
            x, y, r = projected[i]
            
            # Project to boundary
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            
            projected[i] = [x, y, r]
            
        return projected

    def resolve_overlaps(self, circles: np.ndarray) -> np.ndarray:
        """Resolve overlaps by adjusting positions and radii."""
        resolved = circles.copy()
        
        # First, try to reduce radii to resolve overlaps
        for _ in range(20):
            changed = False
            
            # Try to resolve overlaps using geometric approach
            for i in range(len(resolved)):
                x1, y1, r1 = resolved[i]
                
                # Check overlaps with all others
                for j in range(len(resolved)):
                    if i != j:
                        x2, y2, r2 = resolved[j]
                        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        
                        if distance < (r1 + r2):
                            # Reduce radii proportionally
                            total_overlap = (r1 + r2) - distance
                            reduction = total_overlap * 0.3
                            
                            if r1 > 0.001 and r2 > 0.001:
                                new_r1 = max(0.001, r1 - reduction * 0.5)
                                new_r2 = max(0.001, r2 - reduction * 0.5)
                                
                                if new_r1 < r1 or new_r2 < r2:
                                    resolved[i, 2] = new_r1
                                    resolved[j, 2] = new_r2
                                    changed = True
            
            if not changed:
                break
                
        # Then, adjust positions to improve separation
        for _ in range(20):
            changed = False
            
            # Try to move circles apart
            for i in range(len(resolved)):
                x1, y1, r1 = resolved[i]
                
                for j in range(len(resolved)):
                    if i != j:
                        x2, y2, r2 = resolved[j]
                        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        
                        if distance < (r1 + r2):
                            # Move circles apart
                            if distance > 0.001:
                                dx = (x1 - x2) / distance
                                dy = (y1 - y2) / distance
                                
                                separation = 0.001
                                resolved[i, 0] += dx * separation
                                resolved[i, 1] += dy * separation
                                
                                # Ensure containment
                                resolved[i, 0] = np.clip(resolved[i, 0], resolved[i, 2], 1 - resolved[i, 2])
                                resolved[i, 1] = np.clip(resolved[i, 1], resolved[i, 2], 1 - resolved[i, 2])
                                
                                changed = True
                                
            if not changed:
                break
                
        return resolved

    def optimize_local(self, circles: np.ndarray) -> np.ndarray:
        """Perform local optimization on a single circle using a greedy approach.""" 
        optimized = circles.copy()
        
        # For each circle, try to improve its position and radius
        for i in range(len(optimized)):
            # Get current circle
            x, y, r = optimized[i]
            
            # Try to increase radius while maintaining validity
            max_radius = min(x, y, 1 - x, 1 - y)
            
            # Check overlap with neighbors
            neighbors = []
            for j in range(len(optimized)):
                if i != j:
                    neighbors.append(j)
                    
            if len(neighbors) > 0:
                min_dist = float('inf')
                for j in neighbors:
                    x2, y2, r2 = optimized[j]
                    dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                    min_dist = min(min_dist, dist)
                    
                # Safe radius is limited by overlap and boundary constraints
                safe_radius = min(max_radius, min_dist - 0.001) 
                if safe_radius > r:
                    # Increase radius if it doesn't cause overlap
                    optimized[i, 2] = safe_radius
                    
            # Try to improve position to increase radius
            # Try moving in small steps to see if we can increase radius
            original_x, original_y = x, y
            
            # Try a few moves for position improvement
            best_x, best_y, best_r = x, y, r
            best_score = -r  # Minimize negative radius (maximize positive radius)
            
            # Try different positions around current position
            directions = [(0, 0), (0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01)]
            for dx, dy in directions:
                test_x = x + dx
                test_y = y + dy
                
                # Check if new position is valid
                if 0.01 <= test_x <= 0.99 and 0.01 <= test_y <= 0.99:
                    # Check if this might allow for larger radius
                    new_r = min(test_x, test_y, 1 - test_x, 1 - test_y)
                    
                    # Check overlaps with neighbors
                    valid = True
                    for j in neighbors:
                        x2, y2, r2 = optimized[j]
                        dist = np.sqrt((test_x - x2)**2 + (test_y - y2)**2)
                        if dist < (new_r + r2):
                            valid = False
                            break
                            
                    if valid and new_r > r:
                        # This is a potential improvement
                        if new_r > best_r:
                            best_r = new_r
                            best_x, best_y = test_x, test_y
                            
            # Update if we found an improvement
            if best_r > r:
                optimized[i, 0] = best_x
                optimized[i, 1] = best_y
                optimized[i, 2] = best_r
                
        return optimized

    def grid_search_refinement(self, circles: np.ndarray, max_iterations: int = 50) -> np.ndarray:
        """Use grid search to improve solution quality."""
        refined = circles.copy()
        
        for iteration in range(max_iterations):
            # Create a coarser grid for exploration
            grid_points = []
            for i in range(self.small_grid_size):
                for j in range(self.small_grid_size):
                    x = self.small_cell_size * (i + 0.5)
                    y = self.small_cell_size * (j + 0.5)
                    # Only consider interior points
                    if 0.01 <= x <= 0.99 and 0.01 <= y <= 0.99:
                        grid_points.append((x, y))
                        
            # Try different positions for each circle
            for i in range(len(refined)):
                best_x, best_y, best_r = refined[i]
                best_score = -refined[i, 2]  # Negative because we want to maximize
                
                # Try some positions near current circle
                x, y, r = refined[i]
                for dx in [-0.02, -0.01, 0, 0.01, 0.02]:
                    for dy in [-0.02, -0.01, 0, 0.01, 0.02]:
                        test_x = x + dx
                        test_y = y + dy
                        
                        if 0.01 <= test_x <= 0.99 and 0.01 <= test_y <= 0.99:
                            # Compute potential new radius
                            new_r = min(test_x, test_y, 1 - test_x, 1 - test_y)
                            
                            # Check for overlaps with other circles
                            valid = True
                            for j in range(len(refined)):
                                if i != j:
                                    x2, y2, r2 = refined[j]
                                    dist = np.sqrt((test_x - x2)**2 + (test_y - y2)**2)
                                    if dist < (new_r + r2):
                                        valid = False
                                        break
                                        
                            if valid and new_r > best_r:
                                best_r = new_r
                                best_x, best_y = test_x, test_y
                                
                # Update if improvement was found
                if best_r > refined[i, 2]:
                    refined[i, 0] = best_x
                    refined[i, 1] = best_y
                    refined[i, 2] = best_r
                    
        return refined

    def solve(self, n_circles: int = 26, max_iterations: int = 200) -> np.ndarray:
        """Main solving procedure."""
        # Initialize circles
        circles = self.initialize_circles(n_circles)
        
        best_circles = circles.copy()
        best_sum = -self.objective_function(circles)
        
        # Main optimization loop
        for iteration in range(max_iterations):
            if iteration % 10 == 0:
                current_sum = -self.objective_function(circles)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = circles.copy()
                    
            # Try local optimization on each circle
            circles = self.optimize_local(circles)
            
            # Apply grid refinement
            circles = self.grid_search_refinement(circles, 20)
            
            # Resolve any remaining overlaps
            circles = self.resolve_overlaps(circles)
            
            # Ensure feasibility
            circles = self.project_to_feasible_region(circles)
            
            # Early stopping criteria
            if iteration > 50 and abs(best_sum - (-self.objective_function(circles))) < 0.001:
                break
                
        return best_circles

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    packer = GridOptimizedCirclePacker()
    
    # Solve with more iterations
    circles = packer.solve(n_circles=26, max_iterations=300)
    
    # Final validation and refinement
    if not packer.is_valid(circles):
        # Apply final correction
        circles = packer.resolve_overlaps(circles)
        circles = packer.project_to_feasible_region(circles)
        
        # Last resort: if still invalid, fall back to structured arrangement
        if not packer.is_valid(circles):
            circles = np.zeros((26, 3))
            rows = 5
            cols = 5
            spacing_x = 1.0 / (cols + 1)
            spacing_y = 1.0 / (rows + 1)
            radius = min(spacing_x, spacing_y) * 0.3
            
            idx = 0
            for i in range(rows):
                for j in range(cols):
                    if idx >= 26:
                        break
                    x = (j + 1) * spacing_x
                    y = (i + 1) * spacing_y
                    circles[idx] = [x, y, radius]
                    idx += 1
                    
            # Adjust last few circles
            for i in range(idx, 26):
                circles[i] = [0.5, 0.5, 0.01]
    
    return circles

# EVOLVE-BLOCK-END