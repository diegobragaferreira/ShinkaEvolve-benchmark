# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import random
from typing import Tuple, List, Optional
import math

# Constants
GRID_SIZE = 50
CIRCLE_COUNT = 26
MAX_ITERATIONS = 1000
MIN_RADIUS = 0.001
MAX_RADIUS = 0.49

class GridBasedOptimizer:
    def __init__(self):
        self.grid_resolution = GRID_SIZE
        self.cell_size = 1.0 / GRID_SIZE
        self.free_cells = set()
        self.candidate_positions = []
        
    def initialize_grid(self, circles: np.ndarray):
        """Initialize grid with occupied cells based on current circles"""
        self.free_cells.clear()
        
        # Initialize all cells as free
        for i in range(self.grid_resolution):
            for j in range(self.grid_resolution):
                self.free_cells.add((i, j))
        
        # Mark cells as occupied based on existing circles
        for x, y, r in circles:
            self._mark_occupied_cells(x, y, r)
    
    def _mark_occupied_cells(self, cx: float, cy: float, r: float):
        """Mark cells that would be occupied by a circle"""
        # Get cell indices for circle boundaries
        left = max(0, int((cx - r) / self.cell_size))
        right = min(self.grid_resolution - 1, int((cx + r) / self.cell_size))
        bottom = max(0, int((cy - r) / self.cell_size))
        top = min(self.grid_resolution - 1, int((cy + r) / self.cell_size))
        
        for i in range(left, right + 1):
            for j in range(bottom, top + 1):
                # Check if this cell is actually inside the circle
                cell_center_x = (i + 0.5) * self.cell_size
                cell_center_y = (j + 0.5) * self.cell_size
                dist_sq = (cell_center_x - cx)**2 + (cell_center_y - cy)**2
                if dist_sq <= r**2:
                    self.free_cells.discard((i, j))
    
    def find_candidate_positions(self, circles: np.ndarray, min_dist: float = 0.05):
        """Find good candidate positions avoiding existing circles"""
        candidates = []
        
        # Sample potential positions in free cells
        free_list = list(self.free_cells)
        random.shuffle(free_list)
        
        for i, j in free_list[:min(1000, len(free_list))]:  # Limit sample size
            cell_center_x = (i + 0.5) * self.cell_size
            cell_center_y = (j + 0.5) * self.cell_size
            
            # Check if this position is far enough from existing circles
            valid = True
            for x, y, r in circles:
                dist = math.sqrt((cell_center_x - x)**2 + (cell_center_y - y)**2)
                if dist < r + min_dist:
                    valid = False
                    break
            
            if valid:
                candidates.append((cell_center_x, cell_center_y))
                
        return candidates[:100]  # Return top candidates
    
    def optimize_single_circle(self, circles: np.ndarray, idx: int, 
                              candidate_positions: List[Tuple[float, float]]) -> np.ndarray:
        """Optimize one specific circle using local search"""
        optimized = circles.copy()
        
        if len(candidate_positions) == 0:
            return optimized
            
        # Start with current position
        current_x, current_y, current_r = circles[idx]
        
        # Try different positions and radii
        best_x, best_y, best_r = current_x, current_y, current_r
        best_score = float('inf')
        
        # Test several positions and radii
        test_positions = candidate_positions[:20]  # Limit positions tested
        for pos_x, pos_y in test_positions:
            # Try to place with maximum possible radius
            max_r = min(pos_x, 1 - pos_x, pos_y, 1 - pos_y)  # Boundary constraints
            
            # Check overlap with other circles
            min_dist = float('inf')
            for i, (x, y, r) in enumerate(circles):
                if i != idx:
                    dist = math.sqrt((pos_x - x)**2 + (pos_y - y)**2)
                    min_dist = min(min_dist, dist)
            
            # If we can fit a circle without overlap
            if min_dist > 0:
                # Radius is limited by overlap with others and boundaries
                safe_radius = min(max_r, min_dist - 0.001)
                if safe_radius > MIN_RADIUS:
                    # Simple greedy optimization - try various radii
                    try_optimal_radius = min(safe_radius, MAX_RADIUS)
                    test_r = try_optimal_radius
                    
                    # Score is negative sum of radii (we want to maximize)
                    score = -test_r
                    
                    if score < best_score:
                        best_score = score
                        best_x, best_y, best_r = pos_x, pos_y, test_r
                        
        # Update the optimized circle
        optimized[idx] = [best_x, best_y, best_r]
        return optimized

def compute_penalty(circles: np.ndarray) -> float:
    """Compute penalty for constraint violations"""
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
    n = len(circles)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if distance < r1 + r2:
                overlap = (r1 + r2 - distance)
                penalty += overlap**2
                
    return penalty

def is_valid(circles: np.ndarray) -> bool:
    """Check if all circles are within bounds and non-overlapping"""
    n = len(circles)
    
    # Check boundary constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    
    # Check overlap constraints
    if n > 1:
        try:
            positions = [(x, y) for x, y, r in circles]
            tree = cKDTree(positions)
            
            # Query pairs within sum of radii distance
            pairs = tree.query_pairs(r=2.0, output_type='ndarray')
            
            # Check each potential overlapping pair
            for i, j in pairs:
                if i < j:  # Avoid duplicate checking
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < r1 + r2:
                        return False
        except Exception:
            # Fallback to brute force if tree fails
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < r1 + r2:
                        return False
                        
    return True

def calculate_sum_radii(circles: np.ndarray) -> float:
    """Calculate the sum of all radii."""
    return np.sum(circles[:, 2])

def solve_with_gridded_approach():
    """Main gridded optimization solver"""
    # Initialize grid optimizer
    grid_optimizer = GridBasedOptimizer()
    
    # Start with random valid configuration
    circles = np.zeros((CIRCLE_COUNT, 3))
    
    # Generate initial random placement with reasonable radii
    for i in range(CIRCLE_COUNT):
        while True:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            r = random.uniform(0.02, 0.08)
            
            # Ensure it's valid
            valid = True
            for j in range(i):
                prev_x, prev_y, prev_r = circles[j]
                dist = math.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                if dist < r + prev_r:
                    valid = False
                    break
                    
            if valid:
                circles[i] = [x, y, r]
                break
    
    # Iteratively improve the configuration
    for iteration in range(MAX_ITERATIONS):
        # Rebuild grid based on current state
        grid_optimizer.initialize_grid(circles)
        
        # Find candidate positions 
        candidates = grid_optimizer.find_candidate_positions(circles)
        
        # Optimize each circle individually
        improved = False
        for i in range(CIRCLE_COUNT):
            # Only optimize if we have candidates
            new_circles = grid_optimizer.optimize_single_circle(circles, i, candidates)
            
            # Check improvement
            if not is_valid(new_circles):
                continue
                
            old_sum = calculate_sum_radii(circles)
            new_sum = calculate_sum_radii(new_circles)
            
            if new_sum > old_sum:
                circles = new_circles
                improved = True
        
        # If no improvement, try a different strategy
        if not improved and iteration > 50:
            break
            
    return circles

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    # Use the gridded optimization approach
    circles = solve_with_gridded_approach()
    
    # Final validation and refinement
    if not is_valid(circles):
        # Apply refinement process
        for _ in range(10):
            # Simple refinement: push circles away from boundaries and each other
            for i in range(CIRCLE_COUNT):
                x, y, r = circles[i]
                # Adjust for boundaries
                r = min(r, x, 1-x, y, 1-y)
                x = max(r, min(1-r, x))
                y = max(r, min(1-r, y))
                circles[i] = [x, y, r]
                
            # Resolve overlaps
            for _ in range(10):
                changed = False
                for i in range(CIRCLE_COUNT):
                    for j in range(i+1, CIRCLE_COUNT):
                        x1, y1, r1 = circles[i]
                        x2, y2, r2 = circles[j]
                        dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        
                        if dist < r1 + r2:
                            # Move circles apart
                            if dist > 0.001:
                                dx = (x2 - x1) / dist
                                dy = (y2 - y1) / dist
                                move_dist = (r1 + r2 - dist) * 0.5
                                circles[i, 0] -= dx * move_dist * 0.2
                                circles[i, 1] -= dy * move_dist * 0.2
                                circles[j, 0] += dx * move_dist * 0.2
                                circles[j, 1] += dy * move_dist * 0.2
                                changed = True
                
                if not changed:
                    break
    
    # Final boundary clamping
    for i in range(CIRCLE_COUNT):
        x, y, r = circles[i]
        r = max(MIN_RADIUS, min(MAX_RADIUS, r))
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        circles[i] = [x, y, r]
        
    return circles

# EVOLVE-BLOCK-END