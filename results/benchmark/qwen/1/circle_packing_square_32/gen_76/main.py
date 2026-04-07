# EVOLVE-BLOCK-START
import numpy as np
import random
import math
from collections import defaultdict

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

class SpatialGrid:
    def __init__(self, grid_size=20):
        self.grid_size = grid_size
        self.cell_size = 1.0 / grid_size
        self.grid = defaultdict(list)
        
    def _cell_coords(self, x, y):
        """Get the grid cell coordinates for a point"""
        i = int(x / self.cell_size)
        j = int(y / self.cell_size)
        # Handle boundary case
        i = min(i, self.grid_size - 1)
        j = min(j, self.grid_size - 1)
        return (i, j)
    
    def add_circle(self, circle):
        """Add a circle to the grid"""
        x, y, r = circle
        cell = self._cell_coords(x, y)
        self.grid[cell].append(circle)
        
    def remove_circle(self, circle):
        """Remove a circle from the grid"""
        x, y, r = circle
        cell = self._cell_coords(x, y)
        if circle in self.grid[cell]:
            self.grid[cell].remove(circle)
    
    def get_neighbors(self, circle):
        """Get nearby circles in adjacent cells"""
        x, y, r = circle
        neighbors = []
        
        # Check the cell and all 8 adjacent cells
        cell_i, cell_j = self._cell_coords(x, y)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = cell_i + di, cell_j + dj
                if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                    neighbors.extend(self.grid[(ni, nj)])
        
        return neighbors

def check_collision(circle1, circle2):
    """Check if two circles collide"""
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    distance_squared = (x1 - x2)**2 + (y1 - y2)**2
    return distance_squared < (r1 + r2)**2

def compute_max_radius(x, y, circles, grid=None):
    """Compute the maximum radius for a circle at position (x,y) without overlapping existing circles"""
    if len(circles) == 0:
        return min(x, 1-x, y, 1-y)
    
    # Use spatial grid if provided for faster neighbor lookup
    if grid is not None:
        neighbors = grid.get_neighbors((x, y, 0))
        min_distance = float('inf')
        for cx, cy, cr in neighbors:
            distance = np.sqrt((x - cx)**2 + (y - cy)**2)
            min_distance = min(min_distance, distance)
    else:
        # Fallback to brute force for small number of circles
        min_distance = float('inf')
        for cx, cy, cr in circles:
            distance = np.sqrt((x - cx)**2 + (y - cy)**2)
            min_distance = min(min_distance, distance)
    
    # Maximum radius is limited by boundaries and distance to other circles
    boundary_radius = min(x, 1-x, y, 1-y)
    collision_radius = min_distance - 1e-8  # Very small epsilon to avoid numerical issues
    
    return min(boundary_radius, collision_radius) if collision_radius > 0 else 0

def is_valid_position(circle, circles, grid=None):
    """Check if a circle position is valid (within bounds and no collisions)"""
    x, y, r = circle

    # Check boundary constraints
    if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
        return False

    # Use spatial grid for collision detection
    if grid is not None:
        neighbors = grid.get_neighbors(circle)
        for existing_circle in neighbors:
            if check_collision(circle, existing_circle):
                return False
    else:
        # Fallback to brute force
        for existing_circle in circles:
            if check_collision(circle, existing_circle):
                return False

    return True

def place_circles_grid_initialization(circles, max_circles):
    """Initialize circles using a grid-based approach"""
    new_circles = circles.copy()
    placed = 0
    
    # Create spatial grid
    grid = SpatialGrid(grid_size=20)
    
    # Place initial strategic positions
    strategic_positions = [
        (0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9),  # corners
        (0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5),  # edges
        (0.5, 0.5),  # center
        (0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)  # diagonals
    ]
    
    # Place initial strategic circles
    for i, (x, y) in enumerate(strategic_positions[:min(12, max_circles)]):
        if placed >= max_circles:
            break
        max_radius = compute_max_radius(x, y, new_circles[:placed], grid)
        if max_radius > 0:
            new_circle = (x, y, max_radius)
            if is_valid_position(new_circle, new_circles[:placed], grid):
                new_circles[placed] = new_circle
                grid.add_circle(new_circle)
                placed += 1

    # Fill remaining spots using grid-based approach
    attempts = 0
    max_attempts = max_circles * 50
    
    while placed < max_circles and attempts < max_attempts:
        # Generate candidates in a systematic way across the grid
        candidates = []
        
        # Add points from grid cells for better distribution
        for i in range(20):
            for j in range(20):
                # Sample multiple points per cell for better coverage
                for _ in range(2):
                    cell_x = (i + random.random()) / 20.0
                    cell_y = (j + random.random()) / 20.0
                    if 0.01 <= cell_x <= 0.99 and 0.01 <= cell_y <= 0.99:
                        candidates.append((cell_x, cell_y))
        
        # Add random points for exploration
        for _ in range(1000):
            x = random.uniform(0.01, 0.99)
            y = random.uniform(0.01, 0.99)
            candidates.append((x, y))
        
        # Find the best valid circle among candidates
        best_circle = None
        best_radius = 0
        
        # Sample from candidates
        sample_size = min(200, len(candidates))
        sampled_candidates = random.sample(candidates, sample_size)
        
        for x, y in sampled_candidates:
            max_radius = compute_max_radius(x, y, new_circles[:placed], grid)
            if max_radius <= best_radius:
                continue
            test_circle = (x, y, max_radius)
            if is_valid_position(test_circle, new_circles[:placed], grid):
                best_circle = test_circle
                best_radius = max_radius

        if best_circle is not None:
            new_circles[placed] = best_circle
            grid.add_circle(best_circle)
            placed += 1
        else:
            # If we can't find a valid circle, just increment attempts
            attempts += 1
            
        attempts += 1

    return new_circles

def optimize_with_simulated_annealing(circles, iterations=200):
    """Optimize circle positions using simulated annealing with grid acceleration"""
    circles = circles.copy()
    
    # Create spatial grid for efficient collision detection
    grid = SpatialGrid(grid_size=20)
    for circle in circles:
        grid.add_circle(circle)
    
    # Parameters for simulated annealing
    temperature = 0.1
    cooling_rate = 0.995
    min_temperature = 0.001
    
    current_sum = sum(circle[2] for circle in circles)
    
    for i in range(iterations):
        # Decrease temperature
        if temperature > min_temperature:
            temperature *= cooling_rate
            
        # Try to modify one circle
        circle_idx = random.randint(0, len(circles) - 1)
        old_circle = circles[circle_idx]
        
        # Remove from grid
        grid.remove_circle(old_circle)
        
        # Generate new candidate position
        old_x, old_y, old_r = old_circle
        new_x = old_x + random.uniform(-0.03, 0.03)
        new_y = old_y + random.uniform(-0.03, 0.03)
        
        # Bound the new position
        new_x = max(0.01, min(0.99, new_x))
        new_y = max(0.01, min(0.99, new_y))
        
        # Compute new radius
        new_r = compute_max_radius(new_x, new_y, circles[:circle_idx] + circles[circle_idx+1:], grid)
        
        if new_r > 0:
            new_circle = (new_x, new_y, new_r)
            
            # Check validity
            if is_valid_position(new_circle, circles[:circle_idx] + circles[circle_idx+1:], grid):
                # Accept the change with probability based on energy difference
                old_sum = sum(circle[2] for circle in circles)
                new_sum = old_sum - old_r + new_r
                
                if new_sum > old_sum:
                    # Always accept better moves
                    circles[circle_idx] = new_circle
                    grid.add_circle(new_circle)
                    current_sum = new_sum
                else:
                    # Accept worse moves with some probability
                    if random.random() < math.exp((new_sum - old_sum) / temperature):
                        circles[circle_idx] = new_circle
                        grid.add_circle(new_circle)
                        current_sum = new_sum
            else:
                # Revert to original if invalid
                grid.add_circle(old_circle)
        else:
            # Revert to original if invalid radius
            grid.add_circle(old_circle)
    
    return circles

def expand_radii_with_constraints(circles):
    """Systematically expand radii while respecting constraints"""
    # Create spatial grid
    grid = SpatialGrid(grid_size=20)
    for circle in circles:
        grid.add_circle(circle)
    
    # Try to increase each radius step by step
    improved = True
    while improved:
        improved = False
        for i in range(len(circles)):
            old_x, old_y, old_r = circles[i]
            
            # Compute maximum possible radius
            max_r = compute_max_radius(old_x, old_y, 
                                     circles[:i] + circles[i+1:], 
                                     grid)
            
            if max_r > old_r + 1e-6:  # Only if we can increase
                # Test the expanded radius
                test_circle = (old_x, old_y, max_r)
                
                # Check validity with current neighbors
                valid = True
                neighbors = grid.get_neighbors(test_circle)
                for neighbor in neighbors:
                    if not is_valid_position(test_circle, [neighbor], grid):
                        valid = False
                        break
                        
                if valid:
                    # Update in grid and array
                    grid.remove_circle((old_x, old_y, old_r))
                    circles[i] = test_circle
                    grid.add_circle(test_circle)
                    improved = True
    
    return circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    circles = np.zeros((n, 3))
    
    # Step 1: Initialize using grid-based approach
    circles = place_circles_grid_initialization(circles, n)
    
    # Step 2: Refine with simulated annealing
    circles = optimize_with_simulated_annealing(circles, 300)
    
    # Step 3: Expand radii with constraint satisfaction
    circles = expand_radii_with_constraints(circles)
    
    # Step 4: Final optimization with a few more rounds
    circles = optimize_with_simulated_annealing(circles, 100)
    
    return circles

# EVOLVE-BLOCK-END