# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List

# Global constants
RECT_PERIMETER = 4.0
RECT_WIDTH_HEIGHT_RATIO = 1.0  # Square rectangle for simplicity
RECT_WIDTH = RECT_PERIMETER / (2 * (1 + RECT_WIDTH_HEIGHT_RATIO))
RECT_HEIGHT = RECT_PERIMETER / (2 * (1 + RECT_WIDTH_HEIGHT_RATIO))

# Spatial indexing for fast collision detection
class SpatialIndex:
    def __init__(self, cell_size: float = 0.1):
        self.cell_size = cell_size
        self.grid = {}
        
    def _get_cell(self, x: float, y: float) -> Tuple[int, int]:
        return (int(x // self.cell_size), int(y // self.cell_size))
    
    def insert(self, idx: int, x: float, y: float, r: float):
        cell = self._get_cell(x, y)
        if cell not in self.grid:
            self.grid[cell] = []
        self.grid[cell].append((idx, x, y, r))
    
    def get_candidates(self, x: float, y: float, r: float) -> List[Tuple[int, float, float, float]]:
        candidates = []
        cell = self._get_cell(x, y)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell[0] + dx, cell[1] + dy)
                if neighbor_cell in self.grid:
                    candidates.extend(self.grid[neighbor_cell])
        return candidates

def validate_placement(circles: np.ndarray, idx: int, x: float, y: float, r: float, 
                      spatial_index: SpatialIndex) -> bool:
    """Check if placing a circle at (x,y) with radius r is valid."""
    # Check boundary constraints
    if x - r < 0 or x + r > RECT_WIDTH or y - r < 0 or y + r > RECT_HEIGHT:
        return False
    
    # Check collisions with existing circles using spatial index
    candidates = spatial_index.get_candidates(x, y, r)
    for i, cx, cy, cr in candidates:
        # Calculate distance between centers
        dx = x - cx
        dy = y - cy
        distance = np.sqrt(dx * dx + dy * dy)
        # Check if circles overlap
        if distance < (r + cr):
            return False
            
    return True

def compute_max_radius(circles: np.ndarray, x: float, y: float, 
                      spatial_index: SpatialIndex) -> float:
    """Compute maximum radius for a circle at position (x,y) without overlap."""
    max_r = min(x, RECT_WIDTH - x, y, RECT_HEIGHT - y)
    
    # Check collisions with existing circles
    candidates = spatial_index.get_candidates(x, y, max_r)
    for i, cx, cy, cr in candidates:
        dx = x - cx
        dy = y - cy
        distance = np.sqrt(dx * dx + dy * dy)
        max_r = min(max_r, distance - cr)
        
    return max(max_r, 0.001)  # Ensure positive radius

def evaluate_solution(circles: np.ndarray) -> float:
    """Evaluate fitness (sum of radii) for a solution."""
    return np.sum(circles[:, 2])

def initialize_circles(n: int) -> np.ndarray:
    """Initialize circles using a hybrid approach."""
    circles = np.zeros((n, 3))
    
    # Place some circles in strategic positions (corners and center)
    positions = [
        (RECT_WIDTH * 0.1, RECT_HEIGHT * 0.1),          # Bottom-left
        (RECT_WIDTH * 0.9, RECT_HEIGHT * 0.1),          # Bottom-right
        (RECT_WIDTH * 0.1, RECT_HEIGHT * 0.9),          # Top-left
        (RECT_WIDTH * 0.9, RECT_HEIGHT * 0.9),          # Top-right
        (RECT_WIDTH / 2, RECT_HEIGHT / 2),              # Center
    ]
    
    # Initialize with small circles in strategic positions
    for i, (x, y) in enumerate(positions[:min(len(positions), n)]):
        circles[i] = [x, y, 0.02]
    
    # Fill remaining slots with random positions
    for i in range(len(positions), n):
        while True:
            x = random.uniform(0.01, RECT_WIDTH - 0.01)
            y = random.uniform(0.01, RECT_HEIGHT - 0.01)
            # Try to find a valid radius
            r = 0.01
            circles[i] = [x, y, r]
            break
    
    return circles

def local_search_step(circles: np.ndarray, spatial_index: SpatialIndex) -> np.ndarray:
    """Perform one step of local search improving the current solution."""
    new_circles = circles.copy()
    n = len(new_circles)
    
    # Select a random circle to adjust
    idx = random.randint(0, n - 1)
    old_x, old_y, old_r = new_circles[idx]
    
    # Try to improve this circle's position and radius
    best_x, best_y, best_r = old_x, old_y, old_r
    
    # Try to increase radius first
    new_r = compute_max_radius(new_circles, old_x, old_y, spatial_index)
    if new_r > best_r:
        best_r = new_r
        
    # Try small position adjustments
    for _ in range(100):
        dx = random.uniform(-0.02, 0.02)
        dy = random.uniform(-0.02, 0.02)
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Check if new position is valid
        if validate_placement(new_circles, idx, new_x, new_y, best_r, spatial_index):
            best_x, best_y = new_x, new_y
            break
    
    new_circles[idx] = [best_x, best_y, best_r]
    
    # Update spatial index
    spatial_index.grid.clear()
    for i in range(len(new_circles)):
        x, y, r = new_circles[i]
        spatial_index.insert(i, x, y, r)
    
    return new_circles

def simulate_annealing(initial_circles: np.ndarray, max_iter: int = 10000) -> np.ndarray:
    """Run simulated annealing to optimize circle packing."""
    circles = initial_circles.copy()
    spatial_index = SpatialIndex()
    
    # Initialize spatial index
    for i in range(len(circles)):
        x, y, r = circles[i]
        spatial_index.insert(i, x, y, r)
    
    current_fitness = evaluate_solution(circles)
    best_fitness = current_fitness
    best_circles = circles.copy()
    
    # Annealing parameters
    temp = 1.0
    cooling_rate = 0.9999
    min_temp = 1e-5
    
    # Adaptive parameters
    max_steps_per_temp = 100
    
    for iteration in range(max_iter):
        # Cool down temperature
        if temp < min_temp:
            break
            
        # Take multiple steps at current temperature
        for step in range(max_steps_per_temp):
            # Local search step
            new_circles = local_search_step(circles, spatial_index)
            
            new_fitness = evaluate_solution(new_circles)
            
            # Accept or reject the move
            delta = new_fitness - current_fitness
            if delta > 0 or random.random() < np.exp(delta / temp):
                circles = new_circles
                current_fitness = new_fitness
                
                if current_fitness > best_fitness:
                    best_fitness = current_fitness
                    best_circles = circles.copy()
        
        temp *= cooling_rate
        
        # Early stopping based on improvement rate
        if iteration > 1000 and abs(delta) < 1e-6:
            break
    
    return best_circles

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Initialize circles using the hybrid approach
    circles = initialize_circles(21)
    
    # Run simulated annealing for optimization
    optimized_circles = simulate_annealing(circles, max_iter=5000)
    
    # Final refinement with local search
    spatial_index = SpatialIndex()
    for i in range(len(optimized_circles)):
        x, y, r = optimized_circles[i]
        spatial_index.insert(i, x, y, r)
    
    final_circles = optimized_circles.copy()
    for i in range(1000):  # Some final local refinement
        final_circles = local_search_step(final_circles, spatial_index)
    
    return final_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
