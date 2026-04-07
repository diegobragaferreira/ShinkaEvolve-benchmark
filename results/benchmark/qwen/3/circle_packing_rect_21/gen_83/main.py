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
    """Initialize circles using a hybrid approach combining grid placement with smart spacing."""
    circles = np.zeros((n, 3))
    
    # Start with systematic grid-based initialization
    grid_size = int(np.ceil(np.sqrt(n)))
    x_positions = np.linspace(0.05, RECT_WIDTH - 0.05, grid_size)
    y_positions = np.linspace(0.05, RECT_HEIGHT - 0.05, grid_size)
    
    # Fill with grid points first
    filled_positions = []
    for i, x in enumerate(x_positions):
        for j, y in enumerate(y_positions):
            if len(filled_positions) < n:
                filled_positions.append((x, y))
    
    # Add strategic positioning for good spatial distribution
    strategic_positions = [
        (RECT_WIDTH * 0.1, RECT_HEIGHT * 0.1),          # Bottom-left
        (RECT_WIDTH * 0.9, RECT_HEIGHT * 0.1),          # Bottom-right
        (RECT_WIDTH * 0.1, RECT_HEIGHT * 0.9),          # Top-left
        (RECT_WIDTH * 0.9, RECT_HEIGHT * 0.9),          # Top-right
        (RECT_WIDTH / 2, RECT_HEIGHT / 2),              # Center
    ]
    
    # Combine strategic and grid positions
    all_positions = filled_positions + strategic_positions
    unique_positions = list(set(all_positions))[:n]
    
    # Initialize with small circles
    for i, (x, y) in enumerate(unique_positions):
        circles[i] = [x, y, 0.01]
    
    # Fill remaining slots with random positions but ensure some minimum spacing
    for i in range(len(unique_positions), n):
        attempts = 0
        while attempts < 100:
            x = random.uniform(0.05, RECT_WIDTH - 0.05)
            y = random.uniform(0.05, RECT_HEIGHT - 0.05)
            
            # Ensure reasonable spacing from existing circles
            valid = True
            for j in range(i):
                prev_x, prev_y = circles[j, 0], circles[j, 1]
                dist = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                if dist < 0.05:  # Minimum spacing
                    valid = False
                    break
            
            if valid:
                circles[i] = [x, y, 0.01]
                break
            attempts += 1
    
    return circles

def local_search_step(circles: np.ndarray, spatial_index: SpatialIndex) -> np.ndarray:
    """Perform one step of enhanced local search improving the current solution."""
    new_circles = circles.copy()
    n = len(new_circles)
    
    # Select a random circle to adjust
    idx = random.randint(0, n - 1)
    old_x, old_y, old_r = new_circles[idx]
    
    # Try to improve this circle's position and radius
    best_x, best_y, best_r = old_x, old_y, old_r
    best_improvement = 0
    
    # Evaluate multiple potential moves around current position
    candidate_moves = []
    
    # Evaluate radius increase at current position first
    max_radius = compute_max_radius(new_circles, old_x, old_y, spatial_index)
    if max_radius > old_r:
        candidate_moves.append((old_x, old_y, max_radius, max_radius - old_r))
    
    # Try several directional moves with more systematic approach
    directions = [(0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01),
                  (0.005, 0.005), (-0.005, 0.005), (0.005, -0.005), (-0.005, -0.005),
                  (0.02, 0), (0, 0.02), (-0.02, 0), (0, -0.02)]
    
    for dx, dy in directions:
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Keep within bounds
        new_x = max(0.01, min(RECT_WIDTH - 0.01, new_x))
        new_y = max(0.01, min(RECT_HEIGHT - 0.01, new_y))
        
        # Compute max radius at new location
        new_r = compute_max_radius(new_circles, new_x, new_y, spatial_index)
        
        # Check if this is a valid improvement
        if new_r > old_r:
            improvement = new_r - old_r
            candidate_moves.append((new_x, new_y, new_r, improvement))
    
    # Select the best move among candidates
    if candidate_moves:
        best_move = max(candidate_moves, key=lambda x: x[3])  # Max improvement
        best_x, best_y, best_r = best_move[0], best_move[1], best_move[2]
    
    new_circles[idx] = [best_x, best_y, best_r]
    
    # Update spatial index
    spatial_index.grid.clear()
    for i in range(len(new_circles)):
        x, y, r = new_circles[i]
        spatial_index.insert(i, x, y, r)
    
    return new_circles

def simulate_annealing(initial_circles: np.ndarray, max_iter: int = 10000) -> np.ndarray:
    """Run simulated annealing to optimize circle packing with adaptive cooling."""
    circles = initial_circles.copy()
    spatial_index = SpatialIndex()
    
    # Initialize spatial index
    for i in range(len(circles)):
        x, y, r = circles[i]
        spatial_index.insert(i, x, y, r)
    
    current_fitness = evaluate_solution(circles)
    best_fitness = current_fitness
    best_circles = circles.copy()
    
    # Annealing parameters with adaptive behavior
    temp = 1.0
    cooling_rate = 0.9995  # Slightly slower cooling for better exploration
    min_temp = 1e-6
    max_steps_per_temp = 200  # More steps per temperature
    
    # Track recent improvements for adaptive cooling
    recent_improvements = []
    improvement_window = 50
    
    for iteration in range(max_iter):
        # Cool down temperature
        if temp < min_temp:
            break
            
        # Take multiple steps at current temperature
        improvement_count = 0
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
                    improvement_count += 1
        
        # Adaptive cooling: slow down cooling if progress is minimal
        recent_improvements.append(improvement_count)
        if len(recent_improvements) > improvement_window:
            recent_improvements.pop(0)
        
        # Adjust temperature based on recent progress
        avg_improvements = sum(recent_improvements) / len(recent_improvements)
        if avg_improvements < 2 and temp > min_temp * 10:
            # Slow down cooling if few improvements
            temp *= (cooling_rate * 0.9)
        else:
            temp *= cooling_rate
        
        # Early stopping based on improvement rate
        if iteration > 1000 and avg_improvements < 0.1:
            break
    
    return best_circles

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Initialize circles using the improved hybrid approach
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
