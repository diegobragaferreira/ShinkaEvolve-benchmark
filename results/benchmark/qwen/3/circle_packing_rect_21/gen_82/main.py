# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
import random
from typing import Tuple, List
import time

# Global constants
RECT_PERIMETER = 4.0
RECT_WIDTH_HEIGHT_RATIO = 1.0  # Square rectangle for simplicity
RECT_WIDTH = RECT_PERIMETER / (2 * (1 + RECT_WIDTH_HEIGHT_RATIO))
RECT_HEIGHT = RECT_PERIMETER / (2 * (1 + RECT_WIDTH_HEIGHT_RATIO))
MAX_RADIUS = min(RECT_WIDTH, RECT_HEIGHT) / 2.0

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
    """Initialize circles using a multi-scale grid-based approach for better spatial distribution."""
    circles = np.zeros((n, 3))

    # Create a more systematic grid initialization
    # Use a hexagonal-like pattern for better coverage
    grid_size = int(np.ceil(np.sqrt(n)))
    spacing_x = RECT_WIDTH / (grid_size + 1)
    spacing_y = RECT_HEIGHT / (grid_size + 1)

    positions = []
    for i in range(grid_size):
        for j in range(grid_size):
            x = (i + 1) * spacing_x
            y = (j + 1) * spacing_y
            # Add some randomness to avoid perfect grid patterns
            x += random.uniform(-spacing_x/4, spacing_x/4)
            y += random.uniform(-spacing_y/4, spacing_y/4)
            # Keep within bounds
            x = max(0.01, min(RECT_WIDTH - 0.01, x))
            y = max(0.01, min(RECT_HEIGHT - 0.01, y))
            positions.append((x, y))

    # Fill with initial positions, prioritizing corners and center
    corner_positions = [
        (RECT_WIDTH * 0.1, RECT_HEIGHT * 0.1),          # Bottom-left
        (RECT_WIDTH * 0.9, RECT_HEIGHT * 0.1),          # Bottom-right
        (RECT_WIDTH * 0.1, RECT_HEIGHT * 0.9),          # Top-left
        (RECT_WIDTH * 0.9, RECT_HEIGHT * 0.9),          # Top-right
        (RECT_WIDTH / 2, RECT_HEIGHT / 2),              # Center
    ]

    # Add corner positions if possible
    added_positions = []
    for pos in corner_positions:
        if len(added_positions) < n:
            x, y = pos
            x = max(0.01, min(RECT_WIDTH - 0.01, x))
            y = max(0.01, min(RECT_HEIGHT - 0.01, y))
            added_positions.append((x, y))

    # Add grid positions
    for pos in positions:
        if len(added_positions) < n:
            added_positions.append(pos)

    # Initialize circles with appropriate radii
    for i, (x, y) in enumerate(added_positions[:n]):
        # Start with a reasonable initial radius that considers boundaries
        max_boundary_radius = min(x, RECT_WIDTH - x, y, RECT_HEIGHT - y)
        # Also consider if we're near other circles (start small)
        initial_radius = min(0.05, max_boundary_radius)
        circles[i] = [x, y, initial_radius]

    return circles

def local_search_step(circles: np.ndarray, spatial_index: SpatialIndex) -> np.ndarray:
    """Perform one step of enhanced local search improving the current solution."""
    new_circles = circles.copy()
    n = len(new_circles)

    # Select a random circle to adjust
    idx = random.randint(0, n - 1)
    old_x, old_y, old_r = new_circles[idx]

    best_x, best_y, best_r = old_x, old_y, old_r
    best_fitness = evaluate_solution(new_circles)

    # Try multiple strategies for improvement
    # Strategy 1: Increase radius at current position
    new_r = compute_max_radius(new_circles, old_x, old_y, spatial_index)
    if new_r > best_r:
        best_r = new_r

    # Strategy 2: Try several position adjustments around current location
    best_local_move = (old_x, old_y, best_r)
    best_local_fitness = evaluate_solution(new_circles)

    # Sample more diverse movement patterns
    move_patterns = [
        (0, 0),                           # No move
        (0.01, 0), (-0.01, 0),           # Small horizontal
        (0, 0.01), (0, -0.01),           # Small vertical
        (0.005, 0.005), (-0.005, -0.005), # Diagonal
        (0.02, 0.02), (-0.02, -0.02),   # Larger diagonal
    ]

    # Try more thorough neighborhood search
    for dx, dy in move_patterns:
        new_x = old_x + dx
        new_y = old_y + dy

        # Keep within bounds with safety margin
        new_x = max(0.01, min(RECT_WIDTH - 0.01, new_x))
        new_y = max(0.01, min(RECT_HEIGHT - 0.01, new_y))

        # Compute new radius at this position
        new_r = compute_max_radius(new_circles, new_x, new_y, spatial_index)

        # Check validity and evaluate impact
        if new_r > 0.001:  # Valid radius
            # Temporarily update this circle
            temp_circles = new_circles.copy()
            temp_circles[idx] = [new_x, new_y, new_r]

            # Evaluate the change
            temp_fitness = evaluate_solution(temp_circles)

            if temp_fitness > best_local_fitness:
                best_local_fitness = temp_fitness
                best_local_move = (new_x, new_y, new_r)

    # Update with the best move found
    best_x, best_y, best_r = best_local_move

    # Now, we do a more careful check for radius increase
    if best_r < old_r:
        # Try to improve the radius without changing position
        new_r = compute_max_radius(new_circles, old_x, old_y, spatial_index)
        if new_r > best_r:
            best_r = new_r

    new_circles[idx] = [best_x, best_y, best_r]

    # Update spatial index
    spatial_index.grid.clear()
    for i in range(len(new_circles)):
        x, y, r = new_circles[i]
        spatial_index.insert(i, x, y, r)

    return new_circles

def simulate_annealing(initial_circles: np.ndarray, max_iter: int = 10000) -> np.ndarray:
    """Run enhanced simulated annealing to optimize circle packing."""
    circles = initial_circles.copy()
    spatial_index = SpatialIndex()

    # Initialize spatial index
    for i in range(len(circles)):
        x, y, r = circles[i]
        spatial_index.insert(i, x, y, r)

    current_fitness = evaluate_solution(circles)
    best_fitness = current_fitness
    best_circles = circles.copy()

    # Enhanced annealing parameters with multi-scale approach
    temp = 1.0
    cooling_rate = 0.9995  # Slightly slower cooling
    min_temp = 1e-6

    # Adaptive parameters with better convergence criteria
    max_steps_per_temp = 150
    improvement_threshold = 1e-5
    patience = 0
    max_patience = 100

    # Keep track of recent improvements for adaptive cooling
    recent_improvements = []
    improvement_window = 20

    for iteration in range(max_iter):
        # Cool down temperature
        if temp < min_temp:
            break

        # Take multiple steps at current temperature
        improved_in_epoch = False
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
                    improved_in_epoch = True

        # Track improvements for adaptive cooling
        recent_improvements.append(1 if improved_in_epoch else 0)
        if len(recent_improvements) > improvement_window:
            recent_improvements.pop(0)

        # Adaptively adjust cooling rate based on recent progress
        if len(recent_improvements) >= improvement_window:
            recent_improvement_rate = sum(recent_improvements) / len(recent_improvements)
            if recent_improvement_rate < 0.3:
                # Slow down cooling if progress is poor
                temp *= (cooling_rate * 0.9)
            else:
                temp *= cooling_rate

        # Early stopping based on patience
        if improved_in_epoch:
            patience = 0
        else:
            patience += 1

        if patience > max_patience:
            break

    return best_circles

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Multi-scale optimization approach
    # Phase 1: Coarse optimization with fewer iterations
    circles = initialize_circles(21)
    optimized_circles = simulate_annealing(circles, max_iter=1500)

    # Phase 2: Finer optimization with more iterations
    circles_fine = optimized_circles.copy()
    optimized_circles_fine = simulate_annealing(circles_fine, max_iter=3000)

    # Phase 3: Final comprehensive local search
    spatial_index = SpatialIndex()
    for i in range(len(optimized_circles_fine)):
        x, y, r = optimized_circles_fine[i]
        spatial_index.insert(i, x, y, r)

    final_circles = optimized_circles_fine.copy()
    # Increase local search iterations for final refinement
    for i in range(2000):
        final_circles = local_search_step(final_circles, spatial_index)

    return final_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")