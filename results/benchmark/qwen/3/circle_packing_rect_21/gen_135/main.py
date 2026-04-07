# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
import math
from typing import Tuple, List, Optional
from scipy.spatial import Voronoi
import time

class CirclePackingOptimizer:
    """Main optimizer class for 21-circle packing problem."""
    
    def __init__(self):
        self.best_result = None
        self.best_sum = -float('inf')
        self.rect_width = 1.5
        self.rect_height = 0.5
        random.seed(42)
        np.random.seed(42)
    
    def optimize_dimensions(self) -> Tuple[float, float]:
        """Find optimal rectangle dimensions to maximize packing efficiency."""
        best_ratio = 1.5  # empirical choice
        return best_ratio * 1.0, 1.0 / best_ratio
    
    def initialize_strategy(self, n_circles: int, width: float, height: float) -> np.ndarray:
        """Create initial configuration using hybrid approach."""
        circles = np.zeros((n_circles, 3))
        
        # Corner and edge positions
        corner_positions = [
            (width * 0.1, height * 0.1),    # bottom-left
            (width * 0.9, height * 0.1),    # bottom-right
            (width * 0.1, height * 0.9),    # top-left
            (width * 0.9, height * 0.9),    # top-right
            (width * 0.5, height * 0.1),    # bottom-middle
            (width * 0.5, height * 0.9),    # top-middle
            (width * 0.1, height * 0.5),    # left-middle
            (width * 0.9, height * 0.5),    # right-middle
        ]
        
        # Place corner circles
        for i in range(min(len(corner_positions), n_circles)):
            x, y = corner_positions[i]
            circles[i] = [x, y, 0.04]
        
        # Fill remaining with Voronoi-based approach
        remaining = n_circles - len(corner_positions)
        if remaining > 0:
            self._fill_remaining_circles(circles, remaining, len(corner_positions), width, height)
        
        return circles
    
    def _fill_remaining_circles(self, circles: np.ndarray, remaining: int, start_idx: int, 
                              width: float, height: float):
        """Fill remaining circle positions using Voronoi sampling."""
        try:
            # Generate random Voronoi points with boundary bias
            voronoi_points = []
            for _ in range(remaining * 3):  # Generate extra points for safety
                x = random.uniform(0.1 * width, 0.9 * width)
                y = random.uniform(0.1 * height, 0.9 * height)
                voronoi_points.append([x, y])
            
            vor = Voronoi(voronoi_points)
            # Use Voronoi vertices or generate grid if Voronoi fails
            if len(vor.vertices) > 0:
                sample_indices = list(range(min(len(vor.vertices), remaining)))
                for i, idx in enumerate(sample_indices):
                    if i >= remaining:
                        break
                    if idx < len(vor.vertices):
                        x, y = vor.vertices[idx]
                        x = max(0.01, min(width - 0.01, x))
                        y = max(0.01, min(height - 0.01, y))
                        circles[start_idx + i] = [x, y, 0.03]
            else:
                # Fallback to grid
                self._fill_grid_circles(circles, remaining, start_idx, width, height)
        except:
            # Final fallback to grid
            self._fill_grid_circles(circles, remaining, start_idx, width, height)
    
    def _fill_grid_circles(self, circles: np.ndarray, remaining: int, start_idx: int, 
                          width: float, height: float):
        """Fallback grid filling for remaining circles."""
        rows = int(math.ceil(math.sqrt(remaining)))
        cols = int(math.ceil(remaining / rows))
        cell_width = width / (cols + 1)
        cell_height = height / (rows + 1)
        idx = start_idx
        for i in range(rows):
            for j in range(cols):
                if idx >= len(circles):
                    break
                x_offset = 0.0 if i % 2 == 0 else 0.5
                x = (j + 1 + x_offset) * cell_width
                y = (i + 1) * cell_height
                x = max(0.01, min(width - 0.01, x))
                y = max(0.01, min(height - 0.01, y))
                circles[idx] = [x, y, 0.03]
                idx += 1
                if idx >= len(circles):
                    break
    
    def compute_max_radius_vectorized(self, positions: np.ndarray, radii: np.ndarray, 
                                   x: float, y: float, rect_width: float, rect_height: float) -> float:
        """Vectorized computation of max radius for a given position."""
        # Boundary distances
        min_bound = min(x, rect_width - x, y, rect_height - y)
        
        # Vectorized computation of distances to all existing circles
        if len(positions) > 0:
            dx = positions[:, 0] - x
            dy = positions[:, 1] - y
            distances = np.sqrt(dx*dx + dy*dy)
            
            # Avoid self-distance
            mask = distances > 0.0001
            if np.any(mask):
                min_dist_to_others = np.min(distances[mask] - radii[mask])
                actual_min_dist = min_dist_to_others
            else:
                actual_min_dist = float('inf')
        else:
            actual_min_dist = float('inf')
        
        max_radius = min(min_bound, actual_min_dist if actual_min_dist < float('inf') else float('inf'))
        return max(0.001, max_radius)
    
    def validate_configuration_vectorized(self, circles: np.ndarray, 
                                       rect_width: float, rect_height: float) -> bool:
        """Vectorized validation of circle configuration for efficiency."""
        if len(circles) < 2:
            return True
            
        # Check boundary constraints
        if np.any(circles[:, 0] - circles[:, 2] < 0) or \
           np.any(circles[:, 0] + circles[:, 2] > rect_width) or \
           np.any(circles[:, 1] - circles[:, 2] < 0) or \
           np.any(circles[:, 1] + circles[:, 2] > rect_height):
            return False
            
        # Vectorized overlap detection
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Create distance matrix
        dist_matrix = cdist(positions, positions)
        np.fill_diagonal(dist_matrix, float('inf'))  # Self-distances
        
        # Check for overlaps
        min_distances = np.min(dist_matrix, axis=1)
        radii_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Overlap check
        overlaps = min_distances < np.min(radii_sums, axis=0)
        return not np.any(overlaps)
    
    def calculate_radius_sum(self, circles: np.ndarray) -> float:
        """Calculate sum of all radii."""
        return np.sum(circles[:, 2])
    
    def local_refinement_step(self, circles: np.ndarray, rect_width: float, 
                            rect_height: float, iterations: int = 100, 
                            relax_overlap: bool = True) -> np.ndarray:
        """Perform local refinement with optimized search strategy."""
        current = circles.copy()
        current_sum = self.calculate_radius_sum(current)
        
        # Adaptive step sizing
        initial_step_size = 0.05
        min_step_size = 0.001
        step_reduction_factor = 0.95
        
        for iter_num in range(iterations):
            step_size = max(min_step_size, initial_step_size * (step_reduction_factor ** (iter_num // 5)))
            
            # Improved search strategy with better coverage
            search_moves = self._generate_search_moves(step_size, iter_num, iterations)
            
            # Parallel-like processing (batched updates) for better performance
            for i in range(len(current)):
                if not self._try_improve_circle(
                    current, i, search_moves, rect_width, rect_height, 
                    current_sum, relax_overlap, iter_num, iterations
                ):
                    continue
                # Update if improvement found
                current_sum = self.calculate_radius_sum(current)
        
        return current
    
    def _generate_search_moves(self, base_step: float, iter_num: int, total_iterations: int) -> List[Tuple[float, float]]:
        """Generate diverse search moves for optimization."""
        moves = []
        
        # Gaussian perturbations
        moves.extend([
            (base_step * random.gauss(0, 1), base_step * random.gauss(0, 1)),
            (base_step * random.gauss(0, 1), 0),
            (0, base_step * random.gauss(0, 1))
        ])
        
        # Coordinate-specific moves
        moves.extend([
            (base_step * random.uniform(-1, 1), 0),
            (0, base_step * random.uniform(-1, 1))
        ])
        
        # Small random moves
        moves.append((random.uniform(-base_step/2, base_step/2), 
                     random.uniform(-base_step/2, base_step/2)))
        
        # Systematic moves in later iterations
        if iter_num > total_iterations // 3:
            moves.extend([
                (base_step * random.choice([-1, 1]), 0),
                (0, base_step * random.choice([-1, 1])),
                (base_step, base_step),
                (-base_step, -base_step)
            ])
        
        # No-move baseline
        moves.append((0, 0))
        
        return moves
    
    def _try_improve_circle(self, current: np.ndarray, i: int, moves: List[Tuple[float, float]], 
                          rect_width: float, rect_height: float, current_sum: float, 
                          relax_overlap: bool, iter_num: int, total_iterations: int) -> bool:
        """Try to improve a single circle."""
        original_x, original_y, original_r = current[i]
        best_x, best_y, best_r = original_x, original_y, original_r
        best_sum = current_sum
        
        for dx, dy in moves:
            test_x = max(0.001, min(rect_width - 0.001, original_x + dx))
            test_y = max(0.001, min(rect_height - 0.001, original_y + dy))
            
            # Compute max radius
            temp_circles = current.copy()
            temp_circles[i] = [test_x, test_y, 0.01]  # Temporary small radius
            
            positions = temp_circles[:, :2]
            radii = temp_circles[:, 2]
            max_r = self.compute_max_radius_vectorized(
                positions, radii, test_x, test_y, rect_width, rect_height
            )
            
            test_r = min(max_r, max(0.001, original_r + random.uniform(-0.03, 0.03)))
            temp_circles[i] = [test_x, test_y, test_r]
            
            # Validation and update
            if relax_overlap and iter_num < total_iterations // 2:
                valid = (test_x - test_r >= 0 and test_x + test_r <= rect_width and
                        test_y - test_r >= 0 and test_y + test_r <= rect_height)
                if valid:
                    new_sum = self.calculate_radius_sum(temp_circles)
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_x, best_y, best_r = test_x, test_y, test_r
            else:
                if self.validate_configuration_vectorized(temp_circles, rect_width, rect_height):
                    new_sum = self.calculate_radius_sum(temp_circles)
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_x, best_y, best_r = test_x, test_y, test_r
        
        # Apply best update if found
        if best_sum > current_sum:
            current[i] = [best_x, best_y, best_r]
            return True
        return False
    
    def run_multi_start_optimization(self, n_starts: int = 8) -> np.ndarray:
        """Run multiple optimization starts."""
        rect_width, rect_height = self.optimize_dimensions()
        
        # Different initialization strategies
        init_strategies = ['hybrid', 'voronoi', 'hexagonal', 'random']
        
        for start_num in range(n_starts):
            strategy = init_strategies[start_num % len(init_strategies)]
            
            # Initialize circles
            circles = self._create_strategy_circles(strategy, rect_width, rect_height)
            
            # Phased refinement
            refined_1 = self.local_refinement_step(circles, rect_width, rect_height, 30, relax_overlap=True)
            refined_2 = self.local_refinement_step(refined_1, rect_width, rect_height, 50, relax_overlap=True)
            refined_3 = self.local_refinement_step(refined_2, rect_width, rect_height, 80, relax_overlap=False)
            
            final_sum = self.calculate_radius_sum(refined_3)
            
            if final_sum > self.best_sum:
                self.best_sum = final_sum
                self.best_result = refined_3.copy()
                print(f"New best found: {final_sum}")
        
        return self.best_result
    
    def _create_strategy_circles(self, strategy: str, width: float, height: float) -> np.ndarray:
        """Create circles using specific initialization strategy."""
        circles = np.zeros((21, 3))
        
        if strategy == 'hybrid':
            circles = self.initialize_strategy(21, width, height)
        elif strategy == 'voronoi':
            circles = self.initialize_strategy(21, width, height)
            circles = self.local_refinement_step(circles, width, height, 20)
        elif strategy == 'hexagonal':
            rows = int(math.ceil(math.sqrt(21)))
            cols = int(math.ceil(21 / rows))
            cell_width = width / (cols + 1)
            cell_height = height / (rows + 1)
            idx = 0
            for i in range(rows):
                for j in range(cols):
                    if idx >= 21:
                        break
                    x_offset = 0.0 if i % 2 == 0 else 0.5
                    x = (j + 1 + x_offset) * cell_width
                    y = (i + 1) * cell_height
                    x = max(0.01, min(width - 0.01, x))
                    y = max(0.01, min(height - 0.01, y))
                    circles[idx] = [x, y, 0.02]
                    idx += 1
                    if idx >= 21:
                        break
        else:  # random
            for i in range(21):
                x = random.uniform(0.01, width - 0.01)
                y = random.uniform(0.01, height - 0.01)
                circles[i] = [x, y, 0.02]
        
        return circles

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    optimizer = CirclePackingOptimizer()
    final_circles = optimizer.run_multi_start_optimization(8)
    
    # Final validation and cleanup
    if final_circles is not None:
        # Verify validity
        while not optimizer.validate_configuration_vectorized(
            final_circles, optimizer.rect_width, optimizer.rect_height
        ):
            final_circles = optimizer.initialize_strategy(21, optimizer.rect_width, optimizer.rect_height)
            final_circles = optimizer.local_refinement_step(final_circles, optimizer.rect_width, 
                                                          optimizer.rect_height, 50)
    
    return final_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")