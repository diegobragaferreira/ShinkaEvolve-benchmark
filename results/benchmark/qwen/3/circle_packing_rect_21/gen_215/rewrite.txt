# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple, Optional

class CirclePacker:
    def __init__(self, n_circles: int = 21, rect_width: float = 1.25, rect_height: float = 0.75):
        self.n_circles = n_circles
        self.width = rect_width
        self.height = rect_height
        self.best_solution = None
        self.best_sum = 0.0

    def initialize_positions(self) -> np.ndarray:
        """Initialize circle positions using enhanced hybrid strategy."""
        positions = []
        
        # Phase 1: Strategic corner and edge points
        corner_points = [
            [0.1, 0.1], [self.width - 0.1, 0.1],
            [0.1, self.height - 0.1], [self.width - 0.1, self.height - 0.1]
        ]
        
        # Add center point
        center_point = [self.width/2, self.height/2]
        
        # Add edge midpoints
        edge_points = [
            [self.width/2, 0.1], [self.width/2, self.height - 0.1],
            [0.1, self.height/2], [self.width - 0.1, self.height/2]
        ]
        
        # Add all strategic points
        positions.extend(corner_points)
        positions.append(center_point)
        positions.extend(edge_points)
        
        # Phase 2: Hexagonal grid for remaining positions
        remaining_spots = self.n_circles - len(positions)
        if remaining_spots > 0:
            # Use hexagonal packing pattern for better distribution
            rows = max(2, int(np.sqrt(remaining_spots)))
            cols = max(2, int(np.ceil(remaining_spots / rows)))
            
            col_spacing = self.width / (cols + 1)
            row_spacing = self.height / (rows + 1)
            hex_offset = col_spacing * 0.5
            
            # Generate hexagonal pattern
            idx = 0
            for row in range(rows):
                for col in range(cols):
                    if len(positions) < self.n_circles:
                        x = (col + 1) * col_spacing
                        if row % 2 == 1:
                            x += hex_offset
                        y = (row + 1) * row_spacing
                        
                        # Ensure points are within bounds
                        x = max(0.05, min(self.width - 0.05, x))
                        y = max(0.05, min(self.height - 0.05, y))
                        positions.append([x, y])
                        idx += 1
        
        # Ensure exactly n_circles positions
        positions = positions[:self.n_circles]
        
        return np.array(positions)

    def initialize_circles(self, positions: np.ndarray) -> np.ndarray:
        """Create initial circles with small uniform radii."""
        circles = np.zeros((self.n_circles, 3))
        # Base radius based on available space
        base_radius = min(self.width, self.height) * 0.03
        for i in range(self.n_circles):
            circles[i] = [positions[i][0], positions[i][1], base_radius]
        return circles

    def calculate_max_radius_fast(self, circles: np.ndarray, target_idx: int) -> float:
        """Fast calculation of maximum radius for a specific circle."""
        x, y, _ = circles[target_idx]

        # Boundary constraints
        min_to_edges = min(x, y, self.width - x, self.height - y)

        # Other circles constraints (vectorized)
        other_positions = np.delete(circles, target_idx, axis=0)[:, :2]
        if len(other_positions) == 0:
            return min_to_edges

        distances = np.sqrt(np.sum((other_positions - [x, y])**2, axis=1))
        min_to_others = np.min(distances) if len(distances) > 0 else float('inf')

        # Maximum radius is constrained by both edges and other circles
        max_radius = min(min_to_edges, min_to_others)
        return max(0, max_radius)

    def is_valid_configuration_fast(self, circles: np.ndarray) -> bool:
        """Fast validation of circle configuration using vectorized operations."""
        if len(circles) == 0:
            return False

        # Check boundary constraints
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        radii = circles[:, 2]

        within_bounds = (
            (x_coords - radii >= 0) &
            (x_coords + radii <= self.width) &
            (y_coords - radii >= 0) &
            (y_coords + radii <= self.height)
        )

        if not np.all(within_bounds):
            return False

        # Check overlap constraints
        if len(circles) < 2:
            return True

        positions = circles[:, :2]
        distances = cdist(positions, positions)
        np.fill_diagonal(distances, np.inf)

        radius_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlaps = distances < radius_sums

        return not np.any(overlaps)

    def optimize_single_circle(self, circles: np.ndarray, idx: int,
                             step_size: float) -> np.ndarray:
        """Optimize a single circle's position and radius."""
        new_circles = circles.copy()
        x, y, r = new_circles[idx]

        # Try to maximize radius
        max_radius = self.calculate_max_radius_fast(new_circles, idx)
        if max_radius > 0.001:
            perturbation = random.uniform(-step_size * 0.3, step_size * 0.3)
            new_radius = max(0.001, min(max_radius, r + perturbation))
            new_circles[idx][2] = new_radius

        # Position perturbation with higher probability for better exploration
        if random.random() > 0.5:  # Increased probability from 0.6 to 0.5
            dx = random.uniform(-step_size, step_size)
            dy = random.uniform(-step_size, step_size)
            new_x = max(0.001, min(self.width - 0.001, x + dx))
            new_y = max(0.001, min(self.height - 0.001, y + dy))
            new_circles[idx][0] = new_x
            new_circles[idx][1] = new_y

        return new_circles

    def multi_scale_optimization(self, initial_circles: np.ndarray) -> np.ndarray:
        """Perform multi-scale optimization with adaptive learning."""
        current_circles = initial_circles.copy()
        best_circles = initial_circles.copy()
        best_sum = np.sum(initial_circles[:, 2])

        # Enhanced step sizes for better exploration/exploitation balance
        step_sizes = [0.15, 0.08, 0.03]  # Added finer granularity
        phases = [300, 300, 400]  # iterations per phase

        # Track recent improvements for adaptive behavior
        improvement_window = 20
        recent_improvements = []
        improvement_threshold = 0.005

        for phase_idx, (iterations, step_size) in enumerate(zip(phases, step_sizes)):
            phase_best = current_circles.copy()
            phase_best_sum = np.sum(current_circles[:, 2])

            # Adaptive step size adjustment based on recent performance
            for iteration in range(iterations):
                # Adaptive step size based on recent performance
                if len(recent_improvements) >= improvement_window:
                    avg_improvement = np.mean(recent_improvements[-improvement_window:])
                    if avg_improvement < improvement_threshold * 0.5:
                        # Reduce step size if improvement is slow
                        step_size = max(0.005, step_size * 0.9)
                    elif avg_improvement > improvement_threshold:
                        # Slightly increase step size if improvement is good
                        step_size = min(0.2, step_size * 1.05)

                # Randomly select circles to optimize (more thorough coverage)
                indices = list(range(self.n_circles))
                random.shuffle(indices)
                # Increase selection ratio for better exploration in early phases
                selection_ratio = min(0.9, 0.4 + phase_idx * 0.15)
                selected_indices = indices[:max(1, int(self.n_circles * selection_ratio))]

                # Apply optimizations
                new_circles = current_circles.copy()
                for idx in selected_indices:
                    new_circles = self.optimize_single_circle(new_circles, idx, step_size)

                # Validate and accept if better
                if self.is_valid_configuration_fast(new_circles):
                    new_sum = np.sum(new_circles[:, 2])
                    improvement = new_sum - best_sum
                    recent_improvements.append(max(0, improvement))

                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_circles = new_circles.copy()

                    if new_sum > phase_best_sum:
                        phase_best_sum = new_sum
                        phase_best = new_circles.copy()

                    current_circles = new_circles
                else:
                    # Revert to best in phase if invalid
                    current_circles = phase_best.copy()

                # Early stopping if no improvement in recent iterations
                if len(recent_improvements) > 20 and np.mean(recent_improvements[-10:]) < 1e-6:
                    break

            # Reset for next phase with best found so far
            current_circles = best_circles.copy()

        return best_circles

    def refine_solution(self, circles: np.ndarray) -> np.ndarray:
        """Final refinement of the solution."""
        refined = circles.copy()
        for i in range(self.n_circles):
            # Fine-grained optimization for each circle
            max_radius = self.calculate_max_radius_fast(refined, i)
            if max_radius > 0.001:
                refined[i][2] = max(0.001, min(max_radius, refined[i][2]))

        return refined

    def get_result(self) -> np.ndarray:
        """Return the optimized circles."""
        if self.best_solution is None:
            # Fallback to basic initialization if no optimization completed
            positions = self.initialize_positions()
            self.best_solution = self.initialize_circles(positions)
        return self.best_solution

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create packer instance with rectangle dimensions that maximize area utilization
    packer = CirclePacker(rect_width=1.25, rect_height=0.75)

    # Phase 1: Initialization
    positions = packer.initialize_positions()
    initial_circles = packer.initialize_circles(positions)

    # Phase 2: Multi-scale optimization
    optimized_circles = packer.multi_scale_optimization(initial_circles)

    # Phase 3: Final refinement
    final_circles = packer.refine_solution(optimized_circles)

    # Store best solution
    packer.best_solution = final_circles

    # Ensure minimum radius
    for i in range(packer.n_circles):
        if final_circles[i][2] < 0.001:
            final_circles[i][2] = 0.01

    return final_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")