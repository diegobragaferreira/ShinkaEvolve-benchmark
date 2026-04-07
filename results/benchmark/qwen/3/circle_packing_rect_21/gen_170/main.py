# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
import math
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
        """Initialize circle positions using enhanced Voronoi-based distribution for better spatial coverage."""
        # Use a more sophisticated Voronoi-inspired approach with hexagonal packing principles
        # to create better initial distribution

        # Start with corner points to ensure good boundary coverage
        positions = [
            [0.1, 0.1],           # bottom-left
            [self.width - 0.1, 0.1],  # bottom-right
            [0.1, self.height - 0.1], # top-left
            [self.width - 0.1, self.height - 0.1], # top-right
        ]

        # Add center point
        positions.append([self.width/2, self.height/2])

        # Add edge midpoints for better boundary coverage
        positions.extend([
            [self.width/2, 0.1],   # bottom center
            [self.width/2, self.height - 0.1],  # top center
            [0.1, self.height/2],  # left center
            [self.width - 0.1, self.height/2],  # right center
        ])

        # Add additional strategic points based on hexagonal packing principles
        # This helps to create more uniform spacing in the initial configuration
        hex_rows = 3
        hex_cols = 4
        col_spacing = self.width / (hex_cols + 1)
        row_spacing = self.height / (hex_rows + 1)
        hex_offset = col_spacing * 0.5

        for row in range(hex_rows):
            for col in range(hex_cols):
                if len(positions) < self.n_circles:
                    x = (col + 1) * col_spacing
                    if row % 2 == 1:
                        x += hex_offset
                    y = (row + 1) * row_spacing
                    positions.append([x, y])

        # Fill remaining spots with random points, but apply a rejection sampling
        # approach to ensure better distribution and avoid clustering
        remaining_spots = self.n_circles - len(positions)
        if remaining_spots > 0:
            # Use a more sophisticated approach that considers existing points
            # to avoid placing new points too close to existing ones
            for _ in range(remaining_spots * 5):  # Try more times to get good spots
                if len(positions) >= self.n_circles:
                    break
                x = random.uniform(0.05, self.width - 0.05)
                y = random.uniform(0.05, self.height - 0.05)

                # Check distance to existing points to avoid clustering
                valid_position = True
                for existing_pos in positions:
                    dist = math.sqrt((x - existing_pos[0])**2 + (y - existing_pos[1])**2)
                    if dist < min(self.width, self.height) * 0.1:  # Minimum spacing constraint
                        valid_position = False
                        break

                if valid_position:
                    positions.append([x, y])

        # Ensure exactly n_circles positions
        positions = positions[:self.n_circles]

        return np.array(positions)

    def initialize_circles(self, positions: np.ndarray) -> np.ndarray:
        """Create initial circles with small uniform radii."""
        circles = np.zeros((self.n_circles, 3))
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

        # Position perturbation
        if random.random() > 0.6:
            dx = random.uniform(-step_size, step_size)
            dy = random.uniform(-step_size, step_size)
            new_x = max(0.001, min(self.width - 0.001, x + dx))
            new_y = max(0.001, min(self.height - 0.001, y + dy))
            new_circles[idx][0] = new_x
            new_circles[idx][1] = new_y

        return new_circles

    def multi_scale_optimization(self, initial_circles: np.ndarray) -> np.ndarray:
        """Perform multi-scale optimization with different step sizes."""
        current_circles = initial_circles.copy()
        best_circles = initial_circles.copy()
        best_sum = np.sum(initial_circles[:, 2])

        # Adaptive step sizes for different optimization phases
        step_sizes = [0.15, 0.08, 0.03]
        phases = [300, 300, 400]  # iterations per phase

        for phase_idx, (iterations, step_size) in enumerate(zip(phases, step_sizes)):
            phase_best = current_circles.copy()
            phase_best_sum = np.sum(current_circles[:, 2])

            for iteration in range(iterations):
                # Randomly select circles to optimize
                indices = list(range(self.n_circles))
                random.shuffle(indices)
                selected_indices = indices[:max(1, self.n_circles // 3)]

                # Apply optimizations
                new_circles = current_circles.copy()
                for idx in selected_indices:
                    new_circles = self.optimize_single_circle(new_circles, idx, step_size)

                # Validate and accept if better
                if self.is_valid_configuration_fast(new_circles):
                    new_sum = np.sum(new_circles[:, 2])
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

    # Create packer instance
    packer = CirclePacker()

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