# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple, Optional
import time

class AdaptiveVoronoiPacker:
    def __init__(self, n_circles: int = 21):
        self.n_circles = n_circles
        # Start with a rectangle that tends to work well for circle packing
        self.width = 1.0
        self.height = 1.0
        self.best_solution = None
        self.best_sum = 0.0

    def initialize_positions(self) -> np.ndarray:
        """Improved initialization using hexagonal packing principles."""
        positions = []

        # 1. Corner placements
        corners = [
            [0.1, 0.1], [self.width - 0.1, 0.1],
            [0.1, self.height - 0.1], [self.width - 0.1, self.height - 0.1]
        ]
        positions.extend(corners)

        # 2. Center and edge midpoints
        midpoints = [
            [self.width/2, self.height/2],  # center
            [self.width/2, 0.1],            # top center
            [self.width/2, self.height - 0.1],  # bottom center
            [0.1, self.height/2],          # left center
            [self.width - 0.1, self.height/2]   # right center
        ]
        positions.extend(midpoints)

        # 3. Hexagonal-like grid for remaining positions - more systematic approach
        # Create a hexagonal lattice pattern
        rows = 4
        cols = 5
        col_spacing = self.width / (cols + 1)
        row_spacing = self.height / (rows + 1)
        hex_offset = col_spacing * 0.5

        # Populate grid with offset rows for hexagonal packing
        for row in range(rows):
            for col in range(cols):
                if len(positions) >= self.n_circles:
                    break
                x = (col + 1) * col_spacing
                if row % 2 == 1:
                    x += hex_offset
                y = (row + 1) * row_spacing
                positions.append([x, y])

        # 4. Fill remaining with more strategic random points
        while len(positions) < self.n_circles:
            # Prefer placing points away from corners to reduce conflicts
            x = random.uniform(0.05, self.width - 0.05)
            y = random.uniform(0.05, self.height - 0.05)
            # Bias towards center for better packing
            if random.random() < 0.4:
                # More centrally biased placement
                x = self.width/2 + (x - self.width/2) * 0.6
                y = self.height/2 + (y - self.height/2) * 0.6
            positions.append([x, y])

        # Keep only required number
        positions = positions[:self.n_circles]
        return np.array(positions)

    def initialize_circles(self, positions: np.ndarray) -> np.ndarray:
        """Create initial circles with reasonable radii."""
        circles = np.zeros((self.n_circles, 3))
        # Base radius based on available space - more conservative
        base_radius = min(self.width, self.height) * 0.05
        for i in range(self.n_circles):
            circles[i] = [positions[i][0], positions[i][1], base_radius]
        return circles

    def calculate_max_radius_for_circle(self, circles: np.ndarray, target_idx: int) -> float:
        """Calculate maximum possible radius for a specific circle."""
        x, y, _ = circles[target_idx]

        # Distance to edges (with safety margin)
        min_edge_distance = min(
            x - 0.001,          # left edge
            self.width - x - 0.001,   # right edge
            y - 0.001,          # bottom edge
            self.height - y - 0.001   # top edge
        )

        # Distance to other circles (minimum)
        min_other_distance = float('inf')
        for i in range(len(circles)):
            if i != target_idx:
                other_x, other_y, other_r = circles[i]
                dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                # Distance should be >= sum of radii
                min_other_distance = min(min_other_distance, dist - other_r)

        # Return minimum of edge and other-circle constraints
        return min(min_edge_distance, min_other_distance)

    def optimize_single_circle_radius(self, circles: np.ndarray, target_idx: int,
                                    max_radius: Optional[float] = None) -> float:
        """Optimize a single circle's radius using binary search within constraints."""
        if max_radius is None:
            max_radius = self.calculate_max_radius_for_circle(circles, target_idx)

        if max_radius <= 0.001:
            return 0.001

        # Binary search for optimal radius
        low = 0.001
        high = max_radius
        best_radius = 0.001

        # Instead of fixed iterations, use a more efficient approach with convergence check
        for _ in range(25):  # Increased iterations for better precision
            if abs(high - low) < 1e-6:
                break
            mid = (low + high) / 2
            # Temporarily update the circle with this radius
            temp_circles = circles.copy()
            temp_circles[target_idx][2] = mid

            # Check if this configuration is valid
            if self.validate_configuration(temp_circles):
                best_radius = mid
                low = mid
            else:
                high = mid

        return best_radius

    def validate_configuration(self, circles: np.ndarray) -> bool:
        """Fast validation using vectorized operations."""
        if len(circles) == 0:
            return False

        # Check boundary constraints
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        radii = circles[:, 2]

        within_bounds = (
            (x_coords - radii >= 0.001) &
            (x_coords + radii <= self.width - 0.001) &
            (y_coords - radii >= 0.001) &
            (y_coords + radii <= self.height - 0.001)
        )

        if not np.all(within_bounds):
            return False

        # Check overlap constraints using vectorization
        if len(circles) < 2:
            return True

        positions = circles[:, :2]
        distances = cdist(positions, positions)
        np.fill_diagonal(distances, np.inf)

        radius_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlaps = distances < radius_sums

        return not np.any(overlaps)

    def systematic_local_optimization(self, circles: np.ndarray, step_size: float) -> np.ndarray:
        """Systematic local optimization instead of randomized approach."""
        new_circles = circles.copy()

        # Sort circles by radius (larger first) to prioritize optimization of larger circles
        radius_order = np.argsort(new_circles[:, 2])[::-1]

        # Optimize circles in descending order of size
        for i in radius_order:
            # Get current max radius
            max_radius = self.calculate_max_radius_for_circle(new_circles, i)
            if max_radius > 0.001:
                # Optimize radius using binary search
                optimal_radius = self.optimize_single_circle_radius(new_circles, i, max_radius)
                new_circles[i][2] = optimal_radius

            # Systematically adjust position to improve packing
            x, y, r = new_circles[i]

            # Define search space around current position
            search_space = [
                (x + step_size * 0.5, y),      # right
                (x - step_size * 0.5, y),      # left
                (x, y + step_size * 0.5),      # up
                (x, y - step_size * 0.5),      # down
                (x + step_size * 0.3, y + step_size * 0.3),  # diagonal
                (x - step_size * 0.3, y - step_size * 0.3)   # diagonal
            ]

            # Try each position in search space for better packing
            best_x, best_y = x, y
            best_radius = r
            best_sum = np.sum(new_circles[:, 2])

            for new_x, new_y in search_space:
                new_x = max(0.001 + r, min(self.width - 0.001 - r, new_x))
                new_y = max(0.001 + r, min(self.height - 0.001 - r, new_y))

                # Compute max radius at this new location
                temp_circles = new_circles.copy()
                temp_circles[i] = [new_x, new_y, r]  # keep same radius for now

                max_radius_at_pos = self.calculate_max_radius_for_circle(temp_circles, i)
                if max_radius_at_pos > r:
                    # Try to increase radius at new position
                    temp_circles[i] = [new_x, new_y, max_radius_at_pos]
                    if self.validate_configuration(temp_circles):
                        new_sum = np.sum(temp_circles[:, 2])
                        if new_sum > best_sum:
                            best_sum = new_sum
                            best_x, best_y, best_radius = new_x, new_y, max_radius_at_pos

            new_circles[i] = [best_x, best_y, best_radius]

        return new_circles

    def adaptive_multi_scale_optimization(self, initial_circles: np.ndarray) -> np.ndarray:
        """Improved multi-scale optimization with proper convergence detection."""
        current_circles = initial_circles.copy()
        best_circles = initial_circles.copy()
        best_sum = np.sum(initial_circles[:, 2])

        # Track convergence for early stopping
        convergence_history = []
        max_convergence_window = 50

        # Different optimization phases with varying intensities
        phases = [
            {"iterations": 300, "step_size": 0.15, "selection_ratio": 0.8},
            {"iterations": 300, "step_size": 0.08, "selection_ratio": 0.9},
            {"iterations": 500, "step_size": 0.03, "selection_ratio": 1.0},
            {"iterations": 500, "step_size": 0.01, "selection_ratio": 1.0}
        ]

        for phase_num, phase in enumerate(phases):
            phase_iterations = phase["iterations"]
            step_size = phase["step_size"]
            selection_ratio = phase["selection_ratio"]

            # Track improvement for this phase
            phase_best_sum = best_sum
            phase_improvement_counter = 0
            phase_max_improvement = 0

            for iteration in range(phase_iterations):
                # For the first few iterations, optimize all circles
                if iteration < phase_iterations * 0.1:
                    selected_indices = list(range(self.n_circles))
                else:
                    # Select subset for diversity
                    n_selected = max(1, int(self.n_circles * selection_ratio))
                    indices = list(range(self.n_circles))
                    random.shuffle(indices)
                    selected_indices = indices[:n_selected]

                # Create candidate solution with systematic optimization
                candidate_circles = self.systematic_local_optimization(current_circles.copy(), step_size)

                # Validate and accept if better
                if self.validate_configuration(candidate_circles):
                    new_sum = np.sum(candidate_circles[:, 2])
                    improvement = new_sum - best_sum

                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_circles = candidate_circles.copy()
                        phase_best_sum = new_sum
                        phase_improvement_counter = 0
                        phase_max_improvement = max(phase_max_improvement, improvement)
                    else:
                        phase_improvement_counter += 1

                    current_circles = candidate_circles
                else:
                    # Revert to best solution if invalid
                    current_circles = best_circles.copy()

                # Convergence tracking
                convergence_history.append(best_sum)
                if len(convergence_history) > max_convergence_window:
                    convergence_history.pop(0)

                # Early stopping criteria
                if phase_improvement_counter > 100 and iteration > 100:
                    break

                # Check for convergence
                if len(convergence_history) >= 10:
                    recent_changes = [convergence_history[-i] - convergence_history[-i-1]
                                    for i in range(1, min(10, len(convergence_history)))]
                    if all(abs(change) < 1e-6 for change in recent_changes):
                        break

            # If no significant improvement in this phase, reduce step size
            if phase_best_sum > best_sum:
                # Continue with better solution
                pass
            else:
                # Reduce step size for next phase to explore finer details
                step_size *= 0.5

        return best_circles

    def refine_solution(self, circles: np.ndarray) -> np.ndarray:
        """Final comprehensive refinement with more careful optimization."""
        refined = circles.copy()

        # Multiple passes of refinement
        for pass_num in range(3):
            # First pass: optimize all radii
            for i in range(self.n_circles):
                max_radius = self.calculate_max_radius_for_circle(refined, i)
                if max_radius > 0.001:
                    optimal_radius = self.optimize_single_circle_radius(refined, i, max_radius)
                    refined[i][2] = optimal_radius

            # Second pass: systematic position optimization
            refined = self.systematic_local_optimization(refined, 0.01)

            # Third pass: fine tuning
            for i in range(self.n_circles):
                max_radius = self.calculate_max_radius_for_circle(refined, i)
                if max_radius > 0.001:
                    optimal_radius = self.optimize_single_circle_radius(refined, i, max_radius)
                    refined[i][2] = optimal_radius

        return refined

    def get_result(self) -> np.ndarray:
        """Return the optimized circles."""
        if self.best_solution is None:
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
    packer = AdaptiveVoronoiPacker()

    # Phase 1: Initialization with strategic hexagonal-based points
    positions = packer.initialize_positions()
    initial_circles = packer.initialize_circles(positions)

    # Phase 2: Adaptive multi-scale optimization
    optimized_circles = packer.adaptive_multi_scale_optimization(initial_circles)

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