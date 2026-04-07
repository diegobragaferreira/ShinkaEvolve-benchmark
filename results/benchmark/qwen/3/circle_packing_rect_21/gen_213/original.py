# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
import random
from typing import Tuple, Optional

class AdaptiveVoronoiPacker:
    def __init__(self, n_circles: int = 21):
        self.n_circles = n_circles
        # Start with a rectangle that tends to work well for circle packing
        self.width = 1.25
        self.height = 0.75
        self.best_solution = None
        self.best_sum = 0.0

    def initialize_positions(self) -> np.ndarray:
        """Initialize circle positions using Voronoi-based strategic placement."""
        # Generate initial candidate points using multiple heuristics
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

        # 3. Hexagonal-like grid for remaining positions
        rows = 4
        cols = 5
        col_spacing = self.width / (cols + 1)
        row_spacing = self.height / (rows + 1)
        hex_offset = col_spacing * 0.5

        # Populate grid with offset rows
        for row in range(rows):
            for col in range(cols):
                if len(positions) >= self.n_circles:
                    break
                x = (col + 1) * col_spacing
                if row % 2 == 1:
                    x += hex_offset
                y = (row + 1) * row_spacing
                positions.append([x, y])

        # 4. Fill remaining with random points
        while len(positions) < self.n_circles:
            x = random.uniform(0.05, self.width - 0.05)
            y = random.uniform(0.05, self.height - 0.05)
            positions.append([x, y])

        # Keep only required number
        positions = positions[:self.n_circles]
        return np.array(positions)

    def initialize_circles(self, positions: np.ndarray) -> np.ndarray:
        """Create initial circles with reasonable radii."""
        circles = np.zeros((self.n_circles, 3))
        # Base radius based on available space
        base_radius = min(self.width, self.height) * 0.04
        for i in range(self.n_circles):
            circles[i] = [positions[i][0], positions[i][1], base_radius]
        return circles

    def get_voronoi_constraints(self, circles: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Get Voronoi-based constraints for each circle."""
        # Create Voronoi diagram for current circle positions
        vor = Voronoi(circles[:, :2])

        # Map each circle to its Voronoi region
        voronoi_regions = {}
        for i in range(len(vor.points)):
            # Find which Voronoi region contains this point
            point = vor.points[i]
            # Find closest vertex in Voronoi diagram
            region_vertices = []
            for j, region in enumerate(vor.regions):
                if len(region) > 0:
                    # Compute distance to polygon vertices
                    region_points = [vor.vertices[k] for k in region if k >= 0]
                    if region_points:
                        # Simple approximation: find nearest Voronoi vertex
                        distances = [distance.euclidean(point, v) for v in region_points]
                        if distances:
                            min_dist = min(distances)
                            if min_dist < 100:  # reasonable threshold
                                voronoi_regions[i] = region_points

        return vor.vertices, voronoi_regions

    def calculate_max_radius_for_circle(self, circles: np.ndarray, target_idx: int) -> float:
        """Calculate maximum possible radius for a specific circle using Voronoi constraints."""
        x, y, _ = circles[target_idx]

        # Distance to edges
        min_edge_distance = min(
            x,                  # left edge
            self.width - x,     # right edge
            y,                  # bottom edge
            self.height - y     # top edge
        )

        # Distance to other circles (minimum)
        min_other_distance = float('inf')
        for i in range(len(circles)):
            if i != target_idx:
                other_x, other_y, other_r = circles[i]
                dist = distance.euclidean([x, y], [other_x, other_y])
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

        for _ in range(20):  # Limit iterations to prevent infinite loop
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

    def local_circular_optimization(self, circles: np.ndarray, step_size: float) -> np.ndarray:
        """Perform local circular optimization including radius adjustments."""
        new_circles = circles.copy()

        # Try to improve every circle
        for i in range(self.n_circles):
            # Get current max radius
            max_radius = self.calculate_max_radius_for_circle(new_circles, i)
            if max_radius > 0.001:
                # Optimize radius using binary search
                optimal_radius = self.optimize_single_circle_radius(new_circles, i, max_radius)
                new_circles[i][2] = optimal_radius

            # Try to slightly adjust position for better fit
            if random.random() > 0.7:  # 30% chance of position adjustment
                x, y, r = new_circles[i]
                # Small perturbation
                dx = random.uniform(-step_size * 0.5, step_size * 0.5)
                dy = random.uniform(-step_size * 0.5, step_size * 0.5)
                new_x = max(0.001, min(self.width - 0.001, x + dx))
                new_y = max(0.001, min(self.height - 0.001, y + dy))
                new_circles[i][0] = new_x
                new_circles[i][1] = new_y

        return new_circles

    def adaptive_multi_scale_optimization(self, initial_circles: np.ndarray) -> np.ndarray:
        """Perform multi-scale optimization with adaptive parameters."""
        current_circles = initial_circles.copy()
        best_circles = initial_circles.copy()
        best_sum = np.sum(initial_circles[:, 2])

        # Different optimization phases with varying intensities
        phases = [
            {"iterations": 200, "step_size": 0.15, "selection_ratio": 0.6},
            {"iterations": 200, "step_size": 0.08, "selection_ratio": 0.7},
            {"iterations": 300, "step_size": 0.03, "selection_ratio": 0.8},
            {"iterations": 300, "step_size": 0.015, "selection_ratio": 0.9}
        ]

        for phase in phases:
            phase_iterations = phase["iterations"]
            step_size = phase["step_size"]
            selection_ratio = phase["selection_ratio"]

            # Track progress for early stopping
            last_improvement = 0
            improvement_counter = 0

            for iteration in range(phase_iterations):
                # Select subset of circles for optimization
                n_selected = max(1, int(self.n_circles * selection_ratio))
                indices = list(range(self.n_circles))
                random.shuffle(indices)
                selected_indices = indices[:n_selected]

                # Create candidate solution
                candidate_circles = current_circles.copy()

                # Apply local optimization to selected circles
                for idx in selected_indices:
                    # For this approach, we'll do both position and radius optimization
                    max_radius = self.calculate_max_radius_for_circle(candidate_circles, idx)
                    if max_radius > 0.001:
                        # Optimize radius first for this circle
                        optimal_radius = self.optimize_single_circle_radius(candidate_circles, idx, max_radius)
                        candidate_circles[idx][2] = optimal_radius

                    # Slightly adjust position if selected
                    if random.random() > 0.6:
                        x, y, r = candidate_circles[idx]
                        # Small perturbation
                        dx = random.uniform(-step_size * 0.3, step_size * 0.3)
                        dy = random.uniform(-step_size * 0.3, step_size * 0.3)
                        new_x = max(0.001, min(self.width - 0.001, x + dx))
                        new_y = max(0.001, min(self.height - 0.001, y + dy))
                        candidate_circles[idx][0] = new_x
                        candidate_circles[idx][1] = new_y

                # Validate and accept if better
                if self.validate_configuration(candidate_circles):
                    new_sum = np.sum(candidate_circles[:, 2])
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_circles = candidate_circles.copy()
                        last_improvement = iteration
                        improvement_counter = 0
                    else:
                        improvement_counter += 1

                    current_circles = candidate_circles
                else:
                    # Revert to best solution if invalid
                    current_circles = best_circles.copy()

                # Early stopping if no improvement for a while
                if improvement_counter > 100 and iteration > 100:
                    break

        return best_circles

    def refine_solution(self, circles: np.ndarray) -> np.ndarray:
        """Final comprehensive refinement."""
        refined = circles.copy()

        # Do one final pass with focused optimization
        for i in range(self.n_circles):
            # Maximize each circle's radius independently
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

    # Phase 1: Initialization with strategic Voronoi-based points
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