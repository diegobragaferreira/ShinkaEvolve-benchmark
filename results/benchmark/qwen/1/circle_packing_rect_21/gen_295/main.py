# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import differential_evolution
import random
from typing import Tuple, List, Dict, Any
import time

# Set seeds for determinism
random.seed(42)
np.random.seed(42)

class CirclePackingOptimizer:
    def __init__(self, n_circles: int = 21, rect_width: float = 1.0, rect_height: float = 1.0):
        self.n_circles = n_circles
        self.rect_width = rect_width
        self.rect_height = rect_height
        self.best_solution = None
        self.best_score = -float('inf')

    def distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def is_valid_solution(self, circles: np.ndarray) -> bool:
        """Check if all circles are within bounds and non-overlapping using efficient spatial hashing."""
        n = len(circles)

        # Check bounds - vectorized
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        radii = circles[:, 2]

        if np.any(x_coords - radii < 0) or np.any(x_coords + radii > self.rect_width) or \
           np.any(y_coords - radii < 0) or np.any(y_coords + radii > self.rect_height):
            return False

        # Use spatial indexing to detect overlaps efficiently
        if len(radii) > 0:
            avg_radius = np.mean(radii)
            # Dynamic cell size based on average radius
            cell_size = max(0.001, avg_radius * 1.2)  # Slightly larger than average radius for efficiency
        else:
            cell_size = 0.01  # Fallback

        # Grid dimensions
        grid_width = int(np.ceil(self.rect_width / cell_size)) + 1
        grid_height = int(np.ceil(self.rect_height / cell_size)) + 1

        # Initialize grid with lists for each cell
        grid = {}

        # Place circles into grid cells
        for i in range(n):
            x, y, r = circles[i]
            # Calculate grid bounds for this circle
            min_cell_x = max(0, int((x - r) / cell_size))
            max_cell_x = min(grid_width - 1, int((x + r) / cell_size))
            min_cell_y = max(0, int((y - r) / cell_size))
            max_cell_y = min(grid_height - 1, int((y + r) / cell_size))

            # Add circle to all relevant grid cells
            for gx in range(min_cell_x, max_cell_x + 1):
                for gy in range(min_cell_y, max_cell_y + 1):
                    if (gx, gy) not in grid:
                        grid[(gx, gy)] = []
                    grid[(gx, gy)].append(i)

        # Check for overlaps within each grid cell and adjacent cells
        for i in range(n):
            x1, y1, r1 = circles[i]
            # Get grid cell coordinates
            cell_x = int(x1 / cell_size)
            cell_y = int(y1 / cell_size)

            # Check nearby cells (3x3 neighborhood)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = cell_x + dx, cell_y + dy
                    if (nx, ny) in grid:
                        for j in grid[(nx, ny)]:
                            # Skip self-comparison and ensure we don't double-check
                            if i >= j:
                                continue
                            x2, y2, r2 = circles[j]
                            # Check if circles overlap using squared distances for efficiency
                            dx_squared = (x1 - x2)**2
                            dy_squared = (y1 - y2)**2
                            distance_squared = dx_squared + dy_squared
                            radii_sum = r1 + r2
                            if distance_squared < radii_sum * radii_sum:
                                return False

        return True

    def compute_constraint_distances(self, circles: np.ndarray) -> np.ndarray:
        """Compute minimum constraint distance for each circle (distance to nearest neighbor or boundary).

        This provides better insight into local constraint density than Voronoi areas alone.
        """
        n = len(circles)
        if n == 0:
            return np.array([])

        # Initialize constraint distances array
        constraint_distances = np.full(n, float('inf'))

        # Calculate distances to all other circles
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        radii = circles[:, 2]

        # For each circle, find minimum distance to any other circle
        for i in range(n):
            min_neighbor_dist = float('inf')

            # Check distance to all other circles
            for j in range(n):
                if i != j:
                    dx = x_coords[i] - x_coords[j]
                    dy = y_coords[i] - y_coords[j]
                    distance = np.sqrt(dx*dx + dy*dy)
                    min_neighbor_dist = min(min_neighbor_dist, distance)

            # Also consider boundary constraints
            boundary_dists = [
                x_coords[i],  # distance to left boundary
                self.rect_width - x_coords[i],  # distance to right boundary
                y_coords[i],  # distance to bottom boundary
                self.rect_height - y_coords[i]  # distance to top boundary
            ]

            # Minimum boundary distance
            min_boundary_dist = min(boundary_dists)

            # The constraint distance is the minimum of neighbor distance and boundary distance
            # But only consider it if it's actually constraining (smaller than current radius)
            min_constraint_dist = min(min_neighbor_dist, min_boundary_dist)

            # Ensure we don't consider distances that are larger than what would constrain the circle
            constraint_distances[i] = min_constraint_dist

        return constraint_distances

    def mutate_radius(self, circles: np.ndarray, idx: int, constraint_distances: np.ndarray = None) -> np.ndarray:
        """Mutate radius with adaptive delta based on actual constraint distances."""
        new_circles = circles.copy()
        old_r = new_circles[idx, 2]

        # Adaptive delta based on constraint distances
        # Circles with smaller constraint distances (tighter constraints) get smaller deltas
        max_delta = 0.01
        delta = max_delta

        if constraint_distances is not None and len(constraint_distances) > idx:
            # Use constraint distance as proxy for how much we can safely change
            # Constraint distance of 0.05 suggests we're at tight constraint, so use very small delta
            # Constraint distance of 0.2 suggests more room, so larger delta
            constraint_factor = np.clip(constraint_distances[idx] / 0.2, 0.1, 1.0)
            delta = max_delta * constraint_factor
        else:
            delta = max_delta

        # Random small perturbation with better distribution
        if np.random.random() < 0.7:  # 70% chance of small change
            delta_r = np.random.uniform(-delta*0.5, delta*0.5)
        else:  # 30% chance of larger change
            delta_r = np.random.uniform(-delta, delta)

        new_r = old_r + delta_r

        # Ensure positive radius with better clamping
        new_r = max(0.001, new_r)
        new_circles[idx, 2] = new_r

        return new_circles

    def mutate_position(self, circles: np.ndarray, idx: int, constraint_distances: np.ndarray = None) -> np.ndarray:
        """Mutate position with small random perturbation."""
        new_circles = circles.copy()
        old_x, old_y = new_circles[idx, 0], new_circles[idx, 1]

        # Adaptive step size based on constraint distances and boundary proximity
        max_delta = 0.05
        delta = max_delta

        if constraint_distances is not None and len(constraint_distances) > idx:
            # Based on constraint distance, adapt step size
            # Smaller constraint distance means tighter constraints, so smaller steps
            constraint_factor = np.clip(constraint_distances[idx] / 0.2, 0.1, 1.0)
            delta = max_delta * constraint_factor

        # Consider proximity to boundary
        x, y, r = new_circles[idx]
        boundary_distance = min(x, self.rect_width - x, y, self.rect_height - y)
        boundary_factor = min(1.0, boundary_distance / (r * 2.0))  # Scale based on radius
        boundary_delta = delta * boundary_factor

        # Use minimum of adaptive and boundary delta
        actual_delta = min(delta, boundary_delta)

        # Small random perturbation
        delta_x = np.random.uniform(-actual_delta, actual_delta)
        delta_y = np.random.uniform(-actual_delta, actual_delta)

        new_x = old_x + delta_x
        new_y = old_y + delta_y

        # Ensure within bounds with margin
        margin = 0.01
        new_x = np.clip(new_x, r + margin, self.rect_width - r - margin)
        new_y = np.clip(new_y, r + margin, self.rect_height - r - margin)

        new_circles[idx, 0] = new_x
        new_circles[idx, 1] = new_y

        return new_circles

    def evaluate_fitness(self, circles: np.ndarray) -> float:
        """Evaluate fitness as sum of radii."""
        return np.sum(circles[:, 2])

    def generate_smart_initialization(self) -> np.ndarray:
        """Generate a more intelligent initial pattern using density-based approach."""
        circles = np.zeros((self.n_circles, 3))

        # Start with a denser hexagonal pattern and then scale down
        rows = int(np.sqrt(self.n_circles))
        cols = int(self.n_circles / rows) + 1

        # Use more compact spacing initially
        spacing_x = self.rect_width / (cols + 0.5)
        spacing_y = self.rect_height / (rows + 0.5)

        # Initial radius that allows for good packing
        initial_radius = min(spacing_x, spacing_y) * 0.3

        for i in range(self.n_circles):
            row = i // cols
            col = i % cols
            x = (col + 0.5) * spacing_x
            y = (row + 0.5) * spacing_y

            # Offset every other row for hexagonal arrangement
            if row % 2 == 1:
                x += spacing_x / 2

            circles[i] = [x, y, initial_radius]

        return circles

    def generate_triangular_pattern(self) -> np.ndarray:
        """Generate triangular pattern."""
        circles = np.zeros((self.n_circles, 3))

        # Arrange in triangular pattern with better spacing
        sqrt_n = int(np.ceil(np.sqrt(self.n_circles)))
        spacing_x = self.rect_width / (sqrt_n + 1.5)
        spacing_y = self.rect_height / (sqrt_n + 1.5)

        idx = 0
        for i in range(sqrt_n):
            for j in range(sqrt_n):
                if idx >= self.n_circles:
                    break
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                # Slight offset for triangular pattern
                if i % 2 == 1:
                    x += spacing_x / 2
                circles[idx] = [x, y, 0.03]
                idx += 1
            if idx >= self.n_circles:
                break

        return circles

    def generate_square_pattern(self) -> np.ndarray:
        """Generate square grid pattern."""
        circles = np.zeros((self.n_circles, 3))

        sqrt_n = int(np.ceil(np.sqrt(self.n_circles)))
        spacing_x = self.rect_width / (sqrt_n + 1.5)
        spacing_y = self.rect_height / (sqrt_n + 1.5)

        idx = 0
        for i in range(sqrt_n):
            for j in range(sqrt_n):
                if idx >= self.n_circles:
                    break
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                circles[idx] = [x, y, 0.03]
                idx += 1
            if idx >= self.n_circles:
                break

        return circles

    def generate_random_pattern(self) -> np.ndarray:
        """Generate random initial pattern."""
        circles = np.zeros((self.n_circles, 3))

        # Generate random positions and reasonable initial radii
        for i in range(self.n_circles):
            x = np.random.uniform(0.05, self.rect_width - 0.05)
            y = np.random.uniform(0.05, self.rect_height - 0.05)
            r = np.random.uniform(0.02, 0.06)  # More varied initial radii
            circles[i] = [x, y, r]

        return circles

    def enhanced_local_optimization(self, circles: np.ndarray, max_iter: int = 50, patience: int = 10) -> np.ndarray:
        """Enhanced local optimization with constraint-distance-aware mutation."""
        current_circles = circles.copy()
        best_circles = current_circles.copy()
        best_fitness = self.evaluate_fitness(current_circles)

        patience_counter = 0
        last_improvement = 0

        # Get constraint distances for adaptive mutation
        constraint_distances = self.compute_constraint_distances(current_circles)

        # Identify boundary circles for special treatment
        boundary_indices = []
        for i in range(len(current_circles)):
            x, y, r = current_circles[i]
            # Check if circle is near boundary
            if (x <= r + 0.05 or x >= self.rect_width - r - 0.05 or
                y <= r + 0.05 or y >= self.rect_height - r - 0.05):
                boundary_indices.append(i)

        for iteration in range(max_iter):
            improved = False

            # Process boundary circles first (they're more constrained)
            if boundary_indices:
                for i in boundary_indices:
                    # Special handling for boundary circles with smaller deltas
                    mutated_rad = self.mutate_radius(current_circles, i, constraint_distances)
                    mutated_pos = self.mutate_position(current_circles, i, constraint_distances)

                    # Test both mutations
                    rad_fitness = self.evaluate_fitness(mutated_rad)
                    pos_fitness = self.evaluate_fitness(mutated_pos)

                    # Choose better mutation
                    if rad_fitness > pos_fitness:
                        if self.is_valid_solution(mutated_rad):
                            current_circles = mutated_rad
                            improved = True
                    else:
                        if self.is_valid_solution(mutated_pos):
                            current_circles = mutated_pos
                            improved = True

            # Process remaining circles
            remaining_indices = [i for i in range(len(current_circles)) if i not in boundary_indices]
            for i in remaining_indices:
                # Try position mutation with adaptive parameter
                mutated_pos = self.mutate_position(current_circles, i, constraint_distances)

                # Try radius mutation with adaptive delta
                mutated_rad = self.mutate_radius(current_circles, i, constraint_distances)

                # Evaluate both mutations
                pos_fitness = self.evaluate_fitness(mutated_pos)
                rad_fitness = self.evaluate_fitness(mutated_rad)

                # Choose the better one
                if pos_fitness > rad_fitness:
                    if self.is_valid_solution(mutated_pos):
                        current_circles = mutated_pos
                        improved = True
                else:
                    if self.is_valid_solution(mutated_rad):
                        current_circles = mutated_rad
                        improved = True

            # Update best solution and constraint distances for next iteration
            current_fitness = self.evaluate_fitness(current_circles)
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_circles = current_circles.copy()
                patience_counter = 0
                last_improvement = iteration
                # Recompute constraint distances after significant improvement
                constraint_distances = self.compute_constraint_distances(current_circles)
            else:
                patience_counter += 1

            # Early stopping if no improvement for too long
            if patience_counter >= patience or (iteration - last_improvement > max_iter // 2):
                break

        # Final boundary check and adjustment
        if not self.is_valid_solution(best_circles):
            # Try to fix boundary violations
            for i in range(len(best_circles)):
                x, y, r = best_circles[i]
                # Reposition to safe zone if needed
                if x - r < 0:
                    x = r + 0.01
                elif x + r > self.rect_width:
                    x = self.rect_width - r - 0.01
                if y - r < 0:
                    y = r + 0.01
                elif y + r > self.rect_height:
                    y = self.rect_height - r - 0.01
                best_circles[i] = [x, y, r]

        return best_circles

    def optimize(self) -> np.ndarray:
        """Main optimization loop with multiple initialization strategies."""
        # Try multiple initialization strategies with better prioritization
        initial_patterns = [
            self.generate_smart_initialization(),  # Best for dense packing
            self.generate_triangular_pattern(),
            self.generate_square_pattern(),
            self.generate_random_pattern()
        ]

        # Multi-start optimization with improved refinement
        for seed_pattern in initial_patterns:
            # Apply local optimization to get better starting points
            optimized_pattern = self.enhanced_local_optimization(seed_pattern, max_iter=25)

            # Further refine with more iterations for better quality
            final_circles = self.enhanced_local_optimization(optimized_pattern, max_iter=30)

            score = self.evaluate_fitness(final_circles)
            if score > self.best_score and self.is_valid_solution(final_circles):
                self.best_score = score
                self.best_solution = final_circles.copy()

        # Final fine-tuning with extended search
        if self.best_solution is not None:
            # Apply more extensive local optimization with higher patience
            self.best_solution = self.enhanced_local_optimization(
                self.best_solution, max_iter=60, patience=15
            )

        # Ensure final validity
        if self.best_solution is None:
            # Fallback to simple initialization
            self.best_solution = self.generate_random_pattern()
            self.best_solution = self.enhanced_local_optimization(
                self.best_solution, max_iter=80
            )

        return self.best_solution

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    optimizer = CirclePackingOptimizer(n_circles=21, rect_width=1.0, rect_height=1.0)
    return optimizer.optimize()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")