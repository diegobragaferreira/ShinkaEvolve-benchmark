# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
import random
import math
from typing import Tuple, Optional

class CirclePackingOptimizer:
    def __init__(self, rect_width: float = 1.0, rect_height: float = 1.0, n_circles: int = 21):
        self.rect_width = rect_width
        self.rect_height = rect_height
        self.n_circles = n_circles
        self.circles = np.zeros((n_circles, 3))

    def initialize_circles(self) -> None:
        """Initialize circles using enhanced hexagonal packing with Voronoi-based seeding"""
        from scipy.spatial import Voronoi
        import numpy as np

        # Generate initial candidate points using a combination of strategies
        candidate_points = []

        # Strategy 1: Hexagonal grid with Voronoi-derived refinement
        rows = 4
        cols = 6
        x_spacing = self.rect_width / (cols + 1)
        y_spacing = self.rect_height / (rows + 1)

        # Create base hexagonal grid
        for i in range(rows):
            for j in range(cols):
                x = (j + 1) * x_spacing
                y = (i + 1) * y_spacing
                if i % 2 == 1:
                    x += x_spacing * 0.5
                # Add small randomization
                x += random.uniform(-0.005, 0.005)
                y += random.uniform(-0.005, 0.005)
                candidate_points.append([x, y])

        # Strategy 2: Voronoi cell centroid analysis for better distribution
        if len(candidate_points) < self.n_circles:
            # Create a more dense initial grid for Voronoi analysis
            dense_points = []
            for i in range(8):
                for j in range(8):
                    x = (j + 1) * (self.rect_width / 9)
                    y = (i + 1) * (self.rect_height / 9)
                    dense_points.append([x, y])

            # Add some random points for diversity
            for _ in range(10):
                x = random.uniform(0.05, self.rect_width - 0.05)
                y = random.uniform(0.05, self.rect_height - 0.05)
                dense_points.append([x, y])

            # Generate Voronoi diagram and get centroids
            try:
                vor = Voronoi(dense_points)
                # Use Voronoi cell centroids as potential seed points
                for region in vor.regions:
                    if region and -1 not in region:  # Skip infinite regions
                        points_in_region = [vor.vertices[i] for i in region if
                                          0 <= vor.vertices[i][0] <= self.rect_width and
                                          0 <= vor.vertices[i][1] <= self.rect_height]
                        if points_in_region:
                            # Compute centroid
                            centroid_x = sum(p[0] for p in points_in_region) / len(points_in_region)
                            centroid_y = sum(p[1] for p in points_in_region) / len(points_in_region)
                            if 0.05 <= centroid_x <= self.rect_width - 0.05 and \
                               0.05 <= centroid_y <= self.rect_height - 0.05:
                                candidate_points.append([centroid_x, centroid_y])
            except:
                # Fallback to simple random points if Voronoi fails
                for _ in range(5):
                    x = random.uniform(0.05, self.rect_width - 0.05)
                    y = random.uniform(0.05, self.rect_height - 0.05)
                    candidate_points.append([x, y])

        # Strategy 3: Corner and edge seeding
        corner_seeds = [
            [0.1, 0.1],
            [self.rect_width - 0.1, 0.1],
            [0.1, self.rect_height - 0.1],
            [self.rect_width - 0.1, self.rect_height - 0.1]
        ]

        edge_seeds = [
            [self.rect_width/2, 0.1],
            [self.rect_width/2, self.rect_height - 0.1],
            [0.1, self.rect_height/2],
            [self.rect_width - 0.1, self.rect_height/2]
        ]

        # Add corner and edge points
        for point in corner_seeds:
            if len(candidate_points) < self.n_circles:
                candidate_points.append(point)
        for point in edge_seeds:
            if len(candidate_points) < self.n_circles:
                candidate_points.append(point)

        # Select final points based on distance criteria to ensure good distribution
        selected_points = []
        for point in candidate_points:
            if len(selected_points) >= self.n_circles:
                break
            # Check if this point is sufficiently far from existing selected points
            is_far_enough = True
            for selected_point in selected_points:
                distance = np.sqrt((point[0] - selected_point[0])**2 + (point[1] - selected_point[1])**2)
                if distance < 0.1:  # Minimum distance threshold
                    is_far_enough = False
                    break

            if is_far_enough:
                selected_points.append(point)

        # Fill remaining slots if needed
        while len(selected_points) < self.n_circles:
            x = random.uniform(0.05, self.rect_width - 0.05)
            y = random.uniform(0.05, self.rect_height - 0.05)
            selected_points.append([x, y])

        # Assign to circles array with initial radius
        for i in range(min(self.n_circles, len(selected_points))):
            self.circles[i] = [selected_points[i][0], selected_points[i][1], 0.015]

        # Fill remaining circles with random positions
        for i in range(len(selected_points), self.n_circles):
            x = random.uniform(0.05, self.rect_width - 0.05)
            y = random.uniform(0.05, self.rect_height - 0.05)
            self.circles[i] = [x, y, 0.015]

    def compute_max_radius_vectorized(self, circles: np.ndarray, index: int) -> float:
        """Vectorized computation of maximum radius for a circle"""
        x, y, _ = circles[index]

        # Minimum distance to boundaries
        min_dist_to_boundaries = min(x, self.rect_width - x, y, self.rect_height - y)

        # Check collisions with other circles using vectorized operations
        min_dist_to_others = float('inf')

        # Get indices of all circles except the target
        other_indices = np.arange(len(circles)) != index
        other_circles = circles[other_indices]

        if len(other_circles) > 0:
            # Vectorized distance calculation
            dx = x - other_circles[:, 0]
            dy = y - other_circles[:, 1]
            distances = np.sqrt(dx*dx + dy*dy)

            # Calculate radius constraints (distance - radius)
            radius_constraints = distances - other_circles[:, 2]

            # Find minimum positive constraint
            positive_constraints = radius_constraints[radius_constraints > 0]
            if len(positive_constraints) > 0:
                min_dist_to_others = np.min(positive_constraints)

        # Return the minimum of all constraints
        max_radius = min(min_dist_to_boundaries, min_dist_to_others)

        return max(0.001, max_radius)

    def validate_configuration(self, circles: np.ndarray) -> bool:
        """Validate that all circles are within bounds and non-overlapping"""
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Check boundary conditions
            if x - r < 0 or x + r > self.rect_width or y - r < 0 or y + r > self.rect_height:
                return False

            # Check overlap with other circles
            for j in range(i + 1, len(circles)):
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                # Overlap occurs when distance < sum of radii
                if distance < r + r2 - 1e-10:
                    return False

        return True

    def local_search_step(self, circles: np.ndarray, step_size: float = 0.02, iteration: int = 0) -> Tuple[np.ndarray, bool]:
        """Perform one local search step with improved radius computation"""
        improved = False
        new_circles = circles.copy()

        # Track radius improvements for convergence detection
        old_sum = np.sum(new_circles[:, 2])

        # Shuffle circle order to avoid bias
        circle_indices = list(range(self.n_circles))
        random.shuffle(circle_indices)

        for i in circle_indices:
            # Compute maximum possible radius at current location
            max_radius = self.compute_max_radius_vectorized(new_circles, i)

            # Try to increase radius if possible
            if max_radius > new_circles[i, 2]:
                old_radius = new_circles[i, 2]
                new_circles[i, 2] = max_radius
                improved = True

        new_sum = np.sum(new_circles[:, 2])
        return new_circles, improved

    def perturb_positions(self, circles: np.ndarray, step_size: float = 0.02) -> np.ndarray:
        """Apply random perturbations to positions to escape local minima"""
        new_circles = circles.copy()
        for _ in range(3):
            i = random.randint(0, self.n_circles - 1)
            # Apply perturbation to position
            new_circles[i, 0] += random.uniform(-step_size, step_size)
            new_circles[i, 1] += random.uniform(-step_size, step_size)

            # Clamp to rectangle bounds
            new_circles[i, 0] = np.clip(new_circles[i, 0], 0.01, self.rect_width - 0.01)
            new_circles[i, 1] = np.clip(new_circles[i, 1], 0.01, self.rect_height - 0.01)

            # Recompute max radius after perturbation
            max_radius = self.compute_max_radius_vectorized(new_circles, i)
            new_circles[i, 2] = max_radius

        return new_circles

    def adaptive_step_size(self, iteration: int, improvement_history: list, base_step_size: float = 0.02) -> float:
        """Adaptively adjust step size based on recent improvement trends"""
        if len(improvement_history) < 5:
            return base_step_size

        recent_improvements = improvement_history[-5:]
        avg_improvement = np.mean(recent_improvements)

        # If improvement is very small, reduce step size to allow fine-tuning
        if avg_improvement < 0.0001:
            return base_step_size * 0.5
        elif avg_improvement < 0.001:
            return base_step_size * 0.7
        else:
            return base_step_size

    def optimize(self) -> np.ndarray:
        """Main optimization loop with enhanced parameters"""
        # Initialize with a good starting point
        self.initialize_circles()

        best_circles = None
        best_radius_sum = 0

        # Track improvement history for adaptive step sizing
        improvement_history = []

        # Multi-start approach with more iterations (10 vs 5)
        for start_iter in range(10):
            # Reset circles for this iteration
            current_circles = self.circles.copy()

            # Enhanced multi-scale optimization with better parameters
            for phase in range(4):  # Increased from 3 to 4 phases
                # Improved phase parameters
                if phase == 0:
                    max_iterations = 150
                    base_step_size = 0.08
                    temp = 1.0
                    decay_factor = 0.95
                elif phase == 1:
                    max_iterations = 200
                    base_step_size = 0.04
                    temp = 0.6
                    decay_factor = 0.97
                elif phase == 2:
                    max_iterations = 250
                    base_step_size = 0.01
                    temp = 0.2
                    decay_factor = 0.99
                else:  # Final fine-tuning phase
                    max_iterations = 300
                    base_step_size = 0.005
                    temp = 0.05
                    decay_factor = 0.995

                for iteration in range(max_iterations):
                    # Adapt step size based on recent convergence
                    current_step_size = self.adaptive_step_size(iteration, improvement_history, base_step_size)

                    # Perform local search step
                    old_sum = np.sum(current_circles[:, 2])
                    current_circles, improved = self.local_search_step(current_circles, current_step_size, iteration)
                    new_sum = np.sum(current_circles[:, 2])

                    # Track improvement for adaptive step sizing
                    improvement = new_sum - old_sum
                    improvement_history.append(improvement)
                    if len(improvement_history) > 10:
                        improvement_history.pop(0)

                    # Simulated annealing-like temperature-based acceptance
                    if random.random() < math.exp(-iteration * 0.005):  # Cooling factor
                        # Occasionally accept worse solutions to escape local minima
                        pass

                    # Periodic perturbations to escape local minima
                    if iteration % 15 == 0 and iteration > 0:
                        current_circles = self.perturb_positions(current_circles, current_step_size)

                    # More aggressive early stopping
                    if not improved and iteration > 100:
                        break

            # Validate and score configuration
            if self.validate_configuration(current_circles):
                radius_sum = np.sum(current_circles[:, 2])
                if radius_sum > best_radius_sum:
                    best_radius_sum = radius_sum
                    best_circles = current_circles.copy()

        # Fallback if no valid configuration found
        if best_circles is None:
            current_circles = self.circles.copy()
            # More iterations for fallback
            for _ in range(400):
                current_circles, _ = self.local_search_step(current_circles)
            best_circles = current_circles

        # Final boundary correction with more aggressive radius clamping
        for i in range(self.n_circles):
            x, y, r = best_circles[i]
            # Ensure circles are within bounds and radius is reasonable
            r = min(r, x, self.rect_width - x, y, self.rect_height - y)
            if r <= 0.001:
                r = 0.015
            best_circles[i] = [x, y, r]

        return best_circles

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions - since perimeter = 4, width + height = 2
    optimizer = CirclePackingOptimizer()
    return optimizer.optimize()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")