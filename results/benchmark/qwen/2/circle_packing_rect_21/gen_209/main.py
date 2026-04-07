# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import math
import random
import time
from typing import Tuple

class CirclePackingOptimizer:
    def __init__(self, n_circles: int = 21, container_width: float = 1.3, container_height: float = 0.7):
        self.n_circles = n_circles
        self.container_width = container_width
        self.container_height = container_height
        self.spatial_tree = None

    def initialize_grid_placement(self) -> np.ndarray:
        """Initialize circles using adaptive grid-based approach with random perturbations"""
        circles = np.zeros((self.n_circles, 3))

        # Adaptive grid sizing based on circle count and container dimensions
        # Use more sophisticated grid calculation
        aspect_ratio = self.container_width / self.container_height

        # Determine grid dimensions with better aspect ratio handling
        if aspect_ratio > 1.2:
            # Wide container - favor more columns
            cols = max(1, int(math.ceil(math.sqrt(self.n_circles * aspect_ratio * 1.1))))
            rows = max(1, int(math.ceil(self.n_circles / cols)))
        elif aspect_ratio < 0.8:
            # Tall container - favor more rows
            rows = max(1, int(math.ceil(math.sqrt(self.n_circles / aspect_ratio * 1.1))))
            cols = max(1, int(math.ceil(self.n_circles / rows)))
        else:
            # Balanced container
            cols = max(1, int(math.ceil(math.sqrt(self.n_circles * aspect_ratio))))
            rows = max(1, int(math.ceil(self.n_circles / cols)))

        # Ensure sufficient grid cells
        while cols * rows < self.n_circles:
            if aspect_ratio >= 1.2:
                cols += 1
            elif aspect_ratio <= 0.8:
                rows += 1
            else:
                cols += 1

        # Place circles in grid with random perturbations
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= self.n_circles:
                    break
                # Position with offset for hexagonal packing-like effect
                x = (j + 0.5) * self.container_width / cols
                y = (i + 0.5) * self.container_height / rows

                # Add random perturbation with adaptive magnitude
                perturbation = min(0.02, 0.1 * min(self.container_width / cols, self.container_height / rows))
                x += random.uniform(-perturbation, perturbation)
                y += random.uniform(-perturbation, perturbation)

                # Ensure within bounds
                x = max(0.01, min(self.container_width - 0.01, x))
                y = max(0.01, min(self.container_height - 0.01, y))

                # Initial radius with adaptive scaling
                base_radius = min(0.1, self.container_width / (cols * 2.5), self.container_height / (rows * 2.5))
                r = base_radius * random.uniform(0.8, 1.0)
                circles[idx] = [x, y, r]
                idx += 1

        # Fill remaining positions if needed
        if idx < self.n_circles:
            for i in range(idx, self.n_circles):
                x = random.uniform(0.01, self.container_width - 0.01)
                y = random.uniform(0.01, self.container_height - 0.01)
                r = random.uniform(0.01, min(0.1, self.container_width/8, self.container_height/8))
                circles[i] = [x, y, r]

        return circles

    def update_spatial_tree(self, circles: np.ndarray):
        """Update the spatial tree for efficient neighbor queries"""
        self.spatial_tree = cKDTree(circles[:, :2])

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def is_valid_position(self, x, y, r, existing_circles: np.ndarray) -> bool:
        """Check if a circle position is valid (boundary and overlap constraints)"""
        # Check boundary constraints
        if x - r < 0 or x + r > self.container_width or y - r < 0 or y + r > self.container_height:
            return False

        # Use spatial tree for efficient neighbor checking
        try:
            if self.spatial_tree is not None:
                nearby_indices = self.spatial_tree.query_ball_point([x, y], r * 2)
            else:
                nearby_indices = range(len(existing_circles))

            # Check overlap with nearby circles
            for i in nearby_indices:
                existing_x, existing_y, existing_r = existing_circles[i]
                if i != len(existing_circles):  # Skip the circle being placed
                    distance = self.calculate_distance([x, y], [existing_x, existing_y])
                    if distance < (r + existing_r):
                        return False
        except Exception:
            # Fallback to direct checking if spatial tree fails
            for i in range(len(existing_circles)):
                existing_x, existing_y, existing_r = existing_circles[i]
                distance = self.calculate_distance([x, y], [existing_x, existing_y])
                if distance < (r + existing_r):
                    return False
        return True

    def calculate_fitness_with_penalties(self, circles: np.ndarray) -> Tuple[float, int]:
        """
        Calculate fitness with constraint awareness to guide optimization properly

        Returns:
            Tuple of (fitness_score, number_of_violations)
        """
        total_radius = np.sum(circles[:, 2])
        total_penalty = 0.0
        violations = 0

        # Check boundary violations with proper penalty calculation
        for i in range(self.n_circles):
            x, y, r = circles[i]
            # Check if circle is within bounds
            if x - r < 0 or x + r > self.container_width or y - r < 0 or y + r > self.container_height:
                # Calculate how far outside boundaries we are
                boundary_violation = 0
                if x < r:
                    boundary_violation += r - x
                elif x > self.container_width - r:
                    boundary_violation += x - (self.container_width - r)
                if y < r:
                    boundary_violation += r - y
                elif y > self.container_height - r:
                    boundary_violation += y - (self.container_height - r)

                # Apply penalty based on severity of violation
                total_penalty += 10000 * boundary_violation**2
                violations += 1

        # Check overlap violations with penalty based on overlap magnitude
        # Use spatial tree for efficient overlap detection
        try:
            points = circles[:, :2]
            tree = cKDTree(points)
            # Use a safe search radius for overlap checking
            max_radius = np.max(circles[:, 2])
            pairs = tree.query_pairs(2*max_radius, output_type='ndarray')

            for i, j in pairs:
                if i < j:  # Avoid double counting
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    if distance < (r1 + r2):
                        overlap = (r1 + r2) - distance
                        # Penalize based on overlap magnitude - quadratic penalty for better convergence
                        total_penalty += 50000 * overlap**2
                        violations += 1

        except Exception:
            # Fallback to direct checking if spatial tree fails
            for i in range(self.n_circles):
                for j in range(i+1, self.n_circles):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    if distance < (r1 + r2):
                        overlap = (r1 + r2) - distance
                        total_penalty += 50000 * overlap**2
                        violations += 1

        # Return fitness: sum of radii minus penalties
        fitness = total_radius - total_penalty
        return fitness, violations

    def optimize_single_circle(self, circles: np.ndarray, idx: int) -> Tuple[bool, np.ndarray]:
        """Optimize a single circle by trying various positions and radii"""
        current_x, current_y, current_r = circles[idx]
        best_radius = current_r
        best_position = [current_x, current_y]
        best_valid = False

        # Try several radius values with adaptive spacing
        max_radius = min(0.25, self.container_width/3, self.container_height/3)

        # Use exponential spacing for more efficient search
        if current_r < max_radius:
            # Use a more refined approach for radius search
            radius_steps = []
            # Start from current radius and go up to max_radius
            r_current = current_r
            while r_current <= max_radius:
                radius_steps.append(r_current)
                r_current *= 1.2  # Exponential growth for efficient search
            if not radius_steps or radius_steps[-1] < max_radius:
                radius_steps.append(max_radius)
        else:
            radius_steps = [current_r, max_radius]

        # Try several positions near current location with dense grid
        position_grid = [(dx, dy) for dx in [-0.1, -0.05, 0, 0.05, 0.1]
                        for dy in [-0.1, -0.05, 0, 0.05, 0.1]]

        # Also try some positions in the opposite direction for better exploration
        position_grid.extend([(dx, dy) for dx in [-0.08, -0.04, 0.04, 0.08]
                             for dy in [-0.08, -0.04, 0.04, 0.08]])

        for r_trial in radius_steps:
            # Skip if radius is too small or too large
            if r_trial < 0.001 or r_trial > max_radius:
                continue

            for dx, dy in position_grid:
                trial_x = current_x + dx
                trial_y = current_y + dy

                # Ensure within bounds
                trial_x = max(r_trial, min(self.container_width - r_trial, trial_x))
                trial_y = max(r_trial, min(self.container_height - r_trial, trial_y))

                # Test if this placement works
                test_circles = circles.copy()
                test_circles[idx] = [trial_x, trial_y, r_trial]

                # Check validity using spatial tree for efficiency
                if self.is_valid_position(trial_x, trial_y, r_trial, test_circles):
                    if r_trial > best_radius:
                        best_radius = r_trial
                        best_position = [trial_x, trial_y]
                        best_valid = True

        if best_valid:
            circles[idx] = [best_position[0], best_position[1], best_radius]

        return best_valid, circles

    def refine_configuration(self, circles: np.ndarray) -> np.ndarray:
        """Apply final refinement to resolve any remaining issues"""
        # Apply boundary corrections
        for i in range(self.n_circles):
            x, y, r = circles[i]
            # Correct boundary violations
            if x - r < 0:
                circles[i, 0] = r
            elif x + r > self.container_width:
                circles[i, 0] = self.container_width - r

            if y - r < 0:
                circles[i, 1] = r
            elif y + r > self.container_height:
                circles[i, 1] = self.container_height - r

        # Resolve overlaps through iterative adjustment
        for _ in range(200):  # Limited iterations to prevent infinite loop
            improved = False
            for i in range(self.n_circles):
                x, y, r = circles[i]
                # Try to reduce radius if overlapping
                for j in range(self.n_circles):
                    if i != j:
                        x2, y2, r2 = circles[j]
                        distance = self.calculate_distance([x, y], [x2, y2])
                        if distance < (r + r2):
                            new_r = max(1e-6, (distance - 0.001) / 2)
                            if new_r < r:
                                circles[i, 2] = new_r
                                improved = True
                                break

            if not improved:
                break

        return circles

    def optimize(self, max_iterations: int = 2000) -> np.ndarray:
        """Main optimization routine with hybrid approach"""
        # Initialize
        circles = self.initialize_grid_placement()
        self.update_spatial_tree(circles)

        # Track fitness instead of raw sum for better optimization guidance
        best_fitness, _ = self.calculate_fitness_with_penalties(circles)
        best_circles = circles.copy()

        # For tracking improvement
        last_improvement_iteration = 0
        patience = 100  # How many iterations to wait before early stopping

        # Phase 1: Global exploration with larger steps
        for iteration in range(max_iterations // 2):
            # Select multiple circles to optimize in batches
            batch_size = max(1, self.n_circles // 8)
            selected_indices = random.sample(range(self.n_circles), min(batch_size, self.n_circles))

            for idx in selected_indices:
                improved, circles = self.optimize_single_circle(circles, idx)

                # Update spatial tree after modification
                if improved:
                    self.update_spatial_tree(circles)

            # Periodic evaluation of the solution
            if iteration % 30 == 0:
                current_fitness, _ = self.calculate_fitness_with_penalties(circles)
                if current_fitness > best_fitness:
                    best_fitness = current_fitness
                    best_circles = circles.copy()
                    last_improvement_iteration = iteration

        # Phase 2: Fine-grained refinement with smaller steps
        for iteration in range(max_iterations // 2):
            # Select circles more systematically for fine-tuning
            selected_indices = random.sample(range(self.n_circles), min(5, self.n_circles))

            for idx in selected_indices:
                improved, circles = self.optimize_single_circle(circles, idx)

                # Update spatial tree after modification
                if improved:
                    self.update_spatial_tree(circles)

            # Periodic evaluation with higher frequency
            if iteration % 10 == 0:
                current_fitness, _ = self.calculate_fitness_with_penalties(circles)
                if current_fitness > best_fitness:
                    best_fitness = current_fitness
                    best_circles = circles.copy()
                    last_improvement_iteration = iteration + max_iterations // 2

            # Early stopping if no significant improvement
            if iteration - last_improvement_iteration > patience:
                break

        # Final refinement with improved overlap resolution
        final_circles = self.refine_configuration(best_circles.copy())

        # One final fitness check
        final_fitness, _ = self.calculate_fitness_with_penalties(final_circles)
        if final_fitness > best_fitness:
            return final_circles
        else:
            return best_circles

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set up container dimensions (perimeter = 4, so width + height = 2)
    # Using optimized dimensions from successful implementations
    optimizer = CirclePackingOptimizer(n_circles=21, container_width=1.3, container_height=0.7)
    return optimizer.optimize()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")