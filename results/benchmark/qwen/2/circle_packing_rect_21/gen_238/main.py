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
        return True

    def optimize_single_circle(self, circles: np.ndarray, idx: int) -> Tuple[bool, np.ndarray]:
        """Optimize a single circle using aggressive local search with adaptive strategies"""
        current_x, current_y, current_r = circles[idx]
        best_radius = current_r
        best_position = [current_x, current_y]
        best_valid = False

        # Start with more aggressive local search
        # Try multiple radius values with varying strategies
        max_radius = min(0.25, self.container_width/3, self.container_height/3)

        # Use a mix of linear and logarithmic spacing
        radius_steps = []
        # Dense sampling near current radius
        radius_steps.extend(np.linspace(current_r, max_radius, 6)[1:])
        # Logarithmic spacing for larger radii
        log_space = np.logspace(np.log10(current_r), np.log10(max_radius), 6, base=10)
        radius_steps.extend(log_space[1:])
        radius_steps = sorted(set(radius_steps))  # Remove duplicates

        # Expanded position search with different step sizes
        position_grids = [
            [(dx, dy) for dx in [-0.12, -0.06, 0, 0.06, 0.12] for dy in [-0.12, -0.06, 0, 0.06, 0.12]],
            [(dx, dy) for dx in [-0.08, -0.04, 0, 0.04, 0.08] for dy in [-0.08, -0.04, 0, 0.04, 0.08]],
            [(dx, dy) for dx in [-0.04, 0, 0.04] for dy in [-0.04, 0, 0.04]]
        ]

        # Try each grid with decreasing priority
        for grid in position_grids:
            for r_trial in radius_steps:
                for dx, dy in grid:
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

    def aggressive_local_optimization(self, circles: np.ndarray, max_iterations: int = 100) -> np.ndarray:
        """Apply intensive local optimization to further improve solution"""
        current_circles = circles.copy()
        self.update_spatial_tree(current_circles)

        for iteration in range(max_iterations):
            improved = False

            # Try to improve each circle aggressively
            for i in range(self.n_circles):
                # Get current circle
                x, y, r = current_circles[i]

                # Try to increase radius as much as possible while staying valid
                # Calculate maximum possible radius
                max_possible_radius = min(
                    x, self.container_width - x,
                    y, self.container_height - y
                )

                # Try to expand radius with overlap constraints
                best_radius = r
                best_x, best_y = x, y

                # Check if we can safely increase radius and move
                max_radius_search = min(max_possible_radius, 0.25)
                if max_radius_search > r:
                    # Try several radius values
                    for trial_radius in np.linspace(r, max_radius_search, 10):
                        # Try positions around current location
                        for dx in np.linspace(-0.1, 0.1, 5):
                            for dy in np.linspace(-0.1, 0.1, 5):
                                trial_x = x + dx
                                trial_y = y + dy

                                # Ensure within bounds
                                trial_x = max(trial_radius, min(self.container_width - trial_radius, trial_x))
                                trial_y = max(trial_radius, min(self.container_height - trial_radius, trial_y))

                                # Test validity
                                test_circles = current_circles.copy()
                                test_circles[i] = [trial_x, trial_y, trial_radius]

                                if self.is_valid_position(trial_x, trial_y, trial_radius, test_circles):
                                    if trial_radius > best_radius:
                                        best_radius = trial_radius
                                        best_x, best_y = trial_x, trial_y
                                        improved = True

                        # Optimization: if we found a good improvement, stop early
                        if improved:
                            break

                if improved:
                    current_circles[i] = [best_x, best_y, best_radius]
                    self.update_spatial_tree(current_circles)

            # If no improvement, stop early
            if not improved:
                break

        return current_circles

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

        # Resolve overlaps through iterative adjustment with probabilistic resolution
        for iteration in range(200):  # Limited iterations to prevent infinite loop
            improved = False
            # Track which circles were modified in this iteration
            modified_circles = set()

            for i in range(self.n_circles):
                if i in modified_circles:
                    continue

                x, y, r = circles[i]
                # Try to reduce radius if overlapping
                for j in range(self.n_circles):
                    if i != j and j not in modified_circles:
                        x2, y2, r2 = circles[j]
                        distance = self.calculate_distance([x, y], [x2, y2])
                        if distance < (r + r2):
                            # Calculate overlap severity
                            overlap = (r + r2) - distance

                            # Probabilistic resolution based on overlap severity
                            # More severe overlaps are more likely to be resolved
                            resolution_prob = min(1.0, overlap * 10.0)

                            if random.random() < resolution_prob:
                                # Reduce radius to resolve overlap
                                new_r = max(1e-6, (distance - 0.001) / 2)
                                if new_r < r:
                                    circles[i, 2] = new_r
                                    modified_circles.add(i)
                                    improved = True
                                    break

            # Break if no improvement made in this iteration
            if not improved:
                break

        return circles

    def calculate_fitness_with_penalties(self, circles: np.ndarray, generation: int = 0, max_generations: int = 1000) -> Tuple[float, int]:
        """
        Calculate fitness with adaptive constraint awareness to guide optimization properly

        Returns:
            Tuple of (fitness_score, number_of_violations)
        """
        total_radius = np.sum(circles[:, 2])
        total_penalty = 0.0
        violations = 0

        # Adaptive penalty coefficients that change based on optimization stage
        # Early stage: more tolerant to violations to encourage exploration
        # Later stage: stricter penalties to enforce feasibility
        exploration_factor = max(0.1, 1.0 - (generation / max_generations))
        penalty_multiplier = 1.0 + (0.5 * (1.0 - exploration_factor))

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

                # Apply adaptive penalty based on violation severity and stage
                base_penalty = 5000 * boundary_violation**2
                adaptive_penalty = base_penalty * penalty_multiplier * exploration_factor
                total_penalty += adaptive_penalty
                violations += 1

        # Check overlap violations with adaptive penalty based on overlap magnitude
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
                        # Adaptive penalty based on overlap magnitude and optimization stage
                        base_penalty = 20000 * overlap**2
                        adaptive_penalty = base_penalty * penalty_multiplier * exploration_factor
                        total_penalty += adaptive_penalty
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
                        base_penalty = 20000 * overlap**2
                        adaptive_penalty = base_penalty * penalty_multiplier * exploration_factor
                        total_penalty += adaptive_penalty
                        violations += 1

        # Return fitness: sum of radii minus penalties
        fitness = total_radius - total_penalty
        return fitness, violations

    def optimize(self, max_iterations: int = 2000) -> np.ndarray:
        """Main optimization routine with hybrid approach"""
        # Initialize
        circles = self.initialize_grid_placement()
        self.update_spatial_tree(circles)

        best_sum = np.sum(circles[:, 2])
        best_circles = circles.copy()

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
            if iteration % 50 == 0:
                current_sum = np.sum(circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = circles.copy()

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
            if iteration % 20 == 0:
                current_sum = np.sum(circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = circles.copy()

        # Phase 3: Aggressive local optimization
        final_circles = self.aggressive_local_optimization(best_circles.copy())

        # Final refinement
        final_circles = self.refine_configuration(final_circles.copy())

        return final_circles

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