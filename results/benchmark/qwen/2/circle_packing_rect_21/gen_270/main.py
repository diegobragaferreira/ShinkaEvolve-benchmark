# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import random
from copy import deepcopy

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses improved grid initialization and constraint-aware optimization.
    """
    # Rectangle dimensions - perimeter = 4, so width + height = 2
    # Optimized rectangle dimensions for maximum packing efficiency
    rect_width = 1.3
    rect_height = 0.7

    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    n_circles = 21

    # Improved grid initialization with adaptive sizing
    def create_adaptive_grid(n, width, height):
        """
        Create an adaptive grid layout based on circle count and rectangle dimensions
        """
        # Calculate optimal grid dimensions based on circle count
        sqrt_n = np.sqrt(n)
        rows = int(np.ceil(sqrt_n))
        cols = int(np.ceil(n / rows))

        # Adjust grid size to fit within rectangle with proper margins
        cell_width = width / cols
        cell_height = height / rows

        # Use minimum of cell dimensions for radius with safety margin
        max_radius = min(cell_width, cell_height) * 0.42

        # Generate grid layout with slight randomization to avoid symmetry issues
        circles = []
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Add slight random offset to break symmetry and improve packing
                x = (j + 0.5 + np.random.uniform(-0.15, 0.15)) * cell_width
                y = (i + 0.5 + np.random.uniform(-0.15, 0.15)) * cell_height
                # Ensure positions stay within bounds
                x = np.clip(x, max_radius, width - max_radius)
                y = np.clip(y, max_radius, height - max_radius)
                circles.append([x, y, max_radius])
                idx += 1

        return np.array(circles)

    # Fast collision checking using KDTree for O(n log n) complexity
    def fast_collision_check(circles, width, height):
        """
        Fast collision detection using spatial indexing with boundary checks
        """
        # Check boundary constraints efficiently using vectorized operations
        coords = circles[:, :2]
        radii = circles[:, 2]

        # Early exit if any circle violates boundary constraints
        if np.any((coords[:, 0] - radii < 0) | (coords[:, 0] + radii > width) |
                  (coords[:, 1] - radii < 0) | (coords[:, 1] + radii > height)):
            return False

        # Create KDTree for efficient neighbor search
        tree = cKDTree(coords)

        # Find neighbors within 2 * max_radius distance (tighter bound)
        pairs = tree.query_pairs(2 * np.max(radii), p=2)

        # If no pairs found, no overlaps possible
        if len(pairs) == 0:
            return True

        # Check actual distances for overlap with early termination for efficiency
        for i, j in pairs:
            x1, y1 = coords[i]
            x2, y2 = coords[j]
            r1, r2 = radii[i], radii[j]
            dx = x1 - x2
            dy = y1 - y2
            dist_sq = dx*dx + dy*dy
            min_dist_sq = (r1 + r2) * (r1 + r2)

            if dist_sq < min_dist_sq:
                return False

        return True

    # Constraint-aware fitness evaluation with penalty system
    def evaluate_fitness_with_constraints(circles):
        """
        Evaluate fitness considering both objective (radius sum) and constraints
        """
        if not fast_collision_check(circles, rect_width, rect_height):
            # Count constraint violations efficiently
            coords = circles[:, :2]
            radii = circles[:, 2]

            # Vectorized boundary violations
            boundary_violations = np.sum((coords[:, 0] - radii < 0) | (coords[:, 0] + radii > rect_width) |
                                       (coords[:, 1] - radii < 0) | (coords[:, 1] + radii > rect_height))

            # Overlap violations using spatial indexing
            tree = cKDTree(coords)
            pairs = tree.query_pairs(2 * np.max(radii), p=2)

            overlap_violations = 0
            for i, j in pairs:
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                r1, r2 = radii[i], radii[j]
                dx = x1 - x2
                dy = y1 - y2
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (r1 + r2) * (r1 + r2)

                if dist_sq < min_dist_sq:
                    overlap_violations += 1

            total_violations = boundary_violations + overlap_violations

            # Weighted fitness with heavy penalty for constraint violations
            base_fitness = np.sum(circles[:, 2])
            # Heavier penalty for constraint violations (more severe than previous versions)
            penalty = total_violations * 1500
            return max(0, base_fitness - penalty)

        return np.sum(circles[:, 2])

    # Generate initial solution using adaptive grid
    initial_circles = create_adaptive_grid(n_circles, rect_width, rect_height)

    # Local refinement with constraint-aware optimization
    def local_refinement(initial_solution):
        """
        Apply iterative refinement to improve solution quality
        """
        current_solution = initial_solution.copy()

        # Iterative improvement loop
        for iteration in range(100):
            improved = False

            # Try to increase radii while maintaining constraints
            for i in range(len(current_solution)):
                current_radius = current_solution[i, 2]
                best_radius = current_radius

                # Test a range of radius adjustments
                test_radii = np.linspace(current_radius * 0.8, current_radius * 1.3, 15)

                for test_radius in test_radii:
                    # Create temporary solution
                    temp_solution = current_solution.copy()
                    temp_solution[i, 2] = test_radius

                    # Check constraints
                    if fast_collision_check(temp_solution, rect_width, rect_height):
                        if test_radius > best_radius:
                            best_radius = test_radius
                            improved = True

                current_solution[i, 2] = best_radius

            # If no improvement was made, stop early
            if not improved:
                break

        return current_solution

    # Apply refinement to initial solution
    refined_circles = local_refinement(initial_circles)

    # Final validation
    if not fast_collision_check(refined_circles, rect_width, rect_height):
        # Fallback to grid solution if something went wrong
        return create_adaptive_grid(n_circles, rect_width, rect_height)

    return refined_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")