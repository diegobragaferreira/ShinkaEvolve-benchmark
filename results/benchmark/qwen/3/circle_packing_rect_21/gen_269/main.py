# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
import random
import math
from typing import Tuple, List

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2, using 1.5 x 0.5 for good aspect ratio
    rect_width, rect_height = 1.5, 0.5

    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    def create_strategic_initialization(n_circles: int, width: float, height: float) -> np.ndarray:
        """Create initial configuration using strategic corner and edge placements"""
        circles = np.zeros((n_circles, 3))
        
        # Strategic corner and edge positions
        strategic_positions = [
            (width * 0.1, height * 0.1),    # bottom-left
            (width * 0.9, height * 0.1),    # bottom-right
            (width * 0.1, height * 0.9),    # top-left
            (width * 0.9, height * 0.9),    # top-right
            (width * 0.5, height * 0.1),    # bottom-middle
            (width * 0.5, height * 0.9),    # top-middle
            (width * 0.1, height * 0.5),    # left-middle
            (width * 0.9, height * 0.5),    # right-middle
            (width * 0.25, height * 0.25),   # quarter positions
            (width * 0.75, height * 0.25),
            (width * 0.25, height * 0.75),
            (width * 0.75, height * 0.75),
        ]
        
        # Place circles at strategic positions first
        for i in range(min(len(strategic_positions), n_circles)):
            x, y = strategic_positions[i]
            circles[i] = [x, y, 0.03]
        
        # Fill remaining positions with hexagonal grid pattern
        remaining = n_circles - len(strategic_positions)
        if remaining > 0:
            # Use hexagonal grid for remaining positions
            rows = int(math.ceil(math.sqrt(remaining)))
            cols = int(math.ceil(remaining / rows))
            
            cell_width = width / (cols + 1)
            cell_height = height / (rows + 1)
            
            idx = len(strategic_positions)
            for i in range(rows):
                for j in range(cols):
                    if idx >= n_circles:
                        break
                    x_offset = 0.0 if i % 2 == 0 else 0.5
                    x = (j + 1 + x_offset) * cell_width
                    y = (i + 1) * cell_height
                    # Ensure within bounds
                    x = max(0.01, min(width - 0.01, x))
                    y = max(0.01, min(height - 0.01, y))
                    circles[idx] = [x, y, 0.02]
                    idx += 1
                    if idx >= n_circles:
                        break
        
        return circles

    def create_voronoi_initialization(n_circles: int, width: float, height: float) -> np.ndarray:
        """Create initial configuration using Voronoi-based spatial distribution"""
        circles = np.zeros((n_circles, 3))

        # Generate initial points for Voronoi diagram
        # Start with corner and edge positions for good coverage
        initial_points = [
            (width * 0.1, height * 0.1),    # bottom-left
            (width * 0.9, height * 0.1),    # bottom-right
            (width * 0.1, height * 0.9),    # top-left
            (width * 0.9, height * 0.9),    # top-right
            (width * 0.5, height * 0.1),    # bottom-middle
            (width * 0.5, height * 0.9),    # top-middle
            (width * 0.1, height * 0.5),    # left-middle
            (width * 0.9, height * 0.5),    # right-middle
        ]

        # Add random points to ensure good coverage
        additional_points = []
        for _ in range(n_circles - len(initial_points)):
            x = random.uniform(0.05 * width, 0.95 * width)
            y = random.uniform(0.05 * height, 0.95 * height)
            additional_points.append([x, y])

        # Combine all points
        all_points = initial_points + additional_points[:n_circles - len(initial_points)]

        # Create Voronoi diagram
        try:
            vor = Voronoi(all_points)

            # Get Voronoi vertices that are inside our rectangle
            valid_vertices = []
            for vertex in vor.vertices:
                if (0 <= vertex[0] <= width) and (0 <= vertex[1] <= height):
                    valid_vertices.append(vertex)

            # Use Voronoi vertices as center points
            chosen_centers = valid_vertices[:min(n_circles, len(valid_vertices))]
            if len(chosen_centers) < n_circles:
                # Fill missing circles with random points
                for i in range(len(chosen_centers), n_circles):
                    x = random.uniform(0.05 * width, 0.95 * width)
                    y = random.uniform(0.05 * height, 0.95 * height)
                    chosen_centers.append([x, y])

            # Assign positions with initial radii based on Voronoi-based density
            for i, (x, y) in enumerate(chosen_centers[:n_circles]):
                # Base radius estimation based on Voronoi cell size
                circles[i] = [x, y, 0.02]

        except Exception:
            # Fallback to simple initialization if Voronoi fails
            for i in range(n_circles):
                x = random.uniform(0.05 * width, 0.95 * width)
                y = random.uniform(0.05 * height, 0.95 * height)
                circles[i] = [x, y, 0.02]

        return circles

    def compute_max_radius_at_position_vectorized(x: float, y: float, existing_circles: np.ndarray,
                                                rect_width: float, rect_height: float) -> float:
        """Vectorized computation of maximum possible radius for a circle at given position"""
        # Distance to boundaries
        min_bound = min(x, rect_width - x, y, rect_height - y)

        # Vectorized distance calculation to all existing circles
        if len(existing_circles) > 0:
            positions = existing_circles[:, :2]
            radii = existing_circles[:, 2]

            # Calculate distances to all existing circles
            dx = positions[:, 0] - x
            dy = positions[:, 1] - y
            distances = np.sqrt(dx*dx + dy*dy)

            # Avoid self-distance and compute min distance to other circles
            # Set self-distances to infinity to avoid them
            distances = np.where(distances == 0, float('inf'), distances)

            # Min distance minus sum of radii
            min_dist = np.min(distances)
            if len(distances) > 0:
                # Calculate min distance to other circles (not self)
                min_dist_to_others = np.min(distances - radii)
                actual_min_dist = min(min_dist, min_dist_to_others)
            else:
                actual_min_dist = min_dist
        else:
            actual_min_dist = float('inf')

        # Take minimum of boundary and other-circle distances
        max_radius = min(min_bound, actual_min_dist if actual_min_dist < float('inf') else float('inf'))
        return max(0.001, max_radius)

    def is_valid_configuration_vectorized(circles: np.ndarray, rect_width: float, rect_height: float) -> bool:
        """Vectorized validation of circle configuration"""
        # Check boundary constraints efficiently
        if np.any(circles[:, 0] - circles[:, 2] < 0) or \
           np.any(circles[:, 0] + circles[:, 2] > rect_width) or \
           np.any(circles[:, 1] - circles[:, 2] < 0) or \
           np.any(circles[:, 1] + circles[:, 2] > rect_height):
            return False

        # Check overlap constraints efficiently using vectorized computation
        if len(circles) < 2:
            return True

        # Use vectorized computation for overlap detection
        positions = circles[:, :2]
        radii = circles[:, 2]

        # Create distance matrix
        dist_matrix = cdist(positions, positions)

        # Set diagonal to infinity (self-distances)
        np.fill_diagonal(dist_matrix, float('inf'))

        # Minimum distances between circles
        min_distances = np.min(dist_matrix, axis=1)

        # Minimum sum of radii for each circle pair
        radii_sums = radii[:, np.newaxis] + radii[np.newaxis, :]

        # Check overlaps - vectorized operation
        overlap_mask = min_distances < np.min(radii_sums, axis=0)

        return not np.any(overlap_mask)

    def calculate_radius_sum(circles: np.ndarray) -> float:
        """Calculate sum of all radii"""
        return np.sum(circles[:, 2])

    def local_refinement_step(circles: np.ndarray, rect_width: float, rect_height: float,
                            iterations: int = 100, relax_overlap: bool = True) -> np.ndarray:
        """Perform local refinement to improve circle configuration with enhanced strategies"""
        current = circles.copy()
        current_sum = calculate_radius_sum(current)

        # Adaptive step sizes with dynamic adjustments
        initial_step_size = 0.05

        for iter_num in range(iterations):
            # Gradually tighten overlap constraints over time
            overlap_tolerance_factor = 1.0 if not relax_overlap else max(0.1, 1.0 - (iter_num / iterations) * 0.8)

            # Dynamic step size based on iteration count  
            step_size = initial_step_size * (1.0 - (iter_num / iterations) * 0.7)
            step_size = max(0.001, step_size)

            # Try to improve each circle
            for i in range(len(current)):
                original_x, original_y, original_r = current[i]

                # Try several moves with random directions
                best_x, best_y, best_r = original_x, original_y, original_r
                best_sum = current_sum

                # Enhanced perturbation strategy with multiple directions
                test_moves = [
                    # Standard Gaussian perturbations
                    (step_size * random.gauss(0, 1), step_size * random.gauss(0, 1)),
                    (step_size * random.gauss(0, 1), 0),
                    (0, step_size * random.gauss(0, 1)),
                    # Coordinate-specific perturbations
                    (step_size * random.uniform(-1, 1), 0),
                    (0, step_size * random.uniform(-1, 1)),
                    # Small random moves
                    (random.uniform(-step_size/2, step_size/2), random.uniform(-step_size/2, step_size/2)),
                    # No move (for baseline)
                    (0, 0)
                ]

                # Add gradient-guided moves for better convergence in later iterations
                if iter_num > iterations // 3 and len(current) > 1:
                    # Calculate influence from neighboring circles for gradient guidance
                    # Find closest circle to guide movement direction
                    positions = current[:, :2]
                    distances = np.sqrt(np.sum((positions - [original_x, original_y])**2, axis=1))
                    closest_idx = np.argmin(distances)

                    if closest_idx != i:  # Not self
                        # Move towards or away from nearest neighbor depending on relative radii
                        dx = positions[closest_idx, 0] - original_x
                        dy = positions[closest_idx, 1] - original_y
                        dist = max(0.001, np.sqrt(dx*dx + dy*dy))

                        # Normalize direction and scale by step_size
                        dx_norm = dx / dist
                        dy_norm = dy / dist

                        # Adjust move direction based on whether we're closer or further than neighbor
                        neighbor_r = current[closest_idx, 2]
                        if original_r < neighbor_r:
                            # Move away from neighbor to increase own radius
                            test_moves.append((-dx_norm * step_size * 0.5, -dy_norm * step_size * 0.5))
                        else:
                            # Move towards neighbor to possibly increase both radii
                            test_moves.append((dx_norm * step_size * 0.3, dy_norm * step_size * 0.3))

                # Add directional bias for strategic improvement in later iterations
                if iter_num > iterations // 3:  # Later iterations
                    # Add more systematic searches
                    test_moves.extend([
                        (step_size * random.choice([-1, 1]), 0),
                        (0, step_size * random.choice([-1, 1])),
                    ])

                for dx, dy in test_moves:
                    test_x = max(0.001, min(rect_width - 0.001, original_x + dx))
                    test_y = max(0.001, min(rect_height - 0.001, original_y + dy))

                    # Compute maximum possible radius at new position
                    temp_circles = current.copy()
                    temp_circles[i] = [test_x, test_y, 0.01]  # Temporarily small
                    max_r = compute_max_radius_at_position_vectorized(test_x, test_y, temp_circles, rect_width, rect_height)
                    test_r = min(max_r, max(0.001, original_r + random.uniform(-0.02, 0.02)))

                    # Apply final adjustment
                    temp_circles[i] = [test_x, test_y, test_r]

                    # Validate and calculate new sum
                    if relax_overlap and iter_num < iterations // 2:
                        # With relaxed overlap checking in early phase
                        valid = True
                        # Just check boundary constraints
                        if (test_x - test_r < 0 or test_x + test_r > rect_width or
                            test_y - test_r < 0 or test_y + test_r > rect_height):
                            valid = False

                        if valid:
                            new_sum = calculate_radius_sum(temp_circles)
                            if new_sum > best_sum:
                                best_sum = new_sum
                                best_x, best_y, best_r = test_x, test_y, test_r
                    else:
                        # Strict validation in later phases
                        if is_valid_configuration_vectorized(temp_circles, rect_width, rect_height):
                            new_sum = calculate_radius_sum(temp_circles)
                            if new_sum > best_sum:
                                best_sum = new_sum
                                best_x, best_y, best_r = test_x, test_y, test_r

                # Update if improvement found
                if best_sum > current_sum:
                    current[i] = [best_x, best_y, best_r]
                    current_sum = best_sum

        return current

    def multi_scale_optimization() -> np.ndarray:
        """Run multi-scale optimization to handle different levels of detail"""
        best_circles = None
        best_sum = -float('inf')

        # Scale 1: Coarse optimization with large steps - use strategic initialization
        print("Starting coarse optimization...")
        coarse_circles = create_strategic_initialization(21, rect_width, rect_height)
        coarse_circles = local_refinement_step(coarse_circles, rect_width, rect_height, 20, relax_overlap=True)
        coarse_sum = calculate_radius_sum(coarse_circles)
        print(f"Coarse optimization sum: {coarse_sum}")

        # Scale 2: Medium optimization with medium steps - use Voronoi initialization
        print("Starting medium optimization...")
        medium_circles = create_voronoi_initialization(21, rect_width, rect_height)
        medium_circles = local_refinement_step(medium_circles, rect_width, rect_height, 50, relax_overlap=True)
        medium_sum = calculate_radius_sum(medium_circles)
        print(f"Medium optimization sum: {medium_sum}")

        # Scale 3: Fine optimization with small steps
        print("Starting fine optimization...")
        fine_circles = medium_circles.copy()
        fine_circles = local_refinement_step(fine_circles, rect_width, rect_height, 100, relax_overlap=False)
        fine_sum = calculate_radius_sum(fine_circles)
        print(f"Fine optimization sum: {fine_sum}")

        # Return best of all scales
        if fine_sum > best_sum:
            best_sum = fine_sum
            best_circles = fine_circles.copy()

        # Also try with different initializations
        print("Trying alternative initialization...")
        alt_circles = create_voronoi_initialization(21, rect_width, rect_height)
        alt_circles = local_refinement_step(alt_circles, rect_width, rect_height, 30, relax_overlap=True)
        alt_circles = local_refinement_step(alt_circles, rect_width, rect_height, 70, relax_overlap=False)
        alt_sum = calculate_radius_sum(alt_circles)
        print(f"Alternative initialization sum: {alt_sum}")

        if alt_sum > best_sum:
            best_sum = alt_sum
            best_circles = alt_circles.copy()

        return best_circles

    # Main optimization workflow
    # Multi-scale optimization for better convergence
    final_circles = multi_scale_optimization()

    # Final validation and cleanup
    if final_circles is not None:
        # Double-check validity and ensure all constraints
        while True:
            valid = True

            # Check boundaries and overlaps using vectorized method
            if not is_valid_configuration_vectorized(final_circles, rect_width, rect_height):
                valid = False

            if not valid:
                # Reinitialize if invalid
                final_circles = create_strategic_initialization(21, rect_width, rect_height)
                final_circles = local_refinement_step(final_circles, rect_width, rect_height, 50)
                continue

            # Final validation check
            if is_valid_configuration_vectorized(final_circles, rect_width, rect_height):
                break
            else:
                # Retry with different initialization
                final_circles = create_voronoi_initialization(21, rect_width, rect_height)
                final_circles = local_refinement_step(final_circles, rect_width, rect_height, 50)

    # Final optimization pass to fine-tune
    if final_circles is not None:
        final_circles = local_refinement_step(final_circles, rect_width, rect_height, 30, relax_overlap=False)

    return final_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")