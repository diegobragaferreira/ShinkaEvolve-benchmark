# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
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

    def create_hybrid_initialization(n_circles: int, width: float, height: float) -> np.ndarray:
        """Create initial configuration using hybrid Voronoi and strategic placement"""
        circles = np.zeros((n_circles, 3))

        # Strategic corner and edge positions for good spatial distribution
        strategic_points = [
            (width * 0.1, height * 0.1),    # bottom-left
            (width * 0.9, height * 0.1),    # bottom-right
            (width * 0.1, height * 0.9),    # top-left
            (width * 0.9, height * 0.9),    # top-right
            (width * 0.5, height * 0.1),    # bottom-middle
            (width * 0.5, height * 0.9),    # top-middle
            (width * 0.1, height * 0.5),    # left-middle
            (width * 0.9, height * 0.5),    # right-middle
            (width * 0.5, height * 0.5),    # center
        ]

        # Fill with strategic points first
        points_used = 0
        for x, y in strategic_points[:min(n_circles, len(strategic_points))]:
            circles[points_used] = [x, y, 0.02]
            points_used += 1

        # Fill remaining with random points in the interior
        for i in range(points_used, n_circles):
            x = random.uniform(0.05 * width, 0.95 * width)
            y = random.uniform(0.05 * height, 0.95 * height)
            circles[i] = [x, y, 0.02]

        return circles

    def compute_max_radius_at_position(x: float, y: float, existing_circles: np.ndarray,
                                    rect_width: float, rect_height: float) -> float:
        """Compute maximum possible radius for a circle at given position"""
        # Distance to boundaries
        min_bound = min(x, rect_width - x, y, rect_height - y)
        
        # Distance to other circles
        min_dist_to_others = float('inf')
        for cx, cy, r in existing_circles:
            if cx != x or cy != y:  # Skip self
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                min_dist_to_others = min(min_dist_to_others, dist - r)
        
        # Take minimum of boundary and other-circle distances
        max_radius = min(min_bound, min_dist_to_others if min_dist_to_others != float('inf') else float('inf'))
        return max(0.001, max_radius)

    def compute_max_radius_vectorized(positions: np.ndarray, radii: np.ndarray, 
                                    x: float, y: float, rect_width: float, rect_height: float) -> float:
        """Vectorized computation of maximum radius - faster for batch operations"""
        # Boundary constraints
        min_bound = min(x, rect_width - x, y, rect_height - y)
        
        # Avoid division by zero
        if len(positions) == 0:
            return max(0.001, min_bound)
        
        # Vectorized distance computation
        dx = positions[:, 0] - x
        dy = positions[:, 1] - y
        distances = np.sqrt(dx*dx + dy*dy)
        
        # Min distance to other circles
        min_dist_to_others = np.min(distances - radii) if len(distances) > 0 else float('inf')
        
        # Take minimum
        max_radius = min(min_bound, min_dist_to_others if min_dist_to_others != float('inf') else float('inf'))
        return max(0.001, max_radius)

    def is_valid_configuration(circles: np.ndarray, rect_width: float, rect_height: float) -> bool:
        """Check if configuration is valid (no overlaps, within bounds)"""
        # Check boundary constraints
        for x, y, r in circles:
            if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
                return False
        
        # Check overlaps
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dx = x1 - x2
                dy = y1 - y2
                distance = math.sqrt(dx*dx + dy*dy)
                if distance < (r1 + r2):
                    return False
        return True

    def is_valid_configuration_vectorized(circles: np.ndarray, rect_width: float, rect_height: float) -> bool:
        """Vectorized validation for higher performance"""
        # Check boundary constraints efficiently
        if np.any(circles[:, 0] - circles[:, 2] < 0) or \
           np.any(circles[:, 0] + circles[:, 2] > rect_width) or \
           np.any(circles[:, 1] - circles[:, 2] < 0) or \
           np.any(circles[:, 1] + circles[:, 2] > rect_height):
            return False

        # Check overlap constraints using vectorized operation
        if len(circles) < 2:
            return True

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
        """Perform local refinement with adaptive parameters"""
        current = circles.copy()
        current_sum = calculate_radius_sum(current)

        # Adaptive step sizes with dynamic adjustments
        initial_step_size = 0.05

        for iter_num in range(iterations):
            # Gradually tighten overlap constraints over time
            overlap_tolerance_factor = 1.0 if not relax_overlap else max(0.1, 1.0 - (iter_num / iterations) * 0.8)

            # Dynamic step size based on iteration count and convergence rate
            if iter_num < iterations * 0.3:
                step_size = initial_step_size  # Large steps for initial exploration
            elif iter_num < iterations * 0.7:
                step_size = initial_step_size * 0.5  # Medium steps for refinement
            else:
                step_size = initial_step_size * 0.2  # Small steps for fine-tuning
                
            # Further reduce step size if convergence is slow
            if iter_num > 50 and iter_num % 10 == 0:
                step_size *= 0.8

            # Try to improve each circle
            for i in range(len(current)):
                original_x, original_y, original_r = current[i]

                # Try several moves with enhanced strategies
                best_x, best_y, best_r = original_x, original_y, original_r
                best_sum = current_sum

                # Enhanced perturbation strategy with multiple approaches
                test_moves = []
                
                # Generate moves with different strategies
                if iter_num < iterations * 0.5:  # Early iterations - broad exploration
                    # Include more diverse moves
                    test_moves.extend([
                        (step_size * random.gauss(0, 1), step_size * random.gauss(0, 1)),  # Gaussian
                        (step_size * random.uniform(-1, 1), 0),  # X-axis only
                        (0, step_size * random.uniform(-1, 1)),  # Y-axis only
                        (step_size * random.choice([-1, 1]), 0),  # Large X-step
                        (0, step_size * random.choice([-1, 1])),  # Large Y-step
                    ])
                else:  # Later iterations - focused exploitation
                    # More precise moves
                    test_moves.extend([
                        (step_size * random.uniform(-0.5, 0.5), step_size * random.uniform(-0.5, 0.5)),  # Small random
                        (step_size * random.choice([-0.5, 0, 0.5]), 0),  # Small X-step
                        (0, step_size * random.choice([-0.5, 0, 0.5])),  # Small Y-step
                    ])
                
                # Always include no-move case for baseline
                test_moves.append((0, 0))

                # Add systematic moves in later iterations
                if iter_num > iterations * 0.6:
                    # Add structured moves in last phase
                    test_moves.extend([
                        (step_size * 0.1, 0),
                        (-step_size * 0.1, 0),
                        (0, step_size * 0.1),
                        (0, -step_size * 0.1),
                    ])

                for dx, dy in test_moves:
                    test_x = max(0.001, min(rect_width - 0.001, original_x + dx))
                    test_y = max(0.001, min(rect_height - 0.001, original_y + dy))

                    # Compute maximum possible radius at new position
                    temp_circles = current.copy()
                    temp_circles[i] = [test_x, test_y, 0.01]  # Temporarily small
                    
                    # Use vectorized version for performance
                    max_r = compute_max_radius_vectorized(
                        temp_circles[:, :2], temp_circles[:, 2], 
                        test_x, test_y, rect_width, rect_height
                    )
                    test_r = max(0.001, min(max_r, original_r + random.uniform(-0.02, 0.02)))

                    # Apply final adjustment
                    temp_circles[i] = [test_x, test_y, test_r]

                    # Validate and calculate new sum with appropriate tolerance
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

    def adaptive_evolution_optimization() -> np.ndarray:
        """Advanced evolutionary optimization with multiple strategies"""
        best_circles = None
        best_sum = -float('inf')
        
        # Strategy 1: Multi-scale with progressive refinement
        print("Running multi-scale optimization...")
        
        # Scale 1: Coarse optimization with large steps
        coarse_circles = create_hybrid_initialization(21, rect_width, rect_height)
        coarse_circles = local_refinement_step(coarse_circles, rect_width, rect_height, 20, relax_overlap=True)
        coarse_sum = calculate_radius_sum(coarse_circles)
        print(f"Coarse optimization sum: {coarse_sum}")

        # Scale 2: Medium optimization with medium steps
        medium_circles = coarse_circles.copy()
        medium_circles = local_refinement_step(medium_circles, rect_width, rect_height, 50, relax_overlap=True)
        medium_sum = calculate_radius_sum(medium_circles)
        print(f"Medium optimization sum: {medium_sum}")

        # Scale 3: Fine optimization with small steps
        fine_circles = medium_circles.copy()
        fine_circles = local_refinement_step(fine_circles, rect_width, rect_height, 100, relax_overlap=False)
        fine_sum = calculate_radius_sum(fine_circles)
        print(f"Fine optimization sum: {fine_sum}")

        # Keep the best from scales
        if fine_sum > best_sum:
            best_sum = fine_sum
            best_circles = fine_circles.copy()

        # Strategy 2: Alternative initialization
        print("Trying alternative initialization...")
        alt_circles = create_hybrid_initialization(21, rect_width, rect_height)
        alt_circles = local_refinement_step(alt_circles, rect_width, rect_height, 30, relax_overlap=True)
        alt_circles = local_refinement_step(alt_circles, rect_width, rect_height, 70, relax_overlap=False)
        alt_sum = calculate_radius_sum(alt_circles)
        print(f"Alternative initialization sum: {alt_sum}")

        if alt_sum > best_sum:
            best_sum = alt_sum
            best_circles = alt_circles.copy()

        # Strategy 3: Evolved random approach
        print("Running evolved random approach...")
        evolved_circles = create_hybrid_initialization(21, rect_width, rect_height)
        # More aggressive refinement
        evolved_circles = local_refinement_step(evolved_circles, rect_width, rect_height, 150, relax_overlap=False)
        evolved_sum = calculate_radius_sum(evolved_circles)
        print(f"Evolved random approach sum: {evolved_sum}")

        if evolved_sum > best_sum:
            best_sum = evolved_sum
            best_circles = evolved_circles.copy()

        return best_circles

    # Main optimization workflow
    final_circles = adaptive_evolution_optimization()

    # Final validation
    if final_circles is not None:
        # Ensure final configuration is valid
        while True:
            valid = is_valid_configuration_vectorized(final_circles, rect_width, rect_height)
            
            if valid:
                break
            else:
                # Reinitialize if invalid
                final_circles = create_hybrid_initialization(21, rect_width, rect_height)
                final_circles = local_refinement_step(final_circles, rect_width, rect_height, 50)
                continue

        # Final optimization pass
        final_circles = local_refinement_step(final_circles, rect_width, rect_height, 50, relax_overlap=False)

    return final_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")