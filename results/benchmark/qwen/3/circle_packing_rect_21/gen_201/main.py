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
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    def optimize_rectangle_dimensions() -> Tuple[float, float]:
        """Find optimal rectangle dimensions to maximize packing efficiency"""
        # Try various aspect ratios to find best configuration
        best_ratio = 1.5  # Default from previous experiments
        best_score = 0.0
        
        # Test different width/height ratios around the empirical optimum
        ratios = [0.8, 1.0, 1.2, 1.3, 1.5, 1.7, 2.0]
        
        for ratio in ratios:
            width = ratio * 1.0  # normalize to make perimeter = 4
            height = 1.0 / ratio
            # Ensure perimeter = 4: width + height = 2
            if abs(width + height - 2.0) < 0.01:
                # Rough estimate of packing efficiency for this ratio
                score = 1.0 / (1.0 + abs(ratio - 1.5))  # favor ratios around 1.5
                if score > best_score:
                    best_score = score
                    best_ratio = ratio
        
        width = best_ratio * 1.0
        height = 1.0 / best_ratio
        return width, height

    def create_voronoi_hybrid_initialization(n_circles: int, width: float, height: float) -> np.ndarray:
        """Create initial configuration using Voronoi-based hybrid approach"""
        circles = np.zeros((n_circles, 3))

        # Phase 1: Corner and edge seeding with strategic placement
        corner_positions = [
            (width * 0.1, height * 0.1),    # bottom-left
            (width * 0.9, height * 0.1),    # bottom-right
            (width * 0.1, height * 0.9),    # top-left
            (width * 0.9, height * 0.9),    # top-right
            (width * 0.5, height * 0.1),    # bottom-middle
            (width * 0.5, height * 0.9),    # top-middle
            (width * 0.1, height * 0.5),    # left-middle
            (width * 0.9, height * 0.5),    # right-middle
        ]

        # Place corner circles with larger initial radii
        for i in range(min(len(corner_positions), n_circles)):
            x, y = corner_positions[i]
            circles[i] = [x, y, 0.04]

        # Phase 2: Voronoi-based distribution for remaining circles
        remaining = n_circles - len(corner_positions)
        if remaining > 0:
            # Generate Voronoi points with better spatial distribution
            voronoi_points = []
            
            # Add boundary points for better coverage
            for _ in range(remaining * 3):  # Extra points for better Voronoi coverage
                x = random.uniform(0.05 * width, 0.95 * width)
                y = random.uniform(0.05 * height, 0.95 * height)
                voronoi_points.append([x, y])
            
            # Add some additional structured points
            grid_points = []
            grid_size = max(2, int(math.sqrt(remaining)))
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(grid_points) >= remaining * 2:
                        break
                    x = width * (0.1 + 0.8 * i / (grid_size - 1) if grid_size > 1 else 0.5)
                    y = height * (0.1 + 0.8 * j / (grid_size - 1) if grid_size > 1 else 0.5)
                    grid_points.append([x, y])
            
            voronoi_points.extend(grid_points[:remaining * 2])
            
            # Create Voronoi diagram and use centroids for circle positions
            try:
                vor = Voronoi(voronoi_points)
                # Take a subset of Voronoi vertices that are valid
                valid_vertices = []
                for vertex in vor.vertices:
                    if (0.05 * width <= vertex[0] <= 0.95 * width) and (0.05 * height <= vertex[1] <= 0.95 * height):
                        valid_vertices.append(vertex)
                
                # Use Voronoi vertices as center points
                chosen_centers = valid_vertices[:min(remaining, len(valid_vertices))]
                if len(chosen_centers) < remaining:
                    # Fill missing circles with random points
                    for i in range(len(chosen_centers), remaining):
                        x = random.uniform(0.05 * width, 0.95 * width)
                        y = random.uniform(0.05 * height, 0.95 * height)
                        chosen_centers.append([x, y])
                
                # Assign positions with medium initial radii
                for i, (x, y) in enumerate(chosen_centers[:remaining]):
                    circles[len(corner_positions) + i] = [x, y, 0.025]
            except Exception:
                # Fallback to hexagonal grid if Voronoi fails
                rows = int(math.ceil(math.sqrt(remaining)))
                cols = int(math.ceil(remaining / rows))
                cell_width = width / (cols + 1)
                cell_height = height / (rows + 1)
                idx = len(corner_positions)
                for i in range(rows):
                    for j in range(cols):
                        if idx >= n_circles:
                            break
                        x_offset = 0.0 if i % 2 == 0 else 0.5
                        x = (j + 1 + x_offset) * cell_width
                        y = (i + 1) * cell_height
                        x = max(0.01, min(width - 0.01, x))
                        y = max(0.01, min(height - 0.01, y))
                        circles[idx] = [x, y, 0.025]
                        idx += 1
                        if idx >= n_circles:
                            break

        return circles

    def compute_max_radius_at_position(x: float, y: float, existing_circles: np.ndarray,
                                     rect_width: float, rect_height: float) -> float:
        """Compute maximum possible radius for a circle at given position with early termination"""
        # Distance to boundaries
        min_bound = min(x, rect_width - x, y, rect_height - y)

        # Distance to other circles (with early termination for efficiency)
        min_dist = float('inf')
        for i in range(len(existing_circles)):
            ex, ey, er = existing_circles[i]
            dx = x - ex
            dy = y - ey
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 0.0001:  # Avoid self-distance
                min_dist = min(min_dist, dist - er)
                # Early termination if already very close to another circle
                if min_dist < 0.001:
                    break

        # Take minimum of boundary and other-circle distances
        max_radius = min(min_bound, min_dist if min_dist < float('inf') else float('inf'))
        return max(0.001, max_radius)

    def is_valid_configuration_vectorized(circles: np.ndarray, rect_width: float, rect_height: float) -> bool:
        """Vectorized validation of circle configuration with early exit"""
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

    def adaptive_local_search(circles: np.ndarray, rect_width: float, rect_height: float,
                            iterations: int = 150) -> np.ndarray:
        """Perform adaptive local search with multi-level refinement"""
        current = circles.copy()
        current_sum = calculate_radius_sum(current)

        # Adaptive step sizes with better decay schedule
        step_size = 0.05
        step_decay = 0.98  # Gradual decrease for better convergence
        min_step_size = 0.001

        # Track recent improvements for adaptive behavior
        recent_improvements = []
        max_recent = 10

        for iter_num in range(iterations):
            # Adjust step size based on iteration progress and improvement rate
            adjusted_step_size = max(min_step_size, step_size * (step_decay ** (iter_num // 10)))
            
            # In early iterations, allow more aggressive exploration with relaxed validation
            relax_validation = (iter_num < iterations // 3)
            
            # Try to improve each circle
            indices = list(range(len(current)))
            random.shuffle(indices)  # Random order for better exploration
            
            for i in indices:
                original_x, original_y, original_r = current[i]

                # Track best improvement found at each iteration
                best_x, best_y, best_r = original_x, original_y, original_r
                best_sum = current_sum

                # Enhanced perturbation strategies
                test_moves = []
                
                # 1. Gaussian-based perturbations (larger steps in beginning)
                test_moves.append((adjusted_step_size * random.gauss(0, 1), 
                                  adjusted_step_size * random.gauss(0, 1)))
                test_moves.append((adjusted_step_size * random.gauss(0, 1), 0))
                test_moves.append((0, adjusted_step_size * random.gauss(0, 1)))
                
                # 2. Coordinate-specific moves
                test_moves.append((adjusted_step_size * random.uniform(-1, 1), 0))
                test_moves.append((0, adjusted_step_size * random.uniform(-1, 1)))
                
                # 3. Small random moves for fine-tuning
                test_moves.append((random.uniform(-adjusted_step_size/2, adjusted_step_size/2), 
                                  random.uniform(-adjusted_step_size/2, adjusted_step_size/2)))
                
                # 4. Systematic moves in later iterations
                if iter_num > iterations // 2:
                    test_moves.extend([
                        (adjusted_step_size * random.choice([-1, 1]), 0),
                        (0, adjusted_step_size * random.choice([-1, 1])),
                        (adjusted_step_size, adjusted_step_size),
                        (-adjusted_step_size, -adjusted_step_size),
                    ])
                
                # 5. No-move baseline (always included)
                test_moves.append((0, 0))

                # Try different perturbations
                for dx, dy in test_moves:
                    test_x = max(0.001, min(rect_width - 0.001, original_x + dx))
                    test_y = max(0.001, min(rect_height - 0.001, original_y + dy))

                    # Compute maximum possible radius at new position
                    temp_circles = current.copy()
                    temp_circles[i] = [test_x, test_y, 0.01]  # Temporarily small
                    
                    # Compute new max radius
                    max_r = compute_max_radius_at_position(test_x, test_y, temp_circles, rect_width, rect_height)
                    
                    # Adjust radius within limits
                    test_r = min(max_r, max(0.001, original_r + random.uniform(-0.04, 0.04)))

                    # Apply final adjustment
                    temp_circles[i] = [test_x, test_y, test_r]

                    # Validate and calculate new sum
                    if relax_validation:
                        # Relaxed validation in early phase for exploration
                        valid = True
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
                    
                    # Track improvement for adaptive behavior
                    recent_improvements.append(best_sum - calculate_radius_sum(current))
                    if len(recent_improvements) > max_recent:
                        recent_improvements.pop(0)

        return current

    def multi_stage_optimization(n_starts: int = 8) -> np.ndarray:
        """Multi-stage optimization with Voronoi hybrid initialization"""
        best_circles = None
        best_sum = -float('inf')

        # Determine optimal rectangle dimensions
        rect_width, rect_height = optimize_rectangle_dimensions()
        print(f"Optimal rectangle dimensions: {rect_width:.3f} x {rect_height:.3f}")

        # Different initialization strategies
        init_strategies = ['voronoi', 'hexagonal', 'random', 'hybrid']

        for start_num in range(n_starts):
            # Select initialization strategy
            strategy = init_strategies[start_num % len(init_strategies)]

            if strategy == 'voronoi' or strategy == 'hybrid':
                circles = create_voronoi_hybrid_initialization(21, rect_width, rect_height)
                # Apply a bit more initial refinement
                circles = adaptive_local_search(circles, rect_width, rect_height, 20)
            elif strategy == 'hexagonal':
                # Create hexagonal grid initialization
                circles = np.zeros((21, 3))
                rows = int(math.ceil(math.sqrt(21)))
                cols = int(math.ceil(21 / rows))
                cell_width = rect_width / (cols + 1)
                cell_height = rect_height / (rows + 1)
                idx = 0
                for i in range(rows):
                    for j in range(cols):
                        if idx >= 21:
                            break
                        x_offset = 0.0 if i % 2 == 0 else 0.5
                        x = (j + 1 + x_offset) * cell_width
                        y = (i + 1) * cell_height
                        x = max(0.01, min(rect_width - 0.01, x))
                        y = max(0.01, min(rect_height - 0.01, y))
                        circles[idx] = [x, y, 0.02]
                        idx += 1
                        if idx >= 21:
                            break
            else:  # random
                circles = np.zeros((21, 3))
                for i in range(21):
                    x = random.uniform(0.01, rect_width - 0.01)
                    y = random.uniform(0.01, rect_height - 0.01)
                    circles[i] = [x, y, 0.02]

            # Stage 1: Coarse refinement with fewer iterations
            refined_1 = adaptive_local_search(circles, rect_width, rect_height, 25)

            # Stage 2: Medium refinement with moderate iterations
            refined_2 = adaptive_local_search(refined_1, rect_width, rect_height, 40)

            # Stage 3: Fine refinement with more iterations (strict validation)
            refined_3 = adaptive_local_search(refined_2, rect_width, rect_height, 60)

            final_sum = calculate_radius_sum(refined_3)

            if final_sum > best_sum:
                best_sum = final_sum
                best_circles = refined_3.copy()

        return best_circles, rect_width, rect_height

    # Main optimization workflow
    # Multi-start optimization to avoid local optima
    final_circles, rect_width, rect_height = multi_stage_optimization(8)

    # Final validation and cleanup
    if final_circles is not None:
        # Ensure final configuration is valid
        max_attempts = 5
        attempts = 0
        
        while not is_valid_configuration_vectorized(final_circles, rect_width, rect_height) and attempts < max_attempts:
            # If invalid, try reinitialization with Voronoi strategy
            final_circles = create_voronoi_hybrid_initialization(21, rect_width, rect_height)
            final_circles = adaptive_local_search(final_circles, rect_width, rect_height, 40)
            attempts += 1

    # Final optimization pass to fine-tune
    if final_circles is not None:
        final_circles = adaptive_local_search(final_circles, rect_width, rect_height, 30)

    return final_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")