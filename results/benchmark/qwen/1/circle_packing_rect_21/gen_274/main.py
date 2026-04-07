# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
import time
from collections import defaultdict
from scipy.spatial import Voronoi

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Rectangle dimensions (width + height = 2)
    rect_width = 1.0
    rect_height = 1.0

    # Number of circles
    n = 21

    def generate_initial_patterns(width, height, n):
        """Generate multiple initial patterns and return the best one"""
        patterns = []

        # Hexagonal packing
        hex_pattern = generate_hexagonal_pattern(width, height, n)
        patterns.append(hex_pattern)

        # Grid-based packing
        grid_pattern = generate_grid_pattern(width, height, n)
        patterns.append(grid_pattern)

        # Spiral pattern
        spiral_pattern = generate_spiral_pattern(width, height, n)
        patterns.append(spiral_pattern)

        # Random with constraints
        random_pattern = generate_random_constrained_pattern(width, height, n)
        patterns.append(random_pattern)

        # Evaluate all patterns and select best
        best_pattern = None
        best_fitness = -float('inf')

        for pattern in patterns:
            fitness = evaluate_fitness(pattern, width, height)
            if fitness > best_fitness:
                best_fitness = fitness
                best_pattern = pattern

        return best_pattern if best_pattern is not None else generate_hexagonal_pattern(width, height, n)

    def generate_hexagonal_pattern(width, height, n):
        """Generate initial hexagonal packing pattern"""
        circles = np.zeros((n, 3))

        # Determine grid parameters
        rows = int(np.sqrt(n))
        cols = int(np.ceil(n / rows))

        # Calculate spacing
        margin = 0.05
        max_radius = min(width, height) * 0.08

        # Create hexagonal grid
        x_spacing = max_radius * 2.5
        y_spacing = max_radius * 2.165  # sqrt(3)/2 * 2

        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = margin + j * x_spacing
                y = margin + i * y_spacing

                if i % 2 == 1:
                    x += x_spacing / 2

                # Adjust for bounds
                x = max(max_radius, min(width - max_radius, x))
                y = max(max_radius, min(height - max_radius, y))

                circles[idx] = [x, y, max_radius]
                idx += 1

        return circles

    def generate_grid_pattern(width, height, n):
        """Generate initial grid pattern"""
        circles = np.zeros((n, 3))

        # Find grid dimensions
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        # Calculate spacing
        margin = 0.05
        cell_width = (width - 2 * margin) / cols
        cell_height = (height - 2 * margin) / rows
        max_radius = min(cell_width, cell_height) * 0.4

        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = margin + j * cell_width + cell_width / 2
                y = margin + i * cell_height + cell_height / 2
                circles[idx] = [x, y, max_radius]
                idx += 1

        return circles

    def generate_spiral_pattern(width, height, n):
        """Generate initial spiral pattern"""
        circles = np.zeros((n, 3))
        center_x, center_y = width / 2, height / 2
        max_radius = min(width, height) * 0.1
        angle_step = 2 * np.pi / 5
        radius_step = 0.05

        for i in range(n):
            angle = i * angle_step
            radius = i * radius_step
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)

            # Keep within bounds
            x = max(max_radius, min(width - max_radius, x))
            y = max(max_radius, min(height - max_radius, y))

            circles[i] = [x, y, max_radius]

        return circles

    def generate_random_constrained_pattern(width, height, n):
        """Generate random pattern with basic constraints"""
        circles = np.zeros((n, 3))
        max_radius = min(width, height) * 0.08

        # Precompute some random values to avoid repeated calls
        random_x = np.random.uniform(0, width, n)
        random_y = np.random.uniform(0, height, n)
        random_r = np.random.uniform(0.005, max_radius, n)

        for i in range(n):
            x = random_x[i]
            y = random_y[i]
            radius = random_r[i]

            # Check if this circle overlaps with existing ones
            valid = True
            for j in range(i):
                existing_x, existing_y, existing_r = circles[j]
                dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                if dist < (radius + existing_r):
                    valid = False
                    break

            if valid:
                circles[i] = [x, y, radius]
            else:
                # If failed, try again with different random values
                attempts = 0
                while not valid and attempts < 100:
                    x = np.random.uniform(max_radius, width - max_radius)
                    y = np.random.uniform(max_radius, height - max_radius)
                    radius = np.random.uniform(0.005, max_radius)

                    valid = True
                    for j in range(i):
                        existing_x, existing_y, existing_r = circles[j]
                        dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                        if dist < (radius + existing_r):
                            valid = False
                            break
                    attempts += 1

                if valid:
                    circles[i] = [x, y, radius]
                else:
                    # If still failing, just place at random valid position
                    circles[i] = [np.random.uniform(max_radius, width - max_radius),
                                  np.random.uniform(max_radius, height - max_radius),
                                  np.random.uniform(0.005, max_radius)]

        return circles

    def evaluate_fitness(individual, width, height):
        """Evaluate fitness of an individual - sum of radii with penalty for violations"""
        circles = individual.copy()
        total_radius = np.sum(circles[:, 2])

        # Penalty for boundary violations
        penalty = 0

        # Vectorized boundary check
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        radii = circles[:, 2]

        # Check if any circle violates bounds
        boundary_violations = (x_coords - radii < 0) | (x_coords + radii > width) | \
                              (y_coords - radii < 0) | (y_coords + radii > height)
        if np.any(boundary_violations):
            penalty -= 1000 * np.sum(boundary_violations)

        # Penalty for overlaps using spatial hashing for efficiency
        if len(circles) > 1:
            # Spatial hash grid approach for collision detection
            try:
                # Use spatial grid for O(n) collision detection instead of O(n^2)
                collisions = detect_collisions_spatial_hash(circles, width, height)
                if collisions > 0:
                    penalty -= 1000 * collisions
            except:
                # Fallback to classic O(n^2) method if spatial hash fails
                coords = circles[:, :2]
                radii = circles[:, 2]
                distances = cdist(coords, coords)
                # Create mask for upper triangle (avoid double counting)
                mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
                # Compute overlap penalties
                overlap_distances = distances[mask]
                overlap_radii = (radii[:, None] + radii[None, :])[mask]
                overlaps = overlap_distances < overlap_radii
                if np.any(overlaps):
                    overlap_penalty = -np.sum(overlap_radii[overlaps] - overlap_distances[overlaps]) * 100
                    penalty += overlap_penalty

        return total_radius + penalty

    def get_voronoi_criticality(individual):
        """Calculate criticality based on Voronoi cell areas - more accurate constraint density measure"""
        circles = individual.copy()
        n = len(circles)

        if n <= 1:
            return np.ones(n) * 0.01

        try:
            # Use scipy's Voronoi for better reliability
            # Add boundary points to create bounded Voronoi diagram
            points = circles[:, :2].copy()

            # Add corner and edge points to properly bound the Voronoi regions
            boundary_margin = 0.01
            boundary_points = [
                [boundary_margin, boundary_margin],
                [1-boundary_margin, boundary_margin],
                [boundary_margin, 1-boundary_margin],
                [1-boundary_margin, 1-boundary-margin],
                [0.5, boundary_margin],
                [0.5, 1-boundary_margin],
                [boundary_margin, 0.5],
                [1-boundary_margin, 0.5]
            ]
            points = np.vstack([points, boundary_points])

            vor = Voronoi(points)

            # Compute Voronoi cell areas for each original circle
            criticality_scores = np.zeros(n)

            for i in range(n):
                # Find the Voronoi region for this circle
                region_idx = np.where(vor.point_region == i)[0]
                if len(region_idx) > 0 and region_idx[0] < len(vor.regions):
                    region = vor.regions[region_idx[0]]
                    if -1 not in region and len(region) >= 3:
                        # Compute area of Voronoi cell using shoelace formula
                        vertices = np.array([vor.vertices[j] for j in region if j >= 0 and j < len(vor.vertices)])
                        if len(vertices) >= 3:
                            x = vertices[:, 0]
                            y = vertices[:, 1]
                            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                            # Criticality inversely proportional to cell area (smaller = more constrained)
                            criticality_scores[i] = 1.0 / (area + 1e-10)
                        else:
                            criticality_scores[i] = 1.0
                    else:
                        criticality_scores[i] = 1.0
                else:
                    criticality_scores[i] = 1.0

        except Exception as e:
            # Fallback to distance-based criticality if Voronoi fails
            print(f"Voronoi failure: {e}")
            # Vectorized computation of distances to all others
            coords = circles[:, :2]
            distances = cdist(coords, coords)

            # Set diagonal to infinity to exclude self-distances
            np.fill_diagonal(distances, np.inf)

            # Minimum distances for each circle
            min_distances = np.min(distances, axis=1)

            # Criticality is inverse of minimum distance
            # Add small epsilon to avoid division by zero
            criticality_scores = 1.0 / (min_distances + 1e-8)

        # Also consider boundary constraints
        for i in range(n):
            x, y, r = circles[i]
            # Distance to nearest boundary
            min_boundary_dist = min(x, y, 1-x, 1-y)
            # If very close to boundary, increase criticality
            if min_boundary_dist < 0.05:
                criticality_scores[i] *= (1 + 10 * (0.05 - min_boundary_dist))

        # Normalize
        if np.max(criticality_scores) > 0:
            criticality_scores = criticality_scores / np.max(criticality_scores)

        # Ensure minimum values
        criticality_scores = np.maximum(criticality_scores, 0.01)

        return criticality_scores

    def voronoi_local_refinement(circles, width, height, max_iter=10):
        """Apply local refinement using Voronoi-based optimization"""
        refined_circles = circles.copy()

        # Compute Voronoi diagram for current configuration
        try:
            # Add boundary points to make Voronoi more meaningful
            points = refined_circles[:, :2].copy()
            boundary_points = [
                [0, 0], [width, 0], [0, height], [width, height],
                [width/2, 0], [width/2, height],
                [0, height/2], [width, height/2]
            ]
            points = np.vstack([points, boundary_points])

            vor = Voronoi(points)

            # For each circle, compute its Voronoi region and try to move it to a better position
            for i in range(len(refined_circles)):
                # Get Voronoi region for this circle
                region_idx = np.where(vor.point_region == i)[0][0] if i in vor.point_region else -1

                if region_idx != -1 and region_idx < len(vor.regions):
                    region = vor.regions[region_idx]
                    if -1 not in region and len(region) >= 3:
                        # Find the center of the Voronoi cell
                        vertices = np.array([vor.vertices[j] for j in region if j >= 0])
                        if len(vertices) >= 3:
                            # Compute mean of vertices (center of Voronoi cell)
                            cell_center = np.mean(vertices, axis=0)

                            # Try moving the circle towards the center of its Voronoi cell
                            # but keep it within bounds and away from others
                            x, y, r = refined_circles[i]

                            # Move towards cell center but consider constraints
                            direction_x = cell_center[0] - x
                            direction_y = cell_center[1] - y

                            # Normalize direction
                            distance = np.sqrt(direction_x**2 + direction_y**2)
                            if distance > 0:
                                direction_x /= distance
                                direction_y /= distance

                                # Try a small step towards cell center
                                step_size = min(0.01, distance * 0.1)
                                new_x = x + direction_x * step_size
                                new_y = y + direction_y * step_size

                                # Keep within bounds
                                new_x = np.clip(new_x, r, width - r)
                                new_y = np.clip(new_y, r, height - r)

                                # Check if this movement improves overlap constraints
                                refined_circles[i] = [new_x, new_y, r]

        except Exception as e:
            # If Voronoi fails, do nothing
            pass

        return refined_circles

    def detect_collisions_spatial_hash(circles, width, height, grid_size=None):
        """Detect collisions using spatial hashing for O(n) complexity"""
        if grid_size is None:
            # Estimate grid size based on average radius
            avg_radius = np.mean(circles[:, 2])
            grid_size = max(avg_radius, 0.001) * 2

        # Create spatial hash grid
        grid = defaultdict(list)
        cell_size = grid_size

        # Hash each circle to its grid cell
        for i, (x, y, r) in enumerate(circles):
            # Only consider circles that are within bounds
            if x >= r and x <= width - r and y >= r and y <= height - r:
                cell_row = int(y // cell_size)
                cell_col = int(x // cell_size)
                grid[(cell_row, cell_col)].append(i)

                # Also add to neighboring cells to handle edge cases
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        neighbor_cell = (cell_row + dr, cell_col + dc)
                        if neighbor_cell != (cell_row, cell_col):
                            grid[neighbor_cell].append(i)

        # Check for collisions within each cell and neighbors
        collisions = 0
        checked_pairs = set()

        for cell, indices in grid.items():
            if len(indices) > 1:
                # Check all pairs in this cell
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        idx1, idx2 = indices[i], indices[j]
                        # Skip if already checked
                        if (idx1, idx2) in checked_pairs or (idx2, idx1) in checked_pairs:
                            continue

                        # Check actual distance
                        x1, y1, r1 = circles[idx1]
                        x2, y2, r2 = circles[idx2]

                        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        if dist < (r1 + r2):
                            collisions += 1
                            checked_pairs.add((idx1, idx2))

        return collisions

    def get_voronoi_criticality(individual):
        """Calculate criticality based on Voronoi cell areas - more accurate constraint density measure"""
        circles = individual.copy()
        n = len(circles)

        if n <= 1:
            return np.ones(n) * 0.01

        try:
            # Use scipy's Voronoi for better reliability
            from scipy.spatial import Voronoi

            # Add boundary points to create bounded Voronoi diagram
            points = circles[:, :2].copy()

            # Add corner and edge points to properly bound the Voronoi regions
            boundary_margin = 0.01
            boundary_points = [
                [boundary_margin, boundary_margin],
                [1-boundary_margin, boundary_margin],
                [boundary_margin, 1-boundary_margin],
                [1-boundary_margin, 1-boundary_margin],
                [0.5, boundary_margin],
                [0.5, 1-boundary_margin],
                [boundary_margin, 0.5],
                [1-boundary_margin, 0.5]
            ]
            points = np.vstack([points, boundary_points])

            vor = Voronoi(points)

            # Compute Voronoi cell areas for each original circle
            criticality_scores = np.zeros(n)

            for i in range(n):
                # Find the Voronoi region for this circle
                region_idx = np.where(vor.point_region == i)[0]
                if len(region_idx) > 0 and region_idx[0] < len(vor.regions):
                    region = vor.regions[region_idx[0]]
                    if -1 not in region and len(region) >= 3:
                        # Compute area of Voronoi cell using shoelace formula
                        vertices = np.array([vor.vertices[j] for j in region if j >= 0 and j < len(vor.vertices)])
                        if len(vertices) >= 3:
                            x = vertices[:, 0]
                            y = vertices[:, 1]
                            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                            # Criticality inversely proportional to cell area (smaller = more constrained)
                            criticality_scores[i] = 1.0 / (area + 1e-10)
                        else:
                            criticality_scores[i] = 1.0
                    else:
                        criticality_scores[i] = 1.0
                else:
                    criticality_scores[i] = 1.0

        except Exception as e:
            # Fallback to distance-based criticality if Voronoi fails
            print(f"Voronoi failure: {e}")
            # Vectorized computation of distances to all others
            coords = circles[:, :2]
            distances = cdist(coords, coords)

            # Set diagonal to infinity to exclude self-distances
            np.fill_diagonal(distances, np.inf)

            # Minimum distances for each circle
            min_distances = np.min(distances, axis=1)

            # Criticality is inverse of minimum distance
            # Add small epsilon to avoid division by zero
            criticality_scores = 1.0 / (min_distances + 1e-8)

        # Also consider boundary constraints
        for i in range(n):
            x, y, r = circles[i]
            # Distance to nearest boundary
            min_boundary_dist = min(x, y, 1-x, 1-y)
            # If very close to boundary, increase criticality
            if min_boundary_dist < 0.05:
                criticality_scores[i] *= (1 + 10 * (0.05 - min_boundary_dist))

        # Normalize
        if np.max(criticality_scores) > 0:
            criticality_scores = criticality_scores / np.max(criticality_scores)

        # Ensure minimum values
        criticality_scores = np.maximum(criticality_scores, 0.01)

        return criticality_scores

    def mut_radius(individual, indpb=0.2):
        """Mutation operator that modifies only the radius of selected circles with adaptive strength"""
        mutated_individual = individual.copy()
        n = len(mutated_individual)

        # Get criticality scores
        criticality = get_voronoi_criticality(mutated_individual)

        # Sort by criticality (most critical first)
        sorted_indices = np.argsort(-criticality)  # Descending order

        # Mutate top 40% of critical circles (focus on the most constrained)
        num_mutations = int(n * 0.4)
        mutation_indices = sorted_indices[:num_mutations]

        for i in range(num_mutations):
            idx = mutation_indices[i]
            if random.random() < indpb:
                old_radius = mutated_individual[idx, 2]

                # Adaptive mutation strength based on criticality
                # High criticality (constrained) = small mutation
                # Low criticality (loosely constrained) = large mutation
                adaptive_strength = 0.005 * (1.0 / (criticality[idx] + 0.001))
                adaptive_strength = min(adaptive_strength, 0.02)  # Cap maximum mutation

                # Small random change to radius
                delta = np.random.normal(0, adaptive_strength)
                new_radius = max(0.001, old_radius + delta)
                mutated_individual[idx, 2] = new_radius

        return mutated_individual,

    def improved_local_search(individual, width, height, max_iter=50):
        """Improved local search that incorporates Voronoi-based refinement"""
        current_individual = individual.copy()

        # First apply Voronoi refinement to get better initial positioning
        current_individual = voronoi_local_refinement(current_individual, width, height)

        for iteration in range(max_iter):
            # Get criticality scores for the current configuration
            criticality = get_voronoi_criticality(current_individual)

            # Create test individual
            test_individual = current_individual.copy()

            # Focus on top 30% most critical circles for refinement
            sorted_indices = np.argsort(-criticality)
            num_changes = max(1, int(len(criticality) * 0.3))

            # Select circles to perturb
            for i in range(num_changes):
                idx = sorted_indices[i]

                # Apply Voronoi-based refinement for the circle
                x, y, r = test_individual[idx]

                # Perturb position and radius using adaptive approach
                # Position adjustment based on Voronoi properties
                new_x = max(0.005, min(0.995, x + np.random.normal(0, 0.005 * (1.0 / (criticality[idx] + 0.001)))))
                new_y = max(0.005, min(0.995, y + np.random.normal(0, 0.005 * (1.0 / (criticality[idx] + 0.001)))))

                # Radius adjustment with stronger adaptation
                radius_adaptive_factor = 0.002 * (1.0 / (criticality[idx] + 0.001))
                new_r = max(0.001, r + np.random.normal(0, radius_adaptive_factor))

                test_individual[idx] = [new_x, new_y, new_r]

            # Validate and accept if improvement
            if evaluate_fitness(test_individual, width, height) > evaluate_fitness(current_individual, width, height):
                current_individual = test_individual.copy()

                # Apply Voronoi refinement after each improvement
                current_individual = voronoi_local_refinement(current_individual, width, height)

        return current_individual

    def crossover(parent1, parent2):
        """Crossover operator that exchanges radii of most critical circles"""
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Get criticality scores for both parents
        crit1 = get_voronoi_criticality(parent1)
        crit2 = get_voronoi_criticality(parent2)

        # Exchange radii of circles with highest criticality
        combined_criticality = np.maximum(crit1, crit2)
        sorted_indices = np.argsort(-combined_criticality)

        # Exchange radii for top 30% of circles
        num_exchanges = int(len(parent1) * 0.3)
        for i in range(num_exchanges):
            idx = sorted_indices[i]
            child1[idx, 2], child2[idx, 2] = child2[idx, 2], child1[idx, 2]

        return child1, child2

    def is_valid_solution(circles, width, height):
        """Check if solution is valid - faster version using spatial hash"""
        # Check boundary constraints
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
                return False

        # Check overlap constraints using spatial hash
        if len(circles) > 1:
            collisions = detect_collisions_spatial_hash(circles, width, height)
            return collisions == 0

        return True

    # Main algorithm - optimized version
    start_time = time.time()

    # Initialize with best pattern
    best_individual = generate_initial_patterns(rect_width, rect_height, n)

    # Apply improved local search to improve the initial solution
    best_individual = improved_local_search(best_individual, rect_width, rect_height, max_iter=100)

    # Apply additional Voronoi refinement
    best_individual = voronoi_local_refinement(best_individual, rect_width, rect_height, max_iter=20)

    # Early termination check
    if time.time() - start_time > 50:
        return best_individual

    # Final polishing with enhanced local search
    best_individual = improved_local_search(best_individual, rect_width, rect_height, max_iter=50)

    # Final validation
    if not is_valid_solution(best_individual, rect_width, rect_height):
        # Fallback to structured pattern
        best_individual = generate_hexagonal_pattern(rect_width, rect_height, n)

    return best_individual

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")