# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import KDTree
import random
import time
from collections import defaultdict

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Rectangle dimensions: width + height = 2, optimized ratio
    rect_width = 1.2
    rect_height = 0.8

    n = 21

    # Generate adaptive hexagonal grid initial placement using proven method
    def generate_adaptive_hexagonal_grid(num_circles, width, height):
        # Calculate grid dimensions using a simpler but more effective approach
        # Based on the square root of number of circles and container aspect ratio
        sqrt_n = np.ceil(np.sqrt(num_circles))
        cols = int(sqrt_n)
        rows = int(np.ceil(num_circles / sqrt_n))

        # Calculate spacing based on container dimensions
        margin = 0.1
        usable_width = width - 2 * margin
        usable_height = height - 2 * margin

        # Calculate spacing for hexagonal packing
        spacing_x = usable_width / cols * 0.85 if cols > 0 else usable_width
        spacing_y = usable_height / rows * 0.85 if rows > 0 else usable_height

        # For hexagonal packing, adjust vertical spacing
        hex_spacing_y = spacing_y * np.sqrt(3) / 2

        # Ensure we don't exceed container bounds
        max_cols = int(usable_width / spacing_x) if spacing_x > 0 else cols
        max_rows = int(usable_height / hex_spacing_y) if hex_spacing_y > 0 else rows

        if max_cols < cols or max_rows < rows:
            # Reduce spacing to fit
            adjusted_spacing_x = usable_width / max_cols * 0.95
            adjusted_spacing_y = usable_height / max_rows * 0.95
            spacing_x = adjusted_spacing_x
            hex_spacing_y = adjusted_spacing_y * np.sqrt(3) / 2

        circles = []
        y_offset = margin
        x_offset = margin

        # Create hexagonal grid with proper offsetting
        circle_count = 0
        for i in range(max_rows):
            y = y_offset + i * hex_spacing_y
            x_start = x_offset + (i % 2) * spacing_x / 2
            for j in range(max_cols):
                if circle_count >= num_circles:
                    break
                x = x_start + j * spacing_x
                if x < width - margin and y < height - margin:
                    # Start with radius based on spacing
                    r = min(0.05, spacing_x / 4, hex_spacing_y / 4)
                    circles.append([x, y, r])
                    circle_count += 1
            if circle_count >= num_circles:
                break

        # Fill remaining spots with random positions
        while len(circles) < num_circles:
            x = np.random.uniform(margin, width - margin)
            y = np.random.uniform(margin, height - margin)
            r = np.random.uniform(0.01, 0.05)
            circles.append([x, y, r])

        return np.array(circles)

    # Initial configuration
    circles = generate_adaptive_hexagonal_grid(n, rect_width, rect_height)

    # Efficient constraint validation with spatial indexing and constraint-aware penalties
    def calculate_fitness_with_spatial_indexing(circles_array):
        total_radius = np.sum(circles_array[:, 2])

        penalty = 0

        # Boundary penalties with constraint-aware scaling
        for i in range(n):
            cx, cy, r = circles_array[i]
            boundary_violation = 0

            # Check all four boundaries and accumulate violations
            if cx - r < 0.01:
                boundary_violation += (r - cx)**2
            if cx + r > rect_width - 0.01:
                boundary_violation += (cx + r - rect_width)**2
            if cy - r < 0.01:
                boundary_violation += (r - cy)**2
            if cy + r > rect_height - 0.01:
                boundary_violation += (cy + r - rect_height)**2

            # Apply constraint-aware penalty - scale with violation magnitude
            if boundary_violation > 0:
                penalty += 100000 * boundary_violation

        # Overlap penalties using spatial indexing for efficiency
        points = circles_array[:, :2]
        tree = KDTree(points)

        # Query for all possible overlapping pairs efficiently
        # This reduces from O(n^2) to near O(n log n) for overlap checking
        for i in range(n):
            cx, cy, r = circles_array[i]

            # Find neighbors within 2*(r + safety_margin) distance
            neighbor_indices = tree.query_ball_point([cx, cy], 2*(r + 0.01) + 0.001)

            for j in neighbor_indices:
                if i != j:
                    other_cx, other_cy, other_r = circles_array[j]
                    dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                    overlap = (r + other_r) - dist

                    if overlap > 0:
                        # Apply constraint-aware penalty that scales with overlap magnitude
                        penalty += 1000000 * overlap**2

        # Add additional penalty for very tight constraints that might cause numerical instability
        constraint_penalty = 0
        for i in range(n):
            cx, cy, r = circles_array[i]
            # Penalize very small radii that might cause numerical issues
            if r < 0.001:
                constraint_penalty += 1000000 * (0.001 - r)**2

        return total_radius - penalty - constraint_penalty

    # Fast local refinement focused on improvement regions with enhanced strategy
    def adaptive_refinement(circles_array, max_iter=80):
        best_circles = circles_array.copy()
        best_fitness = calculate_fitness_with_spatial_indexing(best_circles)

        # Track circles that have been modified recently
        recent_updates = set()

        for iteration in range(max_iter):
            improved = False

            # Select circles to try improving, prioritizing those with high overlap risk
            # First, identify circles that are likely to benefit from radius increase
            candidates = list(range(n))

            # Sort candidates by how constrained they are (more neighbors = more constrained)
            candidate_scores = []
            points = best_circles[:, :2]
            tree = KDTree(points)

            for i in candidates:
                cx, cy, r = best_circles[i]
                neighbor_indices = tree.query_ball_point([cx, cy], 2*(r + 0.01) + 0.001)
                # Score inversely proportional to number of neighbors (fewer neighbors = more flexibility)
                candidate_scores.append((i, len(neighbor_indices)))

            # Sort by constraint level (most constrained first)
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            candidates = [idx for idx, _ in candidate_scores]

            # Shuffle for variety but keep high-constraint ones first
            random.shuffle(candidates[:min(5, len(candidates))])

            # Process a subset of candidates each iteration to maintain speed
            processed_count = 0

            for i in candidates:
                if processed_count > 10:  # Limit per iteration
                    break
                processed_count += 1

                cx, cy, r = best_circles[i]

                # Compute max allowable radius
                max_radius = float('inf')

                # Boundary constraints
                max_radius = min(max_radius, cx - 0.01)
                max_radius = min(max_radius, rect_width - cx - 0.01)
                max_radius = min(max_radius, cy - 0.01)
                max_radius = min(max_radius, rect_height - cy - 0.01)

                # Overlap constraints using spatial indexing on current neighbors
                points = best_circles[:, :2]
                tree = KDTree(points)
                neighbor_indices = tree.query_ball_point([cx, cy], 2*(r + 0.01) + 0.001)

                for j in neighbor_indices:
                    if i != j:
                        other_cx, other_cy, other_r = best_circles[j]
                        dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                        max_radius = min(max_radius, dist - other_r - 0.001)

                # Try to increase radius aggressively
                if max_radius > r and max_radius > 0.001:
                    # Try multiple increments in sequence
                    increments = [0.001, 0.002, 0.005, 0.01, 0.02]
                    for incr in increments:
                        new_r = min(r + incr, max_radius)
                        if new_r <= r + 0.0001:  # Skip very small increases
                            continue

                        # Quick validation with spatial indexing
                        temp_circles = best_circles.copy()
                        temp_circles[i, 2] = new_r

                        # Check validity using spatial indexing
                        valid = True
                        temp_points = temp_circles[:, :2]
                        temp_tree = KDTree(temp_points)
                        temp_neighbor_indices = temp_tree.query_ball_point([cx, cy], 2*(new_r + 0.01) + 0.001)

                        for k in temp_neighbor_indices:
                            if k != i:
                                other_cx, other_cy, other_r = temp_circles[k]
                                dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                                if dist < new_r + other_r:
                                    valid = False
                                    break

                        if valid:
                            test_circles = best_circles.copy()
                            test_circles[i, 2] = new_r
                            test_fitness = calculate_fitness_with_spatial_indexing(test_circles)

                            if test_fitness > best_fitness:
                                best_fitness = test_fitness
                                best_circles = test_circles
                                improved = True
                                break  # Take the first valid improvement

                if improved:
                    recent_updates.add(i)
                    if len(recent_updates) > 10:
                        recent_updates.pop()

            # Occasionally try position perturbations for global exploration
            if not improved and iteration % 4 == 0:
                for _ in range(3):  # Try up to 3 position changes
                    i = random.randint(0, n - 1)
                    x_old, y_old, r = best_circles[i]

                    # Scale perturbation based on iteration and radius
                    scale = max(0.005, 0.03 - iteration * 0.001)
                    dx = np.random.uniform(-scale, scale)
                    dy = np.random.uniform(-scale, scale)

                    new_x = x_old + dx
                    new_y = y_old + dy

                    # Check bounds
                    if (0.01 + r <= new_x <= rect_width - 0.01 - r and
                        0.01 + r <= new_y <= rect_height - 0.01 - r):

                        temp_circles = best_circles.copy()
                        temp_circles[i, 0] = new_x
                        temp_circles[i, 1] = new_y

                        # Validate overlap
                        valid = True
                        temp_points = temp_circles[:, :2]
                        temp_tree = KDTree(temp_points)
                        temp_neighbor_indices = temp_tree.query_ball_point([new_x, new_y], 2*(r + 0.01) + 0.001)

                        for k in temp_neighbor_indices:
                            if k != i:
                                other_cx, other_cy, other_r = temp_circles[k]
                                dist = np.sqrt((new_x - other_cx)**2 + (new_y - other_cy)**2)
                                if dist < r + other_r:
                                    valid = False
                                    break

                        if valid:
                            test_circles = best_circles.copy()
                            test_circles[i, 0] = new_x
                            test_circles[i, 1] = new_y
                            test_fitness = calculate_fitness_with_spatial_indexing(test_circles)

                            if test_fitness > best_fitness:
                                best_fitness = test_fitness
                                best_circles = test_circles
                                improved = True

            # If no improvement, reduce search scope
            if not improved:
                break

        return best_circles

    # Multi-stage optimization approach
    # Stage 1: Coarse refinement with basic local search
    coarse_solution = adaptive_refinement(circles, max_iter=30)
    stage1_fitness = calculate_fitness_with_spatial_indexing(coarse_solution)

    # Stage 2: Enhanced refinement focusing on specific problem areas
    enhanced_solution = adaptive_refinement(coarse_solution, max_iter=50)
    stage2_fitness = calculate_fitness_with_spatial_indexing(enhanced_solution)

    # Use the best of the two stages
    best_solution = enhanced_solution if stage2_fitness > stage1_fitness else coarse_solution
    best_fitness = max(stage1_fitness, stage2_fitness)

    # Stage 3: Global optimization using a simplified evolutionary approach with enhanced selection
    # Only optimize positions and radii of the most constrained circles
    def selective_evolution():
        # Identify most constrained circles (those with tight space or many neighbors)
        constrained_circles = []
        points = best_solution[:, :2]
        tree = KDTree(points)

        # Calculate constraint score for each circle (number of neighbors)
        constraint_scores = []
        for i in range(n):
            cx, cy, r = best_solution[i]
            neighbor_indices = tree.query_ball_point([cx, cy], 2*(r + 0.01) + 0.001)
            constraint_scores.append((i, len(neighbor_indices)))

        # Sort by constraint score (highly constrained first)
        constraint_scores.sort(key=lambda x: x[1], reverse=True)
        constrained_circles = [idx for idx, _ in constraint_scores[:min(8, n)]]

        # If we have few constrained circles, add some random ones for diversity
        if len(constrained_circles) < 5:
            random_candidates = list(set(range(n)) - set(constrained_circles))
            additional = random.sample(random_candidates, min(5 - len(constrained_circles), len(random_candidates)))
            constrained_circles.extend(additional)

        # Create a simpler evolutionary approach that focuses on these
        population_size = 25
        generations = 25
        population = [best_solution.copy()]

        # Add diverse individuals around current solution
        for _ in range(population_size - 1):
            individual = best_solution.copy()
            # Perturb a few selected circles
            for idx in random.sample(constrained_circles, min(6, len(constrained_circles))):
                # Position perturbation with adaptive scale
                individual[idx, 0] += np.random.normal(0, 0.015)
                individual[idx, 1] += np.random.normal(0, 0.015)
                # Radius adjustment with larger range
                individual[idx, 2] *= np.random.uniform(0.8, 1.2)

                # Clamp values
                individual[idx, 0] = np.clip(individual[idx, 0], 0.05, rect_width - 0.05)
                individual[idx, 1] = np.clip(individual[idx, 1], 0.05, rect_height - 0.05)
                individual[idx, 2] = max(0.001, individual[idx, 2])

            population.append(individual)

        # Evolutionary process
        for gen in range(generations):
            fitnesses = [calculate_fitness_with_spatial_indexing(ind) for ind in population]
            best_idx = np.argmax(fitnesses)

            if fitnesses[best_idx] > best_fitness:
                best_fitness = fitnesses[best_idx]
                best_solution = population[best_idx].copy()

            # Selection: keep top 60%
            sorted_indices = np.argsort(fitnesses)[::-1][:int(population_size * 0.6)]
            selected_population = [population[i] for i in sorted_indices]

            # Create offspring with enhanced crossover
            new_population = [selected_population[0]]  # Elitism

            while len(new_population) < population_size:
                parent1 = random.choice(selected_population)
                parent2 = random.choice(selected_population)

                # Enhanced crossover - blend properties with some randomness
                child = parent1.copy()

                # Crossover: mix positions and radii
                crossover_point = np.random.randint(1, n)
                child[crossover_point:, 0] = parent2[crossover_point:, 0]  # x positions
                child[crossover_point:, 1] = parent2[crossover_point:, 1]  # y positions
                child[crossover_point:, 2] = parent2[crossover_point:, 2]  # radii

                # Mutation with increased diversity
                for i in range(n):
                    if random.random() < 0.15:  # Higher mutation rate
                        if random.random() < 0.6:  # Position mutation (more likely)
                            child[i, 0] += np.random.normal(0, 0.015)
                            child[i, 1] += np.random.normal(0, 0.015)
                            child[i, 0] = np.clip(child[i, 0], 0.05, rect_width - 0.05)
                            child[i, 1] = np.clip(child[i, 1], 0.05, rect_height - 0.05)
                        else:  # Radius mutation
                            child[i, 2] *= np.random.uniform(0.85, 1.15)
                            child[i, 2] = max(0.001, child[i, 2])

                new_population.append(child)

            population = new_population[:population_size]

        return best_solution

    # Run selective evolutionary optimization
    try:
        evolved_solution = selective_evolution()
        final_fitness = calculate_fitness_with_spatial_indexing(evolved_solution)

        if final_fitness > best_fitness:
            best_solution = evolved_solution
            best_fitness = final_fitness

    except Exception:
        # If evolution fails, proceed with current best
        pass

    # Final fine-tuning
    final_solution = adaptive_refinement(best_solution, max_iter=40)
    final_fitness = calculate_fitness_with_spatial_indexing(final_solution)

    if final_fitness > best_fitness:
        best_solution = final_solution

    # Ensure all constraints are satisfied one final time
    final_points = best_solution[:, :2]
    final_tree = KDTree(final_points)

    # Do a final validation pass
    valid = True
    for i in range(n):
        cx, cy, r = best_solution[i]

        # Check boundary constraints
        if cx - r < 0.01 or cx + r > rect_width - 0.01 or \
           cy - r < 0.01 or cy + r > rect_height - 0.01:
            valid = False
            break

        # Check overlap constraints
        neighbor_indices = final_tree.query_ball_point([cx, cy], 2*(r + 0.01) + 0.001)
        for j in neighbor_indices:
            if i != j:
                other_cx, other_cy, other_r = best_solution[j]
                dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                if dist < r + other_r:
                    valid = False
                    break

    # If validation fails, do one last adaptive refinement
    if not valid:
        best_solution = adaptive_refinement(best_solution, max_iter=30)

    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")