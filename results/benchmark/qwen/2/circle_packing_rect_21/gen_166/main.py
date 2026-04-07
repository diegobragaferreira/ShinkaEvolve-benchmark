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

    # Generate adaptive hexagonal grid initial placement
    def generate_adaptive_hexagonal_grid(num_circles, width, height):
        # Use a more intelligent approach based on packing density and container dimensions
        # Calculate optimal grid dimensions for maximum packing efficiency
        # For 21 circles, we'll compute the ideal rectangular grid that approximates hexagonal packing

        # Target packing density for circles in a plane (hexagonal close packing)
        target_density = np.pi / (2 * np.sqrt(3))  # ~0.9069

        # Estimate area needed for all circles
        # We'll use a heuristic: total area needed for 21 circles
        # with radius such that they fill about 80% of the container area
        container_area = width * height
        estimated_circle_area = container_area * 0.8 / num_circles

        # From circle area, estimate radius
        avg_radius = np.sqrt(estimated_circle_area / np.pi)

        # Determine grid dimensions
        # Try to balance rows and columns to match container aspect ratio
        aspect_ratio = width / height

        # Use a more adaptive approach:
        # For 21 circles, a 5x5 grid is often effective with hexagonal offsetting
        ideal_cols = max(1, int(np.sqrt(num_circles * aspect_ratio)))
        ideal_rows = max(1, int(np.ceil(num_circles / ideal_cols)))

        # Ensure we don't exceed container bounds
        max_cols = max(1, int(width / (2 * avg_radius)))
        max_rows = max(1, int(height / (2 * avg_radius)))

        # Use the minimum of ideal and max to prevent overfilling
        cols = min(ideal_cols, max_cols)
        rows = min(ideal_rows, max_rows)

        # If too few columns or rows, adjust to maintain reasonable packing
        if cols * rows < num_circles:
            cols = max(1, int(np.ceil(np.sqrt(num_circles * aspect_ratio))))
            rows = max(1, int(np.ceil(num_circles / cols)))

        # Recalculate based on actual container dimensions
        cell_width = width / cols * 0.9 if cols > 0 else width
        cell_height = height / rows * 0.9 if rows > 0 else height

        # For hexagonal packing, adjust spacing
        spacing_x = cell_width
        spacing_y = cell_height * np.sqrt(3) / 2

        # Ensure spacing doesn't exceed container bounds
        spacing_x = min(spacing_x, width * 0.95)
        spacing_y = min(spacing_y, height * 0.95)

        circles = []
        y_offset = 0.1
        x_offset = 0.1

        circle_count = 0

        # Create hexagonal grid with proper offsetting
        for i in range(rows):
            y = y_offset + i * spacing_y
            x_start = x_offset + (i % 2) * spacing_x / 2
            for j in range(cols):
                if circle_count >= num_circles:
                    break
                x = x_start + j * spacing_x
                if x < width - 0.1 and y < height - 0.1:
                    # Start with an adaptive radius based on the actual computed spacing
                    r = min(avg_radius, spacing_x / 4, spacing_y / 4)
                    # Make radius slightly smaller to allow for better optimization later
                    r = min(r, 0.1)
                    circles.append([x, y, r])
                    circle_count += 1
            if circle_count >= num_circles:
                break

        # Fill remaining spots with strategic random positions that consider the overall packing
        while len(circles) < num_circles:
            # Place random circles in the center region to have more room for expansion
            x = np.random.uniform(0.15 * width, 0.85 * width)
            y = np.random.uniform(0.15 * height, 0.85 * height)
            # Use a smaller radius for random placements to ensure they'll fit
            r = np.random.uniform(0.015, min(0.04, avg_radius * 0.8))
            circles.append([x, y, r])

        return np.array(circles)

    # Initial configuration
    circles = generate_adaptive_hexagonal_grid(n, rect_width, rect_height)

    # Efficient constraint validation with spatial indexing
    def calculate_fitness_with_spatial_indexing(circles_array):
        total_radius = np.sum(circles_array[:, 2])

        penalty = 0

        # Boundary penalties (quadratic)
        for i in range(n):
            cx, cy, r = circles_array[i]
            if cx - r < 0.01:
                penalty += 10000 * (r - cx)**2
            if cx + r > rect_width - 0.01:
                penalty += 10000 * (cx + r - rect_width)**2
            if cy - r < 0.01:
                penalty += 10000 * (r - cy)**2
            if cy + r > rect_height - 0.01:
                penalty += 10000 * (cy + r - rect_height)**2

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
                        penalty += 100000 * overlap**2

        return total_radius - penalty

    # Fast local refinement focused on improvement regions
    def adaptive_refinement(circles_array, max_iter=80):
        best_circles = circles_array.copy()
        best_fitness = calculate_fitness_with_spatial_indexing(best_circles)

        # Track circles that have been modified recently
        recent_updates = set()

        for iteration in range(max_iter):
            improved = False

            # Select circles to try improving, prioritizing those with high overlap risk
            candidates = list(range(n))
            # Shuffle for variety
            random.shuffle(candidates)

            # Process a subset of candidates each iteration to maintain speed
            processed_count = 0

            for i in candidates:
                if processed_count > 8:  # Limit per iteration
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

                # Try to increase radius in multiple steps
                if max_radius > r and max_radius > 0.001:
                    # Binary search approach for more precise improvement
                    low = r
                    high = min(max_radius, r + 0.05)  # Cap increment
                    best_new_r = r

                    # Test a few key increments
                    test_increments = [0.002, 0.005, 0.01, 0.02]
                    for incr in test_increments:
                        new_r = min(r + incr, high)
                        if new_r <= r:
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
                                best_new_r = new_r
                                improved = True

                        if improved:
                            break

                if improved:
                    recent_updates.add(i)
                    if len(recent_updates) > 10:
                        recent_updates.pop()

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

    # Stage 3: Global optimization using a simplified evolutionary approach
    # Only optimize positions and radii of the most constrained circles
    def selective_evolution():
        # Identify most constrained circles (those with tight space or many neighbors)
        constrained_circles = []
        points = best_solution[:, :2]
        tree = KDTree(points)

        for i in range(n):
            cx, cy, r = best_solution[i]
            neighbor_indices = tree.query_ball_point([cx, cy], 2*(r + 0.01) + 0.001)
            if len(neighbor_indices) > 3:  # Circle surrounded by many others
                constrained_circles.append(i)

        # If we have few constrained circles, pick randomly
        if len(constrained_circles) < 5:
            constrained_circles = random.sample(list(range(n)), min(10, n))

        # Create a simpler evolutionary approach that focuses on these
        population_size = 20
        generations = 20
        population = [best_solution.copy()]

        # Add diverse individuals around current solution
        for _ in range(population_size - 1):
            individual = best_solution.copy()
            # Perturb a few selected circles
            for idx in random.sample(constrained_circles, min(5, len(constrained_circles))):
                individual[idx, 0] += np.random.normal(0, 0.02)
                individual[idx, 1] += np.random.normal(0, 0.02)
                individual[idx, 2] *= np.random.uniform(0.85, 1.15)

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

            # Selection
            sorted_indices = np.argsort(fitnesses)[::-1][:population_size//2]
            selected_population = [population[i] for i in sorted_indices]

            # Create offspring
            new_population = [selected_population[0]]  # Elitism

            while len(new_population) < population_size:
                parent1 = random.choice(selected_population)
                parent2 = random.choice(selected_population)

                # Simple crossover - swap segments
                child = parent1.copy()
                crossover_point = np.random.randint(1, n)
                child[crossover_point:, 0] = parent2[crossover_point:, 0]  # x positions
                child[crossover_point:, 1] = parent2[crossover_point:, 1]  # y positions
                child[crossover_point:, 2] = parent2[crossover_point:, 2]  # radii

                # Mutation
                for i in range(n):
                    if random.random() < 0.1:
                        if random.random() < 0.7:  # Position mutation
                            child[i, 0] += np.random.normal(0, 0.01)
                            child[i, 1] += np.random.normal(0, 0.01)
                            child[i, 0] = np.clip(child[i, 0], 0.05, rect_width - 0.05)
                            child[i, 1] = np.clip(child[i, 1], 0.05, rect_height - 0.05)
                        else:  # Radius mutation
                            child[i, 2] *= np.random.uniform(0.9, 1.1)
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
