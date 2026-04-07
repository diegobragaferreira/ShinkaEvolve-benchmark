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

    # Generate adaptive initial placement using a more sophisticated approach
    def generate_optimized_initial_placement(num_circles, width, height):
        # Calculate theoretical maximum packing density for circles in rectangle
        # Using hexagonal packing approximation: ~0.9069 for infinite plane
        # But for finite rectangles we need to account for boundary effects

        # Estimate initial circle radius based on how much space we have
        # We'll try to use about 70% of the available area for circles
        total_area = width * height
        estimated_circle_area = total_area * 0.7 / num_circles

        # From π*r² = estimated_circle_area, solve for r
        estimated_radius = np.sqrt(estimated_circle_area / np.pi)

        # Create a more uniform distribution - start with triangular lattice pattern
        # This gives better initial distribution than simple hexagonal grid

        # For better results with 21 circles, let's use a more intelligent approach
        # Calculate rows and columns based on golden ratio for visual balance
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        cols = int(np.ceil(np.sqrt(num_circles * phi)))
        rows = int(np.ceil(num_circles / cols))

        # Adjust spacing to fit container better
        cell_width = width / cols * 0.9  # Slight margin
        cell_height = height / rows * 0.9

        # For triangular/hexagonal packing
        spacing_x = cell_width
        spacing_y = cell_height * np.sqrt(3) / 2

        # Ensure we have enough space
        if spacing_x <= 0 or spacing_y <= 0:
            spacing_x = width / 5
            spacing_y = height / 5

        circles = []
        y_offset = 0.05
        x_offset = 0.05

        circle_count = 0

        # Create triangular lattice pattern with offset rows
        for i in range(rows):
            y = y_offset + i * spacing_y
            x_start = x_offset + (i % 2) * spacing_x / 2
            for j in range(cols):
                if circle_count >= num_circles:
                    break
                x = x_start + j * spacing_x

                # Check if within bounds
                if x >= width - 0.05 or y >= height - 0.05:
                    continue

                # Set initial radius to be proportional to spacing
                r = min(estimated_radius, spacing_x/3, spacing_y/3)
                if r < 0.001:
                    r = 0.01

                circles.append([x, y, r])
                circle_count += 1

            if circle_count >= num_circles:
                break

        # If we didn't reach enough circles, add random ones
        while len(circles) < num_circles:
            x = np.random.uniform(0.05, width - 0.05)
            y = np.random.uniform(0.05, height - 0.05)
            # Use a smaller random radius
            r = np.random.uniform(0.005, estimated_radius * 0.8)
            circles.append([x, y, r])

        return np.array(circles)

    # Initial configuration
    circles = generate_optimized_initial_placement(n, rect_width, rect_height)

    # Enhanced constraint validation with constraint violation awareness
    def calculate_enhanced_fitness(circles_array, use_violation_penalty=True):
        total_radius = np.sum(circles_array[:, 2])

        penalty = 0
        violation_count = 0

        # Boundary constraints (stronger penalties for being very out of bounds)
        for i in range(n):
            cx, cy, r = circles_array[i]

            # Calculate how much we violate bounds
            left_violation = max(0, r - cx)
            right_violation = max(0, cx + r - rect_width)
            bottom_violation = max(0, r - cy)
            top_violation = max(0, cy + r - rect_height)

            if left_violation > 0 or right_violation > 0 or bottom_violation > 0 or top_violation > 0:
                violation_count += 1
                # Quadratic penalty for boundary violations, with higher weight for severe violations
                penalty += 50000 * (left_violation**2 + right_violation**2 +
                                  bottom_violation**2 + top_violation**2)

        # Overlap constraints using spatial indexing for efficiency
        points = circles_array[:, :2]
        tree = KDTree(points)

        # Query for all possible overlapping pairs efficiently
        # This reduces from O(n^2) to near O(n log n) for overlap checking
        for i in range(n):
            cx, cy, r = circles_array[i]

            # Find neighbors within 2*(r + safety_margin) distance
            # Use a slightly larger threshold to catch potential violations
            neighbor_indices = tree.query_ball_point([cx, cy], 2*(r + 0.015) + 0.001)

            for j in neighbor_indices:
                if i != j:
                    other_cx, other_cy, other_r = circles_array[j]
                    dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                    overlap = (r + other_r) - dist

                    if overlap > 0:
                        violation_count += 1
                        # Use a stronger penalty for overlaps (they're critical)
                        penalty += 200000 * overlap**2

        # Add a penalty term based on constraint violations to encourage feasible solutions
        if use_violation_penalty and violation_count > 0:
            penalty += 10000 * violation_count

        return total_radius - penalty, violation_count

    # Enhanced fitness function for better optimization guidance
    def calculate_fitness_with_spatial_indexing(circles_array):
        return calculate_enhanced_fitness(circles_array, use_violation_penalty=True)[0]

    # Enhanced local refinement with early termination and more aggressive improvement strategies
    def enhanced_local_refinement(circles_array, max_iter=80):
        best_circles = circles_array.copy()
        best_fitness, _ = calculate_enhanced_fitness(best_circles)

        # Track circles that have been modified recently
        recent_updates = set()

        for iteration in range(max_iter):
            improved = False

            # Early termination condition: if no improvement in last few iterations
            if iteration > 10 and not improved:
                break

            # Select circles to try improving, with priority on those that might be underutilized
            candidates = list(range(n))
            # Shuffle for variety, but prioritize circles with less overlap initially
            random.shuffle(candidates)

            # Process a subset of candidates each iteration to maintain speed
            processed_count = 0

            for i in candidates:
                if processed_count > 10:  # Limit per iteration
                    break
                processed_count += 1

                cx, cy, r = best_circles[i]

                # Compute max allowable radius
                max_radius = float('inf')

                # Boundary constraints - stricter margins
                max_radius = min(max_radius, cx - 0.005)
                max_radius = min(max_radius, rect_width - cx - 0.005)
                max_radius = min(max_radius, cy - 0.005)
                max_radius = min(max_radius, rect_height - cy - 0.005)

                # Overlap constraints using spatial indexing on current neighbors
                points = best_circles[:, :2]
                tree = KDTree(points)
                # Widen the query slightly for more thorough checking
                neighbor_indices = tree.query_ball_point([cx, cy], 2*(r + 0.015) + 0.001)

                for j in neighbor_indices:
                    if i != j:
                        other_cx, other_cy, other_r = best_circles[j]
                        dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                        max_radius = min(max_radius, dist - other_r - 0.001)

                # Try to increase radius with multiple approaches
                if max_radius > r and max_radius > 0.001:
                    # First try small increments to find quick wins
                    small_increments = [0.001, 0.002, 0.003, 0.005]
                    found_improvement = False

                    for incr in small_increments:
                        new_r = min(r + incr, max_radius)
                        if new_r <= r:
                            continue

                        # Quick validation with spatial indexing for potential overlap
                        temp_circles = best_circles.copy()
                        temp_circles[i, 2] = new_r

                        # Check validity using spatial indexing - only check neighbors
                        valid = True
                        temp_points = temp_circles[:, :2]
                        temp_tree = KDTree(temp_points)

                        # Check only neighbors of this circle, not all pairs
                        temp_neighbor_indices = temp_tree.query_ball_point([cx, cy], 2*(new_r + 0.015) + 0.001)

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
                            test_fitness, _ = calculate_enhanced_fitness(test_circles)

                            if test_fitness > best_fitness:
                                best_fitness = test_fitness
                                best_circles = test_circles
                                improved = True
                                found_improvement = True
                                break

                    # If no improvement with small steps, try larger ones
                    if not found_improvement:
                        large_increment = min(0.01, max_radius - r)
                        if large_increment > 0.001:
                            new_r = min(r + large_increment, max_radius)
                            temp_circles = best_circles.copy()
                            temp_circles[i, 2] = new_r

                            valid = True
                            temp_points = temp_circles[:, :2]
                            temp_tree = KDTree(temp_points)
                            temp_neighbor_indices = temp_tree.query_ball_point([cx, cy], 2*(new_r + 0.015) + 0.001)

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
                                test_fitness, _ = calculate_enhanced_fitness(test_circles)

                                if test_fitness > best_fitness:
                                    best_fitness = test_fitness
                                    best_circles = test_circles
                                    improved = True

                if improved:
                    recent_updates.add(i)
                    if len(recent_updates) > 10:
                        recent_updates.pop()

            # If no improvement, reduce search scope or stop
            if not improved:
                # Reduce the iteration count to terminate earlier if needed
                pass

        return best_circles

    # Multi-stage optimization approach
    # Stage 1: Coarse refinement with basic local search
    coarse_solution = enhanced_local_refinement(circles, max_iter=30)
    stage1_fitness, _ = calculate_enhanced_fitness(coarse_solution)

    # Stage 2: Enhanced refinement focusing on specific problem areas
    enhanced_solution = enhanced_local_refinement(coarse_solution, max_iter=50)
    stage2_fitness, _ = calculate_enhanced_fitness(enhanced_solution)

    # Use the best of the two stages
    best_solution = enhanced_solution if stage2_fitness > stage1_fitness else coarse_solution
    best_fitness, _ = max(stage1_fitness, stage2_fitness)

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

                # Simple crossover
                child = parent1.copy()
                for i in range(n):
                    if random.random() < 0.3:
                        child[i, 0] = parent2[i, 0]
                        child[i, 1] = parent2[i, 1]
                        child[i, 2] = parent2[i, 2]

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
        final_fitness, _ = calculate_enhanced_fitness(evolved_solution)

        if final_fitness > best_fitness:
            best_solution = evolved_solution
            best_fitness = final_fitness

    except Exception:
        # If evolution fails, proceed with current best
        pass

    # Final fine-tuning with enhanced refinement
    final_solution = enhanced_local_refinement(best_solution, max_iter=40)
    final_fitness, _ = calculate_enhanced_fitness(final_solution)

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