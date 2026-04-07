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

    # Generate adaptive hexagonal grid initial placement with better packing density
    def generate_adaptive_hexagonal_grid(num_circles, width, height):
        # Use a more sophisticated approach for optimal packing
        # Target hexagonal close packing density
        target_density = np.pi / (2 * np.sqrt(3))  # ~0.9069

        # Estimate area needed for all circles with higher packing efficiency
        container_area = width * height
        # Use 90% of container area for circle packing to leave some margin
        estimated_circle_area = container_area * 0.9 / num_circles

        # From circle area, estimate radius
        avg_radius = np.sqrt(estimated_circle_area / np.pi)

        # Determine grid dimensions that work well for hexagonal packing
        # For 21 circles, try to get a configuration that fills the container effectively
        aspect_ratio = width / height

        # More systematic approach to grid sizing
        cols = int(np.ceil(np.sqrt(num_circles * aspect_ratio)))
        rows = int(np.ceil(num_circles / cols))

        # Ensure reasonable dimensions
        cols = max(1, min(cols, int(width / (2 * avg_radius * 0.9))))
        rows = max(1, min(rows, int(height / (2 * avg_radius * 0.9))))

        # Adjust if we're underfilling
        if cols * rows < num_circles:
            cols = max(1, int(np.ceil(np.sqrt(num_circles * aspect_ratio))))
            rows = max(1, int(np.ceil(num_circles / cols)))

        # Recalculate actual cell sizes
        cell_width = width / cols * 0.95 if cols > 0 else width
        cell_height = height / rows * 0.95 if rows > 0 else height

        # Hexagonal packing spacing
        spacing_x = cell_width
        spacing_y = cell_height * np.sqrt(3) / 2

        # Ensure we don't exceed container bounds
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
                    # Start with a carefully calculated radius
                    r = min(avg_radius * 0.9, spacing_x / 4, spacing_y / 4)
                    r = max(0.005, min(r, 0.1))  # Bound radius
                    circles.append([x, y, r])
                    circle_count += 1
            if circle_count >= num_circles:
                break

        # Fill remaining spots with strategic random positions
        while len(circles) < num_circles:
            # Place random circles with preference for center area where more expansion is possible
            x = np.random.uniform(0.1, width - 0.1)
            y = np.random.uniform(0.1, height - 0.1)
            # Use a radius that's a bit smaller than average to leave room for growth
            r = np.random.uniform(0.01, min(0.05, avg_radius * 0.7))
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

        # Additional penalty for numerical stability
        constraint_penalty = 0
        for i in range(n):
            cx, cy, r = circles_array[i]
            # Penalize very small radii that might cause numerical issues
            if r < 0.001:
                constraint_penalty += 1000000 * (0.001 - r)**2

        return total_radius - penalty - constraint_penalty

    # Fast local refinement with enhanced greedy improvement and early termination
    def adaptive_refinement(circles_array, max_iter=80):
        best_circles = circles_array.copy()
        best_fitness = calculate_fitness_with_spatial_indexing(best_circles)

        # Track recent improvements for better search direction
        improvement_history = deque(maxlen=10)

        for iteration in range(max_iter):
            improved = False

            # Select circles in a way that focuses on promising candidates
            # First prioritize circles that are near boundaries or have tight constraints
            candidates = list(range(n))

            # Shuffle initially but bias towards circles that could benefit most
            random.shuffle(candidates)

            # Try more aggressive improvement attempts
            processed_count = 0
            for i in candidates:
                if processed_count > 10:  # Limit per iteration
                    break
                processed_count += 1

                cx, cy, r = best_circles[i]

                # Compute max allowable radius more efficiently
                max_radius = float('inf')

                # Boundary constraints
                max_radius = min(max_radius, cx - 0.01)
                max_radius = min(max_radius, rect_width - cx - 0.01)
                max_radius = min(max_radius, cy - 0.01)
                max_radius = min(max_radius, rect_height - cy - 0.01)

                # Overlap constraints using spatial indexing
                points = best_circles[:, :2]
                tree = KDTree(points)
                neighbor_indices = tree.query_ball_point([cx, cy], 2*(r + 0.01) + 0.001)

                # Check overlaps with neighbors more carefully
                for j in neighbor_indices:
                    if i != j:
                        other_cx, other_cy, other_r = best_circles[j]
                        dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                        max_radius = min(max_radius, dist - other_r - 0.001)

                # Try to increase radius more aggressively with different strategies
                if max_radius > r and max_radius > 0.001:
                    # Try different increment strategies
                    increments_to_try = [
                        0.001, 0.002, 0.005, 0.01, 0.02, 0.03
                    ]

                    for incr in increments_to_try:
                        new_r = min(r + incr, max_radius)
                        if new_r <= r:
                            continue

                        # Quick validation - check just immediate neighbors
                        temp_circles = best_circles.copy()
                        temp_circles[i, 2] = new_r

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
                                break  # Early exit on first success for this circle

                if improved:
                    improvement_history.append(True)
                else:
                    improvement_history.append(False)

            # Stop early if we haven't improved in recent iterations
            if len(improvement_history) == 10 and all(not x for x in list(improvement_history)[-5:]):
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

    # Stage 3: Enhanced global optimization with two-phase evolutionary approach
    def selective_evolution():
        # Identify most constrained circles (those with tight space)
        constrained_circles = []
        points = best_solution[:, :2]
        tree = KDTree(points)

        for i in range(n):
            cx, cy, r = best_solution[i]
            neighbor_indices = tree.query_ball_point([cx, cy], 2*(r + 0.01) + 0.001)
            if len(neighbor_indices) > 2:  # More aggressive constraint detection
                constrained_circles.append(i)

        # If we have few constrained circles, pick more strategically
        if len(constrained_circles) < 5:
            # Pick circles that are at edges or have small radii
            edge_candidates = []
            for i in range(n):
                cx, cy, r = best_solution[i]
                if (cx < 0.15 or cx > rect_width - 0.15 or
                    cy < 0.15 or cy > rect_height - 0.15):
                    edge_candidates.append(i)
                if r < 0.03:  # Small radii are often constrained
                    edge_candidates.append(i)
            constrained_circles = random.sample(edge_candidates,
                                               min(8, len(edge_candidates))) if edge_candidates else list(range(n))[:8]

        # Two-phase evolutionary approach: position-first, then radius
        population_size = 25
        generations = 25

        # Phase 1: Position-focused evolution (more aggressive)
        population = [best_solution.copy()]

        # Add diverse individuals with position mutations
        for _ in range(population_size - 1):
            individual = best_solution.copy()
            # Mutate positions more aggressively
            for idx in random.sample(constrained_circles, min(6, len(constrained_circles))):
                individual[idx, 0] += np.random.normal(0, 0.03)  # Larger mutation
                individual[idx, 1] += np.random.normal(0, 0.03)
                # Keep radius same for now
                individual[idx, 0] = np.clip(individual[idx, 0], 0.05, rect_width - 0.05)
                individual[idx, 1] = np.clip(individual[idx, 1], 0.05, rect_height - 0.05)
            population.append(individual)

        # Evolve positions first
        for gen in range(15):  # Fewer generations for position phase
            fitnesses = [calculate_fitness_with_spatial_indexing(ind) for ind in population]
            best_idx = np.argmax(fitnesses)

            if fitnesses[best_idx] > best_fitness:
                best_fitness = fitnesses[best_idx]
                best_solution = population[best_idx].copy()

            # Selection - keep top performers
            sorted_indices = np.argsort(fitnesses)[::-1][:population_size//2]
            selected_population = [population[i] for i in sorted_indices]

            # Create offspring with crossover
            new_population = [selected_population[0]]  # Elitism

            while len(new_population) < population_size:
                parent1 = random.choice(selected_population)
                parent2 = random.choice(selected_population)

                # Crossover with bias toward parent1
                child = parent1.copy()
                crossover_point = np.random.randint(1, n)
                child[crossover_point:, :2] = parent2[crossover_point:, :2]  # Positions only

                # Mutation with different rates
                for i in range(n):
                    if random.random() < 0.15:  # Higher mutation rate for position
                        child[i, 0] += np.random.normal(0, 0.015)
                        child[i, 1] += np.random.normal(0, 0.015)
                        child[i, 0] = np.clip(child[i, 0], 0.05, rect_width - 0.05)
                        child[i, 1] = np.clip(child[i, 1], 0.05, rect_height - 0.05)

                new_population.append(child)

            population = new_population[:population_size]

        # Phase 2: Radius-focused evolution (fine-tune)
        # Start with current best solution and evolve radii
        population = [best_solution.copy()]

        # Add diverse individuals with radius mutations
        for _ in range(population_size - 1):
            individual = best_solution.copy()
            # Mutate radii more aggressively
            for idx in random.sample(constrained_circles, min(6, len(constrained_circles))):
                individual[idx, 2] *= np.random.uniform(0.8, 1.2)  # Wider range
                individual[idx, 2] = max(0.001, individual[idx, 2])
            population.append(individual)

        # Evolve radii
        for gen in range(10):  # Fewer gens for radius phase
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

                # Crossover for radii
                child = parent1.copy()
                crossover_point = np.random.randint(1, n)
                child[crossover_point:, 2] = parent2[crossover_point:, 2]  # Radii only

                # Mutation for radii
                for i in range(n):
                    if random.random() < 0.2:  # Higher mutation rate for radius
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