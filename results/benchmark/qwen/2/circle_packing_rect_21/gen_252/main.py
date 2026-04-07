# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import random
import time

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

    # Hybrid initialization combining hexagonal packing with strategic random placement
    def generate_hybrid_initialization(num_circles, width, height):
        circles = []
        
        # Phase 1: Hexagonal grid initialization
        avg_radius = min(width, height) * 0.08  # Estimate based on container size
        spacing = 2 * avg_radius * 0.9  # Slight safety margin
        
        # Hexagonal grid dimensions
        rows = int(np.ceil(np.sqrt(num_circles) * 1.1))
        cols = int(np.ceil(num_circles / rows))
        
        # Adjust grid to container bounds
        grid_width = cols * spacing
        grid_height = rows * spacing * np.sqrt(3) / 2
        
        # Scale grid if necessary
        if grid_width > width or grid_height > height:
            scale_factor = min(width / grid_width, height / grid_height)
            spacing *= scale_factor
        
        # Create hexagonal grid
        y_offset = spacing * 0.5
        x_offset = spacing * 0.5
        
        circle_count = 0
        for i in range(rows):
            y = y_offset + i * spacing * np.sqrt(3) / 2
            x_start = x_offset + (i % 2) * spacing / 2
            for j in range(cols):
                if circle_count >= num_circles:
                    break
                x = x_start + j * spacing
                if x < width - spacing * 0.5 and y < height - spacing * 0.5:
                    r = min(avg_radius, spacing * 0.4)
                    circles.append([x, y, r])
                    circle_count += 1
            if circle_count >= num_circles:
                break
        
        # Phase 2: Fill remaining spots with random placement
        while len(circles) < num_circles:
            x = np.random.uniform(spacing * 0.5, width - spacing * 0.5)
            y = np.random.uniform(spacing * 0.5, height - spacing * 0.5)
            # Use a radius that allows for easy expansion
            r = np.random.uniform(avg_radius * 0.3, avg_radius * 0.7)
            circles.append([x, y, r])

        return np.array(circles)

    # Initial configuration with physics-based pre-processing
    circles = generate_hybrid_initialization(n, rect_width, rect_height)
    
    # Physics-based initial refinement to quickly eliminate obvious overlaps
    def physics_preprocess(circles_array, width, height, iterations=100):
        positions = circles_array[:, :2]
        radii = circles_array[:, 2]
        
        # Distance matrix for quick overlap checking
        for _ in range(iterations):
            # Simple force-based repulsion (similar to the second approach)
            forces = np.zeros_like(positions)
            
            # Compute forces between all pairs
            distances = cdist(positions, positions)
            
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist > 0:
                        if dist < (radii[i] + radii[j]):  # Overlapping
                            force_magnitude = 1000.0 / (dist * dist)
                        else:  # Not overlapping, weak repulsion
                            force_magnitude = 0.1 / (dist * dist)
                        
                        fx = force_magnitude * dx / dist
                        fy = force_magnitude * dy / dist
                        
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
            
            # Apply boundary forces
            for i in range(len(positions)):
                r = radii[i]
                if positions[i, 0] - r < 0:
                    forces[i, 0] += 500 * (0 - (positions[i, 0] - r))
                if positions[i, 0] + r > width:
                    forces[i, 0] -= 500 * ((positions[i, 0] + r) - width)
                if positions[i, 1] - r < 0:
                    forces[i, 1] += 500 * (0 - (positions[i, 1] - r))
                if positions[i, 1] + r > height:
                    forces[i, 1] -= 500 * ((positions[i, 1] + r) - height)
            
            # Update positions
            dt = 0.01
            for i in range(len(positions)):
                # Limit velocity
                max_velocity = 0.05
                forces[i, 0] = np.clip(forces[i, 0], -max_velocity, max_velocity)
                forces[i, 1] = np.clip(forces[i, 1], -max_velocity, max_velocity)
                
                positions[i, 0] += forces[i, 0] * dt
                positions[i, 1] += forces[i, 1] * dt
                
                # Apply boundary constraints
                positions[i, 0] = np.clip(positions[i, 0], r, width - r)
                positions[i, 1] = np.clip(positions[i, 1], r, height - r)
        
        # Update circles array with new positions
        circles_array[:, :2] = positions
        return circles_array

    # Apply physics-based preprocessing
    circles = physics_preprocess(circles, rect_width, rect_height, 50)

    # Efficient constraint validation with spatial indexing
    def calculate_fitness_with_spatial_indexing(circles_array):
        total_radius = np.sum(circles_array[:, 2])

        penalty = 0

        # Boundary penalties with stronger scaling
        for i in range(n):
            cx, cy, r = circles_array[i]
            boundary_violation = 0
            
            if cx - r < 0.005:
                boundary_violation += (r - cx)**2
            if cx + r > rect_width - 0.005:
                boundary_violation += (cx + r - rect_width)**2
            if cy - r < 0.005:
                boundary_violation += (r - cy)**2
            if cy + r > rect_height - 0.005:
                boundary_violation += (cy + r - rect_height)**2

            if boundary_violation > 0:
                penalty += 50000 * boundary_violation

        # Overlap penalties using spatial indexing
        points = circles_array[:, :2]
        tree = KDTree(points)

        for i in range(n):
            cx, cy, r = circles_array[i]
            neighbor_indices = tree.query_ball_point([cx, cy], 2*(r + 0.005))

            for j in neighbor_indices:
                if i != j:
                    other_cx, other_cy, other_r = circles_array[j]
                    dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                    overlap = (r + other_r) - dist

                    if overlap > 0:
                        penalty += 500000 * overlap**2

        # Add penalty for numerically unstable configurations
        constraint_penalty = 0
        for i in range(n):
            r = circles_array[i, 2]
            if r < 0.0005:
                constraint_penalty += 1000000 * (0.0005 - r)**2

        return total_radius - penalty - constraint_penalty

    # Fast local refinement with binary search for optimal radius increases
    def adaptive_refinement(circles_array, max_iter=60):
        best_circles = circles_array.copy()
        best_fitness = calculate_fitness_with_spatial_indexing(best_circles)

        for iteration in range(max_iter):
            improved = False
            # Process circles in shuffled order for better exploration
            indices = list(range(n))
            random.shuffle(indices)
            
            # Limit number of circles processed per iteration
            processed_count = 0
            
            for i in indices:
                if processed_count >= 10:  # Limit per iteration
                    break
                processed_count += 1
                
                cx, cy, r = best_circles[i]

                # Compute max allowable radius
                max_radius = float('inf')

                # Boundary constraints
                max_radius = min(max_radius, cx - 0.005)
                max_radius = min(max_radius, rect_width - cx - 0.005)
                max_radius = min(max_radius, cy - 0.005)
                max_radius = min(max_radius, rect_height - cy - 0.005)

                # Overlap constraints using spatial indexing
                points = best_circles[:, :2]
                tree = KDTree(points)
                neighbor_indices = tree.query_ball_point([cx, cy], 2*(r + 0.005))

                for j in neighbor_indices:
                    if i != j:
                        other_cx, other_cy, other_r = best_circles[j]
                        dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                        max_radius = min(max_radius, dist - other_r - 0.0005)

                # Try increasing radius with binary search approach for precision
                if max_radius > r and max_radius > 0.0005:
                    # Test with different increment strategies
                    increments = [0.001, 0.002, 0.005, 0.01, 0.02]
                    
                    for incr in increments:
                        new_r = min(r + incr, max_radius)
                        if new_r <= r:
                            continue
                            
                        # Quick validation with spatial indexing
                        temp_circles = best_circles.copy()
                        temp_circles[i, 2] = new_r
                        
                        # Check validity
                        valid = True
                        temp_points = temp_circles[:, :2]
                        temp_tree = KDTree(temp_points)
                        temp_neighbor_indices = temp_tree.query_ball_point([cx, cy], 2*(new_r + 0.005))

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
                                break
                            
                if improved:
                    break

        return best_circles

    # Multi-stage optimization approach
    # Stage 1: Coarse refinement
    coarse_solution = adaptive_refinement(circles, max_iter=20)
    stage1_fitness = calculate_fitness_with_spatial_indexing(coarse_solution)

    # Stage 2: Enhanced refinement
    enhanced_solution = adaptive_refinement(coarse_solution, max_iter=30)
    stage2_fitness = calculate_fitness_with_spatial_indexing(enhanced_solution)

    # Use the best of the two stages
    best_solution = enhanced_solution if stage2_fitness > stage1_fitness else coarse_solution
    best_fitness = max(stage1_fitness, stage2_fitness)

    # Stage 3: Targeted evolutionary optimization (more focused than previous versions)
    def targeted_evolution():
        # Identify most constrained circles (those with tight space)
        constrained_circles = []
        points = best_solution[:, :2]
        tree = KDTree(points)

        for i in range(n):
            cx, cy, r = best_solution[i]
            neighbor_indices = tree.query_ball_point([cx, cy], 2*(r + 0.005))
            if len(neighbor_indices) > 4:  # Circle surrounded by many others
                constrained_circles.append(i)

        # If we have few constrained circles, pick randomly
        if len(constrained_circles) < 5:
            constrained_circles = random.sample(list(range(n)), min(8, n))

        # Evolutionary approach
        population_size = 15
        generations = 15
        population = [best_solution.copy()]

        # Add diverse individuals
        for _ in range(population_size - 1):
            individual = best_solution.copy()
            # Perturb selected constrained circles
            selected_indices = random.sample(constrained_circles, min(4, len(constrained_circles)))
            for idx in selected_indices:
                individual[idx, 0] += np.random.normal(0, 0.015)
                individual[idx, 1] += np.random.normal(0, 0.015)
                individual[idx, 2] *= np.random.uniform(0.8, 1.2)

                # Clamp values
                individual[idx, 0] = np.clip(individual[idx, 0], 0.01, rect_width - 0.01)
                individual[idx, 1] = np.clip(individual[idx, 1], 0.01, rect_height - 0.01)
                individual[idx, 2] = max(0.0005, individual[idx, 2])

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

                # Single-point crossover
                child = parent1.copy()
                crossover_point = np.random.randint(1, n)
                child[crossover_point:, 0] = parent2[crossover_point:, 0]  # x positions
                child[crossover_point:, 1] = parent2[crossover_point:, 1]  # y positions
                child[crossover_point:, 2] = parent2[crossover_point:, 2]  # radii

                # Mutation with variable rates
                for i in range(n):
                    if random.random() < 0.15:
                        if random.random() < 0.6:  # Position mutation
                            child[i, 0] += np.random.normal(0, 0.008)
                            child[i, 1] += np.random.normal(0, 0.008)
                            child[i, 0] = np.clip(child[i, 0], 0.01, rect_width - 0.01)
                            child[i, 1] = np.clip(child[i, 1], 0.01, rect_height - 0.01)
                        else:  # Radius mutation
                            child[i, 2] *= np.random.uniform(0.9, 1.15)
                            child[i, 2] = max(0.0005, child[i, 2])

                new_population.append(child)

            population = new_population[:population_size]

        return best_solution

    # Run targeted evolutionary optimization
    try:
        evolved_solution = targeted_evolution()
        final_fitness = calculate_fitness_with_spatial_indexing(evolved_solution)

        if final_fitness > best_fitness:
            best_solution = evolved_solution
            best_fitness = final_fitness

    except Exception:
        pass

    # Final fine-tuning
    final_solution = adaptive_refinement(best_solution, max_iter=25)
    final_fitness = calculate_fitness_with_spatial_indexing(final_solution)

    if final_fitness > best_fitness:
        best_solution = final_solution

    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")