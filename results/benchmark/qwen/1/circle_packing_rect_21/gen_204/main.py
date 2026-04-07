# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from deap import base, creator, tools, algorithms
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

    # Rectangle dimensions (width + height = 2)
    rect_width = 1.0
    rect_height = 1.0

    # Number of circles
    n = 21

    def initialize_population():
        """Initialize population with structured patterns"""
        pop = []
        # Try different initial patterns
        patterns = [
            # Hexagonal packing
            lambda: generate_hexagonal_pattern(rect_width, rect_height, n),
            # Grid-based packing
            lambda: generate_grid_pattern(rect_width, rect_height, n),
            # Random with constraints
            lambda: generate_random_constrained_pattern(rect_width, rect_height, n)
        ]

        # Try all patterns and pick the best one
        best_pattern = None
        best_fitness = -float('inf')

        for pattern_func in patterns:
            try:
                individual = pattern_func()
                fitness = evaluate_fitness(individual, rect_width, rect_height)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_pattern = individual
            except:
                continue

        return [best_pattern] if best_pattern is not None else [generate_hexagonal_pattern(rect_width, rect_height, n)]

    def generate_hexagonal_pattern(width, height, n):
        """Generate initial hexagonal packing pattern"""
        circles = np.zeros((n, 3))

        # Determine grid parameters
        rows = int(np.sqrt(n))
        cols = int(np.ceil(n / rows))

        # Calculate spacing
        margin = 0.1
        max_radius = min(width, height) * 0.1

        # Create hexagonal grid
        x_spacing = max_radius * 2.5
        y_spacing = max_radius * 2.165  # sqrt(3)/2 * 2

        for i in range(n):
            row = i // cols
            col = i % cols

            x = margin + col * x_spacing
            y = margin + row * y_spacing

            if row % 2 == 1:
                x += x_spacing / 2

            # Adjust for bounds
            x = max(max_radius, min(width - max_radius, x))
            y = max(max_radius, min(height - max_radius, y))

            circles[i] = [x, y, max_radius]

        return circles

    def generate_grid_pattern(width, height, n):
        """Generate initial grid pattern"""
        circles = np.zeros((n, 3))

        # Find grid dimensions
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        # Calculate spacing
        margin = 0.1
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

    def generate_random_constrained_pattern(width, height, n):
        """Generate random pattern with basic constraints"""
        circles = np.zeros((n, 3))
        max_radius = min(width, height) * 0.1
        attempts = 0

        for i in range(n):
            attempts = 0
            valid = False
            while not valid and attempts < 1000:
                x = np.random.uniform(max_radius, width - max_radius)
                y = np.random.uniform(max_radius, height - max_radius)
                radius = np.random.uniform(0.01, max_radius)

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
                attempts += 1

        return circles

    def evaluate_fitness(individual, width, height):
        """Evaluate fitness of an individual - sum of radii with penalty for violations"""
        circles = individual.copy()
        total_radius = np.sum(circles[:, 2])

        # Penalty for boundary violations
        penalty = 0
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
                penalty -= 1000

        # Penalty for overlaps
        overlap_penalty = 0
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < (r1 + r2):
                    overlap_penalty -= (r1 + r2 - dist) * 100

        return total_radius + penalty + overlap_penalty

    def get_constraint_density_criticality(individual):
        """Calculate criticality based on local constraint density - how much space is available around each circle"""
        circles = individual.copy()
        n = len(circles)

        # Vectorized approach to compute minimum distances to neighbors
        if n <= 1:
            return np.ones(n) * 0.5

        # Get all centers
        centers = circles[:, :2]
        radii = circles[:, 2]

        # Compute pairwise distances between all centers
        # Using broadcasting for efficiency
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))

        # Mask to exclude self-distances
        np.fill_diagonal(distances, np.inf)

        # For each circle, find the minimum distance to any other circle
        min_distances = np.min(distances, axis=1)

        # Also compute minimum distance to boundaries
        boundary_distances = np.minimum(
            centers[:, 0],  # distance to left boundary
            np.minimum(
                1.0 - centers[:, 0],  # distance to right boundary
                np.minimum(
                    centers[:, 1],  # distance to bottom boundary
                    1.0 - centers[:, 1]  # distance to top boundary
                )
            )
        )

        # Combine boundary and neighbor distances to get overall constraint measure
        # Circles that are close to boundaries or neighbors are more constrained
        # Criticality = 1/(1 + scaled_min_distances) where lower values = more constrained
        combined_constraints = np.minimum(min_distances, boundary_distances)

        # Normalize: circles with smaller combined constraints are more critical
        # We want to map constraint levels to criticality [0,1] where 0 = most constrained
        if np.max(combined_constraints) > 0:
            # Invert so that smaller distances (more constrained) get higher criticality values
            normalized = 1.0 / (1.0 + combined_constraints / np.max(combined_constraints) * 10.0)
        else:
            normalized = np.ones(n) * 0.5

        return normalized

    def mut_radius(individual, indpb=0.2):
        """Mutation operator that modifies only the radius of selected circles with adaptive scaling"""
        mutated_individual = individual.copy()
        n = len(mutated_individual)

        # Get criticality scores based on constraint density
        criticality = get_constraint_density_criticality(mutated_individual)

        # Sort by criticality (most critical first - lowest values = most constrained)
        sorted_indices = np.argsort(criticality)  # Ascending order (0 = most constrained)

        # Mutate top 40% of critical circles (focus on the most constrained)
        num_mutations = int(n * 0.4)
        mutation_indices = sorted_indices[:num_mutations]

        for i in range(num_mutations):
            idx = mutation_indices[i]
            if random.random() < indpb:
                old_radius = mutated_individual[idx, 2]
                # Adaptive mutation based on criticality - more aggressive in low-criticality regions
                # Circles with low criticality (well-constrained) get more aggressive mutations
                adaptive_scale = 0.01 + (1.0 - criticality[idx]) * 0.03

                # Small random change to radius with adaptive scale
                delta = np.random.normal(0, adaptive_scale)
                new_radius = max(0.001, old_radius + delta)
                mutated_individual[idx, 2] = new_radius

        return mutated_individual,

    def crossover(parent1, parent2):
        """Crossover operator that exchanges radii of most critical circles with enhanced logic"""
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Get criticality scores for both parents based on constraint density
        crit1 = get_constraint_density_criticality(parent1)
        crit2 = get_constraint_density_criticality(parent2)

        # Exchange radii of circles with lowest criticality (most constrained)
        # Combined criticality: circles that are constrained in either parent should be more likely to swap
        combined_criticality = np.minimum(crit1, crit2)  # Lower = more constrained
        sorted_indices = np.argsort(combined_criticality)  # Ascending order

        # Exchange radii for top 30% of circles with probabilistic selection
        num_exchanges = int(len(parent1) * 0.3)
        for i in range(num_exchanges):
            idx = sorted_indices[i]
            # Apply crossover with probability based on criticality
            # More constrained circles (lower criticality) have higher chance of exchange
            crossover_prob = 0.7 + 0.3 * (1.0 - combined_criticality[idx])  # Higher prob for more constrained
            if random.random() < crossover_prob:
                child1[idx, 2], child2[idx, 2] = child2[idx, 2], child1[idx, 2]

        return child1, child2

    def is_valid_solution(circles, width, height):
        """Check if solution is valid"""
        # Check boundary constraints
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
                return False

        # Check overlap constraints
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < (r1 + r2):
                    return False

        return True

    # Main algorithm
    start_time = time.time()

    # Initialize population
    population = initialize_population()

    # Use the best initial pattern as starting point
    best_individual = population[0].copy()

    # Local optimization to improve the initial solution
    for _ in range(200):
        # Create a copy to work with
        test_individual = best_individual.copy()

        # Get constraint density criticality to identify most constrained circles
        criticality = get_constraint_density_criticality(test_individual)

        # Select circles that are most constrained (lowest criticality) to try to improve
        # These are the circles that would benefit most from optimization
        sorted_indices = np.argsort(criticality)  # Ascending order (0 = most constrained)
        num_to_improve = min(10, len(test_individual))  # Improve up to 10 most constrained circles
        selected_indices = sorted_indices[:num_to_improve]

        # Try small improvements for selected circles
        for idx in selected_indices:
            old_x, old_y, old_r = test_individual[idx]

            # For constrained circles, try to increase radius first (they're more likely to benefit)
            # Then try small position adjustments if that doesn't work
            test_r = old_r * 1.05  # Try increasing radius by 5%
            test_r = min(test_r, min(old_x, old_y, 1.0-old_x, 1.0-old_y) * 0.95)  # Respect bounds

            # Check if increased radius is valid
            valid = True
            for other_idx in range(len(test_individual)):
                if other_idx != idx:
                    ox, oy, oradius = test_individual[other_idx]
                    dist = np.sqrt((old_x - ox)**2 + (old_y - oy)**2)
                    if dist < (test_r + oradius):
                        valid = False
                        break

            if valid and test_r > old_r:
                # Accept radius increase
                test_individual[idx] = [old_x, old_y, test_r]
                continue

            # If we couldn't increase radius, try position adjustments
            # Make adjustments that help avoid collisions or stay within bounds
            new_x = max(0.01, min(0.99, old_x + np.random.normal(0, 0.005)))
            new_y = max(0.01, min(0.99, old_y + np.random.normal(0, 0.005)))
            new_r = old_r

            # Check if this adjustment helps
            valid = True
            for other_idx in range(len(test_individual)):
                if other_idx != idx:
                    ox, oy, oradius = test_individual[other_idx]
                    dist = np.sqrt((new_x - ox)**2 + (new_y - oy)**2)
                    if dist < (new_r + oradius):
                        valid = False
                        break

            # If valid, update
            if valid:
                test_individual[idx] = [new_x, new_y, new_r]

        # If this improves the fitness, accept it
        old_fitness = evaluate_fitness(best_individual, rect_width, rect_height)
        new_fitness = evaluate_fitness(test_individual, rect_width, rect_height)

        if new_fitness > old_fitness:
            best_individual = test_individual.copy()

    # Evolutionary Algorithm
    # Define DEAP classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", creator.Individual, best_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register operators
    toolbox.register("evaluate", lambda ind: evaluate_fitness(ind, rect_width, rect_height))
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mut_radius)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Run evolution
    pop = toolbox.population(n=30)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run the evolutionary algorithm
    try:
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3,
                                          ngen=100, stats=stats, halloffame=hof, verbose=False)
        best_individual = hof[0]
    except:
        # If evolutionary fails, return the local optimized solution
        pass

    # Final validation and cleanup
    if not is_valid_solution(best_individual, rect_width, rect_height):
        # Reinitialize with better pattern if needed
        best_individual = generate_hexagonal_pattern(rect_width, rect_height, n)

    # Final local fine-tuning
    for _ in range(100):
        test_individual = best_individual.copy()

        # Get constraint density criticality to focus on the most constrained circles
        criticality = get_constraint_density_criticality(test_individual)

        # Sort by criticality (ascending = most constrained first)
        sorted_indices = np.argsort(criticality)

        # Perturb top 10 most constrained circles, but with smarter approach
        for i in range(min(10, len(test_individual))):
            idx = sorted_indices[i]
            old_x, old_y, old_r = test_individual[idx]

            # Strategy 1: Try to increase radius first (these are most constrained)
            test_r = old_r * 1.02  # Small increase
            test_r = min(test_r, min(old_x, old_y, 1.0-old_x, 1.0-old_y) * 0.9)  # Respect bounds

            # Check if this works without conflicts
            valid = True
            for other_idx in range(len(test_individual)):
                if other_idx != idx:
                    ox, oy, oradius = test_individual[other_idx]
                    dist = np.sqrt((old_x - ox)**2 + (old_y - oy)**2)
                    if dist < (test_r + oradius):
                        valid = False
                        break

            if valid and test_r > old_r:
                test_individual[idx] = [old_x, old_y, test_r]
                continue

            # Strategy 2: If cannot increase radius, try small position adjustments to reduce conflicts
            step_x = np.random.normal(0, 0.003)  # Smaller step
            step_y = np.random.normal(0, 0.003)  # Smaller step
            new_x = max(0.01, min(0.99, old_x + step_x))
            new_y = max(0.01, min(0.99, old_y + step_y))
            new_r = old_r

            # Check if this adjustment helps avoid collisions
            valid = True
            for other_idx in range(len(test_individual)):
                if other_idx != idx:
                    ox, oy, oradius = test_individual[other_idx]
                    dist = np.sqrt((new_x - ox)**2 + (new_y - oy)**2)
                    if dist < (new_r + oradius):
                        valid = False
                        break

            if valid:
                test_individual[idx] = [new_x, new_y, new_r]

        # Validate and accept improvement
        if evaluate_fitness(test_individual, rect_width, rect_height) > evaluate_fitness(best_individual, rect_width, rect_height):
            best_individual = test_individual.copy()

    # Ensure final solution is valid
    final_circles = best_individual.copy()
    if not is_valid_solution(final_circles, rect_width, rect_height):
        # Fallback to structured pattern
        final_circles = generate_hexagonal_pattern(rect_width, rect_height, n)

    return final_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")