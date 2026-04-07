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

    def get_geometric_criticality(individual):
        """Calculate criticality based on actual geometric constraints - minimum distance to neighbors"""
        circles = individual.copy()
        n = len(circles)

        # Calculate pairwise distances
        positions = circles[:, :2]
        radii = circles[:, 2]

        # Compute distance matrix
        distances = np.sqrt(np.sum((positions[:, np.newaxis] - positions[np.newaxis, :])**2, axis=2))

        # For each circle, find the minimum distance to any other circle
        # This represents how close it is to being constrained by neighboring circles
        criticality_scores = np.full(n, np.inf)

        for i in range(n):
            # Find minimum distance to any other circle (excluding itself)
            distances_from_i = distances[i, :]
            distances_from_i[i] = np.inf  # Exclude self-distance
            min_neighbor_distance = np.min(distances_from_i)

            # Criticality is inverse of distance to nearest neighbor
            # But we also need to consider the radii
            if min_neighbor_distance < np.inf and min_neighbor_distance > 0:
                # Criticality increases when circles are closer together relative to their radii
                # We want to avoid very tight packing that makes changes impossible
                min_radius = radii[i]
                closest_other_radius = radii[np.argmin(distances_from_i)]
                combined_radius = min_radius + closest_other_radius

                # The criticality reflects how much we're "squeezed"
                # Small distance relative to combined radii = high criticality
                if combined_radius > 0:
                    normalized_distance = min_neighbor_distance / combined_radius
                    criticality_scores[i] = normalized_distance
                else:
                    criticality_scores[i] = 1.0
            else:
                # If no neighbors or distance is infinite, not critical
                criticality_scores[i] = 1.0

        # Normalize criticality scores to [0,1] range where 0 = most constrained, 1 = least constrained
        # We invert because lower values mean tighter constraints
        if np.max(criticality_scores) > 0:
            # Normalize to [0,1] where 0 = most constrained, 1 = least constrained
            criticality_scores = 1.0 / (1.0 + criticality_scores)
        else:
            criticality_scores = np.ones(n)

        return criticality_scores

    def mut_radius(individual, indpb=0.2):
        """Mutation operator that modifies only the radius of selected circles with adaptive scaling"""
        mutated_individual = individual.copy()
        n = len(mutated_individual)

        # Get geometric criticality scores
        criticality = get_geometric_criticality(mutated_individual)

        # Sort by criticality (most critical first) - lower criticality means more constrained
        sorted_indices = np.argsort(criticality)  # Ascending order (0 = most constrained)

        # Mutate top 40% of critical circles (focus on the most constrained)
        num_mutations = int(n * 0.4)
        mutation_indices = sorted_indices[:num_mutations]

        for i in range(num_mutations):
            idx = mutation_indices[i]
            if random.random() < indpb:
                old_radius = mutated_individual[idx, 2]

                # Adaptive mutation based on geometric criticality
                # More aggressive mutation in less constrained regions (higher criticality values)
                # Less aggressive mutation in highly constrained regions (lower criticality values)
                if criticality[idx] < 0.3:  # Very constrained
                    adaptive_scale = 0.005
                elif criticality[idx] < 0.6:  # Moderately constrained
                    adaptive_scale = 0.01
                else:  # Less constrained
                    adaptive_scale = 0.02

                # Small random change to radius with adaptive scale
                delta = np.random.normal(0, adaptive_scale)
                new_radius = max(0.001, old_radius + delta)
                mutated_individual[idx, 2] = new_radius

        return mutated_individual,

    def crossover(parent1, parent2):
        """Crossover operator that exchanges radii of most critical circles with enhanced logic"""
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Get geometric criticality scores for both parents
        crit1 = get_geometric_criticality(parent1)
        crit2 = get_geometric_criticality(parent2)

        # Exchange radii of circles with lowest criticality (most constrained)
        # This preserves the most constrained configurations
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

        # Try small perturbations
        # Select circles based on geometric criticality to focus on most constrained
        criticality = get_geometric_criticality(best_individual)
        # Select top 5 most constrained circles
        selected_indices = np.argsort(criticality)[:5]

        for idx in selected_indices:
            old_x, old_y, old_r = test_individual[idx]
            # Small random adjustments - adapt step size based on criticality
            # More constrained circles get smaller steps to avoid violation
            step_size_x = 0.005 if criticality[idx] < 0.3 else 0.01
            step_size_y = 0.005 if criticality[idx] < 0.3 else 0.01
            step_size_r = 0.002 if criticality[idx] < 0.3 else 0.005

            new_x = max(0.01, min(0.99, old_x + np.random.normal(0, step_size_x)))
            new_y = max(0.01, min(0.99, old_y + np.random.normal(0, step_size_y)))
            new_r = max(0.001, old_r + np.random.normal(0, step_size_r))

            # Check if this violates constraints
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

        # Focus on most critical circles (based on geometric constraint)
        criticality = get_geometric_criticality(test_individual)
        # Sort by criticality to get most constrained circles first
        sorted_indices = np.argsort(criticality)  # Lowest values = most constrained

        # Perturb top 10 circles (or all if fewer)
        num_perturb = min(10, len(test_individual))
        for i in range(num_perturb):
            idx = sorted_indices[i]
            old_x, old_y, old_r = test_individual[idx]

            # Make small adjustments, adapt step size based on constraint level
            step_size_x = 0.002 if criticality[idx] < 0.3 else 0.005
            step_size_y = 0.002 if criticality[idx] < 0.3 else 0.005
            step_size_r = 0.001 if criticality[idx] < 0.3 else 0.002

            new_x = max(0.01, min(0.99, old_x + np.random.normal(0, step_size_x)))
            new_y = max(0.01, min(0.99, old_y + np.random.normal(0, step_size_y)))
            new_r = max(0.001, old_r + np.random.normal(0, step_size_r))

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