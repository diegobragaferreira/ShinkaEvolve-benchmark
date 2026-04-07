# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from deap import base, creator, tools, algorithms
import random
import time
from scipy.spatial.distance import cdist

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
            # Spiral pattern
            lambda: generate_spiral_pattern(rect_width, rect_height, n),
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
            except Exception as e:
                continue

        return [best_pattern] if best_pattern is not None else [generate_hexagonal_pattern(rect_width, rect_height, n)]

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
        attempts = 0

        for i in range(n):
            attempts = 0
            valid = False
            while not valid and attempts < 1000:
                x = np.random.uniform(max_radius, width - max_radius)
                y = np.random.uniform(max_radius, height - max_radius)
                radius = np.random.uniform(0.005, max_radius)

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

        # Penalty for overlaps - use vectorized computation for efficiency
        if len(circles) > 1:
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
        """Calculate criticality based on Voronoi diagram - how much room each circle has"""
        circles = individual.copy()
        n = len(circles)

        # Generate Voronoi diagram
        points = circles[:, :2]  # x,y coordinates
        try:
            vor = Voronoi(points)
        except:
            # Fallback if Voronoi fails - return uniform criticality
            return np.ones(n) * 0.01

        criticality_scores = np.zeros(n)

        # For each circle, calculate minimum distance to nearest circle center
        # This gives a better measure of how much space is available
        coords = circles[:, :2]
        for i in range(n):
            # Get distances to all other circles
            distances = np.sqrt(np.sum((coords - coords[i])**2, axis=1))
            # Exclude self-distance
            distances[i] = float('inf')
            # Minimum distance to any other circle center
            min_distance = np.min(distances)
            # Criticality is inversely proportional to this distance
            # Smaller distances = more constrained = higher criticality
            if min_distance > 0:
                # Use inverse relationship but bound it
                criticality_scores[i] = 1.0 / (min_distance + 0.001)
            else:
                criticality_scores[i] = 1000  # Very constrained

        # Also consider boundary constraints
        for i in range(n):
            x, y, r = circles[i]
            # Distance to nearest boundary
            min_boundary_dist = min(x, y, 1-x, 1-y)
            # If very close to boundary, increase criticality
            if min_boundary_dist < 0.05:
                criticality_scores[i] *= (1 + 5 * (0.05 - min_boundary_dist))

        # Normalize and ensure minimum value
        if np.max(criticality_scores) > 0:
            criticality_scores = criticality_scores / np.max(criticality_scores)

        # Default fallback values
        criticality_scores[np.isnan(criticality_scores)] = 0.01
        criticality_scores[criticality_scores <= 0] = 0.01

        return criticality_scores

    def mut_radius(individual, indpb=0.2):
        """Mutation operator that modifies only the radius of selected circles with adaptive strength based on Voronoi criticality"""
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
        """Check if solution is valid"""
        # Check boundary constraints
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
                return False

        # Check overlap constraints - vectorized for efficiency
        if len(circles) > 1:
            coords = circles[:, :2]
            radii = circles[:, 2]
            distances = cdist(coords, coords)
            # Create mask for upper triangle (avoid double counting)
            mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
            # Check overlaps
            overlap_distances = distances[mask]
            overlap_radii = (radii[:, None] + radii[None, :])[mask]
            if np.any(overlap_distances < overlap_radii):
                return False

        return True

    # Main algorithm
    start_time = time.time()

    # Initialize population
    population = initialize_population()

    # Use the best initial pattern as starting point
    best_individual = population[0].copy()

    # Local optimization to improve the initial solution
    for _ in range(300):
        # Create a copy to work with
        test_individual = best_individual.copy()

        # Select 5 circles to adjust
        selected_indices = np.random.choice(len(test_individual), 5, replace=False)

        for idx in selected_indices:
            old_x, old_y, old_r = test_individual[idx]
            # Small random adjustments
            new_x = max(0.005, min(0.995, old_x + np.random.normal(0, 0.01)))
            new_y = max(0.005, min(0.995, old_y + np.random.normal(0, 0.01)))
            new_r = max(0.001, old_r + np.random.normal(0, 0.005))

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
    pop = toolbox.population(n=40)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run the evolutionary algorithm
    try:
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3,
                                          ngen=150, stats=stats, halloffame=hof, verbose=False)
        best_individual = hof[0]
    except:
        # If evolutionary fails, return the local optimized solution
        pass

    # Final validation and cleanup
    if not is_valid_solution(best_individual, rect_width, rect_height):
        # Reinitialize with better pattern if needed
        best_individual = generate_hexagonal_pattern(rect_width, rect_height, n)

    # Final local fine-tuning
    for _ in range(150):
        test_individual = best_individual.copy()

        # Focus on most critical circles
        criticality = get_voronoi_criticality(test_individual)
        sorted_indices = np.argsort(-criticality)

        # Perturb top 10 circles
        for i in range(min(10, len(test_individual))):
            idx = sorted_indices[i]
            old_x, old_y, old_r = test_individual[idx]

            # Make small adjustments
            new_x = max(0.005, min(0.995, old_x + np.random.normal(0, 0.005)))
            new_y = max(0.005, min(0.995, old_y + np.random.normal(0, 0.005)))
            new_r = max(0.001, old_r + np.random.normal(0, 0.002))

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