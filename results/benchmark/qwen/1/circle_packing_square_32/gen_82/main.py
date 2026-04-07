# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import random
from copy import deepcopy

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """

    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    n_circles = 32

    def validate_circle_placement(circles):
        """Check if all circles are within bounds and non-overlapping."""
        # Check containment
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False

        # Check overlaps
        positions = circles[:, :2]
        radii = circles[:, 2]

        # Compute pairwise distances
        distances = cdist(positions, positions)

        # Check for overlaps
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                dist = distances[i, j]
                if dist < radii[i] + radii[j]:
                    return False

        return True

    def calculate_fitness(circles):
        """Calculate fitness as sum of radii, penalized for constraint violations."""
        total_radius = np.sum(circles[:, 2])

        # If invalid configuration, return very low fitness
        if not validate_circle_placement(circles):
            return -1000000

        return total_radius

    def initialize_voronoi_seeds(n_points=50):
        """Generate initial seed points using Voronoi diagram approach with boundary weighting."""
        # Generate more points than needed to ensure good distribution
        points = []

        # Add corner points for better boundary coverage
        for x in [0, 0.5, 1]:
            for y in [0, 0.5, 1]:
                points.append([x, y])

        # Add edge midpoints
        for x in [0.25, 0.5, 0.75]:
            points.append([x, 0])
            points.append([x, 1])
            points.append([0, x])
            points.append([1, x])

        # Add interior points
        for _ in range(20):
            points.append([np.random.random(), np.random.random()])

        # Generate Voronoi diagram
        vor = Voronoi(np.array(points))

        # Get Voronoi vertices and centroids as candidate positions
        candidates = []
        weights = []

        # Add vertices with boundary weighting
        for vertex in vor.vertices:
            if 0 <= vertex[0] <= 1 and 0 <= vertex[1] <= 1:
                # Calculate distance to boundaries
                dist_to_left = vertex[0]
                dist_to_right = 1 - vertex[0]
                dist_to_bottom = vertex[1]
                dist_to_top = 1 - vertex[1]

                # Weight based on proximity to boundaries (1.5x weight for near-boundary points)
                min_dist = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
                weight = 1.0
                if min_dist <= 0.1:
                    weight = 1.5  # Higher weight for points near boundaries

                candidates.append(vertex)
                weights.append(weight)

        # Add centroids of regions with boundary weighting
        for region in vor.regions:
            if len(region) > 0 and -1 not in region:
                centroid_x = np.mean([vor.vertices[i][0] for i in region])
                centroid_y = np.mean([vor.vertices[i][1] for i in region])
                if 0 <= centroid_x <= 1 and 0 <= centroid_y <= 1:
                    # Calculate distance to boundaries
                    dist_to_left = centroid_x
                    dist_to_right = 1 - centroid_x
                    dist_to_bottom = centroid_y
                    dist_to_top = 1 - centroid_y

                    # Weight based on proximity to boundaries
                    min_dist = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
                    weight = 1.0
                    if min_dist <= 0.1:
                        weight = 1.5  # Higher weight for points near boundaries

                    candidates.append([centroid_x, centroid_y])
                    weights.append(weight)

        # Remove duplicates
        unique_candidates = []
        unique_weights = []
        seen = set()
        for i, cand in enumerate(candidates):
            coord_tuple = (round(cand[0], 5), round(cand[1], 5))
            if coord_tuple not in seen:
                seen.add(coord_tuple)
                unique_candidates.append(cand)
                unique_weights.append(weights[i])

        # Sort by weights (descending) and take top n_points
        sorted_indices = np.argsort(unique_weights)[::-1]
        selected_candidates = [unique_candidates[i] for i in sorted_indices[:n_points]]

        return selected_candidates

    def generate_initial_population(n_circles, n_individuals=20):
        """Generate an initial population of circle configurations."""
        population = []

        # Get Voronoi-based seed points
        seed_points = initialize_voronoi_seeds(100)

        for _ in range(n_individuals):
            # Randomly sample positions from seed points
            selected_positions = random.sample(seed_points, min(n_circles, len(seed_points)))

            # Initialize circles with small radii
            circles = np.zeros((n_circles, 3))
            for i in range(len(selected_positions)):
                circles[i, 0] = selected_positions[i][0]
                circles[i, 1] = selected_positions[i][1]
                circles[i, 2] = 0.02  # Small initial radius

            # Try to increase radii while maintaining validity
            for i in range(len(selected_positions)):
                max_radius = min(
                    circles[i, 0],
                    1 - circles[i, 0],
                    circles[i, 1],
                    1 - circles[i, 1]
                )

                # Find maximum compatible radius with other circles
                for j in range(len(selected_positions)):
                    if i != j:
                        dist = np.sqrt(
                            (circles[i, 0] - circles[j, 0])**2 +
                            (circles[i, 1] - circles[j, 1])**2
                        )
                        max_radius = min(max_radius, dist - circles[j, 2])

                # Ensure positive radius
                if max_radius > 0.001:
                    circles[i, 2] = min(max_radius, 0.1)  # Limit initial growth

            population.append(circles.copy())

        return population

    def mutate_circles(circles, mutation_rate=0.3, max_radius_change=0.05):
        """Mutate circles configuration."""
        mutated = deepcopy(circles)

        # Mutate positions and radii
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Mutate position
                mutated[i, 0] += np.random.normal(0, 0.02)
                mutated[i, 1] += np.random.normal(0, 0.02)

                # Clamp to bounds
                mutated[i, 0] = max(0, min(1, mutated[i, 0]))
                mutated[i, 1] = max(0, min(1, mutated[i, 1]))

                # Mutate radius
                mutated[i, 2] += np.random.normal(0, max_radius_change)
                mutated[i, 2] = max(0.001, mutated[i, 2])  # Minimum radius

        return mutated

    def crossover(parent1, parent2):
        """Perform crossover between two parent configurations."""
        child = deepcopy(parent1)

        # Crossover at random points
        crossover_point = random.randint(1, len(child)-1)
        for i in range(crossover_point, len(child)):
            child[i] = parent2[i].copy()

        return child

    def local_refinement(circles, iterations=50):
        """Apply local refinement to improve configuration."""
        refined = deepcopy(circles)

        for _ in range(iterations):
            improved = False

            # Try to increase each circle's radius
            for i in range(len(refined)):
                # Save current state
                old_pos = refined[i, :2].copy()
                old_rad = refined[i, 2]

                # Calculate max possible radius at this location
                max_radius = min(
                    refined[i, 0],
                    1 - refined[i, 0],
                    refined[i, 1],
                    1 - refined[i, 1]
                )

                # Consider overlap constraints
                for j in range(len(refined)):
                    if i != j:
                        dist = np.sqrt(
                            (refined[i, 0] - refined[j, 0])**2 +
                            (refined[i, 1] - refined[j, 1])**2
                        )
                        max_radius = min(max_radius, dist - refined[j, 2])

                # Increase radius if beneficial and valid
                if max_radius > old_rad and max_radius > 0.001:
                    refined[i, 2] = max_radius
                    improved = True

                # Restore old state if invalid
                if not validate_circle_placement(refined):
                    refined[i, 0] = old_pos[0]
                    refined[i, 1] = old_pos[1]
                    refined[i, 2] = old_rad

            if not improved:
                break

        return refined

    # Main evolutionary algorithm
    population_size = 30
    generations = 100
    elite_size = 5

    # Generate initial population
    population = generate_initial_population(n_circles, population_size)

    best_fitness = float('-inf')
    best_individual = None

    # Evolution loop
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            fitness = calculate_fitness(individual)
            fitness_scores.append(fitness)

            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = deepcopy(individual)

        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]

        # Keep elite
        elites = population[:elite_size]

        # Generate new population through selection, crossover, and mutation
        new_population = elites[:]

        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = 3
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]

            # Select parent
            parent = population[winner_index]

            # Apply mutation
            mutated = mutate_circles(parent)

            # Try local refinement
            refined = local_refinement(mutated)

            new_population.append(refined)

        population = new_population

    # Final local refinement
    if best_individual is not None:
        final_solution = local_refinement(best_individual)
    else:
        # If no good solution found, use the best from the last generation
        final_solution = local_refinement(population[0])

    return final_solution

# EVOLVE-BLOCK-END