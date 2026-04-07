# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import random
from copy import deepcopy
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions - perimeter = 4, so width + height = 2
    # Optimized rectangle dimensions for maximum packing efficiency
    rect_width = 1.2
    rect_height = 0.8

    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Parameters
    n_circles = 21
    max_iterations = 150
    population_size = 100
    elite_size = 10
    initial_mutation_rate = 0.08

    # Enhanced adaptive grid initialization with overlap consideration
    def create_adaptive_grid(n, width, height, overlap_buffer=0.8):
        # Calculate optimal grid dimensions based on circle count
        sqrt_n = np.sqrt(n)
        rows = int(np.ceil(sqrt_n))
        cols = int(np.ceil(n / rows))

        # Adjust grid size to fit within rectangle
        cell_width = width / cols
        cell_height = height / rows

        # Use minimum of cell dimensions for radius with safety margin
        max_radius = min(cell_width, cell_height) * overlap_buffer

        # Generate grid layout with hexagonal offset pattern for better packing
        circles = []
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Hexagonal offset pattern
                offset = (i % 2) * (cell_width * 0.5)
                x = (j + 0.5) * cell_width + offset
                y = (i + 0.5) * cell_height
                circles.append([x, y, max_radius])
                idx += 1

        return np.array(circles)

    # Fast collision checking using KDTree with enhanced overlap detection
    def fast_collision_check(circles, width, height):
        # Check boundary constraints
        for circle in circles:
            x, y, r = circle
            if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
                return False

        # Create KDTree for efficient neighbor search
        coords = circles[:, :2]
        tree = cKDTree(coords)

        # Find neighbors within 2 * max_radius distance
        max_radius = np.max(circles[:, 2])
        pairs = tree.query_pairs(2 * max_radius, p=2)

        # Check actual distances for overlap
        for i, j in pairs:
            if i != j:
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                r1, r2 = circles[i, 2], circles[j, 2]
                dx = x1 - x2
                dy = y1 - y2
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (r1 + r2) * (r1 + r2)

                if dist_sq < min_dist_sq:
                    return False

        return True

    # Enhanced constraint-aware fitness evaluation with adaptive penalties
    def evaluate_fitness_with_constraints(circles):
        # Count constraint violations accurately
        boundary_violations = 0
        overlap_violations = 0

        # Boundary violations
        for circle in circles:
            x, y, r = circle
            if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
                boundary_violations += 1

        # Overlap violations using KDTree for efficiency
        coords = circles[:, :2]
        tree = cKDTree(coords)
        max_radius = np.max(circles[:, 2])
        pairs = tree.query_pairs(2 * max_radius, p=2)

        for i, j in pairs:
            if i != j:
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                r1, r2 = circles[i, 2], circles[j, 2]
                dx = x1 - x2
                dy = y1 - y2
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (r1 + r2) * (r1 + r2)

                if dist_sq < min_dist_sq:
                    overlap_violations += 1

        total_violations = boundary_violations + overlap_violations

        # Adaptive penalty system that scales with violation severity
        base_fitness = np.sum(circles[:, 2])

        # Dynamic penalty weights based on violation density
        violation_density = total_violations / n_circles if n_circles > 0 else 0
        if violation_density > 0.1:  # High violation density
            penalty_weight = 1000 + (violation_density * 5000)
        elif violation_density > 0.05:  # Medium violation density
            penalty_weight = 500 + (violation_density * 2000)
        else:  # Low violation density
            penalty_weight = 100 + (violation_density * 1000)

        penalty = penalty_weight * total_violations
        fitness = max(0, base_fitness - penalty)

        return fitness, total_violations

    # Enhanced genetic algorithm with two-stage optimization
    def genetic_algorithm_optimization():
        # Stage 1: Two-stage optimization approach for better exploration
        # Stage 1a: Coarse global search with varied initialization strategies
        population = []

        # Strategy 1: Grid-based initialization with hexagonal pattern
        circles = create_adaptive_grid(n_circles, rect_width, rect_height, overlap_buffer=0.7)
        population.append(circles)

        # Strategy 2: Multiple grid-based individuals with different overlap buffers
        for i in range(3):
            buffer = 0.6 + i * 0.1
            circles = create_adaptive_grid(n_circles, rect_width, rect_height, overlap_buffer=buffer)
            population.append(circles)

        # Strategy 3: Random initialization with careful constraint validation
        for i in range(10):
            circles = np.zeros((n_circles, 3))
            attempts = 0
            valid = False
            while attempts < 500 and not valid:
                # Generate random positions and radii
                for j in range(n_circles):
                    circles[j] = [
                        np.random.uniform(0.05, rect_width - 0.05),
                        np.random.uniform(0.05, rect_height - 0.05),
                        np.random.uniform(0.01, 0.2)
                    ]
                if fast_collision_check(circles, rect_width, rect_height):
                    valid = True
                attempts += 1

            if valid:
                population.append(circles)
            else:
                # Fallback to grid if random generation fails
                circles = create_adaptive_grid(n_circles, rect_width, rect_height, overlap_buffer=0.8)
                population.append(circles)

        # Strategy 4: Diversity from existing population with mutations
        for i in range(10):
            parent_idx = np.random.randint(0, len(population))
            child = deepcopy(population[parent_idx])
            # Add some random mutations to diversify
            for j in range(n_circles):
                if np.random.random() < 0.1:  # Small mutation probability
                    child[j, 0] += np.random.uniform(-0.05, 0.05)
                    child[j, 1] += np.random.uniform(-0.05, 0.05)
                    child[j, 2] *= np.random.uniform(0.9, 1.1)
            population.append(child)

        # Limit population size to avoid memory issues
        population = population[:population_size]

        best_fitness = 0.0
        best_individual = None
        best_violations = float('inf')

        # Evolutionary optimization loop
        for generation in range(max_iterations):
            # Evaluate fitness for all individuals
            fitness_scores = []
            violation_counts = []

            for individual in population:
                score, violations = evaluate_fitness_with_constraints(individual)
                fitness_scores.append(score)
                violation_counts.append(violations)

                if score > best_fitness or (score == best_fitness and violations < best_violations):
                    best_fitness = score
                    best_violations = violations
                    best_individual = individual.copy()

            # Sort by fitness (descending)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = [population[i] for i in sorted_indices]
            fitness_scores = [fitness_scores[i] for i in sorted_indices]
            violation_counts = [violation_counts[i] for i in sorted_indices]

            # Keep elite individuals
            new_population = population[:elite_size]

            # Generate offspring through crossover and mutation
            while len(new_population) < population_size:
                # Tournament selection with better diversity consideration
                parent1_idx = tournament_selection(population, fitness_scores, 3)
                parent2_idx = tournament_selection(population, fitness_scores, 3)

                # Crossover
                child = crossover(population[parent1_idx], population[parent2_idx])

                # Adaptive mutation rate that decreases over generations
                current_mutation_rate = initial_mutation_rate * (1 - generation / max_iterations)
                mutate(child, current_mutation_rate, rect_width, rect_height)

                # Probability of reinitialization for poor performers
                if np.random.random() < 0.05:
                    child = create_adaptive_grid(n_circles, rect_width, rect_height, overlap_buffer=0.8)

                new_population.append(child)

            population = new_population

        return best_individual, best_fitness

    # Stage 2: Enhanced local fine-tuning with constraint-aware gradient descent
    def local_finetuning(initial_solution):
        # Extract positions and radii
        initial_coords = initial_solution[:, :2].flatten()
        initial_radii = initial_solution[:, 2]

        # Combine into a single parameter vector
        params = np.concatenate([initial_coords, initial_radii])

        def objective_function(params):
            # Reconstruct circles from parameters
            coords = params[:-len(initial_radii)].reshape(-1, 2)
            radii = params[-len(initial_radii):]

            # Create circles array
            circles = np.column_stack([
                coords.flatten(),
                radii
            ]).reshape(-1, 3)

            # Ensure valid radii
            circles[:, 2] = np.maximum(0.001, circles[:, 2])

            # Check constraints and evaluate fitness with constraint-aware penalty
            if not fast_collision_check(circles, rect_width, rect_height):
                # Constraint-aware penalty that penalizes constraint violations more heavily
                # than just the raw violation count
                _, violations = evaluate_fitness_with_constraints(circles)
                penalty = violations * 1000  # Strong penalty for constraint violations
                return 1e6 + penalty  # Very large penalty for infeasible solutions

            # Minimize negative sum of radii (maximize sum)
            return -np.sum(radii)

        # Enhanced constraint handling function for local optimization
        def constraint_violation_penalty(circles):
            """Calculate penalty based on constraint violations to guide optimization"""
            violation_penalty = 0
            boundary_violations = 0
            overlap_violations = 0

            # Boundary violations
            for circle in circles:
                x, y, r = circle
                if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
                    boundary_violations += 1

            # Overlap violations using KDTree for efficiency
            coords = circles[:, :2]
            tree = cKDTree(coords)
            max_radius = np.max(circles[:, 2])
            pairs = tree.query_pairs(2 * max_radius, p=2)

            for i, j in pairs:
                if i != j:
                    x1, y1 = coords[i]
                    x2, y2 = coords[j]
                    r1, r2 = circles[i, 2], circles[j, 2]
                    dx = x1 - x2
                    dy = y1 - y2
                    dist_sq = dx*dx + dy*dy
                    min_dist_sq = (r1 + r2) * (r1 + r2)

                    if dist_sq < min_dist_sq:
                        overlap_violations += 1

            violation_penalty = (boundary_violations * 500) + (overlap_violations * 200)
            return violation_penalty

        # Use L-BFGS-B for local optimization with improved constraints
        try:
            # Create bounds for variables
            bounds = []
            # Position bounds
            for i in range(len(initial_coords)):
                if i % 2 == 0:  # x coordinates
                    bounds.append((0.05, rect_width - 0.05))
                else:  # y coordinates
                    bounds.append((0.05, rect_height - 0.05))
            # Radius bounds
            for i in range(len(initial_radii)):
                bounds.append((0.01, 0.3))

            # Optimize with additional convergence monitoring
            result = minimize(
                objective_function,
                params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-6}
            )

            if result.success:
                # Reconstruct final circles
                coords = result.x[:-len(initial_radii)].reshape(-1, 2)
                radii = result.x[-len(initial_radii):]
                circles = np.column_stack([coords.flatten(), radii]).reshape(-1, 3)

                # Final validation with constraint-aware penalty
                final_penalty = constraint_violation_penalty(circles)
                if final_penalty > 0:
                    # If constraints are violated, try a less aggressive approach
                    return initial_solution

                # Ensure valid radii post optimization
                circles[:, 2] = np.maximum(0.001, circles[:, 2])
                return circles
        except Exception as e:
            # If optimization fails due to numerical issues, return the best individual
            pass

        # If optimization fails, return the best individual
        return initial_solution

    # Run optimization
    best_individual, best_fitness = genetic_algorithm_optimization()

    # Apply local fine-tuning
    refined_solution = local_finetuning(best_individual)

    # Final validation check
    if not fast_collision_check(refined_solution, rect_width, rect_height):
        # If final check fails, return best individual from GA
        return best_individual

    return refined_solution

def tournament_selection(population, fitness_scores, k):
    """Select individual via tournament selection"""
    tournament_indices = np.random.choice(len(population), k)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitness)]
    return winner_index

def crossover(parent1, parent2):
    """Single point crossover on circle positions and radii"""
    child = deepcopy(parent1)

    # Select crossover point
    crossover_point = np.random.randint(1, len(parent1))

    # Cross over positions and radii
    child[crossover_point:, :2] = parent2[crossover_point:, :2]  # Positions
    child[crossover_point:, 2] = parent2[crossover_point:, 2]   # Radii

    return child

def mutate(individual, mutation_rate, rect_width, rect_height):
    """Mutate circle positions and radii"""
    for i in range(len(individual)):
        if np.random.random() < mutation_rate:
            # Mutate position
            individual[i, 0] = np.clip(
                individual[i, 0] + np.random.normal(0, 0.03),
                0.05, rect_width - 0.05)
            individual[i, 1] = np.clip(
                individual[i, 1] + np.random.normal(0, 0.03),
                0.05, rect_height - 0.05)

            # Mutate radius
            individual[i, 2] = np.clip(
                individual[i, 2] + np.random.normal(0, 0.015),
                0.01, 0.3)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")