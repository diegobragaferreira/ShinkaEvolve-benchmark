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
    # Optimized rectangle dimensions for maximum packing efficiency (1.6:0.4 ratio)
    rect_width = 1.6
    rect_height = 0.4

    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Parameters
    n_circles = 21
    max_iterations = 200
    population_size = 180
    elite_size = 20
    initial_mutation_rate = 0.12

    # Improved hybrid initialization with hexagonal packing for better starting configurations
    def create_hybrid_initialization(n, width, height):
        """
        Create hybrid initialization combining hexagonal packing and grid-based approaches
        """
        # Hexagonal packing approach (better for dense packing)
        # Estimate circle radius based on area
        total_area = n * np.pi * 0.2**2  # Initial estimate
        rect_area = width * height
        density_factor = 0.9  # Typical hexagonal packing density

        # Estimate circle radius based on available space and density
        estimated_radius = np.sqrt(total_area / (np.pi * density_factor))

        # Hexagonal packing parameters
        hex_radius = estimated_radius * 0.9  # Slightly smaller to allow gaps
        hex_spacing_x = hex_radius * 2
        hex_spacing_y = hex_radius * np.sqrt(3)

        # Determine grid size for hexagonal packing
        cols = max(1, int(width / hex_spacing_x) + 1)
        rows = max(1, int(height / hex_spacing_y) + 1)

        circles = []
        idx = 0
        # Offset first row to center the pattern
        y_offset = hex_radius
        for i in range(rows):
            x_offset = hex_radius if i % 2 == 0 else hex_radius * 1.5
            for j in range(cols):
                if idx >= n:
                    break
                x = x_offset + j * hex_spacing_x
                y = y_offset + i * hex_spacing_y

                # Ensure within bounds with margin
                if x >= hex_radius and x <= width - hex_radius and \
                   y >= hex_radius and y <= height - hex_radius:
                    circles.append([x, y, hex_radius])
                    idx += 1

        # Fill remaining circles with grid if needed
        if len(circles) < n:
            grid_circles = create_adaptive_grid(n - len(circles), width, height)
            circles.extend(grid_circles.tolist())

        return np.array(circles)

    def create_adaptive_grid(n, width, height):
        """
        Create an adaptive grid layout based on circle count and rectangle dimensions
        """
        # Calculate optimal grid dimensions based on circle count
        sqrt_n = np.sqrt(n)
        rows = int(np.ceil(sqrt_n))
        cols = int(np.ceil(n / rows))

        # Adjust grid size to fit within rectangle with proper margins
        cell_width = width / cols
        cell_height = height / rows

        # Use minimum of cell dimensions for radius with safety margin
        max_radius = min(cell_width, cell_height) * 0.42

        # Generate grid layout with slight randomization to avoid symmetry issues
        circles = []
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Add slight random offset to break symmetry and improve packing
                x = (j + 0.5 + np.random.uniform(-0.15, 0.15)) * cell_width
                y = (i + 0.5 + np.random.uniform(-0.15, 0.15)) * cell_height
                # Ensure positions stay within bounds
                x = np.clip(x, max_radius, width - max_radius)
                y = np.clip(y, max_radius, height - max_radius)
                circles.append([x, y, max_radius])
                idx += 1

        return np.array(circles)

    # Fast collision checking using KDTree for O(n log n) complexity
    def fast_collision_check(circles, width, height):
        """
        Fast collision detection using spatial indexing with boundary checks
        """
        # Check boundary constraints efficiently using vectorized operations
        coords = circles[:, :2]
        radii = circles[:, 2]

        # Early exit if any circle violates boundary constraints
        if np.any((coords[:, 0] - radii < 0) | (coords[:, 0] + radii > width) |
                  (coords[:, 1] - radii < 0) | (coords[:, 1] + radii > height)):
            return False

        # Create KDTree for efficient neighbor search
        tree = cKDTree(coords)

        # Find neighbors within 2 * max_radius distance (tighter bound)
        pairs = tree.query_pairs(2 * np.max(radii), p=2)

        # If no pairs found, no overlaps possible
        if len(pairs) == 0:
            return True

        # Check actual distances for overlap with early termination for efficiency
        for i, j in pairs:
            x1, y1 = coords[i]
            x2, y2 = coords[j]
            r1, r2 = radii[i], radii[j]
            dx = x1 - x2
            dy = y1 - y2
            dist_sq = dx*dx + dy*dy
            min_dist_sq = (r1 + r2) * (r1 + r2)

            if dist_sq < min_dist_sq:
                return False

        return True

    # Constraint-aware fitness evaluation with penalty system
    def evaluate_fitness_with_constraints(circles):
        """
        Evaluate fitness considering both objective (radius sum) and constraints
        """
        if not fast_collision_check(circles, rect_width, rect_height):
            # Count constraint violations efficiently
            coords = circles[:, :2]
            radii = circles[:, 2]

            # Vectorized boundary violations
            boundary_violations = np.sum((coords[:, 0] - radii < 0) | (coords[:, 0] + radii > rect_width) |
                                       (coords[:, 1] - radii < 0) | (coords[:, 1] + radii > rect_height))

            # Overlap violations using spatial indexing
            tree = cKDTree(coords)
            pairs = tree.query_pairs(2 * np.max(radii), p=2)

            overlap_violations = 0
            for i, j in pairs:
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                r1, r2 = radii[i], radii[j]
                dx = x1 - x2
                dy = y1 - y2
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (r1 + r2) * (r1 + r2)

                if dist_sq < min_dist_sq:
                    overlap_violations += 1

            total_violations = boundary_violations + overlap_violations

            # Weighted fitness with heavy penalty for constraint violations
            base_fitness = np.sum(circles[:, 2])
            # Heavier penalty for constraint violations (more severe than previous versions)
            penalty = total_violations * 2000
            return max(0, base_fitness - penalty)

        return np.sum(circles[:, 2])

    # Enhanced genetic algorithm with two-stage optimization and probabilistic reset
    def genetic_algorithm_optimization():
        """
        Perform evolutionary optimization with enhanced strategies for better convergence
        """
        # Stage 1: Coarse global optimization
        population = []

        # Create diverse initial population with hybrid approach
        for i in range(population_size):
            if i == 0:
                # First individual: hexagonal packing for high-quality initial solution
                circles = create_hybrid_initialization(n_circles, rect_width, rect_height)
            elif i < 30:
                # Some individuals: hybrid initialization
                circles = create_hybrid_initialization(n_circles, rect_width, rect_height)
                # Add more substantial noise for diversity
                for j in range(n_circles):
                    circles[j, 0] += np.random.uniform(-0.08, 0.08)
                    circles[j, 1] += np.random.uniform(-0.08, 0.08)
                    circles[j, 2] *= np.random.uniform(0.85, 1.15)
            elif i < 60:
                # Some individuals: adaptive grid with moderate perturbations
                circles = create_adaptive_grid(n_circles, rect_width, rect_height)
                # Add more substantial noise for diversity
                for j in range(n_circles):
                    circles[j, 0] += np.random.uniform(-0.07, 0.07)
                    circles[j, 1] += np.random.uniform(-0.07, 0.07)
                    circles[j, 2] *= np.random.uniform(0.9, 1.1)
            else:
                # Random valid configurations with better initialization strategy
                circles = np.zeros((n_circles, 3))
                attempts = 0
                valid = False
                while attempts < 1000 and not valid:
                    # Generate random positions and radii with better distribution
                    for j in range(n_circles):
                        circles[j] = [
                            np.random.uniform(0.05, rect_width - 0.05),
                            np.random.uniform(0.05, rect_height - 0.05),
                            np.random.uniform(0.01, 0.25)
                        ]
                    # Quick coarse validation to reduce expensive full checks
                    if np.all((circles[:, 0] - circles[:, 2] > 0) &
                             (circles[:, 0] + circles[:, 2] < rect_width) &
                             (circles[:, 1] - circles[:, 2] > 0) &
                             (circles[:, 1] + circles[:, 2] < rect_height)):
                        if fast_collision_check(circles, rect_width, rect_height):
                            valid = True
                    attempts += 1

                if not valid:
                    # Fallback to grid if random generation fails
                    circles = create_adaptive_grid(n_circles, rect_width, rect_height)

            population.append(circles)

        best_fitness = 0.0
        best_individual = None
        stagnation_counter = 0
        stagnant_threshold = 15  # Stop if no improvement for 15 generations

        # Evolutionary optimization loop with progress tracking
        for generation in range(max_iterations):
            # Evaluate fitness for all individuals
            fitness_scores = []
            for individual in population:
                score = evaluate_fitness_with_constraints(individual)
                fitness_scores.append(score)

                if score > best_fitness:
                    best_fitness = score
                    best_individual = individual.copy()
                    stagnation_counter = 0  # Reset stagnation counter on improvement
                elif score == best_fitness:
                    stagnation_counter += 1  # Increment if no improvement

            # Sort by fitness (descending)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = [population[i] for i in sorted_indices]
            fitness_scores = [fitness_scores[i] for i in sorted_indices]

            # Keep elite individuals for better convergence
            new_population = population[:elite_size]

            # Generate offspring through crossover and mutation
            while len(new_population) < population_size:
                # Tournament selection with larger tournament size for better selection pressure
                parent1_idx = tournament_selection(population, fitness_scores, 6)
                parent2_idx = tournament_selection(population, fitness_scores, 6)

                # Crossover with better recombination
                child = crossover(population[parent1_idx], population[parent2_idx])

                # Adaptive mutation rate that decreases over generations for exploitation
                current_mutation_rate = initial_mutation_rate * (1 - generation / max_iterations)
                mutate(child, current_mutation_rate, rect_width, rect_height)

                # Probability of reinitialization for maintaining diversity
                if np.random.random() < 0.12:  # Increased probability
                    child = create_adaptive_grid(n_circles, rect_width, rect_height)

                new_population.append(child)

            # Apply probabilistic reset every 10 generations or when stagnating
            if generation % 10 == 0 or stagnation_counter > 10:
                for i in range(len(new_population)):
                    if np.random.random() < 0.10:  # 10% chance of reset
                        new_population[i] = create_adaptive_grid(n_circles, rect_width, rect_height)

            population = new_population

            # Early stopping if no improvement for too many generations
            if stagnation_counter >= stagnant_threshold:
                break

        return best_individual, best_fitness

    # Stage 2: Local fine-tuning using gradient-based optimization
    def local_finetuning(initial_solution):
        """
        Apply local optimization for fine-tuning the global solution
        """
        # Extract positions and radii
        initial_coords = initial_solution[:, :2].flatten()
        initial_radii = initial_solution[:, 2]

        # Combine into a single parameter vector for optimization
        params = np.concatenate([initial_coords, initial_radii])

        def objective_function(params):
            """
            Objective function for local optimization (minimize negative sum of radii)
            """
            # Reconstruct circles from parameters
            coords = params[:-len(initial_radii)].reshape(-1, 2)
            radii = params[-len(initial_radii):]

            # Create circles array
            circles = np.column_stack([
                coords.flatten(),
                radii
            ]).reshape(-1, 3)

            # Ensure valid radii (positive and within reasonable bounds)
            circles[:, 2] = np.maximum(0.001, np.minimum(0.35, circles[:, 2]))

            # Check constraints and evaluate fitness
            if not fast_collision_check(circles, rect_width, rect_height):
                # Heavy penalty for constraint violations to enforce feasibility
                return 1e8

            # Minimize negative sum of radii (maximize sum)
            return -np.sum(radii)

        # Use L-BFGS-B for local optimization with improved bounds and settings
        try:
            # Create bounds for variables (tighter bounds for better convergence)
            bounds = []
            # Position bounds (tighter margins for better confinement)
            for i in range(len(initial_coords)):
                if i % 2 == 0:  # x coordinates
                    bounds.append((0.05, rect_width - 0.05))
                else:  # y coordinates
                    bounds.append((0.05, rect_height - 0.05))
            # Radius bounds (slightly higher upper bound for flexibility)
            for i in range(len(initial_radii)):
                bounds.append((0.01, 0.35))

            # Optimize with more iterations to balance quality vs speed
            result = minimize(
                objective_function,
                params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-7, 'gtol': 1e-7}
            )

            if result.success:
                # Reconstruct final circles from optimization result
                coords = result.x[:-len(initial_radii)].reshape(-1, 2)
                radii = result.x[-len(initial_radii):]
                # Apply final bounds to ensure validity
                radii = np.maximum(0.001, np.minimum(0.35, radii))
                circles = np.column_stack([coords.flatten(), radii]).reshape(-1, 3)
                return circles
        except Exception:
            # If optimization fails for any reason, return the best solution from GA
            pass

        # Return the original solution if local optimization fails or doesn't improve
        return initial_solution

    # Run optimization with robust error handling
    try:
        best_individual, best_fitness = genetic_algorithm_optimization()

        # Apply local fine-tuning
        refined_solution = local_finetuning(best_individual)

        # Final validation check
        if not fast_collision_check(refined_solution, rect_width, rect_height):
            # If final check fails due to numerical issues, return best from GA
            return best_individual

        return refined_solution

    except Exception as e:
        # Fallback to a robust grid initialization if anything goes wrong
        print(f"Error occurred during optimization: {e}")
        print("Returning grid-based solution as fallback.")
        return create_adaptive_grid(n_circles, rect_width, rect_height)

def tournament_selection(population, fitness_scores, k):
    """Select individual via tournament selection with better diversity"""
    tournament_indices = np.random.choice(len(population), k, replace=False)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitness)]
    return winner_index

def crossover(parent1, parent2):
    """Single point crossover with improved recombination"""
    child = deepcopy(parent1)

    # Select crossover point with variation to encourage diversity
    crossover_point = np.random.randint(1, len(parent1) - 1)

    # Cross over positions and radii with better blending
    child[crossover_point:, :2] = parent2[crossover_point:, :2]  # Positions
    child[crossover_point:, 2] = parent2[crossover_point:, 2]   # Radii

    # Add some noise to children to prevent premature convergence
    for i in range(len(child)):
        if np.random.random() < 0.05:  # 5% chance of additional variation
            child[i, 0] += np.random.normal(0, 0.01)
            child[i, 1] += np.random.normal(0, 0.01)
            child[i, 2] += np.random.normal(0, 0.005)

    return child

def mutate(individual, mutation_rate, rect_width, rect_height):
    """Mutate circle positions and radii with adaptive variance"""
    for i in range(len(individual)):
        if np.random.random() < mutation_rate:
            # Mutate position with controlled variance
            individual[i, 0] = np.clip(
                individual[i, 0] + np.random.normal(0, 0.035),
                0.05, rect_width - 0.05)
            individual[i, 1] = np.clip(
                individual[i, 1] + np.random.normal(0, 0.035),
                0.05, rect_height - 0.05)

            # Mutate radius with adaptive variance
            individual[i, 2] = np.clip(
                individual[i, 2] + np.random.normal(0, 0.018),
                0.01, 0.3)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")