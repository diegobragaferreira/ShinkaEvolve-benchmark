# EVOLVE-BLOCK-START
import numpy as np
from deap import base, creator, tools, algorithms
import random
from scipy.spatial import cKDTree
import math
from scipy.optimize import minimize
import warnings
from collections import Counter

# Global constants
POP_SIZE = 200
NGEN = 150
MUTPB = 0.15
CXPB = 0.5
BOUND_LOW = 0.0
BOUND_UP = 1.0
ELITISM_COUNT = 5

# Define the fitness and individual classes for DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

class CirclePacker:
    """Handles circle packing constraints and validation."""

    @staticmethod
    def check_containment(circles):
        """Check containment constraints efficiently"""
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        radii = circles[:, 2]

        # Check boundaries for all circles at once using vectorized operations
        containment_violations = (
            (x_coords - radii < BOUND_LOW) |
            (x_coords + radii > BOUND_UP) |
            (y_coords - radii < BOUND_LOW) |
            (y_coords + radii > BOUND_UP)
        )

        return np.sum(containment_violations)

    @staticmethod
    def calculate_overlap_penalty(circles):
        """Calculate overlap penalty using efficient spatial indexing with optimizations"""
        if len(circles) <= 1:
            return 0.0

        # Build KDTree for efficient neighbor search
        tree = cKDTree(circles[:, :2])

        penalty = 0.0
        radii = circles[:, 2]

        # For each circle, find neighbors within sum of radii
        for i in range(len(circles)):
            x1, y1, r1 = circles[i]

            # Query nearby points (within 2*(r1+r2) distance)
            # Use a more efficient search by finding neighbors within reasonable range
            max_distance = 2 * (r1 + np.max(radii)) if len(radii) > 0 else 2 * r1

            neighbors = tree.query_ball_point([x1, y1], max_distance)

            # Check overlaps with neighbors
            for j in neighbors:
                if i != j:
                    x2, y2, r2 = circles[j]
                    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < r1 + r2:
                        penalty += 1000 * (r1 + r2 - distance)

        return penalty

class CircleEvaluator:
    """Handles circle evaluation and fitness calculation."""

    @staticmethod
    def eval_circles(individual):
        """Evaluate the fitness of an individual (set of circles)"""
        # Convert individual to circles array
        circles = np.array(individual).reshape(-1, 3)

        # Calculate sum of radii
        total_radius = np.sum(circles[:, 2])

        # Check constraints
        containment_violations = CirclePacker.check_containment(circles)
        overlap_penalty = CirclePacker.calculate_overlap_penalty(circles)

        # Combine penalties with weighted scheme
        total_penalty = 10000 * containment_violations + overlap_penalty

        # Return fitness (higher is better)
        return (total_radius - total_penalty,)

class Initializer:
    """Handles circle initialization strategies."""

    @staticmethod
    def generate_grid_refined_initialization():
        """Generate improved initial circle positions using grid refinement approach"""
        n = 26
        circles = []

        # Start with a systematic grid approach
        # Calculate grid dimensions based on the number of circles
        grid_size = int(math.ceil(math.sqrt(n)))
        if grid_size < 2:
            grid_size = 2

        # Create spacing that accounts for optimal packing density
        # For 26 circles in unit square, we expect roughly sqrt(26) ~ 5.1 grid cells
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)

        # Place circles systematically on grid
        placed_positions = set()
        count = 0

        # First, fill grid points with circles
        for i in range(1, grid_size + 1):
            if count >= n:
                break
            for j in range(1, grid_size + 1):
                if count >= n:
                    break

                x = i * spacing_x
                y = j * spacing_y

                # Ensure we don't go out of bounds and avoid duplicates
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))

                # Skip if already placed
                if (round(x, 3), round(y, 3)) not in placed_positions:
                    # Calculate max possible radius considering boundaries
                    min_dist_to_bound = min(x, 1-x, y, 1-y)
                    r = min(0.15, min_dist_to_bound/2)

                    # Add variance to make it more realistic
                    r *= random.uniform(0.8, 1.2)
                    r = max(0.005, min(0.15, r))

                    circles.extend([x, y, r])
                    placed_positions.add((round(x, 3), round(y, 3)))
                    count += 1

        # If we didn't place enough circles, add strategic positions
        if count < n:
            # Add corner and edge positions for better distribution
            additional_positions = [
                (0.5, 0.5),  # center
                (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),  # corners
                (0.5, 0.25), (0.5, 0.75), (0.25, 0.5), (0.75, 0.5),  # midpoints
                (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9),  # near corners
            ]

            for pos in additional_positions:
                if count >= n:
                    break
                x, y = pos
                # Ensure we don't go out of bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))

                # Skip if already placed
                if (round(x, 3), round(y, 3)) not in placed_positions:
                    # Calculate max possible radius
                    min_dist_to_bound = min(x, 1-x, y, 1-y)
                    r = min(0.12, min_dist_to_bound/2)

                    # Add some variance
                    r *= random.uniform(0.85, 1.15)
                    r = max(0.005, min(0.15, r))

                    circles.extend([x, y, r])
                    placed_positions.add((round(x, 3), round(y, 3)))
                    count += 1

        # Fill any remaining positions with random placement but with smart constraints
        while len(circles) < n * 3:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            # Use a more informed radius distribution based on expected optimal spacing
            r = random.uniform(0.005, 0.12)
            circles.extend([x, y, r])

        return circles[:n*3]

    @staticmethod
    def init_individual():
        """Initialize an individual with better initialization"""
        individual = Initializer.generate_better_initialization()

        # Add more variance via mutation-like perturbations
        for i in range(len(individual)):
            if i % 3 == 2:  # This is a radius
                # Mutate radius with bounded adjustment
                individual[i] = max(0.001, min(0.4, individual[i] * random.uniform(0.9, 1.1)))
            else:  # This is x or y coordinate
                # Mutate position with bounded adjustment
                individual[i] = max(BOUND_LOW, min(BOUND_UP, individual[i] + random.gauss(0, 0.015)))

        return individual

class MutationHandler:
    """Handles mutation operations."""

    @staticmethod
    def mutate_individual(individual, generation=0):
        """Mutate an individual with three-phase adaptive mutation rates"""
        # Three-phase adaptive mutation scheduling
        if generation <= 60:
            # Phase 1: High exploration
            gen_rate = 0.15
        elif generation <= 120:
            # Phase 2: Balanced refinement
            gen_rate = 0.05
        else:
            # Phase 3: Fine-tuning
            gen_rate = 0.015

        for i in range(len(individual)):
            if random.random() < gen_rate:
                if i % 3 == 2:  # This is a radius
                    # Mutate radius with bounded adjustment
                    individual[i] = max(0.001, min(0.4, individual[i] * random.uniform(0.85, 1.15)))
                else:  # This is x or y coordinate
                    # Mutate position with bounded adjustment
                    individual[i] = max(BOUND_LOW, min(BOUND_UP, individual[i] + random.gauss(0, 0.025)))

        return individual,

class CrossoverHandler:
    """Handles crossover operations with constraint awareness."""

    @staticmethod
    def adaptive_constraint_aware_crossover(ind1, ind2):
        """Perform adaptive crossover that weights probability based on overlap risk"""
        # Convert to numpy arrays for easier manipulation
        arr1 = np.array(ind1).reshape(-1, 3)
        arr2 = np.array(ind2).reshape(-1, 3)

        # Calculate overlap risks between corresponding circles in parents
        crossover_probs = []
        for i in range(len(arr1)):
            x1, y1, r1 = arr1[i]
            x2, y2, r2 = arr2[i]

            # Calculate distance between parent circles
            distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

            # Determine crossover probability based on overlap risk
            # If circles are very far apart, higher chance of crossover
            # If circles are close together, lower chance to preserve good configuration
            if distance < 1.5 * (r1 + r2):  # High overlap risk zone
                crossover_prob = 0.3  # Low crossover probability
            else:  # Low overlap risk zone
                crossover_prob = 0.8  # High crossover probability

            crossover_probs.append(crossover_prob)

        # Perform crossover with adaptive probabilities
        for i in range(len(ind1)):
            if random.random() < crossover_probs[i]:
                # Swap genes with probability based on overlap risk
                if i % 3 == 0:  # x coordinate
                    ind1[i], ind2[i] = ind2[i], ind1[i]
                elif i % 3 == 1:  # y coordinate
                    ind1[i], ind2[i] = ind2[i], ind1[i]
                else:  # radius
                    ind1[i], ind2[i] = ind2[i], ind1[i]

        # Check if offspring violate constraints and repair if needed
        def repair_if_needed(circles):
            # Repair containment violations
            for i in range(len(circles)):
                x, y, r = circles[i]
                # Bound the circle to stay within unit square
                x = max(r, min(1-r, x))
                y = max(r, min(1-r, y))
                circles[i] = [x, y, r]

            # Repair overlap violations through iterative adjustment
            changed = True
            iterations = 0
            while changed and iterations < 20:
                changed = False
                for i in range(len(circles)):
                    x1, y1, r1 = circles[i]
                    for j in range(len(circles)):
                        if i != j:
                            x2, y2, r2 = circles[j]
                            distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                            if distance < r1 + r2:
                                # Adjust positions to resolve overlap
                                overlap = (r1 + r2) - distance
                                # Move circles apart along the line connecting their centers
                                dx = x2 - x1
                                dy = y2 - y1
                                if dx == 0 and dy == 0:
                                    # Random movement if they're at the same point
                                    angle = random.uniform(0, 2*math.pi)
                                    dx = math.cos(angle)
                                    dy = math.sin(angle)

                                # Normalize
                                norm = math.sqrt(dx*dx + dy*dy)
                                dx /= norm
                                dy /= norm

                                # Move both circles apart
                                move_amount = overlap / 2.0
                                circles[i][0] -= dx * move_amount
                                circles[i][1] -= dy * move_amount
                                circles[j][0] += dx * move_amount
                                circles[j][1] += dy * move_amount

                                changed = True

                iterations += 1

            return circles

        # Repair both offspring
        repaired_ind1 = repair_if_needed(arr1.copy())
        repaired_ind2 = repair_if_needed(arr2.copy())

        # Convert back to individual format
        ind1[:] = repaired_ind1.flatten().tolist()
        ind2[:] = repaired_ind2.flatten().tolist()

        return ind1, ind2

class SelectionHandler:
    """Handles selection operations."""

    @staticmethod
    def adaptive_tournament_selection(population, k, diversity_threshold=0.1):
        """Adaptive tournament selection based on population diversity"""
        # Calculate diversity metric
        if len(population) < 2:
            return tools.selTournament(population, k, tournsize=3)

        # Compute diversity based on average distance between individuals
        distances = []
        for i in range(min(20, len(population))):
            for j in range(i+1, min(20, len(population))):
                dist = np.linalg.norm(np.array(population[i]) - np.array(population[j]))
                distances.append(dist)

        avg_diversity = np.mean(distances) if distances else 0

        # Adjust tournament size based on diversity
        if avg_diversity > diversity_threshold:
            tournsize = max(3, min(7, int(5 + avg_diversity * 10)))  # Higher diversity = larger tournaments
        else:
            tournsize = max(3, min(7, int(3 + avg_diversity * 20)))   # Lower diversity = smaller tournaments

        return tools.selTournament(population, k, tournsize=tournsize)

class LocalRefiner:
    """Handles local refinement strategies."""

    @staticmethod
    def hierarchical_local_refinement(circles_array, max_iter=100):
        """Apply hierarchical local optimization based on overlap severity classification"""

        def count_overlap_violations(circles):
            """Count number of overlap violations"""
            count = 0
            if len(circles) <= 1:
                return 0

            for i in range(len(circles)):
                x1, y1, r1 = circles[i]
                for j in range(i+1, len(circles)):
                    x2, y2, r2 = circles[j]
                    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < r1 + r2:
                        count += 1
            return count

        def objective(params):
            circles = params.reshape(-1, 3)
            # Calculate sum of radii (negative because we want to maximize)
            sum_radii = -np.sum(circles[:, 2])

            # Penalty for constraint violations
            penalty = 0

            # Boundary constraints
            x_coords = circles[:, 0]
            y_coords = circles[:, 1]
            radii = circles[:, 2]

            # Check containment violations
            containment_violations = (
                (x_coords - radii < BOUND_LOW) |
                (x_coords + radii > BOUND_UP) |
                (y_coords - radii < BOUND_LOW) |
                (y_coords + radii > BOUND_UP)
            )
            penalty += 10000 * np.sum(containment_violations)

            # Overlap penalties
            if len(circles) > 1:
                for i in range(len(circles)):
                    x1, y1, r1 = circles[i]
                    for j in range(i+1, len(circles)):
                        x2, y2, r2 = circles[j]
                        distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        if distance < r1 + r2:
                            penalty += 1000 * (r1 + r2 - distance)

            return sum_radii + penalty

        def constraint_handling(circles):
            """Handle boundary and overlap constraints manually"""
            circles_copy = circles.copy()
            for i in range(len(circles_copy)):
                x, y, r = circles_copy[i]
                # Fix containment
                x = max(r, min(1-r, x))
                y = max(r, min(1-r, y))
                circles_copy[i] = [x, y, r]
            return circles_copy

        def greedy_radius_expansion(circles, max_iter=50):
            """Greedy approach to expand radii while maintaining constraints"""
            circles_copy = circles.copy()
            changed = True
            iterations = 0

            while changed and iterations < max_iter:
                changed = False
                for i in range(len(circles_copy)):
                    x, y, r = circles_copy[i]
                    # Try to increase radius while staying within boundaries
                    max_possible_r = min(x, 1-x, y, 1-y)

                    # Check for overlap violations with others
                    can_expand = True
                    for j in range(len(circles_copy)):
                        if i != j:
                            x2, y2, r2 = circles_copy[j]
                            distance = math.sqrt((x - x2)**2 + (y - y2)**2)
                            if distance < r + r2:  # Would create overlap
                                can_expand = False
                                break

                    if can_expand and r < max_possible_r:
                        # Increase radius slightly
                        new_r = min(max_possible_r, r + 0.001)
                        if new_r > r + 1e-6:
                            circles_copy[i] = [x, y, new_r]
                            changed = True

                iterations += 1
            return circles_copy

        def physics_repulsion_refinement(circles, max_iter=50):
            """Apply physics-inspired repulsion to resolve overlaps"""
            circles_copy = circles.copy()
            for iteration in range(max_iter):
                # Compute forces from all other circles on each circle
                forces = np.zeros_like(circles_copy)

                for i in range(len(circles_copy)):
                    x1, y1, r1 = circles_copy[i]
                    fx, fy = 0, 0

                    for j in range(len(circles_copy)):
                        if i != j:
                            x2, y2, r2 = circles_copy[j]
                            dx = x2 - x1
                            dy = y2 - y1
                            distance = max(1e-8, math.sqrt(dx*dx + dy*dy))

                            # Force repulsion
                            if distance < (r1 + r2):
                                # Overlapping circles
                                force_magnitude = 1000 * (r1 + r2 - distance)
                                fx += force_magnitude * dx / distance
                                fy += force_magnitude * dy / distance

                            # Boundary forces to keep within square
                            if x1 - r1 < 0:
                                fx += 100 * (r1 - x1)
                            if x1 + r1 > 1:
                                fx += 100 * (1 - (x1 + r1))
                            if y1 - r1 < 0:
                                fy += 100 * (r1 - y1)
                            if y1 + r1 > 1:
                                fy += 100 * (1 - (y1 + r1))

                    forces[i] = [fx, fy, 0]  # We ignore radius changes for simplicity

                # Update positions
                circles_copy[:, :2] += 0.001 * forces[:, :2]

                # Apply boundary constraints to keep within unit square
                circles_copy[:, 0] = np.clip(circles_copy[:, 0],
                                            circles_copy[:, 2], 1 - circles_copy[:, 2])
                circles_copy[:, 1] = np.clip(circles_copy[:, 1],
                                            circles_copy[:, 2], 1 - circles_copy[:, 2])

            return circles_copy

        # Start with constraint handling
        circles_array = constraint_handling(circles_array)

        # Classify solution based on overlap severity
        overlap_count = count_overlap_violations(circles_array)

        # Apply refinement strategy based on overlap severity
        if overlap_count == 0:
            # No overlaps - use L-BFGS-B for fine-tuning
            # Flatten to optimize
            flat_params = circles_array.flatten()

            try:
                result = minimize(
                    objective,
                    flat_params,
                    method='L-BFGS-B',
                    bounds=[(0, 1) if i%3 != 2 else (0.001, 0.4) for i in range(len(flat_params))],
                    options={'maxiter': max_iter//2, 'ftol': 1e-6},
                    tol=1e-6
                )

                if result.success:
                    optimized_circles = result.x.reshape(-1, 3)
                    # Apply constraint fixes to results
                    optimized_circles = constraint_handling(optimized_circles)
                    return optimized_circles
            except Exception:
                pass

        elif overlap_count <= 5:
            # Low overlap - light refinement
            # First try simple L-BFGS if available
            flat_params = circles_array.flatten()
            try:
                result = minimize(
                    objective,
                    flat_params,
                    method='L-BFGS-B',
                    bounds=[(0, 1) if i%3 != 2 else (0.001, 0.4) for i in range(len(flat_params))],
                    options={'maxiter': max_iter//3, 'ftol': 1e-6},
                    tol=1e-6
                )

                if result.success:
                    optimized_circles = result.x.reshape(-1, 3)
                    # Apply constraint fixes to results
                    optimized_circles = constraint_handling(optimized_circles)
                    return optimized_circles
            except Exception:
                pass

            # Fall back to simple constraint handling and minimal optimization
            return constraint_handling(circles_array)

        elif overlap_count <= 15:
            # Medium overlap - moderate refinement
            # Apply physics repulsion to resolve overlaps
            refined = physics_repulsion_refinement(circles_array, max_iter//2)

            # Then do some local optimization
            flat_params = refined.flatten()
            try:
                result = minimize(
                    objective,
                    flat_params,
                    method='L-BFGS-B',
                    bounds=[(0, 1) if i%3 != 2 else (0.001, 0.4) for i in range(len(flat_params))],
                    options={'maxiter': max_iter//4, 'ftol': 1e-6},
                    tol=1e-6
                )

                if result.success:
                    optimized_circles = result.x.reshape(-1, 3)
                    # Apply constraint fixes to results
                    optimized_circles = constraint_handling(optimized_circles)
                    return optimized_circles
            except Exception:
                pass

            return refined

        else:
            # High overlap - intensive refinement
            # First perform greedy radius expansion
            expanded = greedy_radius_expansion(circles_array, max_iter//4)

            # Then apply physics repulsion
            repulsed = physics_repulsion_refinement(expanded, max_iter//3)

            # Finally optimize with L-BFGS
            flat_params = repulsed.flatten()
            try:
                result = minimize(
                    objective,
                    flat_params,
                    method='L-BFGS-B',
                    bounds=[(0, 1) if i%3 != 2 else (0.001, 0.4) for i in range(len(flat_params))],
                    options={'maxiter': max_iter//4, 'ftol': 1e-6},
                    tol=1e-6
                )

                if result.success:
                    optimized_circles = result.x.reshape(-1, 3)
                    # Apply constraint fixes to results
                    optimized_circles = constraint_handling(optimized_circles)
                    return optimized_circles
            except Exception:
                pass

            return repulsed

        # Fallback to original
        return circles_array

class EvolutionaryOptimizer:
    """Main evolutionary optimization class."""

    def __init__(self):
        self.toolbox = base.Toolbox()
        self._setup_toolbox()

    def _setup_toolbox(self):
        """Setup DEAP toolbox with registered components."""
        self.toolbox.register("individual", tools.initIterate, creator.Individual, Initializer.generate_grid_refined_initialization)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", CircleEvaluator.eval_circles)
        self.toolbox.register("mate", CrossoverHandler.adaptive_constraint_aware_crossover)
        self.toolbox.register("mutate", lambda ind, gen: MutationHandler.mutate_individual(ind, gen))
        self.toolbox.register("select", SelectionHandler.adaptive_tournament_selection)

    def optimize(self):
        """Run the evolutionary optimization process."""
        # Create population
        pop = self.toolbox.population(n=POP_SIZE)

        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Evolution loop with enhanced adaptive parameters
        for gen in range(NGEN):
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:  # Using constant mutation probability for now
                    self.toolbox.mutate(mutant, gen)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Elitism: keep best individuals
            if ELITISM_COUNT > 0:
                best_individuals = tools.selBest(pop, ELITISM_COUNT)
                offspring[:ELITISM_COUNT] = best_individuals

            # The population is entirely replaced by the offspring
            pop[:] = offspring

        # Find the best individual
        best_ind = tools.selBest(pop, 1)[0]
        circles = np.array(best_ind).reshape(-1, 3)

        return circles

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create and run the optimizer
    optimizer = EvolutionaryOptimizer()
    circles = optimizer.optimize()

    # Apply enhanced local refinement to improve the final solution
    circles = LocalRefiner.hierarchical_local_refinement(circles)

    return circles

# EVOLVE-BLOCK-END