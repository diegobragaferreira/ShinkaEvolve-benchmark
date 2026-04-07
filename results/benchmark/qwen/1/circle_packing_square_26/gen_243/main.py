# EVOLVE-BLOCK-START
import numpy as np
from deap import base, creator, tools, algorithms
import random
import math
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Uses adaptive Voronoi-based initialization combined with evolutionary algorithm.
    Implements constraint-aware operators and local optimization.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    N_CIRCLES = 26
    POP_SIZE = 80
    GEN_COUNT = 60
    MUTPB = 0.25
    CXPB = 0.4

    # Create fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Generate initial Voronoi-based configuration
    initial_circles = generate_voronoi_initial_config(N_CIRCLES)

    def eval_circle_placement(individual):
        """Evaluate placement fitness with penalty for constraints"""
        circles = np.array(individual).reshape(-1, 3)

        # Extract positions and radii
        positions = circles[:, :2]
        radii = circles[:, 2]

        # Calculate total radius (objective)
        total_radius = np.sum(radii)

        # Penalty for constraint violations
        penalty = 0

        # Check containment constraints
        for i, (pos, r) in enumerate(zip(positions, radii)):
            x, y = pos
            # Circle must be fully inside unit square
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                penalty += 10000

        # Check overlap constraints using KDTree for efficiency
        try:
            tree = cKDTree(positions)
            pairs = tree.query_pairs(radii.sum() + 0.001, p=2)
            for i, j in pairs:
                r_i = radii[i]
                r_j = radii[j]
                pos_i = positions[i]
                pos_j = positions[j]

                # Distance between centers
                dist = np.sqrt(np.sum((pos_i - pos_j)**2))

                # Must not overlap (distance >= sum of radii)
                if dist < (r_i + r_j):
                    penalty += 1000 * (r_i + r_j - dist)
        except:
            # Fallback to brute force if KDTree fails
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    pos_i = circles[i, :2]
                    pos_j = circles[j, :2]
                    r_i = circles[i, 2]
                    r_j = circles[j, 2]

                    # Distance between centers
                    dist = np.sqrt(np.sum((pos_i - pos_j)**2))

                    # Must not overlap (distance >= sum of radii)
                    if dist < (r_i + r_j):
                        penalty += 1000 * (r_i + r_j - dist)

        # Return fitness (total_radius - penalty)
        return (total_radius - penalty,)

    def mutate_circle(individual):
        """Mutate a circle placement with adaptive parameters"""
        # Calculate population diversity for adaptive mutation
        individual_array = np.array(individual).reshape(-1, 3)
        radii = individual_array[:, 2]
        diversity = np.std(radii) / (np.mean(radii) + 1e-8) if np.mean(radii) > 1e-8 else 0

        # Adaptive mutation rate based on diversity
        adaptive_mutation_rate = MUTPB * (1 - min(0.8, diversity))

        for i in range(len(individual)):
            # Mutate with adaptive probability
            if random.random() < adaptive_mutation_rate:
                idx = i % 3
                if idx == 2:  # radius index
                    old_r = individual[i]
                    # Smaller mutations for larger radii to maintain feasibility
                    mutation_strength = 0.015 * (1 + diversity)
                    new_r = old_r + random.gauss(0, mutation_strength)
                    individual[i] = max(0.001, min(0.5, new_r))
                else:  # position indices (x, y)
                    old_val = individual[i]
                    # Mutate position with adaptive strength
                    mutation_strength = 0.02 * (1 + diversity)
                    new_val = old_val + random.gauss(0, mutation_strength)
                    individual[i] = max(0, min(1, new_val))
        return individual,

    def create_individual():
        """Create a random valid individual"""
        # Start with a slightly perturbed version of initial configuration
        individual = initial_circles.flatten().tolist()
        for i in range(len(individual)):
            if i % 3 < 2:  # x or y coordinate
                # Small random perturbation
                individual[i] += random.uniform(-0.02, 0.02)
                individual[i] = max(0, min(1, individual[i]))
            else:  # radius
                # Small random perturbation
                individual[i] *= random.uniform(0.9, 1.1)
                individual[i] = max(0.001, min(0.5, individual[i]))
        return creator.Individual(individual)

    def cx_constraint_aware(ind1, ind2):
        """Crossover that maintains constraints with enhanced repair"""
        # Perform uniform crossover with higher recombination rate
        tools.cxUniform(ind1, ind2, indpb=0.7)

        # More thorough repair of offspring
        temp_ind = np.array(ind1).reshape(-1, 3)

        # Enhanced containment repair with better boundary handling
        for i in range(len(temp_ind)):
            x, y, r = temp_ind[i]
            # More robust boundary checking and fixing
            original_r = r
            if x - r < 0:
                x = r + 0.001
            elif x + r > 1:
                x = 1 - r - 0.001
            if y - r < 0:
                y = r + 0.001
            elif y + r > 1:
                y = 1 - r - 0.001

            # Ensure radius is still valid after position adjustment
            max_radius = min(x, 1-x, y, 1-y) * 0.99
            if r > max_radius:
                r = max_radius

            temp_ind[i] = [x, y, r]

        # Enhanced overlap resolution with iterative improvement
        for iteration in range(10):  # Allow a few rounds of overlap fixing
            any_changes = False
            for i in range(len(temp_ind)):
                for j in range(i+1, len(temp_ind)):
                    pos_i = temp_ind[i, :2]
                    pos_j = temp_ind[j, :2]
                    r_i = temp_ind[i, 2]
                    r_j = temp_ind[j, 2]
                    dist = np.sqrt(np.sum((pos_i - pos_j)**2))

                    if dist < (r_i + r_j):
                        any_changes = True
                        # Move both circles apart using more sophisticated method
                        dx, dy = pos_i - pos_j
                        total_dist = np.sqrt(dx*dx + dy*dy) + 1e-8
                        dx /= total_dist
                        dy /= total_dist

                        # Calculate overlap amount and separate by that amount
                        overlap = (r_i + r_j) - dist
                        separation = overlap * 0.5

                        # Apply movement to both circles, but prefer moving the smaller one
                        # Move smaller circle more aggressively
                        smaller_idx = i if r_i < r_j else j
                        larger_idx = j if r_i < r_j else i

                        if smaller_idx == i:
                            new_x = max(r_i, min(1-r_i, pos_i[0] + dx * separation))
                            new_y = max(r_i, min(1-r_i, pos_i[1] + dy * separation))
                            temp_ind[i] = [new_x, new_y, r_i]
                        else:
                            new_x = max(r_j, min(1-r_j, pos_i[0] - dx * separation))
                            new_y = max(r_j, min(1-r_j, pos_i[1] - dy * separation))
                            temp_ind[i] = [new_x, new_y, r_i]

                        if larger_idx == j:
                            new_x = max(r_j, min(1-r_j, pos_j[0] - dx * separation))
                            new_y = max(r_j, min(1-r_j, pos_j[1] - dy * separation))
                            temp_ind[j] = [new_x, new_y, r_j]
                        else:
                            new_x = max(r_i, min(1-r_i, pos_j[0] + dx * separation))
                            new_y = max(r_i, min(1-r_i, pos_j[1] + dy * separation))
                            temp_ind[j] = [new_x, new_y, r_j]

            if not any_changes:
                break

        # Return repaired individuals
        ind1[:] = temp_ind.flatten()
        return ind1, ind2

    # Initialize toolbox
    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_circle_placement)
    toolbox.register("mate", cx_constraint_aware)
    toolbox.register("mutate", mutate_circle)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create initial population
    population = toolbox.population(n=POP_SIZE)

    # Run evolution
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    try:
        population, logbook = algorithms.eaSimple(
            population, toolbox, cxpb=CXPB, mutpb=MUTPB,
            ngen=GEN_COUNT, stats=stats, halloffame=hof, verbose=False
        )
    except Exception as e:
        # Fallback to simple heuristic if GA fails
        print(f"GA failed with error: {e}")
        return heuristic_circle_packing()

    # Return best solution
    best_individual = hof[0]
    result = np.array(best_individual).reshape(-1, 3)

    # Apply local optimization to refine further
    refined_result = local_optimization(result.copy())

    return refined_result

def generate_voronoi_initial_config(n):
    """Generate initial circle configuration based on enhanced Voronoi-like principles"""
    circles = np.zeros((n, 3))

    # Create a more sophisticated hexagonal grid pattern for better distribution
    # This creates a honeycomb-like structure with optimized spacing
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))

    # Calculate spacing for hexagonal packing
    if rows * cols < n:
        rows += 1

    # Determine spacing to fill the unit square
    spacing_x = 0.9 / (cols + 1) if cols > 0 else 0.1
    spacing_y = 0.9 / (rows + 1) if rows > 0 else 0.1

    # Generate points in hexagonal pattern
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= n:
                break
            # Hexagonal offset pattern for better coverage
            x_offset = 0 if i % 2 == 0 else spacing_x / 2
            x = (j + 1) * spacing_x + x_offset + random.uniform(-spacing_x/6, spacing_x/6)
            y = (i + 1) * spacing_y + random.uniform(-spacing_y/6, spacing_y/6)

            # Initial radius based on local density considerations
            # Larger radii in sparser regions, smaller in denser regions
            r = min(spacing_x, spacing_y) * random.uniform(0.25, 0.45)

            circles[count] = [x, y, r]
            count += 1

    # Ensure all circles are within bounds and adjust radii accordingly
    for i in range(n):
        x, y, r = circles[i]
        # Bound radii to stay within square
        bound_r = min(r, x, 1-x, y, 1-y)
        # Use a more conservative approach to avoid boundary issues
        circles[i] = [x, y, bound_r * 0.95]

    return circles

def local_optimization(circles):
    """Refine the solution with enhanced local optimization"""
    # More aggressive optimization with multiple phases
    for iteration in range(300):  # Increase iterations for better convergence
        improved = False

        # Phase 1: Aggressive radius maximization
        for i in range(len(circles)):
            # Try to increase each circle's radius significantly
            original_r = circles[i, 2]
            max_increase = min(
                circles[i, 0], 1 - circles[i, 0],
                circles[i, 1], 1 - circles[i, 1]
            ) - original_r

            if max_increase > 0.001:  # Only optimize if there's room
                # More precise binary search with fine granularity
                low = 0
                high = max_increase
                best_radius = original_r

                for _ in range(15):  # More iterations for precision
                    test_r = (low + high) / 2
                    test_r = min(test_r, max_increase)

                    # Check if this change is feasible
                    valid = True
                    test_pos = circles[i, :2]
                    test_r_new = original_r + test_r

                    # Check overlap with all other circles (more comprehensive)
                    for j in range(len(circles)):
                        if i != j:
                            pos_j = circles[j, :2]
                            r_j = circles[j, 2]
                            dist = np.sqrt(np.sum((test_pos - pos_j)**2))
                            if dist < (test_r_new + r_j):
                                valid = False
                                break

                    if valid:
                        best_radius = original_r + test_r
                        low = test_r
                    else:
                        high = test_r

                if best_radius > original_r + 0.0001:  # Only update if significant improvement
                    circles[i, 2] = best_radius
                    improved = True

        # Phase 2: Position refinement for overlap reduction
        if not improved:
            for i in range(len(circles)):
                # Try to improve circle positioning
                original_pos = circles[i, :2].copy()
                best_pos = original_pos.copy()
                best_radius = circles[i, 2]
                best_score = best_radius

                # Try a more exhaustive grid search around current location
                search_grid = [-0.03, -0.015, 0, 0.015, 0.03]
                for dx in search_grid:
                    for dy in search_grid:
                        test_x = max(0.01, min(0.99, circles[i, 0] + dx))
                        test_y = max(0.01, min(0.99, circles[i, 1] + dy))

                        # Test if this position is valid
                        valid = True
                        test_r = circles[i, 2]

                        # Check overlap with other circles
                        for j in range(len(circles)):
                            if i != j:
                                pos_j = circles[j, :2]
                                r_j = circles[j, 2]
                                dist = np.sqrt((test_x - pos_j[0])**2 + (test_y - pos_j[1])**2)
                                if dist < (test_r + r_j):
                                    valid = False
                                    break

                        if valid:
                            score = test_r  # Focus on radius maximization
                            if score > best_score:
                                best_score = score
                                best_pos = [test_x, test_y]

                # Apply improvement if any
                if best_score > circles[i, 2] + 0.0001 or not np.array_equal(best_pos, original_pos):
                    circles[i, :2] = best_pos
                    improved = True

        # Terminate if no improvement for a while
        if iteration > 100 and not improved:
            break

    return circles

def heuristic_circle_packing() -> np.ndarray:
    """Fallback method using a more structured approach"""
    # Simple grid-based arrangement with some refinement
    n = 26
    circles = np.zeros((n, 3))

    # Try a hexagonal packing pattern
    rows = 5
    cols = 5
    if n < rows * cols:
        rows = math.ceil(n / cols)

    # Create regular grid points
    spacing_x = 0.9 / (cols + 1)
    spacing_y = 0.9 / (rows + 1)

    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= n:
                break
            x = (j + 1) * spacing_x
            y = (i + 1) * spacing_y
            # Set reasonable initial radius
            r = min(spacing_x, spacing_y) * 0.4
            circles[count] = [x, y, r]
            count += 1

    # Refine positions to avoid overlaps
    # Simple iterative improvement
    for _ in range(100):
        improved = False
        for i in range(n):
            # Try to move circle to reduce overlaps
            best_pos = circles[i, :2].copy()
            best_rad = circles[i, 2]
            best_score = -1000

            # Check nearby positions
            for dx in [-0.02, -0.01, 0, 0.01, 0.02]:
                for dy in [-0.02, -0.01, 0, 0.01, 0.02]:
                    test_x = max(0.01, min(0.99, circles[i, 0] + dx))
                    test_y = max(0.01, min(0.99, circles[i, 1] + dy))
                    test_r = circles[i, 2]

                    # Check constraint violations
                    valid = True
                    for j in range(n):
                        if i != j:
                            dist = np.sqrt((test_x - circles[j, 0])**2 + (test_y - circles[j, 1])**2)
                            if dist < (test_r + circles[j, 2]):
                                valid = False
                                break

                    if valid:
                        score = test_r  # Just maximize radius for now
                        if score > best_score:
                            best_score = score
                            best_pos = [test_x, test_y]

            # Apply best improvement if found
            if best_score > circles[i, 2]:
                circles[i, :2] = best_pos
                circles[i, 2] = best_score
                improved = True

        if not improved:
            break

    return circles


# EVOLVE-BLOCK-END