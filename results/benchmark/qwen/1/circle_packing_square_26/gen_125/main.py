# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import random
from deap import base, creator, tools, algorithms
import time
from numba import jit
import copy

# Global constants
POP_SIZE = 100
GENERATIONS = 50
MUT_PB = 0.1
CROSSOVER_PB = 0.8
TOURNAMENT_SIZE = 3
BENCHMARK = 2.6358627564136983

# Initialize DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0.0, 1.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=78)  # 26*3 = 78
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

@jit(nopython=True)
def check_validity_jit(circles_np):
    """Fast validity check using Numba"""
    n = len(circles_np)
    for i in range(n):
        x, y, r = circles_np[i]
        # Check containment
        if r > x or r > y or r > 1-x or r > 1-y:
            return False
        # Check overlap with all previous circles
        for j in range(i):
            x2, y2, r2 = circles_np[j]
            dx = x - x2
            dy = y - y2
            dist_sq = dx*dx + dy*dy
            min_dist_sq = (r+r2)*(r+r2)
            if dist_sq < min_dist_sq:
                return False
    return True

def decode_individual(individual):
    """Convert individual to circles array"""
    circles = np.array(individual).reshape(-1, 3)
    return circles

def evaluate(individual):
    """Evaluate fitness of individual"""
    circles = decode_individual(individual)

    # Normalize radii to be within valid bounds [0, 0.5]
    # We'll use a heuristic to scale them properly
    for i in range(len(circles)):
        circles[i][2] = max(0.001, min(0.5, circles[i][2]))

    # Check if valid
    if not check_validity_jit(circles):
        return (0.0,)

    # Return sum of radii
    total_radius = np.sum(circles[:, 2])
    return (total_radius,)

def check_collision(x1, y1, r1, x2, y2, r2):
    """Check if two circles collide"""
    dx = x1 - x2
    dy = y1 - y2
    distance_squared = dx*dx + dy*dy
    min_distance_squared = (r1 + r2) * (r1 + r2)
    return distance_squared < min_distance_squared

def constraint_violation(x, y, r):
    """Return constraint violations for a circle"""
    violations = []
    # Check containment
    if r > x:
        violations.append(r - x)
    if r > y:
        violations.append(r - y)
    if r > 1 - x:
        violations.append(r - (1 - x))
    if r > 1 - y:
        violations.append(r - (1 - y))
    return violations

def repair_individual(individual):
    """Repair an individual to satisfy constraints"""
    circles = decode_individual(individual)

    # Repair containment constraints
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Adjust radius if necessary
        min_radius = min(x, y, 1-x, 1-y)
        if r > min_radius:
            circles[i][2] = min_radius

        # Ensure radius is positive
        if circles[i][2] <= 0:
            circles[i][2] = 0.001

    # Repair overlap constraints
    for i in range(len(circles)):
        x1, y1, r1 = circles[i]
        for j in range(i):
            x2, y2, r2 = circles[j]
            if check_collision(x1, y1, r1, x2, y2, r2):
                # Reduce radii to resolve collision
                # Simple approach: reduce both radii proportionally
                overlap_dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                needed_separation = r1 + r2
                excess = overlap_dist - needed_separation

                if excess > 0:
                    reduction = excess / 2
                    circles[i][2] = max(0.001, circles[i][2] - reduction)
                    circles[j][2] = max(0.001, circles[j][2] - reduction)

    # Flatten back to individual
    return list(circles.flatten())

def initialize_population(pop_size, n_circles=26):
    """Initialize population with enhanced Voronoi-based starting solutions"""
    def create_enhanced_voronoi_placement():
        """Create better distributed initial circle positions using enhanced Voronoi approach"""
        import numpy as np
        from scipy.spatial import Voronoi, distance
        from scipy.spatial.distance import cdist

        # Generate initial points using a more structured approach
        # Create a grid with some randomness to avoid regular patterns
        points_per_side = max(5, int(np.ceil(np.sqrt(n_circles)) + 2))
        x_coords = np.linspace(0.05, 0.95, points_per_side)
        y_coords = np.linspace(0.05, 0.95, points_per_side)

        # Create grid points with slight jitter
        initial_points = []
        for x in x_coords:
            for y in y_coords:
                # Add slight jitter to make distribution less regular
                jitter_x = random.uniform(-0.02, 0.02)
                jitter_y = random.uniform(-0.02, 0.02)
                new_x = np.clip(x + jitter_x, 0.05, 0.95)
                new_y = np.clip(y + jitter_y, 0.05, 0.95)
                initial_points.append([new_x, new_y])

        # If we don't have enough points, add random ones
        if len(initial_points) < n_circles:
            extra_points = n_circles - len(initial_points)
            for _ in range(extra_points):
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
                initial_points.append([x, y])

        initial_points = np.array(initial_points[:n_circles])

        # Apply Lloyd relaxation to improve point distribution
        # This helps create more uniform spacing between points
        for _ in range(5):  # Run a few iterations of Lloyd relaxation
            try:
                vor = Voronoi(initial_points)
                # Calculate centroids of Voronoi regions
                new_points = []
                for i in range(len(initial_points)):
                    # Get Voronoi vertices for point i
                    region = vor.point_region[i]
                    if region != len(vor.regions) or region == -1:
                        # Skip infinite regions
                        new_points.append(initial_points[i])
                        continue

                    vertices = vor.regions[region]
                    if not vertices or -1 in vertices:
                        new_points.append(initial_points[i])
                        continue

                    # Compute centroid of the polygon formed by vertices
                    vertex_points = [vor.vertices[v] for v in vertices if v >= 0]
                    if len(vertex_points) < 3:
                        new_points.append(initial_points[i])
                        continue

                    # Compute centroid
                    centroid_x = np.mean([v[0] for v in vertex_points])
                    centroid_y = np.mean([v[1] for v in vertex_points])

                    # Clip to keep within bounds
                    centroid_x = np.clip(centroid_x, 0.05, 0.95)
                    centroid_y = np.clip(centroid_y, 0.05, 0.95)
                    new_points.append([centroid_x, centroid_y])

                initial_points = np.array(new_points)
            except:
                # If Voronoi fails, just use original points
                break

        # Now create circles from these points
        circles = []
        for x, y in initial_points:
            # Compute max possible radius
            max_r = min(x, y, 1-x, 1-y)
            if max_r > 0.01:
                # Use a reasonable fraction of max radius
                r = random.uniform(max_r * 0.3, max_r * 0.7)
                circles.append((x, y, r))

        # If we don't have enough circles, add more
        if len(circles) < n_circles:
            # Use grid approach for remaining circles
            rows = int(np.ceil(np.sqrt(n_circles - len(circles))))
            cols = int(np.ceil((n_circles - len(circles)) / rows))
            spacing_x = 1.0 / (cols + 2)
            spacing_y = 1.0 / (rows + 2)

            for i in range(n_circles - len(circles)):
                row = i // cols
                col = i % cols
                x = (col + 1.5) * spacing_x
                y = (row + 1.5) * spacing_y
                max_r = min(x, y, 1-x, 1-y)
                if max_r > 0.01:
                    r = random.uniform(max_r * 0.4, max_r * 0.6)
                    circles.append((x, y, r))

        # Ensure we have exactly n_circles
        while len(circles) < n_circles:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            max_r = min(x, y, 1-x, 1-y)
            if max_r > 0.01:
                r = random.uniform(max_r * 0.2, max_r * 0.5)
                circles.append((x, y, r))

        circles = circles[:n_circles]
        return circles

    population = []
    for _ in range(pop_size):
        try:
            circles = create_enhanced_voronoi_placement()
            individual = []
            for x, y, r in circles:
                individual.extend([x, y, r])
            population.append(individual)
        except Exception as e:
            # Fallback to simple approach if enhanced method fails
            circles = []
            rows = int(np.ceil(np.sqrt(n_circles)))
            cols = int(np.ceil(n_circles / rows))
            spacing_x = 1.0 / (cols + 1)
            spacing_y = 1.0 / (rows + 1)

            for i in range(n_circles):
                row = i // cols
                col = i % cols
                x = (col + 1) * spacing_x
                y = (row + 1) * spacing_y
                max_r = min(x, y, 1-x, 1-y)
                if max_r > 0.01:
                    r = max_r * 0.3
                    circles.append((x, y, r))
                else:
                    circles.append((0.5, 0.5, 0.01))

            individual = []
            for x, y, r in circles:
                individual.extend([x, y, r])
            population.append(individual)

    return population

def cxTwoPointCustom(ind1, ind2):
    """Custom two-point crossover that respects the circle structure"""
    if random.random() > CROSSOVER_PB:
        return ind1, ind2

    size = len(ind1)
    point1 = random.randint(1, size//3 - 1)
    point2 = random.randint(point1, size//3 - 1)

    for i in range(point1, point2):
        ind1[i], ind2[i] = ind2[i], ind1[i]

    # Each circle needs to remain valid after crossover
    # So we do a simple repair after crossover
    ind1_repaired = repair_individual(ind1)
    ind2_repaired = repair_individual(ind2)

    return ind1_repaired, ind2_repaired

def mutGaussianCustom(individual, mu, sigma, indpb):
    """Custom Gaussian mutation that respects constraints"""
    if random.random() > MUT_PB:
        return individual,

    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            # For positions (0,1,2,3...): add Gaussian noise
            # For radii (2,5,8,...): ensure they stay in [0.001, 0.5]
            if i % 3 == 2:  # This is a radius index
                old_r = individual[i]
                new_r = old_r + random.gauss(mu, sigma)
                individual[i] = max(0.001, min(0.5, new_r))  # Clamp to valid range
            else:  # This is a coordinate (x or y)
                old_coord = individual[i]
                new_coord = old_coord + random.gauss(mu, sigma)
                individual[i] = max(0.0, min(1.0, new_coord))  # Clamp to unit square

    # Repair individual to ensure it's valid
    repaired = repair_individual(individual)
    return repaired,

def constraint_aware_local_search(circles, max_iter=100):
    """
    Apply constraint-aware local search to improve circle configurations.
    This performs gradient-based refinement while respecting containment and overlap constraints.
    """
    circles = np.array(circles)

    for iteration in range(max_iter):
        improved = False

        # Calculate forces between all pairs of circles
        forces = np.zeros_like(circles[:, :2])

        # Compute repulsion forces
        for i in range(len(circles)):
            x1, y1, r1 = circles[i]
            force_x, force_y = 0.0, 0.0

            for j in range(len(circles)):
                if i != j:
                    x2, y2, r2 = circles[j]
                    dx = x1 - x2
                    dy = y1 - y2
                    dist_sq = dx*dx + dy*dy
                    dist = np.sqrt(dist_sq)

                    if dist < r1 + r2:  # Collision
                        # Strong repulsion force
                        if dist > 0.001:
                            force_mag = (r1 + r2 - dist) * 10.0
                            force_x += dx / dist * force_mag
                            force_y += dy / dist * force_mag
                        improved = True
                    elif dist < r1 + r2 + 0.01:  # Near collision
                        # Moderate repulsion
                        if dist > 0.001:
                            force_mag = (r1 + r2 + 0.01 - dist) * 2.0
                            force_x -= dx / dist * force_mag
                            force_y -= dy / dist * force_mag
                        improved = True

            # Apply force to center of circle
            forces[i] = [force_x, force_y]

        # Apply forces with momentum
        learning_rate = 0.01
        for i in range(len(circles)):
            x, y, r = circles[i]
            dx, dy = forces[i]

            # Move circle
            new_x = x + dx * learning_rate
            new_y = y + dy * learning_rate

            # Keep within bounds
            new_x = np.clip(new_x, r, 1-r)
            new_y = np.clip(new_y, r, 1-r)

            # Update position
            circles[i][0] = new_x
            circles[i][1] = new_y

            # Attempt to slightly increase radius if space allows
            # but only if we're not currently in collision
            if dist > r1 + r2 + 0.01:
                max_radius = min(new_x, new_y, 1-new_x, 1-new_y)
                if r < max_radius and r < 0.5:
                    circles[i][2] = min(max_radius, r + 0.001)

        # Boundary correction
        for i in range(len(circles)):
            x, y, r = circles[i]
            circles[i][0] = np.clip(x, r, 1-r)
            circles[i][1] = np.clip(y, r, 1-r)

        # Early termination if no significant improvement
        if not improved and iteration > 20:
            break

    return circles

def run_evolution():
    """Main evolutionary algorithm with local search enhancement"""
    # Initialize population
    pop = initialize_population(POP_SIZE)

    # Register functions with toolbox
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", cxTwoPointCustom)
    toolbox.register("mutate", mutGaussianCustom, mu=0, sigma=0.05, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run evolution
    hof = tools.HallOfFame(1)

    try:
        pop, logbook = algorithms.eaSimple(
            pop, toolbox, cxpb=CROSSOVER_PB, mutpb=MUT_PB,
            ngen=GENERATIONS, stats=stats, halloffame=hof, verbose=True
        )
    except Exception as e:
        print(f"Evolution error: {e}")
        # Fallback to simpler approach
        return None

    # Apply local search to the best individual
    if len(hof) > 0:
        best_individual = hof[0]
        circles = decode_individual(best_individual)
        refined_circles = constraint_aware_local_search(circles)
        return refined_circles.flatten().tolist()

    return hof[0]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    start_time = time.time()
    try:
        # Run the evolution
        final_individual = run_evolution()

        if final_individual is None:
            # If evolution failed, return a fallback solution
            circles = np.zeros((26, 3))
            return circles

        # Decode the best individual
        circles = decode_individual(final_individual)

        # Final validation step
        circles = np.array(circles)
        if not check_validity_jit(circles):
            # Apply final repair
            circles = repair_individual(final_individual)
            circles = np.array(circles).reshape(-1, 3)

        # Ensure correct shape
        while len(circles) < 26:
            circles = np.vstack([circles, [0.5, 0.5, 0.01]])

        if len(circles) > 26:
            circles = circles[:26]

        # Ensure all circles have valid radii
        circles[:, 2] = np.maximum(0.001, circles[:, 2])

        print(f"Total evaluation time: {time.time() - start_time:.2f}s")
        print(f"Sum of radii: {np.sum(circles[:, 2]):.6f}")
        print(f"Benchmark ratio: {np.sum(circles[:, 2]) / BENCHMARK:.6f}")

        return circles

    except Exception as e:
        print(f"Unexpected error in circle_packing26: {e}")
        # Return a basic fallback
        circles = np.zeros((26, 3))
        return circles

# EVOLVE-BLOCK-END