# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import random
from deap import base, creator, tools, algorithms
import time
from numba import jit
import copy
from sklearn.cluster import KMeans

# Global constants
POP_SIZE = 150
GENERATIONS = 80
MUT_PB = 0.15
CROSSOVER_PB = 0.8
TOURNAMENT_SIZE = 3
BENCHMARK = 2.6358627564136983
N_CIRCLES = 26

# Initialize DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0.0, 1.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=N_CIRCLES*3)  # 26*3 = 78
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

    # Normalize radii to be within valid bounds [0.001, 0.5]
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

    # Repair overlap constraints - more sophisticated approach
    for iter_count in range(5):  # Multiple iterations for better resolution
        any_changes = False
        for i in range(len(circles)):
            x1, y1, r1 = circles[i]
            for j in range(i):
                x2, y2, r2 = circles[j]
                if check_collision(x1, y1, r1, x2, y2, r2):
                    # Calculate overlap distance
                    dx = x2 - x1
                    dy = y2 - y1
                    dist = np.sqrt(dx*dx + dy*dy)

                    if dist > 0:
                        # Separate circles along the line connecting their centers
                        overlap = (r1 + r2) - dist
                        if overlap > 0:
                            # Move circles apart
                            move_amount = overlap * 0.5
                            dx_norm = dx / dist
                            dy_norm = dy / dist

                            circles[i][0] -= dx_norm * move_amount
                            circles[i][1] -= dy_norm * move_amount
                            circles[j][0] += dx_norm * move_amount
                            circles[j][1] += dy_norm * move_amount

                            any_changes = True

        if not any_changes:
            break

    # Final boundary corrections
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Ensure containment
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        circles[i][0] = x
        circles[i][1] = y

    # Flatten back to individual
    return list(circles.flatten())

def initialize_population(pop_size, n_circles=26):
    """Initialize population with better starting solutions using Voronoi-inspired placement"""
    def create_voronoi_initialization():
        # Generate initial points using k-means clustering to distribute well
        points = np.random.rand(200, 2)
        kmeans = KMeans(n_clusters=n_circles, random_state=42, n_init=10)
        kmeans.fit(points)
        centroids = kmeans.cluster_centers_

        circles = []
        for i, (cx, cy) in enumerate(centroids):
            # Compute max radius at this position
            max_r = min(cx, cy, 1-cx, 1-cy)
            if max_r < 0.01:
                continue

            # Place with reasonable radius
            r = random.uniform(max_r/4, max_r/2)
            circles.append((cx, cy, r))

        # If we don't have enough circles, fill in with additional positions
        while len(circles) < n_circles:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            max_r = min(x, y, 1-x, 1-y)
            if max_r > 0.01:
                r = random.uniform(max_r/4, max_r/2)
                circles.append((x, y, r))

        # Trim to exact number needed
        circles = circles[:n_circles]

        return circles

    population = []
    for _ in range(pop_size):
        circles = create_voronoi_initialization()
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

    for i in range(point1*3, point2*3):  # Adjust indices to account for 3 components per circle
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

def local_optimize_solution(circles):
    """Apply local optimization to further improve solution"""
    circles = np.array(circles)

    # Simple gradient descent-style optimization
    learning_rate = 0.01
    max_iterations = 100

    for iteration in range(max_iterations):
        improved = False
        # Try moving each circle slightly
        for i in range(len(circles)):
            x, y, r = circles[i]

            # Calculate forces from nearby circles
            force_x, force_y = 0.0, 0.0
            for j in range(len(circles)):
                if i != j:
                    x2, y2, r2 = circles[j]
                    dx = x - x2
                    dy = y - y2
                    dist = np.sqrt(dx*dx + dy*dy)

                    if dist < r + r2:  # Collision or near collision
                        # Repel force (if overlapping)
                        if dist > 0.001:
                            force_x += dx / dist * (r + r2 - dist) * 0.1
                            force_y += dy / dist * (r + r2 - dist) * 0.1
                    elif dist < (r + r2 + 0.01):  # Near contact
                        # Slight repulsion
                        if dist > 0.001:
                            force_x -= dx / dist * (r + r2 + 0.01 - dist) * 0.01
                            force_y -= dy / dist * (r + r2 + 0.01 - dist) * 0.01

        # Apply forces
        if abs(force_x) > 0.0001 or abs(force_y) > 0.0001:
            improved = True
            # Move circle
            circles[i][0] = np.clip(x + force_x * learning_rate, r, 1-r)
            circles[i][1] = np.clip(y + force_y * learning_rate, r, 1-r)

        # If no improvement, break early
        if not improved:
            break

    # Final validity repair
    circles = repair_individual(circles.flatten())
    circles = np.array(circles).reshape(-1, 3)

    return circles

def run_evolution():
    """Main evolutionary algorithm with adaptive parameters"""
    # Initialize population
    pop = initialize_population(POP_SIZE)

    # Register functions with toolbox
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", cxTwoPointCustom)
    toolbox.register("mutate", mutGaussianCustom, mu=0, sigma=0.03, indpb=0.1)
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
            ngen=GENERATIONS, stats=stats, halloffame=hof, verbose=False
        )
    except Exception as e:
        print(f"Evolution error: {e}")
        # Fallback to simpler approach
        return None

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
            circles = np.zeros((N_CIRCLES, 3))
            return circles

        # Decode the best individual
        circles = decode_individual(final_individual)

        # Final validation step
        circles = np.array(circles)
        if not check_validity_jit(circles):
            # Apply final repair
            circles = repair_individual(final_individual)
            circles = np.array(circles).reshape(-1, 3)

        # Apply local optimization
        circles = local_optimize_solution(circles)

        # Ensure correct shape
        while len(circles) < N_CIRCLES:
            circles = np.vstack([circles, [0.5, 0.5, 0.01]])

        if len(circles) > N_CIRCLES:
            circles = circles[:N_CIRCLES]

        # Ensure all circles have valid radii
        circles[:, 2] = np.maximum(0.001, circles[:, 2])

        print(f"Total evaluation time: {time.time() - start_time:.2f}s")
        print(f"Sum of radii: {np.sum(circles[:, 2]):.6f}")
        print(f"Benchmark ratio: {np.sum(circles[:, 2]) / BENCHMARK:.6f}")

        return circles

    except Exception as e:
        print(f"Unexpected error in circle_packing26: {e}")
        # Return a basic fallback
        circles = np.zeros((N_CIRCLES, 3))
        return circles

# EVOLVE-BLOCK-END