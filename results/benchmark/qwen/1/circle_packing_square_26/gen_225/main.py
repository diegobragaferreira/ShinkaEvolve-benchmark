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
GENERATIONS = 100
MUT_PB = 0.15
CROSSOVER_PB = 0.8
TOURNAMENT_SIZE = 3
BENCHMARK = 2.6358627564136983
N_CIRCLES = 26

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Initialize DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0.0, 1.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=N_CIRCLES*3)  # 26*3 = 78
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

@jit(nopython=True)
def check_validity_jit(circles_np):
    """Fast validity check using Numba with early termination"""
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

def repair_individual(individual):
    """Repair an individual to satisfy constraints with improved algorithm"""
    circles = decode_individual(individual)

    # Repair containment constraints first
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Adjust radius if necessary
        min_radius = min(x, y, 1-x, 1-y)
        if r > min_radius:
            circles[i][2] = min_radius

        # Ensure radius is positive
        if circles[i][2] <= 0:
            circles[i][2] = 0.001

    # Repair overlap constraints using iterative approach with better separation
    max_iter = 10
    for iter_count in range(max_iter):  # Multiple iterations for better resolution
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
                            # Move circles apart with better weight distribution
                            move_amount = overlap * 0.5
                            dx_norm = dx / dist
                            dy_norm = dy / dist

                            # Weighted separation based on radii
                            w1 = r1 / (r1 + r2) if (r1 + r2) > 0 else 0.5
                            w2 = r2 / (r1 + r2) if (r1 + r2) > 0 else 0.5

                            circles[i][0] -= dx_norm * move_amount * w1
                            circles[i][1] -= dy_norm * move_amount * w1
                            circles[j][0] += dx_norm * move_amount * w2
                            circles[j][1] += dy_norm * move_amount * w2

                            any_changes = True

        if not any_changes:
            break

    # Final boundary corrections with more precise clamping
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
    """Initialize population with better starting solutions using enhanced Voronoi-inspired placement"""
    def create_enhanced_voronoi_initialization():
        # Create a more sophisticated initial distribution using multiple strategies
        circles = []

        # Strategy 1: Start with clustered points using k-means
        # Generate candidate points
        candidate_points = []
        n_candidates = max(200, n_circles * 10)

        # Generate points with strategic distribution
        for _ in range(n_candidates):
            # Use a combination of uniform and clustered sampling
            if random.random() < 0.7:
                # More uniform distribution
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
            else:
                # Clustered around center
                angle = random.uniform(0, 2*np.pi)
                radius = random.uniform(0, 0.3)
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                # Ensure within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
            candidate_points.append([x, y])

        candidate_points = np.array(candidate_points)

        # Use KMeans to cluster into approximately n_circles clusters
        kmeans = KMeans(n_clusters=n_circles, random_state=42, n_init=20)
        kmeans.fit(candidate_points)
        centroids = kmeans.cluster_centers_

        # Create circles at centroids with refined radius calculation
        for i, (cx, cy) in enumerate(centroids):
            # Compute max radius at this position (distance to nearest boundary)
            max_r = min(cx, cy, 1-cx, 1-cy)

            # If too close to boundary, skip or reduce radius significantly
            if max_r < 0.02:
                continue

            # Calculate average distance to other centroids to estimate density
            distances = np.sqrt(np.sum((centroids - [cx, cy])**2, axis=1))
            distances = distances[distances > 0]  # Exclude self-distance

            if len(distances) > 0:
                avg_distance = np.mean(distances)
                # Use a fraction of the average distance as radius constraint
                r = min(max_r, avg_distance * 0.25)
            else:
                # If no neighbors, just use max possible radius
                r = max_r * 0.3

            # Ensure reasonable minimum and maximum radius
            r = max(0.01, min(0.25, r))
            circles.append((cx, cy, r))

        # If we don't have enough circles due to boundary constraints,
        # fill in with additional random positions
        while len(circles) < n_circles:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)

            # Check distance to all existing circles
            valid_pos = True
            for cx, cy, cr in circles:
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < cr + 0.01:  # Minimum spacing requirement
                    valid_pos = False
                    break

            if valid_pos:
                max_r = min(x, y, 1-x, 1-y)
                if max_r > 0.01:
                    r = min(max_r * 0.3, 0.2)  # Cap at reasonable value
                    circles.append((x, y, r))

        # Trim to exact number needed
        circles = circles[:n_circles]

        return circles

    population = []
    for _ in range(pop_size):
        circles = create_enhanced_voronoi_initialization()
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

    # Crossover entire circles instead of individual elements
    for i in range(point1*3, point2*3, 3):  # Adjust indices to account for 3 components per circle
        # Swap x, y, r for each circle
        ind1[i], ind2[i] = ind2[i], ind1[i]  # x
        ind1[i+1], ind2[i+1] = ind2[i+1], ind1[i+1]  # y
        ind1[i+2], ind2[i+2] = ind2[i+2], ind1[i+2]  # r

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

def constraint_aware_local_search(circles):
    """Apply constraint-aware local optimization to further improve solution"""
    # Convert to numpy array for efficient computation
    circles = np.array(circles)

    # More sophisticated optimization using projected gradient descent
    max_iterations = 100
    learning_rate = 0.05

    for iteration in range(max_iterations):
        improved = False
        # Calculate gradients for each circle
        gradients = np.zeros_like(circles[:, :2])  # Gradient for positions only

        # Calculate forces/gradient components
        for i in range(len(circles)):
            x, y, r = circles[i]

            # Accumulate forces from all other circles
            for j in range(len(circles)):
                if i != j:
                    x2, y2, r2 = circles[j]
                    dx = x - x2
                    dy = y - y2
                    dist = np.sqrt(dx*dx + dy*dy)

                    # Compute force magnitude and direction
                    if dist < r + r2:  # Collision
                        # Strong repulsion when overlapping
                        if dist > 0.001:
                            force_magnitude = (r + r2 - dist) * 2.0
                            gradients[i, 0] += dx / dist * force_magnitude
                            gradients[i, 1] += dy / dist * force_magnitude
                    elif dist < (r + r2 + 0.03):  # Near contact
                        # Gentle repulsion for nearby circles
                        if dist > 0.001:
                            force_magnitude = (r + r2 + 0.03 - dist) * 0.5
                            gradients[i, 0] -= dx / dist * force_magnitude
                            gradients[i, 1] -= dy / dist * force_magnitude

        # Apply gradient updates with proper boundary handling
        for i in range(len(circles)):
            x, y, r = circles[i]

            # Apply gradient descent step (but project onto feasible region)
            new_x = x + gradients[i, 0] * learning_rate
            new_y = y + gradients[i, 1] * learning_rate

            # Clamp to valid positions respecting circle radius
            new_x = np.clip(new_x, r, 1 - r)
            new_y = np.clip(new_y, r, 1 - r)

            # Only accept change if it improves validity
            if new_x != x or new_y != y:
                # Test if the new position is valid
                temp_circles = circles.copy()
                temp_circles[i, 0] = new_x
                temp_circles[i, 1] = new_y

                if check_validity_jit(temp_circles):
                    circles[i, 0] = new_x
                    circles[i, 1] = new_y
                    improved = True
                else:
                    # Try smaller step or skip this update
                    pass

        # Early stopping if no significant improvement
        if not improved and iteration > 20:
            break

        # Reduce learning rate over time
        learning_rate *= 0.95

    # Final validity repair to ensure constraints are met
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
        circles = constraint_aware_local_search(circles)

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