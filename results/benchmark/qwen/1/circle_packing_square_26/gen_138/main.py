# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import KDTree, Voronoi
from deap import base, creator, tools, algorithms
import random
import time
from scipy.spatial.distance import cdist
from collections import defaultdict

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

def create_voronoi_initialization(n_circles, n_candidates=1000):
    """Create initial configuration using enhanced Voronoi-based approach"""
    # Generate random candidate points with better distribution
    candidates = np.random.rand(n_candidates, 2)

    # Add strategic boundary and center points for better coverage
    boundary_points = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1],  # Corners
        [0.5, 0], [0.5, 1], [0, 0.5], [1, 0.5],  # Midpoints
        [0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75],  # Diagonals
        [0.5, 0.5]  # Center
    ])
    candidates = np.vstack([candidates, boundary_points])

    # Compute Voronoi diagram
    try:
        vor = Voronoi(candidates)
    except:
        # Fallback to simpler approach if Voronoi computation fails
        points = np.random.rand(n_circles, 2)
        circles = []
        for x, y in points:
            max_radius = min(x, 1-x, y, 1-y)
            circles.append([x, y, max_radius])
        return circles

    # Get Voronoi vertices and regions
    regions = []
    for i in range(len(vor.points)):
        region = []
        for j, vertex_idx in enumerate(vor.point_region[i]):
            if vertex_idx != -1:
                region.append(vor.vertices[vertex_idx])
        regions.append(region)

    # Calculate Voronoi cell properties for selection
    cell_properties = []
    for i, (point, region) in enumerate(zip(vor.points, regions)):
        if len(region) > 0:
            # Calculate area of Voronoi cell
            area = calculate_polygon_area(np.array(region))

            # Calculate centroid of the cell
            centroid = np.mean(np.array(region), axis=0)

            # Calculate distance from centroid to center (0.5, 0.5)
            distance_to_center = np.sqrt((centroid[0] - 0.5)**2 + (centroid[1] - 0.5)**2)

            # Store properties: (area, distance_from_center, point)
            cell_properties.append((area, distance_to_center, point))

    # Sort by area (descending) and select top candidates, prioritizing centrality
    cell_properties.sort(key=lambda x: (x[0] * 0.7 - x[1] * 0.3), reverse=True)
    selected_points = [prop[2] for prop in cell_properties[:n_circles]]

    # Add more points if needed
    if len(selected_points) < n_circles:
        # Fill with random points
        additional_points = np.random.rand(n_circles - len(selected_points), 2)
        selected_points.extend(additional_points)

    selected_points = selected_points[:n_circles]

    # Enhanced placement with better radius calculation
    circles = []
    for i, (x, y) in enumerate(selected_points):
        # Calculate maximum possible radius that accounts for geometry
        max_radius = min(x, 1-x, y, 1-y)

        # If in the center region, allow larger radii
        center_distance = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
        if center_distance < 0.3:
            # Allow larger radius in center
            max_radius = min(max_radius, 0.25)
        else:
            # Reduce radius for edges to allow better packing
            max_radius = min(max_radius, 0.2)

        # Ensure reasonable minimum
        max_radius = max(0.005, max_radius)

        if max_radius > 0:
            circles.append([x, y, max_radius])
        else:
            # Fallback to small circle if boundary constrained
            circles.append([x, y, 0.01])

    return circles

def calculate_polygon_area(vertices):
    """Calculate area of polygon using shoelace formula"""
    if len(vertices) < 3:
        return 0
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(len(x)-1)) +
                     x[-1] * y[0] - x[0] * y[-1])

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n_circles = 26
    max_radius = 0.5

    # Precompute some constants
    min_distance = 0.001  # Minimum distance between circles for numerical stability
    max_generation = 100

    def is_valid_position(x, y, r, existing_circles):
        """Check if a circle at (x,y) with radius r is valid"""
        # Check boundary constraints
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False

        # Check overlap with existing circles
        for cx, cy, cr in existing_circles:
            distance = np.sqrt((x - cx)**2 + (y - cy)**2)
            if distance < r + cr + min_distance:
                return False
        return True

    def create_initial_population(pop_size):
        """Create an initial population with good starting configurations"""
        population = []

        # Generate multiple initial solutions using different strategies
        for _ in range(pop_size):
            # Strategy 1: Voronoi-based initialization
            circles = create_voronoi_initialization(n_circles)
            
            # Strategy 2: Greedy placement for refinement
            if len(circles) < n_circles:
                for i in range(n_circles - len(circles)):
                    best_x, best_y, best_r = 0.5, 0.5, 0.0
                    attempts = 0
                    while attempts < 1000:
                        x = np.random.uniform(0.01, 0.99)
                        y = np.random.uniform(0.01, 0.99)
                        r = min(x, 1-x, y, 1-y)
                        
                        if is_valid_position(x, y, r, circles):
                            best_x, best_y, best_r = x, y, r
                            break
                        attempts += 1
                    circles.append([best_x, best_y, best_r])
            
            # Strategy 3: Improve existing circles
            for _ in range(100):
                improved = False
                for i in range(len(circles)):
                    # Try to increase radius
                    x, y, r = circles[i]
                    max_r = min(x, 1-x, y, 1-y)
                    if max_r > r:
                        # Binary search for maximum valid radius
                        low, high = r, max_r
                        best_r = r
                        for _ in range(10):
                            test_r = (low + high) / 2
                            if is_valid_position(x, y, test_r, circles):
                                best_r = test_r
                                low = test_r
                            else:
                                high = test_r
                        
                        if best_r > r:
                            circles[i] = [x, y, best_r]
                            improved = True
                
                if not improved:
                    break
                    
            population.append(circles)

        return population

    def evaluate(individual):
        """Evaluate fitness - sum of radii"""
        total_radius = sum(circle[2] for circle in individual)
        return (total_radius,)

    def mutate(individual):
        """Custom mutation operator that respects constraints"""
        # Pick a random circle to modify
        idx = random.randint(0, len(individual) - 1)
        x, y, r = individual[idx]

        # Create a new, slightly modified circle
        new_x = max(0.01, min(0.99, x + random.gauss(0, 0.015)))
        new_y = max(0.01, min(0.99, y + random.gauss(0, 0.015)))

        # Recalculate max possible radius at new location
        new_r = min(new_x, 1-new_x, new_y, 1-new_y)

        # Adjust radius to ensure no overlaps
        for i, (cx, cy, cr) in enumerate(individual):
            if i != idx:
                distance = np.sqrt((new_x - cx)**2 + (new_y - cy)**2)
                max_radius_allowed = distance - cr - min_distance
                if max_radius_allowed > 0:
                    new_r = min(new_r, max_radius_allowed)

        # Ensure minimum positive radius
        new_r = max(0.001, new_r)

        individual[idx] = [new_x, new_y, new_r]
        return individual,

    def crossover(ind1, ind2):
        """Custom crossover that maintains validity with repair"""
        # Simple uniform crossover of positions and radii
        for i in range(len(ind1)):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]

        # Repair individuals to maintain constraints
        for ind in [ind1, ind2]:
            # Ensure all circles are valid after crossover
            for i in range(len(ind)):
                x, y, r = ind[i]
                # Ensure boundary constraints
                if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                    # Shrink radius and reposition to center
                    r = min(x, 1-x, y, 1-y)
                    r = max(0.001, r)
                    # Reposition to center of feasible area
                    x = max(r, min(1-r, x))
                    y = max(r, min(1-r, y))
                    ind[i] = [x, y, r]

                # Check for overlaps and adjust
                for j in range(len(ind)):
                    if i != j:
                        cx, cy, cr = ind[j]
                        distance = np.sqrt((x - cx)**2 + (y - cy)**2)
                        if distance < r + cr + min_distance:
                            # Reduce radius to prevent overlap
                            r = max(0.001, distance - cr - min_distance)
                            ind[i] = [x, y, r]

        return ind1, ind2

    def constraint_aware_local_search(individual):
        """Enhanced local search with better constraint handling"""
        # Try to improve each circle's radius and position
        improved = True
        iterations = 0
        
        while improved and iterations < 50:
            improved = False
            iterations += 1
            
            # Try to increase radii
            for i in range(len(individual)):
                x, y, r = individual[i]
                max_r = min(x, 1-x, y, 1-y)
                
                if max_r > r:
                    # Binary search for best radius
                    low, high = r, max_r
                    best_radius = r
                    for _ in range(10):
                        test_r = (low + high) / 2
                        if is_valid_position(x, y, test_r, individual):
                            best_radius = test_r
                            low = test_r
                        else:
                            high = test_r
                    
                    if best_radius > r:
                        individual[i] = [x, y, best_radius]
                        improved = True
                        
            # Try to fine-tune positions to reduce overlaps
            for i in range(len(individual)):
                x, y, r = individual[i]
                best_x, best_y = x, y
                best_radius = r
                best_score = r
                
                # Try small movements
                for dx in [-0.005, -0.002, 0, 0.002, 0.005]:
                    for dy in [-0.005, -0.002, 0, 0.002, 0.005]:
                        new_x = max(0.01, min(0.99, x + dx))
                        new_y = max(0.01, min(0.99, y + dy))
                        
                        # Check validity
                        if is_valid_position(new_x, new_y, r, individual):
                            score = r  # Maximize radius
                            if score > best_score:
                                best_score = score
                                best_x, best_y = new_x, new_y
                        
                if best_x != x or best_y != y:
                    individual[i] = [best_x, best_y, r]
                    improved = True

        return individual

    # Setup DEAP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", lambda: create_initial_population(1)[0])
    toolbox.register("population", lambda: create_initial_population(30))
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=5)

    # Create initial population
    pop = toolbox.population()

    # Run evolution
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run evolution with adaptive parameters
    start_time = time.time()
    try:
        # Adaptive evolution with different parameters for early vs late generations
        for generation in range(max_generation):
            # Adjust parameters based on generation and diversity
            current_gen = generation
            adaptive_cxpb = max(0.6, 0.8 - (current_gen / max_generation) * 0.5)
            adaptive_mutpb = max(0.1, 0.3 - (current_gen / max_generation) * 0.2)
            
            # Run evolution step with adaptive parameters
            pop, log = algorithms.eaSimple(pop, toolbox, cxpb=adaptive_cxpb, mutpb=adaptive_mutpb,
                                          ngen=1, stats=stats, halloffame=hof,
                                          verbose=False)
            
            # Apply local search to best individual periodically
            if generation % 10 == 0 and len(hof) > 0:
                hof[0] = constraint_aware_local_search(hof[0])
            
            # Early stopping check
            if time.time() - start_time > 55:  # Leave 5 seconds for final processing
                break
                
    except Exception as e:
        print(f"Error in evolution: {e}")

    # Return the best individual found
    best_individual = hof[0] if len(hof) > 0 else pop[0]

    # Apply final local search
    final_individual = constraint_aware_local_search(best_individual.copy())
    
    # Convert to final result format (ensure all circles are valid)
    final_circles = []
    for x, y, r in final_individual:
        # Final validation and adjustment
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        final_circles.append([x, y, r])

    return np.array(final_circles)

# EVOLVE-BLOCK-END