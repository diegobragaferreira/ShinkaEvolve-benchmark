# EVOLVE-BLOCK-START
import numpy as np
import random
from deap import base, creator, tools, algorithms
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import time
from shapely.geometry import Polygon, Point
import math
from joblib import Parallel, delayed
import multiprocessing

def create_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Create vertices of a regular hexagon given center, rotation, and side length"""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return vertices

def check_hexagon_containment(hex_vertices, outer_hex_vertices):
    """Check if all vertices of inner hexagon are within outer hexagon with buffer to avoid floating point issues"""
    outer_polygon = Polygon(outer_hex_vertices)
    # Use a small buffer to prevent floating point precision issues
    buffered_outer = outer_polygon.buffer(1e-6)
    for vertex in hex_vertices:
        if not buffered_outer.contains(Point(vertex[0], vertex[1])):
            return False
    return True

def sat_separating_axis_test(vertices1, vertices2):
    """Separating Axis Theorem implementation for hexagon overlap detection"""
    # Get all edge normals for both polygons
    def get_edges(vertices):
        edges = []
        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            edges.append(edge)
        return edges

    def get_normals(edges):
        normals = []
        for edge in edges:
            # Normal vector (perpendicular to edge)
            normal = (-edge[1], edge[0])
            # Normalize
            length = math.sqrt(normal[0]**2 + normal[1]**2)
            if length > 1e-10:
                normal = (normal[0]/length, normal[1]/length)
            normals.append(normal)
        return normals

    edges1 = get_edges(vertices1)
    edges2 = get_edges(vertices2)
    normals1 = get_normals(edges1)
    normals2 = get_normals(edges2)

    # Test all normals as potential separating axes
    all_normals = normals1 + normals2
    for axis in all_normals:
        # Project both polygons onto the axis
        proj1 = [vertex[0]*axis[0] + vertex[1]*axis[1] for vertex in vertices1]
        proj2 = [vertex[0]*axis[0] + vertex[1]*axis[1] for vertex in vertices2]

        min1, max1 = min(proj1), max(proj1)
        min2, max2 = min(proj2), max(proj2)

        # Check if projections overlap
        if max1 < min2 or max2 < min1:
            return False  # Found separating axis

    return True  # No separating axis found, polygons overlap

def check_hexagon_overlap_fast(hex1_vertices, hex2_vertices):
    """Fast bounding box overlap check"""
    # Get bounding boxes
    x1 = [v[0] for v in hex1_vertices]
    y1 = [v[1] for v in hex1_vertices]
    x2 = [v[0] for v in hex2_vertices]
    y2 = [v[1] for v in hex2_vertices]

    min_x1, max_x1 = min(x1), max(x1)
    min_y1, max_y1 = min(y1), max(y1)
    min_x2, max_x2 = min(x2), max(x2)
    min_y2, max_y2 = min(y2), max(y2)

    # Check if bounding boxes overlap
    if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
        return False
    return True

def check_hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using SAT with Shapely buffer as backup for precision"""
    # Fast preliminary check
    if not check_hexagon_overlap_fast(hex1_vertices, hex2_vertices):
        return False

    # Primary SAT test
    if sat_separating_axis_test(hex1_vertices, hex2_vertices):
        # If SAT says they overlap, double-check with Shapely for robustness
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        # Use a small buffer to prevent floating point precision issues
        buffered_poly1 = poly1.buffer(1e-6)
        buffered_poly2 = poly2.buffer(1e-6)
        return buffered_poly1.intersects(buffered_poly2)

    return False

def calculate_outer_hex_side_length(inner_hex_data):
    """Calculate minimum outer hexagon side length that contains all inner hexagons using tight bounding"""
    if len(inner_hex_data) == 0:
        return 1000

    # Get all vertices of inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = create_hexagon_vertices(center_x, center_y, angle)
        all_vertices.extend(vertices)

    if not all_vertices:
        return 1000

    # Calculate tight bounding box
    xs = [v[0] for v in all_vertices]
    ys = [v[1] for v in all_vertices]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Calculate the diagonal of the bounding box
    bbox_width = max_x - min_x
    bbox_height = max_y - min_y

    # For a hexagon, we need to ensure all points fit within the hexagon's circumscribed circle
    # The diagonal of the bounding box gives us information about the required side length
    diagonal = math.sqrt(bbox_width**2 + bbox_height**2)

    # For a regular hexagon, the relationship between circumradius and side length:
    # Circumradius = side_length
    # We want to fit our bounding box diagonal into the hexagon
    # So the outer hexagon's side length should be at least diagonal/sqrt(3)
    side_length = diagonal / math.sqrt(3)

    # Add a small margin to ensure full containment
    side_length *= 1.1

    return side_length

def generate_initial_polar_patterns():
    """Generate multiple hexagonal lattice patterns using polar coordinates"""
    patterns = []

    # Pattern 1: Simple concentric rings
    pattern1 = np.array([
        [0.0, 0.0, 0.0],      # center (r, theta, angle)
        [1.73, 0.0, 0.0],     # ring 1, angle 0
        [1.73, 60.0, 0.0],    # ring 1, angle 60
        [1.73, 120.0, 0.0],   # ring 1, angle 120
        [1.73, 180.0, 0.0],   # ring 1, angle 180
        [1.73, 240.0, 0.0],   # ring 1, angle 240
        [1.73, 300.0, 0.0],   # ring 1, angle 300
        [3.46, 0.0, 0.0],     # ring 2, angle 0
        [3.46, 60.0, 0.0],    # ring 2, angle 60
        [3.46, 120.0, 0.0],   # ring 2, angle 120
        [3.46, 180.0, 0.0],   # ring 2, angle 180
    ])
    patterns.append(pattern1[:11])  # Trim to 11 elements

    # Pattern 2: Hexagonal grid with optimized spacing
    pattern2 = np.array([
        [0.0, 0.0, 0.0],
        [1.73, 0.0, 0.0],
        [1.73, 120.0, 0.0],
        [1.73, 240.0, 0.0],
        [3.46, 0.0, 0.0],
        [3.46, 120.0, 0.0],
        [3.46, 240.0, 0.0],
        [1.73, 60.0, 0.0],
        [1.73, 180.0, 0.0],
        [1.73, 300.0, 0.0],
        [3.46, 60.0, 0.0],
    ])
    patterns.append(pattern2[:11])

    # Pattern 3: Spiral-like arrangement
    pattern3 = np.array([
        [0.0, 0.0, 0.0],
        [1.73, 0.0, 0.0],
        [1.73, 60.0, 0.0],
        [1.73, 120.0, 0.0],
        [1.73, 180.0, 0.0],
        [1.73, 240.0, 0.0],
        [1.73, 300.0, 0.0],
        [3.46, 30.0, 0.0],
        [3.46, 90.0, 0.0],
        [3.46, 150.0, 0.0],
        [3.46, 210.0, 0.0],
    ])
    patterns.append(pattern3[:11])

    # Pattern 4: Star pattern with radial symmetry
    pattern4 = np.array([
        [0.0, 0.0, 0.0],
        [1.73, 0.0, 0.0],
        [1.73, 120.0, 0.0],
        [1.73, 240.0, 0.0],
        [3.46, 0.0, 0.0],
        [3.46, 60.0, 0.0],
        [3.46, 120.0, 0.0],
        [3.46, 180.0, 0.0],
        [3.46, 240.0, 0.0],
        [3.46, 300.0, 0.0],
        [5.19, 0.0, 0.0],
    ])
    patterns.append(pattern4[:11])

    return patterns

def evaluate_solution_polar(individual):
    """Evaluate fitness of a polar solution - convert to Cartesian and check constraints"""
    # Convert from polar to Cartesian coordinates
    cartesian_coords = []
    for r, theta, angle in individual:
        x = r * math.cos(math.radians(theta))
        y = r * math.sin(math.radians(theta))
        cartesian_coords.append([x, y, angle])

    hex_data = np.array(cartesian_coords)

    # Create outer hexagon vertices (assuming centered at origin)
    outer_side_length = calculate_outer_hex_side_length(hex_data)

    # Check constraints
    try:
        # Check containment for all inner hexagons
        outer_hex_vertices = create_hexagon_vertices(0, 0, 0, outer_side_length)

        # Check if all hexagons are contained and non-overlapping
        total_penalty = 0

        # Check containment
        for i in range(len(hex_data)):
            center_x, center_y, angle = hex_data[i]
            inner_hex_vertices = create_hexagon_vertices(center_x, center_y, angle)
            if not check_hexagon_containment(inner_hex_vertices, outer_hex_vertices):
                total_penalty += 10000  # Large penalty for containment violation

        # Check overlaps with early termination
        for i in range(len(hex_data)):
            for j in range(i+1, len(hex_data)):
                center_x1, center_y1, angle1 = hex_data[i]
                center_x2, center_y2, angle2 = hex_data[j]

                hex1_vertices = create_hexagon_vertices(center_x1, center_y1, angle1)
                hex2_vertices = create_hexagon_vertices(center_x2, center_y2, angle2)

                if check_hexagon_overlap(hex1_vertices, hex2_vertices):
                    total_penalty += 10000  # Large penalty for overlap

        # Return fitness (inverse of outer hex side length + penalties)
        if total_penalty > 0:
            return (1.0 / outer_side_length - total_penalty,),  # Return tuple for DEAP

        return (1.0 / outer_side_length,),  # Maximize 1/outer_side_length

    except Exception as e:
        return (-10000,),  # Very poor fitness for invalid solutions

def create_polar_individual():
    """Create a randomized polar coordinate individual"""
    individual = []
    # Center hexagon
    individual.append([0.0, 0.0, random.uniform(0, 360)])

    # Ring 1 hexagons
    for i in range(6):
        r = 1.73  # distance from center
        theta = i * 60  # angles evenly spaced
        angle = random.uniform(0, 360)
        individual.append([r, theta, angle])

    # Ring 2 hexagons (only 5 to make 11 total)
    for i in range(5):
        r = 3.46  # distance from center
        theta = i * 72  # angles evenly spaced
        angle = random.uniform(0, 360)
        individual.append([r, theta, angle])

    return individual

def polar_to_cartesian(polar_data):
    """Convert polar coordinates to Cartesian"""
    cartesian = []
    for r, theta, angle in polar_data:
        x = r * math.cos(math.radians(theta))
        y = r * math.sin(math.radians(theta))
        cartesian.append([x, y, angle])
    return np.array(cartesian)

def cartesian_to_polar(cartesian_data):
    """Convert Cartesian coordinates to polar"""
    polar = []
    for x, y, angle in cartesian_data:
        r = math.sqrt(x**2 + y**2)
        theta = math.degrees(math.atan2(y, x)) % 360
        polar.append([r, theta, angle])
    return polar

def adaptive_mutation_polar(individual, sigma=1.0):
    """Apply mutation to polar coordinates with special handling for radial distances"""
    mutated = []
    for i, (r, theta, angle) in enumerate(individual):
        # Mutate radial component (with bounded constraints)
        if i > 0:  # Only mutate non-center hexagons' radial distance
            new_r = r + random.gauss(0, sigma)
            # Ensure radial distance stays reasonable
            new_r = max(0.1, min(10.0, new_r))
        else:
            new_r = r  # Don't mutate center distance

        # Mutate angular components
        new_theta = (theta + random.gauss(0, sigma * 2)) % 360
        new_angle = (angle + random.gauss(0, sigma * 3)) % 360

        mutated.append([new_r, new_theta, new_angle])

    return mutated

def run_single_polar_optimization(toolbox, num_hexagons, max_generations=100):
    """Run a single optimization attempt using polar coordinates"""
    try:
        # Generate initial population with polar patterns
        pop = []
        polar_patterns = generate_initial_polar_patterns()

        # Add pattern-based individuals
        for pattern in polar_patterns:
            # Convert to polar with noise
            individual = [[r + random.uniform(-0.2, 0.2),
                          theta + random.uniform(-10, 10),
                          angle + random.uniform(-30, 30)]
                         for r, theta, angle in pattern]
            # Ensure we're within valid ranges
            individual = [[max(0.1, min(10.0, r)),
                          theta % 360,
                          angle % 360] for r, theta, angle in individual]
            pop.append(individual)

        # Fill remaining slots with random individuals
        while len(pop) < 50:
            pop.append(create_polar_individual())

        # Convert to DEAP individuals
        deap_pop = [creator.Individual(ind) for ind in pop]

        hof = tools.HallOfFame(1)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Run evolutionary algorithm
        pop, logbook = algorithms.eaSimple(deap_pop, toolbox, cxpb=0.8, mutpb=0.3,
                                          ngen=max_generations, stats=stats, halloffame=hof, verbose=False)

        # Get best individual
        if len(hof) > 0:
            best_individual = hof[0]

            # Convert polar to Cartesian for final evaluation
            cartesian_best = polar_to_cartesian(best_individual)

            # Final local refinement
            def objective(x_flat):
                # Reshape to polar
                polar_data = []
                for i in range(0, len(x_flat), 3):
                    r = x_flat[i]
                    theta = x_flat[i+1]
                    angle = x_flat[i+2]
                    polar_data.append([r, theta, angle])

                # Convert to cartesian
                cartesian = polar_to_cartesian(polar_data)

                # Calculate outer hexagon side length
                outer_side_length = calculate_outer_hex_side_length(cartesian)

                # Check constraints
                outer_hex_vertices = create_hexagon_vertices(0, 0, 0, outer_side_length)
                total_penalty = 0

                # Check containment
                for i in range(len(cartesian)):
                    center_x, center_y, angle = cartesian[i]
                    inner_hex_vertices = create_hexagon_vertices(center_x, center_y, angle)
                    if not check_hexagon_containment(inner_hex_vertices, outer_hex_vertices):
                        total_penalty += 10000

                # Check overlaps
                for i in range(len(cartesian)):
                    for j in range(i+1, len(cartesian)):
                        center_x1, center_y1, angle1 = cartesian[i]
                        center_x2, center_y2, angle2 = cartesian[j]

                        hex1_vertices = create_hexagon_vertices(center_x1, center_y1, angle1)
                        hex2_vertices = create_hexagon_vertices(center_x2, center_y2, angle2)

                        if check_hexagon_overlap(hex1_vertices, hex2_vertices):
                            total_penalty += 10000

                # Return fitness
                if total_penalty > 0:
                    return 1.0 / outer_side_length - total_penalty
                return 1.0 / outer_side_length

            # Flatten the polar solution for local refinement
            x0 = []
            for r, theta, angle in best_individual:
                x0.extend([r, theta, angle])

            # Define bounds
            bounds = []
            for i in range(len(x0)):
                if i % 3 == 0:  # r (radial distance)
                    bounds.append((0.1, 10.0))
                elif i % 3 == 1:  # theta (angular position)
                    bounds.append((0.0, 360.0))
                else:  # angle (rotation)
                    bounds.append((0.0, 360.0))

            try:
                result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 30})
                refined_solution = result.x.reshape(-1, 3)
                cartesian_refined = polar_to_cartesian(refined_solution)
                refined_outer_side_length = calculate_outer_hex_side_length(cartesian_refined)
                refined_fitness = 1.0 / refined_outer_side_length
                return (cartesian_refined, np.array([0, 0, 0]), refined_outer_side_length, refined_fitness)
            except:
                # Fallback to original
                outer_side_length = calculate_outer_hex_side_length(cartesian_best)
                current_fitness = 1.0 / outer_side_length
                return (cartesian_best, np.array([0, 0, 0]), outer_side_length, current_fitness)

    except Exception as e:
        return None

    return None

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses polar coordinate evolutionary optimization with hierarchical approach.
    """
    # Set up evolutionary algorithm with polar coordinates
    random.seed(42)
    np.random.seed(42)

    # Create fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Register mutation and crossover operators for polar coordinates
    def mutate_polar(individual):
        return adaptive_mutation_polar(individual, sigma=0.5)

    def mate_polar(ind1, ind2):
        # Simple uniform crossover for polar coordinates
        child1 = []
        child2 = []
        for i in range(len(ind1)):
            if random.random() < 0.5:
                child1.append(ind1[i][:])  # Copy the element
                child2.append(ind2[i][:])  # Copy the element
            else:
                child1.append(ind2[i][:])  # Copy the element
                child2.append(ind1[i][:])  # Copy the element
        return child1, child2

    # Register operators
    toolbox.register("individual", create_polar_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_solution_polar)
    toolbox.register("mate", mate_polar)
    toolbox.register("mutate", mutate_polar)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Run multiple optimization attempts in parallel
    results = Parallel(n_jobs=min(4, multiprocessing.cpu_count()))(
        delayed(run_single_polar_optimization)(toolbox, 11, 100)
        for _ in range(4)
    )

    # Filter out None results
    valid_results = [r for r in results if r is not None]

    if valid_results:
        # Find the best result among all attempts
        best_result = max(valid_results, key=lambda x: x[3])  # Compare by fitness
        return best_result[0], best_result[1], best_result[2]
    else:
        # Fallback to original configuration if optimization fails
        print("Using fallback configuration")
        inner_hex_data = np.array([
            [0, 0, 0],  # center
            [-2.5, 0, 0],  # left
            [2.5, 0, 0],  # right
            [-1.25, 2.17, 0],  # top-left
            [1.25, 2.17, 0],  # top-right
            [-1.25, -2.17, 0],  # bottom-left
            [1.25, -2.17, 0],  # bottom-right
            [-3.75, 2.17, 0],  # far top-left
            [3.75, 2.17, 0],  # far top-right
            [-3.75, -2.17, 0],  # far bottom-left
            [3.75, -2.17, 0],  # far bottom-right
        ])
        outer_hex_side_length = 8
        return inner_hex_data, np.array([0, 0, 0]), outer_hex_side_length

# EVOLVE-BLOCK-END