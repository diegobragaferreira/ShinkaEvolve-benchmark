# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid
import time
import warnings
from numba import jit
import random

# Constants
UNIT_HEX_RADIUS = 1.0  # Side length of unit hexagon
UNIT_HEX_APOGEE = np.sqrt(3)/2  # Distance from center to corner of unit hexagon

@jit(nopython=True)
def distance_point_to_line(point, line_start, line_end):
    """Fast computation of point-to-line distance"""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    # Vector from line_start to line_end
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx*dx + dy*dy

    if length_sq == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)

    # Project point onto line
    t = max(0, min(1, ((px - x1)*dx + (py - y1)*dy) / length_sq))
    proj_x = x1 + t*dx
    proj_y = y1 + t*dy

    return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

@jit(nopython=True)
def hexagon_vertices_numba(center_x, center_y, rotation_deg, side_length):
    """Fast computation of hexagon vertices using numba"""
    vertices = np.empty((6, 2), dtype=np.float64)
    angle_offset = rotation_deg * np.pi / 180.0
    for i in range(6):
        angle = angle_offset + i * np.pi / 3.0
        vertices[i, 0] = center_x + side_length * np.cos(angle)
        vertices[i, 1] = center_y + side_length * np.sin(angle)
    return vertices

@jit(nopython=True)
def point_in_polygon_fast(point_x, point_y, polygon_vertices):
    """Fast point-in-polygon test using ray casting algorithm"""
    n = len(polygon_vertices)
    inside = False

    p1x, p1y = polygon_vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_vertices[i % n]
        if point_y > min(p1y, p2y):
            if point_y <= max(p1y, p2y):
                if point_x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (point_y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or point_x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

@jit(nopython=True)
def project_polygon_onto_axis_numba(vertices, axis):
    """Project polygon vertices onto axis and return min/max projections"""
    projections = np.empty(len(vertices), dtype=np.float64)
    for i in range(len(vertices)):
        projections[i] = vertices[i, 0] * axis[0] + vertices[i, 1] * axis[1]
    return np.min(projections), np.max(projections)

@jit(nopython=True)
def get_hexagon_axes_numba(vertices):
    """Get axes (normals) of a hexagon for SAT collision detection"""
    axes = np.empty((6, 2), dtype=np.float64)
    for i in range(6):
        p1 = vertices[i]
        p2 = vertices[(i+1) % 6]
        edge = p2 - p1
        # Normal vector (perpendicular to edge)
        normal = np.array([-edge[1], edge[0]])
        # Normalize
        norm = np.sqrt(normal[0]*normal[0] + normal[1]*normal[1])
        if norm > 1e-10:
            normal = normal / norm
        axes[i] = normal
    return axes

@jit(nopython=True)
def sat_collision_check_numba(hex1_vertices, hex2_vertices):
    """Use Separating Axis Theorem to detect collision between hexagons"""
    # Get axes for both polygons
    axes1 = get_hexagon_axes_numba(hex1_vertices)
    axes2 = get_hexagon_axes_numba(hex2_vertices)
    all_axes = np.vstack([axes1, axes2])

    # Check each axis
    for axis in all_axes:
        min1, max1 = project_polygon_onto_axis_numba(hex1_vertices, axis)
        min2, max2 = project_polygon_onto_axis_numba(hex2_vertices, axis)

        # Check for separation
        if max1 < min2 or max2 < min1:
            return False  # No overlap on this axis, so they don't collide

    return True  # Overlap on all axes, so they collide

class HexagonPacker:
    def __init__(self):
        self.n_inner = 11
        self.unit_hex_radius = UNIT_HEX_RADIUS
        self.unit_hex_apogee = UNIT_HEX_APOGEE
        self.max_evaluations = 10000
        self.generation_limit = 500
        self.population_size = 100
        self.tournament_size = 5
        self.crossover_rate = 0.8
        self.mutation_rate = 0.1

    def create_unit_hexagon(self, center=(0,0), rotation=0):
        """Create a unit regular hexagon as a Shapely Polygon"""
        angle_offset = np.deg2rad(rotation)
        points = []
        for i in range(6):
            angle = angle_offset + i * np.pi/3
            x = center[0] + self.unit_hex_radius * np.cos(angle)
            y = center[1] + self.unit_hex_radius * np.sin(angle)
            points.append((x, y))
        return Polygon(points)

    def get_hexagon_vertices_fast(self, center, rotation):
        """Fast hexagon vertex computation using numba"""
        return hexagon_vertices_numba(center[0], center[1], rotation, self.unit_hex_radius)

    def sat_collision_check(self, hex1, hex2):
        """Fast SAT-based collision detection using numba-compiled functions"""
        # Convert Shapely polygons to vertices arrays for fast SAT
        hex1_vertices = np.array(list(hex1.exterior.coords)[:-1])  # Remove duplicate last point
        hex2_vertices = np.array(list(hex2.exterior.coords)[:-1])

        # Use compiled SAT check
        return sat_collision_check_numba(hex1_vertices, hex2_vertices)

    def check_containment(self, inner_hexagon, outer_hexagon):
        """Fast containment check using custom point-in-polygon test"""
        # Create vertices of outer hexagon
        outer_vertices = np.array(list(outer_hexagon.exterior.coords)[:-1])

        # Get vertices of inner hexagon
        inner_vertices = np.array(list(inner_hexagon.exterior.coords)[:-1])

        # Check if all vertices of inner hexagon are within outer hexagon
        for vertex in inner_vertices:
            if not point_in_polygon_fast(vertex[0], vertex[1], outer_vertices):
                return False
        return True

    def check_overlap(self, hex1, hex2):
        """Fast overlap check using SAT with early termination"""
        # Use SAT for robust collision detection
        try:
            return self.sat_collision_check(hex1, hex2)
        except:
            # Fallback to Shapely intersection if SAT fails
            return hex1.intersects(hex2)

    def evaluate_constraints(self, inner_params, outer_radius):
        """Comprehensive constraint evaluation with early termination and enhanced validation"""
        # Pre-compute all inner hexagon vertices for fast access
        inner_vertices = []
        inner_centers = []
        inner_rotations = []

        for i in range(self.n_inner):
            x, y, angle = inner_params[3*i:3*i+3]
            inner_centers.append((x, y))
            inner_rotations.append(angle)
            vertices = self.get_hexagon_vertices_fast((x, y), angle)
            inner_vertices.append(vertices)

        # Create outer hexagon vertices (unit size then scaled)
        outer_vertices = self.get_hexagon_vertices_fast((0, 0), 0)
        scaled_outer_vertices = outer_vertices * outer_radius

        # Check containment (early termination)
        for i in range(self.n_inner):
            # Check containment using point-in-polygon test
            vertices = inner_vertices[i]
            for vertex in vertices:
                if not point_in_polygon_fast(vertex[0], vertex[1], scaled_outer_vertices):
                    return False, False, 0.0  # containment violated

        # Check overlaps (early termination with SAT-based collision detection)
        for i in range(self.n_inner):
            for j in range(i+1, self.n_inner):
                # Fast SAT collision check using pre-computed vertices
                if sat_collision_check_numba(inner_vertices[i], inner_vertices[j]):
                    return False, False, 0.0  # overlap violated

        return True, True, 1.0 / outer_radius  # valid solution

    def objective_function(self, params):
        """Objective function to minimize: negative of 1/outer_radius (i.e., maximize 1/outer_radius)"""
        # params: [x1,y1,a1, x2,y2,a2, ..., x11,y11,a11, outer_radius]
        n = self.n_inner
        outer_radius = params[-1]

        # Extract inner hexagon parameters
        inner_params = params[:-1]

        # Check constraints
        containment_ok, overlap_ok, inv_radius = self.evaluate_constraints(inner_params, outer_radius)

        # If any constraint violated, return large penalty
        if not (containment_ok and overlap_ok):
            # Much larger penalty for constraint violations to discourage invalid solutions
            return 100000.0 + abs(outer_radius) * 100.0

        # Return negative of inverse radius to minimize (maximize 1/outer_radius)
        return -inv_radius

    def create_individual(self):
        """Create a random valid individual"""
        individual = []
        # Generate initial positions and rotations
        base_positions = [
            (0.0, 0.0),       # center
            (-1.8, 0.0),      # left
            (1.8, 0.0),       # right
            (0.0, 1.8),       # top
            (0.0, -1.8),      # bottom
            (-1.3, 1.3),      # top-left
            (1.3, 1.3),       # top-right
            (-1.3, -1.3),     # bottom-left
            (1.3, -1.3),      # bottom-right
            (-2.2, 0.0),      # further left
            (2.2, 0.0),       # further right
        ]

        for i, (cx, cy) in enumerate(base_positions):
            # Add small random variation to avoid symmetry issues
            jitter_x = np.random.normal(0, 0.15)
            jitter_y = np.random.normal(0, 0.15)
            individual.extend([cx + jitter_x, cy + jitter_y, np.random.uniform(0, 360)])

        # Add outer radius estimate
        # Calculate the maximum distance from center to any hexagon center + apogee
        max_dist = 0
        for cx, cy in base_positions:
            dist = np.sqrt(cx**2 + cy**2) + self.unit_hex_apogee
            max_dist = max(max_dist, dist)

        individual.append(max_dist + 0.3)  # Add a small margin
        return np.array(individual)

    def create_population(self):
        """Create a population of individuals"""
        return [self.create_individual() for _ in range(self.population_size)]

    def tournament_selection(self, population, fitnesses):
        """Select parent using tournament selection"""
        tournament_indices = np.random.choice(len(population), self.tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmin(tournament_fitnesses)]
        return population[winner_index].copy()

    def crossover(self, parent1, parent2):
        """Perform uniform crossover between parents"""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        child1 = parent1.copy()
        child2 = parent2.copy()
        # Crossover for positions and rotations
        for i in range(len(parent1)-1):  # exclude outer_radius
            if np.random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]

        return child1, child2

    def mutate(self, individual):
        """Mutate an individual"""
        mutated = individual.copy()
        for i in range(len(mutated)-1):  # exclude outer_radius
            if np.random.random() < self.mutation_rate:
                if i % 3 == 0:  # x position
                    mutated[i] += np.random.normal(0, 0.2)
                elif i % 3 == 1:  # y position
                    mutated[i] += np.random.normal(0, 0.2)
                else:  # rotation
                    mutated[i] += np.random.normal(0, 15)
                    mutated[i] = mutated[i] % 360
        # Mutate outer radius
        if np.random.random() < self.mutation_rate:
            mutated[-1] += np.random.normal(0, 0.1)
            mutated[-1] = max(3.0, mutated[-1])
        return mutated

    def genetic_algorithm(self):
        """Main genetic algorithm implementation"""
        population = self.create_population()
        best_individual = None
        best_fitness = float('inf')

        for generation in range(self.generation_limit):
            # Evaluate fitness for entire population
            fitnesses = []
            for individual in population:
                fitness = self.objective_function(individual)
                fitnesses.append(fitness)

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()

            # Check early termination
            if generation % 50 == 0:
                if abs(best_fitness) < 0.2545:  # Close to target
                    break

            # Create new population
            new_population = []

            # Elitism: keep best individual
            new_population.append(best_individual.copy())

            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)

                child1, child2 = self.crossover(parent1, parent2)

                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.extend([child1, child2])

            # Trim to exact population size
            population = new_population[:self.population_size]

        return best_individual

    def simulated_annealing_refinement(self, initial_params, max_iterations=500):
        """Refine solution using simulated annealing with adaptive cooling"""
        current_params = initial_params.copy()
        current_fitness = self.objective_function(current_params)
        best_params = current_params.copy()
        best_fitness = current_fitness

        # Adaptive cooling schedule
        initial_temp = 10.0
        final_temp = 0.01
        alpha = 0.95
        temp = initial_temp

        iteration = 0
        while iteration < max_iterations and temp > final_temp:
            # Generate neighbor
            neighbor_params = current_params.copy()

            # Perturb one random parameter
            idx = np.random.randint(len(neighbor_params))
            if idx < len(neighbor_params) - 1:  # position or rotation
                if idx % 3 == 0:  # x position
                    neighbor_params[idx] += np.random.normal(0, 0.05)
                elif idx % 3 == 1:  # y position
                    neighbor_params[idx] += np.random.normal(0, 0.05)
                else:  # rotation
                    neighbor_params[idx] += np.random.normal(0, 5)
                    neighbor_params[idx] = neighbor_params[idx] % 360
            else:  # outer radius
                neighbor_params[idx] += np.random.normal(0, 0.02)
                neighbor_params[idx] = max(3.0, neighbor_params[idx])

            # Evaluate neighbor
            neighbor_fitness = self.objective_function(neighbor_params)

            # Accept or reject based on Metropolis criterion
            if neighbor_fitness < current_fitness or \
               np.random.random() < np.exp(-(neighbor_fitness - current_fitness) / temp):
                current_params = neighbor_params
                current_fitness = neighbor_fitness

                if current_fitness < best_fitness:
                    best_params = current_params.copy()
                    best_fitness = current_fitness

            # Cool down
            temp *= alpha
            iteration += 1

        return best_params

    def optimize_solution(self):
        """Main optimization routine using hybrid GA + SA approach"""
        # Step 1: Global search with genetic algorithm
        ga_start = time.time()
        ga_best = self.genetic_algorithm()
        ga_time = time.time() - ga_start

        # Step 2: Local refinement with simulated annealing
        sa_start = time.time()
        final_params = self.simulated_annealing_refinement(ga_best)
        sa_time = time.time() - sa_start

        return final_params

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses hybrid evolutionary optimization (GA + SA) to find the best arrangement.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    try:
        packer = HexagonPacker()

        # Run hybrid optimization
        final_params = packer.optimize_solution()

        # Extract results
        n = 11
        inner_params = final_params[:-1]
        outer_radius = final_params[-1]

        # Validate solution
        containment_ok, overlap_ok, inv_radius = packer.evaluate_constraints(inner_params, outer_radius)

        if containment_ok and overlap_ok:
            # Format output
            inner_hex_data = np.zeros((n, 3))
            for i in range(n):
                inner_hex_data[i] = inner_params[3*i:3*i+3]

            outer_hex_data = np.array([0, 0, 0])

            return inner_hex_data, outer_hex_data, outer_radius

    except Exception as e:
        warnings.warn(f"Error in optimization: {str(e)}")
        pass

    # Fallback to original method if optimization fails
    inner_hex_data = np.array([
        [0, 0, 0],        # center
        [-2.5, 0, 0],     # left
        [2.5, 0, 0],      # right
        [-1.25, 2.17, 0], # top-left
        [1.25, 2.17, 0],  # top-right
        [-1.25, -2.17, 0], # bottom-left
        [1.25, -2.17, 0], # bottom-right
        [-3.75, 2.17, 0], # far top-left
        [3.75, 2.17, 0],  # far top-right
        [-3.75, -2.17, 0], # far bottom-left
        [3.75, -2.17, 0], # far bottom-right
    ])

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 8  # large enough to contain all inner hexagons

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END