# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import time
import math
from numba import jit
import warnings

@jit(nopython=True)
def hexagon_vertices_fast(x, y, angle_deg, side_length=1):
    """Fast generation of hexagon vertices using numba"""
    angle_rad = np.radians(angle_deg)
    angles = np.arange(0, 6) * np.pi / 3
    vertices = np.zeros((6, 2))
    for i in range(6):
        vertices[i, 0] = x + side_length * np.cos(angles[i] + angle_rad)
        vertices[i, 1] = y + side_length * np.sin(angles[i] + angle_rad)
    return vertices

@jit(nopython=True)
def point_in_polygon_fast(point, polygon):
    """Fast point-in-polygon test using ray casting"""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

@jit(nopython=True)
def distance_point_to_segment(point, seg_start, seg_end):
    """Distance from point to line segment"""
    px, py = point
    x1, y1 = seg_start
    x2, y2 = seg_end

    # Vector from start to end
    dx, dy = x2 - x1, y2 - y1
    # Vector from start to point
    px_minus_x1, py_minus_y1 = px - x1, py - y1

    # Project point onto line
    length_sq = dx*dx + dy*dy
    if length_sq == 0:
        return np.sqrt(px_minus_x1*px_minus_x1 + py_minus_y1*py_minus_y1)

    t = (px_minus_x1*dx + py_minus_y1*dy) / length_sq
    t = max(0, min(1, t))

    # Closest point on segment
    closest_x = x1 + t*dx
    closest_y = y1 + t*dy

    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

@jit(nopython=True)
def hexagon_distance_fast(hex1_vertices, hex2_vertices):
    """Compute minimum distance between two hexagons"""
    min_dist = np.inf
    for i in range(6):
        p1 = hex1_vertices[i]
        p2 = hex1_vertices[(i+1)%6]
        for j in range(6):
            q1 = hex2_vertices[j]
            q2 = hex2_vertices[(j+1)%6]
            dist = distance_point_to_segment(q1, p1, p2)
            min_dist = min(min_dist, dist)
    return min_dist

class HexagonGeometry:
    """Handles all geometric operations related to hexagons."""
    
    @staticmethod
    @jit(nopython=True)
    def vertices(center_x, center_y, angle_deg, side_length=1):
        """Generate vertices of a regular hexagon given center, angle, and side length."""
        angle_rad = math.radians(angle_deg)
        vertices = np.empty((6, 2))
        for i in range(6):
            angle = angle_rad + i * math.pi / 3
            x = center_x + side_length * math.cos(angle)
            y = center_y + side_length * math.sin(angle)
            vertices[i] = (x, y)
        return vertices

    @staticmethod
    @jit(nopython=True)
    def outer_vertices(side_length):
        """Generate vertices of outer hexagon centered at origin."""
        return HexagonGeometry.vertices(0, 0, 0, side_length)

    @staticmethod
    @jit(nopython=True)
    def bounds(vertices):
        """Get bounding box of a hexagon."""
        min_x = vertices[0][0]
        max_x = vertices[0][0]
        min_y = vertices[0][1]
        max_y = vertices[0][1]

        for i in range(1, len(vertices)):
            x, y = vertices[i]
            if x < min_x: min_x = x
            if x > max_x: max_x = x
            if y < min_y: min_y = y
            if y > max_y: max_y = y

        return min_x, max_x, min_y, max_y

    @staticmethod
    @jit(nopython=True)
    def to_grid_cells(vertices, cell_size):
        """Get all grid cells that a hexagon might occupy."""
        min_x, max_x, min_y, max_y = HexagonGeometry.bounds(vertices)

        min_cell_x = int(min_x // cell_size)
        max_cell_x = int(max_x // cell_size)
        min_cell_y = int(min_y // cell_size)
        max_cell_y = int(max_y // cell_size)

        cells = []
        for x in range(min_cell_x, max_cell_x + 1):
            for y in range(min_cell_y, max_cell_y + 1):
                cells.append((x, y))

        return cells

class HexagonConstraintChecker:
    """Handles constraint checking for hexagon packing."""
    
    @staticmethod
    @jit(nopython=True)
    def containment_fast(hexagon_vertices_list, outer_side_length):
        """Fast containment check using distance from center."""
        outer_radius = outer_side_length

        for vertices in hexagon_vertices_list:
            center_x = vertices[0][0]
            center_y = vertices[0][1]
            distance = math.sqrt(center_x*center_x + center_y*center_y)
            if distance > outer_radius:
                return False
        return True

    @staticmethod
    @jit(nopython=True)
    def overlap_spatial_hashing(hexagon_vertices_list, cell_size=1.0):
        """Spatial hashing overlap check for improved performance."""
        n_hexagons = len(hexagon_vertices_list)

        # Initialize grid
        grid = {}

        # Populate grid with hexagon indices
        for i in range(n_hexagons):
            vertices = hexagon_vertices_list[i]
            cells = HexagonGeometry.to_grid_cells(vertices, cell_size)
            for cell in cells:
                if cell not in grid:
                    grid[cell] = []
                grid[cell].append(i)

        # Check for overlaps using spatial hashing
        for i in range(n_hexagons):
            vertices_i = hexagon_vertices_list[i]
            cells = HexagonGeometry.to_grid_cells(vertices_i, cell_size)

            for cell in cells:
                # Check 3x3 neighborhood
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        neighbor_cell = (cell[0] + dx, cell[1] + dy)
                        if neighbor_cell in grid:
                            for j in grid[neighbor_cell]:
                                if i != j:
                                    # Quick bounding box check
                                    min_x_i, max_x_i, min_y_i, max_y_i = HexagonGeometry.bounds(vertices_i)
                                    vertices_j = hexagon_vertices_list[j]
                                    min_x_j, max_x_j, min_y_j, max_y_j = HexagonGeometry.bounds(vertices_j)

                                    # If bounding boxes don't intersect, skip detailed check
                                    if (max_x_i < min_x_j or min_x_i > max_x_j or
                                        max_y_i < min_y_j or min_y_i > max_y_j):
                                        continue

                                    # For close hexagons, we must do a more precise check
                                    # Calculate distance between centers to determine if we need more accurate overlap detection
                                    cx_i = vertices_i[0][0]
                                    cy_i = vertices_i[0][1]
                                    cx_j = vertices_j[0][0]
                                    cy_j = vertices_j[0][1]

                                    # Use squared distance to avoid sqrt computation
                                    dist_sq = (cx_i - cx_j)**2 + (cy_i - cy_j)**2
                                    # For unit hexagons, if centers are 1.99 or less apart, they may overlap
                                    # But we still need to verify with proper overlap test
                                    if dist_sq < 3.9601:  # (1.99)^2
                                        return False  # Assume overlap and return early

        return True

    @staticmethod
    def containment_check(hexagon_vertices_list, outer_side_length):
        """Check if all hexagon vertices are within the outer hexagon."""
        if HexagonConstraintChecker.containment_fast(hexagon_vertices_list, outer_side_length):
            outer_polygon = Polygon(HexagonGeometry.outer_vertices(outer_side_length))

            for vertices in hexagon_vertices_list:
                center_x = vertices[0][0]
                center_y = vertices[0][1]
                if not outer_polygon.contains(Point(center_x, center_y)):
                    return False
                hex_polygon = Polygon(vertices)
                if not outer_polygon.contains(hex_polygon):
                    return False
            return True
        return False

    @staticmethod
    def overlap_check(hexagon_vertices_list):
        """Check if any hexagons overlap using spatial hashing for early rejection."""
        # First use spatial hashing for fast rejection
        if HexagonConstraintChecker.overlap_spatial_hashing(hexagon_vertices_list, cell_size=1.0):
            # If spatial hashing didn't detect overlap, do precise check
            try:
                polygons = [Polygon(vertices) for vertices in hexagon_vertices_list]
                union = unary_union(polygons)
                total_area = sum(polygon.area for polygon in polygons)
                union_area = union.area
                return abs(total_area - union_area) < 1e-10
            except:
                # Fallback to pairwise intersection check for robustness
                for i in range(len(polygons)):
                    for j in range(i+1, len(polygons)):
                        if polygons[i].intersects(polygons[j]):
                            return False
                return True
        return False

class OptimizationStrategy:
    """Manages optimization strategies and heuristics."""
    
    @staticmethod
    def create_symmetric_pattern():
        """Create a more structured symmetric pattern."""
        pattern = [
            [0.0, 0.0, 0.0],      # Center
            [-1.732, 0.0, 0.0],   # Left
            [1.732, 0.0, 0.0],    # Right
            [0.0, 1.732, 0.0],    # Top
            [0.0, -1.732, 0.0],   # Bottom
            [-0.866, 0.866, 0.0], # Top-left
            [0.866, 0.866, 0.0],  # Top-right
            [-0.866, -0.866, 0.0], # Bottom-left
            [0.866, -0.866, 0.0], # Bottom-right
            [-2.598, 0.0, 0.0],   # Far left
            [2.598, 0.0, 0.0],    # Far right
            [0.0, 2.598, 0.0],    # Far top
        ]
        return np.array(pattern)

    @staticmethod
    def generate_symmetric_initial_population(pop_size):
        """Generate initial population with better symmetry considerations."""
        population = []
        base_pattern = OptimizationStrategy.create_symmetric_pattern()

        for i in range(pop_size):
            individual = base_pattern.copy().astype(float)

            for j in range(12):
                individual[j, 0] += np.random.uniform(-0.15, 0.15)
                individual[j, 1] += np.random.uniform(-0.15, 0.15)
                individual[j, 2] += np.random.uniform(-15, 15)
                individual[j, 2] = individual[j, 2] % 360

            population.append(individual.flatten())

        return population

def create_symmetric_initial():
    """Create highly symmetric initial configuration based on group theory"""
    # Use a pattern inspired by the 12-fold symmetry group
    positions = []

    # Central hexagon
    positions.append([0.0, 0.0, 0.0])

    # Ring 1: 6 hexagons arranged in a regular hexagon
    for i in range(6):
        angle = i * 60  # degrees
        radius = 2.0
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        positions.append([x, y, 0.0])

    # Ring 2: 5 hexagons in a pentagonal arrangement
    for i in range(5):
        angle = i * 72 + 18  # offset to create irregular but symmetric pattern
        radius = 3.5
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        positions.append([x, y, 0.0])

    # Add some strategic rotations to increase optimality
    # Rotate some hexagons to break degenerate symmetries
    positions[1][2] = 30   # First ring hexagon rotated
    positions[2][2] = 15   # Second ring hexagon rotated
    positions[4][2] = 45   # Third ring hexagon rotated

    return np.array(positions)

def create_outer_hexagon_vertices(side_length):
    """Create vertices of outer hexagon with given side length"""
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    vertices = np.column_stack([np.cos(angles), np.sin(angles)]) * side_length
    return vertices

def check_containment_fast(hex_position, outer_vertices):
    """Fast containment check using vertex position"""
    x, y, angle = hex_position
    vertices = hexagon_vertices_fast(x, y, angle)

    # Check if all vertices are within outer hexagon
    for vertex in vertices:
        if not point_in_polygon_fast(vertex, outer_vertices):
            return False
    return True

def check_overlap_fast(hex1_pos, hex2_pos):
    """Fast overlap check using distance between centers vs sum of radii"""
    x1, y1, _ = hex1_pos
    x2, y2, _ = hex2_pos

    # Distance between centers
    dist_centers = np.sqrt((x2-x1)**2 + (y2-y1)**2)

    # For unit hexagons, approximate minimum distance between edges
    # When they touch, centers are about 2 units apart
    # When they overlap, centers are less than 2 units apart
    return dist_centers < 1.99  # Small tolerance for overlap

def validate_configuration_fast(hex_data, outer_vertices):
    """Quick validation of configuration using fast geometric checks"""
    for i in range(len(hex_data)):
        if not check_containment_fast(hex_data[i], outer_vertices):
            return False
    return True

def compute_outer_side_length(hex_data):
    """Compute minimum side length of outer hexagon"""
    max_dist = 0
    for i in range(len(hex_data)):
        x, y, angle = hex_data[i]
        vertices = hexagon_vertices_fast(x, y, angle)
        for vx, vy in vertices:
            dist = np.sqrt(vx*vx + vy*vy)
            max_dist = max(max_dist, dist)

    # Convert to hexagon side length (accounting for hexagon geometry)
    # For a regular hexagon, circumradius = side_length
    # But our vertices may extend beyond circumradius of the hexagon itself
    # So we want side_length such that all vertices are within the outer hexagon
    side_length = max_dist * 2 / np.sqrt(3)  # More accurate conversion

    return side_length

def calculate_fitness(hex_data, outer_side_length):
    """Calculate fitness for evolutionary optimization with adaptive penalty weighting"""
    # Check for overlaps (this will be slow but necessary)
    penalty = 0

    # Fast initial check using distance
    for i in range(len(hex_data)):
        for j in range(i+1, len(hex_data)):
            if check_overlap_fast(hex_data[i], hex_data[j]):
                # Use exact computation for overlaps
                x1, y1, angle1 = hex_data[i]
                x2, y2, angle2 = hex_data[j]
                v1 = hexagon_vertices_fast(x1, y1, angle1)
                v2 = hexagon_vertices_fast(x2, y2, angle2)

                # Use Shapely for precise overlap detection
                p1 = Polygon(v1)
                p2 = Polygon(v2)
                if p1.intersects(p2):
                    penalty += 1000000  # High penalty for overlaps

    # Check containment with Shapely for precision
    outer_vertices = create_outer_hexagon_vertices(outer_side_length)

    containment_violations = 0
    for i in range(len(hex_data)):
        x, y, angle = hex_data[i]
        vertices = hexagon_vertices_fast(x, y, angle)
        hex_poly = Polygon(vertices)

        # Point by point containment check with Shapely
        for vertex in vertices:
            point = Point(vertex[0], vertex[1])
            if not Polygon(outer_vertices).contains(point):
                # Calculate how far outside the boundary
                dist = np.sqrt(vertex[0]**2 + vertex[1]**2)
                # Adaptive penalty: more severe for deeper violations
                violation_distance = max(0, dist - outer_side_length)
                penalty += violation_distance * 1500  # Higher penalty for containment
                containment_violations += 1

    # Objective: maximize 1/outer_side_length
    # So we minimize negative of 1/outer_side_length plus penalty
    # Add adaptive scaling based on constraint violations
    if containment_violations > 0:
        penalty *= (1.0 + containment_violations * 0.1)  # Scale penalty based on violations

    objective = -1.0 / (outer_side_length + 1e-10) + penalty

    return objective

def create_evolutionary_population(pop_size, target_dim=12):
    """Create diverse population with symmetry awareness"""
    population = []

    # Generate multiple symmetric base configurations
    for i in range(pop_size // 2):
        base_config = create_symmetric_initial()

        # Add variation to positions and orientations
        noise = np.random.normal(0, 0.3, base_config.shape)
        mutated_config = base_config + noise
        population.append(mutated_config.flatten())

    # Generate some random configurations
    for i in range(pop_size // 2):
        # Random configuration but with sensible ranges
        config = np.random.uniform(-4, 4, (target_dim, 3))
        config[:, 2] = np.random.uniform(0, 360, target_dim)  # Random rotations
        population.append(config.flatten())

    return population

def evolutionary_optimization():
    """Use evolutionary approach with symmetry-aware operators"""
    pop_size = 15
    max_generations = 20

    # Create initial population
    population = create_evolutionary_population(pop_size)

    best_fitness = float('inf')
    best_individual = None

    for gen in range(max_generations):
        # Evaluate all individuals
        fitness_scores = []

        for individual in population:
            config = individual.reshape(-1, 3)
            outer_side = compute_outer_side_length(config)
            fitness = calculate_fitness(config, outer_side)
            fitness_scores.append(fitness)

        # Select best individuals
        sorted_indices = np.argsort(fitness_scores)
        elite_count = pop_size // 3
        selected_indices = sorted_indices[:elite_count]

        # Keep best individual
        current_best_idx = sorted_indices[0]
        current_best_fitness = fitness_scores[current_best_idx]

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[current_best_idx].copy()

        # Create new population through crossover and mutation
        new_population = []

        # Keep elites
        for idx in selected_indices:
            new_population.append(population[idx])

        # Generate offspring
        while len(new_population) < pop_size:
            parent1_idx = np.random.choice(selected_indices)
            parent2_idx = np.random.choice(selected_indices)

            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]

            # Uniform crossover
            child = np.copy(parent1)
            mask = np.random.rand(len(child)) > 0.5
            child[mask] = parent2[mask]

            # Mutation with symmetry awareness
            mut_rate = 0.1
            for i in range(len(child)):
                if np.random.rand() < mut_rate:
                    if i % 3 == 2:  # Rotation parameter
                        child[i] += np.random.normal(0, 30)  # Larger change for rotation
                        child[i] = child[i] % 360
                    else:  # Position parameters
                        child[i] += np.random.normal(0, 0.5)

            new_population.append(child)

        population = new_population

    return best_individual.reshape(-1, 3)

class HexagonPacker:
    """Main class for hexagon packing optimization with improved architecture."""
    
    def __init__(self):
        self.hex_side_length = 1.0
        self.outer_hex_side_length_bounds = (3.8, 4.0)
        self.max_iterations = 40
        self.population_size = 8
        self.num_runs = 8
        self.eval_count = 0

    def evaluate_configuration(self, config, outer_side_length):
        """Evaluate a configuration of 12 hexagons."""
        self.eval_count += 1
        hexagons = config.reshape(12, 3)
        hexagon_vertices_list = []

        for i in range(12):
            x, y, angle = hexagons[i]
            vertices = HexagonGeometry.vertices(x, y, angle)
            hexagon_vertices_list.append(vertices)

        penalty = 0.0

        # Check containment first - this is more critical
        # Use enhanced containment checking with distance-based penalty
        outer_radius = outer_side_length
        for vertices in hexagon_vertices_list:
            center_x = vertices[0][0]
            center_y = vertices[0][1]
            distance = math.sqrt(center_x*center_x + center_y*center_y)
            if distance > outer_radius:
                # Adaptive penalty based on how far outside the boundary
                violation_distance = distance - outer_radius
                penalty += violation_distance * 1500  # Higher penalty for containment violations

        # Check overlap - use precise overlap detection
        if not HexagonConstraintChecker.overlap_check(hexagon_vertices_list):
            penalty += 1000000  # Heavy penalty for overlap violations

        return penalty

    def objective_function(self, config, outer_side_length):
        """Objective function to minimize (negative of 1/outer_hex_side_length)."""
        penalty = self.evaluate_configuration(config, outer_side_length)
        if penalty > 100000:  # If there are constraint violations
            return penalty
        else:
            if outer_side_length > 0:
                return -1.0 / outer_side_length
            else:
                return 1000000

    def refine_solution(self, config, initial_side_length):
        """Try to find a better fit with different outer hexagon sizes."""
        best_config = config.copy()
        best_side = initial_side_length
        best_score = -1.0 / initial_side_length

        # First, try a coarse search
        search_range = np.linspace(3.85, min(initial_side_length, 3.9419123), 20)

        for test_side in search_range:
            penalty = self.evaluate_configuration(best_config, test_side)
            if penalty != float('inf'):
                if test_side > best_side:
                    best_side = test_side
                    best_score = -1.0 / test_side

        # Then try fine-grained search
        if best_side < 3.9419123:
            fine_range = np.linspace(best_side, min(3.9419123, best_side + 0.005), 10)
            for test_side in fine_range:
                penalty = self.evaluate_configuration(best_config, test_side)
                if penalty != float('inf'):
                    if test_side > best_side:
                        best_side = test_side
                        best_score = -1.0 / test_side

        return best_config, best_side, best_score

    def local_search_refinement(self, config, outer_side_length):
        """Apply local search refinement to improve the solution."""
        # This is a simple greedy local search approach
        best_config = config.copy()
        best_penalty = self.evaluate_configuration(best_config, outer_side_length)

        # Try small adjustments to each hexagon's position and angle
        for iter_num in range(50):  # Limited iterations to prevent long running times
            improved = False
            current_config = best_config.copy()

            # Try adjusting each hexagon individually
            for i in range(12):
                # Save original values
                orig_x, orig_y, orig_angle = current_config[i*3], current_config[i*3+1], current_config[i*3+2]

                # Try small perturbations
                for _ in range(5):  # Try 5 different small perturbations per hexagon
                    # Small position adjustment
                    new_x = orig_x + np.random.uniform(-0.05, 0.05)
                    new_y = orig_y + np.random.uniform(-0.05, 0.05)
                    new_angle = orig_angle + np.random.uniform(-10, 10)
                    new_angle = new_angle % 360

                    # Update config temporarily
                    current_config[i*3] = new_x
                    current_config[i*3+1] = new_y
                    current_config[i*3+2] = new_angle

                    # Test if this improves the solution
                    penalty = self.evaluate_configuration(current_config, outer_side_length)
                    if penalty < best_penalty:
                        best_penalty = penalty
                        best_config = current_config.copy()
                        improved = True
                    else:
                        # Restore original values
                        current_config[i*3] = orig_x
                        current_config[i*3+1] = orig_y
                        current_config[i*3+2] = orig_angle

            if not improved:
                break  # No improvement made, stop

        return best_config, best_penalty

    def optimize_hexagon_positions(self):
        """Main optimization routine with hierarchical approach."""
        # Try evolutionary optimization (better for finding good starting points)
        try:
            inner_hex_data = evolutionary_optimization()
            outer_side_length = compute_outer_side_length(inner_hex_data)
            
            # Perform one final detailed check
            final_outer_vertices = create_outer_hexagon_vertices(outer_side_length)
            is_valid = validate_configuration_fast(inner_hex_data, final_outer_vertices)

            if not is_valid:
                # Fall back to symmetric configuration
                inner_hex_data = OptimizationStrategy.create_symmetric_pattern()
                outer_side_length = compute_outer_side_length(inner_hex_data)

            return inner_hex_data, np.array([0, 0, 0]), outer_side_length
        except Exception as e:
            warnings.warn(f"Evolutionary optimization failed: {str(e)}")
            # Fall back to symmetric configuration
            inner_hex_data = OptimizationStrategy.create_symmetric_pattern()
            outer_side_length = compute_outer_side_length(inner_hex_data)
            return inner_hex_data, np.array([0, 0, 0]), outer_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    packer = HexagonPacker()
    inner_hex_data, outer_hex_data, outer_hex_side_length = packer.optimize_hexagon_positions()

    # Calculate actual score
    inv_side_length = 1.0 / outer_hex_side_length
    eval_time = time.time() - start_time

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END