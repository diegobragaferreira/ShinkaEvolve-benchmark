# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from scipy.optimize import differential_evolution, minimize
import random
import time
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')

# Constants
NUM_INNER_HEX = 11
UNIT_HEX_RADIUS = 1.0
HEX_VERTICES = 6
ROTATION_STEPS = 12  # 30 degree increments

@jit(nopython=True)
def get_hexagon_vertices(x, y, angle_deg, radius=1.0):
    """Get vertices of a regular hexagon given position and angle - JIT compiled"""
    vertices = np.zeros((6, 2))
    angle_rad = np.radians(angle_deg)
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i] = [x + radius * np.cos(theta), y + radius * np.sin(theta)]
    return vertices

@jit(nopython=True)
def point_in_hexagon(point_x, point_y, hex_center_x, hex_center_y, hex_radius, angle_deg):
    """Fast point-in-hexagon test - JIT compiled"""
    # Transform point to hexagon's local coordinate system
    angle_rad = np.radians(angle_deg)
    rel_x = point_x - hex_center_x
    rel_y = point_y - hex_center_y

    # Rotate point back to align with hexagon axes
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    rot_x = rel_x * cos_a - rel_y * sin_a
    rot_y = rel_x * sin_a + rel_y * cos_a

    # Hexagon width = 2 * radius * cos(π/6) = radius * √3
    hex_width = hex_radius * np.sqrt(3)
    half_width = hex_width / 2

    # Check bounds
    if abs(rot_x) > half_width:
        return False

    # Maximum y based on x position in hexagon
    # For a regular hexagon with side length r, height = r * √3
    max_y = hex_radius * np.sqrt(3) / 2

    if abs(rot_y) > max_y:
        return False

    return True

def estimate_min_outer_radius(inner_hex_data):
    """
    Estimate minimum outer hexagon radius that can contain all inner hexagons
    using a more precise geometric approach.
    """
    if len(inner_hex_data) == 0:
        return 1000.0

    # Find the maximum distance from origin to any vertex of any inner hexagon
    max_distance = 0.0

    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        vertices = get_hexagon_vertices(x, y, angle)

        # Calculate distance from center to each vertex
        for vertex in vertices:
            distance = np.sqrt(vertex[0]**2 + vertex[1]**2)
            max_distance = max(max_distance, distance)

    # Add a small buffer to ensure complete containment
    # The outer hexagon needs to be large enough so that any vertex of inner hexagons
    # lies inside the outer hexagon
    return max_distance * 1.05  # Add 5% buffer

def calculate_outer_hex_side_length(inner_hex_data, outer_hex_center=(0, 0)):
    """Calculate minimum outer hexagon side length needed to contain all inner hexagons"""
    if len(inner_hex_data) == 0:
        return 1000.0

    max_distance = 0.0
    center_x, center_y = outer_hex_center

    # For each inner hexagon, check all 6 vertices
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        vertices = get_hexagon_vertices(x, y, angle)

        # Calculate distance from center to each vertex
        for vertex in vertices:
            distance = np.sqrt((vertex[0] - center_x)**2 + (vertex[1] - center_y)**2)
            max_distance = max(max_distance, distance)

    # Account for hexagon radius
    # The outer hexagon needs to be large enough so that any vertex of inner hexagons
    # lies inside the outer hexagon
    return max_distance * 2.0 / np.sqrt(3)  # Convert circumradius to side length

def check_containment_sat(hex_vertices, outer_center=(0, 0), outer_radius=1000.0):
    """Check if hexagon vertices are within the outer hexagon using SAT-like approach"""
    outer_center_x, outer_center_y = outer_center
    # Check if all vertices are within the outer hexagon
    # Outer hexagon circumscribed circle has radius = outer_radius * sqrt(3)/2
    outer_circumradius = outer_radius * np.sqrt(3) / 2

    for vertex in hex_vertices:
        dist_from_center = np.sqrt((vertex[0] - outer_center_x)**2 + (vertex[1] - outer_center_y)**2)
        if dist_from_center > outer_circumradius:
            return False
    return True

def check_overlap_sat(hex1_vertices, hex2_vertices):
    """SAT-based overlap detection - much faster than Shapely for many comparisons"""
    # Simple SAT test: check if there's a separating axis
    # For hexagons, we only need to test 12 axes (each edge normal)
    def get_edges(vertices):
        edges = []
        n = len(vertices)
        for i in range(n):
            edges.append(vertices[i] - vertices[(i+1)%n])
        return edges

    def project_polygon_onto_axis(vertices, axis):
        projections = []
        for vertex in vertices:
            proj = np.dot(vertex, axis)
            projections.append(proj)
        return min(projections), max(projections)

    edges1 = get_edges(hex1_vertices)
    edges2 = get_edges(hex2_vertices)

    # Get normals to all edges (perpendicular vectors)
    normals1 = []
    normals2 = []

    for edge in edges1:
        # Normal vector (perpendicular to edge)
        normal = np.array([-edge[1], edge[0]])
        normals1.append(normal / np.linalg.norm(normal))  # Normalize

    for edge in edges2:
        # Normal vector (perpendicular to edge)
        normal = np.array([-edge[1], edge[0]])
        normals2.append(normal / np.linalg.norm(normal))  # Normalize

    # Test all axes
    all_normals = normals1 + normals2

    for axis in all_normals:
        min1, max1 = project_polygon_onto_axis(hex1_vertices, axis)
        min2, max2 = project_polygon_onto_axis(hex2_vertices, axis)

        # Check for overlap
        if max1 < min2 or max2 < min1:
            return False  # No overlap along this axis

    return True  # Overlap detected

def check_overlap_shapely(hex1_vertices, hex2_vertices):
    """Fallback to Shapely overlap detection"""
    try:
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)
    except:
        return False

def evaluate_individual(individual):
    """Evaluate fitness of individual solution with improved error handling"""
    try:
        # Reshape individual into (11, 3) array of (x, y, angle)
        hex_data = np.array(individual).reshape(-1, 3)

        # Calculate required outer hex side length
        outer_side_length = calculate_outer_hex_side_length(hex_data)

        # Initialize penalty
        penalty = 0.0

        # Check containment constraints
        # Outer hex is centered at origin with calculated side length
        outer_radius = outer_side_length * np.sqrt(3) / 2  # Circumradius

        # Check each hexagon for containment using optimized checks
        for i in range(NUM_INNER_HEX):
            x, y, angle = hex_data[i]
            vertices = get_hexagon_vertices(x, y, angle)

            # Quick containment check using SAT-like approach
            if not check_containment_sat(vertices, (0, 0), outer_radius):
                penalty += 1000000.0  # Heavy penalty

        # Check for overlaps between hexagons using SAT
        overlap_pairs = 0
        for i in range(NUM_INNER_HEX):
            for j in range(i+1, NUM_INNER_HEX):
                x1, y1, angle1 = hex_data[i]
                x2, y2, angle2 = hex_data[j]

                vertices1 = get_hexagon_vertices(x1, y1, angle1)
                vertices2 = get_hexagon_vertices(x2, y2, angle2)

                # Use SAT-based check for efficiency
                if check_overlap_sat(vertices1, vertices2):
                    penalty += 1000000.0  # Heavy penalty
                    overlap_pairs += 1

        # Fitness is negative inverse of side length plus penalties
        # We want to minimize side length, so maximize 1/side_length
        fitness = -1.0 / outer_side_length
        if penalty > 0:
            fitness -= penalty  # Add penalty for constraint violations

        # Add a bonus for non-overlapping solutions
        if penalty == 0 and overlap_pairs == 0:
            fitness += 0.5  # Small bonus for valid solutions

        return (fitness,)
    except Exception as e:
        return (-1000000.0,)

def generate_initial_solutions(num_solutions=50):
    """Generate diverse initial solutions using multiple strategies"""
    solutions = []

    # Strategy 1: Grid-based placements
    for _ in range(num_solutions // 2):
        individual = []
        # Center hexagon at origin with fixed orientation
        individual.extend([0, 0, 0])

        # Generate grid pattern around center
        grid_positions = [
            (0, 2.5), (0, -2.5), (2.5, 0), (-2.5, 0),
            (2.5, 2.5), (-2.5, 2.5), (2.5, -2.5), (-2.5, -2.5),
            (3.5, 1.5), (-3.5, 1.5), (3.5, -1.5)
        ]

        for i, pos in enumerate(grid_positions[:10]):  # Leave one for center
            x = pos[0] + random.uniform(-0.3, 0.3)
            y = pos[1] + random.uniform(-0.3, 0.3)
            angle = random.uniform(0, 360.0)
            individual.extend([x, y, angle])

        solutions.append(individual)

    # Strategy 2: Spiral-based placements
    for _ in range(num_solutions // 4):
        individual = []
        # Center hexagon at origin
        individual.extend([0, 0, 0])

        # Spiral pattern
        for i in range(10):
            angle = i * 36.0  # 10 degrees per step
            radius = 1.0 + i * 0.5
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            angle_rot = random.uniform(0, 360.0)
            individual.extend([x, y, angle_rot])

        solutions.append(individual)

    # Strategy 3: Random placements with good bounds
    for _ in range(num_solutions // 4):
        individual = []
        # Center hexagon at origin
        individual.extend([0, 0, 0])

        # Random placements within reasonable bounds
        for _ in range(10):
            x = random.uniform(-4.0, 4.0)
            y = random.uniform(-4.0, 4.0)
            angle = random.uniform(0, 360.0)
            individual.extend([x, y, angle])

        solutions.append(individual)

    return solutions

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """

    # Generate diverse starting solutions
    initial_solutions = generate_initial_solutions(50)

    # Run differential evolution for global search
    bounds = [(-10, 10), (-10, 10), (0, 360)] * NUM_INNER_HEX

    try:
        # Use differential evolution for global optimization
        result = differential_evolution(
            lambda x: -evaluate_individual(x)[0],
            bounds,
            seed=42,
            maxiter=100,
            popsize=20,
            mutation=(0.5, 1),
            recombination=0.7,
            disp=False,
            tol=1e-6
        )

        best_individual = result.x
        best_fitness = -result.fun

    except Exception as e:
        # Fallback to best initial solution if nothing worked
        print(f"Differential evolution failed: {e}")
        best_individual = None
        best_fitness = -float('inf')

        # Try to find the best among initial solutions
        for sol in initial_solutions:
            try:
                fitness = evaluate_individual(sol)[0]
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = sol
            except:
                continue

    # Perform local refinement if we have a solution
    if best_individual is not None:
        try:
            # Local optimization with L-BFGS
            refined_result = minimize(
                lambda x: -evaluate_individual(x)[0],
                best_individual,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50},
                tol=1e-6
            )

            if refined_result.success:
                best_individual = refined_result.x
                best_fitness = -refined_result.fun

        except Exception as e:
            print(f"L-BFGS refinement failed: {e}")
            pass

    # Convert best individual to desired format
    if best_individual is None:
        # Fallback to simple grid if nothing worked
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
        outer_hex_data = np.array([0, 0, 0])  # centered at origin
        outer_hex_side_length = 8  # large enough to contain all inner hexagons
    else:
        # Convert best individual to hex data
        inner_hex_data = np.array(best_individual).reshape(-1, 3)
        outer_hex_side_length = calculate_outer_hex_side_length(inner_hex_data)
        outer_hex_data = np.array([0, 0, 0])  # outer hexagon centered at origin

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END