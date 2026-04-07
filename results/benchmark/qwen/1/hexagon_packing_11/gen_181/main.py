# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import math

# Constants for hexagon geometry
HEX_RADIUS = 1.0  # Unit hexagon radius
HEX_APOGEE = HEX_RADIUS * math.sqrt(3)/2  # Distance from center to side midpoint
HEX_HEIGHT = 2 * HEX_APOGEE  # Height of hexagon
HEX_WIDTH = 2 * HEX_RADIUS  # Width of hexagon

# Spatial indexing parameters for efficient collision detection
GRID_CELL_SIZE = 2.0  # Grid cell size should be larger than hexagon diameter for good performance

def get_hexagon_vertices(center_x, center_y, angle_degrees):
    """Get vertices of a unit regular hexagon given center and rotation"""
    # Convert angle to radians
    angle_rad = math.radians(angle_degrees)

    # Vertices of a unit hexagon centered at origin, pointing up
    base_vertices = []
    for i in range(6):
        theta = angle_rad + i * math.pi/3
        x = HEX_RADIUS * math.cos(theta)
        y = HEX_RADIUS * math.sin(theta)
        base_vertices.append((x, y))

    # Translate to center
    vertices = [(x + center_x, y + center_y) for x, y in base_vertices]
    return np.array(vertices)

def build_spatial_grid(hex_data):
    """Build a spatial grid for efficient collision detection"""
    grid = {}

    # Determine appropriate grid bounds based on hexagon extents
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    for cx, cy, angle in hex_data:
        vertices = get_hexagon_vertices(cx, cy, angle)
        for x, y in vertices:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

    # Expand bounds slightly to account for edge cases
    margin = HEX_WIDTH
    min_x -= margin
    max_x += margin
    min_y -= margin
    max_y += margin

    # Calculate grid dimensions
    grid_width = max_x - min_x
    grid_height = max_y - min_y

    # Adjust cell size based on problem size
    avg_cell_size = GRID_CELL_SIZE
    num_cols = int(math.ceil(grid_width / avg_cell_size))
    num_rows = int(math.ceil(grid_height / avg_cell_size))

    # For each hexagon, determine which grid cells it occupies
    for i, (cx, cy, angle) in enumerate(hex_data):
        # Get the bounding box of the hexagon
        vertices = get_hexagon_vertices(cx, cy, angle)
        min_x_h = min(v[0] for v in vertices)
        max_x_h = max(v[0] for v in vertices)
        min_y_h = min(v[1] for v in vertices)
        max_y_h = max(v[1] for v in vertices)

        # Determine grid cells that this hexagon covers
        min_col = int((min_x_h - min_x) // avg_cell_size)
        max_col = int((max_x_h - min_x) // avg_cell_size)
        min_row = int((min_y_h - min_y) // avg_cell_size)
        max_row = int((max_y_h - min_y) // avg_cell_size)

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                if (row, col) not in grid:
                    grid[(row, col)] = []
                grid[(row, col)].append(i)

    return grid, min_x, max_x, min_y, max_y

def get_potential_collisions(grid, hex_data, hex_index, min_x, max_x, min_y, max_y):
    """Get potential collision partners from spatial grid"""
    # Get the bounding box of the hexagon we're checking
    cx, cy, angle = hex_data[hex_index]
    vertices = get_hexagon_vertices(cx, cy, angle)
    min_x_h = min(v[0] for v in vertices)
    max_x_h = max(v[0] for v in vertices)
    min_y_h = min(v[1] for v in vertices)
    max_y_h = max(v[1] for v in vertices)

    # Determine grid cells that this hexagon covers
    avg_cell_size = GRID_CELL_SIZE
    min_col = int((min_x_h - min_x) // avg_cell_size)
    max_col = int((max_x_h - min_x) // avg_cell_size)
    min_row = int((min_y_h - min_y) // avg_cell_size)
    max_row = int((max_y_h - min_y) // avg_cell_size)

    # Collect potential candidates
    candidates = set()
    for row in range(min_row - 1, max_row + 2):
        for col in range(min_col - 1, max_col + 2):
            if (row, col) in grid:
                candidates.update(grid[(row, col)])

    return list(candidates)

def check_hexagon_containment(hex_vertices, outer_center_x, outer_center_y, outer_radius):
    """Check if hexagon vertices are contained within outer hexagon"""
    # For a regular hexagon centered at origin, we can check distance from center
    for vertex in hex_vertices:
        x, y = vertex
        dx = x - outer_center_x
        dy = y - outer_center_y
        distance = math.sqrt(dx*dx + dy*dy)
        if distance >= outer_radius:
            return False
    return True

def hexagon_collision(hex1_vertices, hex2_vertices):
    """Check if two hexagons collide using Separating Axis Theorem"""
    # Quick bounding box check first
    min1_x = min(v[0] for v in hex1_vertices)
    max1_x = max(v[0] for v in hex1_vertices)
    min1_y = min(v[1] for v in hex1_vertices)
    max1_y = max(v[1] for v in hex1_vertices)

    min2_x = min(v[0] for v in hex2_vertices)
    max2_x = max(v[0] for v in hex2_vertices)
    min2_y = min(v[1] for v in hex2_vertices)
    max2_y = max(v[1] for v in hex2_vertices)

    # If bounding boxes don't overlap, no collision possible
    if max1_x < min2_x or max2_x < min1_x or max1_y < min2_y or max2_y < min1_y:
        return False

    # Get all edges of both hexagons
    edges1 = []
    edges2 = []

    for i in range(6):
        p1 = hex1_vertices[i]
        p2 = hex1_vertices[(i+1)%6]
        edge = (p2[0]-p1[0], p2[1]-p1[1])
        edges1.append(edge)

        p1 = hex2_vertices[i]
        p2 = hex2_vertices[(i+1)%6]
        edge = (p2[0]-p1[0], p2[1]-p1[1])
        edges2.append(edge)

    # Combine all potential separating axes
    all_axes = edges1 + edges2

    # Normalize axes and check for zero-length vectors
    normalized_axes = []
    for axis in all_axes:
        length = math.sqrt(axis[0]**2 + axis[1]**2)
        if length > 0:
            normalized_axes.append((axis[0]/length, axis[1]/length))

    if not normalized_axes:
        return False

    # Check projection overlap on each axis
    for axis in normalized_axes:
        # Project both hexagons onto this axis
        proj1 = []
        proj2 = []

        for v in hex1_vertices:
            dot = v[0]*axis[0] + v[1]*axis[1]
            proj1.append(dot)

        for v in hex2_vertices:
            dot = v[0]*axis[0] + v[1]*axis[1]
            proj2.append(dot)

        min1, max1 = min(proj1), max(proj1)
        min2, max2 = min(proj2), max(proj2)

        # If projections don't overlap, then there's separation
        if max1 < min2 or max2 < min1:
            return False

    return True

def calculate_outer_hex_radius(inner_hex_data, outer_center=(0,0)):
    """Calculate minimum radius needed for outer hexagon to contain all inner hexagons"""
    max_distance = 0

    for i in range(len(inner_hex_data)):
        center_x = inner_hex_data[i][0]
        center_y = inner_hex_data[i][1]
        angle = inner_hex_data[i][2]

        # Get vertices of this hexagon
        vertices = get_hexagon_vertices(center_x, center_y, angle)

        # Find maximum distance from outer center to any vertex
        for x, y in vertices:
            distance = math.sqrt((x - outer_center[0])**2 + (y - outer_center[1])**2)
            max_distance = max(max_distance, distance)

    # Add buffer to ensure complete containment
    return max_distance + HEX_RADIUS

def evaluate_fitness(individual):
    """
    Evaluate the fitness of a solution configuration
    individual: array of shape (33,) containing [x1,y1,a1,x2,y2,a2,...,x11,y11,a11]
    Returns negative value because we want to maximize 1/R (minimize R)
    """
    # Reshape individual into hexagon data
    inner_hex_data = individual.reshape(-1, 3)

    # Try different outer hexagon sizes and check feasibility
    # Start with a reasonable estimate
    outer_radius = calculate_outer_hex_radius(inner_hex_data)

    # Check collisions and containment
    num_collisions = 0
    num_out_of_bounds = 0

    # Build spatial grid for efficient collision detection
    grid, min_x, max_x, min_y, max_y = build_spatial_grid(inner_hex_data)

    # Check all hexagon pairs for collision using spatial indexing
    for i in range(len(inner_hex_data)):
        vertices_i = get_hexagon_vertices(
            inner_hex_data[i][0],
            inner_hex_data[i][1],
            inner_hex_data[i][2]
        )

        # Check containment first
        if not check_hexagon_containment(vertices_i, 0, 0, outer_radius):
            num_out_of_bounds += 1
            # Early termination if containment fails
            if num_out_of_bounds > 0:
                penalty = 1000 * (num_collisions + num_out_of_bounds)
                return 1000000 + penalty  # Large penalty for invalid solutions

        # Efficiently get potential collision partners using spatial indexing
        potential_collisions = get_potential_collisions(grid, inner_hex_data, i, min_x, max_x, min_y, max_y)
        for j in potential_collisions:
            if i >= j:
                continue

            vertices_j = get_hexagon_vertices(
                inner_hex_data[j][0],
                inner_hex_data[j][1],
                inner_hex_data[j][2]
            )

            if hexagon_collision(vertices_i, vertices_j):
                num_collisions += 1
                # Early termination if collision found
                if num_collisions > 0:
                    penalty = 1000 * (num_collisions + num_out_of_bounds)
                    return 1000000 + penalty  # Large penalty for invalid solutions

    # Penalty for collisions or out of bounds
    penalty = 1000 * (num_collisions + num_out_of_bounds)

    # If invalid configuration, return poor fitness
    if num_collisions > 0 or num_out_of_bounds > 0:
        return 1000000 + penalty  # Large penalty for invalid solutions

    # Return inverse of outer radius (we want to maximize 1/R)
    return 1.0 / outer_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    np.random.seed(42)  # For reproducibility

    # Better initial configuration based on improved hexagonal lattice arrangement
    # Using a more careful spacing to allow for better packing
    initial_guess = np.array([
        [0, 0, 0],           # center
        [-2.1, 0, 0],        # left
        [2.1, 0, 0],         # right
        [-1.05, 1.8, 0],     # top-left
        [1.05, 1.8, 0],      # top-right
        [-1.05, -1.8, 0],    # bottom-left
        [1.05, -1.8, 0],     # bottom-right
        [-3.15, 1.8, 0],     # far top-left
        [3.15, 1.8, 0],      # far top-right
        [-3.15, -1.8, 0],    # far bottom-left
        [3.15, -1.8, 0],     # far bottom-right
    ]).flatten()

    # Bounds for optimization: positions (-10, 10), rotations (0, 360)
    bounds = []
    for _ in range(11):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])  # x, y, angle

    # Run optimization with bounds
    try:
        result = differential_evolution(
            func=evaluate_fitness,
            bounds=bounds,
            maxiter=100,
            popsize=15,
            seed=42,
            disp=False,
            tol=1e-6
        )

        if result.success:
            # Get the best solution
            best_individual = result.x
            inner_hex_data = best_individual.reshape(-1, 3)

            # Calculate final outer hexagon radius
            outer_radius = calculate_outer_hex_radius(inner_hex_data)

            # Create outer hexagon data (centered at origin)
            outer_hex_data = np.array([0, 0, 0])

            # Return the best solution found
            return inner_hex_data, outer_hex_data, outer_radius
        else:
            # Fallback to initial guess if optimization fails
            pass
    except Exception:
        # If optimization fails, fall back to initial guess
        pass

    # Fallback to initial configuration if anything goes wrong
    inner_hex_data = np.array([
        [0, 0, 0],           # center
        [-2.2, 0, 0],        # left
        [2.2, 0, 0],         # right
        [-1.1, 1.9, 0],      # top-left
        [1.1, 1.9, 0],       # top-right
        [-1.1, -1.9, 0],     # bottom-left
        [1.1, -1.9, 0],      # bottom-right
        [-3.3, 1.9, 0],      # far top-left
        [3.3, 1.9, 0],       # far top-right
        [-3.3, -1.9, 0],     # far bottom-left
        [3.3, -1.9, 0],      # far bottom-right
    ])

    outer_radius = calculate_outer_hex_radius(inner_hex_data)
    outer_hex_data = np.array([0, 0, 0])

    return inner_hex_data, outer_hex_data, outer_radius


# EVOLVE-BLOCK-END