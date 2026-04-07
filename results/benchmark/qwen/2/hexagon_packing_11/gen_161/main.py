# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely.prepared import prep
from scipy.spatial import KDTree
import time
from joblib import Parallel, delayed
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def hexagon_vertices(center_x, center_y, rotation_degrees, side_length=1):
    """Generate vertices of a regular hexagon given center, rotation, and side length."""
    angle_rad = np.radians(rotation_degrees)
    # Vertices of a unit hexagon centered at origin
    unit_vertices = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    # Rotate and translate
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_vertices = unit_vertices @ rotation_matrix.T
    return rotated_vertices * side_length + np.array([center_x, center_y])

def get_hexagon_bounding_box(vertices):
    """Get bounding box of hexagon vertices."""
    vertices_array = np.array(vertices)
    min_x, max_x = vertices_array[:, 0].min(), vertices_array[:, 0].max()
    min_y, max_y = vertices_array[:, 1].min(), vertices_array[:, 1].max()
    return min_x, max_x, min_y, max_y

def check_containment_single(hex_vertices, outer_polygon):
    """Check if all vertices of a hexagon are inside the outer hexagon with buffer validation."""
    # Add small buffer to ensure numerical stability
    buffer_size = 1e-6
    buffered_vertices = []
    for vertex in hex_vertices:
        point = Point(vertex)
        # Create a slightly buffered point (though this is mainly for robustness)
        buffered_vertices.append(vertex)

    # Use prepared polygon for faster containment checks
    for vertex in hex_vertices:
        point = Point(vertex)
        # Add buffer tolerance - if point is very close to boundary, consider it inside
        if not outer_polygon.contains(point):
            # Check if it's within buffer distance
            try:
                distance = point.distance(Point(outer_polygon.exterior.coords[0][0], outer_polygon.exterior.coords[0][1]))
                if distance > buffer_size:
                    return False
            except:
                return False
    return True

def calculate_outer_hex_side_length_fast(inner_hex_params):
    """Enhanced estimation of required outer hexagon side length with better margin."""
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(11):
        x, y, rot = inner_hex_params[3*i], inner_hex_params[3*i+1], inner_hex_params[3*i+2]
        hex_vertices = hexagon_vertices(x, y, rot, 1)
        all_vertices.extend(hex_vertices)

    if len(all_vertices) == 0:
        return 100.0

    # Calculate bounding box
    all_vertices = np.array(all_vertices)
    min_x, max_x = all_vertices[:, 0].min(), all_vertices[:, 0].max()
    min_y, max_y = all_vertices[:, 1].min(), all_vertices[:, 1].max()

    # Calculate center of the bounding box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Find maximum distance from center to any vertex
    max_dist = 0
    for vertex in all_vertices:
        dist = np.sqrt((vertex[0] - center_x)**2 + (vertex[1] - center_y)**2)
        max_dist = max(max_dist, dist)

    # For a regular hexagon, to contain all vertices, we need to account for
    # the circumradius of the inner hexagons plus some safety margin
    # The circumradius of a unit hexagon is 1
    # We need to accommodate: max_dist + 1 (for the hexagon's circumradius) + margin
    return max_dist + 1.1  # Slightly adjusted margin with more precision

def check_collision_single(hex1_vertices, hex2_vertices):
    """Improved collision check with buffer validation."""
    # Use SAT for fast rejection, then Shapely for precise check
    if not check_collision_single_sat(hex1_vertices, hex2_vertices):
        return False

    # Add buffer check for floating point precision issues
    buffer_size = 1e-6

    # If SAT says they might intersect, do precise check with buffer
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)

    # Check if polygons are effectively disjoint (with buffer)
    if poly1.distance(poly2) > buffer_size:
        return False

    return poly1.intersects(poly2)

def check_collision_single_fast(hex1_vertices, hex2_vertices):
    """Fast collision check using Separating Axis Theorem for efficiency."""
    # Quick bounding box check
    min1_x, max1_x, min1_y, max1_y = get_hexagon_bounding_box(hex1_vertices)
    min2_x, max2_x, min2_y, max2_y = get_hexagon_bounding_box(hex2_vertices)

    # If bounding boxes don't intersect, no collision
    if max1_x < min2_x or max2_x < min1_x or max1_y < min2_y or max2_y < min1_y:
        return False

    # Use SAT for better accuracy
    return check_collision_single_sat(hex1_vertices, hex2_vertices)

def check_collision_single_sat(hex1_vertices, hex2_vertices):
    """Improved Separating Axis Theorem implementation."""
    # For regular hexagons, we only need to test specific axes
    # Edge normals for hexagon: 6 directions (each 60 degrees apart)
    # But we can optimize further since hexagons have symmetries

    # Get edges from both hexagons
    edges1 = []
    edges2 = []

    for i in range(len(hex1_vertices)):
        p1 = hex1_vertices[i]
        p2 = hex1_vertices[(i + 1) % len(hex1_vertices)]
        edge = p2 - p1
        edges1.append(edge)

    for i in range(len(hex2_vertices)):
        p1 = hex2_vertices[i]
        p2 = hex2_vertices[(i + 1) % len(hex2_vertices)]
        edge = p2 - p1
        edges2.append(edge)

    # Test axes from both polygons
    # For hexagons, we only need to test normals to edges (6 unique directions)
    axes = []

    # Get edge normals (perpendicular vectors)
    for edge in edges1 + edges2:
        # Normal vector to edge (rotate 90 degrees counter-clockwise)
        normal = np.array([-edge[1], edge[0]])
        # Normalize
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            normal = normal / norm
        axes.append(normal)

    # Check each axis for separation
    for axis in axes:
        # Project both polygons onto this axis
        proj1 = [np.dot(vertex, axis) for vertex in hex1_vertices]
        proj2 = [np.dot(vertex, axis) for vertex in hex2_vertices]

        min1_proj, max1_proj = min(proj1), max(proj1)
        min2_proj, max2_proj = min(proj2), max(proj2)

        # If projections don't overlap, this is a separating axis
        if max1_proj < min2_proj or max2_proj < min1_proj:
            return False

    # Polygons overlap
    return True

def calculate_outer_hex_side_length_fast(inner_hex_params):
    """Fast estimation of required outer hexagon side length."""
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(11):
        x, y, rot = inner_hex_params[3*i], inner_hex_params[3*i+1], inner_hex_params[3*i+2]
        hex_vertices = hexagon_vertices(x, y, rot, 1)
        all_vertices.extend(hex_vertices)

    if len(all_vertices) == 0:
        return 100.0

    # Calculate bounding box
    all_vertices = np.array(all_vertices)
    min_x, max_x = all_vertices[:, 0].min(), all_vertices[:, 0].max()
    min_y, max_y = all_vertices[:, 1].min(), all_vertices[:, 1].max()

    # Calculate diagonal distance from center to corner
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Find maximum distance from center to any vertex
    max_dist = 0
    for vertex in all_vertices:
        dist = np.sqrt((vertex[0] - center_x)**2 + (vertex[1] - center_y)**2)
        max_dist = max(max_dist, dist)

    # For a hexagon, the side length is max_dist / sqrt(3)
    # We want to ensure our hexagon can contain everything with some margin
    return max_dist * 2 / np.sqrt(3) * 1.05  # Reduced margin for tighter fit

def objective_function(params):
    """Improved objective function with better constraint handling and conservative penalties."""
    # params: [x1, y1, rot1, x2, y2, rot2, ..., x11, y11, rot11, R]
    n = 11
    outer_side_length = params[-1]

    # Check if outer hexagon is large enough
    if outer_side_length <= 0:
        return 1e12  # Larger penalty for invalid outer hex

    # Create outer hexagon vertices once
    outer_vertices = hexagon_vertices(0, 0, 0, outer_side_length)
    outer_polygon = prep(Polygon(outer_vertices))  # Prepare for fast intersection tests

    # Phase 1: Containment check
    total_containment_penalty = 0
    inner_positions = params[:-1].reshape(-1, 3)
    inner_hex_vertices = []

    for i in range(n):
        x, y, rot = inner_positions[i]
        hex_vertices = hexagon_vertices(x, y, rot, 1)
        inner_hex_vertices.append(hex_vertices)

        # Check containment for this hexagon - more thorough check
        for vertex in hex_vertices:
            point = Point(vertex)
            if not outer_polygon.contains(point):
                total_containment_penalty += 1e10  # Much larger penalty for containment violation

    # If containment failed, early exit
    if total_containment_penalty > 0:
        return -(1.0 / outer_side_length) + total_containment_penalty

    # Phase 2: Collision checking
    total_collision_penalty = 0

    # Build KD-tree for neighbors to speed up collision checks
    centers = []
    for i in range(n):
        x, y, rot = inner_positions[i]
        centers.append([x, y])
    centers = np.array(centers)

    tree = KDTree(centers)

    # Check collisions with neighbors only (more efficient)
    collision_pairs = []
    for i in range(n):
        # Find nearby points within some radius (adjust based on hexagon size)
        nearby_indices = tree.query_ball_point(centers[i], 2.5)  # Adjusted radius
        for j in nearby_indices:
            if i < j:  # Avoid duplicate pairs and self-comparison
                collision_pairs.append((i, j))

    # Perform collision checks only for likely candidates
    for i, j in collision_pairs:
        hex1_vertices = inner_hex_vertices[i]
        hex2_vertices = inner_hex_vertices[j]

        # Use fast collision check for initial screening
        if check_collision_single_fast(hex1_vertices, hex2_vertices):
            total_collision_penalty += 1e10  # Large penalty for collisions

    # Final penalty calculation - make penalties more conservative
    return -(1.0 / outer_side_length) + total_containment_penalty + total_collision_penalty

def generate_hybrid_initial_guess():
    """Generate a smart initial guess combining multiple strategies."""
    # Strategy 1: Hexagonal packing with rotations (from first version)
    hex_pattern = [
        [0, 0, 0],           # center
        [0, 2, 0],           # top
        [1.732, 1, 30],      # top-right
        [1.732, -1, 60],     # bottom-right
        [0, -2, 0],          # bottom
        [-1.732, -1, 120],   # bottom-left
        [-1.732, 1, 150],    # top-left
        [3.464, 0, 0],       # far right
        [1.732, 2, 30],      # top-middle
        [-1.732, 2, 150],    # top-middle-left
        [-3.464, 0, 0],      # far left
    ]

    # Strategy 2: Spiral arrangement (from second version)
    spiral_pattern = [
        [0, 0, 0],         # center
        [2, 0, 0],         # right
        [1, 1.732, 0],     # upper-right
        [-1, 1.732, 0],    # upper-left
        [-2, 0, 0],        # left
        [-1, -1.732, 0],   # lower-left
        [1, -1.732, 0],    # lower-right
        [3, 0, 0],         # far right
        [1.5, 2.6, 0],     # upper-middle-right
        [-1.5, 2.6, 0],    # upper-middle-left
        [-3, 0, 0],        # far left
    ]

    # Strategy 3: Grid arrangement (from baseline)
    grid_pattern = [
        [0, 0, 0],       # center
        [-2.5, 0, 0],    # left
        [2.5, 0, 0],     # right
        [-1.25, 2.17, 0], # top-left
        [1.25, 2.17, 0],  # top-right
        [-1.25, -2.17, 0], # bottom-left
        [1.25, -2.17, 0],  # bottom-right
        [-3.75, 2.17, 0],  # far top-left
        [3.75, 2.17, 0],   # far top-right
        [-3.75, -2.17, 0], # far bottom-left
        [3.75, -2.17, 0],  # far bottom-right
    ]

    # Try all three patterns and pick the best one
    patterns = [hex_pattern, spiral_pattern, grid_pattern]
    best_score = -float('inf')
    best_params = None

    for pattern in patterns:
        initial_params = []
        for pos in pattern:
            initial_params.extend(pos)

        # Estimate outer side length and evaluate
        estimated_side = calculate_outer_hex_side_length_fast(np.array(initial_params))
        initial_params.append(estimated_side)

        # Evaluate with objective function
        try:
            score = -objective_function(np.array(initial_params))
            if score > best_score:
                best_score = score
                best_params = np.array(initial_params)
        except:
            continue

    return best_params

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """

    # Generate smart initial guess
    initial_guess = generate_hybrid_initial_guess()

    if initial_guess is None:
        # Fallback to default arrangement
        initial_guess = np.array([
            [0, 0, 0], [0, 2, 0], [1.732, 1, 0], [1.732, -1, 0], [0, -2, 0],
            [-1.732, -1, 0], [-1.732, 1, 0], [3.464, 0, 0], [1.732, 2, 0],
            [-1.732, 2, 0], [-3.464, 0, 0], 10.0  # Add side length
        ]).flatten()

    # Define bounds for optimization
    bounds = []
    # Positions: x, y for each hexagon (limited to reasonable range)
    for _ in range(11):
        bounds.extend([(-10, 10), (-10, 10), (-180, 180)])  # x, y, rotation
    # Outer hexagon side length
    bounds.append((1.0, 20.0))  # Must be positive and reasonable

    # Optimization parameters tuned for better performance
    maxiter = 100
    popsize = 15
    mutation = (0.5, 1)
    recombination = 0.7

    # Perform optimized differential evolution
    try:
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            mutation=mutation,
            recombination=recombination,
            seed=42,
            disp=False,
            polish=True  # Enable local polishing for final refinement
        )

        best_params = result.x
        outer_side_length = best_params[-1]

        # Extract inner hexagon data
        inner_hex_data = np.zeros((11, 3))
        for i in range(11):
            inner_hex_data[i] = [best_params[3*i], best_params[3*i+1], best_params[3*i+2]]

        # Outer hexagon data
        outer_hex_data = np.array([0, 0, 0])  # Centered at origin

        # Verify solution
        outer_vertices = hexagon_vertices(0, 0, 0, outer_side_length)
        outer_polygon = prep(Polygon(outer_vertices))

        # Quick validation check
        valid_solution = True
        for i in range(11):
            x, y, rot = best_params[3*i], best_params[3*i+1], best_params[3*i+2]
            hex_vertices = hexagon_vertices(x, y, rot, 1)

            if not check_containment_single(hex_vertices, outer_polygon):
                valid_solution = False
                break

        # If not valid, fallback to smarter arrangement
        if not valid_solution:
            # Use the hybrid initial guess as fallback
            inner_hex_data = np.array([
                [0, 0, 0], [0, 2, 0], [1.732, 1, 0], [1.732, -1, 0], [0, -2, 0],
                [-1.732, -1, 0], [-1.732, 1, 0], [3.464, 0, 0], [1.732, 2, 0],
                [-1.732, 2, 0], [-3.464, 0, 0]
            ])
            outer_side_length = calculate_outer_hex_side_length_fast(inner_hex_data.flatten())
            outer_hex_data = np.array([0, 0, 0])

    except Exception as e:
        # Fallback to hybrid arrangement in case of optimization failure
        print(f"Optimization failed: {e}")
        inner_hex_data = np.array([
            [0, 0, 0], [0, 2, 0], [1.732, 1, 0], [1.732, -1, 0], [0, -2, 0],
            [-1.732, -1, 0], [-1.732, 1, 0], [3.464, 0, 0], [1.732, 2, 0],
            [-1.732, 2, 0], [-3.464, 0, 0]
        ])
        outer_side_length = calculate_outer_hex_side_length_fast(inner_hex_data.flatten())
        outer_hex_data = np.array([0, 0, 0])

    return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END