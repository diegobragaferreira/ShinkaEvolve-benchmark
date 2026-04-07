# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.validation import make_valid
import warnings
warnings.filterwarnings('ignore')

def hexagon_vertices(center, side_length, angle_degrees):
    """Generate vertices of a regular hexagon given center, side length, and rotation"""
    angle_rad = np.radians(angle_degrees)
    # Vertices of a regular hexagon with side length 1, centered at origin, unrotated
    base_vertices = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    # Rotate and translate
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    vertices = base_vertices @ rotation_matrix.T + center

    return vertices

def check_containment(hexagon_vertices, outer_hex_vertices):
    """Check if hexagon is fully contained within outer hexagon"""
    inner_poly = Polygon(hexagon_vertices)
    outer_poly = Polygon(outer_hex_vertices)

    # Check if inner polygon is completely contained within outer
    return outer_poly.contains(inner_poly)

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap"""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)

    # Check if polygons intersect
    return poly1.intersects(poly2)

def compute_outer_hex_side_length(inner_hex_data):
    """Estimate the minimum side length of outer hexagon needed to contain all inner hexagons"""
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i, :2]
        angle = inner_hex_data[i, 2]
        vertices = hexagon_vertices(center, 1.0, angle)
        all_vertices.extend(vertices)

    all_vertices = np.array(all_vertices)

    # Find the bounding box of all vertices
    min_x, max_x = np.min(all_vertices[:, 0]), np.max(all_vertices[:, 0])
    min_y, max_y = np.min(all_vertices[:, 1]), np.max(all_vertices[:, 1])

    # The side length of the outer hexagon should be sufficient to contain this bounding box
    # For a regular hexagon, if we know the extent in x and y directions,
    # we need to determine the minimum side length such that it contains all points

    # The distance from center to any vertex of a hexagon with side length s is s
    # So we need to ensure that the outer hexagon has enough radius
    half_width = (max_x - min_x) / 2
    half_height = (max_y - min_y) / 2

    # In a regular hexagon, the distance from center to vertices is equal to side length
    # We need to account for both width and height constraints
    side_length_estimate = max(half_width, half_height) * (2 / np.sqrt(3))

    return side_length_estimate

def objective_function(params):
    """Objective function to maximize 1/outer_hex_side_length"""
    num_hex = 11
    # Parse parameters: 11 hexagons * 3 params each = 33 params
    # But we want to optimize over positions and rotations only
    # First 22 parameters: x,y positions for 11 hexagons
    # Last 11 parameters: rotations for 11 hexagons

    positions = params[:22].reshape(-1, 2)
    rotations = params[22:]

    # Construct data array
    inner_hex_data = np.column_stack([positions, rotations])

    # Compute estimated outer hexagon side length
    outer_side_length = compute_outer_hex_side_length(inner_hex_data)

    # Small penalty for negative side lengths
    if outer_side_length <= 0:
        return 1e6

    # Since we want to maximize 1/outer_side_length, we minimize its negative
    return -1.0 / outer_side_length

def constraint_containment(params, outer_hex_vertices):
    """Constraint function ensuring all inner hexagons are contained"""
    num_hex = 11
    positions = params[:22].reshape(-1, 2)
    rotations = params[22:]

    # Construct data array
    inner_hex_data = np.column_stack([positions, rotations])

    for i in range(num_hex):
        center = inner_hex_data[i, :2]
        angle = inner_hex_data[i, 2]
        vertices = hexagon_vertices(center, 1.0, angle)

        if not check_containment(vertices, outer_hex_vertices):
            return False  # Violates containment constraint

    return True

def constraint_overlap(params):
    """Constraint function ensuring no overlaps between hexagons"""
    num_hex = 11
    positions = params[:22].reshape(-1, 2)
    rotations = params[22:]

    # Construct data array
    inner_hex_data = np.column_stack([positions, rotations])

    # Check all pairs for overlap
    for i in range(num_hex):
        for j in range(i+1, num_hex):
            center1 = inner_hex_data[i, :2]
            angle1 = inner_hex_data[i, 2]
            center2 = inner_hex_data[j, :2]
            angle2 = inner_hex_data[j, 2]

            vertices1 = hexagon_vertices(center1, 1.0, angle1)
            vertices2 = hexagon_vertices(center2, 1.0, angle2)

            if check_overlap(vertices1, vertices2):
                return False  # Violates non-overlap constraint

    return True

def initialize_parameters():
    """Initialize parameters for optimization"""
    # Start with a better initial configuration
    # Place center hexagon at origin, surround with others in a pattern
    positions = np.array([
        [0, 0],      # center
        [-2.0, 0],   # left
        [2.0, 0],    # right
        [0, 2.0],    # top
        [0, -2.0],   # bottom
        [-1.5, 1.5], # top-left
        [1.5, 1.5],  # top-right
        [-1.5, -1.5],# bottom-left
        [1.5, -1.5], # bottom-right
        [-2.5, 0],   # left far
        [2.5, 0],    # right far
    ])

    rotations = np.zeros(11)  # All horizontal initially

    # Combine into single parameter vector
    return np.concatenate([positions.flatten(), rotations])

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """

    # Initialize parameters
    initial_params = initialize_parameters()

    # Define bounds for optimization
    bounds = []
    # Position bounds: reasonable limits to keep hexagons within a reasonable area
    for _ in range(22):
        bounds.append((-10, 10))  # x and y coordinates
    # Rotation bounds: 0 to 360 degrees
    for _ in range(11):
        bounds.append((0, 360))

    # Define constraints
    def constraint_func(params):
        return constraint_overlap(params)

    # Use scipy.optimize to find optimal configuration
    try:
        result = minimize(objective_function,
                         initial_params,
                         method='L-BFGS-B',
                         bounds=bounds,
                         options={'maxiter': 1000})

        if result.success:
            # Extract optimized parameters
            positions = result.x[:22].reshape(-1, 2)
            rotations = result.x[22:]
            inner_hex_data = np.column_stack([positions, rotations])

            # Compute final outer hexagon side length
            outer_side_length = compute_outer_hex_side_length(inner_hex_data)

            # If the optimization didn't work well, fall back to original
            if outer_side_length < 1e-6:
                raise ValueError("Poor optimization result")
        else:
            raise RuntimeError("Optimization failed")

    except Exception as e:
        # Fallback to the original configuration
        print(f"Fallback to original configuration due to error: {e}")
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

        outer_side_length = 8  # large enough to contain all inner hexagons

    # Outer hexagon centered at origin with calculated side length
    outer_hex_data = np.array([0, 0, 0])

    return inner_hex_data, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END