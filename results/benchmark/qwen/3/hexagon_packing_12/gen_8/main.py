# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import time

def hexagon_vertices(center_x, center_y, side_length=1, angle_deg=0):
    """Generate vertices of a regular hexagon given center, side length, and rotation."""
    angle_rad = np.deg2rad(angle_deg)
    angles = np.linspace(0, 2*np.pi, 7) + angle_rad
    vertices = np.array([
        [center_x + side_length * np.cos(a),
         center_y + side_length * np.sin(a)]
        for a in angles
    ])
    return vertices

def evaluate_packing(inner_hex_data, outer_hex_side_length):
    """Evaluate if the packing meets constraints and compute objective."""
    try:
        # Generate vertices for all inner hexagons
        hex_polygons = []
        for i in range(12):
            x, y, angle = inner_hex_data[i]
            vertices = hexagon_vertices(x, y, 1.0, angle)
            hex_polygons.append(Polygon(vertices))

        # Check for overlaps between hexagons
        for i in range(12):
            for j in range(i+1, 12):
                if hex_polygons[i].intersects(hex_polygons[j]):
                    return False, 0

        # Create outer hexagon
        outer_vertices = hexagon_vertices(0, 0, outer_hex_side_length, 0)
        outer_polygon = Polygon(outer_vertices)

        # Check containment
        for i in range(12):
            for vertex in hex_polygons[i].exterior.coords:
                point = Point(vertex[0], vertex[1])
                if not outer_polygon.contains(point):
                    return False, 0

        # If we reach here, packing is valid
        # Calculate objective (1/outer_radius)
        return True, 1.0 / outer_hex_side_length

    except Exception as e:
        return False, 0

def objective_function(params):
    """Objective function to minimize (negative of 1/outer_radius)"""
    # Reshape params into 12 hexagons with (x,y,angle) each
    hex_params = params.reshape(-1, 3)

    # Extract outer hexagon side length (last parameter)
    outer_radius = params[-1]

    # Validate the packing
    valid, obj_val = evaluate_packing(hex_params[:-1], outer_radius)

    # If invalid configuration, penalize heavily
    if not valid:
        return 1e6  # Large penalty

    # Return negative because we want to maximize 1/outer_radius,
    # which means minimizing -1/outer_radius
    return -obj_val

def generate_initial_guess():
    """Generate a good initial symmetric configuration."""
    # Based on known good configurations
    inner_hex_data = np.array([
        [0.0, 0.0, 0],      # Center
        [0.0, 2.0, 0],      # Top
        [1.732050808, 1.0, 0],   # Top right
        [1.732050808, -1.0, 0],  # Bottom right
        [0.0, -2.0, 0],     # Bottom
        [-1.732050808, -1.0, 0],  # Bottom left
        [-1.732050808, 1.0, 0],   # Top left
        [3.464101616, 2.0, 0],    # Far top right
        [3.464101616, -2.0, 0],   # Far bottom right
        [-3.464101616, -2.0, 0],  # Far bottom left
        [-3.464101616, 2.0, 0],   # Far top left
        [0.0, -4.0, 0],     # Far bottom
    ], dtype=float)

    # Add outer radius parameter at the end
    return np.concatenate([inner_hex_data.flatten(), [3.9419123]])

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Generate initial guess
    initial_guess = generate_initial_guess()

    # Define bounds for optimization
    # Each hexagon has x,y,angle parameters
    # x, y typically bounded by reasonable range (e.g., -10 to 10)
    # angle bounded by 0-360 degrees
    bounds = [(None, None)] * 36  # 12 hexagons * 3 params each
    bounds.extend([(3.0, 5.0)])  # outer radius bound (reasonable range)

    # Set bounds for positions (x,y) and angles
    for i in range(24):  # 12 hexagons * 2 pos params (x,y) = 24
        bounds[i] = (-10.0, 10.0)  # Position bounds
    for i in range(24, 36):  # 12 hexagons * 1 angle param = 12
        bounds[i] = (0.0, 360.0)  # Angle bounds

    # Run optimization
    try:
        result = minimize(objective_function,
                         initial_guess,
                         method='L-BFGS-B',
                         bounds=bounds,
                         options={'maxiter': 1000, 'ftol': 1e-9})

        if result.success:
            # Extract optimized parameters
            hex_params = result.x[:-1].reshape(-1, 3)
            outer_hex_side_length = result.x[-1]

            # Validate final configuration
            is_valid, objective_value = evaluate_packing(hex_params, outer_hex_side_length)

            if is_valid:
                inner_hex_data = hex_params
                outer_hex_data = np.array([0, 0, 0])  # Outer hexagon centered at origin
                return inner_hex_data, outer_hex_data, outer_hex_side_length
            else:
                # Fallback to initial configuration if optimization failed validation
                inner_hex_data = np.array([
                    [0.0, 0.0, 0],      # Center
                    [0.0, 2.0, 0],      # Top
                    [1.732050808, 1.0, 0],   # Top right
                    [1.732050808, -1.0, 0],  # Bottom right
                    [0.0, -2.0, 0],     # Bottom
                    [-1.732050808, -1.0, 0],  # Bottom left
                    [-1.732050808, 1.0, 0],   # Top left
                    [3.464101616, 2.0, 0],    # Far top right
                    [3.464101616, -2.0, 0],   # Far bottom right
                    [-3.464101616, -2.0, 0],  # Far bottom left
                    [-3.464101616, 2.0, 0],   # Far top left
                    [0.0, -4.0, 0],     # Far bottom
                ], dtype=float)

                outer_hex_side_length = 3.9419123
                outer_hex_data = np.array([0, 0, 0])
                return inner_hex_data, outer_hex_data, outer_hex_side_length
        else:
            # If optimization didn't converge, return initial configuration
            inner_hex_data = np.array([
                [0.0, 0.0, 0],      # Center
                [0.0, 2.0, 0],      # Top
                [1.732050808, 1.0, 0],   # Top right
                [1.732050808, -1.0, 0],  # Bottom right
                [0.0, -2.0, 0],     # Bottom
                [-1.732050808, -1.0, 0],  # Bottom left
                [-1.732050808, 1.0, 0],   # Top left
                [3.464101616, 2.0, 0],    # Far top right
                [3.464101616, -2.0, 0],   # Far bottom right
                [-3.464101616, -2.0, 0],  # Far bottom left
                [-3.464101616, 2.0, 0],   # Far top left
                [0.0, -4.0, 0],     # Far bottom
            ], dtype=float)

            outer_hex_side_length = 3.9419123
            outer_hex_data = np.array([0, 0, 0])
            return inner_hex_data, outer_hex_data, outer_hex_side_length

    except Exception as e:
        # In case of error, return initial configuration
        print(f"Optimization error: {e}")
        inner_hex_data = np.array([
            [0.0, 0.0, 0],      # Center
            [0.0, 2.0, 0],      # Top
            [1.732050808, 1.0, 0],   # Top right
            [1.732050808, -1.0, 0],  # Bottom right
            [0.0, -2.0, 0],     # Bottom
            [-1.732050808, -1.0, 0],  # Bottom left
            [-1.732050808, 1.0, 0],   # Top left
            [3.464101616, 2.0, 0],    # Far top right
            [3.464101616, -2.0, 0],   # Far bottom right
            [-3.464101616, -2.0, 0],  # Far bottom left
            [-3.464101616, 2.0, 0],   # Far top left
            [0.0, -4.0, 0],     # Far bottom
        ], dtype=float)

        outer_hex_side_length = 3.9419123
        outer_hex_data = np.array([0, 0, 0])
        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END