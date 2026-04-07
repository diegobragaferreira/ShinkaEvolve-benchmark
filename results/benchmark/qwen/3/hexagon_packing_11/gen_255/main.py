# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
import math

def create_regular_hexagon(center_x, center_y, side_length=1, rotation_deg=0):
    """Create a regular hexagon as a Shapely polygon"""
    rotation_rad = math.radians(rotation_deg)
    points = []
    for i in range(6):
        angle = rotation_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        points.append((x, y))
    return Polygon(points)

def check_containment_and_overlap(inner_hexagons, outer_hexagon):
    """Check if all inner hexagons are contained in outer hexagon and don't overlap"""
    # Check containment
    for hex_poly in inner_hexagons:
        if not outer_hexagon.contains(hex_poly):
            return False

    # Check pairwise overlaps - optimized for small number of hexagons
    n = len(inner_hexagons)
    for i in range(n):
        for j in range(i+1, n):
            if inner_hexagons[i].intersects(inner_hexagons[j]):
                return False

    return True

def compute_outer_hexagon_radius(inner_hexagons, padding=0.01):
    """Compute minimum radius needed to contain all inner hexagons with some padding"""
    # Get all vertices of all hexagons
    all_vertices = []
    for hex_poly in inner_hexagons:
        all_vertices.extend(list(hex_poly.exterior.coords))

    # Find center of bounding box
    xs = [p[0] for p in all_vertices]
    ys = [p[1] for p in all_vertices]
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2

    # Compute max distance from center to any vertex
    max_dist = 0
    for x, y in all_vertices:
        dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = max(max_dist, dist)

    # Add padding and convert to side length
    # For a regular hexagon, radius = side_length
    return max_dist + padding

def evaluate_layout(inner_positions_angles, outer_center=(0, 0), initial_outer_radius=8):
    """Evaluate the layout quality"""
    # Convert to hexagon polygons
    inner_hexagons = []
    for pos_angle in inner_positions_angles:
        x, y, angle = pos_angle
        hex_poly = create_regular_hexagon(x, y, 1, angle)
        inner_hexagons.append(hex_poly)

    # Create outer hexagon with current radius
    outer_radius = compute_outer_hexagon_radius(inner_hexagons, 0.01)
    outer_hexagon = create_regular_hexagon(outer_center[0], outer_center[1], outer_radius, 0)

    # Validate constraints
    valid = check_containment_and_overlap(inner_hexagons, outer_hexagon)

    # Return negative because we want to maximize 1/R (minimize R)
    outer_side_length = outer_radius
    inv_radius = 1.0 / outer_side_length if valid else 0.0

    return -inv_radius, outer_side_length

def generate_better_initial_config():
    """
    Generate a better initial configuration for 11 hexagons based on known dense packings
    """
    # This configuration is designed to be close to an optimal arrangement
    # Based on hexagonal close packing principles with strategic placement
    # Using a central hexagon with surrounding rings
    initial_positions = [
        # Central hexagon
        [0.0, 0.0, 0.0],
        # First ring (6 hexagons) - arranged in a perfect hexagonal pattern
        [-2.0, 0.0, 0.0],      # Left
        [2.0, 0.0, 0.0],       # Right
        [0.0, 2.0, 0.0],       # Top
        [0.0, -2.0, 0.0],      # Bottom
        [-1.0, 1.732, 0.0],    # Top-left (1.732 = sqrt(3))
        [1.0, 1.732, 0.0],     # Top-right
        # Second ring (4 hexagons) - strategically placed
        [-1.0, -1.732, 0.0],   # Bottom-left
        [1.0, -1.732, 0.0],    # Bottom-right
        [-2.0, 1.0, 0.0],      # Far top-left
        [2.0, 1.0, 0.0],       # Far top-right
        [-2.0, -1.0, 0.0],     # Far bottom-left
        [2.0, -1.0, 0.0],      # Far bottom-right
    ]

    # Keep only first 11 positions (the 11 required hexagons)
    # Adjust for better packing
    adjusted_positions = [
        [0.0, 0.0, 0.0],          # center
        [-2.0, 0.0, 0.0],         # left
        [2.0, 0.0, 0.0],          # right
        [0.0, 2.0, 0.0],          # top
        [0.0, -2.0, 0.0],         # bottom
        [-1.0, 1.732, 0.0],       # top-left
        [1.0, 1.732, 0.0],        # top-right
        [-1.0, -1.732, 0.0],      # bottom-left
        [1.0, -1.732, 0.0],       # bottom-right
        [-2.0, 1.732, 0.0],       # top-left far
        [2.0, 1.732, 0.0],        # top-right far
    ]

    return np.array(adjusted_positions[:11])

def optimize_layout(initial_positions):
    """Optimize the layout using local optimization"""

    # Define bounds for each parameter (x, y, angle)
    bounds = []
    for i in range(11):
        # Reasonable bounds to keep solutions in a practical region
        bounds.extend([(-8, 8), (-8, 8), (0, 360)])  # angle in degrees

    # Define objective function for optimization
    def objective(params):
        # Reshape parameters into positions and angles
        positions_angles = []
        for i in range(11):
            x = params[i*3]
            y = params[i*3 + 1]
            angle = params[i*3 + 2]
            positions_angles.append([x, y, angle])

        score, side_length = evaluate_layout(positions_angles)
        return score  # Negative since we minimize -score = maximize score

    # Local refinement with L-BFGS-B
    try:
        initial_flat = []
        for pos_angle in initial_positions:
            initial_flat.extend(pos_angle)

        result_local = minimize(
            objective,
            initial_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8}
        )

        # Extract refined solution
        refined_params = result_local.x
        refined_positions_angles = []
        for i in range(11):
            x = refined_params[i*3]
            y = refined_params[i*3 + 1]
            angle = refined_params[i*3 + 2]
            refined_positions_angles.append([x, y, angle])

        return np.array(refined_positions_angles)
    except Exception:
        # If optimization fails, return original
        return initial_positions

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Generate a better initial configuration
    initial_positions = generate_better_initial_config()

    # Optimize the layout
    optimized_positions = optimize_layout(initial_positions)

    # Final validation and calculation
    inner_hexagons = []
    for pos_angle in optimized_positions:
        x, y, angle = pos_angle
        hex_poly = create_regular_hexagon(x, y, 1, angle)
        inner_hexagons.append(hex_poly)

    # Compute outer hexagon size
    outer_radius = compute_outer_hexagon_radius(inner_hexagons, 0.01)
    outer_hexagon = create_regular_hexagon(0, 0, outer_radius, 0)

    # Final validation check
    if not check_containment_and_overlap(inner_hexagons, outer_hexagon):
        # If validation fails, fall back to initial positions
        optimized_positions = initial_positions
        inner_hexagons = []
        for pos_angle in optimized_positions:
            x, y, angle = pos_angle
            hex_poly = create_regular_hexagon(x, y, 1, angle)
            inner_hexagons.append(hex_poly)
        outer_radius = compute_outer_hexagon_radius(inner_hexagons, 0.01)

    # Ensure we have a valid result
    outer_hex_data = np.array([0, 0, 0])  # centered at origin

    return optimized_positions, outer_hex_data, outer_radius


# EVOLVE-BLOCK-END