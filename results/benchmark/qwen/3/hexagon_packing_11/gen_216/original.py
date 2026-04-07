# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
import math
from scipy.optimize import differential_evolution, minimize
import time

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

    # Check pairwise overlaps
    for i in range(len(inner_hexagons)):
        for j in range(i+1, len(inner_hexagons)):
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

def generate_initial_config():
    """Generate an initial configuration for 11 hexagons"""
    # Enhanced initial configuration based on known good packings
    # This configuration is designed to be more compact than simple lattice layouts

    # Using hexagon side length = 1, distance between centers = sqrt(3)
    hex_spacing = math.sqrt(3)  # distance between adjacent hexagon centers

    # Known good packing pattern from literature-inspired layout
    initial_positions = [
        # Center hexagon
        [0, 0, 0],

        # Surrounding hexagons in a compact formation
        [-hex_spacing, 0, 0],     # left
        [hex_spacing, 0, 0],      # right
        [0, hex_spacing, 0],      # top
        [0, -hex_spacing, 0],    # bottom
        [-hex_spacing/2, hex_spacing/2, 0],  # top-left
        [hex_spacing/2, hex_spacing/2, 0],   # top-right
        [-hex_spacing/2, -hex_spacing/2, 0], # bottom-left
        [hex_spacing/2, -hex_spacing/2, 0],  # bottom-right
        [-hex_spacing * 1.5, hex_spacing/2, 0],   # extended top-left
        [hex_spacing * 1.5, hex_spacing/2, 0],    # extended top-right
    ]

    # Fill remaining positions with symmetrically placed hexagons
    while len(initial_positions) < 11:
        initial_positions.append([0, 0, 0])  # placeholder for unused positions

    return np.array(initial_positions[:11])

def local_refinement_step(initial_positions_angles, max_iter=200):
    """
    Apply local optimization to refine the solution using L-BFGS-B with adaptive settings
    """
    # Define bounds for local optimization (x,y in [-15,15], angle in [0,360])
    bounds = []
    for i in range(11):
        bounds.extend([(-15, 15), (-15, 15), (0, 360)])

    # Define objective function for local optimization
    def objective(params):
        # Reshape parameters into positions and angles
        positions_angles = []
        for i in range(11):
            x = params[i*3]
            y = params[i*3 + 1]
            angle = params[i*3 + 2]
            positions_angles.append([x, y, angle])

        score, side_length = evaluate_layout(positions_angles)
        return score  # Negative because we minimize -score = maximize score

    # Use L-BFGS-B for local refinement with higher precision
    result = minimize(
        objective,
        initial_positions_angles.flatten(),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-10}
    )

    # Return refined positions
    refined_positions = result.x.reshape(-1, 3)
    return refined_positions

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """

    # First, let's try several different approaches to get the best possible result

    # Approach 1: Multi-stage optimization with progressive refinement
    best_result = None
    best_score = -float('inf')

    # Try multiple random seeds for differential evolution
    for seed in [42, 123, 456, 789]:
        try:
            # Generate initial configuration
            initial_positions = generate_initial_config()
            inner_hex_data = initial_positions.copy()

            # Set up optimization bounds for each of the 11 hexagons (x, y, angle)
            bounds = []
            for i in range(11):
                # x and y coordinates bounded to prevent extreme positions
                bounds.extend([(-15, 15), (-15, 15), (0, 360)])

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
                return score  # Negative because we minimize -score = maximize score

            # Use differential evolution for global optimization with increased precision
            result = differential_evolution(
                objective,
                bounds,
                maxiter=200,  # More iterations for better convergence
                popsize=30,   # Larger population size
                seed=seed,
                tol=1e-9,     # Tighter tolerance
                mutation=(0.5, 1),
                recombination=0.7,
                disp=False
            )

            # Extract best solution from global search
            best_params = result.x
            final_positions_angles = []
            for i in range(11):
                x = best_params[i*3]
                y = best_params[i*3 + 1]
                angle = best_params[i*3 + 2]
                final_positions_angles.append([x, y, angle])

            # Apply local refinement to the global optimum with more iterations
            refined_positions = local_refinement_step(np.array(final_positions_angles), max_iter=300)

            # Evaluate final result after refinement
            final_score, final_side_length = evaluate_layout(refined_positions)

            # Check if this result is better than our current best
            if final_score > best_score:
                best_score = final_score
                best_result = (refined_positions, final_side_length)

        except Exception as e:
            print(f"Seed {seed} failed: {e}")
            continue

    # If we found a good solution, use it; otherwise, fall back to initial configuration
    if best_result is not None:
        best_inner_data, best_outer_side_length = best_result
    else:
        # Fallback to initial configuration if all optimization attempts failed
        print("All optimization attempts failed, using initial configuration")
        inner_hex_data = generate_initial_config()
        best_inner_data = inner_hex_data.copy()
        best_outer_side_length = 8.0

    # Final validation and refinement
    inner_hexagons = []
    for pos_angle in best_inner_data:
        x, y, angle = pos_angle
        hex_poly = create_regular_hexagon(x, y, 1, angle)
        inner_hexagons.append(hex_poly)

    # Compute current outer hexagon size
    outer_radius = compute_outer_hexagon_radius(inner_hexagons, 0.01)
    outer_hexagon = create_regular_hexagon(0, 0, outer_radius, 0)

    # Validate constraints - if we have overlaps, fallback to initial configuration
    if not check_containment_and_overlap(inner_hexagons, outer_hexagon):
        print("Final validation failed, falling back to initial configuration")
        inner_hex_data = generate_initial_config()
        best_inner_data = inner_hex_data.copy()
        inner_hexagons = []
        for pos_angle in best_inner_data:
            x, y, angle = pos_angle
            hex_poly = create_regular_hexagon(x, y, 1, angle)
            inner_hexagons.append(hex_poly)
        outer_radius = compute_outer_hexagon_radius(inner_hexagons, 0.01)

    # Ensure we're returning the correct data format
    outer_hex_data = np.array([0, 0, 0])  # centered at origin

    # Return results
    return best_inner_data, outer_hex_data, outer_radius

# EVOLVE-BLOCK-END