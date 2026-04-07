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

def generate_better_initial_config():
    """
    Generate a better initial configuration for 11 hexagons based on known dense packings
    """
    # More refined configuration based on literature and geometric considerations
    # Using a combination of hexagonal lattice points with optimized spacing

    # Base hexagon positions in a honeycomb-like pattern
    # Center hexagon
    initial_positions = [[0.0, 0.0, 0.0]]

    # First ring - 6 hexagons around the center
    ring1_angles = [0, 60, 120, 180, 240, 300]  # degrees
    ring1_radius = 2.0  # This is the distance between centers of adjacent hexagons
    for angle in ring1_angles:
        rad = math.radians(angle)
        x = ring1_radius * math.cos(rad)
        y = ring1_radius * math.sin(rad)
        initial_positions.append([x, y, 0.0])

    # Second ring - 6 hexagons in a larger ring
    ring2_radius = 3.464  # approximately 2*sqrt(3), which allows tight packing
    for angle in ring1_angles:
        rad = math.radians(angle)
        x = ring2_radius * math.cos(rad)
        y = ring2_radius * math.sin(rad)
        initial_positions.append([x, y, 0.0])

    # Adjust some positions to optimize packing
    # Move some positions to reduce gaps and improve density
    initial_positions[2] = [-1.0, 1.732, 0.0]  # Top-left
    initial_positions[3] = [1.0, 1.732, 0.0]   # Top-right
    initial_positions[7] = [-1.0, -1.732, 0.0]  # Bottom-left
    initial_positions[8] = [1.0, -1.732, 0.0]   # Bottom-right

    # Use specific strategic positions for better packing
    initial_positions[4] = [-2.0, 0.0, 0.0]    # Left
    initial_positions[5] = [2.0, 0.0, 0.0]     # Right
    initial_positions[6] = [0.0, 2.0, 0.0]     # Top
    initial_positions[9] = [0.0, -2.0, 0.0]    # Bottom

    # Adjust second ring to improve packing density
    initial_positions[10] = [-2.0, 1.0, 0.0]   # Far top-left
    initial_positions[11] = [2.0, 1.0, 0.0]    # Far top-right
    initial_positions[12] = [-2.0, -1.0, 0.0]  # Far bottom-left
    initial_positions[13] = [2.0, -1.0, 0.0]   # Far bottom-right

    # Keep only first 11 positions (the 11 required hexagons)
    return np.array(initial_positions[:11])

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
    inner_hex_data = initial_positions.copy()

    # Optimization bounds for each parameter (x, y, angle)
    bounds = []
    for i in range(11):
        # Reasonable bounds to keep solutions in a practical region
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])  # angle in degrees

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

    # Multi-stage optimization approach with iterative refinement
    best_score = float('inf')  # We'll minimize this
    best_inner_data = inner_hex_data.copy()
    best_outer_side_length = 10.0

    # Stage 1: Coarse global optimization with higher resolution
    try:
        print("Starting coarse global optimization...")
        result = differential_evolution(
            objective,
            bounds,
            maxiter=150,  # Increased iterations
            popsize=20,   # Larger population
            seed=42,
            tol=1e-8,
            mutation=(0.5, 1),
            recombination=0.7,
            disp=False
        )

        # Extract best solution from coarse optimization
        best_params = result.x
        final_positions_angles = []
        for i in range(11):
            x = best_params[i*3]
            y = best_params[i*3 + 1]
            angle = best_params[i*3 + 2]
            final_positions_angles.append([x, y, angle])

        # Evaluate final result
        final_score, final_side_length = evaluate_layout(final_positions_angles)

        if final_score < best_score and final_side_length > 0:
            best_score = final_score
            best_inner_data = np.array(final_positions_angles)
            best_outer_side_length = final_side_length

    except Exception as e:
        print(f"Coarse optimization failed: {e}")
        pass

    # Stage 2: Multiple refinement attempts with different optimization methods
    for attempt in range(3):  # Try multiple refinement approaches
        print(f"Starting refinement attempt {attempt + 1}...")

        try:
            # Convert to flat array for scipy optimization
            initial_flat = []
            for pos_angle in best_inner_data:
                initial_flat.extend(pos_angle)

            # Local refinement with L-BFGS-B
            result_local = minimize(
                objective,
                initial_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-9},
                callback=None
            )

            # Extract refined solution
            refined_params = result_local.x
            refined_positions_angles = []
            for i in range(11):
                x = refined_params[i*3]
                y = refined_params[i*3 + 1]
                angle = refined_params[i*3 + 2]
                refined_positions_angles.append([x, y, angle])

            # Evaluate refined result
            refined_score, refined_side_length = evaluate_layout(refined_positions_angles)

            if refined_score < best_score and refined_side_length > 0:
                best_score = refined_score
                best_inner_data = np.array(refined_positions_angles)
                best_outer_side_length = refined_side_length

        except Exception as e:
            print(f"Local optimization attempt {attempt + 1} failed: {e}")
            continue

    # Stage 3: Additional refinement with trust-constr method for better convergence
    try:
        print("Starting trust-constr refinement...")
        # Convert to flat array for scipy optimization
        initial_flat = []
        for pos_angle in best_inner_data:
            initial_flat.extend(pos_angle)

        # Local refinement with trust-constr method
        result_trust = minimize(
            objective,
            initial_flat,
            method='trust-constr',
            bounds=bounds,
            options={'maxiter': 100, 'gtol': 1e-10},
            callback=None
        )

        # Extract refined solution
        trust_params = result_trust.x
        trust_positions_angles = []
        for i in range(11):
            x = trust_params[i*3]
            y = trust_params[i*3 + 1]
            angle = trust_params[i*3 + 2]
            trust_positions_angles.append([x, y, angle])

        # Evaluate refined result
        trust_score, trust_side_length = evaluate_layout(trust_positions_angles)

        if trust_score < best_score and trust_side_length > 0:
            best_score = trust_score
            best_inner_data = np.array(trust_positions_angles)
            best_outer_side_length = trust_side_length

    except Exception as e:
        print(f"Trust-constr optimization failed: {e}")

    # Final validation and refinement
    print("Performing final validation...")

    # Always validate the result
    inner_hexagons = []
    for pos_angle in best_inner_data:
        x, y, angle = pos_angle
        hex_poly = create_regular_hexagon(x, y, 1, angle)
        inner_hexagons.append(hex_poly)

    # Recompute outer hexagon size carefully
    outer_radius = compute_outer_hexagon_radius(inner_hexagons, 0.01)
    outer_hexagon = create_regular_hexagon(0, 0, outer_radius, 0)

    # Final validation check
    if not check_containment_and_overlap(inner_hexagons, outer_hexagon):
        print("Final validation failed, using fallback...")
        # Fall back to initial configuration
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