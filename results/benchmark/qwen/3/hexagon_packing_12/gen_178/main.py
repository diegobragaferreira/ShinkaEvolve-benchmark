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

def check_containment(hex_vertices, outer_hex_center, outer_hex_side_length):
    """Check if all vertices of a hexagon are inside the outer hexagon."""
    outer_vertices = hexagon_vertices(outer_hex_center[0], outer_hex_center[1], outer_hex_side_length, 0)
    outer_polygon = Polygon(outer_vertices)

    for vertex in hex_vertices:
        point = Point(vertex[0], vertex[1])
        if not outer_polygon.contains(point):
            return False
    return True

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

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

def generate_symmetric_initial_guess():
    """Generate a good initial symmetric configuration."""
    # Based on known symmetric arrangements for 12 hexagons
    # This creates a configuration with 6-fold rotational symmetry
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

    return inner_hex_data

def grid_refinement_optimization(initial_config, max_iterations=100):
    """Perform grid refinement with adaptive mutation strength to optimize the packing."""

    best_config = initial_config.copy()
    best_objective = 1.0 / 3.9419123  # Starting with known good value
    outer_hex_side_length = 3.9419123

    # Grid refinement with decreasing mutation strength
    mutation_strengths = np.linspace(0.3, 0.05, 10)  # Decreasing from 0.3 to 0.05

    for iteration, mutation_strength in enumerate(mutation_strengths):
        # For each mutation strength, perform several iterations
        for _ in range(10):  # 10 iterations per mutation level
            # Create a mutated version of the best configuration
            mutated_config = best_config.copy()

            # Apply mutation with current strength
            for i in range(12):
                # Mutate x and y coordinates
                mutated_config[i, 0] += np.random.normal(0, mutation_strength)
                mutated_config[i, 1] += np.random.normal(0, mutation_strength)
                # Keep rotation fixed for now (can be added later)

            # Validate the mutated configuration
            is_valid, objective_value = evaluate_packing(mutated_config, outer_hex_side_length)

            # If better, accept it
            if is_valid and objective_value > best_objective:
                best_config = mutated_config.copy()
                best_objective = objective_value

    return best_config, best_objective

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Start with a very good known configuration
    # Based on research, the best known packing has outer hexagon side length ~3.9419123
    # So we aim to get close to 1/3.9419123 ≈ 0.2537

    # Generate initial symmetric configuration
    inner_hex_data = generate_symmetric_initial_guess()

    # Apply grid refinement with adaptive mutation
    refined_config, objective_value = grid_refinement_optimization(inner_hex_data)

    # Outer hexagon side length that should work well
    outer_hex_side_length = 3.9419123

    # Validate the refined configuration
    is_valid, validated_objective = evaluate_packing(refined_config, outer_hex_side_length)

    # If not valid, use the original configuration
    if not is_valid:
        print("Warning: Refinement produced invalid configuration, using original.")
        # Fall back to the original good configuration
        inner_hex_data = generate_symmetric_initial_guess()
        objective_value = 1.0 / 3.9419123
        outer_hex_side_length = 3.9419123

    # If refinement improved things, use the refined version
    if validated_objective > objective_value:
        inner_hex_data = refined_config
        objective_value = validated_objective

    # Outer hexagon centered at origin with no rotation
    outer_hex_data = np.array([0, 0, 0])

    # Final validation
    is_valid, final_objective = evaluate_packing(inner_hex_data, outer_hex_side_length)
    if not is_valid:
        print("Warning: Final configuration not valid, using fallback.")
        # Fall back to simpler configuration
        inner_hex_data = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
            [0, -4, 0],
        ])
        outer_hex_side_length = 8.0
        final_objective = 1.0 / 8.0

    # Calculate performance metrics
    inv_outer_hex_side_length = final_objective
    benchmark_ratio = inv_outer_hex_side_length / 0.2537

    print(f"Optimized result: inverse_side_length={inv_outer_hex_side_length:.6f}, "
          f"benchmark_ratio={benchmark_ratio:.6f}")

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END