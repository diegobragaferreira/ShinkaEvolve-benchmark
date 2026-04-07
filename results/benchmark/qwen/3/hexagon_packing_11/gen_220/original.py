# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
import time
import random
from copy import deepcopy

def hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, angle, and side length"""
    angle_rad = np.radians(angle_deg)
    angles = np.linspace(0, 2*np.pi, 7) + angle_rad
    vertices = np.array([
        [center_x + side_length * np.cos(a), center_y + side_length * np.sin(a)]
        for a in angles
    ])
    return vertices

def check_containment(hex_vertices, outer_hex_vertices):
    """Check if hexagon vertices are contained within outer hexagon using Shapely"""
    inner_polygon = Polygon(hex_vertices)
    outer_polygon = Polygon(outer_hex_vertices)
    return outer_polygon.contains(inner_polygon)

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely"""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def compute_outer_hexagon_radius(inner_positions, inner_angles, initial_radius_estimate=5.0):
    """Compute minimum outer hexagon radius that contains all inner hexagons with fast binary search"""
    left = initial_radius_estimate
    right = 20.0
    best_radius = right

    # Fast binary search with early exit criteria
    precision_threshold = 1e-5
    max_iterations = 50

    for _ in range(max_iterations):
        if right - left <= precision_threshold:
            break
        mid = (left + right) / 2.0
        outer_vertices = hexagon_vertices(0, 0, 0, mid)
        valid = True

        for i, (pos, angle) in enumerate(zip(inner_positions, inner_angles)):
            hex_vertices = hexagon_vertices(pos[0], pos[1], angle)
            if not check_containment(hex_vertices, outer_vertices):
                valid = False
                break

        if valid:
            best_radius = mid
            right = mid
        else:
            left = mid

    return best_radius

def evaluate_fitness(inner_positions, inner_angles, max_radius=20.0):
    """Evaluate fitness: higher is better, maximize 1/radius"""
    outer_radius = compute_outer_hexagon_radius(inner_positions, inner_angles)

    # Check all constraints
    total_penalty = 0

    # Check containment for all inner hexagons
    outer_vertices = hexagon_vertices(0, 0, 0, outer_radius)
    for i, (pos, angle) in enumerate(zip(inner_positions, inner_angles)):
        hex_vertices = hexagon_vertices(pos[0], pos[1], angle)
        if not check_containment(hex_vertices, outer_vertices):
            total_penalty += 10000  # Large penalty for containment violation

    # Check overlaps between all pairs of inner hexagons
    for i in range(len(inner_positions)):
        for j in range(i+1, len(inner_positions)):
            hex1_vertices = hexagon_vertices(inner_positions[i][0], inner_positions[i][1], inner_angles[i])
            hex2_vertices = hexagon_vertices(inner_positions[j][0], inner_positions[j][1], inner_angles[j])
            if check_overlap(hex1_vertices, hex2_vertices):
                total_penalty += 10000  # Large penalty for overlap violation

    # Fitness is negative of the radius plus penalties
    fitness = -outer_radius - total_penalty

    return fitness, outer_radius

def generate_voronoi_initial_config():
    """Generate initial configuration using Voronoi sampling for better spatial distribution"""
    # Predefined seed points for better initial layout
    seed_points = [
        [0, 0],           # center
        [-2.5, 0],        # left
        [2.5, 0],         # right
        [-1.25, 2.17],    # top-left
        [1.25, 2.17],     # top-right
        [-1.25, -2.17],   # bottom-left
        [1.25, -2.17],    # bottom-right
        [-3.75, 2.17],    # far top-left
        [3.75, 2.17],     # far top-right
        [-3.75, -2.17],   # far bottom-left
        [3.75, -2.17],    # far bottom-right
    ]

    # Add jitter to seed points for diversity
    individual = np.array(seed_points)
    for i in range(len(individual)):
        individual[i][0] += random.uniform(-0.2, 0.2)
        individual[i][1] += random.uniform(-0.2, 0.2)

    # Add rotation information
    rotation_array = np.array([0] * 11)  # All at 0 degrees initially

    return individual, rotation_array

def voronoi_based_optimization():
    """Main optimization using Voronoi-inspired configuration generation"""
    # Generate initial configuration using Voronoi-like approach
    initial_positions, initial_angles = generate_voronoi_initial_config()

    # Flatten positions for optimization
    initial_vars = np.hstack([initial_positions.flatten(), initial_angles])

    # Optimization bounds
    bounds = []
    # Position bounds
    for _ in range(len(initial_positions)):
        bounds.extend([(-10, 10), (-10, 10)])  # Larger bounds for flexibility
    # Angle bounds (0-360 degrees)
    for _ in range(len(initial_angles)):
        bounds.extend([(0, 360)])

    def objective_func(vars):
        # Unpack variables
        positions_flat = vars[:-11]  # All positions
        angles = vars[-11:]          # All angles

        # Reshape positions
        positions = positions_flat.reshape(-1, 2)

        # Evaluate fitness (we want to maximize 1/radius, so we minimize -1/radius)
        try:
            fitness, _ = evaluate_fitness(positions, angles)
            return -fitness  # Minimize negative fitness to maximize fitness
        except:
            return 1000000  # Penalty for invalid configurations

    # Use a simple bounded optimization approach
    try:
        # First do a coarse optimization
        result = minimize(
            objective_func,
            initial_vars,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
        )

        if result.success:
            final_vars = result.x
            positions_flat = final_vars[:-11]
            angles = final_vars[-11:]
            positions = positions_flat.reshape(-1, 2)

            # Evaluate final fitness
            final_fitness, outer_radius = evaluate_fitness(positions, angles)

            return positions, angles, outer_radius
        else:
            # Fallback to initial solution
            return initial_positions, initial_angles, 20.0
    except Exception:
        return initial_positions, initial_angles, 20.0

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Use Voronoi-based optimization approach
    positions, angles, outer_radius = voronoi_based_optimization()

    # Format output as required
    inner_hex_data = np.column_stack([positions, angles])
    outer_hex_data = np.array([0, 0, 0])  # centered at origin

    return inner_hex_data, outer_hex_data, outer_radius

# EVOLVE-BLOCK-END