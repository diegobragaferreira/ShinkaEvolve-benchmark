# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import time
from numba import jit


@jit(nopython=True)
def hexagon_vertices_jit(center_x, center_y, angle_deg, side_length=1):
    """Return vertices of a regular hexagon with given center, rotation, and side length."""
    angle_rad = np.radians(angle_deg)
    # Vertices of a regular hexagon with side length 1, centered at origin
    base_vertices = np.array([
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
    rotated_vertices = base_vertices @ rotation_matrix.T

    return rotated_vertices + np.array([center_x, center_y])


@jit(nopython=True)
def check_overlap_jit(hex1_vertices, hex2_vertices):
    """Check if two hexagon vertices overlap using separating axis theorem."""
    # Get all edges of both hexagons
    def get_edges(vertices):
        edges = []
        n = len(vertices)
        for i in range(n):
            edges.append(vertices[i] - vertices[(i+1)%n])
        return edges

    edges1 = get_edges(hex1_vertices)
    edges2 = get_edges(hex2_vertices)

    # Project both hexagons onto each edge direction
    all_axes = edges1 + edges2
    for axis in all_axes:
        # Normalize axis
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0:
            continue
        norm_axis = axis / axis_norm

        # Project both polygons onto this axis
        proj1 = [np.dot(vertex, norm_axis) for vertex in hex1_vertices]
        proj2 = [np.dot(vertex, norm_axis) for vertex in hex2_vertices]

        # Check for overlap
        min1, max1 = min(proj1), max(proj1)
        min2, max2 = min(proj2), max(proj2)

        # If no overlap, then they don't intersect
        if max1 < min2 or max2 < min1:
            return False

    return True


@jit(nopython=True)
def calculate_min_enclosing_hexagon_jit(inner_hex_data, scale_factor=1.0):
    """Calculate the minimum side length of the hexagon needed to contain all inner hexagons."""
    # Get all vertices of all inner hexagons
    all_vertices = np.empty((len(inner_hex_data) * 6, 2))
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = hexagon_vertices_jit(center_x, center_y, angle)
        all_vertices[i*6:(i+1)*6] = vertices

    # Find bounding circle radius
    centroid_x = np.mean(all_vertices[:, 0])
    centroid_y = np.mean(all_vertices[:, 1])
    distances = np.sqrt(np.sum((all_vertices - np.array([centroid_x, centroid_y]))**2, axis=1))
    max_distance = np.max(distances)

    # For a regular hexagon, side length = max_distance * sqrt(3)/2
    # But we also want to ensure we have enough margin for the hexagon itself
    # Scale factor is used to ensure we have room for proper containment
    side_length = max_distance * 2 / np.sqrt(3) * scale_factor

    return side_length, centroid_x, centroid_y


def evaluate_solution(solution_array):
    """Evaluate how good a solution is by returning negative inverse side length
    (negative because we maximize the inverse, but optimization minimizes)."""
    # Reshape solution array into 12 hexagons with (x, y, angle) each
    inner_hex_data = solution_array.reshape(-1, 3)

    # Calculate the minimum enclosing hexagon
    min_side_length, _ = calculate_min_enclosing_hexagon(inner_hex_data)

    # Check all constraints
    num_hex = len(inner_hex_data)
    penalty = 0.0

    # Check containment - this is a simplified check
    # For now we just make sure the hexagons aren't too far out
    for i in range(num_hex):
        center_x, center_y, angle = inner_hex_data[i]
        # Check if the hexagon is contained within a reasonable boundary
        vertices = hexagon_vertices(center_x, center_y, angle)
        max_dist = np.max(np.sqrt(np.sum((vertices - np.array([0, 0]))**2, axis=1)))
        if max_dist > min_side_length:
            penalty += 1000 * (max_dist - min_side_length)

    # Check overlaps - compute pairwise distances and penalize overlaps
    for i in range(num_hex):
        for j in range(i+1, num_hex):
            vertices1 = hexagon_vertices(*inner_hex_data[i])
            vertices2 = hexagon_vertices(*inner_hex_data[j])

            if check_overlap(vertices1, vertices2):
                penalty += 10000  # Heavy penalty for overlap

    # Return negative inverse side length plus penalty
    # This makes our optimization minimize the negative inverse, which maximizes the inverse
    objective_value = -1.0 / min_side_length + penalty

    return objective_value


def hexagon_packing_12():
    """
    Constructs an optimized packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    # Number of variables: 12 hexagons * 3 parameters each = 36
    num_variables = 12 * 3

    # Define bounds for each parameter: x, y in [-5, 5], angle in [0, 360)
    bounds = []
    for i in range(12):
        bounds.extend([(-5, 5), (-5, 5), (0, 360)])

    # Use differential evolution to find the optimal solution
    result = differential_evolution(
        evaluate_solution,
        bounds,
        maxiter=100,
        popsize=15,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        disp=True
    )

    print(f"Optimization completed in {time.time() - start_time:.2f} seconds")
    print(f"Best objective value: {result.fun}")

    # Extract the best solution
    best_solution = result.x
    inner_hex_data = best_solution.reshape(-1, 3)

    # Calculate the resulting outer hexagon side length
    min_side_length, centroid = calculate_min_enclosing_hexagon(inner_hex_data, 1.05)

    # Center the outer hexagon at the centroid of inner hexagons
    outer_hex_data = np.array([centroid[0], centroid[1], 0])

    return inner_hex_data, outer_hex_data, min_side_length


# EVOLVE-BLOCK-END