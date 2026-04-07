# EVOLVE-BLOCK-START
import numpy as np
from numba import njit
import math


@njit
def generate_hexagon_vertices_numba(center_x: float, center_y: float, angle_deg: float) -> np.ndarray:
    """Generate vertices of a regular hexagon given center and rotation using numba JIT compilation."""
    # Precomputed constants
    sqrt3_over_2 = math.sqrt(3) / 2
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Vertices of a unit hexagon centered at origin
    base_vertices = np.array([
        [1.0, 0.0],
        [0.5, sqrt3_over_2],
        [-0.5, sqrt3_over_2],
        [-1.0, 0.0],
        [-0.5, -sqrt3_over_2],
        [0.5, -sqrt3_over_2]
    ])

    # Rotate and translate
    rotated_vertices = np.empty_like(base_vertices)
    for i in range(6):
        x, y = base_vertices[i]
        rotated_vertices[i, 0] = x * cos_a - y * sin_a + center_x
        rotated_vertices[i, 1] = x * sin_a + y * cos_a + center_y

    return rotated_vertices


@njit
def point_in_polygon_numba(point: np.ndarray, polygon_vertices: np.ndarray) -> bool:
    """Check if a point is inside a polygon using ray casting algorithm."""
    x, y = point
    n = len(polygon_vertices)
    inside = False

    p1x, p1y = polygon_vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_vertices[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


@njit
def distance_point_to_segment(point: np.ndarray, segment_start: np.ndarray, segment_end: np.ndarray) -> float:
    """Calculate the distance from a point to a line segment."""
    px, py = point
    x1, y1 = segment_start
    x2, y2 = segment_end

    # Vector from segment_start to segment_end
    dx = x2 - x1
    dy = y2 - y1

    # Length squared of segment
    length_sq = dx*dx + dy*dy

    if length_sq == 0.0:
        # Segment is actually a point
        return math.sqrt((px - x1)**2 + (py - y1)**2)

    # Project point onto line defined by segment
    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    t = max(0.0, min(1.0, t))  # Clamp to [0, 1] to stay within segment

    # Closest point on segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    # Distance from point to closest point
    return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)


@njit
def hexagon_overlap_area_numba(hex1_vertices: np.ndarray, hex2_vertices: np.ndarray) -> float:
    """Calculate approximate overlap area between two hexagons."""
    # Use a simple sampling approach for overlap estimation
    # This is a rough approximation but sufficient for our purposes
    overlap_samples = 0
    total_samples = 100

    for i in range(total_samples):
        # Generate random point in first hexagon
        # We'll sample from a bounding box around the first hexagon
        min_x, max_x = hex1_vertices[:, 0].min(), hex1_vertices[:, 0].max()
        min_y, max_y = hex1_vertices[:, 1].min(), hex1_vertices[:, 1].max()

        px = min_x + (max_x - min_x) * np.random.random()
        py = min_y + (max_y - min_y) * np.random.random()

        point = np.array([px, py])
        if point_in_polygon_numba(point, hex1_vertices):
            if point_in_polygon_numba(point, hex2_vertices):
                overlap_samples += 1

    # Estimate overlap area proportionally
    hex1_area = 6 * 0.5 * 1.0 * 1.0 * math.sqrt(3) / 2
    return overlap_samples / total_samples * hex1_area


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    n = 12
    # Simple grid arrangement of inner hexagons
    inner_hex_data = np.array(
        [
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
            [3.75, -2.17, 0],  # far bottom-right,
            [0, -4, 0],  # far bottom-center
        ]
    )

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 4.0  # reduced from 8 for better packing

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END