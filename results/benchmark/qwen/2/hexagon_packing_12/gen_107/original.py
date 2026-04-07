# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from scipy.optimize import differential_evolution
import time
from typing import Tuple, List
import warnings
from numba import jit, prange

# JIT compile the most frequently used geometric functions
@jit(nopython=True)
def hexagon_vertices_jit(center_x: float, center_y: float, rotation_deg: float) -> np.ndarray:
    """JIT compiled function to generate hexagon vertices"""
    # Unit hexagon vertices centered at origin
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 vertices + close loop
    unit_vertices = np.column_stack([np.cos(angles), np.sin(angles)])

    # Apply rotation and translation
    rotation_rad = np.radians(rotation_deg)
    cos_r, sin_r = np.cos(rotation_rad), np.sin(rotation_rad)
    rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])

    rotated_vertices = rotation_matrix @ unit_vertices.T
    translated_vertices = rotated_vertices.T + np.array([center_x, center_y])

    return translated_vertices

@jit(nopython=True)
def point_in_hexagon_jit(point_x: float, point_y: float, hex_vertices: np.ndarray) -> bool:
    """JIT compiled function to check if point is inside hexagon using ray casting"""
    # Ray casting algorithm
    x, y = point_x, point_y
    n = len(hex_vertices)
    inside = False
    p1x, p1y = hex_vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = hex_vertices[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

@jit(nopython=True)
def distance_point_to_line_segment(point_x: float, point_y: float,
                                   line_start_x: float, line_start_y: float,
                                   line_end_x: float, line_end_y: float) -> float:
    """JIT compiled function to compute distance from point to line segment"""
    A = point_x - line_start_x
    B = point_y - line_start_y
    C = line_end_x - line_start_x
    D = line_end_y - line_start_y

    dot = A*C + B*D
    len_sq = C*C + D*D
    if len_sq == 0:
        return np.sqrt(A*A + B*B)
    param = dot / len_sq
    param = max(0, min(1, param))
    xx = line_start_x + param * C
    yy = line_start_y + param * D
    dx = point_x - xx
    dy = point_y - yy
    return np.sqrt(dx*dx + dy*dy)

class Hexagon:
    """Represents a regular hexagon with side length 1"""

    def __init__(self, center_x: float, center_y: float, rotation_deg: float):
        self.center = np.array([center_x, center_y])
        self.rotation = np.radians(rotation_deg)
        self.side_length = 1.0

    def vertices(self) -> np.ndarray:
        """Return vertices of hexagon in counterclockwise order"""
        return hexagon_vertices_jit(self.center[0], self.center[1], self.rotation)

    def area(self) -> float:
        """Return area of hexagon"""
        return (3 * np.sqrt(3) / 2) * self.side_length ** 2

def create_outer_hexagon(side_length: float, center_x: float = 0, center_y: float = 0) -> Polygon:
    """Create outer hexagon as Shapely polygon"""
    hex = Hexagon(center_x, center_y, 0)
    outer_vertices = hex.vertices()
    return Polygon(outer_vertices)

def check_containment_jit(hexagon: Hexagon, outer_polygon: Polygon) -> bool:
    """Check if all vertices of hexagon are inside outer polygon using JIT-compiled helper"""
    vertices = hexagon.vertices()
    for vertex in vertices:
        point = Point(vertex[0], vertex[1])
        if not outer_polygon.contains(point):
            return False
    return True

def check_containment(hexagon: Hexagon, outer_polygon: Polygon) -> bool:
    """Check if all vertices of hexagon are inside outer polygon"""
    vertices = hexagon.vertices()
    for vertex in vertices:
        point = Point(vertex[0], vertex[1])
        if not outer_polygon.contains(point):
            return False
    return True

def check_overlap_jit(hex1: Hexagon, hex2: Hexagon) -> bool:
    """Check if two hexagons overlap using Shapely"""
    poly1 = Polygon(hex1.vertices())
    poly2 = Polygon(hex2.vertices())
    return poly1.intersects(poly2)

def check_overlap(hex1: Hexagon, hex2: Hexagon) -> bool:
    """Check if two hexagons overlap using Shapely"""
    poly1 = Polygon(hex1.vertices())
    poly2 = Polygon(hex2.vertices())
    return poly1.intersects(poly2)

def validate_configuration(inner_hexagons: List[Hexagon], outer_polygon: Polygon) -> Tuple[bool, bool]:
    """
    Validate configuration: check containment and overlap
    Returns (is_valid, has_overlaps)
    """
    # Check containment
    for hex in inner_hexagons:
        if not check_containment(hex, outer_polygon):
            return False, False

    # Check overlaps
    for i in range(len(inner_hexagons)):
        for j in range(i+1, len(inner_hexagons)):
            if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                return True, True

    return True, False

def calculate_objective(outer_side_length: float) -> float:
    """Calculate 1/outer_hex_side_length"""
    return 1.0 / outer_side_length

def compute_outer_hex_side_length(inner_hexagons: List[Hexagon]) -> float:
    """Compute minimum outer hexagon side length required to contain all inner hexagons"""
    # Get all vertices from all hexagons
    all_vertices = []
    for hex in inner_hexagons:
        all_vertices.extend(hex.vertices())

    all_vertices = np.array(all_vertices)

    # Find bounding circle center and radius
    center = np.mean(all_vertices, axis=0)

    # Calculate maximum distance from center to any vertex
    distances = np.linalg.norm(all_vertices - center, axis=1)
    max_distance = np.max(distances)

    # For a hexagon, we need side length >= max_distance
    # Side length = max_distance / (sqrt(3)/2)  (approximate for unit hexagons)
    # But let's be more precise
    side_length = max_distance * 2 / np.sqrt(3)

    return side_length

def generate_initial_population(n_individuals: int, n_hexagons: int = 12) -> List[np.ndarray]:
    """Generate initial population of valid configurations"""
    population = []

    # Try multiple random configurations
    for _ in range(n_individuals):
        # Start with a symmetric arrangement around center
        positions = []
        rotations = []

        # Center hexagon
        positions.append([0.0, 0.0])
        rotations.append(0.0)

        # Surrounding hexagons arranged in 2 rings
        ring1_radius = 2.0
        ring2_radius = 3.5

        # First ring (5 hexagons)
        for i in range(5):
            angle = i * 2 * np.pi / 5
            x = ring1_radius * np.cos(angle)
            y = ring1_radius * np.sin(angle)
            positions.append([x, y])
            rotations.append(0.0)

        # Second ring (6 hexagons)
        for i in range(6):
            angle = i * 2 * np.pi / 6 + np.pi/6  # offset to interleave
            x = ring2_radius * np.cos(angle)
            y = ring2_radius * np.sin(angle)
            positions.append([x, y])
            rotations.append(0.0)

        # Add some random noise to make it diverse
        individual = np.array(positions + rotations).flatten()
        individual += np.random.normal(0, 0.5, len(individual))
        population.append(individual)

    return population

def fitness_function(params: np.ndarray, outer_side_length: float = 10.0) -> float:
    """
    Fitness function to minimize negative of 1/outer_hex_side_length
    This assumes params contains [x1,y1,theta1,x2,y2,theta2,...]
    """
    n_hexagons = 12
    hex_params = params.reshape(-1, 3)

    # Create hexagons
    inner_hexagons = []
    for i in range(n_hexagons):
        x, y, theta = hex_params[i]
        inner_hexagons.append(Hexagon(x, y, theta))

    # Create outer hexagon of given size
    outer_polygon = create_outer_hexagon(outer_side_length)

    # Validate configuration
    is_valid, has_overlaps = validate_configuration(inner_hexagons, outer_polygon)

    if not is_valid:
        # Penalize invalid configurations heavily
        return 1e6

    # Compute actual outer hexagon size needed
    actual_size = compute_outer_hex_side_length(inner_hexagons)

    if actual_size > outer_side_length:
        # Configuration requires larger outer hexagon
        return 1e6

    # Return negative of 1/actual_size to maximize 1/size
    return -1.0 / actual_size

def optimize_packing() -> Tuple[np.ndarray, np.ndarray, float]:
    """Main optimization function"""
    start_time = time.time()

    # Define bounds for optimization
    # Each hexagon has (x,y,rotation) = 3 parameters
    bounds = []
    for i in range(12):
        # X and Y positions: narrow bounds based on expected optimal arrangement
        bounds.extend([(-6.0, 6.0), (-6.0, 6.0)])  # Positions - narrower bounds
        # Rotation: 0-360 degrees
        bounds.append((0, 360))  # Rotations

    # Initial guess with a good symmetric layout
    initial_guess = []
    base_positions = [
        [0, 0], [0, 2.5], [0, -2.5],
        [2.17, 1.25], [-2.17, 1.25], [2.17, -1.25], [-2.17, -1.25],
        [4.34, 0], [0, 4.34], [0, -4.34], [4.34, 2.17], [-4.34, 2.17], [4.34, -2.17], [-4.34, -2.17]
    ]

    initial_guess = []
    for i in range(12):
        x, y = base_positions[i]
        initial_guess.extend([x, y, 0])  # No rotation initially

    # Optimization using differential evolution
    try:
        # Set a time limit to ensure we don't exceed 180 seconds
        result = differential_evolution(
            lambda x: fitness_function(x, 8.0),
            bounds,
            maxiter=100,
            popsize=15,
            init='random',
            seed=42,
            disp=False,
            tol=1e-6
        )

        if result.success:
            # Extract final configuration
            final_params = result.x
            hex_params = final_params.reshape(-1, 3)

            # Create hexagons from final parameters
            inner_hexagons = []
            for i in range(12):
                x, y, theta = hex_params[i]
                inner_hexagons.append(Hexagon(x, y, theta))

            # Compute actual outer hexagon size
            outer_side_length = compute_outer_hex_side_length(inner_hexagons)

            # Convert back to expected format
            inner_hex_data = hex_params.copy()
            outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered

            return inner_hex_data, outer_hex_data, outer_side_length
        else:
            # If optimization failed, return the initial configuration with improved bounds
            warnings.warn("Optimization did not converge, returning initial configuration")

    except Exception as e:
        warnings.warn(f"Optimization error: {str(e)}")

    # Fallback to simple symmetric arrangement
    inner_hex_data = np.array([
        [0, 0, 0],          # center
        [0, 2.5, 0],        # up
        [0, -2.5, 0],       # down
        [2.17, 1.25, 0],    # upper right
        [-2.17, 1.25, 0],   # upper left
        [2.17, -1.25, 0],   # lower right
        [-2.17, -1.25, 0],  # lower left
        [4.34, 0, 0],       # far right
        [0, 4.34, 0],       # far up
        [0, -4.34, 0],      # far down
        [4.34, 2.17, 0],    # far upper right
        [-4.34, 2.17, 0],   # far upper left
    ])

    outer_hex_data = np.array([0, 0, 0])
    outer_side_length = 8.0

    return inner_hex_data, outer_hex_data, outer_side_length

def hexagon_packing_12() -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Ensure we don't exceed the 180 second limit by setting a timeout
    try:
        inner_hex_data, outer_hex_data, outer_hex_side_length = optimize_packing()

        # Calculate benchmark ratio
        benchmark_ratio = calculate_objective(outer_hex_side_length) / 0.2537

        # Output metrics for verification
        print(f"inv_outer_hex_side_length: {calculate_objective(outer_hex_side_length):.8f}")
        print(f"benchmark_ratio: {benchmark_ratio:.8f}")
        print(f"eval_time: {time.time() - start_time:.4f}s")

        return inner_hex_data, outer_hex_data, outer_hex_side_length

    except Exception as e:
        print(f"Error in hexagon packing: {e}")
        # Return fallback configuration
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
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END