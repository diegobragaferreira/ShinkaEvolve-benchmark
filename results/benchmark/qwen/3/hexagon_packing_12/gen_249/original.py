# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
from scipy.spatial.distance import cdist
import time


class HexagonPacker:
    """Main class managing hexagon packing operations"""

    def __init__(self):
        self.hex_side_length = 1.0
        self.outer_center = np.array([0.0, 0.0])

    def hexagon_vertices(self, center_x, center_y, size=1, angle_deg=0):
        """Generate vertices of a regular hexagon given center, size, and rotation."""
        angle_rad = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            angle = angle_rad + i * np.pi / 3
            x = center_x + size * np.cos(angle)
            y = center_y + size * np.sin(angle)
            vertices.append((x, y))
        return np.array(vertices)

    def get_outer_hexagon(self, outer_radius):
        """Get vertices of the outer hexagon with given radius."""
        return self.hexagon_vertices(self.outer_center[0], self.outer_center[1], outer_radius, 0)

    def validate_containment(self, hex_vertices, outer_radius):
        """Check if all vertices of a hexagon are inside the outer hexagon."""
        outer_vertices = self.get_outer_hexagon(outer_radius)
        outer_polygon = Polygon(outer_vertices)

        for vertex in hex_vertices:
            point = Point(vertex[0], vertex[1])
            if not outer_polygon.contains(point):
                return False
        return True

    def validate_overlap(self, hex1_vertices, hex2_vertices):
        """Check if two hexagons overlap using Shapely."""
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)

    def calculate_max_distance_from_center(self, hex_data):
        """Calculate maximum distance from center to any hexagon vertex."""
        max_dist = 0
        for i in range(len(hex_data)):
            cx, cy, _ = hex_data[i]
            # Calculate distance to center plus hexagon radius
            dist = np.sqrt(cx**2 + cy**2) + self.hex_side_length
            max_dist = max(max_dist, dist)
        return max_dist

    def evaluate_configuration(self, hex_data, outer_radius):
        """Evaluate current configuration: returns (validity, inv_radius)."""
        # Check for overlaps
        for i in range(len(hex_data)):
            hex1_vertices = self.hexagon_vertices(hex_data[i][0], hex_data[i][1],
                                                self.hex_side_length, hex_data[i][2])
            for j in range(i+1, len(hex_data)):
                hex2_vertices = self.hexagon_vertices(hex_data[j][0], hex_data[j][1],
                                                    self.hex_side_length, hex_data[j][2])
                if self.validate_overlap(hex1_vertices, hex2_vertices):
                    return False, 0

        # Check containment
        for i in range(len(hex_data)):
            hex_vertices = self.hexagon_vertices(hex_data[i][0], hex_data[i][1],
                                               self.hex_side_length, hex_data[i][2])
            if not self.validate_containment(hex_vertices, outer_radius):
                return False, 0

        # Return inverse of outer radius
        return True, 1.0 / outer_radius


def generate_multiple_initial_configurations():
    """Generate multiple symmetric initial configurations for better exploration"""
    configs = []

    # Configuration 1: Highly symmetric arrangement
    config1 = [
        [0, 0, 0],           # center
        [-1.732, 0, 0],      # left
        [1.732, 0, 0],       # right
        [-0.866, 1.5, 0],    # top-left
        [0.866, 1.5, 0],     # top-right
        [-0.866, -1.5, 0],   # bottom-left
        [0.866, -1.5, 0],    # bottom-right
        [-2.598, 1.5, 0],    # far top-left
        [2.598, 1.5, 0],     # far top-right
        [-2.598, -1.5, 0],   # far bottom-left
        [2.598, -1.5, 0],    # far bottom-right
        [0, -3, 0]           # far bottom-center
    ]

    # Configuration 2: Rotated version
    config2 = [
        [0, 0, 30],          # center
        [-1.732, 0, 30],     # left
        [1.732, 0, 30],      # right
        [-0.866, 1.5, 30],   # top-left
        [0.866, 1.5, 30],    # top-right
        [-0.866, -1.5, 30],  # bottom-left
        [0.866, -1.5, 30],   # bottom-right
        [-2.598, 1.5, 30],   # far top-left
        [2.598, 1.5, 30],    # far top-right
        [-2.598, -1.5, 30],  # far bottom-left
        [2.598, -1.5, 30],   # far bottom-right
        [0, -3, 30]          # far bottom-center
    ]

    # Configuration 3: Another symmetric arrangement
    config3 = [
        [0, 0, 0],           # center
        [-2.0, 0, 0],        # left
        [2.0, 0, 0],         # right
        [-1.0, 1.732, 0],    # top-left
        [1.0, 1.732, 0],     # top-right
        [-1.0, -1.732, 0],   # bottom-left
        [1.0, -1.732, 0],    # bottom-right
        [-3.0, 1.732, 0],    # far top-left
        [3.0, 1.732, 0],     # far top-right
        [-3.0, -1.732, 0],   # far bottom-left
        [3.0, -1.732, 0],    # far bottom-right
        [0, -3.464, 0]       # far bottom-center
    ]

    configs.append(np.array(config1))
    configs.append(np.array(config2))
    configs.append(np.array(config3))

    return configs


def optimize_positions_and_orientations(initial_config):
    """Optimize both positions and orientations using differential evolution"""
    packer = HexagonPacker()

    def objective(params):
        # Reshape parameters back to hex_data format (positions + orientations)
        hex_data = initial_config.copy()
        idx = 0
        for i in range(len(hex_data)):
            hex_data[i][0] = params[idx]
            hex_data[i][1] = params[idx + 1]
            hex_data[i][2] = params[idx + 2]  # orientation
            idx += 3

        # Calculate outer radius for this configuration
        outer_radius = packer.calculate_max_distance_from_center(hex_data)

        # Evaluate configuration
        validity, inv_radius = packer.evaluate_configuration(hex_data, outer_radius)

        if not validity:
            return 1e10  # Large penalty for invalid configurations
        return -inv_radius  # Negative because we maximize

    # Flatten initial configuration - now including orientations
    initial_params = []
    for i in range(len(initial_config)):
        initial_params.extend([initial_config[i][0], initial_config[i][1], initial_config[i][2]])

    # Bounds for positions: [-5, 5] for each coordinate, orientations: [0, 360)
    bounds = [(-5, 5), (-5, 5), (0, 360)] * len(initial_config)

    # Use differential evolution for global optimization
    result = differential_evolution(
        objective,
        bounds,
        maxiter=100,
        popsize=15,
        tol=1e-6,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42
    )

    # Reconstruct optimized configuration
    optimized_config = initial_config.copy()
    idx = 0
    for i in range(len(optimized_config)):
        optimized_config[i][0] = result.x[idx]
        optimized_config[i][1] = result.x[idx + 1]
        optimized_config[i][2] = result.x[idx + 2]
        idx += 3

    return optimized_config


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Generate multiple initial configurations
    initial_configs = generate_multiple_initial_configurations()

    best_config = None
    best_inv_radius = 0
    best_outer_radius = float('inf')

    # Try each initial configuration and optimize
    for initial_config in initial_configs:
        # Optimize positions and orientations
        optimized_config = optimize_positions_and_orientations(initial_config)

        # Final validation and calculation
        packer = HexagonPacker()
        outer_radius = packer.calculate_max_distance_from_center(optimized_config)

        # Verify final configuration
        validity, inv_radius = packer.evaluate_configuration(optimized_config, outer_radius)

        if validity and inv_radius > best_inv_radius:
            best_inv_radius = inv_radius
            best_config = optimized_config.copy()
            best_outer_radius = outer_radius

    # If no valid solution found, fall back to a known working configuration
    if best_config is None:
        # Use the simple arrangement that was previously working
        fallback_config = np.array([
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
        ])
        best_config = fallback_config
        best_outer_radius = 8.0
        best_inv_radius = 1.0 / best_outer_radius

    # Prepare return values
    inner_hex_data = np.array(best_config)
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = best_outer_radius * 2  # approximate side length

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END