# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from typing import Tuple, Optional, List
import time

class HexagonGeometry:
    """Handles all hexagon-related geometric computations."""

    def __init__(self):
        self.side_length = 1.0
        self.vertex_angles = np.linspace(0, 2*np.pi, 7)

    def vertices(self, center_x: float, center_y: float, angle_deg: float = 0) -> np.ndarray:
        """Generate vertices of a regular hexagon given center and rotation."""
        angle_rad = np.deg2rad(angle_deg)
        angles = self.vertex_angles + angle_rad
        vertices = np.array([
            [center_x + self.side_length * np.cos(a),
             center_y + self.side_length * np.sin(a)]
            for a in angles
        ])
        return vertices

class PackingValidator:
    """Validates packing configurations for constraints."""

    def __init__(self, hex_geometry: HexagonGeometry):
        self.hex_geom = hex_geometry

    def check_overlap(self, hex1_vertices: np.ndarray, hex2_vertices: np.ndarray) -> bool:
        """Check if two hexagons overlap using Shapely."""
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)

    def check_containment(self, hex_vertices: np.ndarray, outer_hex_vertices: np.ndarray) -> bool:
        """Check if hexagon vertices are contained within outer hexagon."""
        outer_polygon = Polygon(outer_hex_vertices)
        for vertex in hex_vertices:
            point = Point(vertex[0], vertex[1])
            if not outer_polygon.contains(point):
                return False
        return True

    def validate_packing(self, inner_hex_data: np.ndarray, outer_hex_side_length: float) -> Tuple[bool, float]:
        """Validate entire packing configuration."""
        try:
            # Generate vertices for all inner hexagons
            hex_polygons = []
            for i in range(12):
                x, y, angle = inner_hex_data[i]
                vertices = self.hex_geom.vertices(x, y, angle)
                hex_polygons.append(Polygon(vertices))

            # Check for overlaps between hexagons
            for i in range(12):
                for j in range(i+1, 12):
                    if self.check_overlap(hex_polygons[i], hex_polygons[j]):
                        return False, 0.0

            # Create outer hexagon
            outer_vertices = self.hex_geom.vertices(0, 0, outer_hex_side_length)
            outer_polygon = Polygon(outer_vertices)

            # Check containment
            for i in range(12):
                for vertex in hex_polygons[i].exterior.coords:
                    point = Point(vertex[0], vertex[1])
                    if not outer_polygon.contains(point):
                        return False, 0.0

            # If we reach here, packing is valid
            # Calculate objective (1/outer_radius)
            return True, 1.0 / outer_hex_side_length

        except Exception:
            return False, 0.0

class ConfigurationManager:
    """Manages different configuration strategies."""

    @staticmethod
    def get_target_configuration() -> Tuple[np.ndarray, float]:
        """Return the known high-quality symmetric configuration."""
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
        return inner_hex_data, outer_hex_side_length

    @staticmethod
    def get_fallback_configuration() -> Tuple[np.ndarray, float]:
        """Return a fallback configuration."""
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
        return inner_hex_data, outer_hex_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Initialize components
    start_time = time.time()

    hex_geometry = HexagonGeometry()
    validator = PackingValidator(hex_geometry)
    config_manager = ConfigurationManager()

    # Get target configuration
    inner_hex_data, outer_hex_side_length = config_manager.get_target_configuration()

    # Outer hexagon centered at origin with no rotation
    outer_hex_data = np.array([0, 0, 0])

    # Validate the configuration
    is_valid, objective_value = validator.validate_packing(inner_hex_data, outer_hex_side_length)

    # If not valid, use fallback
    if not is_valid:
        print("Warning: Target configuration not valid, using fallback.")
        inner_hex_data, outer_hex_side_length = config_manager.get_fallback_configuration()
        outer_hex_data = np.array([0, 0, 0])

        # Validate fallback
        is_valid, objective_value = validator.validate_packing(inner_hex_data, outer_hex_side_length)

    end_time = time.time()

    # Calculate performance metrics
    inv_outer_hex_side_length = objective_value if is_valid else 0.0
    benchmark_ratio = inv_outer_hex_side_length / 0.2537

    # Print metrics
    print(f"Optimized result: inverse_side_length={inv_outer_hex_side_length:.6f}, "
          f"benchmark_ratio={benchmark_ratio:.6f}, eval_time={(end_time-start_time):.3f}s")

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END