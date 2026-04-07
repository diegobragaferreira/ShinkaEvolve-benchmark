# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
from scipy.spatial import cKDTree
import time
from functools import lru_cache

class Hexagon:
    """Represents a regular hexagon with position and rotation."""

    def __init__(self, center_x, center_y, angle_degrees, side_length=1):
        self.center_x = center_x
        self.center_y = center_y
        self.angle_degrees = angle_degrees
        self.side_length = side_length
        self._vertices = None

    @property
    def vertices(self):
        """Cached generation of hexagon vertices."""
        if self._vertices is None:
            self._vertices = self._generate_vertices()
        return self._vertices

    def _generate_vertices(self):
        """Generate vertices of a regular hexagon."""
        angle_rad = np.radians(self.angle_degrees)
        angles = np.linspace(0, 2*np.pi, 7) + angle_rad  # 6 sides + closing vertex
        vertices = []
        for angle in angles:
            x = self.center_x + self.side_length * np.cos(angle)
            y = self.center_y + self.side_length * np.sin(angle)
            vertices.append((x, y))
        return np.array(vertices)

    def get_bounding_circle(self):
        """Get the bounding circle (center and radius) of the hexagon."""
        cx, cy = self.center_x, self.center_y
        # Distance from center to corner of unit hexagon is 1
        r = self.side_length
        return cx, cy, r

class HexagonPacker:
    """Handles the core hexagon packing logic."""

    def __init__(self, num_inner_hexagons=12):
        self.num_inner_hexagons = num_inner_hexagons
        self.hexagons = []
        self.outer_hexagon = None

    def create_symmetric_layout(self, params):
        """Create hexagon layout based on symmetric parameters."""
        # params[0]: middle ring radius
        # params[1]: outer ring radius
        # params[2]: middle ring angle offset
        # params[3]: outer ring angle offset
        # params[4]: outer hexagon angle (rotation)
        # params[5]: outer hexagon center x
        # params[6]: outer hexagon center y
        # params[7]: center hexagon rotation
        # params[8]: middle ring rotation

        # Clear existing hexagons
        self.hexagons.clear()

        middle_radius = params[0]
        outer_radius = params[1]
        middle_angle_offset = params[2]
        outer_angle_offset = params[3]

        # Layer 1: Center (1 hexagon)
        center_hex = Hexagon(0.0, 0.0, params[7])
        self.hexagons.append(center_hex)

        # Layer 2: Middle ring (6 hexagons)
        for i in range(6):
            angle = (i * 60 + middle_angle_offset) % 360
            rad = middle_radius
            x = rad * np.cos(np.radians(angle))
            y = rad * np.sin(np.radians(angle))
            hexagon = Hexagon(x, y, params[8])
            self.hexagons.append(hexagon)

        # Layer 3: Outer ring (5 hexagons)
        for i in range(5):
            angle = (i * 72 + outer_angle_offset) % 360
            rad = outer_radius
            x = rad * np.cos(np.radians(angle))
            y = rad * np.sin(np.radians(angle))
            hexagon = Hexagon(x, y, 0.0)
            self.hexagons.append(hexagon)

        return self.hexagons

    def calculate_outer_radius(self):
        """Calculate required outer hexagon radius based on inner hexagons."""
        max_dist = 0
        for hexagon in self.hexagons:
            for vertex in hexagon.vertices:
                dist = np.sqrt(vertex[0]**2 + vertex[1]**2)
                max_dist = max(max_dist, dist)
        return max_dist * 1.02  # Add 2% buffer for numerical stability

    def create_outer_hexagon(self, center_x, center_y, angle, radius):
        """Create the outer hexagon."""
        self.outer_hexagon = Hexagon(center_x, center_y, angle, radius)
        return self.outer_hexagon

    def check_containment_all(self):
        """Check if all inner hexagons are contained within outer hexagon."""
        if not self.outer_hexagon:
            return False
        outer_vertices = self.outer_hexagon.vertices

        for hexagon in self.hexagons:
            inner_polygon = Polygon(hexagon.vertices)
            outer_polygon = Polygon(outer_vertices)
            if not outer_polygon.contains(inner_polygon):
                return False
        return True

    @staticmethod
    @lru_cache(maxsize=1000)
    def _cached_overlap_check(hex1_vertices_tuple, hex2_vertices_tuple):
        """Cached overlap check between two hexagons."""
        polygon1 = Polygon(hex1_vertices_tuple)
        polygon2 = Polygon(hex2_vertices_tuple)
        return polygon1.intersects(polygon2)

    def fast_overlap_check(self, hex1, hex2):
        """Fast overlap check using bounding circles for early rejection."""
        cx1, cy1, r1 = hex1.get_bounding_circle()
        cx2, cy2, r2 = hex2.get_bounding_circle()

        # Fast circle overlap test
        dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        if dist >= (r1 + r2):
            return False  # Definitely don't overlap

        # If close enough, do precise overlap check
        return self._cached_overlap_check(
            tuple(tuple(v) for v in hex1.vertices),
            tuple(tuple(v) for v in hex2.vertices)
        )

    def check_overlaps(self):
        """Check for overlaps between hexagons with optimized spatial queries."""
        if len(self.hexagons) < 2:
            return False

        # Build spatial index for efficient neighbor querying
        hex_centers = np.array([[h.center_x, h.center_y] for h in self.hexagons])
        tree = cKDTree(hex_centers)

        # Find neighbors within a reasonable distance (2x hexagon diameter)
        pairs_to_check = tree.query_pairs(r=3.0, p=np.inf)

        # Check overlaps using spatial acceleration
        for i, j in pairs_to_check:
            # Skip center with itself
            if i == 0 and j == 0:
                continue

            # Check overlap between hexagons i and j
            if self.fast_overlap_check(self.hexagons[i], self.hexagons[j]):
                return True  # Found overlap

        # Additional specific overlap checks for critical pairs
        # Center with all others
        for i in range(1, len(self.hexagons)):  # center with all other hexagons
            if self.fast_overlap_check(self.hexagons[0], self.hexagons[i]):
                return True

        # Middle ring vs outer ring
        for i in range(1, 7):  # middle ring
            for j in range(7, 12):  # outer ring
                if self.fast_overlap_check(self.hexagons[i], self.hexagons[j]):
                    return True

        # Middle ring self-intersection
        for i in range(1, 7):
            for j in range(i+1, 7):
                if self.fast_overlap_check(self.hexagons[i], self.hexagons[j]):
                    return True

        # Outer ring self-intersection
        for i in range(7, 12):
            for j in range(i+1, 12):
                if self.fast_overlap_check(self.hexagons[i], self.hexagons[j]):
                    return True

        return False

class OptimizationEngine:
    """Handles the optimization process with configurable parameters."""

    def __init__(self, packer):
        self.packer = packer
        self.bounds = [
            (1.0, 4.0),     # middle ring radius
            (2.0, 6.0),     # outer ring radius
            (-180, 180),    # middle ring angle offset
            (-180, 180),    # outer ring angle offset
            (-180, 180),    # outer hex angle
            (-5.0, 5.0),    # outer center x
            (-5.0, 5.0),    # outer center y
            (-180, 180),    # center rotation
            (-180, 180)     # middle rotation
        ]
        self.maxiter = 30
        self.popsize = 20

    def evaluate(self, params):
        """Evaluate the configuration and return fitness score."""
        # Update the packer with new parameters
        self.packer.create_symmetric_layout(params)

        # Calculate outer hexagon parameters
        outer_radius = self.packer.calculate_outer_radius()
        outer_center_x, outer_center_y, outer_angle = params[5:8]

        # Create outer hexagon
        self.packer.create_outer_hexagon(outer_center_x, outer_center_y, outer_angle, outer_radius)

        # Check constraints
        total_penalty = 0

        # Check containment
        if not self.packer.check_containment_all():
            total_penalty += 10000

        # Check overlaps
        if self.packer.check_overlaps():
            total_penalty += 10000

        # Return negative inverse of outer radius plus penalties
        return -(1.0 / (outer_radius + total_penalty + 1e-8))

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    try:
        # Initialize components
        packer = HexagonPacker(12)
        optimizer = OptimizationEngine(packer)

        # Initial smart symmetric configuration
        initial_params = np.array([
            2.15,     # middle ring radius
            3.6,      # outer ring radius
            30.0,     # middle ring angle offset
            0.0,      # outer ring angle offset
            0.0,      # outer hexagon angle
            0.0,      # outer hexagon center x
            0.0,      # outer hexagon center y
            0.0,      # center hexagon rotation
            0.0       # middle ring rotation
        ])

        # Run optimization
        result = differential_evolution(
            optimizer.evaluate,
            optimizer.bounds,
            maxiter=optimizer.maxiter,
            popsize=optimizer.popsize,
            seed=42,
            disp=False,
            atol=1e-6,
            ftol=1e-6
        )

        # Extract optimized parameters
        optimized_params = result.x

        # Recreate final configuration
        packer.create_symmetric_layout(optimized_params)

        # Calculate exact outer radius needed
        outer_radius_final = packer.calculate_outer_radius()
        outer_center_x, outer_center_y, outer_angle = optimized_params[5:8]
        packer.create_outer_hexagon(outer_center_x, outer_center_y, outer_angle, outer_radius_final)

        # Final validation
        valid = True
        if not packer.check_containment_all() or packer.check_overlaps():
            valid = False

        if not valid:
            # Fallback to previous working configuration
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
            outer_hex_side_length = 8
            return inner_hex_data, outer_hex_data, outer_hex_side_length

        # Format output with optimized positions
        inner_hex_data = np.zeros((12, 3))
        for i, hexagon in enumerate(packer.hexagons):
            inner_hex_data[i] = [hexagon.center_x, hexagon.center_y, hexagon.angle_degrees]

        outer_hex_data = np.array([outer_center_x, outer_center_y, outer_angle])
        outer_hex_side_length = outer_radius_final

        return inner_hex_data, outer_hex_data, outer_hex_side_length

    except Exception as e:
        # Fallback if optimization fails
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
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END