# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
from scipy.spatial.distance import cdist
from numba import jit, prange
import time
import warnings
from collections import defaultdict

@jit(nopython=True)
def hexagon_vertices_jit(x, y, angle_deg, side_length=1):
    """Fast generation of hexagon vertices using numba"""
    angle_rad = np.radians(angle_deg)
    angles = np.arange(0, 6) * np.pi / 3
    vertices = np.zeros((6, 2))
    for i in range(6):
        vertices[i, 0] = x + side_length * np.cos(angles[i] + angle_rad)
        vertices[i, 1] = y + side_length * np.sin(angles[i] + angle_rad)
    return vertices

@jit(nopython=True)
def point_in_polygon_fast(point, polygon):
    """Fast point-in-polygon test using ray casting"""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

class SpatialHashGrid:
    """Spatial hash grid for efficient neighbor lookups"""

    def __init__(self, cell_size=3.0):
        self.cell_size = cell_size
        self.grid = defaultdict(list)

    def clear(self):
        self.grid.clear()

    def insert(self, hex_id, center_x, center_y):
        """Insert hexagon into spatial hash grid"""
        cell_x = int(center_x // self.cell_size)
        cell_y = int(center_y // self.cell_size)
        key = (cell_x, cell_y)
        self.grid[key].append(hex_id)

    def get_neighbors(self, center_x, center_y):
        """Get all hexagons in the same and neighboring cells"""
        cell_x = int(center_x // self.cell_size)
        cell_y = int(center_y // self.cell_size)

        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (cell_x + dx, cell_y + dy)
                if key in self.grid:
                    neighbors.extend(self.grid[key])
        return neighbors

class HexGridOptimizer:
    """Hexagonal grid-based optimizer for hexagon packing with enhanced symmetry awareness"""

    def __init__(self):
        self.hex_radius = 1.0
        self.hex_apothem = self.hex_radius * np.sqrt(3) / 2
        self.hex_height = 2 * self.hex_apothem
        self.hex_width = 2 * self.hex_radius
        self.grid_spacing = self.hex_width * 1.05  # Slightly larger than hex width to ensure separation

    def create_outer_hexagon(self, side_length: float, center_x: float = 0, center_y: float = 0) -> Polygon:
        """Create outer hexagon as Shapely polygon"""
        vertices = hexagon_vertices_jit(center_x, center_y, 0, side_length)
        return Polygon(vertices)

    def generate_symmetric_base_configs(self):
        """Generate multiple symmetric base configurations for diverse optimization starts"""
        configs = []

        # Config 1: Classic honeycomb pattern with central hexagon
        positions1 = [[0, 0, 0]]  # center
        radius = 2.0
        for i in range(6):
            angle = i * 60
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            positions1.append([x, y, 0])
        # Add second ring
        radius *= 1.2
        for i in range(6):
            angle = i * 60 + 30
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            positions1.append([x, y, 0])
        configs.append(np.array(positions1).flatten())

        # Config 2: Star pattern with radial symmetry
        positions2 = [[0, 0, 0]]
        # Add positions in radial pattern
        for i in range(6):
            angle = i * 60
            x = 2.2 * np.cos(np.radians(angle))
            y = 2.2 * np.sin(np.radians(angle))
            positions2.append([x, y, 0])
        # Add additional strategic positions
        positions2.extend([
            [0, -3.5, 0],
            [3.5, 0, 0],
            [-3.5, 0, 0],
            [0, 3.5, 0]
        ])
        configs.append(np.array(positions2).flatten())

        # Config 3: Modified hexagonal pattern
        positions3 = [[0, 0, 0]]
        radius = 1.8
        for i in range(6):
            angle = i * 60
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            positions3.append([x, y, 0])
        # Add second ring with offset
        radius = 3.2
        for i in range(6):
            angle = i * 60 + 30
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            positions3.append([x, y, 0])
        configs.append(np.array(positions3).flatten())

        return configs

    def compute_outer_hex_side_length(self, hex_positions):
        """Compute minimum outer hexagon side length required to contain all inner hexagons"""
        # Get all vertices from all hexagons
        all_vertices = []
        for pos in hex_positions:
            x, y, angle = pos
            vertices = hexagon_vertices_jit(x, y, angle)
            all_vertices.extend(vertices)

        all_vertices = np.array(all_vertices)

        # Find bounding circle center and radius
        center = np.mean(all_vertices, axis=0)

        # Calculate maximum distance from center to any vertex
        distances = np.linalg.norm(all_vertices - center, axis=1)
        max_distance = np.max(distances)

        # For a hexagon, we need side length >= max_distance * 2 / sqrt(3)
        side_length = max_distance * 2 / np.sqrt(3)

        return side_length

    def check_containment(self, hex_position, outer_polygon):
        """Check if all vertices of hexagon are inside outer polygon"""
        x, y, angle = hex_position
        vertices = hexagon_vertices_jit(x, y, angle)
        for vertex in vertices:
            point = Point(vertex[0], vertex[1])
            if not outer_polygon.contains(point):
                return False
        return True

    def check_overlap_fast(self, pos1, pos2):
        """Fast overlap check using distance approximation"""
        x1, y1, _ = pos1
        x2, y2, _ = pos2
        dx = x1 - x2
        dy = y1 - y2
        distance_sq = dx*dx + dy*dy
        # Two unit hexagons overlap if their centers are less than 2 units apart
        return distance_sq < 4.0

    def calculate_penalty(self, hex_positions, outer_side_length):
        """Calculate penalty based on constraints with adaptive weighting"""
        penalty = 0.0

        # Create outer polygon
        outer_polygon = self.create_outer_hexagon(outer_side_length)

        # Check containment penalties - these are critical
        for pos in hex_positions:
            if not self.check_containment(pos, outer_polygon):
                penalty += 1e9  # High penalty for containment violations

        # Check overlap penalties using spatial hashing for efficiency
        spatial_grid = SpatialHashGrid(cell_size=2.5)

        # Insert all hexagons into spatial hash
        for i, pos in enumerate(hex_positions):
            spatial_grid.insert(i, pos[0], pos[1])

        # Check overlaps using spatial hash
        for i, pos1 in enumerate(hex_positions):
            neighbors = spatial_grid.get_neighbors(pos1[0], pos1[1])
            for j in neighbors:
                if i >= j:  # Avoid duplicate checks and self-checking
                    continue
                pos2 = hex_positions[j]
                if self.check_overlap_fast(pos1, pos2):
                    # Higher penalty for overlaps to ensure they're eliminated
                    penalty += 1e8

        return penalty

    def evaluate_objective(self, params, outer_side_length=10.0):
        """Evaluate the objective function with better penalty handling"""
        # Reshape params to hexagon positions
        positions = params.reshape(-1, 3)

        # Calculate penalty
        penalty = self.calculate_penalty(positions, outer_side_length)

        # Calculate actual outer side length
        actual_side_length = self.compute_outer_hex_side_length(positions)

        # Additional safety check
        if actual_side_length > outer_side_length:
            penalty += 1e9

        # Objective: maximize 1/actual_side_length (minimize -1/actual_side_length)
        # We add penalty to discourage constraint violations
        return -1.0 / actual_side_length + penalty

    def get_initial_solution(self):
        """Generate a good initial solution using multiple strategies"""
        # Generate multiple base configurations
        base_configs = self.generate_symmetric_base_configs()

        # Select the best one based on initial computation
        best_config = None
        best_score = float('inf')

        for config in base_configs:
            # Perturb each configuration for diversity
            perturbed = config.copy()
            # Add small noise to positions (but preserve some structure)
            for i in range(len(config)):
                if i % 3 != 2:  # Not angle parameter
                    perturbed[i] += np.random.normal(0, 0.1)
                else:  # Angle parameter, keep within bounds
                    perturbed[i] = perturbed[i] % 360

            # Check if this configuration gives a reasonable outer side length
            positions = perturbed.reshape(-1, 3)
            side_length = self.compute_outer_hex_side_length(positions)
            if side_length > 1.0 and side_length < 20.0:  # Reasonable bounds
                score = side_length  # Lower side length is better
                if score < best_score:
                    best_score = score
                    best_config = perturbed

        # If no good configuration found, use default
        if best_config is None:
            # Default symmetric configuration
            positions = [
                [0, 0, 0],  # center
                [-2.1, 0, 0],  # left
                [2.1, 0, 0],  # right
                [-1.05, 1.82, 0],  # top-left
                [1.05, 1.82, 0],  # top-right
                [-1.05, -1.82, 0],  # bottom-left
                [1.05, -1.82, 0],  # bottom-right
                [-3.15, 1.82, 0],  # far-top-left
                [3.15, 1.82, 0],  # far-top-right
                [-3.15, -1.82, 0],  # far-bottom-left
                [3.15, -1.82, 0],  # far-bottom-right
                [0, -3.64, 0],  # far-bottom
            ]
            best_config = np.array(positions).flatten()

        return best_config

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    try:
        optimizer = HexGridOptimizer()

        # Get initial solution using enhanced symmetric approaches
        initial_params = optimizer.get_initial_solution()

        # Phase 1: Coarse optimization to find good configuration
        def coarse_objective(params):
            return optimizer.evaluate_objective(params, outer_side_length=8.0)

        # Use L-BFGS-B for coarse optimization
        try:
            result_coarse = minimize(
                coarse_objective,
                initial_params,
                method='L-BFGS-B',
                bounds=[(-10, 10), (-10, 10), (0, 360)] * 12 + [(1.0, 20.0)],
                options={'maxiter': 30, 'ftol': 1e-6, 'gtol': 1e-6},
                tol=1e-6
            )

            if result_coarse.success:
                final_params = result_coarse.x
            else:
                final_params = initial_params
        except Exception as e:
            warnings.warn(f"Coarse optimization failed: {e}")
            final_params = initial_params

        # Phase 2: More aggressive refinement with better bounds
        def refined_objective(params):
            return optimizer.evaluate_objective(params, outer_side_length=8.0)

        try:
            result_fine = minimize(
                refined_objective,
                final_params,
                method='L-BFGS-B',
                bounds=[(-10, 10), (-10, 10), (0, 360)] * 12 + [(1.0, 20.0)],
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-8},
                tol=1e-8
            )

            if result_fine.success:
                final_params = result_fine.x
        except Exception as e:
            warnings.warn(f"Fine optimization failed: {e}")
            # Continue with whatever we have, but don't give up completely

        # Extract final configuration
        positions = final_params.reshape(-1, 3)
        outer_side_length = optimizer.compute_outer_hex_side_length(positions)

        # Create inner hex data
        inner_hex_data = positions.copy()
        outer_hex_data = np.array([0, 0, 0])

        # Final validation with strict checking
        penalty = optimizer.calculate_penalty(positions, outer_side_length)
        if penalty > 1e7:
            # Try another approach if constraints are violated
            warnings.warn("Solution had constraint violations, attempting recovery...")
            # Use a simpler but more reliable approach
            try:
                positions = np.array([
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
                outer_side_length = 8.0
                inner_hex_data = positions
                outer_hex_data = np.array([0, 0, 0])
            except:
                pass

        return inner_hex_data, outer_hex_data, outer_side_length

    except Exception as e:
        warnings.warn(f"Error in hexagon packing: {str(e)}")
        # Fallback to simple grid if optimization fails
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
        outer_side_length = 8.0

        return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END