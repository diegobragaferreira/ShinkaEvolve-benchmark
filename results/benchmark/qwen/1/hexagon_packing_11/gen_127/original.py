# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
from numba import jit
import warnings
import math
from collections import defaultdict
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')

@jit(nopython=True)
def hexagon_vertices(x, y, angle_deg, side_length=1):
    """Calculate vertices of a hexagon given center, angle, and side length"""
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vx = x + side_length * np.cos(theta)
        vy = y + side_length * np.sin(theta)
        vertices.append((vx, vy))
    return np.array(vertices)

def get_hexagon_polygon(x, y, angle_deg, side_length=1):
    """Get shapely polygon representation of hexagon"""
    vertices = hexagon_vertices(x, y, angle_deg, side_length)
    return Polygon(vertices)

def calculate_outer_hexagon_radius(inner_positions, inner_angles):
    """Calculate minimum radius needed to contain all inner hexagons"""
    max_dist = 0
    outer_center = (0, 0)

    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_positions)):
        pos = inner_positions[i]
        angle = inner_angles[i]
        hex_vertices = hexagon_vertices(pos[0], pos[1], angle)
        all_vertices.extend(hex_vertices)

    # Find maximum distance from center
    for vertex in all_vertices:
        dist = np.sqrt((vertex[0] - outer_center[0])**2 + (vertex[1] - outer_center[1])**2)
        max_dist = max(max_dist, dist)

    # Add buffer for safety and account for hexagon shape
    return max_dist * 1.1  # Safety factor

def check_containment(hex_poly, outer_poly):
    """Check if hexagon is completely contained within outer hexagon"""
    return outer_poly.contains(hex_poly) or (outer_poly.intersects(hex_poly) and
                                           outer_poly.intersection(hex_poly).area == hex_poly.area)

def fast_collision_check(hex_poly1, hex_poly2):
    """Fast collision check using bounding boxes"""
    bbox1 = hex_poly1.bounds
    bbox2 = hex_poly2.bounds

    # Quick bounding box overlap test
    if (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or
        bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1]):
        return False

    # More precise check
    return hex_poly1.intersects(hex_poly2)

class GeometricConstraintSolver:
    """Solves the hexagon packing problem using geometric insights and constraint satisfaction"""

    def __init__(self, num_hexagons=11, side_length=1):
        self.num_hexagons = num_hexagons
        self.side_length = side_length
        self.hex_radius = side_length * np.sqrt(3) / 2  # Distance from center to corner

    def initialize_geometrically(self):
        """Initialize using geometric insights instead of random guessing"""
        # Start with a good initial configuration based on known packing patterns
        positions = []
        angles = []

        # Central hexagon
        positions.append([0.0, 0.0])
        angles.append(0.0)

        # Surrounding hexagons in a hexagonal pattern
        for i in range(6):
            angle = i * 60
            radius = 2.0 * self.hex_radius  # Proper spacing for unit hexagons
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            positions.append([x, y])
            angles.append(0.0)

        # Additional positions arranged in a way that maximizes packing density
        # These are derived from mathematical packing considerations
        additional_positions = [
            (-3.0, 1.5), (3.0, 1.5),
            (-3.0, -1.5), (3.0, -1.5),
            (0.0, 3.0), (0.0, -3.0),
            (2.0, 2.0), (-2.0, -2.0),
            (-2.0, 2.0), (2.0, -2.0)
        ]

        for pos in additional_positions:
            if len(positions) < self.num_hexagons:
                positions.append(list(pos))
                angles.append(0.0)

        # Ensure we have exactly the right number
        while len(positions) < self.num_hexagons:
            positions.append([0.0, 0.0])
            angles.append(0.0)

        return np.array(positions[:self.num_hexagons]), np.array(angles[:self.num_hexagons])

    def compute_clustering_score(self, positions):
        """Compute a clustering metric to identify dense regions"""
        if len(positions) < 2:
            return 0.0

        distances = cdist(positions, positions)
        # Avoid diagonal elements (distance to itself)
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        return np.mean(min_distances)

    def build_spatial_grid(self, hexagons, grid_size=5.0):
        """Build spatial grid for fast collision detection"""
        grid = defaultdict(list)
        for i, hex_poly in enumerate(hexagons):
            bbox = hex_poly.bounds
            min_x, min_y, max_x, max_y = bbox
            for x in range(int(min_x/grid_size), int(max_x/grid_size)+1):
                for y in range(int(min_y/grid_size), int(max_y/grid_size)+1):
                    grid[(x,y)].append(i)
        return grid

    def get_collision_candidates(self, grid, hex_index, hex_poly):
        """Get potential collision candidates efficiently"""
        candidates = []
        bbox = hex_poly.bounds
        min_x, min_y, max_x, max_y = bbox

        for x in range(int(min_x/grid_size), int(max_x/grid_size)+1):
            for y in range(int(min_y/grid_size), int(max_y/grid_size)+1):
                candidates.extend(grid.get((x,y), []))
        return [i for i in candidates if i != hex_index]

class Optimizer:
    """Main optimizer using constraint satisfaction with local refinement"""

    def __init__(self, num_hexagons=11, side_length=1):
        self.num_hexagons = num_hexagons
        self.side_length = side_length
        self.solver = GeometricConstraintSolver(num_hexagons, side_length)
        self.grid_size = 5.0

    def evaluate_constraint_violation(self, positions, angles):
        """Calculate constraint violation for optimization"""
        # Create hexagon polygons
        hexagons = []
        for i in range(len(positions)):
            hex_poly = get_hexagon_polygon(positions[i][0], positions[i][1], angles[i])
            hexagons.append(hex_poly)

        # Check containment constraint
        outer_radius = calculate_outer_hexagon_radius(positions, angles)
        outer_hexagon = get_hexagon_polygon(0, 0, 0, outer_radius)

        containment_violations = 0
        overlap_violations = 0

        # Check containment
        for hex_poly in hexagons:
            if not check_containment(hex_poly, outer_hexagon):
                containment_violations += 1

        # Check overlaps (using spatial grid for efficiency)
        grid = self.solver.build_spatial_grid(hexagons, self.grid_size)

        # Check for overlaps
        for i in range(len(hexagons)):
            candidates = self.solver.get_collision_candidates(grid, i, hexagons[i])
            for j in candidates:
                if i != j and fast_collision_check(hexagons[i], hexagons[j]):
                    overlap_violations += 1

        return containment_violations + overlap_violations

    def compute_objective(self, positions, angles):
        """Compute the objective value (negative 1/outer_radius)"""
        outer_radius = calculate_outer_hexagon_radius(positions, angles)
        return -1.0 / outer_radius

    def optimize_single(self, initial_positions, initial_angles):
        """Optimize a single configuration using scipy minimize"""
        # Flatten the initial solution
        initial_solution = np.concatenate([initial_positions.flatten(), initial_angles])

        def objective(x):
            positions = x[:22].reshape(-1, 2)
            angles = x[22:]

            # Compute constraint violations
            constraint_violation = self.evaluate_constraint_violation(positions, angles)

            if constraint_violation > 0:
                # Penalties for constraint violations
                return 1e10 + constraint_violation * 1e6

            # Compute objective
            obj = self.compute_objective(positions, angles)
            return obj

        # Bounds for positions and angles
        bounds = []
        # Position bounds
        for _ in range(22):
            bounds.append((-10.0, 10.0))  # X and Y coordinates
        # Angle bounds
        for _ in range(self.num_hexagons):
            bounds.append((0.0, 360.0))   # Rotation angles

        # Optimize with multiple methods for robustness
        methods = ['L-BFGS-B', 'TNC']
        best_result = None
        best_value = float('inf')

        for method in methods:
            try:
                result = minimize(
                    objective,
                    initial_solution,
                    method=method,
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-5}
                )

                if result.success and result.fun < best_value:
                    best_value = result.fun
                    best_result = result

            except Exception:
                continue

        if best_result is None:
            # Fallback to initial solution
            return initial_positions, initial_angles

        # Extract solution
        final_positions = best_result.x[:22].reshape(-1, 2)
        final_angles = best_result.x[22:]

        return final_positions, final_angles

    def optimize(self):
        """Main optimization loop with multiple restarts"""
        best_positions = None
        best_angles = None
        best_obj = float('inf')

        # Multiple restarts to find better solutions
        num_restarts = 5

        for restart in range(num_restarts):
            # Initialize geometrically
            initial_positions, initial_angles = self.solver.initialize_geometrically()

            # Add small random noise for diversity
            np.random.seed(restart)
            initial_positions += np.random.normal(0, 0.1, initial_positions.shape)
            initial_angles += np.random.normal(0, 2, initial_angles.shape)

            # Optimize this initialization
            final_positions, final_angles = self.optimize_single(initial_positions, initial_angles)

            # Evaluate objective
            current_obj = self.compute_objective(final_positions, final_angles)

            if current_obj < best_obj:
                best_obj = current_obj
                best_positions = final_positions.copy()
                best_angles = final_angles.copy()

        return best_positions, best_angles

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    try:
        # Initialize optimizer
        optimizer = Optimizer(num_hexagons=11, side_length=1.0)

        # Run optimization
        final_positions, final_angles = optimizer.optimize()

        # Create inner hex data
        inner_hex_data = np.column_stack([final_positions, final_angles])

        # Create outer hex data (centered)
        outer_hex_data = np.array([0, 0, 0])

        # Calculate outer hex side length
        outer_radius = calculate_outer_hexagon_radius(final_positions, final_angles)
        # Convert to side length for regular hexagon
        outer_hex_side_length = outer_radius / (np.sqrt(3) / 2)

        elapsed_time = time.time() - start_time
        print(f"Optimization completed in {elapsed_time:.2f} seconds")

        return inner_hex_data, outer_hex_data, outer_hex_side_length

    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback to initial solution
        inner_hex_data = np.array([
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
            [3.75, -2.17, 0],  # far bottom-right
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END