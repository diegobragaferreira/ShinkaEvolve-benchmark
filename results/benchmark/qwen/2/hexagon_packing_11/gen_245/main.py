# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid
from joblib import Parallel, delayed
from numba import jit, prange
import time
import warnings
import math

# Constants
UNIT_HEX_RADIUS = 1.0
UNIT_HEX_APOGEE = np.sqrt(3)/2
BENCHMARK_RATIO = 0.2544

@jit(nopython=True, cache=True)
def hexagon_vertices_jit(center_x, center_y, rotation, side_length):
    """Fast computation of hexagon vertices using numba"""
    points = np.empty((6, 2), dtype=np.float64)
    angle_offset = rotation * np.pi / 180.0
    for i in range(6):
        angle = angle_offset + i * np.pi / 3.0
        points[i, 0] = center_x + side_length * np.cos(angle)
        points[i, 1] = center_y + side_length * np.sin(angle)
    return points

@jit(nopython=True, cache=True)
def point_in_polygon_fast(point_x, point_y, polygon_vertices):
    """Fast point-in-polygon test using ray casting algorithm"""
    n = len(polygon_vertices)
    inside = False

    p1x, p1y = polygon_vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_vertices[i % n]
        if point_y > min(p1y, p2y):
            if point_y <= max(p1y, p2y):
                if point_x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (point_y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or point_x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

class HexagonPacker:
    def __init__(self, num_inner=11):
        self.num_inner = num_inner
        self.unit_hex_radius = UNIT_HEX_RADIUS
        self.unit_hex_apogee = UNIT_HEX_APOGEE

    def create_unit_hexagon(self, center=(0,0), rotation=0):
        """Create a unit regular hexagon as a Shapely Polygon"""
        vertices = hexagon_vertices_jit(center[0], center[1], rotation, self.unit_hex_radius)
        points = [(float(v[0]), float(v[1])) for v in vertices]
        return Polygon(points)

    def validate_polygon(self, polygon):
        """Ensure polygon is valid for geometric operations"""
        if not polygon.is_valid:
            return make_valid(polygon)
        return polygon

    def check_containment_fast(self, inner_hexagon, outer_hexagon):
        """Fast containment check by testing all vertices"""
        # Get all vertices of inner hexagon
        vertices = list(inner_hexagon.exterior.coords)[:-1]
        outer_vertices = list(outer_hexagon.exterior.coords)[:-1]
        outer_vertices_np = np.array(outer_vertices)

        # Test each vertex against outer polygon
        for vertex in vertices:
            if not point_in_polygon_fast(vertex[0], vertex[1], outer_vertices_np):
                return False
        return True

    def check_overlap_fast(self, hex1, hex2):
        """Fast overlap check using Separating Axis Theorem for better numerical precision"""
        # Convert polygons to numpy arrays for faster computation
        hex1_coords = np.array(hex1.exterior.coords)
        hex2_coords = np.array(hex2.exterior.coords)

        # Get edges of both polygons
        edges1 = hex1_coords[:-1] - np.roll(hex1_coords[:-1], 1, axis=0)
        edges2 = hex2_coords[:-1] - np.roll(hex2_coords[:-1], 1, axis=0)

        # Get normals to edges (perpendicular vectors)
        normals1 = np.column_stack([-edges1[:, 1], edges1[:, 0]])
        normals2 = np.column_stack([-edges2[:, 1], edges2[:, 0]])

        # Normalize normals
        norms1 = np.linalg.norm(normals1, axis=1, keepdims=True)
        norms2 = np.linalg.norm(normals2, axis=1, keepdims=True)
        normals1 = normals1 / np.where(norms1 == 0, 1, norms1)
        normals2 = normals2 / np.where(norms2 == 0, 1, norms2)

        # Combine all separating axes
        all_axes = np.vstack([normals1, normals2])

        # Project both polygons onto each axis
        for axis in all_axes:
            proj1 = np.dot(hex1_coords[:-1], axis)
            proj2 = np.dot(hex2_coords[:-1], axis)

            # Check for overlap along this axis
            if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                return False  # No overlap along this axis

        return True  # Overlap detected

    def calculate_outer_hex_radius(self, inner_hex_data, outer_center=(0,0)):
        """Calculate minimum outer hexagon radius needed to contain all inner hexagons"""
        max_dist = 0
        for i in range(len(inner_hex_data)):
            center = inner_hex_data[i][:2]
            dist = np.linalg.norm(np.array(center) - np.array(outer_center))
            # Add distance from center to corner of unit hexagon
            dist += self.unit_hex_apogee
            max_dist = max(max_dist, dist)
        return max_dist

    def evaluate_constraints_parallel(self, inner_params, outer_radius):
        """Parallel constraint evaluation with early termination"""
        inner_hexagons = []

        # Create inner hexagons efficiently
        for i in range(self.num_inner):
            x, y, angle = inner_params[3*i:3*i+3]
            hexagon = self.create_unit_hexagon((x, y), angle)
            inner_hexagons.append(hexagon)

        # Create outer hexagon
        outer_hexagon = self.create_unit_hexagon((0, 0), 0)
        outer_coords = list(outer_hexagon.exterior.coords)
        scaled_coords = [(x*outer_radius, y*outer_radius) for x, y in outer_coords]
        outer_hexagon_scaled = Polygon(scaled_coords)

        # Check containment (early termination)
        for hexagon in inner_hexagons:
            if not self.check_containment_fast(hexagon, outer_hexagon_scaled):
                return False, False, 0.0  # containment violated

        # Check overlaps (early termination)
        for i in range(self.num_inner):
            for j in range(i+1, self.num_inner):
                if self.check_overlap_fast(inner_hexagons[i], inner_hexagons[j]):
                    return False, False, 0.0  # overlap violated

        return True, True, 1.0 / outer_radius  # valid solution

def evaluate_individual(individual, packer):
    """Evaluate a single individual with geometric validation"""
    try:
        # Extract parameters
        n = packer.num_inner
        inner_params = individual[:-1]
        outer_radius = individual[-1]

        # Check constraints
        containment_ok, overlap_ok, inv_radius = packer.evaluate_constraints_parallel(inner_params, outer_radius)

        # If any constraint violated, return large penalty
        if not (containment_ok and overlap_ok):
            return 10000.0 + abs(outer_radius)  # penalty for constraint violations

        # Return negative of inverse radius to minimize (maximize 1/outer_radius)
        return -inv_radius

    except Exception as e:
        return 10000.0  # penalty for exceptions

def generate_enhanced_initialization(packer, num_samples=30):
    """Generate diverse initial configurations using enhanced techniques:
    1. Hybrid pattern combining hexagonal and spiral arrangements
    2. Strategic placement based on known good configurations
    3. Better spatial distribution for improved convergence"""
    configs = []

    # Base patterns for different arrangements
    base_patterns = [
        # Pattern 1: Concentric hexagonal rings
        [
            [0, 0, 0],           # center
            [0, 2.0, 0],         # top
            [1.732, 1.0, 0],     # top-right
            [1.732, -1.0, 0],    # bottom-right
            [0, -2.0, 0],        # bottom
            [-1.732, -1.0, 0],   # bottom-left
            [-1.732, 1.0, 0],    # top-left
            [3.464, 0, 0],       # far right
            [-3.464, 0, 0],      # far left
            [0, 3.464, 0],       # very top
            [0, -3.464, 0],      # very bottom
        ],

        # Pattern 2: Spiral with strategic offsets
        [
            [0, 0, 0],
            [-1.0, 0, 30],
            [1.0, 0, 60],
            [0, 1.5, 90],
            [0, -1.5, 120],
            [1.2, 1.2, 150],
            [-1.2, -1.2, 180],
            [1.2, -1.2, 210],
            [-1.2, 1.2, 240],
            [2.0, 0, 270],
            [0, 2.0, 300]
        ],

        # Pattern 3: Optimized layout with strategic spacing
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.165, 0],
            [1.25, 2.165, 0],
            [-1.25, -2.165, 0],
            [1.25, -2.165, 0],
            [-3.75, 2.165, 0],
            [3.75, 2.165, 0],
            [-3.75, -2.165, 0],
            [3.75, -2.165, 0],
        ]
    ]

    for pattern_idx, base_pattern in enumerate(base_patterns):
        # Create samples from each base pattern
        for _ in range(num_samples // len(base_patterns)):
            config = []
            for i, (cx, cy, base_angle) in enumerate(base_pattern):
                # Add strategic jitter based on pattern position
                if i == 0:  # center hexagon - minimal jitter
                    jitter_x = np.random.uniform(-0.1, 0.1)
                    jitter_y = np.random.uniform(-0.1, 0.1)
                    jitter_angle = np.random.uniform(-5, 5)
                else:  # outer hexagons - more pronounced but controlled jitter
                    jitter_x = np.random.uniform(-0.3, 0.3)
                    jitter_y = np.random.uniform(-0.3, 0.3)
                    jitter_angle = np.random.uniform(-10, 10)

                config.extend([cx + jitter_x, cy + jitter_y, base_angle + jitter_angle])

            # Add outer radius with better distribution
            config.append(3.0 + np.random.exponential(1.5))
            configs.append(config)

    # Add some completely random configurations for diversity
    for _ in range(5):
        config = []
        for i in range(11):
            x = np.random.uniform(-4, 4)
            y = np.random.uniform(-4, 4)
            rotation = np.random.uniform(0, 360)
            config.extend([x, y, rotation])
        config.append(4.0 + np.random.exponential(2.0))
        configs.append(config)

    return configs

def hexagon_packing_evolutionary_optimized():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses optimized hybrid approach with golden spiral initialization and multi-stage refinement.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Initialize packer
    packer = HexagonPacker()

    # Generate initial population using enhanced initialization
    population = generate_enhanced_initialization(packer, 35)

    # Track best solution
    best_inv_radius = -float('inf')
    best_individual = None

    # Multi-start evolutionary search with optimized restart strategy
    seeds = [42, 123, 456, 789, 999, 555, 111, 222]  # More seeds for better exploration

    for seed in seeds:
        try:
            # Set random seed for reproducibility
            np.random.seed(seed)

            # Generate new population for this seed with increased diversity
            current_population = generate_enhanced_initialization(packer, 25)

            # Evaluate population in parallel
            results = Parallel(n_jobs=-1)(delayed(evaluate_individual)(individual, packer) for individual in current_population)

            # Find best in this population
            best_score_in_pop = min(results)
            best_index = results.index(best_score_in_pop)
            best_in_population = current_population[best_index]

            if best_in_population is not None and best_score_in_pop < -best_inv_radius:
                best_inv_radius = -best_score_in_pop
                best_individual = best_in_population[:]

        except Exception as e:
            warnings.warn(f"Evolutionary run failed: {str(e)}")
            continue

    # Local refinement with L-BFGS-B if we have a good candidate
    if best_individual is not None:
        try:
            # Prepare bounds for local optimization - tighter ranges for better convergence
            bounds = []
            for _ in range(packer.num_inner):
                bounds.extend([(-10.0, 10.0), (-10.0, 10.0), (0, 360)])  # x, y, angle
            bounds.append((2.5, 15.0))  # outer radius

            # Define objective function for local optimization
            def local_objective(params):
                return evaluate_individual(params, packer)

            # Run local optimization with moderate iterations
            result = minimize(
                local_objective,
                best_individual,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 60, 'ftol': 1e-8, 'gtol': 1e-6}  # Tighter tolerances
            )

            if result.success:
                refined_individual = result.x
                # Check if refined solution is better
                current_inv_radius = -best_inv_radius
                refined_inv_radius = -local_objective(refined_individual)

                if refined_inv_radius > current_inv_radius:
                    best_individual = refined_individual
                    best_inv_radius = -refined_inv_radius

        except Exception as e:
            warnings.warn(f"Local refinement failed: {str(e)}")

    # Final evaluation and formatting
    if best_individual is not None:
        inner_params = best_individual[:-1]
        outer_radius = best_individual[-1]

        # Validate final solution
        containment_ok, overlap_ok, inv_radius = packer.evaluate_constraints_parallel(inner_params, outer_radius)

        if containment_ok and overlap_ok:
            # Format output
            inner_hex_data = np.zeros((packer.num_inner, 3))
            for i in range(packer.num_inner):
                inner_hex_data[i] = inner_params[3*i:3*i+3]

            outer_hex_data = np.array([0, 0, 0])

            return inner_hex_data, outer_hex_data, outer_radius

    # Fallback to best available configuration from initial samples
    best_sample = None
    best_score = -float('inf')

    for sample in population:
        try:
            inner_params = sample[:-1]
            outer_radius = sample[-1]
            containment_ok, overlap_ok, inv_radius = packer.evaluate_constraints_parallel(inner_params, outer_radius)

            if containment_ok and overlap_ok and inv_radius > best_score:
                best_score = inv_radius
                best_sample = sample
        except Exception:
            continue

    if best_sample is not None:
        inner_params = best_sample[:-1]
        outer_radius = best_sample[-1]

        inner_hex_data = np.zeros((packer.num_inner, 3))
        for i in range(packer.num_inner):
            inner_hex_data[i] = inner_params[3*i:3*i+3]

        outer_hex_data = np.array([0, 0, 0])
        return inner_hex_data, outer_hex_data, outer_radius

    # Ultimate fallback to simple configuration
    inner_hex_data = np.array([
        [0, 0, 0],        # center
        [-2.5, 0, 0],     # left
        [2.5, 0, 0],      # right
        [-1.25, 2.17, 0], # top-left
        [1.25, 2.17, 0],  # top-right
        [-1.25, -2.17, 0], # bottom-left
        [1.25, -2.17, 0], # bottom-right
        [-3.75, 2.17, 0], # far top-left
        [3.75, 2.17, 0],  # far top-right
        [-3.75, -2.17, 0], # far bottom-left
        [3.75, -2.17, 0], # far bottom-right
    ])

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 8  # large enough to contain all inner hexagons

    return inner_hex_data, outer_hex_data, outer_hex_side_length

def hexagon_packing_11():
    return hexagon_packing_evolutionary_optimized()

# EVOLVE-BLOCK-END