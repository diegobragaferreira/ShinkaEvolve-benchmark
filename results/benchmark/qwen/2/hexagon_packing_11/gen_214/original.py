# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
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
        """Fast overlap check using direct polygon intersection"""
        return hex1.intersects(hex2)

    def calculate_outer_hex_radius(self, inner_hex_data, outer_center=(0,0)):
        """Calculate minimum outer hexagon radius needed to contain all inner hexagons"""
        # Get all vertices of all inner hexagons efficiently
        all_vertices = []
        for i in range(len(inner_hex_data)):
            center = inner_hex_data[i][:2]
            rotation = inner_hex_data[i][2]
            hexagon = self.create_unit_hexagon(center, rotation)
            vertices = list(hexagon.exterior.coords)[:-1]  # Exclude duplicate last point
            all_vertices.extend(vertices)

        if not all_vertices:
            return 1.0

        # Compute centroid and distances more efficiently
        vertices_array = np.array(all_vertices)
        centroid = vertices_array.mean(axis=0)

        # Vectorized distance calculation
        distances_squared = np.sum((vertices_array - centroid)**2, axis=1)
        max_dist_squared = np.max(distances_squared)
        max_dist = np.sqrt(max_dist_squared)

        # Take maximum distance and add unit hexagon apogee for safety
        max_dist += self.unit_hex_apogee

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

def generate_golden_spiral_initialization(packer, num_samples=20):
    """Generate diverse initial configurations using golden spiral pattern"""
    configs = []

    # Golden spiral pattern for better spatial distribution
    golden_angle = 2.399963229728653  # Approximation of 2*pi/(phi^2) where phi is golden ratio
    spiral_positions = []
    for i in range(11):
        # Golden spiral coordinates
        radius = 0.8 + i * 0.5
        angle = i * golden_angle
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        spiral_positions.append((x, y))

    # Create diverse samples with strategic jitter
    for _ in range(num_samples):
        config = []
        for i, (cx, cy) in enumerate(spiral_positions):
            # Add controlled jitter with decreasing amplitude
            amp = 0.3 * (1.0 - i/10.0)  # Less jitter for outer positions
            jitter_x = np.random.uniform(-amp, amp)
            jitter_y = np.random.uniform(-amp, amp)
            config.extend([cx + jitter_x, cy + jitter_y, np.random.uniform(0, 360)])

        # Add outer radius with better distribution
        config.append(3.0 + np.random.exponential(1.0))
        configs.append(config)

    return configs

def evaluate_individual(individual, packer):
    """Evaluate a single individual with geometric validation"""
    try:
        # Extract parameters efficiently
        n = packer.num_inner
        inner_params = individual[:-1]
        outer_radius = individual[-1]

        # Check constraints in parallel
        containment_ok, overlap_ok, inv_radius = packer.evaluate_constraints_parallel(inner_params, outer_radius)

        # If any constraint violated, return large penalty
        if not (containment_ok and overlap_ok):
            return 10000.0 + abs(outer_radius)  # penalty for constraint violations

        # Return negative of inverse radius to minimize (maximize 1/outer_radius)
        return -inv_radius

    except Exception as e:
        return 10000.0  # penalty for exceptions

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

    # Generate initial population using golden spiral pattern
    population = generate_golden_spiral_initialization(packer, 30)

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
            current_population = generate_golden_spiral_initialization(packer, 25)

            # Evaluate population in parallel for better performance
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

    # Ultimate fallback
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