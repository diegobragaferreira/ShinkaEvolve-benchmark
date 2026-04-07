# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
from numba import jit, njit
import time
import random
from collections import defaultdict
from typing import List, Tuple, Dict, Any

# Numba-optimized geometric functions
@njit
def hexagon_vertices_jit(center_x: float, center_y: float, angle_deg: float, side_length: float = 1.0) -> np.ndarray:
    """Fast computation of hexagon vertices using Numba JIT."""
    angle_rad = np.radians(angle_deg)
    # Vertices of a regular hexagon with side length 1, centered at origin
    base_vertices = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ], dtype=np.float64)

    # Rotate and translate
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    rotated_vertices = np.empty_like(base_vertices)

    for i in range(6):
        rotated_vertices[i] = np.array([
            base_vertices[i][0] * cos_a - base_vertices[i][1] * sin_a,
            base_vertices[i][0] * sin_a + base_vertices[i][1] * cos_a
        ])

    return rotated_vertices + np.array([center_x, center_y], dtype=np.float64)

@njit
def separate_axis_test_jit(vertices1: np.ndarray, vertices2: np.ndarray) -> bool:
    """Fast SAT overlap detection using Numba."""
    # Get all edges of both hexagons
    edges1 = np.empty((6, 2), dtype=np.float64)
    edges2 = np.empty((6, 2), dtype=np.float64)

    for i in range(6):
        edges1[i] = vertices1[i] - vertices1[(i+1)%6]
        edges2[i] = vertices2[i] - vertices2[(i+1)%6]

    # Project both hexagons onto each edge direction
    all_axes = np.vstack([edges1, edges2])

    for axis in all_axes:
        # Normalize axis
        axis_norm = np.sqrt(axis[0]**2 + axis[1]**2)
        if axis_norm == 0:
            continue
        norm_axis = axis / axis_norm

        # Project both polygons onto this axis
        proj1 = np.empty(6, dtype=np.float64)
        proj2 = np.empty(6, dtype=np.float64)

        for i in range(6):
            proj1[i] = vertices1[i][0] * norm_axis[0] + vertices1[i][1] * norm_axis[1]
            proj2[i] = vertices2[i][0] * norm_axis[0] + vertices2[i][1] * norm_axis[1]

        # Check for overlap
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()

        # If no overlap, then they don't intersect
        if max1 < min2 or max2 < min1:
            return False

    return True

@njit
def fast_containment_check(center_x: float, center_y: float, angle: float, 
                          outer_side_length: float) -> bool:
    """Fast containment check using distance from center."""
    vertices = hexagon_vertices_jit(center_x, center_y, angle)
    max_dist = np.max(np.sqrt(np.sum((vertices - np.array([center_x, center_y]))**2, axis=1)))
    # For unit hexagons, containment requires max distance <= outer_side_length * sqrt(3)/2
    return max_dist <= outer_side_length * np.sqrt(3) / 2

class SpatialHash:
    """Spatial hashing for efficient overlap detection"""
    
    def __init__(self, cell_size: float = 2.0):
        self.cell_size = cell_size
        self.hash_table = defaultdict(list)

    def get_grid_coords(self, x: float, y: float) -> Tuple[int, int]:
        """Get grid coordinates for a point"""
        return int(x / self.cell_size), int(y / self.cell_size)

    def add_hexagon(self, index: int, x: float, y: float):
        """Add a hexagon to the spatial hash"""
        cell_x, cell_y = self.get_grid_coords(x, y)
        self.hash_table[(cell_x, cell_y)].append(index)

    def get_candidates(self, x: float, y: float) -> List[int]:
        """Get candidate hexagons in nearby cells"""
        cell_x, cell_y = self.get_grid_coords(x, y)
        candidates = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (cell_x + dx, cell_y + dy)
                candidates.extend(self.hash_table[key])
        return candidates

class HexagonPackingOptimizer:
    """Main optimization class combining best practices from various approaches"""
    
    def __init__(self):
        self.spatial_hash = SpatialHash()
        
    def create_hexagon_polygon(self, center_x: float, center_y: float, angle_deg: float, 
                              side_length: float = 1.0) -> Polygon:
        """Create a Shapely polygon for a hexagon."""
        vertices = hexagon_vertices_jit(center_x, center_y, angle_deg, side_length)
        return Polygon(vertices)
        
    def calculate_min_enclosing_hexagon_fast(self, inner_hex_data: np.ndarray, 
                                           scale_factor: float = 1.05) -> Tuple[float, np.ndarray]:
        """Fast calculation of minimum enclosing hexagon using Numba."""
        # Get all vertices of all inner hexagons
        all_vertices = np.empty((0, 2), dtype=np.float64)

        for i in range(len(inner_hex_data)):
            center_x, center_y, angle = inner_hex_data[i]
            vertices = hexagon_vertices_jit(center_x, center_y, angle)
            all_vertices = np.vstack([all_vertices, vertices])

        if len(all_vertices) == 0:
            return 1.0, np.array([0., 0.])

        # Find bounding circle radius
        centroid = np.mean(all_vertices, axis=0)
        distances = np.sqrt(np.sum((all_vertices - centroid)**2, axis=1))
        max_distance = np.max(distances)

        # For a regular hexagon, side length = max_distance * sqrt(3)/2
        side_length = max_distance * 2 / np.sqrt(3) * scale_factor

        return side_length, centroid
        
    def check_overlaps_spatial_hash(self, hex_data: np.ndarray, 
                                  vertices_cache: List[np.ndarray] = None) -> bool:
        """Check overlaps using spatial hashing to reduce comparisons."""
        if len(hex_data) <= 1:
            return False

        # Create spatial hash grid
        self.spatial_hash.hash_table.clear()
        for i, (cx, cy, angle) in enumerate(hex_data):
            self.spatial_hash.add_hexagon(i, cx, cy)

        # Check overlaps efficiently using neighbor lists
        for i in range(len(hex_data)):
            cx1, cy1, angle1 = hex_data[i]
            candidates = self.spatial_hash.get_candidates(cx1, cy1)

            # Check overlap only with neighbors
            for j in candidates:
                if i >= j:  # Avoid duplicate checks
                    continue

                cx2, cy2, angle2 = hex_data[j]

                # Early rejection using bounding boxes (simple approach)
                if abs(cx1 - cx2) > 2.0 or abs(cy1 - cy2) > 2.0:
                    continue

                # Use cached vertices if available
                if vertices_cache is not None:
                    vertices1 = vertices_cache[i]
                    vertices2 = vertices_cache[j]
                else:
                    vertices1 = hexagon_vertices_jit(cx1, cy1, angle1)
                    vertices2 = hexagon_vertices_jit(cx2, cy2, angle2)

                # Perform overlap test
                if separate_axis_test_jit(vertices1, vertices2):
                    return True
                    
        return False
        
    def evaluate_solution(self, solution_array: np.ndarray, 
                         penalty_weights: Dict[str, float] = None) -> Tuple[float, float]:
        """Improved evaluation with better containment and overlap checking."""
        if penalty_weights is None:
            penalty_weights = {'containment': 10000, 'overlap': 100000, 'boundary': 5000}

        # Reshape solution array into 12 hexagons with (x, y, angle) each
        inner_hex_data = solution_array.reshape(-1, 3)

        # Calculate the minimum enclosing hexagon
        min_side_length, centroid = self.calculate_min_enclosing_hexagon_fast(inner_hex_data)

        # Check all constraints
        num_hex = len(inner_hex_data)
        penalty = 0.0

        # Create outer hexagon polygon for containment checks (this is what we're optimizing)
        outer_hex = self.create_hexagon_polygon(centroid[0], centroid[1], 0, min_side_length)

        # Check containment for all hexagons - FAST VERSION
        for i in range(num_hex):
            center_x, center_y, angle = inner_hex_data[i]
            # Fast containment check instead of full polygon operation
            if not fast_containment_check(center_x, center_y, angle, min_side_length):
                penalty += penalty_weights['containment']

        # Check overlaps using spatial hashing for efficiency
        # Precompute vertices for spatial hash check
        vertices_cache = []
        for i in range(num_hex):
            vertices = hexagon_vertices_jit(*inner_hex_data[i])
            vertices_cache.append(vertices)

        # Use spatial hash overlap detection
        if self.check_overlaps_spatial_hash(inner_hex_data, vertices_cache):
            penalty += penalty_weights['overlap']

        # Return negative inverse side length plus penalty
        objective_value = -1.0 / min_side_length + penalty

        return objective_value, min_side_length
        
    def generate_symmetric_initial_population(self, pop_size: int, num_hexagons: int = 12) -> List[np.ndarray]:
        """Generate intelligent initial population with symmetry patterns."""
        population = []

        # Base symmetric configuration: hexagonal pattern around center
        # Start with central hexagon and surrounding ring elements
        # Pattern: one center + 6 surrounding + 5 additional
        base_angles = np.linspace(0, 360, 7, endpoint=False)  # 6 surrounding positions + 1 center

        for _ in range(pop_size):
            # Start with a symmetric layout
            hex_config = []

            # Central hexagon
            hex_config.append([0, 0, np.random.uniform(0, 360)])

            # Surrounding hexagons forming a ring
            for i in range(1, 7):  # First 6 surrounding
                radius = 2.0
                angle_rad = np.radians(base_angles[i] + np.random.uniform(-30, 30))
                x = radius * np.cos(angle_rad)
                y = radius * np.sin(angle_rad)
                angle = np.random.uniform(0, 360)
                hex_config.append([x, y, angle])

            # Add remaining 5 hexagons with additional symmetry
            for i in range(7, 12):
                # Distribute around a larger ring
                radius = 3.5
                angle_rad = np.radians(base_angles[i % 6] + np.random.uniform(-15, 15))
                x = radius * np.cos(angle_rad)
                y = radius * np.sin(angle_rad)
                angle = np.random.uniform(0, 360)
                hex_config.append([x, y, angle])

            # Add small random perturbations to break exact symmetry
            for i in range(len(hex_config)):
                hex_config[i][0] += np.random.normal(0, 0.1)
                hex_config[i][1] += np.random.normal(0, 0.1)
                hex_config[i][2] += np.random.uniform(-5, 5)

            population.append(np.array(hex_config).flatten())

        return population

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

    # Define bounds for each parameter: x, y in [-8, 8], angle in [0, 360)
    bounds = []
    for i in range(12):
        bounds.extend([(-8, 8), (-8, 8), (0, 360)])

    # Initialize optimizer
    optimizer = HexagonPackingOptimizer()
    
    best_solution = None
    best_side_length = float('inf')
    best_objective = float('inf')

    # Multi-start optimization with different strategies
    strategies = [
        {"popsize": 20, "maxiter": 50, "mutation": (0.5, 1.0), "recombination": 0.7},
        {"popsize": 30, "maxiter": 80, "mutation": (0.7, 1.0), "recombination": 0.8},
        {"popsize": 40, "maxiter": 100, "mutation": (0.8, 1.0), "recombination": 0.85}
    ]

    for strategy_idx, strategy in enumerate(strategies):
        # Generate better initial population for this run
        initial_pop = optimizer.generate_symmetric_initial_population(strategy["popsize"] - 1)
        # Add a fresh random solution to ensure variety
        random_solution = np.random.uniform(-8, 8, num_variables)
        initial_pop.append(random_solution)

        try:
            result = differential_evolution(
                lambda x: optimizer.evaluate_solution(x)[0],
                bounds,
                maxiter=strategy["maxiter"],
                popsize=strategy["popsize"],
                mutation=strategy["mutation"],
                recombination=strategy["recombination"],
                seed=42 + strategy_idx,
                disp=False,
                init=initial_pop
            )

            # Evaluate final solution
            final_objective, side_length = optimizer.evaluate_solution(result.x)

            if final_objective < best_objective:
                best_objective = final_objective
                best_side_length = side_length
                best_solution = result.x.copy()

        except Exception as e:
            continue

    if best_solution is None:
        # Fallback to simple symmetric configuration
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
            [0, -4, 0],  # far bottom-center
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    # Extract the best solution
    inner_hex_data = best_solution.reshape(-1, 3)

    # Calculate the resulting outer hexagon side length
    min_side_length, centroid = optimizer.calculate_min_enclosing_hexagon_fast(inner_hex_data, 1.05)

    # Center the outer hexagon at the centroid of inner hexagons
    outer_hex_data = np.array([centroid[0], centroid[1], 0])

    # Final verification
    _, final_side_length = optimizer.evaluate_solution(best_solution)

    # Calculate benchmark ratio for reporting
    benchmark_ratio = (1.0 / final_side_length) / 0.2537

    # Print metrics
    print(f"inv_outer_hex_side_length: {1.0/final_side_length:.8f}")
    print(f"benchmark_ratio: {benchmark_ratio:.8f}")
    print(f"eval_time: {time.time() - start_time:.4f}s")

    return inner_hex_data, outer_hex_data, final_side_length

# EVOLVE-BLOCK-END