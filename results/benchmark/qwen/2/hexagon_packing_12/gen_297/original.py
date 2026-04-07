# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import time
from numba import jit
import warnings
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional
import copy

# Core geometric utilities
class HexagonGeometry:
    """Efficient geometric computations for hexagon operations."""

    def __init__(self):
        self.side_length = 1.0
        self.apothem = np.sqrt(3) / 2
        self.height = 2 * self.apothem
        self.width = 2 * self.side_length

    @staticmethod
    @jit(nopython=True)
    def vertices_jit(center_x: float, center_y: float, rotation_deg: float) -> np.ndarray:
        """JIT compiled hexagon vertex calculation."""
        angle_rad = np.radians(rotation_deg)
        # Unit hexagon vertices centered at origin
        base_vertices = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])

        # Rotate and translate
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_vertices = base_vertices @ rotation_matrix.T
        return rotated_vertices + np.array([center_x, center_y])

    def vertices(self, center_x: float, center_y: float, rotation_deg: float) -> np.ndarray:
        """Get hexagon vertices."""
        return self.vertices_jit(center_x, center_y, rotation_deg)

# Efficient constraint validation with spatial hashing
class SpatialHashGrid:
    """Spatial hash grid for efficient neighbor search."""

    def __init__(self, cell_size: float = 2.0):
        self.cell_size = cell_size
        self.grid = defaultdict(list)

    @staticmethod
    @jit(nopython=True)
    def _hash_cell_numba(x: float, y: float, cell_size: float) -> tuple:
        """JIT compiled hash function."""
        return (int(x // cell_size), int(y // cell_size))

    def _hash_cell(self, x: float, y: float) -> tuple:
        """Hash coordinates to grid cell."""
        return self._hash_cell_numba(x, y, self.cell_size)

    def insert(self, hex_id: int, center_x: float, center_y: float):
        """Insert a hexagon into the spatial grid."""
        cell = self._hash_cell(center_x, center_y)
        self.grid[cell].append(hex_id)

    @staticmethod
    @jit(nopython=True)
    def _get_candidates_numba(grid, cell, cell_size) -> list:
        """JIT compiled candidate retrieval."""
        candidates = []
        # Check the cell and its 8 neighboring cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell[0] + dx, cell[1] + dy)
                if neighbor_cell in grid:
                    candidates.extend(grid[neighbor_cell])
        return candidates

    def get_candidates(self, center_x: float, center_y: float) -> list:
        """Get candidate hexagons that might collide with given position."""
        cell = self._hash_cell(center_x, center_y)
        return self._get_candidates_numba(self.grid, cell, self.cell_size)

# Optimized overlap detection using SAT
@jit(nopython=True)
def _get_edges(vertices: np.ndarray) -> np.ndarray:
    """Get edges from vertices."""
    edges = np.empty((len(vertices), 2))
    for i in range(len(vertices)):
        edges[i] = vertices[i] - vertices[(i+1) % len(vertices)]
    return edges

@jit(nopython=True)
def _project_polygon_onto_axis(vertices: np.ndarray, axis: np.ndarray) -> tuple:
    """Project polygon vertices onto an axis."""
    projections = np.empty(len(vertices))
    for i in range(len(vertices)):
        projections[i] = vertices[i, 0] * axis[0] + vertices[i, 1] * axis[1]
    return np.min(projections), np.max(projections)

@jit(nopython=True)
def _hexagon_overlap_sat_jit(hex1_vertices: np.ndarray, hex2_vertices: np.ndarray) -> bool:
    """Separating Axis Theorem for hexagon overlap detection."""
    # Get edges of both hexagons
    edges1 = _get_edges(hex1_vertices)
    edges2 = _get_edges(hex2_vertices)

    # Combine all axes (edges perpendicular to edges)
    all_axes = np.empty((len(edges1) + len(edges2), 2))
    for i in range(len(edges1)):
        # Normal vector to edge (perpendicular)
        all_axes[i] = np.array([-edges1[i, 1], edges1[i, 0]])
        # Normalize
        norm = np.sqrt(all_axes[i, 0]**2 + all_axes[i, 1]**2)
        if norm > 1e-10:
            all_axes[i] /= norm
    for i in range(len(edges2)):
        # Normal vector to edge (perpendicular)
        all_axes[len(edges1) + i] = np.array([-edges2[i, 1], edges2[i, 0]])
        # Normalize
        norm = np.sqrt(all_axes[len(edges1) + i, 0]**2 + all_axes[len(edges1) + i, 1]**2)
        if norm > 1e-10:
            all_axes[len(edges1) + i] /= norm

    # Check each axis
    for axis in all_axes:
        min1, max1 = _project_polygon_onto_axis(hex1_vertices, axis)
        min2, max2 = _project_polygon_onto_axis(hex2_vertices, axis)

        # If no overlap on this axis, polygons don't overlap
        if max1 < min2 or max2 < min1:
            return False

    return True

# Add JIT compilation to the constraint validator methods for better performance
class ConstraintValidator:
    """Robust constraint validation for hexagon packing."""

    def __init__(self, geometry: HexagonGeometry):
        self.geo = geometry
        self.spatial_hash = SpatialHashGrid()

    @staticmethod
    @jit(nopython=True)
    def is_contained_jit(hex_vertices: np.ndarray, outer_center_x: float,
                        outer_center_y: float, outer_side_length: float) -> bool:
        """JIT compiled containment check."""
        # For a regular hexagon, we can check distance from center
        dist_from_center = np.sqrt((hex_vertices[:, 0] - outer_center_x)**2 +
                                  (hex_vertices[:, 1] - outer_center_y)**2)
        # Maximum distance from center in a regular hexagon is side_length * sqrt(3)/2
        max_radius = outer_side_length * np.sqrt(3) / 2
        return np.all(dist_from_center <= max_radius)

    def is_contained(self, hex_vertices: np.ndarray, outer_center_x: float,
                     outer_center_y: float, outer_side_length: float) -> bool:
        """Check if all hexagon vertices are within outer hexagon."""
        return self.is_contained_jit(hex_vertices, outer_center_x, outer_center_y, outer_side_length)

    @staticmethod
    @jit(nopython=True)
    def has_overlap_fast_jit(hex1_vertices: np.ndarray, hex2_vertices: np.ndarray) -> bool:
        """Fast approximate overlap check before precise check."""
        # Simple distance check first
        center1 = np.mean(hex1_vertices, axis=0)
        center2 = np.mean(hex2_vertices, axis=0)
        dist_centers = np.sqrt(np.sum((center1 - center2)**2))

        # If centers are far apart, no overlap
        if dist_centers > 2.0:  # Approximate sum of radii for unit hexagons
            return False

        # Otherwise, use SAT for precise check
        return _hexagon_overlap_sat_jit(hex1_vertices, hex2_vertices)

    def has_overlap_fast(self, hex1_vertices: np.ndarray, hex2_vertices: np.ndarray) -> bool:
        """Fast approximate overlap check before precise check."""
        return self.has_overlap_fast_jit(hex1_vertices, hex2_vertices)

    def has_overlap_with_spatial_hash(self, hex_data: np.ndarray) -> bool:
        """Fast overlap detection using spatial hashing."""
        # Clear existing grid
        self.spatial_hash.grid.clear()

        # Insert all hexagons into spatial grid
        num_hex = len(hex_data)
        for i in range(num_hex):
            center_x, center_y, _ = hex_data[i]
            self.spatial_hash.insert(i, center_x, center_y)

        # Check for overlaps with nearby hexagons only
        for i in range(num_hex):
            center_x, center_y, _ = hex_data[i]
            candidates = self.spatial_hash.get_candidates(center_x, center_y)

            # Check against candidates in neighboring cells
            for j in candidates:
                if i >= j:  # Avoid duplicate comparisons
                    continue

                vertices1 = self.geo.vertices(*hex_data[i])
                vertices2 = self.geo.vertices(*hex_data[j])

                if self.has_overlap_fast(vertices1, vertices2):
                    return True

        return False

# Solution evaluator with performance optimizations
class SolutionEvaluator:
    """Handles solution evaluation with efficient constraint checking."""

    def __init__(self, geometry: HexagonGeometry, validator: ConstraintValidator):
        self.geo = geometry
        self.validator = validator

    def calculate_min_enclosing_hexagon(self, inner_hex_data: np.ndarray,
                                       scale_factor: float = 1.0) -> tuple:
        """Calculate minimum enclosing hexagon side length."""
        # Get all vertices of all inner hexagons
        all_vertices = []
        for i in range(len(inner_hex_data)):
            center_x, center_y, angle = inner_hex_data[i]
            vertices = self.geo.vertices(center_x, center_y, angle)
            all_vertices.extend(vertices)

        all_vertices = np.array(all_vertices)

        # Find bounding circle radius
        centroid = np.mean(all_vertices, axis=0)
        distances = np.sqrt(np.sum((all_vertices - centroid)**2, axis=1))
        max_distance = np.max(distances)

        # For a regular hexagon, side length = max_distance * sqrt(3)/2
        side_length = max_distance * 2 / np.sqrt(3) * scale_factor
        return side_length, centroid

    def evaluate(self, solution_array: np.ndarray, outer_side_length: float = 10.0) -> float:
        """Evaluate solution quality with penalty-based constraints."""
        # Reshape solution array into 12 hexagons with (x, y, angle) each
        inner_hex_data = solution_array.reshape(-1, 3)

        # Calculate the minimum enclosing hexagon
        min_side_length, centroid = self.calculate_min_enclosing_hexagon(inner_hex_data, 1.05)

        # Check all constraints
        num_hex = len(inner_hex_data)
        penalty = 0.0

        # Check containment for all hexagons
        for i in range(num_hex):
            center_x, center_y, angle = inner_hex_data[i]
            vertices = self.geo.vertices(center_x, center_y, angle)

            # Check if hexagon is contained properly
            if not self.validator.is_contained(vertices, centroid[0], centroid[1], min_side_length):
                # Apply higher penalty for containment violations since they are fundamental
                penalty += 15000.0

        # Check overlaps using spatial hashing for efficiency
        if self.validator.has_overlap_with_spatial_hash(inner_hex_data):
            # Apply higher penalty for overlaps with adaptive weighting
            penalty += 50000.0

        # Return negative inverse side length plus penalty
        objective_value = -1.0 / min_side_length + penalty
        return objective_value

# Optimizer with progressive refinement stages
class PackingOptimizer:
    """Main optimization controller with structured approach."""

    def __init__(self, num_hexagons: int = 12):
        self.num_hexagons = num_hexagons
        self.geometry = HexagonGeometry()
        self.validator = ConstraintValidator(self.geometry)
        self.evaluator = SolutionEvaluator(self.geometry, self.validator)
        self.initial_solution_cache = {}

    def _generate_initial_solution(self) -> np.ndarray:
        """Generate an initial symmetric solution with cache support."""
        # Check if cached version exists
        cache_key = "symmetric_12_hex"
        if cache_key in self.initial_solution_cache:
            return self.initial_solution_cache[cache_key]

        # Start with a good symmetric arrangement
        base_positions = [
            [0, 0],           # Center
            [-2.5, 0],        # Left
            [2.5, 0],         # Right
            [-1.25, 2.17],    # Top-left
            [1.25, 2.17],     # Top-right
            [-1.25, -2.17],   # Bottom-left
            [1.25, -2.17],    # Bottom-right
            [-3.75, 2.17],    # Far top-left
            [3.75, 2.17],     # Far top-right
            [-3.75, -2.17],   # Far bottom-left
            [3.75, -2.17],    # Far bottom-right
            [0, -4],          # Far bottom
        ]

        # Flatten and add rotations (zero for now)
        positions_with_angles = np.array(base_positions)
        positions_with_angles = np.column_stack([positions_with_angles, np.zeros(positions_with_angles.shape[0])])

        initial_solution = positions_with_angles.flatten()

        # Cache the solution
        self.initial_solution_cache[cache_key] = initial_solution

        return initial_solution

    def _setup_bounds(self) -> list:
        """Setup parameter bounds for optimization."""
        bounds = []
        for i in range(self.num_hexagons):
            # X and Y positions
            bounds.extend([(-8.0, 8.0), (-8.0, 8.0)])
            # Rotation: 0-360 degrees
            bounds.append((0.0, 360.0))
        return bounds

    def _optimize_stage(self, bounds: list, maxiter: int, popsize: int,
                       mutation: tuple, recombination: float,
                       initial_population: list = None, stage_name: str = "") -> Optional[Dict]:
        """Run optimization stage."""
        try:
            if stage_name:
                print(f"Starting {stage_name}...")

            result = differential_evolution(
                self.evaluator.evaluate,
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                mutation=mutation,
                recombination=recombination,
                seed=42,
                disp=False,
                init=initial_population
            )
            return result
        except Exception as e:
            warnings.warn(f"Optimization stage failed: {e}")
            return None

    def optimize(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Main optimization routine with progressive refinement."""
        start_time = time.time()

        # Setup bounds
        bounds = self._setup_bounds()

        # Stage 1: Coarse optimization
        initial_solution = self._generate_initial_solution()

        # Generate population for better initialization
        initial_pop = [initial_solution]
        for _ in range(9):
            # Add perturbed versions
            perturbed = initial_solution + np.random.normal(0, 0.5, len(initial_solution))
            initial_pop.append(perturbed)

        coarse_result = self._optimize_stage(
            bounds, maxiter=50, popsize=15,
            mutation=(0.8, 1.0), recombination=0.7,
            initial_population=initial_pop,
            stage_name="stage 1: Coarse optimization"
        )

        if coarse_result is None or not coarse_result.success:
            warnings.warn("Coarse optimization failed, using initial solution")
            best_solution = initial_solution
        else:
            best_solution = coarse_result.x

        # Stage 2: Fine optimization
        fine_result = self._optimize_stage(
            bounds, maxiter=100, popsize=25,
            mutation=(1.0, 1.0), recombination=0.8,
            initial_population=[best_solution] + [np.random.normal(best_solution, 1.0) for _ in range(24)],
            stage_name="stage 2: Fine optimization"
        )

        if fine_result is not None and fine_result.success:
            final_solution = fine_result.x
        else:
            final_solution = best_solution

        # Evaluate final solution
        inner_hex_data = final_solution.reshape(-1, 3)
        min_side_length, centroid = self.evaluator.calculate_min_enclosing_hexagon(inner_hex_data, 1.05)

        # Center the outer hexagon at the centroid of inner hexagons
        outer_hex_data = np.array([centroid[0], centroid[1], 0])

        eval_time = time.time() - start_time
        print(f"Optimization completed in {eval_time:.2f} seconds")

        return inner_hex_data, outer_hex_data, min_side_length

# Main entry point with improved error handling and metrics
def hexagon_packing_12() -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Constructs an optimized packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    try:
        optimizer = PackingOptimizer()
        inner_hex_data, outer_hex_data, outer_hex_side_length = optimizer.optimize()

        # Calculate benchmark ratio
        inv_outer_hex_side_length = 1.0 / outer_hex_side_length
        benchmark_ratio = inv_outer_hex_side_length / 0.2537

        # Output metrics for verification
        print(f"inv_outer_hex_side_length: {inv_outer_hex_side_length:.8f}")
        print(f"benchmark_ratio: {benchmark_ratio:.8f}")
        print(f"eval_time: {time.time() - start_time:.4f}s")

        return inner_hex_data, outer_hex_data, outer_hex_side_length

    except Exception as e:
        warnings.warn(f"Error in hexagon packing: {e}")
        # Fallback to simple symmetric arrangement
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

        # Calculate fallback metrics
        inv_outer_hex_side_length = 1.0 / outer_hex_side_length
        benchmark_ratio = inv_outer_hex_side_length / 0.2537
        print(f"Fallback - inv_outer_hex_side_length: {inv_outer_hex_side_length:.8f}")
        print(f"Fallback - benchmark_ratio: {benchmark_ratio:.8f}")
        print(f"eval_time: {time.time() - start_time:.4f}s")

        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END