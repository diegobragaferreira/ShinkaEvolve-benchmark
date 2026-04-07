# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import time
from numba import jit, njit
import warnings
from collections import defaultdict

class HexagonGeometry:
    """Efficient geometric computations for hexagon operations."""

    def __init__(self):
        self.side_length = 1.0
        self.apothem = np.sqrt(3) / 2
        self.height = 2 * self.apothem
        self.width = 2 * self.side_length

    @staticmethod
    @njit
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

    @staticmethod
    @njit
    def _get_edges(vertices: np.ndarray) -> np.ndarray:
        """Get edges from vertices."""
        edges = np.empty((len(vertices), 2))
        for i in range(len(vertices)):
            edges[i] = vertices[i] - vertices[(i+1) % len(vertices)]
        return edges

    @staticmethod
    @njit
    def _project_polygon_onto_axis(vertices: np.ndarray, axis: np.ndarray) -> tuple:
        """Project polygon vertices onto an axis."""
        projections = np.empty(len(vertices))
        for i in range(len(vertices)):
            projections[i] = vertices[i, 0] * axis[0] + vertices[i, 1] * axis[1]
        return np.min(projections), np.max(projections)

    @staticmethod
    @njit
    def _hexagon_overlap_sat_jit(hex1_vertices: np.ndarray, hex2_vertices: np.ndarray) -> bool:
        """Separating Axis Theorem for hexagon overlap detection."""
        # Get edges of both hexagons
        edges1 = HexagonGeometry._get_edges(hex1_vertices)
        edges2 = HexagonGeometry._get_edges(hex2_vertices)

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
            min1, max1 = HexagonGeometry._project_polygon_onto_axis(hex1_vertices, axis)
            min2, max2 = HexagonGeometry._project_polygon_onto_axis(hex2_vertices, axis)

            # If no overlap on this axis, polygons don't overlap
            if max1 < min2 or max2 < min1:
                return False

        return True

    @staticmethod
    @njit
    def _point_in_hexagon_jit(point: np.ndarray, hex_vertices: np.ndarray) -> bool:
        """Fast point-in-polygon test for hexagon."""
        x, y = point[0], point[1]
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

    def vertices(self, center_x: float, center_y: float, rotation_deg: float) -> np.ndarray:
        """Get hexagon vertices."""
        return self.vertices_jit(center_x, center_y, rotation_deg)

>>>>>>> REPLACE
</DIFF>


<DIFF>
<<<<<<< SEARCH
class SymmetricHexagonGrid:
    """Symmetry-aware spatial hash grid for efficient neighbor search."""

    def __init__(self, cell_size: float = 3.0):
        self.cell_size = cell_size
        self.grid = defaultdict(list)

    @staticmethod
    @njit
    def _hash_cell_numba(x: float, y: float, cell_size: float) -> tuple:
        """Hash coordinates to grid cell."""
        return (int(x // cell_size), int(y // cell_size))

    def _hash_cell(self, x: float, y: float) -> tuple:
        """Hash coordinates to grid cell."""
        return self._hash_cell_numba(x, y, self.cell_size)

    def insert(self, hex_id: int, center_x: float, center_y: float):
        """Insert a hexagon into the spatial grid."""
        # Clamp coordinates to prevent extreme values in hash
        clamped_x = np.clip(center_x, -1000.0, 1000.0)
        clamped_y = np.clip(center_y, -1000.0, 1000.0)
        cell = self._hash_cell(clamped_x, clamped_y)
        self.grid[cell].append(hex_id)

    def get_candidates(self, center_x: float, center_y: float) -> list:
        """Get candidate hexagons that might collide with given position."""
        # Clamp coordinates to prevent extreme values in hash
        clamped_x = np.clip(center_x, -1000.0, 1000.0)
        clamped_y = np.clip(center_y, -1000.0, 1000.0)
        cell = self._hash_cell(clamped_x, clamped_y)
        candidates = []

        # Check the cell and its 8 neighboring cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell[0] + dx, cell[1] + dy)
                candidates.extend(self.grid[neighbor_cell])

        return candidates

    def clear(self):
        """Clear the spatial hash grid."""
        self.grid.clear()
=======
class SymmetricHexagonGrid:
    """Symmetry-aware spatial hash grid for efficient neighbor search."""

    def __init__(self, cell_size: float = 3.0):
        self.cell_size = cell_size
        self.grid = defaultdict(list)

    @staticmethod
    @njit
    def _hash_cell_numba(x: float, y: float, cell_size: float) -> tuple:
        """Hash coordinates to grid cell."""
        return (int(x // cell_size), int(y // cell_size))

    @staticmethod
    @njit
    def _clamp_coordinates(x: float, y: float) -> tuple:
        """Clamp coordinates to prevent extreme values in hash."""
        clamped_x = np.clip(x, -1000.0, 1000.0)
        clamped_y = np.clip(y, -1000.0, 1000.0)
        return clamped_x, clamped_y

    def _hash_cell(self, x: float, y: float) -> tuple:
        """Hash coordinates to grid cell."""
        return self._hash_cell_numba(x, y, self.cell_size)

    def insert(self, hex_id: int, center_x: float, center_y: float):
        """Insert a hexagon into the spatial grid."""
        # Clamp coordinates to prevent extreme values in hash
        clamped_x, clamped_y = self._clamp_coordinates(center_x, center_y)
        cell = self._hash_cell(clamped_x, clamped_y)
        self.grid[cell].append(hex_id)

    def get_candidates(self, center_x: float, center_y: float) -> list:
        """Get candidate hexagons that might collide with given position."""
        # Clamp coordinates to prevent extreme values in hash
        clamped_x, clamped_y = self._clamp_coordinates(center_x, center_y)
        cell = self._hash_cell(clamped_x, clamped_y)
        candidates = []

        # Check the cell and its 8 neighboring cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell[0] + dx, cell[1] + dy)
                candidates.extend(self.grid[neighbor_cell])

        return candidates

    def clear(self):
        """Clear the spatial hash grid."""
        self.grid.clear()

@njit
def _point_in_hexagon_jit(point: np.ndarray, hex_vertices: np.ndarray) -> bool:
    """Fast point-in-polygon test for hexagon."""
    x, y = point[0], point[1]
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

class SymmetricHexagonGrid:
    """Symmetry-aware spatial hash grid for efficient neighbor search."""

    def __init__(self, cell_size: float = 3.0):
        self.cell_size = cell_size
        self.grid = defaultdict(list)

    @staticmethod
    @njit
    def _hash_cell_numba(x: float, y: float, cell_size: float) -> tuple:
        """Hash coordinates to grid cell."""
        return (int(x // cell_size), int(y // cell_size))

    def _hash_cell(self, x: float, y: float) -> tuple:
        """Hash coordinates to grid cell."""
        return self._hash_cell_numba(x, y, self.cell_size)

    def insert(self, hex_id: int, center_x: float, center_y: float):
        """Insert a hexagon into the spatial grid."""
        # Clamp coordinates to prevent extreme values in hash
        clamped_x = np.clip(center_x, -1000.0, 1000.0)
        clamped_y = np.clip(center_y, -1000.0, 1000.0)
        cell = self._hash_cell(clamped_x, clamped_y)
        self.grid[cell].append(hex_id)

    def get_candidates(self, center_x: float, center_y: float) -> list:
        """Get candidate hexagons that might collide with given position."""
        # Clamp coordinates to prevent extreme values in hash
        clamped_x = np.clip(center_x, -1000.0, 1000.0)
        clamped_y = np.clip(center_y, -1000.0, 1000.0)
        cell = self._hash_cell(clamped_x, clamped_y)
        candidates = []

        # Check the cell and its 8 neighboring cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell[0] + dx, cell[1] + dy)
                candidates.extend(self.grid[neighbor_cell])

        return candidates

    def clear(self):
        """Clear the spatial hash grid."""
        self.grid.clear()

class ConstraintValidator:
    """Robust constraint validation for hexagon packing."""

    def __init__(self, geometry: HexagonGeometry):
        self.geo = geometry
        # Precompute the approximate radius for spatial hashing
        self.hex_radius_approx = 1.0  # For unit hexagon

    def is_contained(self, hex_vertices: np.ndarray, outer_center_x: float,
                     outer_center_y: float, outer_side_length: float) -> bool:
        """Check if all hexagon vertices are within outer hexagon."""
        # For a regular hexagon, we can check distance from center
        dist_from_center = np.sqrt((hex_vertices[:, 0] - outer_center_x)**2 +
                                  (hex_vertices[:, 1] - outer_center_y)**2)
        # Maximum distance from center in a regular hexagon is side_length * sqrt(3)/2
        max_radius = outer_side_length * np.sqrt(3) / 2
        return np.all(dist_from_center <= max_radius)

    @staticmethod
    @njit
    def _fast_contains_check_jit(hex_vertices: np.ndarray, outer_center_x: float,
                                outer_center_y: float, outer_side_length: float) -> bool:
        """Fast JIT compiled containment check."""
        # For a regular hexagon, we can check distance from center
        # Maximum distance from center in a regular hexagon is side_length * sqrt(3)/2
        max_radius = outer_side_length * np.sqrt(3) / 2
        for i in range(len(hex_vertices)):
            dist_sq = (hex_vertices[i, 0] - outer_center_x)**2 + (hex_vertices[i, 1] - outer_center_y)**2
            if dist_sq > max_radius**2:
                return False
        return True

    def has_overlap_fast(self, hex1_vertices: np.ndarray, hex2_vertices: np.ndarray) -> bool:
        """Fast approximate overlap check before precise check."""
        # Simple distance check first
        center1 = np.mean(hex1_vertices, axis=0)
        center2 = np.mean(hex2_vertices, axis=0)
        dist_centers = np.sqrt(np.sum((center1 - center2)**2))

        # If centers are far apart, no overlap
        if dist_centers > 2.0:  # Approximate sum of radii for unit hexagons
            return False

        # Otherwise, use SAT for precise check
        return HexagonGeometry._hexagon_overlap_sat_jit(hex1_vertices, hex2_vertices)

    def has_overlap_with_spatial_hash(self, hex_data: np.ndarray, spatial_grid: SymmetricHexagonGrid) -> bool:
        """Fast overlap detection using spatial hashing."""
        # Clear existing grid entries
        spatial_grid.clear()

        num_hex = len(hex_data)
        if num_hex <= 1:
            return False

        # Insert all hexagons into spatial grid
        for i in range(num_hex):
            center_x, center_y, _ = hex_data[i]
            spatial_grid.insert(i, center_x, center_y)

        # Check for overlaps with nearby hexagons only
        for i in range(num_hex):
            center_x, center_y, _ = hex_data[i]
            candidates = spatial_grid.get_candidates(center_x, center_y)

            # Check against candidates in neighboring cells
            for j in candidates:
                if i >= j:  # Avoid duplicate comparisons
                    continue

                vertices1 = self.geo.vertices(*hex_data[i])
                vertices2 = self.geo.vertices(*hex_data[j])

                if self.has_overlap_fast(vertices1, vertices2):
                    return True

        return False

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
        # Create spatial grid for faster neighbor search
        spatial_grid = SymmetricHexagonGrid(cell_size=2.0)
        if self.validator.has_overlap_with_spatial_hash(inner_hex_data, spatial_grid):
            # Apply higher penalty for overlaps with adaptive weighting
            penalty += 50000.0

        # Return negative inverse side length plus penalty
        objective_value = -1.0 / min_side_length + penalty
        return objective_value

class PackingOptimizer:
    """Main optimization controller with structured approach."""

    def __init__(self, num_hexagons: int = 12):
        self.num_hexagons = num_hexagons
        self.geometry = HexagonGeometry()
        self.validator = ConstraintValidator(self.geometry)
        self.evaluator = SolutionEvaluator(self.geometry, self.validator)

    def _generate_symmetric_initial_solution(self) -> np.ndarray:
        """Generate an initial symmetric solution with specific group theory constraints."""
        # Start with a highly symmetric arrangement - 12 hexagons arranged in a pattern
        # that respects rotational and reflectional symmetries
        base_positions = [
            [0, 0],           # Center (1st hex)
            [-2.1, 0],        # Left (2nd hex)
            [2.1, 0],         # Right (3rd hex)
            [-1.05, 1.82],    # Top-left (4th hex)
            [1.05, 1.82],     # Top-right (5th hex)
            [-1.05, -1.82],   # Bottom-left (6th hex)
            [1.05, -1.82],    # Bottom-right (7th hex)
            [-3.15, 1.82],    # Far top-left (8th hex)
            [3.15, 1.82],     # Far top-right (9th hex)
            [-3.15, -1.82],   # Far bottom-left (10th hex)
            [3.15, -1.82],    # Far bottom-right (11th hex)
            [0, -3.64],       # Far bottom (12th hex)
        ]

        # Special treatment for hexagon positions to maintain symmetry where possible
        # These are based on known good symmetric arrangements from literature

        # Flatten and add rotations with some symmetry considerations
        positions_with_angles = np.array(base_positions)

        # Apply symmetry-based rotations to leverage known symmetry patterns
        # Group 1: Central 1 hexagon - no rotation needed
        # Group 2: Opposite pairs - same rotation angles
        # Group 3: Diagonal pairs - same rotation angles
        rotations = np.array([
            0,   # center - no rotation
            0,   # left - fixed rotation
            0,   # right - fixed rotation
            0,   # top-left - fixed rotation
            0,   # top-right - fixed rotation
            0,   # bottom-left - fixed rotation
            0,   # bottom-right - fixed rotation
            0,   # far top-left - fixed rotation
            0,   # far top-right - fixed rotation
            0,   # far bottom-left - fixed rotation
            0,   # far bottom-right - fixed rotation
            0    # far bottom - fixed rotation
        ])

        positions_with_angles = np.column_stack([positions_with_angles, rotations])

        return positions_with_angles.flatten()

    def _symmetry_preserving_mutation(self, individual: np.ndarray,
                                    mutation_factor: float = 0.8) -> np.ndarray:
        """Apply mutation that preserves symmetry relationships between hexagons."""
        mutated = individual.copy()

        # Group hexagons by their symmetry relationships
        # Group 1: Center (index 0)
        # Group 2: Opposite pairs: (1,2), (3,4), (5,6)
        # Group 3: Diagonal pairs: (7,8), (9,10)
        # Group 4: Far bottom (index 11)

        # Mutate center hexagon
        mutated[0:3] += np.random.normal(0, mutation_factor * 0.1, 3)  # Small movement

        # Mutate opposite pairs together - keep them symmetrically positioned
        # Pair 1: (1,2) - left/right
        mutated[3:6] += np.random.normal(0, mutation_factor * 0.2, 3)
        mutated[6:9] = [mutated[3] * -1, mutated[4], mutated[5]]  # Mirror x-coordinate

        # Pair 2: (3,4) - top-left/top-right
        mutated[9:12] += np.random.normal(0, mutation_factor * 0.2, 3)
        mutated[12:15] = [mutated[9] * -1, mutated[10], mutated[11]]  # Mirror x-coordinate

        # Pair 3: (5,6) - bottom-left/bottom-right
        mutated[15:18] += np.random.normal(0, mutation_factor * 0.2, 3)
        mutated[18:21] = [mutated[15] * -1, mutated[16], mutated[17]]  # Mirror x-coordinate

        # Pair 4: (7,8) - far top-left/far top-right
        mutated[21:24] += np.random.normal(0, mutation_factor * 0.3, 3)
        mutated[24:27] = [mutated[21] * -1, mutated[22], mutated[23]]  # Mirror x-coordinate

        # Pair 5: (9,10) - far bottom-left/far bottom-right
        mutated[27:30] += np.random.normal(0, mutation_factor * 0.3, 3)
        mutated[30:33] = [mutated[27] * -1, mutated[28], mutated[29]]  # Mirror x-coordinate

        # Mutate far bottom hexagon
        mutated[33:36] += np.random.normal(0, mutation_factor * 0.1, 3)  # Small movement

        # Keep rotations consistent with symmetry
        # For symmetric groups, we'll apply similar rotations to maintain symmetry
        # Center: 0°
        # Left/Right: same rotation
        # Top-left/Top-right: same rotation
        # Bottom-left/Bottom-right: same rotation
        # Far top-left/Far top-right: same rotation
        # Far bottom-left/Far bottom-right: same rotation
        # Far bottom: 0°

        # Apply symmetry to rotations - group by symmetry class
        # Group 1: Center and far bottom - keep rotation as is
        # Group 2: Opposite pairs - same rotation
        # Group 3: Diagonal pairs - same rotation

        # Update rotations to maintain symmetry
        # Left/right pair
        mutated[5] = mutated[8] = mutated[2]  # Same rotation for left and right

        # Top-left/top-right pair
        mutated[11] = mutated[14] = mutated[10]  # Same rotation for top-left and top-right

        # Bottom-left/bottom-right pair
        mutated[17] = mutated[20] = mutated[16]  # Same rotation for bottom-left and bottom-right

        # Far top-left/far top-right pair
        mutated[23] = mutated[26] = mutated[22]  # Same rotation for far top-left and far top-right

        # Far bottom-left/far bottom-right pair
        mutated[29] = mutated[32] = mutated[28]  # Same rotation for far bottom-left and far bottom-right

        return mutated

    def _setup_bounds(self) -> list:
        """Setup parameter bounds for optimization."""
        bounds = []
        for i in range(self.num_hexagons):
            # X and Y positions - wider range to allow for symmetry breaking
            bounds.extend([(-10.0, 10.0), (-10.0, 10.0)])
            # Rotation: 0-360 degrees (but we'll use symmetry constraints)
            bounds.append((0.0, 360.0))
        return bounds

    def _optimize_stage(self, bounds: list, maxiter: int, popsize: int,
                       mutation: tuple, recombination: float,
                       initial_population: list = None) -> dict:
        """Run optimization stage."""
        try:
            # Create a custom mutation function that preserves symmetry
            def symmetry_preserving_mutation(func, indv, f, cr):
                return self._symmetry_preserving_mutation(indv, f)

            result = differential_evolution(
                self.evaluator.evaluate,
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                mutation=mutation,
                recombination=recombination,
                seed=42,
                disp=False,
                init=initial_population,
                callback=lambda x, convergence: None  # Suppress output
            )
            return result
        except Exception as e:
            warnings.warn(f"Optimization stage failed: {e}")
            return None

    def optimize(self) -> tuple:
        """Main optimization routine."""
        start_time = time.time()

        # Setup bounds
        bounds = self._setup_bounds()

        # Stage 1: Coarse optimization with symmetry-preserving initialization
        print("Starting stage 1: Coarse optimization...")
        initial_solution = self._generate_symmetric_initial_solution()

        # Generate population for better initialization with symmetry awareness
        initial_pop = [initial_solution]
        for _ in range(9):
            # Add perturbed versions that maintain some symmetries
            perturbed = initial_solution + np.random.normal(0, 0.5, len(initial_solution))
            # Ensure some rotation constraints are maintained
            initial_pop.append(perturbed)

        coarse_result = self._optimize_stage(
            bounds, maxiter=60, popsize=20,
            mutation=(0.8, 1.0), recombination=0.7,
            initial_population=initial_pop
        )

        if coarse_result is None or not coarse_result.success:
            warnings.warn("Coarse optimization failed, using initial solution")
            best_solution = initial_solution
        else:
            best_solution = coarse_result.x

        # Stage 2: Fine optimization with symmetry preservation
        print("Starting stage 2: Fine optimization...")
        fine_result = self._optimize_stage(
            bounds, maxiter=120, popsize=30,
            mutation=(1.0, 1.0), recombination=0.8,
            initial_population=[best_solution] + [np.random.normal(best_solution, 0.8) for _ in range(29)]
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

def hexagon_packing_12():
    """
    Constructs an optimized packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
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
        # Fallback to better symmetric arrangement
        inner_hex_data = np.array([
            [0, 0, 0],
            [-2.1, 0, 0],
            [2.1, 0, 0],
            [-1.05, 1.82, 0],
            [1.05, 1.82, 0],
            [-1.05, -1.82, 0],
            [1.05, -1.82, 0],
            [-3.15, 1.82, 0],
            [3.15, 1.82, 0],
            [-3.15, -1.82, 0],
            [3.15, -1.82, 0],
            [0, -3.64, 0],
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 7.5
        return inner_hex_data, outer_hex_data, outer_hex_side_length

# Global variable for timing
start_time = time.time()

# EVOLVE-BLOCK-END