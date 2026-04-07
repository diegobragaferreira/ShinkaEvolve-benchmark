# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import time
from numba import jit, prange
import warnings
from shapely.geometry import Polygon
from shapely.ops import unary_union

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

class SpatialHashGrid:
    """Efficient spatial hash grid for neighbor searches."""
    
    def __init__(self, cell_size: float = 2.0):
        self.cell_size = cell_size
        self.grid = {}

    def _hash_cell(self, x: float, y: float) -> tuple:
        """Hash coordinates to grid cell."""
        return (int(x // self.cell_size), int(y // self.cell_size))

    def insert(self, hex_id: int, center_x: float, center_y: float):
        """Insert a hexagon into the spatial grid."""
        cell = self._hash_cell(center_x, center_y)
        if cell not in self.grid:
            self.grid[cell] = []
        self.grid[cell].append(hex_id)

    def get_candidates(self, center_x: float, center_y: float) -> list:
        """Get candidate hexagons that might collide with given position."""
        cell = self._hash_cell(center_x, center_y)
        candidates = []

        # Check the cell and its 8 neighboring cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell[0] + dx, cell[1] + dy)
                if neighbor_cell in self.grid:
                    candidates.extend(self.grid[neighbor_cell])

        return candidates

class HexagonValidator:
    """Robust constraint validation for hexagon packing."""
    
    def __init__(self, geometry: HexagonGeometry):
        self.geo = geometry
        
    @staticmethod
    @jit(nopython=True)
    def _distance_point_to_segment(point, seg_start, seg_end):
        """Fast distance from point to line segment."""
        px, py = point
        x1, y1 = seg_start
        x2, y2 = seg_end

        # Vector from start to end
        dx, dy = x2 - x1, y2 - y1
        # Vector from start to point
        px_minus_x1, py_minus_y1 = px - x1, py - y1

        # Project point onto line
        length_sq = dx*dx + dy*dy
        if length_sq == 0:
            return np.sqrt(px_minus_x1*px_minus_x1 + py_minus_y1*py_minus_y1)

        t = (px_minus_x1*dx + py_minus_y1*dy) / length_sq
        t = max(0, min(1, t))

        # Closest point on segment
        closest_x = x1 + t*dx
        closest_y = y1 + t*dy

        return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

    @staticmethod
    @jit(nopython=True)
    def _hexagon_distance_fast(hex1_vertices, hex2_vertices):
        """Fast computation of minimum distance between hexagons."""
        min_dist = np.inf
        for i in range(6):
            p1 = hex1_vertices[i]
            p2 = hex1_vertices[(i+1)%6]
            for j in range(6):
                q1 = hex2_vertices[j]
                q2 = hex2_vertices[(j+1)%6]
                dist = HexagonValidator._distance_point_to_segment(q1, p1, p2)
                min_dist = min(min_dist, dist)
        return min_dist

    def is_contained(self, hex_vertices: np.ndarray, outer_center_x: float,
                     outer_center_y: float, outer_side_length: float) -> bool:
        """Check if all hexagon vertices are within outer hexagon."""
        # For a regular hexagon, we can check distance from center
        dist_from_center = np.sqrt((hex_vertices[:, 0] - outer_center_x)**2 +
                                  (hex_vertices[:, 1] - outer_center_y)**2)
        # Maximum distance from center in a regular hexagon is side_length * sqrt(3)/2
        max_radius = outer_side_length * np.sqrt(3) / 2
        return np.all(dist_from_center <= max_radius)

    def has_overlap_fast(self, hex1_vertices: np.ndarray, hex2_vertices: np.ndarray) -> bool:
        """Fast approximate overlap check before precise check."""
        # Simple distance check first
        center1 = np.mean(hex1_vertices, axis=0)
        center2 = np.mean(hex2_vertices, axis=0)
        dist_centers = np.sqrt(np.sum((center1 - center2)**2))

        # If centers are far apart, no overlap
        if dist_centers > 2.0:  # Approximate sum of radii for unit hexagons
            return False

        # Otherwise, use fast distance calculation for precise check
        distance = self._hexagon_distance_fast(hex1_vertices, hex2_vertices)
        return distance < 1e-10  # Threshold for overlap detection

    def has_overlap_with_spatial_hash(self, hex_data: np.ndarray, spatial_grid: SpatialHashGrid) -> bool:
        """Fast overlap detection using spatial hashing."""
        num_hex = len(hex_data)
        
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

class OptimizerEngine:
    """Main optimization engine with structured approach."""
    
    def __init__(self):
        self.geometry = HexagonGeometry()
        self.validator = HexagonValidator(self.geometry)
        
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

        # Flatten and add rotations with some symmetry considerations
        positions_with_angles = np.array(base_positions)

        # Apply symmetry-based rotations to leverage known symmetry patterns
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

    def _calculate_min_enclosing_hexagon(self, inner_hex_data: np.ndarray,
                                       scale_factor: float = 1.05) -> tuple:
        """Calculate minimum enclosing hexagon side length."""
        # Get all vertices of all inner hexagons
        all_vertices = []
        for i in range(len(inner_hex_data)):
            center_x, center_y, angle = inner_hex_data[i]
            vertices = self.geometry.vertices(center_x, center_y, angle)
            all_vertices.extend(vertices)

        all_vertices = np.array(all_vertices)

        # Find bounding circle radius
        centroid = np.mean(all_vertices, axis=0)
        distances = np.sqrt(np.sum((all_vertices - centroid)**2, axis=1))
        max_distance = np.max(distances)

        # For a regular hexagon, side length = max_distance * sqrt(3)/2
        side_length = max_distance * 2 / np.sqrt(3) * scale_factor
        return side_length, centroid

    def _evaluate_solution(self, solution_array: np.ndarray, 
                          outer_side_length: float = 10.0) -> float:
        """Evaluate solution quality with penalty-based constraints."""
        # Reshape solution array into 12 hexagons with (x, y, angle) each
        inner_hex_data = solution_array.reshape(-1, 3)

        # Calculate the minimum enclosing hexagon
        min_side_length, centroid = self._calculate_min_enclosing_hexagon(inner_hex_data, 1.05)

        # Check all constraints
        num_hex = len(inner_hex_data)
        penalty = 0.0

        # Check containment for all hexagons
        for i in range(num_hex):
            center_x, center_y, angle = inner_hex_data[i]
            vertices = self.geometry.vertices(center_x, center_y, angle)

            # Check if hexagon is contained properly
            if not self.validator.is_contained(vertices, centroid[0], centroid[1], min_side_length):
                # Apply higher penalty for containment violations since they are fundamental
                penalty += 15000.0

        # Check overlaps using spatial hashing for efficiency
        # Create spatial grid for faster neighbor search
        spatial_grid = SpatialHashGrid(cell_size=2.0)
        if self.validator.has_overlap_with_spatial_hash(inner_hex_data, spatial_grid):
            # Apply higher penalty for overlaps with adaptive weighting
            penalty += 50000.0

        # Return negative inverse side length plus penalty
        objective_value = -1.0 / min_side_length + penalty
        return objective_value

    def optimize(self) -> tuple:
        """Main optimization routine."""
        # Setup bounds
        bounds = []
        for i in range(12):
            # X and Y positions - wider range to allow for symmetry breaking
            bounds.extend([(-10.0, 10.0), (-10.0, 10.0)])
            # Rotation: 0-360 degrees
            bounds.append((0.0, 360.0))

        # Stage 1: Coarse optimization with symmetry-preserving initialization
        initial_solution = self._generate_symmetric_initial_solution()

        # Generate population for better initialization with symmetry awareness
        initial_pop = [initial_solution]
        for _ in range(9):
            # Add perturbed versions that maintain some symmetries
            perturbed = initial_solution + np.random.normal(0, 0.5, len(initial_solution))
            initial_pop.append(perturbed)

        try:
            # Run coarse optimization
            coarse_result = differential_evolution(
                self._evaluate_solution,
                bounds,
                maxiter=50,
                popsize=15,
                mutation=(0.8, 1.0),
                recombination=0.7,
                seed=42,
                disp=False,
                init=initial_pop
            )

            if not coarse_result.success:
                warnings.warn("Coarse optimization failed, using initial solution")
                best_solution = initial_solution
            else:
                best_solution = coarse_result.x

            # Stage 2: Fine optimization
            fine_result = differential_evolution(
                self._evaluate_solution,
                bounds,
                maxiter=100,
                popsize=25,
                mutation=(1.0, 1.0),
                recombination=0.8,
                seed=43,
                disp=False,
                init=[best_solution] + [np.random.normal(best_solution, 0.8) for _ in range(24)]
            )

            if fine_result.success:
                final_solution = fine_result.x
            else:
                final_solution = best_solution

        except Exception as e:
            warnings.warn(f"Optimization failed: {e}")
            final_solution = initial_solution

        # Evaluate final solution
        inner_hex_data = final_solution.reshape(-1, 3)
        min_side_length, centroid = self._calculate_min_enclosing_hexagon(inner_hex_data, 1.05)

        # Center the outer hexagon at the centroid of inner hexagons
        outer_hex_data = np.array([centroid[0], centroid[1], 0])

        return inner_hex_data, outer_hex_data, min_side_length

def hexagon_packing_12():
    """
    Constructs an optimized packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    try:
        engine = OptimizerEngine()
        inner_hex_data, outer_hex_data, outer_hex_side_length = engine.optimize()

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

# EVOLVE-BLOCK-END