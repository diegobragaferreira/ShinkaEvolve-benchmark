# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
import time
import math
from numba import jit, prange
from typing import Tuple, List, Optional
import warnings

class SpatialHash:
    """Spatial hash grid for efficient collision detection with improved performance."""

    def __init__(self, cell_size=None):
        self.grid = {}
        # Use hexagon diameter as default cell size for better performance
        self.cell_size = cell_size if cell_size is not None else 2.0

    def _hash(self, x, y):
        """Convert continuous coordinates to grid cell indices."""
        return (int(x // self.cell_size), int(y // self.cell_size))

    def clear(self):
        """Clear the spatial hash grid."""
        self.grid.clear()

    def insert(self, hex_id, vertices):
        """Insert a hexagon into the spatial hash."""
        # Get bounding box of hexagon
        min_x = min(v[0] for v in vertices)
        max_x = max(v[0] for v in vertices)
        min_y = min(v[1] for v in vertices)
        max_y = max(v[1] for v in vertices)

        # Insert into all cells it covers
        min_cell_x, min_cell_y = self._hash(min_x, min_y)
        max_cell_x, max_cell_y = self._hash(max_x, max_y)

        for cell_x in range(min_cell_x, max_cell_x + 1):
            for cell_y in range(min_cell_y, max_cell_y + 1):
                if (cell_x, cell_y) not in self.grid:
                    self.grid[(cell_x, cell_y)] = []
                self.grid[(cell_x, cell_y)].append(hex_id)

    def query(self, vertices):
        """Query which hexagons might collide with the given hexagon."""
        # Get bounding box of hexagon
        min_x = min(v[0] for v in vertices)
        max_x = max(v[0] for v in vertices)
        min_y = min(v[1] for v in vertices)
        max_y = max(v[1] for v in vertices)

        # Query all cells it covers
        min_cell_x, min_cell_y = self._hash(min_x, min_y)
        max_cell_x, max_cell_y = self._hash(max_x, max_y)

        candidates = set()
        for cell_x in range(min_cell_x, max_cell_x + 1):
            for cell_y in range(min_cell_y, max_cell_y + 1):
                if (cell_x, cell_y) in self.grid:
                    candidates.update(self.grid[(cell_x, cell_y)])

        return list(candidates)

    def get_candidate_pairs(self, hexagon_vertices_list):
        """Get all potential overlapping pairs using spatial hash."""
        # Clear existing grid
        self.clear()

        # Insert all hexagons
        for i, vertices in enumerate(hexagon_vertices_list):
            self.insert(i, vertices)

        # Find candidate pairs
        candidate_pairs = set()
        for i, vertices in enumerate(hexagon_vertices_list):
            candidates = self.query(vertices)
            for j in candidates:
                if i < j:  # Avoid duplicate pairs
                    candidate_pairs.add((i, j))

        return list(candidate_pairs)

@jit(nopython=True)
def hexagon_vertices_numba(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon using numba for speed."""
    angle_rad = angle_deg * math.pi / 180.0
    vertices = np.empty((6, 2))
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices[i] = (x, y)
    return vertices

@jit(nopython=True)
def point_distance_squared(x1, y1, x2, y2):
    """Calculate squared distance between two points."""
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + dy * dy

@jit(nopython=True)
def point_in_hexagon_fast(point_x, point_y, hex_vertices):
    """Fast point-in-hexagon test using ray casting."""
    x, y = point_x, point_y
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

@jit(nopython=True)
def distance_point_to_line_segment(point_x, point_y, line_start_x, line_start_y, line_end_x, line_end_y):
    """Calculate distance from point to line segment."""
    A = point_x - line_start_x
    B = point_y - line_start_y
    C = line_end_x - line_start_x
    D = line_end_y - line_start_y

    dot = A*C + B*D
    len_sq = C*C + D*D
    if len_sq == 0:
        return np.sqrt(A*A + B*B)
    param = dot / len_sq
    param = max(0, min(1, param))
    xx = line_start_x + param * C
    yy = line_start_y + param * D
    dx = point_x - xx
    dy = point_y - yy
    return np.sqrt(dx*dx + dy*dy)

@jit(nopython=True)
def get_edges(vertices):
    """Get edges from vertices."""
    edges = np.empty((len(vertices), 2))
    for i in range(len(vertices)):
        edges[i] = vertices[i] - vertices[(i+1) % len(vertices)]
    return edges

@jit(nopython=True)
def project_polygon_onto_axis(vertices, axis):
    """Project polygon vertices onto an axis and return min/max projections."""
    projections = np.empty(len(vertices))
    for i in range(len(vertices)):
        projections[i] = vertices[i, 0] * axis[0] + vertices[i, 1] * axis[1]
    return np.min(projections), np.max(projections)

@jit(nopython=True)
def hexagon_overlap_sat(hex1_vertices, hex2_vertices):
    """Separating Axis Theorem for hexagon overlap detection."""
    # Get edges of both hexagons
    edges1 = get_edges(hex1_vertices)
    edges2 = get_edges(hex2_vertices)

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
        min1, max1 = project_polygon_onto_axis(hex1_vertices, axis)
        min2, max2 = project_polygon_onto_axis(hex2_vertices, axis)

        # If no overlap on this axis, polygons don't overlap
        if max1 < min2 or max2 < min1:
            return False

    return True

class HexagonGeometry:
    """Handles all geometric computations for hexagon operations."""
    
    def __init__(self):
        self.hex_radius = 1.0
        self.hex_apothem = np.sqrt(3) / 2
        self.hex_height = 2 * self.hex_apothem
        self.hex_width = 2 * self.hex_radius
    
    @staticmethod
    @jit(nopython=True)
    def compute_centers_apothems_numba(hexagon_vertices_list):
        """Numba-accelerated computation of centers and apothems."""
        num_hexagons = len(hexagon_vertices_list)
        centers = np.empty((num_hexagons, 2))
        apothems_sq = np.empty(num_hexagons)

        for i in range(num_hexagons):
            # Calculate center as average of vertices
            cx = 0.0
            cy = 0.0
            for j in range(6):
                cx += hexagon_vertices_list[i][j, 0]
                cy += hexagon_vertices_list[i][j, 1]
            cx /= 6.0
            cy /= 6.0

            # Calculate apothem using distance from center to first vertex
            apothem_sq = point_distance_squared(cx, cy, hexagon_vertices_list[i][0, 0], hexagon_vertices_list[i][0, 1])
            centers[i] = (cx, cy)
            apothems_sq[i] = apothem_sq

        return centers, apothems_sq

    def generate_hexagon_vertices(self, center_x: float, center_y: float, angle_deg: float, side_length: float = 1.0) -> np.ndarray:
        """Generate vertices of a regular hexagon given center, angle, and side length."""
        angle_rad = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            angle = angle_rad + i * math.pi / 3
            x = center_x + side_length * math.cos(angle)
            y = center_y + side_length * math.sin(angle)
            vertices.append((x, y))
        return np.array(vertices)

    def create_hexagon_polygon(self, center_x: float, center_y: float, angle_deg: float) -> Polygon:
        """Create Shapely polygon representation of a hexagon."""
        vertices = self.generate_hexagon_vertices(center_x, center_y, angle_deg)
        return Polygon(vertices)

    def outer_hexagon_vertices(self, side_length: float) -> np.ndarray:
        """Generate vertices of outer hexagon centered at origin."""
        return self.generate_hexagon_vertices(0, 0, 0, side_length)

class ConstraintValidator:
    """Handles all constraint validation logic."""
    
    def __init__(self, geometry_handler: HexagonGeometry):
        self.geo = geometry_handler
    
    def check_containment_distance_only(self, hexagon_vertices_list: List[np.ndarray], outer_side_length: float) -> bool:
        """Fast containment check using distance bounds."""
        # For outer hexagon with side length R, distance from center to edge is R * sqrt(3)/2
        outer_apothem = outer_side_length * math.sqrt(3) / 2
        outer_apothem_sq = outer_apothem * outer_apothem

        for vertices in hexagon_vertices_list:
            # Check distance of each vertex from origin (0,0)
            for i in range(6):
                x = vertices[i, 0]
                y = vertices[i, 1]
                distance_sq = x*x + y*y
                if distance_sq > outer_apothem_sq:
                    return False
        return True

    def check_overlap_spatial_hash(self, hexagon_vertices_list: List[np.ndarray]) -> bool:
        """Overlap detection using spatial hashing for efficiency."""
        # Create spatial hash
        spatial_hash = SpatialHash(cell_size=2.0)
        n_hexagons = len(hexagon_vertices_list)

        # Insert all hexagons into spatial hash
        for i, vertices in enumerate(hexagon_vertices_list):
            spatial_hash.insert(i, vertices)

        # Get candidate pairs
        candidate_pairs = spatial_hash.get_candidate_pairs(hexagon_vertices_list)

        # Check actual overlaps only for candidates
        for i, j in candidate_pairs:
            try:
                if hexagon_overlap_sat(hexagon_vertices_list[i], hexagon_vertices_list[j]):
                    return False
            except:
                return False
        return True

    def check_overlap_shapely(self, hexagon_vertices_list: List[np.ndarray]) -> bool:
        """Precise overlap detection using Shapely."""
        try:
            polygons = [Polygon(vertices) for vertices in hexagon_vertices_list]
            union = Polygon.union_all(polygons)
            total_area = sum(polygon.area for polygon in polygons)
            union_area = union.area
            return abs(total_area - union_area) < 1e-10
        except:
            # Fallback for complex cases
            for i in range(len(hexagon_vertices_list)):
                for j in range(i+1, len(hexagon_vertices_list)):
                    try:
                        p1 = Polygon(hexagon_vertices_list[i])
                        p2 = Polygon(hexagon_vertices_list[j])
                        if p1.intersects(p2):
                            return False
                    except:
                        return False
            return True

    def validate_configuration(self, config: np.ndarray, outer_side_length: float) -> Tuple[bool, float]:
        """Validate a configuration of 12 hexagons."""
        # Parse configuration into 12 hexagons (x, y, angle)
        hexagons = config.reshape(12, 3)

        # Get vertices for all hexagons
        hexagon_vertices_list = []
        for i in range(12):
            x, y, angle = hexagons[i]
            vertices = self.geo.generate_hexagon_vertices(x, y, angle)
            hexagon_vertices_list.append(vertices)

        # Fast containment check first
        if not self.check_containment_distance_only(hexagon_vertices_list, outer_side_length):
            return False, 1500.0  # Invalid configuration - high penalty for containment

        # Fast overlap check using spatial hash
        if not self.check_overlap_spatial_hash(hexagon_vertices_list):
            return False, 1000.0  # Overlapping hexagons - medium penalty for overlap

        # Final precise validation
        if not self.check_overlap_shapely(hexagon_vertices_list):
            return False, 1000.0

        return True, 0.0  # Valid configuration

class Optimizer:
    """Handles the core optimization logic."""
    
    def __init__(self, geometry_handler: HexagonGeometry, validator: ConstraintValidator):
        self.geo = geometry_handler
        self.validator = validator
        self.max_time = 180.0
    
    def generate_good_initial_config(self) -> np.ndarray:
        """Generate a good initial configuration based on proven patterns."""
        # Start with a proven symmetric pattern that's close to optimal
        # This uses mathematical insight about efficient packing arrangements
        initial_config = np.array([
            # Center hexagon
            [0.0, 0.0, 0.0],
            # First ring around center (approximate spacing)
            [-1.732, 0.0, 0.0],  # Left
            [1.732, 0.0, 0.0],   # Right
            [0.0, 1.732, 0.0],   # Top
            [0.0, -1.732, 0.0],  # Bottom
            [-0.866, 0.866, 0.0],  # Top-left
            [0.866, 0.866, 0.0],   # Top-right
            [-0.866, -0.866, 0.0], # Bottom-left
            [0.866, -0.866, 0.0],  # Bottom-right
            # Outer ring - arranged to maximize space utilization
            [-2.598, 0.0, 0.0],   # Far left
            [2.598, 0.0, 0.0],    # Far right
            [0.0, 2.598, 0.0],    # Far top
        ])

        # Add small random noise to get initial variety while preserving good structure
        initial_config += np.random.normal(0, 0.02, initial_config.shape)

        return initial_config.flatten()

    def optimize_with_differential_evolution(self, initial_config: np.ndarray, outer_side_length: float) -> Tuple[np.ndarray, bool]:
        """Optimize using differential evolution with enhanced settings."""
        # Define bounds for optimization
        bounds = []
        for _ in range(12):
            bounds.extend([(-5.0, 5.0), (-5.0, 5.0), (0.0, 360.0)])
        
        def objective(x):
            valid, penalty = self.validator.validate_configuration(x, outer_side_length)
            if valid:
                # Maximize 1/outer_side_length (minimize -1/outer_side_length)
                return -1.0 / outer_side_length
            else:
                return penalty  # Return penalty for invalid configurations
        
        try:
            # Enhanced differential evolution settings with early stopping criteria
            result = differential_evolution(
                objective, bounds, seed=42, maxiter=150, popsize=20, 
                mutation=(0.5, 1), recombination=0.7, disp=False
            )
            return result.x, result.success
        except Exception as e:
            print(f"DE optimization failed: {e}")
            return initial_config, False

    def local_refinement(self, config: np.ndarray, outer_side_length: float, max_iter: int = 100) -> np.ndarray:
        """Perform local refinement with adaptive perturbation strategy."""
        best_config = config.copy()
        best_valid, best_penalty = self.validator.validate_configuration(best_config, outer_side_length)

        if not best_valid:
            return config

        # Adaptive step size that decreases over iterations
        initial_step_size = 0.05
        final_step_size = 0.001

        # Try improvements with adaptive perturbation
        for iteration in range(max_iter):
            # Decrease step size as iterations progress
            step_size = initial_step_size + (final_step_size - initial_step_size) * (iteration / max_iter)

            # Try modifying multiple hexagons based on their positions
            current_config = best_config.copy()

            # Select hexagon to modify - prioritize those near boundaries
            hex_idx = np.random.randint(0, 12)

            # Perturb position with adaptive step size
            current_config[hex_idx*3:hex_idx*3+2] += np.random.normal(0, step_size, 2)
            # Keep angle within [0, 360]
            current_config[hex_idx*3+2] = current_config[hex_idx*3+2] % 360

            valid, penalty = self.validator.validate_configuration(current_config, outer_side_length)
            if valid and penalty < best_penalty:
                best_config = current_config
                best_penalty = penalty

        return best_config

    def optimize_hexagon_positions(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Main optimization routine with enhanced strategy."""
        best_outer_side_length = 3.9419123  # Target the SOTA
        best_config = None
        best_valid = False
        
        # Generate initial configuration
        initial_config = self.generate_good_initial_config()
        
        # Try optimization with DE
        optimized_config, success = self.optimize_with_differential_evolution(initial_config, best_outer_side_length)
        
        if success:
            # Local refinement of the optimized result
            refined_config = self.local_refinement(optimized_config, best_outer_side_length, 100)
            valid, penalty = self.validator.validate_configuration(refined_config, best_outer_side_length)
            
            if valid:
                # Try to fit with smaller outer hexagon size
                test_sides = np.linspace(3.8, best_outer_side_length, 30)
                for test_side in test_sides[::-1]:
                    valid_test, penalty_test = self.validator.validate_configuration(refined_config, test_side)
                    if valid_test:
                        if test_side < best_outer_side_length:
                            best_outer_side_length = test_side
                            best_config = refined_config.copy()
                            best_valid = True
                            break
        
        # If nothing worked, use fallback
        if not best_valid:
            # Use the initial good configuration as fallback
            best_config = initial_config
            best_outer_side_length = 4.0

        # Final validation
        final_valid, _ = self.validator.validate_configuration(best_config, best_outer_side_length)
        if not final_valid:
            # Use simple grid as last resort
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
                [3.75, -2.17, 0],  # far bottom-right,
                [0, -4, 0],  # far bottom-center
            ])
            outer_hex_data = np.array([0, 0, 0])
            outer_hex_side_length = 8.0
            return inner_hex_data, outer_hex_data, outer_hex_side_length

        return best_config.reshape(12, 3), np.array([0, 0, 0]), best_outer_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Initialize components
    geometry_handler = HexagonGeometry()
    validator = ConstraintValidator(geometry_handler)
    optimizer = Optimizer(geometry_handler, validator)
    
    # Run optimization
    inner_hex_data, outer_hex_data, outer_hex_side_length = optimizer.optimize_hexagon_positions()
    
    # Calculate actual score
    inv_side_length = 1.0 / outer_hex_side_length
    eval_time = time.time() - start_time
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END