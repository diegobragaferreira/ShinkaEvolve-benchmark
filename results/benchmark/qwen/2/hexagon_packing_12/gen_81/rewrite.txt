# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from scipy.optimize import differential_evolution, minimize
import time
from typing import Tuple, List
import warnings
from numba import jit
import math

class Hexagon:
    """Represents a regular hexagon with side length 1"""

    def __init__(self, center_x: float, center_y: float, rotation_deg: float):
        self.center = np.array([center_x, center_y])
        self.rotation = np.radians(rotation_deg)
        self.side_length = 1.0

    def vertices(self) -> np.ndarray:
        """Return vertices of hexagon in counterclockwise order"""
        # Unit hexagon vertices centered at origin
        angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 vertices + close loop
        unit_vertices = np.column_stack([np.cos(angles), np.sin(angles)])

        # Apply rotation and translation
        cos_r, sin_r = np.cos(self.rotation), np.sin(self.rotation)
        rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])

        rotated_vertices = rotation_matrix @ unit_vertices.T
        translated_vertices = rotated_vertices.T + self.center

        return translated_vertices

    def area(self) -> float:
        """Return area of hexagon"""
        return (3 * np.sqrt(3) / 2) * self.side_length ** 2

    def bounding_box(self):
        """Return the axis-aligned bounding box (min_x, max_x, min_y, max_y)"""
        # For a unit hexagon with rotation, we compute its extreme coordinates
        vertices = self.vertices()
        mins = np.min(vertices, axis=0)
        maxs = np.max(vertices, axis=0)
        return (mins[0], maxs[0], mins[1], maxs[1])

def create_outer_hexagon(side_length: float, center_x: float = 0, center_y: float = 0) -> Polygon:
    """Create outer hexagon as Shapely polygon"""
    hex = Hexagon(center_x, center_y, 0)
    outer_vertices = hex.vertices()
    return Polygon(outer_vertices)

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

def compute_outer_hex_side_length(inner_hexagons: List[Hexagon]) -> float:
    """Compute minimum outer hexagon side length required to contain all inner hexagons"""
    # Get all vertices from all hexagons
    all_vertices = []
    for hex in inner_hexagons:
        all_vertices.extend(hex.vertices())

    all_vertices = np.array(all_vertices)

    # Find bounding circle center and radius
    center = np.mean(all_vertices, axis=0)

    # Calculate maximum distance from center to any vertex
    distances = np.linalg.norm(all_vertices - center, axis=1)
    max_distance = np.max(distances)

    # For a hexagon, we need side length >= max_distance * 2 / sqrt(3)
    side_length = max_distance * 2 / np.sqrt(3)

    return side_length

def check_containment(hexagon: Hexagon, outer_polygon: Polygon) -> bool:
    """Check if all vertices of hexagon are inside outer polygon"""
    vertices = hexagon.vertices()
    for vertex in vertices:
        point = Point(vertex[0], vertex[1])
        if not outer_polygon.contains(point):
            return False
    return True

# Spatial hashing for efficient overlap detection
class SpatialHash:
    """Spatial hash grid for efficient collision detection"""

    def __init__(self, cell_size=1.5):
        self.cell_size = cell_size
        self.grid = {}

    def _hash(self, x, y):
        """Hash coordinates to grid cell"""
        return (int(x // self.cell_size), int(y // self.cell_size))

    def insert(self, obj, bbox):
        """Insert an object with its bounding box into the spatial hash"""
        min_x, max_x, min_y, max_y = bbox
        x1, y1 = self._hash(min_x, min_y)
        x2, y2 = self._hash(max_x, max_y)

        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                if (x, y) not in self.grid:
                    self.grid[(x, y)] = []
                self.grid[(x, y)].append(obj)

    def query(self, bbox):
        """Query objects that might collide with the given bounding box"""
        min_x, max_x, min_y, max_y = bbox
        x1, y1 = self._hash(min_x, min_y)
        x2, y2 = self._hash(max_x, max_y)

        candidates = set()
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                if (x, y) in self.grid:
                    candidates.update(self.grid[(x, y)])

        return list(candidates)

def check_overlap_fast(hex1: Hexagon, hex2: Hexagon, spatial_hash=None):
    """Fast approximate overlap check using spatial hashing then precise Shapely check"""
    if spatial_hash:
        # Quick spatial hash check first
        bbox1 = hex1.bounding_box()
        bbox2 = hex2.bounding_box()

        # Check if bounding boxes intersect
        if (bbox1[1] < bbox2[0] or bbox2[1] < bbox1[0] or
            bbox1[3] < bbox2[2] or bbox2[3] < bbox1[2]):
            return False

        # If they could potentially overlap, check spatial hash
        # (This is an optimization - we still do the precise check below)
        pass

    # Precise overlap check using Shapely
    poly1 = Polygon(hex1.vertices())
    poly2 = Polygon(hex2.vertices())
    return poly1.intersects(poly2)

def check_overlap(hex1: Hexagon, hex2: Hexagon, spatial_hash=None):
    """Check if two hexagons overlap using either spatial hashing or direct method"""
    return check_overlap_fast(hex1, hex2, spatial_hash)

def validate_configuration(inner_hexagons: List[Hexagon], outer_polygon: Polygon, spatial_hash=None) -> Tuple[bool, bool]:
    """
    Validate configuration: check containment and overlap
    Returns (is_valid, has_overlaps)
    """
    # Check containment first (early termination)
    for hex in inner_hexagons:
        if not check_containment(hex, outer_polygon):
            return False, False

    # Check overlaps using spatial hashing for performance
    if spatial_hash is None:
        # Create spatial hash from all hexagon bounding boxes
        all_bboxes = [hex.bounding_box() for hex in inner_hexagons]
        spatial_hash = SpatialHash(cell_size=2.0)
        for i, bbox in enumerate(all_bboxes):
            spatial_hash.insert(i, bbox)

    # Use spatial hash for efficient overlap checking
    for i in range(len(inner_hexagons)):
        # Query neighbors from spatial hash
        hex1 = inner_hexagons[i]
        bbox1 = hex1.bounding_box()

        # Check nearby hexagons (using spatial hash)
        candidates = spatial_hash.query(bbox1)
        for j in candidates:
            if i != j:  # Don't compare with self
                hex2 = inner_hexagons[j]
                if check_overlap(hex1, hex2, spatial_hash):
                    return True, True

    return True, False

def calculate_objective(outer_side_length: float) -> float:
    """Calculate 1/outer_hex_side_length"""
    return 1.0 / outer_side_length

def generate_initial_population(n_individuals: int, n_hexagons: int = 12) -> List[np.ndarray]:
    """Generate initial population of valid configurations with better starting points"""
    population = []

    # Try multiple configurations with improved symmetry
    for _ in range(n_individuals):
        # Start with a better symmetric arrangement (inspired by known optimal patterns)
        positions = []
        rotations = []

        # Center hexagon
        positions.append([0.0, 0.0])
        rotations.append(0.0)

        # Ring 1: 6 hexagons arranged in a regular hexagon 
        ring1_radius = 2.1
        for i in range(6):
            angle = i * 60  # degrees
            x = ring1_radius * np.cos(np.radians(angle))
            y = ring1_radius * np.sin(np.radians(angle))
            positions.append([x, y])
            rotations.append(0.0)
            
        # Ring 2: 5 hexagons in a pentagonal arrangement
        ring2_radius = 3.6
        for i in range(5):
            angle = i * 72 + 18  # offset to create irregular but symmetric pattern
            x = ring2_radius * np.cos(np.radians(angle))
            y = ring2_radius * np.sin(np.radians(angle))
            positions.append([x, y])
            rotations.append(0.0)
            
        # Add some strategic rotations to break degenerate symmetries
        rotations[1] = 30   # First ring hexagon rotated
        rotations[2] = 15   # Second ring hexagon rotated
        
        # Add some random noise to make it diverse
        individual = np.array(positions + rotations).flatten()
        individual += np.random.normal(0, 0.2, len(individual))
        population.append(individual)
        
    return population

def fitness_function_adaptive(params: np.ndarray, outer_side_length: float = 10.0) -> float:
    """
    Fitness function to minimize negative of 1/outer_hex_side_length with adaptive penalties
    This assumes params contains [x1,y1,theta1,x2,y2,theta2,...]
    """
    n_hexagons = 12
    hex_params = params.reshape(-1, 3)

    # Create hexagons
    inner_hexagons = []
    for i in range(n_hexagons):
        x, y, theta = hex_params[i]
        inner_hexagons.append(Hexagon(x, y, theta))

    # Create outer hexagon of given size
    outer_polygon = create_outer_hexagon(outer_side_length)

    # Early validity check: containment check only (fast)
    for hex in inner_hexagons:
        vertices = hex.vertices()
        has_outside_point = False
        for vertex in vertices:
            point = Point(vertex[0], vertex[1])
            if not outer_polygon.contains(point):
                has_outside_point = True
                break
        if has_outside_point:
            # Heavy penalty for containment violations
            return 1e8

    # Validate complete configuration
    is_valid, has_overlaps = validate_configuration(inner_hexagons, outer_polygon)

    if not is_valid or has_overlaps:
        # Heavier penalties for overlap and containment violations
        return 1e7 + (1e6 if has_overlaps else 0)

    # Compute actual outer hexagon size needed
    actual_size = compute_outer_hex_side_length(inner_hexagons)

    if actual_size > outer_side_length:
        # Configuration requires larger outer hexagon
        return 1e7 + (actual_size - outer_side_length) * 10000
        
    # Return negative of 1/actual_size to maximize 1/size
    # Add a small penalty term to encourage tight packing
    return -1.0 / actual_size - 1e-10 * (actual_size - outer_side_length)**2

def optimize_packing_multistage() -> Tuple[np.ndarray, np.ndarray, float]:
    """Main optimization function with multi-stage approach"""
    start_time = time.time()

    # Stage 1: Coarse global search with differential evolution
    # Define bounds for optimization
    bounds = []
    for i in range(12):
        # X and Y positions: typically within reasonable bounds
        bounds.extend([(-8.0, 8.0), (-8.0, 8.0)])  # Positions
        # Rotation: 0-360 degrees
        bounds.append((0, 360))  # Rotations

    # Better initial guess with mathematical insight
    initial_guess = [
        [0.0, 0.0, 0.0],      # center
        [0.0, 2.1, 0.0],      # up
        [0.0, -2.1, 0.0],     # down
        [1.8, 1.0, 0.0],      # upper right
        [-1.8, 1.0, 0.0],     # upper left
        [1.8, -1.0, 0.0],     # lower right
        [-1.8, -1.0, 0.0],    # lower left
        [3.6, 0.0, 0.0],      # far right
        [0.0, 3.6, 0.0],      # far up
        [0.0, -3.6, 0.0],     # far down
        [2.1, 2.1, 0.0],      # upper right diagonal
        [-2.1, 2.1, 0.0],     # upper left diagonal
    ]

    initial_guess = np.array(initial_guess).flatten()

    # First stage: Global search with reduced complexity
    try:
        # Use fewer iterations for quick initial result
        result_stage1 = differential_evolution(
            lambda x: fitness_function_adaptive(x, 8.0),
            bounds,
            maxiter=30,  # Reduced iterations for speed
            popsize=8,   # Smaller population for efficiency
            init='random',
            seed=42,
            disp=False,
            tol=1e-5
        )

        # Extract stage 1 solution
        stage1_params = result_stage1.x if result_stage1.success else initial_guess
        stage1_hex_params = stage1_params.reshape(-1, 3)
        
        # Create hexagons from stage 1 parameters
        stage1_hexagons = []
        for i in range(12):
            x, y, theta = stage1_hex_params[i]
            stage1_hexagons.append(Hexagon(x, y, theta))
            
        # Compute stage 1 outer hexagon size
        stage1_outer_size = compute_outer_hex_side_length(stage1_hexagons)
        
        # Stage 2: Local refinement using L-BFGS-B
        # Prepare bounds for local optimization (tighter constraints)
        local_bounds = [(max(-6, x - 0.5), min(6, x + 0.5)) for x in stage1_params[::3]]
        local_bounds.extend([(max(-6, y - 0.5), min(6, y + 0.5)) for y in stage1_params[1::3]])
        local_bounds.extend([(0, 360)] * 12)  # rotations: 0-360
        
        def local_objective(params):
            return fitness_function_adaptive(params, stage1_outer_size + 0.5)
            
        # Use L-BFGS-B for final local refinement
        local_result = minimize(
            local_objective,
            stage1_params,
            method='L-BFGS-B',
            bounds=[(b[0], b[1]) for b in local_bounds],
            options={'maxiter': 50, 'ftol': 1e-6},
            tol=1e-6
        )
        
        final_params = local_result.x if local_result.success else stage1_params
        final_hex_params = final_params.reshape(-1, 3)
        
        # Create hexagons from final parameters
        final_hexagons = []
        for i in range(12):
            x, y, theta = final_hex_params[i]
            final_hexagons.append(Hexagon(x, y, theta))
            
        # Compute actual outer hexagon size
        outer_side_length = compute_outer_hex_side_length(final_hexagons)
        
        # Convert back to expected format
        inner_hex_data = final_hex_params.copy()
        outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered
        
        return inner_hex_data, outer_hex_data, outer_side_length
        
    except Exception as e:
        warnings.warn(f"Multistage optimization error: {str(e)}")
        # Fallback to improved symmetric arrangement
        inner_hex_data = np.array([
            [0, 0, 0],           # center
            [0, 2.1, 0],         # up
            [0, -2.1, 0],        # down
            [1.8, 1.0, 0],       # upper right
            [-1.8, 1.0, 0],      # upper left
            [1.8, -1.0, 0],      # lower right
            [-1.8, -1.0, 0],     # lower left
            [3.6, 0, 0],         # far right
            [0, 3.6, 0],         # far up
            [0, -3.6, 0],        # far down
            [2.1, 2.1, 0],       # upper right diagonal
            [-2.1, 2.1, 0],      # upper left diagonal
        ])

        outer_hex_data = np.array([0, 0, 0])
        outer_side_length = 8.0
        
        return inner_hex_data, outer_hex_data, outer_side_length

def hexagon_packing_12() -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Ensure we don't exceed the 180 second limit by setting a timeout
    try:
        inner_hex_data, outer_hex_data, outer_hex_side_length = optimize_packing_multistage()

        # Calculate benchmark ratio
        benchmark_ratio = calculate_objective(outer_hex_side_length) / 0.2537

        # Output metrics for verification
        print(f"inv_outer_hex_side_length: {calculate_objective(outer_hex_side_length):.8f}")
        print(f"benchmark_ratio: {benchmark_ratio:.8f}")
        print(f"eval_time: {time.time() - start_time:.4f}s")

        return inner_hex_data, outer_hex_data, outer_hex_side_length
        
    except Exception as e:
        print(f"Error in hexagon packing: {e}")
        # Return fallback configuration with slightly better arrangement
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
        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END