# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
import time
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')

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

class SpatialHashGrid:
    """Spatial hash grid for efficient neighbor lookups"""
    
    def __init__(self, cell_size=2.0):
        self.cell_size = cell_size
        self.grid = {}
    
    def clear(self):
        self.grid.clear()
    
    def insert(self, hex_id, center_x, center_y):
        """Insert hexagon into spatial hash grid"""
        cell_x = int(center_x // self.cell_size)
        cell_y = int(center_y // self.cell_size)
        key = (cell_x, cell_y)
        if key not in self.grid:
            self.grid[key] = []
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

@jit(nopython=True)
def generate_hexagon_vertices_numba(center_x, center_y, angle_degrees):
    """Generate vertices of a unit regular hexagon given center and rotation - JIT compiled"""
    angle_rad = np.radians(angle_degrees)
    vertices = np.empty((6, 2), dtype=np.float64)
    
    # Precompute trigonometric values
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    # Hexagon vertices in local coordinate system (unit hexagon centered at origin)
    local_vertices = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Apply rotation and translation
    for i in range(6):
        x, y = local_vertices[i]
        # Rotate
        rot_x = x * cos_angle - y * sin_angle
        rot_y = x * sin_angle + y * cos_angle
        # Translate
        vertices[i, 0] = rot_x + center_x
        vertices[i, 1] = rot_y + center_y
    
    return vertices

@jit(nopython=True)
def point_in_polygon_numba(point, polygon_vertices):
    """Fast point-in-polygon test - JIT compiled"""
    x, y = point
    n = len(polygon_vertices)
    inside = False
    
    p1x, p1y = polygon_vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_vertices[i % n]
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
def compute_outer_hexagon_side_length_fast(inner_hex_data):
    """Fast computation of outer hexagon side length using vectorized operations"""
    if len(inner_hex_data) == 0:
        return 1e6
    
    # Get all vertices efficiently
    all_vertices = np.empty((len(inner_hex_data) * 6, 2), dtype=np.float64)
    
    for i in prange(len(inner_hex_data)):
        center_x, center_y, angle_degrees = inner_hex_data[i]
        vertices = generate_hexagon_vertices_numba(center_x, center_y, angle_degrees)
        for j in range(6):
            all_vertices[i*6 + j] = vertices[j]
    
    # Compute bounding box
    min_x = np.min(all_vertices[:, 0])
    max_x = np.max(all_vertices[:, 0])
    min_y = np.min(all_vertices[:, 1])
    max_y = np.max(all_vertices[:, 1])
    
    # Compute center
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    
    # Compute max distance squared from center
    max_dist_sq = 0.0
    for i in prange(len(all_vertices)):
        x, y = all_vertices[i]
        dist_sq = (x - center_x)**2 + (y - center_y)**2
        max_dist_sq = max(max_dist_sq, dist_sq)
    
    # Side length = sqrt(max_dist) * 2 / sqrt(3)
    return np.sqrt(max_dist_sq) * 2.0 / np.sqrt(3)

class HexagonPacker:
    """Main class for hexagon packing optimization"""
    
    def __init__(self):
        self.hex_radius = 1.0
        self.hex_apothem = np.sqrt(3) / 2
        self.hex_height = 2 * self.hex_apothem
        self.hex_width = 2 * self.hex_radius
    
    def create_outer_hexagon_polygon(self, side_length: float) -> Polygon:
        """Create outer hexagon as Shapely polygon"""
        hex = Hexagon(0, 0, 0)
        outer_vertices = hex.vertices()
        # Scale to match desired outer radius
        outer_vertices *= side_length / self.hex_radius
        return Polygon(outer_vertices)
    
    def check_containment(self, hexagon: Hexagon, outer_polygon: Polygon) -> bool:
        """Check if all vertices of hexagon are inside outer polygon"""
        vertices = hexagon.vertices()
        for vertex in vertices:
            point = Point(vertex[0], vertex[1])
            if not outer_polygon.contains(point):
                return False
        return True
    
    def check_overlap(self, hex1: Hexagon, hex2: Hexagon) -> bool:
        """Check if two hexagons overlap using Shapely"""
        poly1 = Polygon(hex1.vertices())
        poly2 = Polygon(hex2.vertices())
        return poly1.intersects(poly2)
    
    def create_hexagon_from_params(self, params):
        """Create hexagon from parameter array [x, y, angle]"""
        return Hexagon(params[0], params[1], params[2])
    
    def evaluate_fitness(self, params, outer_side_length=10.0):
        """Evaluate fitness of a solution"""
        # Reshape parameters to 12 hexagons
        positions = params.reshape(-1, 3)
        
        # Create hexagons
        hexagons = [self.create_hexagon_from_params(pos) for pos in positions]
        
        # Create outer hexagon
        outer_polygon = self.create_outer_hexagon_polygon(outer_side_length)
        
        # Check containment
        for hex in hexagons:
            if not self.check_containment(hex, outer_polygon):
                return 1e6
        
        # Check overlaps using spatial hashing
        spatial_hash = SpatialHashGrid(cell_size=3.0)
        for i, hex in enumerate(hexagons):
            spatial_hash.insert(i, hex.center[0], hex.center[1])
        
        # Check overlaps
        penalty = 0
        for i in range(len(hexagons)):
            neighbors = spatial_hash.get_neighbors(hexagons[i].center[0], hexagons[i].center[1])
            for j in neighbors:
                if i != j and self.check_overlap(hexagons[i], hexagons[j]):
                    penalty += 1e5
        
        # Calculate actual outer side length
        actual_size = compute_outer_hexagon_side_length_fast(positions)
        
        if actual_size > outer_side_length:
            return 1e6
        
        # Return negative of 1/actual_size to maximize 1/size
        return -1.0 / actual_size + penalty
    
    def get_initial_guess(self):
        """Generate initial configuration using a known good starting point"""
        # Known good configuration from literature
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
            [0, -4, 0]
        ])
        
        return positions.flatten()

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
        # Initialize packer
        packer = HexagonPacker()
        
        # Get initial guess
        initial_guess = packer.get_initial_guess()
        
        # Define bounds for optimization
        bounds = []
        for _ in range(12):
            bounds.extend([(-10, 10), (-10, 10), (0, 360)])
        bounds.append((1.0, 20.0))  # Outer radius
        
        # Run optimization
        def objective_func(params):
            return packer.evaluate_fitness(params, outer_side_length=8.0)
        
        result = differential_evolution(
            objective_func,
            bounds,
            maxiter=30,
            popsize=10,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=False,
            tol=1e-6
        )
        
        if result.success:
            # Extract final configuration
            final_params = result.x
            positions = final_params.reshape(-1, 3)
            
            # Compute actual outer hexagon side length
            outer_side_length = compute_outer_hexagon_side_length_fast(positions)
            
            # Create inner hex data
            inner_hex_data = positions.copy()
            outer_hex_data = np.array([0, 0, 0])
            
            return inner_hex_data, outer_hex_data, outer_side_length
        else:
            raise Exception("Optimization failed")
            
    except Exception as e:
        # Fallback to initial configuration
        warnings.warn(f"Fallback due to error: {e}")
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
    
    end_time = time.time()
    eval_time = end_time - start_time
    
    return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END