# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
import math
from numba import jit, prange
import warnings
from scipy.spatial.distance import cdist

# Constants for hexagon geometry
HEX_RADIUS = 1.0
HEX_APODEM = np.sqrt(3) / 2
HEX_HEIGHT = 2 * HEX_APODEM
HEX_WIDTH = 2 * HEX_RADIUS

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
def point_in_polygon(point_x, point_y, polygon_vertices):
    """Ray casting algorithm to check if point is in polygon."""
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

@jit(nopython=True)
def hexagon_overlap_detect(vertices1, vertices2):
    """Simple overlap detection using bounding boxes and point-in-polygon."""
    # Quick bounding box check
    min1_x = min(v[0] for v in vertices1)
    max1_x = max(v[0] for v in vertices1)
    min1_y = min(v[1] for v in vertices1)
    max1_y = max(v[1] for v in vertices1)
    
    min2_x = min(v[0] for v in vertices2)
    max2_x = max(v[0] for v in vertices2)
    min2_y = min(v[1] for v in vertices2)
    max2_y = max(v[1] for v in vertices2)
    
    # If bounding boxes don't intersect, no overlap
    if max1_x < min2_x or max2_x < min1_x or max1_y < min2_y or max2_y < min1_y:
        return False
    
    # Check if any vertex of hex1 is inside hex2
    for vertex in vertices1:
        if point_in_polygon(vertex[0], vertex[1], vertices2):
            return True
    
    # Check if any vertex of hex2 is inside hex1
    for vertex in vertices2:
        if point_in_polygon(vertex[0], vertex[1], vertices1):
            return True
            
    return False

def create_lattice_structure():
    """Create a triangular lattice structure optimized for hexagon packing."""
    # Hexagonal lattice basis vectors
    # In a triangular lattice, the distance between nearest neighbors is 2*radius = 2
    # Basis vectors for a triangular lattice
    basis1 = np.array([2.0, 0.0])  # Horizontal
    basis2 = np.array([1.0, np.sqrt(3)])  # Diagonal
    
    # Generate lattice points that can accommodate 12 hexagons
    # We'll use a 3x4 rectangular arrangement in lattice coordinates
    lattice_points = []
    
    # 3 rows, 4 columns
    for i in range(3):
        for j in range(4):
            # Offset odd rows
            offset = j if i % 2 == 0 else j + 0.5
            point = i * basis2 + offset * basis1
            lattice_points.append(point)
    
    return np.array(lattice_points)

def create_hexagon_positions_from_lattice(lattice_points, center_offset=(0,0), scale=1.0):
    """Convert lattice points to hexagon positions."""
    positions = []
    for i, lp in enumerate(lattice_points):
        x = lp[0] * scale + center_offset[0]
        y = lp[1] * scale + center_offset[1]
        # Alternate rotations for better packing
        rotation = 0 if i % 2 == 0 else 30
        positions.append([x, y, rotation])
    return positions

class LatticeHexagonPacker:
    """Implements lattice-based approach for hexagon packing optimization."""
    
    def __init__(self):
        self.lattice_basis = np.array([
            [2.0, 0.0],
            [1.0, np.sqrt(3)]
        ])
        
    def generate_lattice_points(self):
        """Generate standard 3x4 lattice of points."""
        points = []
        for i in range(3):
            for j in range(4):
                offset = j if i % 2 == 0 else j + 0.5
                point = i * self.lattice_basis[1] + offset * self.lattice_basis[0]
                points.append(point)
        return np.array(points)
    
    def objective_function(self, params, target_ratio=0.2537):
        """
        Objective function for optimization.
        params: [scale, center_x, center_y, rotation0, rotation1, ... rotation11]
        """
        # Unpack parameters
        scale = params[0]
        center_x = params[1] 
        center_y = params[2]
        rotations = params[3:]
        
        # Generate positions from lattice
        lattice_points = self.generate_lattice_points()
        positions = []
        for i, lp in enumerate(lattice_points):
            x = lp[0] * scale + center_x
            y = lp[1] * scale + center_y
            rot = rotations[i] if i < len(rotations) else 0
            positions.append([x, y, rot])
        
        # Create hexagon polygons
        hex_polygons = []
        for pos in positions:
            vertices = hexagon_vertices_numba(pos[0], pos[1], pos[2])
            hex_polygons.append(Polygon(vertices))
        
        # Check for overlaps
        for i in range(len(hex_polygons)):
            for j in range(i+1, len(hex_polygons)):
                if hex_polygons[i].intersects(hex_polygons[j]):
                    return 1e10  # Large penalty for overlaps
        
        # Calculate outer hexagon size needed
        outer_radius = 0
        for pos in positions:
            vertices = hexagon_vertices_numba(pos[0], pos[1], pos[2])
            for vertex in vertices:
                dist = np.sqrt((vertex[0])**2 + (vertex[1])**2)
                outer_radius = max(outer_radius, dist)
        
        # Add safety factor
        outer_radius *= 1.1
        
        # Calculate inverse side length
        inv_side_length = 1.0 / outer_radius
        
        # We want to maximize 1/outer_radius, so minimize negative of it
        return -inv_side_length
    
    def optimize(self):
        """Run the optimization."""
        # Initial parameters: [scale, center_x, center_y, 12 rotation parameters]
        initial_params = [2.0, 0.0, 0.0] + [0.0] * 12
        
        # Bounds for parameters
        bounds = [(1.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]  # scale, center_x, center_y
        for _ in range(12):
            bounds.append((-30.0, 30.0))  # rotation bounds
        
        # Use differential evolution for global optimization
        try:
            result = differential_evolution(
                self.objective_function,
                bounds,
                maxiter=50,
                popsize=20,
                seed=42,
                disp=False,
                tol=1e-6
            )
            
            if result.success:
                return result.x
        except Exception as e:
            warnings.warn(f"Differential evolution failed: {e}")
        
        # Fallback to local optimization
        try:
            result = minimize(
                self.objective_function,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-8}
            )
            if result.success:
                return result.x
        except Exception as e:
            warnings.warn(f"Local optimization failed: {e}")
        
        return initial_params

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Initialize packer
    packer = LatticeHexagonPacker()
    
    # Run optimization
    try:
        opt_params = packer.optimize()
        
        # Extract results
        scale = opt_params[0]
        center_x = opt_params[1]
        center_y = opt_params[2]
        rotations = opt_params[3:]
        
        # Generate hexagon positions 
        lattice_points = packer.generate_lattice_points()
        positions = []
        for i, lp in enumerate(lattice_points):
            x = lp[0] * scale + center_x
            y = lp[1] * scale + center_y
            rot = rotations[i] if i < len(rotations) else 0
            positions.append([x, y, rot])
        
        # Convert to numpy array
        inner_hex_data = np.array(positions)
        
        # Calculate outer hexagon size
        outer_radius = 0
        for pos in positions:
            vertices = hexagon_vertices_numba(pos[0], pos[1], pos[2])
            for vertex in vertices:
                dist = np.sqrt((vertex[0])**2 + (vertex[1])**2)
                outer_radius = max(outer_radius, dist)
        
        outer_radius *= 1.1  # Safety factor
        outer_hex_side_length = outer_radius
        
        # Outer hexagon data (centered at origin)
        outer_hex_data = np.array([0, 0, 0])
        
        # Final validation
        hex_polygons = []
        for pos in positions:
            vertices = hexagon_vertices_numba(pos[0], pos[1], pos[2])
            hex_polygons.append(Polygon(vertices))
        
        # Check overlaps
        has_overlap = False
        for i in range(len(hex_polygons)):
            for j in range(i+1, len(hex_polygons)):
                if hex_polygons[i].intersects(hex_polygons[j]):
                    has_overlap = True
                    break
            if has_overlap:
                break
        
        if has_overlap:
            # Fallback to known good configuration
            inner_hex_data = np.array([
                [0, 0, 0],           # center
                [-2.5, 0, 0],        # left
                [2.5, 0, 0],         # right
                [-1.25, 2.17, 0],    # top-left
                [1.25, 2.17, 0],     # top-right
                [-1.25, -2.17, 0],   # bottom-left
                [1.25, -2.17, 0],    # bottom-right
                [-3.75, 2.17, 0],    # far top-left
                [3.75, 2.17, 0],     # far top-right
                [-3.75, -2.17, 0],   # far bottom-left
                [3.75, -2.17, 0],    # far bottom-right
                [0, -4, 0],          # far bottom-center
            ])
            outer_hex_side_length = 8.0
            outer_hex_data = np.array([0, 0, 0])
            
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}")
        # Fallback to known good configuration
        inner_hex_data = np.array([
            [0, 0, 0],           # center
            [-2.5, 0, 0],        # left
            [2.5, 0, 0],         # right
            [-1.25, 2.17, 0],    # top-left
            [1.25, 2.17, 0],     # top-right
            [-1.25, -2.17, 0],   # bottom-left
            [1.25, -2.17, 0],    # bottom-right
            [-3.75, 2.17, 0],    # far top-left
            [3.75, 2.17, 0],     # far top-right
            [-3.75, -2.17, 0],   # far bottom-left
            [3.75, -2.17, 0],    # far bottom-right
            [0, -4, 0],          # far bottom-center
        ])
        outer_hex_side_length = 8.0
        outer_hex_data = np.array([0, 0, 0])
    
    eval_time = time.time() - start_time
    
    # Calculate benchmark ratio
    benchmark_ratio = (1.0 / outer_hex_side_length) / 0.2537
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END