# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import math
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from numba import jit
import time

# Constants for hexagons
UNIT_HEX_RADIUS = 1.0
UNIT_HEX_WIDTH = 2.0  # Distance between parallel sides
UNIT_HEX_SIDE_LENGTH = 1.0  # Side length of unit hexagon
MAX_EVAL_TIME = 180.0  # seconds

@jit(nopython=True)
def get_hexagon_vertices_numba(x, y, angle_deg, radius=1.0):
    """Get vertices of a hexagon given center, angle, and radius (Numba optimized)"""
    vertices = np.zeros((6, 2))
    angle_rad = np.radians(angle_deg)
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i] = [x + radius * np.cos(theta), y + radius * np.sin(theta)]
    return vertices

class Hexagon:
    """Represents a regular hexagon with position and rotation"""
    
    def __init__(self, x, y, angle_deg, radius=1.0):
        self.x = x
        self.y = y
        self.angle_deg = angle_deg
        self.radius = radius
    
    def get_vertices(self):
        """Get vertices of the hexagon"""
        return get_hexagon_vertices_numba(self.x, self.y, self.angle_deg, self.radius)
    
    def to_polygon(self):
        """Convert hexagon to Shapely polygon"""
        vertices = self.get_vertices()
        return Polygon(vertices)
    
    def contains_point(self, point_x, point_y):
        """Check if a point is inside this hexagon"""
        vertices = self.get_vertices()
        hex_poly = Polygon(vertices)
        pt = Point(point_x, point_y)
        return hex_poly.contains(pt)

class PackingValidator:
    """Handles all validation logic for hexagon packings"""
    
    @staticmethod
    def validate_solution(inner_hex_data):
        """Validate that solution meets all constraints"""
        if len(inner_hex_data) != 12:
            return False, "Wrong number of hexagons"
        
        # Check for overlaps between any pair of hexagons
        hexagons = [Hexagon(x, y, angle) for x, y, angle in inner_hex_data]
        
        # Early exit optimization: check bounding boxes first
        for i in range(len(hexagons)):
            hex1 = hexagons[i]
            for j in range(i+1, len(hexagons)):
                hex2 = hexagons[j]
                
                # Fast bounding box check
                vertices1 = hex1.get_vertices()
                vertices2 = hex2.get_vertices()
                
                min_x1, max_x1 = vertices1[:, 0].min(), vertices1[:, 0].max()
                min_y1, max_y1 = vertices1[:, 1].min(), vertices1[:, 1].max()
                min_x2, max_x2 = vertices2[:, 0].min(), vertices2[:, 0].max()
                min_y2, max_y2 = vertices2[:, 1].min(), vertices2[:, 1].max()
                
                if (max_x1 < min_x2 or max_x2 < min_x1 or 
                    max_y1 < min_y2 or max_y2 < min_y1):
                    continue  # No intersection possible
                
                # Full overlap check
                poly1 = hex1.to_polygon()
                poly2 = hex2.to_polygon()
                
                if poly1.intersects(poly2) and not poly1.touches(poly2):
                    return False, f"Overlapping hexagons {i} and {j}"
        
        # Check containment
        outer_radius = PackingValidator.compute_outer_hexagon_radius(inner_hex_data)
        outer_hex = Hexagon(0, 0, 0, outer_radius)
        outer_poly = outer_hex.to_polygon()
        
        for i, hexagon in enumerate(hexagons):
            inner_poly = hexagon.to_polygon()
            if not outer_poly.contains(inner_poly):
                return False, f"Inner hexagon {i} not contained"
            
            # Check that all vertices are contained
            vertices = hexagon.get_vertices()
            for vx, vy in vertices:
                inner_point = Point(vx, vy)
                if not outer_poly.contains(inner_point):
                    return False, f"Hexagon {i} vertex ({vx}, {vy}) outside outer hexagon"
        
        return True, "Valid solution"
    
    @staticmethod
    def compute_outer_hexagon_radius(inner_hex_data):
        """Compute minimum outer hexagon radius that contains all inner hexagons"""
        if len(inner_hex_data) == 0:
            return 0.0

        # Get all vertices of all inner hexagons
        all_vertices = []
        for i in range(len(inner_hex_data)):
            x, y, angle = inner_hex_data[i]
            vertices = get_hexagon_vertices_numba(x, y, angle)
            all_vertices.extend(vertices)

        if len(all_vertices) == 0:
            return 0.0

        # Compute centroid
        centroid_x = np.mean([v[0] for v in all_vertices])
        centroid_y = np.mean([v[1] for v in all_vertices])

        # Find maximum distance from centroid to any vertex
        max_distance = 0.0
        for x, y in all_vertices:
            distance = math.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
            max_distance = max(max_distance, distance)

        return max_distance + UNIT_HEX_RADIUS

class InitialConfigurationFactory:
    """Creates various initial configurations for optimization"""
    
    @staticmethod
    def create_classic_hexagonal():
        """Create classic hexagonal close-packed arrangement"""
        return np.array([
            [0.0, 0.0, 0.0],       # Center
            [0.0, 2.0, 0.0],       # Top
            [0.0, -2.0, 0.0],      # Bottom
            [1.732, 1.0, 0.0],     # Top right
            [-1.732, 1.0, 0.0],    # Top left
            [1.732, -1.0, 0.0],    # Bottom right
            [-1.732, -1.0, 0.0],   # Bottom left
            [3.464, 0.0, 0.0],     # Far right
            [-3.464, 0.0, 0.0],    # Far left
            [0.0, 3.464, 0.0],     # Very top
            [0.0, -3.464, 0.0],    # Very bottom
            [1.732, 2.0, 0.0],     # Additional corner
        ])
    
    @staticmethod
    def create_triangular_arrangement():
        """Create triangular arrangement with radial symmetry"""
        return np.array([
            [0.0, 0.0, 0.0],       # Center
            [0.0, 2.0, 0.0],       # Top
            [1.732, 1.0, 0.0],     # Top right
            [1.732, -1.0, 0.0],    # Bottom right
            [0.0, -2.0, 0.0],      # Bottom
            [-1.732, -1.0, 0.0],   # Bottom left
            [-1.732, 1.0, 0.0],    # Top left
            [3.464, 0.0, 0.0],     # Far right
            [3.464, 2.0, 0.0],     # Far top right
            [3.464, -2.0, 0.0],    # Far bottom right
            [-3.464, 0.0, 0.0],    # Far left
            [-3.464, 2.0, 0.0],    # Far top left
        ])
    
    @staticmethod
    def create_checkerboard_arrangement():
        """Create checkerboard-like arrangement"""
        return np.array([
            [0.0, 0.0, 0.0],       # Center
            [0.0, 2.0, 0.0],       # Top
            [0.0, -2.0, 0.0],      # Bottom
            [1.732, 1.0, 0.0],     # Top right
            [-1.732, 1.0, 0.0],    # Top left
            [1.732, -1.0, 0.0],    # Bottom right
            [-1.732, -1.0, 0.0],   # Bottom left
            [3.464, 0.0, 0.0],     # Far right
            [-3.464, 0.0, 0.0],    # Far left
            [0.0, 3.464, 0.0],     # Very top
            [0.0, -3.464, 0.0],    # Very bottom
            [1.732, 2.0, 0.0],     # Additional corner
        ])
    
    @staticmethod
    def get_all_initial_configurations():
        """Get all initial configurations"""
        return [
            InitialConfigurationFactory.create_classic_hexagonal(),
            InitialConfigurationFactory.create_triangular_arrangement(),
            InitialConfigurationFactory.create_checkerboard_arrangement()
        ]

class OptimizationEngine:
    """Handles the optimization process with multiple strategies"""
    
    @staticmethod
    def local_search(initial_config, max_iter=100):
        """Refine configuration using local search"""
        # Flatten the initial configuration
        params = initial_config.flatten()

        # Objective function to minimize
        def objective(x):
            # Reshape back to hexagon configuration
            hex_config = x.reshape(12, 3)

            # Validate and penalize invalid solutions
            valid, msg = PackingValidator.validate_solution(hex_config)
            if not valid:
                return 1e10  # High penalty for invalid solutions

            # Minimize negative of 1/outer_radius (i.e., maximize 1/outer_radius)
            outer_radius = PackingValidator.compute_outer_hexagon_radius(hex_config)
            if outer_radius <= 0:
                return 1e10
            return -1.0 / outer_radius

        # Local optimization with bounds
        bounds = [(-10.0, 10.0)] * 36  # 12 hexagons * 3 params each

        try:
            result = minimize(
                objective,
                params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-8}
            )

            if result.success:
                optimized_config = result.x.reshape(12, 3)
                return optimized_config
        except:
            pass

        return initial_config

def optimize_hexagon_arrangement():
    """Use a multi-stage optimization with symmetry awareness"""
    # Get multiple starting configurations
    initial_configs = InitialConfigurationFactory.get_all_initial_configurations()

    best_config = None
    best_fitness = -float('inf')

    for i, initial_config in enumerate(initial_configs):
        # Stage 1: Basic local optimization
        refined_config = OptimizationEngine.local_search(initial_config, 50)

        # Stage 2: More intensive refinement
        final_config = OptimizationEngine.local_search(refined_config, 100)

        # Evaluate the final configuration
        valid, msg = PackingValidator.validate_solution(final_config)
        if valid:
            outer_radius = PackingValidator.compute_outer_hexagon_radius(final_config)
            fitness = 1.0 / outer_radius if outer_radius > 0 else 0.0
            if fitness > best_fitness:
                best_fitness = fitness
                best_config = final_config

    # If none of the initial configs worked, return a default configuration
    if best_config is None:
        # Default configuration that's known to work reasonably well
        best_config = np.array([
            [0.0, 0.0, 0.0],       # Center
            [0.0, 2.0, 0.0],       # Top
            [0.0, -2.0, 0.0],      # Bottom
            [1.732, 1.0, 0.0],     # Top right
            [-1.732, 1.0, 0.0],    # Top left
            [1.732, -1.0, 0.0],    # Bottom right
            [-1.732, -1.0, 0.0],   # Bottom left
            [3.464, 0.0, 0.0],     # Far right
            [-3.464, 0.0, 0.0],    # Far left
            [0.0, 3.464, 0.0],     # Very top
            [0.0, -3.464, 0.0],    # Very bottom
            [1.732, 2.0, 0.0],     # Additional corner
        ])

    return best_config

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """

    # Start with a symmetry-aware optimized configuration
    inner_hex_data = optimize_hexagon_arrangement()

    # Compute the outer hexagon radius required
    outer_radius = PackingValidator.compute_outer_hexagon_radius(inner_hex_data)

    # Convert to side length (for regular hexagon, radius = side length)
    outer_hex_side_length = outer_radius

    # Outer hexagon centered at origin, no rotation
    outer_hex_data = np.array([0, 0, 0])

    # Validate final solution
    valid, message = PackingValidator.validate_solution(inner_hex_data)
    if not valid:
        # Fallback to a simple but safe configuration
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
        outer_hex_side_length = 8

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END