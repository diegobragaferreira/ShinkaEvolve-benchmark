# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon, Point
from joblib import Parallel, delayed
import time
from numba import jit
from scipy.spatial import cKDTree
from functools import lru_cache

@jit(nopython=True)
def distance_point_to_line(point, line_start, line_end):
    """Calculate the shortest distance from a point to a line segment."""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Vector from line_start to line_end
    dx, dy = x2 - x1, y2 - y1
    
    # Length squared of line segment
    length_sq = dx*dx + dy*dy
    
    if length_sq == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Project point onto line
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    
    # Closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

@jit(nopython=True)
def point_in_hexagon_fast(point_x, point_y, hex_center_x, hex_center_y, rotation, side_length):
    """Fast check if a point is inside a regular hexagon using distance to edges."""
    # For a regular hexagon with distance from center to vertex = side_length
    # Distance from center to edge = side_length * sqrt(3)/2
    
    # Transform point to hexagon's coordinate system
    cos_rot = np.cos(rotation)
    sin_rot = np.sin(rotation)
    dx = point_x - hex_center_x
    dy = point_y - hex_center_y
    rot_x = dx * cos_rot + dy * sin_rot
    rot_y = -dx * sin_rot + dy * cos_rot
    
    # Distance from center to edge in x and y directions
    edge_distance_x = side_length * np.sqrt(3) / 2
    edge_distance_y = side_length * 0.5
    
    # Check if point is within bounds
    return abs(rot_x) <= edge_distance_x and abs(rot_y) <= edge_distance_y and \
           abs(rot_x) + abs(rot_y) <= side_length * np.sqrt(3)

class Hexagon:
    """Represents a regular hexagon with position and rotation."""
    
    def __init__(self, center_x, center_y, angle_degrees, side_length=1):
        self.center_x = center_x
        self.center_y = center_y
        self.angle_degrees = angle_degrees
        self.side_length = side_length
        self._vertices = None
        
    @property
    def vertices(self):
        """Cached generation of hexagon vertices."""
        if self._vertices is None:
            self._vertices = self._generate_vertices()
        return self._vertices
        
    def _generate_vertices(self):
        """Generate vertices of a regular hexagon."""
        angle_rad = np.radians(self.angle_degrees)
        angles = np.linspace(0, 2*np.pi, 7) + angle_rad  # 6 sides + closing vertex
        vertices = []
        for angle in angles:
            x = self.center_x + self.side_length * np.cos(angle)
            y = self.center_y + self.side_length * np.sin(angle)
            vertices.append((x, y))
        return np.array(vertices)
        
    def get_bounding_circle(self):
        """Get the bounding circle (center and radius) of the hexagon."""
        cx, cy = self.center_x, self.center_y
        # Distance from center to corner of unit hexagon is 1
        r = self.side_length
        return cx, cy, r

class HexagonPacker:
    """Handles the core hexagon packing logic."""
    
    def __init__(self, num_inner_hexagons=12):
        self.num_inner_hexagons = num_inner_hexagons
        self.hexagons = []
        self.outer_hexagon = None
        
    def create_symmetric_layout(self, params):
        """Create hexagon layout based on symmetric parameters."""
        # params[0]: middle ring radius  
        # params[1]: outer ring radius
        # params[2]: middle ring angle offset
        # params[3]: outer ring angle offset  
        # params[4]: outer hexagon angle (rotation)
        # params[5]: outer hexagon center x
        # params[6]: outer hexagon center y
        # params[7]: center hexagon rotation
        # params[8]: middle ring rotation
        
        # Clear existing hexagons
        self.hexagons.clear()
        
        middle_radius = params[0]
        outer_radius = params[1] 
        middle_angle_offset = params[2]
        outer_angle_offset = params[3]
        
        # Layer 1: Center (1 hexagon)
        center_hex = Hexagon(0.0, 0.0, params[7])
        self.hexagons.append(center_hex)
        
        # Layer 2: Middle ring (6 hexagons)
        for i in range(6):
            angle = (i * 60 + middle_angle_offset) % 360
            rad = middle_radius
            x = rad * np.cos(np.radians(angle))
            y = rad * np.sin(np.radians(angle))
            hexagon = Hexagon(x, y, params[8])
            self.hexagons.append(hexagon)
        
        # Layer 3: Outer ring (5 hexagons)
        for i in range(5):
            angle = (i * 72 + outer_angle_offset) % 360
            rad = outer_radius
            x = rad * np.cos(np.radians(angle))
            y = rad * np.sin(np.radians(angle))
            hexagon = Hexagon(x, y, 0.0)
            self.hexagons.append(hexagon)
            
        return self.hexagons
        
    def calculate_outer_radius(self):
        """Calculate required outer hexagon radius based on inner hexagons."""
        max_dist = 0
        for hexagon in self.hexagons:
            for vertex in hexagon.vertices:
                dist = np.sqrt(vertex[0]**2 + vertex[1]**2)
                max_dist = max(max_dist, dist)
        return max_dist * 1.02  # Add 2% buffer for numerical stability
        
    def create_outer_hexagon(self, center_x, center_y, angle, radius):
        """Create the outer hexagon."""
        self.outer_hexagon = Hexagon(center_x, center_y, angle, radius)
        return self.outer_hexagon
        
    def check_containment_all(self):
        """Check if all inner hexagons are contained within outer hexagon."""
        if not self.outer_hexagon:
            return False
        outer_vertices = self.outer_hexagon.vertices
        
        for hexagon in self.hexagons:
            inner_polygon = Polygon(hexagon.vertices)
            outer_polygon = Polygon(outer_vertices)
            if not outer_polygon.contains(inner_polygon):
                return False
        return True
        
    @staticmethod
    @lru_cache(maxsize=1000)
    def _cached_overlap_check(hex1_vertices_tuple, hex2_vertices_tuple):
        """Cached overlap check between two hexagons."""
        polygon1 = Polygon(hex1_vertices_tuple)
        polygon2 = Polygon(hex2_vertices_tuple)
        return polygon1.intersects(polygon2)
        
    def fast_overlap_check(self, hex1, hex2):
        """Fast overlap check using bounding circles for early rejection."""
        cx1, cy1, r1 = hex1.get_bounding_circle()
        cx2, cy2, r2 = hex2.get_bounding_circle()
        
        # Fast circle overlap test
        dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        if dist >= (r1 + r2):
            return False  # Definitely don't overlap
        
        # If close enough, do precise overlap check
        return self._cached_overlap_check(
            tuple(tuple(v) for v in hex1.vertices),
            tuple(tuple(v) for v in hex2.vertices)
        )
        
    def check_overlaps(self):
        """Check for overlaps between hexagons with optimized spatial queries."""
        if len(self.hexagons) < 2:
            return False
            
        # Build spatial index for efficient neighbor querying
        hex_centers = np.array([[h.center_x, h.center_y] for h in self.hexagons])
        tree = cKDTree(hex_centers)
        
        # Find neighbors within a reasonable distance (2x hexagon diameter)
        pairs_to_check = tree.query_pairs(r=3.0, p=np.inf)
        
        # Check overlaps using spatial acceleration
        for i, j in pairs_to_check:
            # Skip center with itself
            if i == 0 and j == 0:
                continue
                
            # Check overlap between hexagons i and j
            if self.fast_overlap_check(self.hexagons[i], self.hexagons[j]):
                return True  # Found overlap
                
        # Additional specific overlap checks for critical pairs
        # Center with all others
        for i in range(1, len(self.hexagons)):  # center with all other hexagons
            if self.fast_overlap_check(self.hexagons[0], self.hexagons[i]):
                return True
                
        # Middle ring vs outer ring
        for i in range(1, 7):  # middle ring
            for j in range(7, 12):  # outer ring
                if self.fast_overlap_check(self.hexagons[i], self.hexagons[j]):
                    return True
                    
        # Middle ring self-intersection
        for i in range(1, 7):
            for j in range(i+1, 7):
                if self.fast_overlap_check(self.hexagons[i], self.hexagons[j]):
                    return True
                    
        # Outer ring self-intersection  
        for i in range(7, 12):
            for j in range(i+1, 12):
                if self.fast_overlap_check(self.hexagons[i], self.hexagons[j]):
                    return True
                    
        return False

class OptimizationEngine:
    """Handles the optimization process with configurable parameters."""
    
    def __init__(self, packer):
        self.packer = packer
        self.bounds = [
            (1.0, 4.0),     # middle ring radius  
            (2.0, 6.0),     # outer ring radius
            (-180, 180),    # middle ring angle offset
            (-180, 180),    # outer ring angle offset  
            (-180, 180),    # outer hex angle
            (-5.0, 5.0),    # outer center x
            (-5.0, 5.0),    # outer center y
            (-180, 180),    # center rotation
            (-180, 180)     # middle rotation
        ]
        self.maxiter = 30
        self.popsize = 20
        
    def evaluate(self, params):
        """Evaluate the configuration and return fitness score."""
        # Update the packer with new parameters
        self.packer.create_symmetric_layout(params)
        
        # Calculate outer hexagon parameters
        outer_radius = self.packer.calculate_outer_radius()
        outer_center_x, outer_center_y, outer_angle = params[5:8]
        
        # Create outer hexagon
        self.packer.create_outer_hexagon(outer_center_x, outer_center_y, outer_angle, outer_radius)
        
        # Check constraints
        total_penalty = 0
        
        # Check containment
        if not self.packer.check_containment_all():
            total_penalty += 10000
            
        # Check overlaps
        if self.packer.check_overlaps():
            total_penalty += 10000
            
        # Return negative inverse of outer radius plus penalties
        return -(1.0 / (outer_radius + total_penalty + 1e-8))

def create_unit_hexagon(center=(0,0), rotation=0):
    """Create a unit regular hexagon with given center and rotation."""
    angle = rotation * np.pi / 180
    # Vertices of a unit hexagon centered at origin
    hex_vertices = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = np.cos(theta)
        y = np.sin(theta)
        hex_vertices.append((x + center[0], y + center[1]))
    return Polygon(hex_vertices)

def check_containment(inner_hex, outer_hex):
    """Check if inner hexagon is fully contained within outer hexagon."""
    # Check if all vertices of inner hex are inside outer hex
    for point in list(inner_hex.exterior.coords):
        if not outer_hex.contains(Point(point)):
            return False
    return True

def check_overlap(hex1, hex2):
    """Check if two hexagons overlap."""
    return hex1.intersects(hex2)

def parallel_overlap_check(hexagons, start_idx, end_idx):
    """Parallel overlap checking for a subset of hexagon pairs."""
    overlaps = []
    for i in range(start_idx, end_idx):
        for j in range(i + 1, len(hexagons)):
            if check_overlap(hexagons[i], hexagons[j]):
                overlaps.append((i, j))
    return overlaps

def evaluate_configuration_parallel(config):
    """
    Evaluate a configuration with parallel constraint checking.
    config: array of shape (37,) - [x1,y1,theta1,...,x12,y12,theta12,R]
    Returns negative inverse side length (to maximize inverse side length)
    """
    # Extract parameters
    positions_angles = config[:-1].reshape(-1, 3)
    outer_radius = config[-1]
    
    # Create outer hexagon
    outer_hex = create_unit_hexagon((0, 0), 0)
    # Scale the outer hexagon to have side length = outer_radius
    scaled_outer_vertices = []
    for i in range(6):
        theta = i * np.pi / 3
        x = outer_radius * np.cos(theta)
        y = outer_radius * np.sin(theta)
        scaled_outer_vertices.append((x, y))
    outer_hex = Polygon(scaled_outer_vertices)

    # Create inner hexagons
    inner_hexagons = []
    for i in range(12):
        x, y, angle = positions_angles[i]
        inner_hex = create_unit_hexagon((x, y), angle)
        inner_hexagons.append(inner_hex)

        # Check containment early
        if not check_containment(inner_hex, outer_hex):
            return 1e10  # Penalty for violation

    # Check pairwise overlaps in parallel
    # Split work into chunks for parallel processing
    num_pairs = 12 * 11 // 2  # Number of unique pairs
    chunk_size = max(1, num_pairs // 4)  # Process 4 chunks
    
    # Use joblib for parallel overlap checking
    overlap_results = Parallel(n_jobs=-1)(
        delayed(parallel_overlap_check)(inner_hexagons, i*chunk_size, min((i+1)*chunk_size, len(inner_hexagons)))
        for i in range(4)
    )
    
    # Check if any overlaps were found
    for result in overlap_results:
        if result:
            return 1e10  # Penalty for overlap

    # Return negative inverse side length (we want to maximize 1/R)
    return -1.0 / outer_radius

def get_initial_guess_better():
    """Get a better initial guess based on known hexagon packing patterns"""
    # Start with a known dense configuration
    # Arrange in a hexagonal pattern with strategic positioning
    positions_angles = []
    
    # Central hexagon
    positions_angles.append([0.0, 0.0, 0.0])
    
    # First ring (6 hexagons)
    for i in range(6):
        angle = i * np.pi/3
        x = 2.0 * np.cos(angle)
        y = 2.0 * np.sin(angle)
        positions_angles.append([x, y, 0.0])
    
    # Second ring (6 hexagons) 
    for i in range(6):
        angle = i * np.pi/3 + np.pi/6
        x = 3.0 * np.cos(angle)
        y = 3.0 * np.sin(angle)
        positions_angles.append([x, y, 0.0])
        
    # Add reasonable starting outer radius
    initial_radius = 5.5

    # Flatten for optimization
    flat_config = np.array(positions_angles).flatten()
    flat_config = np.append(flat_config, initial_radius)

    return flat_config

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses an enhanced optimization approach combining different strategies.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    try:
        # Use the symmetric optimization approach for better initial configuration
        # Initialize components
        packer = HexagonPacker(12)
        optimizer = OptimizationEngine(packer)
        
        # Initial smart symmetric configuration
        initial_params = np.array([
            2.15,     # middle ring radius  
            3.6,      # outer ring radius
            30.0,     # middle ring angle offset
            0.0,      # outer ring angle offset  
            0.0,      # outer hexagon angle
            0.0,      # outer hexagon center x
            0.0,      # outer hexagon center y
            0.0,      # center hexagon rotation
            0.0       # middle ring rotation
        ])
        
        # Run optimization
        result = differential_evolution(
            optimizer.evaluate, 
            optimizer.bounds, 
            maxiter=optimizer.maxiter, 
            popsize=optimizer.popsize, 
            seed=42, 
            disp=False,
            atol=1e-6,
            ftol=1e-6
        )
        
        # Extract optimized parameters
        optimized_params = result.x
        
        # Recreate final configuration
        packer.create_symmetric_layout(optimized_params)
        
        # Calculate exact outer radius needed
        outer_radius_final = packer.calculate_outer_radius()
        outer_center_x, outer_center_y, outer_angle = optimized_params[5:8]
        packer.create_outer_hexagon(outer_center_x, outer_center_y, outer_angle, outer_radius_final)
        
        # Final validation
        valid = True
        if not packer.check_containment_all() or packer.check_overlaps():
            valid = False
            
        if not valid:
            # Fallback to previous working configuration
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
            outer_hex_side_length = 8
            return inner_hex_data, outer_hex_data, outer_hex_side_length

        # Format output with optimized positions
        inner_hex_data = np.zeros((12, 3))
        for i, hexagon in enumerate(packer.hexagons):
            inner_hex_data[i] = [hexagon.center_x, hexagon.center_y, hexagon.angle_degrees]
            
        outer_hex_data = np.array([outer_center_x, outer_center_y, outer_angle])
        outer_hex_side_length = outer_radius_final
        
        return inner_hex_data, outer_hex_data, outer_hex_side_length
        
    except Exception as e:
        # Fallback if optimization fails - use the differential evolution approach
        # Define bounds for optimization
        # Positions: x,y in [-10, 10], angles in [0, 360]
        # Outer radius should be reasonable
        bounds = []
        for _ in range(12):
            bounds.extend([(-10, 10), (-10, 10), (0, 360)])  # x, y, angle
        bounds.append((2.0, 15.0))  # outer_radius

        # Get initial configuration
        initial_guess = get_initial_guess_better()

        # Phase 1: Coarse global optimization with larger population
        start_time = time.time()

        # Use differential evolution for global optimization with increased population
        result = differential_evolution(
            evaluate_configuration_parallel,
            bounds,
            maxiter=150,
            popsize=25,  # Larger population for better exploration
            seed=42,
            disp=False,
            mutation=(0.5, 1.0),
            recombination=0.7,
            tol=1e-6
        )

        # Phase 2: Local refinement with L-BFGS-B if needed
        if result.fun < -0.25:  # If we haven't reached target yet, do local refinement
            # Refine using L-BFGS-B
            refined_result = minimize(
                evaluate_configuration_parallel,
                result.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-9}
            )
            if refined_result.fun < result.fun:
                result = refined_result

        end_time = time.time()

        # Extract results
        final_config = result.x
        positions_angles = final_config[:-1].reshape(-1, 3)
        outer_hex_side_length = final_config[-1]

        # Convert back to required format
        # The inner hex data is positions_angles
        inner_hex_data = positions_angles.copy()

        # Outer hex is centered at origin
        outer_hex_data = np.array([0, 0, 0])

        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END