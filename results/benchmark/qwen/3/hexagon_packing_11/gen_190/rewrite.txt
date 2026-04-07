# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from scipy.optimize import minimize
from joblib import Parallel, delayed
import warnings
from numba import jit, prange
from typing import Tuple, List, Optional

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

@jit(nopython=True)
def generate_hexagon_vertices_numba(center_x: float, center_y: float, 
                                   side_length: float, rotation_rad: float):
    """Fast JIT version of hexagon vertex generation using Numba."""
    angles = np.linspace(0, 2*np.pi, 7) + rotation_rad
    vertices_x = np.empty(6)
    vertices_y = np.empty(6)
    for i in range(6):
        vertices_x[i] = center_x + side_length * np.cos(angles[i])
        vertices_y[i] = center_y + side_length * np.sin(angles[i])
    return vertices_x, vertices_y

@jit(nopython=True)
def point_in_hexagon_numba(px: float, py: float, hex_center_x: float, hex_center_y: float, 
                          side_length: float, rotation_rad: float):
    """Fast point-in-hexagon check using analytical geometry."""
    # Transform point to hexagon reference frame
    dx = px - hex_center_x
    dy = py - hex_center_y

    # Rotate back to hexagon's local coordinate system
    cos_r = np.cos(-rotation_rad)
    sin_r = np.sin(-rotation_rad)
    local_x = dx * cos_r - dy * sin_r
    local_y = dx * sin_r + dy * cos_r

    # Check distance from center
    dist_to_center = np.sqrt(local_x**2 + local_y**2)
    
    # For regular hexagon with side length s, distance from center to corner is s
    # Distance from center to edge is s * sqrt(3)/2
    edge_distance = side_length * np.sqrt(3) / 2
    
    # If too far or too close, we know the answer
    if dist_to_center > side_length:
        return False
    if dist_to_center < edge_distance:
        return True
        
    # Check if point is inside the hexagon by checking all edges
    # For a hexagon with vertices at angles 0, π/3, 2π/3, π, 4π/3, 5π/3
    # We check if point is on the correct side of each edge
    for i in range(6):
        angle1 = i * np.pi / 3
        angle2 = ((i+1) % 6) * np.pi / 3
        x1 = side_length * np.cos(angle1)
        y1 = side_length * np.sin(angle1)
        x2 = side_length * np.cos(angle2)
        y2 = side_length * np.sin(angle2)
        
        # Vector from point to first edge point
        v1x = x1 - local_x
        v1y = y1 - local_y
        # Vector along edge
        v2x = x2 - x1
        v2y = y2 - y1

        # Cross product to determine which side of the edge the point is on
        cross_product = v1x * v2y - v1y * v2x
        
        # For the correct side, cross product should be non-negative
        # (point on the inside of the edge)
        if cross_product < 0:
            return False
            
    return True

@jit(nopython=True)
def hexagon_overlap_numba(hex1_center_x: float, hex1_center_y: float, hex1_side: float, 
                         hex1_rot: float, hex2_center_x: float, hex2_center_y: float, 
                         hex2_side: float, hex2_rot: float):
    """Fast hexagon overlap check using distance-based approximation."""
    # Fast approximation: if distance between centers is less than sum of circumradii,
    # they might overlap
    dx = hex1_center_x - hex2_center_x
    dy = hex1_center_y - hex2_center_y
    distance = np.sqrt(dx*dx + dy*dy)
    
    # Circumradius of unit hexagon is 1
    return distance < 2.0  # Sum of circumradii

def generate_hexagon_vertices(center_x: float, center_y: float, 
                            side_length: float = 1, rotation_deg: float = 0) -> np.ndarray:
    """Generate vertices of a regular hexagon given center, side length, and rotation."""
    rotation_rad = np.radians(rotation_deg)
    angles = np.linspace(0, 2*np.pi, 7) + rotation_rad
    vertices = np.column_stack([
        center_x + side_length * np.cos(angles),
        center_y + side_length * np.sin(angles)
    ])
    return vertices[:-1]  # Remove duplicate last vertex

def check_containment_fast(hexagon_vertices: np.ndarray, outer_hex_vertices: np.ndarray) -> bool:
    """Fast containment check using Numba JIT."""
    # Get outer hexagon center
    outer_center_x = np.mean(outer_hex_vertices[:, 0])
    outer_center_y = np.mean(outer_hex_vertices[:, 1])
    
    # Simple distance check first
    inner_center_x = np.mean(hexagon_vertices[:, 0])
    inner_center_y = np.mean(hexagon_vertices[:, 1])
    
    dx = inner_center_x - outer_center_x
    dy = inner_center_y - outer_center_y
    distance_from_center = np.sqrt(dx*dx + dy*dy)
    
    # Estimate outer radius based on vertices
    max_outer_dist = 0
    for vertex in outer_hex_vertices:
        dist = np.sqrt((vertex[0] - outer_center_x)**2 + (vertex[1] - outer_center_y)**2)
        max_outer_dist = max(max_outer_dist, dist)
    
    # If center is too far away, definitely not contained
    if distance_from_center > max_outer_dist + 1.0:
        return False
    
    # Use Numba for vertex-by-vertex containment check
    outer_center_x_numba = float(outer_center_x)
    outer_center_y_numba = float(outer_center_y)
    
    # Check that all vertices are inside the outer hexagon
    for vertex in hexagon_vertices:
        px, py = vertex[0], vertex[1]
        # For simplicity, we use a basic approach: check if the point is close enough to center
        # and then verify with shapely for correctness (this is a trade-off for speed vs accuracy)
        pass
    
    # Fall back to original approach for reliability
    outer_polygon = Polygon(outer_hex_vertices)
    for vertex in hexagon_vertices:
        point = Point(vertex[0], vertex[1])
        if not outer_polygon.contains(point):
            return False

    return True

def check_overlap_fast(hex1_vertices: np.ndarray, hex2_vertices: np.ndarray) -> bool:
    """Fast overlap check using distance approximation."""
    # Quick distance check using centroids
    centroid1 = np.mean(hex1_vertices, axis=0)
    centroid2 = np.mean(hex2_vertices, axis=0)

    dx = centroid1[0] - centroid2[0]
    dy = centroid1[1] - centroid2[1]
    distance = np.sqrt(dx*dx + dy*dy)

    # If distance is greater than sum of circumradii, no overlap
    # For unit hexagons, circumradius is 1
    if distance >= 2.0:
        return False
    
    # Use Shapely for final determination (more reliable)
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

class HexagonGeometry:
    """Handles all geometric computations for hexagons with optimized vertex generation"""

    def __init__(self):
        # Pre-compute unit hexagon vertices once for efficiency
        self._unit_vertices = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])

    def get_transformed_vertices(self, center_x: float, center_y: float, angle_deg: float, side_length: float = 1.0) -> np.ndarray:
        """Efficiently compute transformed hexagon vertices with caching"""
        # Get unit vertices and scale
        vertices = self._unit_vertices * side_length

        # Apply rotation
        angle_rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = vertices @ rotation_matrix.T

        # Apply translation
        return rotated + np.array([center_x, center_y])

    def create_hexagon_polygon(self, center_x: float, center_y: float, angle_deg: float, side_length: float = 1.0) -> Polygon:
        """Create shapely polygon for hexagon with precomputed vertices"""
        vertices = self.get_transformed_vertices(center_x, center_y, angle_deg, side_length)
        return Polygon(vertices)

class HexagonValidator:
    """Handles constraint checking for hexagon packing with optimized operations"""

    def __init__(self, geometry: HexagonGeometry):
        self.geometry = geometry

    def check_containment(self, hexagons: List[Tuple[float, float, float]], outer_radius: float) -> bool:
        """Check if all hexagons are contained within outer hexagon of given radius"""
        # Create outer hexagon once
        outer_polygon = self.geometry.create_hexagon_polygon(0.0, 0.0, 0.0, outer_radius)

        # Check each hexagon against outer polygon
        for center_x, center_y, angle_deg in hexagons:
            hex_polygon = self.geometry.create_hexagon_polygon(center_x, center_y, angle_deg)
            if not outer_polygon.contains(hex_polygon):
                return False
        return True

    def check_overlap(self, hexagons: List[Tuple[float, float, float]]) -> bool:
        """Check if any hexagons overlap using spatial indexing for efficiency"""
        # Create polygons once
        polygons = [self.geometry.create_hexagon_polygon(center_x, center_y, angle_deg) 
                   for center_x, center_y, angle_deg in hexagons]
        
        # Check pairwise overlaps
        for i in range(len(polygons)):
            for j in range(i+1, len(polygons)):
                if polygons[i].intersects(polygons[j]):
                    return True
        return False

class SolutionManager:
    """Manages solution representation, validation and output formatting"""

    def __init__(self, geometry: HexagonGeometry, validator: HexagonValidator):
        self.geometry = geometry
        self.validator = validator

    def validate_solution(self, hex_data: np.ndarray, outer_radius: float) -> bool:
        """Validate that solution meets all constraints"""
        # Convert array to list of tuples for validator
        hexagons = [(row[0], row[1], row[2]) for row in hex_data]
        
        # Check constraints
        if not self.validator.check_containment(hexagons, outer_radius):
            return False
        if self.validator.check_overlap(hexagons):
            return False
        return True

    def format_output(self, hex_data: np.ndarray, outer_radius: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Format final solution for output"""
        # Inner hex data
        inner_hex_data = hex_data.copy()
        
        # Outer hex data (centered at origin with zero rotation)
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        
        # Outer hex side length
        outer_hex_side_length = outer_radius
        
        return inner_hex_data, outer_hex_data, outer_hex_side_length

class PackingProblem:
    """Main optimization class coordinating the hexagon packing process"""

    def __init__(self, n_inner_hexagons: int = 11, hex_side_length: float = 1.0):
        self.n_inner_hexagons = n_inner_hexagons
        self.hex_side_length = hex_side_length
        
        # Initialize components
        self.geometry = HexagonGeometry()
        self.validator = HexagonValidator(self.geometry)
        self.solver = SolutionManager(self.geometry, self.validator)

    def create_hexagon_list(self, hex_data: np.ndarray) -> List[Tuple[float, float, float]]:
        """Convert numpy array to list of hexagon tuples"""
        return [(row[0], row[1], row[2]) for row in hex_data]

    def evaluate_fitness(self, hex_data: np.ndarray, outer_radius: float) -> float:
        """Evaluate fitness based on geometric constraints and packing density"""
        # Convert to tuple format for validation
        hexagons = self.create_hexagon_list(hex_data)
        
        # Check constraints
        if not self.validator.check_containment(hexagons, outer_radius):
            return -np.inf  # Invalid - penalty
        
        if self.validator.check_overlap(hexagons):
            return -np.inf  # Invalid - penalty
            
        # Valid configuration - maximize 1/outer_radius (minimize outer_radius)
        return 1.0 / outer_radius

    def find_optimal_radius(self, hex_data: np.ndarray, min_radius: float = 1.0, max_radius: float = 10.0) -> float:
        """Find minimum radius that contains all hexagons using binary search"""
        # Convert to tuple format for validation
        hexagons = self.create_hexagon_list(hex_data)
        
        # First check if configuration fits at all
        if self.validator.check_containment(hexagons, min_radius):
            return min_radius
            
        # Binary search with early termination
        left, right = min_radius, max_radius
        iterations = 0
        max_iterations = 20
        
        # Reduce iterations by checking if left boundary works
        if self.validator.check_containment(hexagons, left):
            return left
            
        while iterations < max_iterations and abs(right - left) > 0.001:
            mid = (left + right) / 2
            if self.validator.check_containment(hexagons, mid):
                right = mid
            else:
                left = mid
            iterations += 1
                
        return right

    def optimize_local(self, hex_data: np.ndarray, outer_radius: float) -> np.ndarray:
        """Refine solution locally using optimization"""
        def objective(params):
            # Reshape params back to hexagon data
            new_data = hex_data.copy()
            for i in range(len(new_data)):
                new_data[i][0] = params[i*3]
                new_data[i][1] = params[i*3+1]
                new_data[i][2] = params[i*3+2]
            
            # Evaluate fitness
            fitness = self.evaluate_fitness(new_data, outer_radius)
            return -fitness  # minimize negative fitness
        
        # Flatten the data for optimization
        initial_params = []
        for i in range(len(hex_data)):
            initial_params.extend([hex_data[i][0], hex_data[i][1], hex_data[i][2]])
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(objective, initial_params, method='L-BFGS-B', 
                            bounds=[(-10, 10), (-10, 10), (0, 360)] * len(hex_data),
                            options={'maxiter': 100})
            if result.success:
                # Reshape optimized result back
                refined_data = hex_data.copy()
                for i in range(len(refined_data)):
                    refined_data[i][0] = result.x[i*3]
                    refined_data[i][1] = result.x[i*3+1]
                    refined_data[i][2] = result.x[i*3+2]
                return refined_data
        except:
            pass
        return hex_data

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Initialize the packing problem
    problem = PackingProblem(n_inner_hexagons=11, hex_side_length=1.0)
    
    # Start with a good heuristic initial configuration
    # Based on hexagonal tiling pattern
    initial_config = np.array([
        [0, 0, 0],        # center
        [-2.5, 0, 0],     # left
        [2.5, 0, 0],      # right
        [-1.25, 2.17, 0], # top-left
        [1.25, 2.17, 0],  # top-right
        [-1.25, -2.17, 0],# bottom-left
        [1.25, -2.17, 0], # bottom-right
        [-3.75, 2.17, 0], # far top-left
        [3.75, 2.17, 0],  # far top-right
        [-3.75, -2.17, 0],# far bottom-left
        [3.75, -2.17, 0], # far bottom-right
    ])
    
    # Simple optimization approach for time constraints
    best_fitness = -np.inf
    best_config = initial_config.copy()
    best_radius = 10.0
    
    # Direct local optimization approach instead of full evolutionary search
    # to reduce execution time while maintaining reasonable quality
    
    # Try local optimization on initial configuration
    current_config = initial_config.copy()
    
    # Run several rounds of local optimization with different random perturbations
    for _ in range(5):  # Reduced iterations for speed
        # Apply small random perturbations to current solution
        perturbed = current_config.copy()
        for i in range(len(perturbed)):
            # Small random perturbations to position
            perturbed[i][0] += np.random.normal(0, 0.2)
            perturbed[i][1] += np.random.normal(0, 0.2)
            # Mutate angle slightly
            perturbed[i][2] += np.random.normal(0, 10)
            # Keep angle in [0, 360)
            perturbed[i][2] = perturbed[i][2] % 360
        
        # Find optimal radius for the perturbed configuration
        radius = problem.find_optimal_radius(perturbed)
        fitness = problem.evaluate_fitness(perturbed, radius)
        
        # Update best solution if better
        if fitness > best_fitness:
            best_fitness = fitness
            best_config = perturbed.copy()
            best_radius = radius
    
    # Final local optimization on best solution found
    refined_config = problem.optimize_local(best_config, best_radius)
    final_radius = problem.find_optimal_radius(refined_config)
    final_fitness = problem.evaluate_fitness(refined_config, final_radius)
    
    if final_fitness > best_fitness:
        best_config = refined_config
        best_radius = final_radius
        best_fitness = final_fitness
    
    # Validate final solution
    if not problem.solver.validate_solution(best_config, best_radius):
        # If validation fails, fall back to initial config
        best_config = initial_config.copy()
        best_radius = problem.find_optimal_radius(best_config)
    
    # Format output
    inner_hex_data, outer_hex_data, outer_hex_side_length = problem.solver.format_output(
        best_config, best_radius
    )
    
    end_time = time.time()
    eval_time = end_time - start_time
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END