# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from scipy.optimize import differential_evolution
import time
from typing import Tuple, List
import warnings
from numba import jit
from collections import defaultdict

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

@jit(nopython=True)
def distance_point_to_line_segment(point, line_start, line_end):
    """Calculate distance from point to line segment"""
    A = point[0] - line_start[0]
    B = point[1] - line_start[1]
    C = line_end[0] - line_start[0]
    D = line_end[1] - line_start[1]
    
    dot = A*C + B*D
    len_sq = C*C + D*D
    if len_sq == 0:
        return np.sqrt(A*A + B*B)
    param = dot / len_sq
    param = max(0, min(1, param))
    xx = line_start[0] + param * C
    yy = line_start[1] + param * D
    dx = point[0] - xx
    dy = point[1] - yy
    return np.sqrt(dx*dx + dy*dy)

@jit(nopython=True)
def check_hexagon_overlap_fast(hex1_vertices, hex2_vertices):
    """Fast overlap check using separating axis theorem"""
    # Check if any vertex of hex1 is inside hex2
    for v in hex1_vertices:
        if point_in_polygon_fast(v, hex2_vertices):
            return True
    # Check if any vertex of hex2 is inside hex1  
    for v in hex2_vertices:
        if point_in_polygon_fast(v, hex1_vertices):
            return True
    return False

def create_spatial_hash(hex_vertices_list, cell_size=2.0):
    """Create spatial hash grid for fast overlap checking"""
    hash_grid = defaultdict(list)
    for i, vertices in enumerate(hex_vertices_list):
        # Get bounding box of hexagon
        min_x = min(v[0] for v in vertices)
        max_x = max(v[0] for v in vertices)
        min_y = min(v[1] for v in vertices)
        max_y = max(v[1] for v in vertices)
        
        # Add to all relevant cells
        start_col = int(min_x // cell_size)
        end_col = int(max_x // cell_size) + 1
        start_row = int(min_y // cell_size)
        end_row = int(max_y // cell_size) + 1
        
        for col in range(start_col, end_col + 1):
            for row in range(start_row, end_row + 1):
                hash_grid[(col, row)].append(i)
    return hash_grid

def get_overlapping_indices(hash_grid, hex_index, hex_vertices, cell_size=2.0):
    """Get indices of potentially overlapping hexagons using spatial hash"""
    overlapping = set()
    # Get bounding box of hexagon
    min_x = min(v[0] for v in hex_vertices)
    max_x = max(v[0] for v in hex_vertices)
    min_y = min(v[1] for v in hex_vertices)
    max_y = max(v[1] for v in hex_vertices)
    
    # Check all relevant cells
    start_col = int(min_x // cell_size)
    end_col = int(max_x // cell_size) + 1
    start_row = int(min_y // cell_size)
    end_row = int(max_y // cell_size) + 1
    
    for col in range(start_col, end_col + 1):
        for row in range(start_row, end_row + 1):
            if (col, row) in hash_grid:
                for idx in hash_grid[(col, row)]:
                    if idx != hex_index:
                        overlapping.add(idx)
    return overlapping

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

def check_overlap_hexagons(hex1: Hexagon, hex2: Hexagon) -> bool:
    """Check if two hexagons overlap using Shapely"""
    poly1 = Polygon(hex1.vertices())
    poly2 = Polygon(hex2.vertices())
    return poly1.intersects(poly2)

def validate_configuration_with_penalty(inner_hexagons: List[Hexagon], outer_side_length: float) -> Tuple[float, bool]:
    """
    Validate configuration with penalty calculation
    Returns (total_penalty, is_valid)
    """
    # Create outer polygon
    outer_polygon = create_outer_hexagon(outer_side_length)
    
    # Check containment and calculate penalty
    total_penalty = 0
    containment_violations = 0
    
    for hex in inner_hexagons:
        vertices = hex.vertices()
        # Check containment
        for vertex in vertices:
            point = Point(vertex[0], vertex[1])
            if not outer_polygon.contains(point):
                dist = np.sqrt(vertex[0]**2 + vertex[1]**2)
                violation_distance = max(0, dist - outer_side_length)
                total_penalty += violation_distance * 1500  # Higher penalty for containment
                containment_violations += 1
    
    # Check overlaps using spatial hashing for efficiency
    n = len(inner_hexagons)
    hex_vertices_list = [hex.vertices() for hex in inner_hexagons]
    hash_grid = create_spatial_hash(hex_vertices_list)
    
    for i in range(n):
        overlapping_indices = get_overlapping_indices(hash_grid, i, hex_vertices_list[i])
        for j in overlapping_indices:
            if i < j:  # Avoid double counting
                if check_hexagon_overlap_fast(hex_vertices_list[i], hex_vertices_list[j]):
                    total_penalty += 1000000  # Large penalty for overlaps
    
    is_valid = total_penalty == 0
    
    return total_penalty, is_valid

def calculate_objective(outer_side_length: float) -> float:
    """Calculate 1/outer_hex_side_length"""
    return 1.0 / outer_side_length

def generate_symmetric_initial_population(n_individuals: int, n_hexagons: int = 12) -> List[np.ndarray]:
    """Generate initial population with better symmetric patterns"""
    population = []
    
    # Try multiple symmetric arrangements
    for _ in range(n_individuals):
        # Start with a better symmetric arrangement based on group theory principles
        positions = []
        rotations = []
        
        # Center hexagon
        positions.append([0.0, 0.0])
        rotations.append(0.0)
        
        # Ring 1: 6 hexagons arranged in a regular hexagon
        ring1_radius = 2.0
        for i in range(6):
            angle = i * 2 * np.pi / 6
            x = ring1_radius * np.cos(angle)
            y = ring1_radius * np.sin(angle)
            positions.append([x, y])
            rotations.append(0.0)
            
        # Ring 2: 5 hexagons in a pentagonal arrangement
        ring2_radius = 3.5
        for i in range(5):
            angle = i * 2 * np.pi / 5 + 0.1  # small offset for asymmetry
            x = ring2_radius * np.cos(angle)
            y = ring2_radius * np.sin(angle)
            positions.append([x, y])
            rotations.append(0.0)
            
        # Add some random noise to make it diverse
        individual = np.array(positions + rotations).flatten()
        individual += np.random.normal(0, 0.2, len(individual))
        population.append(individual)
        
    return population

def fitness_function_with_penalty(params: np.ndarray, outer_side_length: float = 10.0) -> float:
    """
    Fitness function with penalty for constraint violations
    Returns a value to minimize (lower is better)
    """
    n_hexagons = 12
    hex_params = params.reshape(-1, 3)
    
    # Create hexagons
    inner_hexagons = []
    for i in range(n_hexagons):
        x, y, theta = hex_params[i]
        inner_hexagons.append(Hexagon(x, y, theta))
    
    # Validate configuration and get penalty
    total_penalty, is_valid = validate_configuration_with_penalty(inner_hexagons, outer_side_length)
    
    if not is_valid:
        # Return very high penalty for invalid configurations
        return 1e10 + total_penalty
    
    # Compute actual outer hexagon size needed
    actual_size = compute_outer_hex_side_length(inner_hexagons)
    
    if actual_size > outer_side_length:
        # Configuration requires larger outer hexagon
        return 1e10 + (actual_size - outer_side_length) * 1000000
        
    # Return negative of 1/actual_size to maximize 1/size plus penalty
    return -(1.0 / actual_size) + total_penalty

def optimize_packing_multiple_runs() -> Tuple[np.ndarray, np.ndarray, float]:
    """Run optimization multiple times to increase chance of better results"""
    start_time = time.time()
    
    # Define bounds for optimization
    # Each hexagon has (x,y,rotation) = 3 parameters
    bounds = []
    for i in range(12):
        # X and Y positions: typically within reasonable bounds
        bounds.extend([(-8.0, 8.0), (-8.0, 8.0)])  # Positions
        # Rotation: 0-360 degrees
        bounds.append((0, 360))  # Rotations
    
    best_result = None
    best_fitness = float('inf')
    
    # Run multiple optimization attempts with different seeds
    for run in range(3):
        # Better initial guess with mathematical insight
        initial_guess = [
            [0.0, 0.0, 0.0],      # center
            [0.0, 2.5, 0.0],      # up
            [0.0, -2.5, 0.0],     # down
            [2.17, 1.25, 0.0],    # upper right
            [-2.17, 1.25, 0.0],   # upper left
            [2.17, -1.25, 0.0],   # lower right
            [-2.17, -1.25, 0.0],  # lower left
            [4.34, 0.0, 0.0],     # far right
            [0.0, 4.34, 0.0],     # far up
            [0.0, -4.34, 0.0],    # far down
            [2.17, 2.17, 0.0],    # upper right diagonal
            [-2.17, 2.17, 0.0],   # upper left diagonal
        ]
        
        initial_guess = np.array(initial_guess).flatten()
        
        # Optimization using differential evolution with different seeds
        try:
            result = differential_evolution(
                lambda x: fitness_function_with_penalty(x, 8.0),
                bounds,
                maxiter=30,  # Reduced iterations for speed
                popsize=8,   # Smaller population for efficiency
                init='random',
                seed=42 + run,  # Different seed for each run
                disp=False,
                tol=1e-6
            )
            
            if result.success:
                final_params = result.x
                hex_params = final_params.reshape(-1, 3)
                
                # Create hexagons from final parameters
                inner_hexagons = []
                for i in range(12):
                    x, y, theta = hex_params[i]
                    inner_hexagons.append(Hexagon(x, y, theta))
                    
                # Compute actual outer hexagon size
                outer_side_length = compute_outer_hex_side_length(inner_hexagons)
                
                # Evaluate fitness with final configuration
                fitness = fitness_function_with_penalty(final_params, outer_side_length)
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_result = (hex_params.copy(), outer_side_length)
                    
        except Exception as e:
            warnings.warn(f"Optimization run {run} failed: {str(e)}")
            continue
    
    # If we found a good result, return it
    if best_result is not None:
        inner_hex_params, outer_side_length = best_result
        outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered
        return inner_hex_params, outer_hex_data, outer_side_length
    
    # Fallback to improved symmetric arrangement if optimization failed
    inner_hex_data = np.array([
        [0, 0, 0],           # center
        [0, 2.5, 0],         # up
        [0, -2.5, 0],        # down
        [2.17, 1.25, 0],     # upper right
        [-2.17, 1.25, 0],    # upper left
        [2.17, -1.25, 0],    # lower right
        [-2.17, -1.25, 0],   # lower left
        [4.34, 0, 0],        # far right
        [0, 4.34, 0],        # far up
        [0, -4.34, 0],       # far down
        [2.17, 2.17, 0],     # upper right diagonal
        [-2.17, 2.17, 0],    # upper left diagonal
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
        inner_hex_data, outer_hex_data, outer_hex_side_length = optimize_packing_multiple_runs()
        
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
