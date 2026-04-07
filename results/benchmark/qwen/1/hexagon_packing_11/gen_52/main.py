# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import numba
from numba import jit
import time
from joblib import Parallel, delayed

@jit(nopython=True)
def hexagon_vertices(x, y, angle_deg, side_length=1):
    """Calculate vertices of a regular hexagon given center, angle, and side length"""
    angle_rad = np.deg2rad(angle_deg)
    vertices = np.zeros((6, 2))
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i, 0] = x + side_length * np.cos(theta)
        vertices[i, 1] = y + side_length * np.sin(theta)
    return vertices

@jit(nopython=True)
def distance_point_to_hexagon_edges(point, hex_vertices):
    """Calculate minimum distance from point to hexagon edges"""
    min_dist = np.inf
    for i in range(6):
        p1 = hex_vertices[i]
        p2 = hex_vertices[(i+1)%6]
        # Distance from point to line segment
        A = point[0] - p1[0]
        B = point[1] - p1[1]
        C = p2[0] - p1[0]
        D = p2[1] - p1[1]
        
        dot = A*C + B*D
        len_sq = C*C + D*D
        if len_sq == 0:
            dist = np.sqrt(A*A + B*B)
        else:
            param = dot / len_sq
            param = max(0, min(1, param))
            xx = p1[0] + param * C
            yy = p1[1] + param * D
            dx = point[0] - xx
            dy = point[1] - yy
            dist = np.sqrt(dx*dx + dy*dy)
        min_dist = min(min_dist, dist)
    return min_dist

def get_inner_hexagon_polygons(inner_data, side_length=1):
    """Get shapely polygons for all inner hexagons"""
    polygons = []
    for i in range(len(inner_data)):
        x, y, angle = inner_data[i]
        vertices = hexagon_vertices(x, y, angle, side_length)
        polygons.append(Polygon(vertices))
    return polygons

def get_outer_hexagon_polygon(center_x, center_y, side_length, angle_deg=0):
    """Get shapely polygon for outer hexagon"""
    vertices = hexagon_vertices(center_x, center_y, angle_deg, side_length)
    return Polygon(vertices)

def check_containment(inner_data, outer_polygon, side_length=1):
    """Check if all inner hexagons are contained within outer hexagon"""
    inner_polygons = get_inner_hexagon_polygons(inner_data, side_length)
    
    # Check containment for each inner polygon
    for poly in inner_polygons:
        if not outer_polygon.contains(poly):
            return False
    
    return True

def check_overlap(inner_data, side_length=1):
    """Check if any inner hexagons overlap"""
    inner_polygons = get_inner_hexagon_polygons(inner_data, side_length)
    
    # Check pairwise intersections
    for i in range(len(inner_polygons)):
        for j in range(i+1, len(inner_polygons)):
            if inner_polygons[i].intersects(inner_polygons[j]):
                return True
    
    return False

def calculate_outer_hex_side_length(inner_data, padding=0.01):
    """Estimate minimum outer hexagon side length that can contain all inner hexagons"""
    inner_polygons = get_inner_hexagon_polygons(inner_data)
    
    # Find bounding box of all inner hexagons
    all_points = []
    for poly in inner_polygons:
        for point in list(poly.exterior.coords):
            all_points.append(point)
    
    if len(all_points) == 0:
        return 1000
        
    all_points = np.array(all_points)
    min_x, max_x = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    min_y, max_y = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    
    # Calculate distance from center to corners
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Find maximum distance from center to any point
    max_dist = 0
    for point in all_points:
        dist = np.sqrt((point[0] - center_x)**2 + (point[1] - center_y)**2)
        max_dist = max(max_dist, dist)
    
    # Add some padding to ensure containment
    return max_dist * 1.05 + padding

def evaluate_solution(params):
    """Evaluate a solution - returns negative because we want to maximize 1/R"""
    # Reshape params
    inner_positions = params[:22].reshape(-1, 2)  # 11 hexagons x 2 coordinates
    inner_angles = params[22:33]  # 11 angles
    outer_center = params[33:35]  # outer hexagon center
    outer_angle = params[35]  # outer hexagon angle
    outer_side_length = params[36]  # outer hexagon side length
    
    # Create inner hexagon data
    inner_data = np.column_stack([inner_positions, inner_angles])
    
    # Create outer hexagon polygon
    outer_polygon = get_outer_hexagon_polygon(outer_center[0], outer_center[1], outer_side_length, outer_angle)
    
    # Check constraints
    if not check_containment(inner_data, outer_polygon):
        return 1e6  # Large penalty for containment violation
    
    if check_overlap(inner_data):
        return 1e6  # Large penalty for overlap
    
    # Return negative of 1/outer_side_length to maximize 1/outer_side_length
    # The actual value is positive since we're minimizing the negative
    return -1.0 / outer_side_length

def optimize_solution(initial_guess, bounds, maxiter=1000):
    """Run differential evolution optimization"""
    result = differential_evolution(
        evaluate_solution,
        bounds,
        maxiter=maxiter,
        popsize=50,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        disp=False,
        polish=True
    )
    
    return result

def generate_initial_configs():
    """Generate multiple diverse initial configurations"""
    configs = []
    
    # Configuration 1: Hexagonal pattern around center
    center_config = np.array([
        [0, 0, 0],      # Center
        [-1.75, 0, 0],  # Left
        [1.75, 0, 0],   # Right
        [0, 1.75, 0],   # Top
        [0, -1.75, 0],  # Bottom
        [-0.875, 0.875, 0],  # Top-left
        [0.875, 0.875, 0],   # Top-right
        [-0.875, -0.875, 0], # Bottom-left
        [0.875, -0.875, 0],  # Bottom-right
        [-1.75, 1.75, 0],    # Far-top-left
        [1.75, 1.75, 0],     # Far-top-right
    ])
    
    # Configuration 2: Spiral pattern
    spiral_config = np.array([
        [0, 0, 0],      # Center
        [1.5, 0, 0],    # Right
        [0, 1.5, 0],    # Top
        [-1.5, 0, 0],   # Left
        [0, -1.5, 0],   # Bottom
        [1.5, 1.5, 0],  # Top-right
        [-1.5, 1.5, 0], # Top-left
        [-1.5, -1.5, 0], # Bottom-left
        [1.5, -1.5, 0],  # Bottom-right
        [3.0, 0, 0],     # Far right
        [0, 3.0, 0],     # Far top
    ])
    
    # Configuration 3: Linear chain
    chain_config = np.array([
        [0, 0, 0],       # Center
        [1.5, 0, 0],     # Right
        [3.0, 0, 0],     # Right again
        [4.5, 0, 0],     # Right again
        [0, 1.5, 0],     # Up
        [0, 3.0, 0],     # Up again
        [0, 4.5, 0],     # Up again
        [1.5, 1.5, 0],   # Diagonal
        [3.0, 3.0, 0],   # Diagonal
        [4.5, 4.5, 0],   # Diagonal
        [-1.5, -1.5, 0], # Diagonal
    ])
    
    # Configuration 4: Clustered
    cluster_config = np.array([
        [0, 0, 0],       # Center
        [1.25, 0, 0],    # Right
        [-1.25, 0, 0],   # Left
        [0, 1.25, 0],    # Top
        [0, -1.25, 0],   # Bottom
        [1.25, 1.25, 0], # Top-right
        [-1.25, 1.25, 0], # Top-left
        [-1.25, -1.25, 0], # Bottom-left
        [1.25, -1.25, 0],  # Bottom-right
        [2.5, 0, 0],       # Far right
        [0, 2.5, 0],       # Far top
    ])
    
    # Configuration 5: Staggered pattern
    staggered_config = np.array([
        [0, 0, 0],       # Center
        [1.25, 0, 0],    # Right
        [-1.25, 0, 0],   # Left
        [0, 1.25, 0],    # Top
        [0, -1.25, 0],   # Bottom
        [0.625, 1.25, 0], # Top-right
        [-0.625, 1.25, 0], # Top-left
        [-0.625, -1.25, 0], # Bottom-left
        [0.625, -1.25, 0],  # Bottom-right
        [1.875, 1.25, 0],   # Far top-right
        [1.875, -1.25, 0],  # Far bottom-right
    ])
    
    configs = [center_config, spiral_config, chain_config, cluster_config, staggered_config]
    return configs

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Generate diverse initial configurations
    initial_configs = generate_initial_configs()
    
    best_result = None
    best_score = -np.inf
    
    # Try different configurations with optimization
    for i, config in enumerate(initial_configs):
        try:
            # Start with a reasonable estimate of outer hexagon side length
            estimated_side = calculate_outer_hex_side_length(config)
            
            # Set up bounds for optimization
            # Inner positions (x,y) - bounds set to reasonable ranges
            bounds = []
            for j in range(11):
                bounds.extend([(-10, 10), (-10, 10)])  # x,y positions
            
            # Inner angles (0 to 360)
            for _ in range(11):
                bounds.extend([(0, 360)])
            
            # Outer center (x,y) - bounded by reasonable range
            bounds.extend([(-10, 10), (-10, 10)])
            
            # Outer angle (0 to 360)
            bounds.extend([(0, 360)])
            
            # Outer side length (bounded to prevent extreme values)
            bounds.extend([(1, 20)])
            
            # Create initial guess
            initial_guess = []
            # Add inner positions
            for j in range(11):
                initial_guess.extend([config[j][0], config[j][1]])
            # Add inner angles
            for j in range(11):
                initial_guess.extend([config[j][2]])
            # Add outer center
            initial_guess.extend([0, 0])  # Start centered
            # Add outer angle
            initial_guess.extend([0])
            # Add outer side length
            initial_guess.extend([estimated_side])
            
            # Run optimization
            result = optimize_solution(initial_guess, bounds, maxiter=1000)
            
            # Evaluate final solution
            final_score = evaluate_solution(result.x)
            
            if final_score > best_score:
                best_score = final_score
                best_result = result
                
        except Exception as e:
            continue
    
    if best_result is None:
        # Fallback to original method if optimization fails
        n = 11
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
            [3.75, -2.17, 0],  # far bottom-right
        ])
        outer_hex_data = np.array([0, 0, 0])  # centered at origin
        outer_hex_side_length = 8  # large enough to contain all inner hexagons
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    # Extract results
    params = best_result.x
    inner_positions = params[:22].reshape(-1, 2)
    inner_angles = params[22:33]
    outer_center = params[33:35]
    outer_angle = params[35]
    outer_side_length = params[36]
    
    # Construct inner hex data
    inner_hex_data = np.column_stack([inner_positions, inner_angles])
    
    # Construct outer hex data
    outer_hex_data = np.array([outer_center[0], outer_center[1], outer_angle])
    
    return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END
