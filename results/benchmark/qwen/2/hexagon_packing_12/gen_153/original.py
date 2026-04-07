# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
import shapely.geometry as sg
from numba import jit
import time

@jit(nopython=True)
def hexagon_vertices(x, y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given position, angle, and side length"""
    angle_rad = np.radians(angle_deg)
    angles = np.arange(0, 6) * np.pi / 3
    vertices = np.zeros((6, 2))
    for i in range(6):
        vertices[i, 0] = x + side_length * np.cos(angles[i] + angle_rad)
        vertices[i, 1] = y + side_length * np.sin(angles[i] + angle_rad)
    return vertices

@jit(nopython=True)
def distance_point_to_hexagon(point, hex_vertices):
    """Calculate minimum distance from point to hexagon boundary"""
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

@jit(nopython=True)
def check_hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using separating axis theorem"""
    # Check if any vertex of hex1 is inside hex2
    for v in hex1_vertices:
        if point_in_polygon(v, hex2_vertices):
            return True
    # Check if any vertex of hex2 is inside hex1
    for v in hex2_vertices:
        if point_in_polygon(v, hex1_vertices):
            return True
    return False

@jit(nopython=True)
def point_in_polygon(point, polygon):
    """Check if point is inside polygon using ray casting"""
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

def create_initial_config():
    """Create an initial symmetric configuration based on group theory"""
    # Generate 12 hexagon positions in concentric rings
    positions = []

    # Center hexagon
    positions.append([0.0, 0.0, 0.0])

    # First ring around center (6 hexagons)
    for i in range(6):
        angle = i * 60
        radius = 2.0
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        positions.append([x, y, 0.0])

    # Second ring (5 hexagons)
    for i in range(5):
        angle = i * 72
        radius = 3.5
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        positions.append([x, y, 0.0])

    return np.array(positions)

def get_outer_hexagon_radius(inner_hex_data):
    """Compute the minimum radius required to contain all hexagons"""
    max_dist = 0
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        vertices = hexagon_vertices(x, y, angle)
        for vx, vy in vertices:
            dist = np.sqrt(vx*vx + vy*vy)
            max_dist = max(max_dist, dist)
    return max_dist + 1.0  # Add small margin

def evaluate_packing(inner_hex_data):
    """Evaluate how well a packing satisfies constraints"""
    n = len(inner_hex_data)
    penalty = 0

    # Check containment (penalize if hexagons extend beyond boundary)
    outer_radius = get_outer_hexagon_radius(inner_hex_data)
    outer_hex_vertices = hexagon_vertices(0, 0, 0, outer_radius)

    for i in range(n):
        x, y, angle = inner_hex_data[i]
        vertices = hexagon_vertices(x, y, angle)

        # Check if all vertices are inside outer hexagon
        for vx, vy in vertices:
            point = np.array([vx, vy])
            if not point_in_polygon(point, outer_hex_vertices):
                # Calculate penalty based on how much it extends beyond
                dist = np.sqrt(vx*vx + vy*vy)
                penalty += (dist - outer_radius + 0.5)**2

    # Check overlaps between hexagons (penalize overlapping areas)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, angle1 = inner_hex_data[i]
            x2, y2, angle2 = inner_hex_data[j]

            vertices1 = hexagon_vertices(x1, y1, angle1)
            vertices2 = hexagon_vertices(x2, y2, angle2)

            if check_hexagon_overlap(vertices1, vertices2):
                penalty += 1000000  # Large penalty for overlaps
                break

    return penalty

def optimize_packing():
    """Main optimization routine to find best packing"""
    # Start with a good initial configuration
    initial_config = create_initial_config()

    # Flatten the parameters for optimization
    initial_params = initial_config.flatten()

    def objective(params):
        # Reshape back to positions
        config = params.reshape(-1, 3)

        # Evaluate current penalty
        penalty = evaluate_packing(config)

        # The objective is to minimize penalty, which means maximize 1/R
        # We use a negative to convert to minimization problem
        outer_radius = get_outer_hexagon_radius(config)
        return penalty - 1.0/(outer_radius + 1e-6)  # Adding small epsilon for numerical stability

    def constraint_func(params):
        # This is a constraint: all hexagons should be completely inside
        config = params.reshape(-1, 3)
        return get_outer_hexagon_radius(config) - 100.0  # Upper bound

    cons = {'type': 'ineq', 'fun': constraint_func}

    # Perform optimization
    result = minimize(objective, initial_params, method='SLSQP', constraints=cons,
                      options={'maxiter': 1000, 'ftol': 1e-6})

    final_config = result.x.reshape(-1, 3)
    return final_config

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Optimize the packing configuration
    start_time = time.time()

    try:
        inner_hex_data = optimize_packing()
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        print(f"Fallback due to optimization error: {e}")
        inner_hex_data = create_initial_config()

    # Compute final outer hexagon size
    outer_hex_side_length = get_outer_hexagon_radius(inner_hex_data)

    # Create outer hexagon data (centered, no rotation)
    outer_hex_data = np.array([0, 0, 0])  # centered at origin, no rotation

    # Validate final configuration
    final_penalty = evaluate_packing(inner_hex_data)
    if final_penalty > 10000:  # If there are major overlaps or violations
        # Fallback to the original simpler solution
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
        outer_hex_side_length = 8  # large enough to contain all inner hexagons

    end_time = time.time()

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END