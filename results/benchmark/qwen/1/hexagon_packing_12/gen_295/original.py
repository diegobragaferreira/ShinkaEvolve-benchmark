# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
import time
from numba import jit
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

@jit(nopython=True)
def hexagon_vertices(x, y, angle_deg, side_length=1):
    """Compute vertices of a hexagon given center, rotation, and side length."""
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    # Vertices of regular hexagon with side length 1 centered at origin
    base_verts = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    # Rotate and translate
    rotated_verts = np.empty_like(base_verts)
    for i in range(6):
        x_orig, y_orig = base_verts[i]
        rotated_verts[i] = [
            x + side_length * (x_orig * cos_a - y_orig * sin_a),
            y + side_length * (x_orig * sin_a + y_orig * cos_a)
        ]

    return rotated_verts

@jit(nopython=True)
def point_in_hexagon(px, py, hx, hy, angle_deg, side_length=1):
    """Fast point-in-hexagon test using winding number."""
    vertices = hexagon_vertices(hx, hy, angle_deg, side_length)
    # Ray casting method
    n = len(vertices)
    inside = False
    p1x, p1y = vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = vertices[i % n]
        if py > min(p1y, p2y):
            if py <= max(p1y, p2y):
                if px <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or px <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

@jit(nopython=True)
def distance_point_to_line(px, py, x1, y1, x2, y2):
    """Calculate distance from point to line segment."""
    # Vector from (x1,y1) to (x2,y2)
    dx = x2 - x1
    dy = y2 - y1
    
    # Length squared of line segment
    length_sq = dx*dx + dy*dy
    
    if length_sq == 0:
        # Line segment is a point
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Project point onto line
    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    t = max(0, min(1, t))  # Clamp projection to line segment
    
    # Find closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    # Distance to closest point
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

@jit(nopython=True)
def hexagon_distance(h1_x, h1_y, h1_angle, h2_x, h2_y, h2_angle):
    """Compute minimum distance between two hexagons."""
    v1 = hexagon_vertices(h1_x, h1_y, h1_angle)
    v2 = hexagon_vertices(h2_x, h2_y, h2_angle)
    
    min_dist = np.inf
    for i in range(6):
        for j in range(6):
            # Distance between vertices
            dist = np.sqrt((v1[i,0]-v2[j,0])**2 + (v1[i,1]-v2[j,1])**2)
            if dist < min_dist:
                min_dist = dist
                
            # Distance from vertex to edge of other hexagon
            dist = distance_point_to_line(v1[i,0], v1[i,1], v2[j,0], v2[j,1], v2[(j+1)%6,0], v2[(j+1)%6,1])
            if dist < min_dist:
                min_dist = dist
                
            dist = distance_point_to_line(v2[j,0], v2[j,1], v1[i,0], v1[i,1], v1[(i+1)%6,0], v1[(i+1)%6,1])
            if dist < min_dist:
                min_dist = dist
    
    return min_dist

def compute_hexagon_polygon(x, y, angle_deg, side_length=1):
    """Convert hexagon parameters to shapely polygon."""
    vertices = hexagon_vertices(x, y, angle_deg, side_length)
    return Polygon(vertices)

def compute_outer_hexagon_radius(inner_hex_data):
    """Calculate minimum outer hexagon radius required to contain all inner hexagons."""
    max_dist = 0
    for i in range(len(inner_hex_data)):
        center_x, center_y, _ = inner_hex_data[i]
        # Distance from origin to hexagon center plus hexagon circumradius
        dist_to_center = np.sqrt(center_x**2 + center_y**2)
        hex_circumradius = 1.0  # Unit hexagon circumradius
        max_dist = max(max_dist, dist_to_center + hex_circumradius)
    return max_dist

def get_hexagon_vertices_array(hex_data):
    """Get all hexagon vertices as a numpy array for fast distance calculations."""
    n = len(hex_data)
    vertices_list = []
    
    for i in range(n):
        x, y, angle = hex_data[i]
        verts = hexagon_vertices(x, y, angle)
        vertices_list.append(verts)
    
    return vertices_list

def check_all_constraints(inner_hex_data, outer_radius):
    """Efficiently check all constraints for a configuration."""
    outer_hex_poly = compute_hexagon_polygon(0, 0, 0, outer_radius)
    
    # Check containment of all hexagons
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        hex_poly = compute_hexagon_polygon(x, y, angle)
        if not outer_hex_poly.contains(hex_poly):
            return False, "Containment violation"
    
    # Check overlaps between all pairs
    for i in range(len(inner_hex_data)):
        for j in range(i):
            x1, y1, angle1 = inner_hex_data[i]
            x2, y2, angle2 = inner_hex_data[j]
            hex1_poly = compute_hexagon_polygon(x1, y1, angle1)
            hex2_poly = compute_hexagon_polygon(x2, y2, angle2)
            
            if hex1_poly.intersects(hex2_poly) and not hex1_poly.touches(hex2_poly):
                return False, "Overlap violation"
    
    return True, "Valid"

def evaluate_constraint_violation(inner_hex_data, outer_radius):
    """Calculate constraint violation penalty for a configuration."""
    outer_hex_poly = compute_hexagon_polygon(0, 0, 0, outer_radius)
    
    penalty = 0.0
    
    # Check containment violations
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        hex_poly = compute_hexagon_polygon(x, y, angle)
        if not outer_hex_poly.contains(hex_poly):
            # Calculate how much it penetrates
            try:
                diff = outer_hex_poly.difference(hex_poly)
                if hasattr(diff, 'area'):
                    penalty += diff.area * 1000
            except:
                penalty += 10000
    
    # Check overlap violations
    for i in range(len(inner_hex_data)):
        for j in range(i):
            x1, y1, angle1 = inner_hex_data[i]
            x2, y2, angle2 = inner_hex_data[j]
            hex1_poly = compute_hexagon_polygon(x1, y1, angle1)
            hex2_poly = compute_hexagon_polygon(x2, y2, angle2)
            
            if hex1_poly.intersects(hex2_poly) and not hex1_poly.touches(hex2_poly):
                try:
                    overlap = hex1_poly.intersection(hex2_poly)
                    if hasattr(overlap, 'area'):
                        penalty += overlap.area * 1000
                except:
                    penalty += 10000
    
    return penalty

def solve_single_hexagon_position(hex_index, fixed_hex_data, outer_radius):
    """Optimize position of a single hexagon while keeping others fixed."""
    def objective(params):
        # Update the hexagon being optimized
        new_hex_data = fixed_hex_data.copy()
        new_hex_data[hex_index] = [params[0], params[1], params[2]]  # x, y, angle
        
        # Check constraints and return penalty
        valid, message = check_all_constraints(new_hex_data, outer_radius)
        if valid:
            # Return negative of inverse outer radius (since we want to maximize 1/R)
            return -1.0 / outer_radius
        else:
            # Return penalty for constraint violations
            return evaluate_constraint_violation(new_hex_data, outer_radius) + 1e6
    
    # Starting point
    x_start, y_start, angle_start = fixed_hex_data[hex_index]
    
    # Bounds for optimization
    bounds = [(-5, 5), (-5, 5), (0, 360)]
    
    # Use L-BFGS-B for local optimization
    try:
        result = minimize(
            objective,
            [x_start, y_start, angle_start],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 20, 'disp': False}
        )
        
        if result.success:
            return result.x
    except:
        pass
    
    # Return original if optimization fails
    return fixed_hex_data[hex_index]

def progressive_refinement(initial_config, max_iterations=50):
    """Perform progressive refinement of hexagon configuration."""
    # Copy initial configuration
    current_config = initial_config.copy()
    
    # Compute initial outer radius
    outer_radius = compute_outer_hexagon_radius(current_config)
    
    # Main refinement loop
    for iteration in range(max_iterations):
        improved = False
        
        # Try to improve each hexagon one at a time
        for i in range(len(current_config)):
            # Solve for this hexagon with all others fixed
            original_pos = current_config[i].copy()
            new_pos = solve_single_hexagon_position(i, current_config, outer_radius)
            
            # Check if improvement occurred
            if not np.allclose(original_pos, new_pos, atol=1e-4):
                current_config[i] = new_pos
                improved = True
                
        # Check if we can reduce outer radius
        new_outer_radius = compute_outer_hexagon_radius(current_config)
        if new_outer_radius < outer_radius - 1e-3:
            outer_radius = new_outer_radius
            improved = True
            
        # Early stopping if no improvements
        if not improved:
            break
            
    return current_config, outer_radius

def generate_initial_configuration():
    """Generate an improved initial configuration based on mathematical insight."""
    # Create a configuration inspired by optimal hexagon packings
    
    # Central hexagon
    config = [[0, 0, 0]]
    
    # First ring of 6 hexagons (at distance sqrt(3) from center)
    for i in range(6):
        angle = i * 60
        r = 1.732  # sqrt(3) for efficient packing
        x = r * np.cos(np.radians(angle))
        y = r * np.sin(np.radians(angle))
        config.append([x, y, 0])
    
    # Second ring of 6 hexagons (at distance 2*sqrt(3) from center)
    for i in range(6):
        angle = i * 60 + 30  # offset
        r = 3.464  # 2*sqrt(3)
        x = r * np.cos(np.radians(angle))
        y = r * np.sin(np.radians(angle))
        config.append([x, y, 0])
    
    # Add strategic center point
    config.append([0, -4.5, 0])
    
    return np.array(config[:12])

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
        # Generate initial configuration
        initial_config = generate_initial_configuration()
        
        # Apply progressive refinement
        refined_config, outer_radius = progressive_refinement(initial_config, max_iterations=30)
        
        # Final constraint check
        valid, message = check_all_constraints(refined_config, outer_radius)
        if not valid:
            # Re-optimize if needed
            refined_config, outer_radius = progressive_refinement(initial_config, max_iterations=50)
        
        # Convert to final format
        inner_hex_data = refined_config
        outer_hex_side_length = outer_radius * np.sqrt(3)  # Convert circumradius to side length
        outer_hex_data = np.array([0, 0, 0])
        
        end_time = time.time()
        eval_time = end_time - start_time
        
        return inner_hex_data, outer_hex_data, outer_hex_side_length
        
    except Exception as e:
        # Fallback to grid configuration if anything goes wrong
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
        outer_hex_side_length = 8  # Large enough to contain all inner hexagons

        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END