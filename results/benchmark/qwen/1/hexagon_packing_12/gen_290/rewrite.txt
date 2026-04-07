# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import time
from numba import jit
import warnings

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
    # Use ray casting method for simplicity and speed
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
def compute_min_distance_hexagon_hexagon(h1_x, h1_y, h1_angle, h2_x, h2_y, h2_angle):
    """Compute minimum distance between two hexagons using analytical approach."""
    v1 = hexagon_vertices(h1_x, h1_y, h1_angle)
    v2 = hexagon_vertices(h2_x, h2_y, h2_angle)

    min_dist = np.inf
    # Check vertex-to-vertex distances
    for i in range(6):
        for j in range(6):
            dist = np.sqrt((v1[i,0]-v2[j,0])**2 + (v1[i,1]-v2[j,1])**2)
            if dist < min_dist:
                min_dist = dist

    # Check vertex-to-edge distances
    for i in range(6):
        for j in range(6):
            # Distance from vertex v1[i] to edge v2[j]-v2[(j+1)%6]
            dist = distance_point_to_line(v1[i,0], v1[i,1], v2[j,0], v2[j,1], v2[(j+1)%6,0], v2[(j+1)%6,1])
            if dist < min_dist:
                min_dist = dist

            # Distance from vertex v2[j] to edge v1[i]-v1[(i+1)%6]
            dist = distance_point_to_line(v2[j,0], v2[j,1], v1[i,0], v1[i,1], v1[(i+1)%6,0], v1[(i+1)%6,1])
            if dist < min_dist:
                min_dist = dist

    return min_dist

def compute_hexagon_polygon(x, y, angle_deg, side_length=1):
    """Convert hexagon parameters to shapely polygon."""
    vertices = hexagon_vertices(x, y, angle_deg, side_length)
    return Polygon(vertices)

def compute_voronoi_area(points):
    """Compute areas of Voronoi regions for given points."""
    vor = Voronoi(points)
    areas = []
    for region in vor.regions:
        if len(region) > 0 and -1 not in region:
            # Compute polygon area
            vertices = [vor.vertices[i] for i in region]
            if len(vertices) >= 3:
                poly = Polygon(vertices)
                areas.append(poly.area)
            else:
                areas.append(0)
        else:
            areas.append(0)
    return areas

def generate_initial_voronoi_config():
    """Generate an initial configuration based on Voronoi tiling principles."""
    # Generate points arranged in a Voronoi-like pattern that naturally forms hexagonal structure
    points = []
    
    # Central point
    points.append([0.0, 0.0])
    
    # First ring - 6 points forming a hexagon
    for i in range(6):
        angle = i * np.pi/3
        radius = 1.732  # sqrt(3) - appropriate for tight packing
        points.append([radius * np.cos(angle), radius * np.sin(angle)])
    
    # Second ring - 6 points in a larger hexagon
    for i in range(6):
        angle = i * np.pi/3 + np.pi/6  # offset by 30 degrees for better packing
        radius = 3.464  # 2*sqrt(3)
        points.append([radius * np.cos(angle), radius * np.sin(angle)])
    
    # Additional points to make 12 total
    points.append([0.0, 3.0])
    points.append([0.0, -3.0])
    points.append([2.598, 1.5])  # sqrt(3)*0.5, 1.5
    points.append([-2.598, 1.5])
    points.append([2.598, -1.5])
    points.append([-2.598, -1.5])
    
    # Trim to exactly 12 points
    points = points[:12]
    
    # Add small random perturbations to break symmetry
    np.random.seed(42)
    for i in range(len(points)):
        points[i][0] += np.random.normal(0, 0.05)
        points[i][1] += np.random.normal(0, 0.05)
    
    return np.array(points)

def compute_voronoi_constraints(points):
    """Compute how well the Voronoi diagram matches hexagonal packing."""
    try:
        vor = Voronoi(points)
        # Count number of vertices per region (should be 6 for hexagons)
        vertex_counts = []
        for region in vor.regions:
            if len(region) > 0 and -1 not in region:
                vertex_counts.append(len(region))
        
        # Compute variance in vertex counts - lower is better
        if len(vertex_counts) > 0:
            mean_vertices = np.mean(vertex_counts)
            var_vertices = np.var(vertex_counts)
        else:
            mean_vertices = 0
            var_vertices = 1000
        
        # Compute compactness of regions
        areas = compute_voronoi_area(points)
        if len(areas) > 0:
            mean_area = np.mean(areas)
            var_area = np.var(areas)
        else:
            mean_area = 0
            var_area = 1000
            
        return {
            'mean_vertices': mean_vertices,
            'var_vertices': var_vertices,
            'mean_area': mean_area,
            'var_area': var_area
        }
    except:
        return {'mean_vertices': 0, 'var_vertices': 1000, 'mean_area': 0, 'var_area': 1000}

def evaluate_voronoi_fitness(points, outer_radius):
    """Evaluate fitness of Voronoi configuration."""
    try:
        # Create hexagon polygons
        hexagons = []
        for i in range(len(points)):
            x, y = points[i]
            # Use fixed rotation for simplicity
            hex_poly = compute_hexagon_polygon(x, y, 0.0, 1.0)
            hexagons.append(hex_poly)
        
        # Create outer hexagon polygon
        outer_hex_poly = compute_hexagon_polygon(0, 0, 0.0, outer_radius)
        
        # Check containment
        containment_valid = True
        for hex_poly in hexagons:
            center = hex_poly.centroid
            if not outer_hex_poly.contains(center):
                containment_valid = False
                break
            
        if not containment_valid:
            return 1e10
        
        # Check overlaps
        overlap_penalty = 0.0
        for i in range(len(hexagons)):
            for j in range(i+1, len(hexagons)):
                if hexagons[i].intersects(hexagons[j]) and not hexagons[i].touches(hexagons[j]):
                    try:
                        overlap = hexagons[i].intersection(hexagons[j])
                        if hasattr(overlap, 'area') and overlap.area > 0:
                            overlap_penalty += overlap.area
                    except:
                        overlap_penalty += 1000
                        
        if overlap_penalty > 0:
            return 1e10 + overlap_penalty * 10000
        
        # If valid, return inverse of outer radius
        return -1.0 / outer_radius
        
    except Exception as e:
        return 1e10

def voronoi_hexagon_packing_optimization(max_iter=1000):
    """Optimize using Voronoi-based approach with iterative improvement."""
    # Generate initial Voronoi configuration
    points = generate_initial_voronoi_config()
    
    # Estimate initial outer radius
    max_dist = 0
    for point in points:
        dist = np.sqrt(point[0]**2 + point[1]**2)
        max_dist = max(max_dist, dist)
    outer_radius = max_dist + 2.0  # Add margin for hexagon size
    
    best_points = points.copy()
    best_outer_radius = outer_radius
    best_fitness = evaluate_voronoi_fitness(points, outer_radius)
    
    # Iterative improvement using gradient-based approach
    learning_rate = 0.01
    
    for iteration in range(max_iter):
        # Compute current fitness
        current_fitness = evaluate_voronoi_fitness(points, outer_radius)
        
        # If we're improving or at a good solution, keep going
        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_points = points.copy()
            best_outer_radius = outer_radius
            
        # Simple gradient descent step
        # Compute gradients by finite differences
        gradients = np.zeros_like(points)
        epsilon = 1e-4
        
        for i in range(len(points)):
            for j in range(len(points[i])):
                # Perturb point
                points_plus = points.copy()
                points_minus = points.copy()
                points_plus[i, j] += epsilon
                points_minus[i, j] -= epsilon
                
                # Evaluate fitness
                fitness_plus = evaluate_voronoi_fitness(points_plus, outer_radius)
                fitness_minus = evaluate_voronoi_fitness(points_minus, outer_radius)
                
                # Compute gradient
                gradients[i, j] = (fitness_plus - fitness_minus) / (2 * epsilon)
        
        # Update points
        points -= learning_rate * gradients
        
        # Keep points within reasonable bounds
        for i in range(len(points)):
            # Limit to reasonable area
            dist = np.sqrt(points[i, 0]**2 + points[i, 1]**2)
            if dist > 10:
                points[i] = points[i] / dist * 10
                
        # Adjust outer radius based on current configuration
        new_max_dist = 0
        for point in points:
            dist = np.sqrt(point[0]**2 + point[1]**2)
            new_max_dist = max(new_max_dist, dist)
        outer_radius = new_max_dist + 2.0  # Add margin for hexagons
        
        # Early stopping if we're getting close to target
        if abs(best_fitness) < 0.2530:  # Near target value
            if iteration > max_iter // 2:
                break
    
    return best_points, best_outer_radius

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
        # Use Voronoi-based optimization
        points, outer_radius = voronoi_hexagon_packing_optimization()
        
        # Convert points to hexagon parameter format
        # Each point represents the center of a hexagon with 0 rotation
        inner_hex_data = np.zeros((12, 3))
        for i in range(12):
            inner_hex_data[i] = [points[i][0], points[i][1], 0.0]
        
        # Create outer hexagon data (centered at origin, no rotation)
        outer_hex_data = np.array([0, 0, 0])
        
        # Ensure we don't exceed time limits
        end_time = time.time()
        eval_time = end_time - start_time
        
        return inner_hex_data, outer_hex_data, outer_radius
        
    except Exception as e:
        # Fallback to a known good configuration if optimization fails
        warnings.warn(f"Voronoi optimization failed: {e}")
        
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
        outer_hex_side_length = 8  # Large enough to contain all inner hexagons

        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END