# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import random
from scipy.spatial import Voronoi, distance
import time
from scipy.optimize import minimize

# Constants
NUM_INNER_HEXAGONS = 11
UNIT_HEXAGON_RADIUS = 1.0  # Distance from center to corner for unit hexagon
MAX_EVAL_TIME = 180.0  # seconds

# Precomputed unit hexagon vertices (centered at origin)
def get_unit_hexagon_vertices():
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 angles + close the loop
    vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    return vertices

UNIT_HEXAGON_VERTICES = get_unit_hexagon_vertices()

def rotate_point(point, angle_rad):
    """Rotate a point around origin"""
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([point[0]*cos_a - point[1]*sin_a, point[0]*sin_a + point[1]*cos_a])

def hexagon_vertices(center, angle_rad, scale=1.0):
    """Get vertices of a hexagon at given position and rotation"""
    rotated_vertices = np.array([rotate_point(v, angle_rad) for v in UNIT_HEXAGON_VERTICES])
    return rotated_vertices * scale + np.array(center)

def point_in_polygon(point, polygon):
    """Fast point-in-polygon check"""
    return polygon.contains(Point(point))

def calculate_outer_hexagon_radius(inner_hex_data, outer_center=[0,0], outer_angle=0):
    """Calculate minimum radius needed for outer hexagon to contain all inner hexagons"""
    max_dist = 0
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = np.radians(inner_hex_data[i][2])

        # Get all vertices of this hexagon
        vertices = hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)

        # Calculate max distance from outer center to any vertex
        for vertex in vertices:
            dist = np.linalg.norm(np.array(vertex) - np.array(outer_center))
            max_dist = max(max_dist, dist)

    return max_dist

def validate_solution(inner_hex_data, outer_center=[0,0], outer_angle=0):
    """Validate solution: check containment and non-overlap"""
    # Precompute all hexagon polygons once for reuse
    hex_polygons = []
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = np.radians(inner_hex_data[i][2])
        vertices = hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)
        hex_polygons.append(Polygon(vertices))

    # Calculate outer radius once
    outer_radius = calculate_outer_hexagon_radius(inner_hex_data, outer_center, outer_angle)

    # Check containment using the outer hexagon polygon
    outer_vertices = hexagon_vertices(outer_center, outer_angle, outer_radius)
    outer_polygon = Polygon(outer_vertices)

    # Check if all inner hexagons are contained within outer hexagon
    for hex_poly in hex_polygons:
        # Fast check: if any vertex is outside, reject
        for vertex in hex_poly.exterior.coords[:-1]:  # Exclude closing vertex
            if not outer_polygon.contains(Point(vertex)):
                return False

    # Check overlaps efficiently using direct polygon intersection
    for i in range(len(hex_polygons)):
        for j in range(i+1, len(hex_polygons)):
            if hex_polygons[i].intersects(hex_polygons[j]):
                return False

    return True

def evaluate_fitness(inner_hex_data, outer_center=[0,0], outer_angle=0):
    """Evaluate fitness (negative of outer hexagon radius for maximization)"""
    # Calculate minimum outer radius needed
    outer_radius = calculate_outer_hexagon_radius(inner_hex_data, outer_center, outer_angle)

    # If solution is invalid, penalize heavily
    if not validate_solution(inner_hex_data, outer_center, outer_angle):
        return -1e10  # Very poor fitness

    # Return negative radius (we want to minimize radius, so maximize negative value)
    return -outer_radius

def create_voronoi_based_configuration():
    """Create initial configuration based on Voronoi tessellation theory"""
    # Generate points that naturally form hexagonal patterns
    # Start with central point
    points = [[0, 0]]
    
    # Add surrounding points in hexagonal arrangement
    # Layer 1: 6 points around the center
    layer1_radius = 2.0  # Distance between centers of adjacent hexagons
    for i in range(6):
        angle = i * np.pi / 3
        x = layer1_radius * np.cos(angle)
        y = layer1_radius * np.sin(angle)
        points.append([x, y])
    
    # Layer 2: Additional points for the 11th hexagon
    # Place one more hexagon in a strategic location
    layer2_radius = 3.5
    angles = [np.pi/6, 5*np.pi/6, 3*np.pi/2, 7*np.pi/6, 11*np.pi/6, np.pi/2]
    for i in range(5):
        angle = angles[i]
        x = layer2_radius * np.cos(angle)
        y = layer2_radius * np.sin(angle)
        points.append([x, y])
    
    # Ensure we have exactly 11 points
    while len(points) < 11:
        # Add random points to fill
        points.append([random.uniform(-4, 4), random.uniform(-4, 4)])
    
    # Trim to exactly 11 points
    points = points[:11]
    
    # Create configuration array with default angles
    config = []
    for point in points:
        config.append([point[0], point[1], 0.0])
    
    return np.array(config)

def voronoi_optimization_step(config, max_iter=100):
    """Optimize configuration using Voronoi-based geometric refinement"""
    # Create Voronoi diagram from current configuration
    points = config[:, :2]
    
    try:
        vor = Voronoi(points)
        
        # Refine the configuration by moving points to Voronoi cell centroids
        new_points = []
        for i, point in enumerate(points):
            # Get Voronoi vertices for this cell
            region_indices = np.where(vor.point_region == i)[0]
            if len(region_indices) > 0:
                # Find the centroid of the Voronoi region
                region = vor.regions[region_indices[0]]
                if len(region) > 0 and -1 not in region:
                    vertices = [vor.vertices[r] for r in region if r >= 0]
                    if len(vertices) > 0:
                        centroid = np.mean(vertices, axis=0)
                        # Keep points within reasonable bounds
                        bounded_centroid = [
                            np.clip(centroid[0], -8, 8),
                            np.clip(centroid[1], -8, 8)
                        ]
                        new_points.append(bounded_centroid)
                    else:
                        new_points.append(point)
                else:
                    new_points.append(point)
            else:
                new_points.append(point)
        
        # Update configuration with refined points
        refined_config = config.copy()
        for i in range(min(len(new_points), len(refined_config))):
            refined_config[i][:2] = new_points[i]
            # Preserve angles
            if i >= 1:  # Keep some structural arrangement
                refined_config[i][2] = config[i][2]
        
        return refined_config
        
    except Exception:
        # If Voronoi fails, return original
        return config

def geometric_refinement(config, max_iter=50):
    """Apply geometric refinement to improve arrangement"""
    refined_config = config.copy()
    
    # Iteratively improve the configuration
    for iter_count in range(max_iter):
        # Try to move hexagons closer together while maintaining non-overlap
        for i in range(len(refined_config)):
            # Save current position
            current_pos = refined_config[i][:2].copy()
            
            # Try nearby positions that maintain some structure
            best_pos = current_pos.copy()
            best_score = evaluate_fitness(refined_config)
            
            # Test small displacements
            for dx in [-0.2, -0.1, 0, 0.1, 0.2]:
                for dy in [-0.2, -0.1, 0, 0.1, 0.2]:
                    test_config = refined_config.copy()
                    test_config[i][0] = current_pos[0] + dx
                    test_config[i][1] = current_pos[1] + dy
                    
                    if validate_solution(test_config):
                        score = evaluate_fitness(test_config)
                        if score > best_score:
                            best_score = score
                            best_pos = [current_pos[0] + dx, current_pos[1] + dy]
            
            # Apply best move
            refined_config[i][:2] = best_pos
            
    return refined_config

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Start with Voronoi-based configuration
    best_config = create_voronoi_based_configuration()
    best_radius = -evaluate_fitness(best_config) if evaluate_fitness(best_config) != float('-inf') else 10.0
    
    # Apply Voronoi refinement
    refined_config = voronoi_optimization_step(best_config)
    refined_radius = -evaluate_fitness(refined_config) if evaluate_fitness(refined_config) != float('-inf') else 10.0
    
    if refined_radius < best_radius:
        best_config = refined_config
        best_radius = refined_radius
    
    # Apply geometric refinement
    final_config = geometric_refinement(best_config)
    final_radius = -evaluate_fitness(final_config) if evaluate_fitness(final_config) != float('-inf') else 10.0
    
    if final_radius < best_radius:
        best_config = final_config
        best_radius = final_radius
    
    # Final verification and improvement
    if not validate_solution(best_config):
        # Fall back to a known good configuration
        best_config = np.array([
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
        ])
        best_radius = 8.0
    
    # Return result
    inner_hex_data = best_config
    outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered at origin
    outer_hex_side_length = best_radius
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END