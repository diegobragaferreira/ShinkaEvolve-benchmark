# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
import random
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import math

# Constants
NUM_INNER_HEXAGONS = 11
UNIT_HEXAGON_RADIUS = 1.0
MAX_EVAL_TIME = 180.0

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

def point_in_polygon(point, polygon):
    """Fast point-in-polygon check"""
    return polygon.contains(Point(point))

def is_contained_in_outer_hexagon(hexagon_vertices_list, outer_center, outer_angle, outer_radius):
    """Check if hexagon is fully contained in outer hexagon"""
    outer_vertices = hexagon_vertices(outer_center, outer_angle, outer_radius)
    outer_polygon = Polygon(outer_vertices)
    for vertex in hexagon_vertices_list:
        if not point_in_polygon(vertex, outer_polygon):
            return False
    return True

def check_overlap_fast(hex1_vertices, hex2_vertices):
    """Fast overlap check using polygon intersection"""
    try:
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)
    except:
        return False

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

    # Check overlaps efficiently
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

def compute_repulsion_force(pos1, pos2, radius1=UNIT_HEXAGON_RADIUS, radius2=UNIT_HEXAGON_RADIUS):
    """Compute repulsive force between two hexagon centers"""
    diff = np.array(pos2) - np.array(pos1)
    distance = np.linalg.norm(diff)
    
    if distance < 1e-6:
        return np.array([0.0, 0.0])
    
    # Add some padding for proper separation
    min_dist = radius1 + radius2
    if distance < min_dist:
        # Repel when too close
        force_magnitude = (min_dist - distance) * 10000
        force_direction = diff / distance
        return force_direction * force_magnitude
    else:
        return np.array([0.0, 0.0])

def compute_attraction_force(pos, target, strength=0.1):
    """Compute attraction force toward target"""
    diff = np.array(target) - np.array(pos)
    distance = np.linalg.norm(diff)
    if distance > 1e-6:
        force_direction = diff / distance
        return force_direction * strength * distance
    return np.array([0.0, 0.0])

def apply_boundary_constraints(positions, outer_radius):
    """Apply boundary constraints to push hexagons back into valid region"""
    updated_positions = []
    for pos in positions:
        # Calculate distance from center
        distance_from_center = np.linalg.norm(pos)
        if distance_from_center > outer_radius - 0.5:  # Keep some buffer
            # Push back toward center
            direction = -pos / distance_from_center
            new_pos = pos + direction * (distance_from_center - (outer_radius - 0.5))
            updated_positions.append(new_pos)
        else:
            updated_positions.append(pos)
    return updated_positions

def simulate_hexagon_system(initial_positions, outer_radius, iterations=1000):
    """Run physics simulation to optimize hexagon arrangement"""
    # Convert initial positions to numpy array for easier manipulation
    positions = [list(pos) for pos in initial_positions]
    
    # Simulate physics-like relaxation
    for iteration in range(iterations):
        # Compute forces for each hexagon
        forces = [np.array([0.0, 0.0]) for _ in range(len(positions))]
        
        # Repulsive forces between overlapping hexagons
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                pos_i = np.array(positions[i])
                pos_j = np.array(positions[j])
                
                # Simple distance-based repulsion
                diff = pos_j - pos_i
                distance = np.linalg.norm(diff)
                if distance < 2.0 * UNIT_HEXAGON_RADIUS:
                    # Apply repulsion force
                    if distance > 1e-6:
                        repulse_force = diff / distance * (2.0 * UNIT_HEXAGON_RADIUS - distance) * 1000
                        forces[i] += repulse_force
                        forces[j] -= repulse_force
        
        # Attractive forces toward center (stronger for outer hexagons)
        center = np.array([0.0, 0.0])
        for i in range(len(positions)):
            pos = np.array(positions[i])
            # Attract towards center
            attraction_force = compute_attraction_force(pos, center, strength=0.01)
            forces[i] += attraction_force
            
            # Additional force to maintain spacing
            for j in range(len(positions)):
                if i != j:
                    pos_j = np.array(positions[j])
                    diff = pos_j - pos
                    distance = np.linalg.norm(diff)
                    if distance < 3.0:
                        # Repel if too close to other hexagons
                        if distance > 1e-6:
                            repulse_force = diff / distance * (3.0 - distance) * 100
                            forces[i] += repulse_force

        # Apply forces
        dt = 0.01
        for i in range(len(positions)):
            positions[i] = np.array(positions[i]) + forces[i] * dt
        
        # Apply boundary constraints
        positions = apply_boundary_constraints(positions, outer_radius)
        
        # Occasionally adjust to prevent getting stuck
        if iteration % 100 == 0:
            # Small random perturbations to escape local minima
            for i in range(len(positions)):
                if random.random() < 0.1:
                    positions[i] += np.random.normal(0, 0.01, 2)
    
    return positions

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Initial configuration - start with a known good arrangement
    base_config = [
        [0, 0, 0],           # center
        [-2.5, 0, 0],       # left
        [2.5, 0, 0],        # right
        [-1.25, 2.17, 0],   # top-left
        [1.25, 2.17, 0],    # top-right
        [-1.25, -2.17, 0],  # bottom-left
        [1.25, -2.17, 0],   # bottom-right
        [-3.75, 2.17, 0],   # far top-left
        [3.75, 2.17, 0],    # far top-right
        [-3.75, -2.17, 0],  # far bottom-left
        [3.75, -2.17, 0],   # far bottom-right
    ]
    
    # Initialize with slight random variation
    initial_positions = []
    for i, (x, y, angle) in enumerate(base_config):
        if i == 0:
            # Center hexagon - very stable, don't move much
            initial_positions.append([random.uniform(x-0.1, x+0.1), random.uniform(y-0.1, y+0.1), angle])
        else:
            # Other hexagons - add more variation
            initial_positions.append([random.uniform(x-0.5, x+0.5), random.uniform(y-0.5, y+0.5), angle])
    
    # Run physics simulation to relax the configuration
    # First, estimate reasonable outer radius from initial config
    initial_positions_array = np.array([[p[0], p[1]] for p in initial_positions])
    estimated_outer_radius = max(np.linalg.norm(pos) for pos in initial_positions_array) + 1.0
    
    # Run simulation to optimize positions
    try:
        # Run physics simulation
        optimized_positions = simulate_hexagon_system(initial_positions, estimated_outer_radius, 500)
        
        # Create the final configuration with optimized positions
        final_config = []
        for i, (pos, orig) in enumerate(zip(optimized_positions, initial_positions)):
            final_config.append([pos[0], pos[1], orig[2]])  # Keep original angles
            
        final_positions = np.array(final_config)
        
        # Validate and refine solution if needed
        if validate_solution(final_positions):
            outer_radius = calculate_outer_hexagon_radius(final_positions)
        else:
            # Fall back to initial configuration
            final_positions = np.array(initial_positions)
            outer_radius = calculate_outer_hexagon_radius(final_positions)
            
    except Exception as e:
        # If simulation fails, fall back to initial configuration
        final_positions = np.array(initial_positions)
        outer_radius = calculate_outer_hexagon_radius(final_positions)
    
    # Final optimization via gradient descent on the best configuration found
    try:
        # Convert to flat representation for scipy optimize
        def objective(params):
            # Convert flat params back to positions
            positions = []
            for i in range(NUM_INNER_HEXAGONS):
                x = params[i*2]
                y = params[i*2 + 1]
                positions.append([x, y])
            
            # Create full config
            config = []
            for i, (pos, orig) in enumerate(zip(positions, initial_positions)):
                config.append([pos[0], pos[1], orig[2]])
            
            # Evaluate fitness
            fitness = evaluate_fitness(config)
            return -fitness  # Minimize negative fitness (maximize fitness)
        
        # Flatten initial configuration
        initial_flat = []
        for pos in final_positions:
            initial_flat.extend([pos[0], pos[1]])
        
        # Optimize using scipy minimize
        result = minimize(objective, initial_flat, method='L-BFGS-B', 
                         options={'maxiter': 1000, 'ftol': 1e-10})
        
        # Reconstruct positions from optimized result
        optimized_positions = []
        for i in range(NUM_INNER_HEXAGONS):
            x = result.x[i*2]
            y = result.x[i*2 + 1]
            optimized_positions.append([x, y])
        
        # Create final config with optimized positions and original angles
        final_config = []
        for i, (pos, orig) in enumerate(zip(optimized_positions, initial_positions)):
            final_config.append([pos[0], pos[1], orig[2]])
            
        final_positions = np.array(final_config)
        
        # Verify final solution
        if validate_solution(final_positions):
            outer_radius = calculate_outer_hexagon_radius(final_positions)
        else:
            # If optimization resulted in invalid solution, use previous best
            pass
            
    except Exception as e:
        # If optimization fails, continue with existing solution
        pass
    
    # Return final result
    inner_hex_data = final_positions
    outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered at origin
    outer_hex_side_length = outer_radius
    
    # Final validation
    if not validate_solution(inner_hex_data):
        # Fallback to known good configuration
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
        ])
        outer_hex_side_length = 8.0
        outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END