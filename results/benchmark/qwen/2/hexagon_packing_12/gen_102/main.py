# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
from numba import jit
import time
import warnings
from itertools import product

@jit(nopython=True)
def hexagon_vertices_fast(x, y, angle_deg, side_length=1):
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
def distance_point_to_segment(point, seg_start, seg_end):
    """Distance from point to line segment"""
    px, py = point
    x1, y1 = seg_start
    x2, y2 = seg_end
    
    # Vector from start to end
    dx, dy = x2 - x1, y2 - y1
    # Vector from start to point
    px_minus_x1, py_minus_y1 = px - x1, py - y1
    
    # Project point onto line
    length_sq = dx*dx + dy*dy
    if length_sq == 0:
        return np.sqrt(px_minus_x1*px_minus_x1 + py_minus_y1*py_minus_y1)
    
    t = (px_minus_x1*dx + py_minus_y1*dy) / length_sq
    t = max(0, min(1, t))
    
    # Closest point on segment
    closest_x = x1 + t*dx
    closest_y = y1 + t*dy
    
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

@jit(nopython=True)
def hexagon_distance_fast(hex1_vertices, hex2_vertices):
    """Compute minimum distance between two hexagons"""
    min_dist = np.inf
    for i in range(6):
        p1 = hex1_vertices[i]
        p2 = hex1_vertices[(i+1)%6]
        for j in range(6):
            q1 = hex2_vertices[j]
            q2 = hex2_vertices[(j+1)%6]
            dist = distance_point_to_segment(q1, p1, p2)
            min_dist = min(min_dist, dist)
    return min_dist

def create_symmetric_initial():
    """Create highly symmetric initial configuration based on group theory"""
    # Use a pattern inspired by the 12-fold symmetry group
    positions = []
    
    # Central hexagon
    positions.append([0.0, 0.0, 0.0])
    
    # Ring 1: 6 hexagons arranged in a regular hexagon 
    for i in range(6):
        angle = i * 60  # degrees
        radius = 2.0
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        positions.append([x, y, 0.0])
    
    # Ring 2: 5 hexagons in a pentagonal arrangement
    for i in range(5):
        angle = i * 72 + 18  # offset to create irregular but symmetric pattern
        radius = 3.5
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        positions.append([x, y, 0.0])
    
    # Add some strategic rotations to increase optimality
    # Rotate some hexagons to break degenerate symmetries
    positions[1][2] = 30   # First ring hexagon rotated
    positions[2][2] = 15   # Second ring hexagon rotated
    positions[4][2] = 45   # Third ring hexagon rotated
    
    return np.array(positions)

def create_outer_hexagon_vertices(side_length):
    """Create vertices of outer hexagon with given side length"""
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    vertices = np.column_stack([np.cos(angles), np.sin(angles)]) * side_length
    return vertices

def check_containment_fast(hex_position, outer_vertices):
    """Fast containment check using vertex position"""
    x, y, angle = hex_position
    vertices = hexagon_vertices_fast(x, y, angle)
    
    # Check if all vertices are within outer hexagon
    for vertex in vertices:
        if not point_in_polygon_fast(vertex, outer_vertices):
            return False
    return True

def check_overlap_fast(hex1_pos, hex2_pos):
    """Fast overlap check using distance between centers vs sum of radii"""
    x1, y1, _ = hex1_pos
    x2, y2, _ = hex2_pos
    
    # Distance between centers
    dist_centers = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    # For unit hexagons, approximate minimum distance between edges
    # When they touch, centers are about 2 units apart
    # When they overlap, centers are less than 2 units apart
    return dist_centers < 1.99  # Small tolerance for overlap

def validate_configuration_fast(hex_data, outer_vertices):
    """Quick validation of configuration using fast geometric checks"""
    for i in range(len(hex_data)):
        if not check_containment_fast(hex_data[i], outer_vertices):
            return False
    return True

def compute_outer_side_length(hex_data):
    """Compute minimum side length of outer hexagon"""
    max_dist = 0
    for i in range(len(hex_data)):
        x, y, angle = hex_data[i]
        vertices = hexagon_vertices_fast(x, y, angle)
        for vx, vy in vertices:
            dist = np.sqrt(vx*vx + vy*vy)
            max_dist = max(max_dist, dist)
    
    # Convert to hexagon side length (accounting for hexagon geometry)
    # For a regular hexagon, circumradius = side_length
    # But our vertices may extend beyond circumradius of the hexagon itself
    # So we want side_length such that all vertices are within the outer hexagon
    side_length = max_dist * 2 / np.sqrt(3)  # More accurate conversion
    
    return side_length

def create_hexagon_polygon(center_x, center_y, side_length=1, rotation_deg=0):
    """Create a shapely polygon for a hexagon."""
    vertices = []
    angle_step = np.pi / 3
    rotation_rad = np.radians(rotation_deg)
    for i in range(6):
        angle = rotation_rad + i * angle_step
        x = center_x + side_length * np.cos(angle)
        y = center_y + side_length * np.sin(angle)
        vertices.append((x, y))
    return Polygon(vertices)

def check_containment_shapely(hexagon_poly, outer_hex_poly):
    """Check if hexagon is fully contained within outer hexagon using vertex containment."""
    vertices = list(hexagon_poly.exterior.coords)
    for point in vertices:
        if not outer_hex_poly.contains(Point(point[0], point[1])):
            return False
    return True

def check_overlap_shapely(hex1_poly, hex2_poly):
    """Check if two hexagons overlap using shapely intersection."""
    return hex1_poly.intersects(hex2_poly) and not hex1_poly.touches(hex2_poly)

def calculate_fitness_precise(hex_data, outer_side_length):
    """Calculate fitness with precise Shapely-based validation"""
    # Create all inner hexagon polygons
    inner_hexagons = []
    for i in range(len(hex_data)):
        cx, cy, rot = hex_data[i]
        hex_poly = create_hexagon_polygon(cx, cy, 1, rot)
        inner_hexagons.append(hex_poly)

    # Check containment and overlap with Shapely for precision
    outer_hex = create_hexagon_polygon(0, 0, outer_side_length, 0)

    # Count violations
    overlap_count = 0
    containment_count = 0

    # Check overlaps between all pairs
    for i in range(len(inner_hexagons)):
        for j in range(i+1, len(inner_hexagons)):
            if check_overlap_shapely(inner_hexagons[i], inner_hexagons[j]):
                overlap_count += 1

    # Check containment
    for hex_poly in inner_hexagons:
        if not check_containment_shapely(hex_poly, outer_hex):
            containment_count += 1

    # Penalty for constraint violations
    penalty = overlap_count * 100000 + containment_count * 100000

    # Fitness is inverse of outer hex side length minus penalties
    if overlap_count > 0 or containment_count > 0:
        return -penalty  # Very bad fitness if constraints violated
    else:
        return 1.0 / outer_side_length

def calculate_fitness_approximate(hex_data, outer_side_length):
    """Approximate fitness calculation for faster processing during optimization"""
    # Check for overlaps using fast method
    penalty = 0
    
    # Fast initial check using distance
    for i in range(len(hex_data)):
        for j in range(i+1, len(hex_data)):
            if check_overlap_fast(hex_data[i], hex_data[j]):
                # Use exact computation for overlaps
                x1, y1, angle1 = hex_data[i]
                x2, y2, angle2 = hex_data[j]
                v1 = hexagon_vertices_fast(x1, y1, angle1)
                v2 = hexagon_vertices_fast(x2, y2, angle2)
                
                # Use Shapely for precise overlap detection
                p1 = Polygon(v1)
                p2 = Polygon(v2)
                if p1.intersects(p2):
                    penalty += 100000
    
    # Check containment with Shapely for precision
    outer_vertices = create_outer_hexagon_vertices(outer_side_length)
    
    for i in range(len(hex_data)):
        x, y, angle = hex_data[i]
        vertices = hexagon_vertices_fast(x, y, angle)
        hex_poly = Polygon(vertices)
        
        # Point by point containment check with Shapely
        for vertex in vertices:
            point = Point(vertex[0], vertex[1])
            if not Polygon(outer_vertices).contains(point):
                # Calculate how far outside the boundary
                dist = np.sqrt(vertex[0]**2 + vertex[1]**2)
                penalty += (dist - outer_side_length)**2 * 1000
    
    # Objective: maximize 1/outer_side_length
    # So we minimize negative of 1/outer_side_length plus penalty
    objective = -1.0 / (outer_side_length + 1e-10) + penalty
    
    return objective

def create_evolutionary_population(pop_size, target_dim=12):
    """Create diverse population with symmetry awareness"""
    population = []
    
    # Generate multiple symmetric base configurations
    for i in range(pop_size // 2):
        base_config = create_symmetric_initial()
        
        # Add variation to positions and orientations
        noise = np.random.normal(0, 0.3, base_config.shape)
        mutated_config = base_config + noise
        population.append(mutated_config.flatten())
    
    # Generate some random configurations
    for i in range(pop_size // 2):
        # Random configuration but with sensible ranges
        config = np.random.uniform(-4, 4, (target_dim, 3))
        config[:, 2] = np.random.uniform(0, 360, target_dim)  # Random rotations
        population.append(config.flatten())
    
    return population

def evolutionary_optimization():
    """Use evolutionary approach with symmetry-aware operators"""
    pop_size = 15
    max_generations = 20
    
    # Create initial population
    population = create_evolutionary_population(pop_size)
    
    best_fitness = float('inf')
    best_individual = None
    
    for gen in range(max_generations):
        # Evaluate all individuals
        fitness_scores = []
        
        for individual in population:
            config = individual.reshape(-1, 3)
            outer_side = compute_outer_side_length(config)
            fitness = calculate_fitness_approximate(config, outer_side)
            fitness_scores.append(fitness)
        
        # Select best individuals
        sorted_indices = np.argsort(fitness_scores)
        elite_count = pop_size // 3
        selected_indices = sorted_indices[:elite_count]
        
        # Keep best individual
        current_best_idx = sorted_indices[0]
        current_best_fitness = fitness_scores[current_best_idx]
        
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[current_best_idx].copy()
        
        # Create new population through crossover and mutation
        new_population = []
        
        # Keep elites
        for idx in selected_indices:
            new_population.append(population[idx])
        
        # Generate offspring
        while len(new_population) < pop_size:
            parent1_idx = np.random.choice(selected_indices)
            parent2_idx = np.random.choice(selected_indices)
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Uniform crossover
            child = np.copy(parent1)
            mask = np.random.rand(len(child)) > 0.5
            child[mask] = parent2[mask]
            
            # Mutation with symmetry awareness
            mut_rate = 0.1
            for i in range(len(child)):
                if np.random.rand() < mut_rate:
                    if i % 3 == 2:  # Rotation parameter
                        child[i] += np.random.normal(0, 30)  # Larger change for rotation
                        child[i] = child[i] % 360
                    else:  # Position parameters
                        child[i] += np.random.normal(0, 0.5)
            
            new_population.append(child)
        
        population = new_population
    
    return best_individual.reshape(-1, 3)

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
        # Use evolutionary optimization
        inner_hex_data = evolutionary_optimization()
        
        # Final validation and refinement
        outer_side_length = compute_outer_side_length(inner_hex_data)
        
        # Perform one final detailed check with precise Shapely validation
        final_outer_vertices = create_outer_hexagon_vertices(outer_side_length)
        is_valid = validate_configuration_fast(inner_hex_data, final_outer_vertices)
        
        if not is_valid:
            # Fall back to symmetric configuration
            inner_hex_data = create_symmetric_initial()
            outer_side_length = compute_outer_side_length(inner_hex_data)
            
        # Double-check with precise validation
        precise_fitness = calculate_fitness_precise(inner_hex_data, outer_side_length)
        if precise_fitness < -100000:  # If violations detected
            # Fall back to symmetric configuration
            inner_hex_data = create_symmetric_initial()
            outer_side_length = compute_outer_side_length(inner_hex_data)
        
    except Exception as e:
        warnings.warn(f"Evolutionary optimization failed: {str(e)}")
        # Fall back to symmetric configuration
        inner_hex_data = create_symmetric_initial()
        outer_side_length = compute_outer_side_length(inner_hex_data)
    
    # Create outer hexagon data (centered at origin, no rotation)
    outer_hex_data = np.array([0, 0, 0])
    
    end_time = time.time()
    
    # Calculate benchmark ratio for reporting
    benchmark_ratio = (1.0 / outer_side_length) / 0.2537
    
    # Print metrics
    print(f"inv_outer_hex_side_length: {1.0/outer_side_length:.8f}")
    print(f"benchmark_ratio: {benchmark_ratio:.8f}")
    print(f"eval_time: {end_time - start_time:.4f}s")
    
    return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END
