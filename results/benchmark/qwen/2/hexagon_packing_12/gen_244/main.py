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

def create_advanced_symmetric_initial():
    """Create advanced symmetric initial configuration using group theory principles"""
    # Create a configuration based on known optimal symmetric arrangements
    # This follows the principle of D6 symmetry (hexagonal symmetry) enhanced with
    # additional strategic placements

    positions = []

    # Central hexagon
    positions.append([0.0, 0.0, 0.0])

    # First ring: 6 hexagons arranged in a perfect hexagon
    # Using proper spacing for tight packing
    ring1_radius = 1.732  # sqrt(3) ≈ 1.732 - optimal spacing for unit hexagons
    for i in range(6):
        angle = i * 60  # degrees - perfect hexagonal arrangement
        x = ring1_radius * np.cos(np.radians(angle))
        y = ring1_radius * np.sin(np.radians(angle))
        positions.append([x, y, 0.0])

    # Second ring: 5 hexagons arranged in approximate pentagon with symmetry breaking
    ring2_radius = 3.0  # Slightly larger for better packing
    for i in range(5):
        # Offset by 18 degrees to break perfect symmetry and allow optimization
        angle = i * 72 + 18
        x = ring2_radius * np.cos(np.radians(angle))
        y = ring2_radius * np.sin(np.radians(angle))
        positions.append([x, y, 0.0])

    # Add strategic rotations to break degenerate symmetries and improve optimization
    # These rotations are chosen to maintain overall symmetry while allowing local optimization
    positions[1][2] = 30   # First ring - rotate to break degeneracy
    positions[2][2] = 15   # Second ring - rotate to allow better packing
    positions[4][2] = 45   # Third ring - rotate strategically
    positions[7][2] = 90   # Outer ring - add rotational diversity
    positions[8][2] = 270  # Another outer ring - mirror rotation for symmetry

    # Create reflection variant for diversity
    reflected_positions = []
    for pos in positions:
        x, y, angle = pos
        # Reflect across x-axis for diversity
        reflected_positions.append([x, -y, 360 - angle])

    return np.array(positions), np.array(reflected_positions)

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

def calculate_fitness(hex_data, outer_side_length):
    """Calculate fitness for evolutionary optimization with optimized calculations"""
    penalty = 0

    # Precompute outer hexagon vertices once
    outer_vertices = create_outer_hexagon_vertices(outer_side_length)
    outer_polygon = Polygon(outer_vertices)

    # Check containment efficiently - batch operation for better performance
    containment_violations = 0

    # Fast initial containment check using distance from center
    max_radius = outer_side_length * np.sqrt(3) / 2  # Apothem of outer hexagon

    for i in range(len(hex_data)):
        x, y, angle = hex_data[i]
        # Quick distance check first
        dist_from_center = np.sqrt(x*x + y*y)
        if dist_from_center > max_radius:
            penalty += 1500000  # High penalty for containment violations
            containment_violations += 1
            continue

        # For valid positions, check detailed containment
        vertices = hexagon_vertices_fast(x, y, angle)
        hex_poly = Polygon(vertices)

        # Check if any vertex is outside outer polygon
        for vertex in vertices:
            point = Point(vertex[0], vertex[1])
            if not outer_polygon.contains(point):
                # Calculate how far outside the boundary
                dist = np.sqrt(vertex[0]**2 + vertex[1]**2)
                # Adaptive penalty: more severe for deeper violations
                violation_distance = max(0, dist - outer_side_length)
                penalty += violation_distance * 1500  # Higher penalty for containment
                containment_violations += 1
                break  # No need to check other vertices once one fails

    # Efficient overlap checking using spatial locality
    # Create bounding box for each hexagon to reduce overlap checks
    overlap_penalties = 0
    n_hexagons = len(hex_data)

    # Check pairs more efficiently
    for i in range(n_hexagons):
        x1, y1, angle1 = hex_data[i]
        v1 = hexagon_vertices_fast(x1, y1, angle1)

        for j in range(i+1, n_hexagons):
            x2, y2, angle2 = hex_data[j]

            # Fast distance check - only proceed to detailed check if close enough
            dist_centers = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            # For unit hexagons, if centers are more than 2 units apart, no overlap possible
            if dist_centers >= 2.0:
                continue

            v2 = hexagon_vertices_fast(x2, y2, angle2)

            # Use Shapely for precise overlap detection
            p1 = Polygon(v1)
            p2 = Polygon(v2)
            if p1.intersects(p2):
                overlap_penalties += 1000000
                # Early termination if we already detected overlaps
                if overlap_penalties > 1000000:
                    break

    penalty += overlap_penalties

    # Objective: maximize 1/outer_side_length
    # So we minimize negative of 1/outer_side_length plus penalty
    # Add adaptive scaling based on constraint violations
    if containment_violations > 0:
        penalty *= (1.0 + containment_violations * 0.1)  # Scale penalty based on violations

    objective = -1.0 / (outer_side_length + 1e-10) + penalty

    return objective

def create_evolutionary_population(pop_size, target_dim=12):
    """Create diverse population with symmetry awareness and reflection diversity"""
    population = []

    # Generate multiple symmetric base configurations (original and reflected)
    for i in range(pop_size // 3):
        # Original symmetric configuration
        base_config_orig, base_config_reflect = create_advanced_symmetric_initial()

        # Add variation to positions and orientations for original
        noise = np.random.normal(0, 0.2, base_config_orig.shape)
        mutated_config_orig = base_config_orig + noise
        population.append(mutated_config_orig.flatten())

        # Add variation to positions and orientations for reflected
        noise_reflect = np.random.normal(0, 0.2, base_config_reflect.shape)
        mutated_config_reflect = base_config_reflect + noise_reflect
        population.append(mutated_config_reflect.flatten())

    # Generate some random configurations for exploration
    for i in range(pop_size // 3, pop_size):
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
            fitness = calculate_fitness(config, outer_side)
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

        # Perform one final detailed check
        final_outer_vertices = create_outer_hexagon_vertices(outer_side_length)
        is_valid = validate_configuration_fast(inner_hex_data, final_outer_vertices)

        if not is_valid:
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