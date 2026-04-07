# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import cKDTree
import time
import random
from copy import deepcopy
import math

def hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, angle, and side length"""
    angle_rad = np.radians(angle_deg)
    angles = np.linspace(0, 2*np.pi, 7) + angle_rad
    vertices = np.array([
        [center_x + side_length * np.cos(a), center_y + side_length * np.sin(a)]
        for a in angles
    ])
    return vertices

def check_containment(hex_vertices, outer_hex_vertices):
    """Check if hexagon vertices are contained within outer hexagon using Shapely"""
    inner_polygon = Polygon(hex_vertices)
    outer_polygon = Polygon(outer_hex_vertices)
    return outer_polygon.contains(inner_polygon)

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely"""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def compute_outer_hexagon_radius(inner_positions, inner_angles, initial_radius_estimate=5.0):
    """Compute minimum outer hexagon radius that contains all inner hexagons with enhanced precision"""
    # Binary search for tightest fit with faster convergence
    left = initial_radius_estimate
    right = 20.0
    best_radius = right

    precision_threshold = 1e-6
    max_iterations = 100
    iterations = 0

    while right - left > precision_threshold and iterations < max_iterations:
        mid = (left + right) / 2.0
        outer_vertices = hexagon_vertices(0, 0, 0, mid)
        valid = True

        # Check all inner hexagons
        for i, (pos, angle) in enumerate(zip(inner_positions, inner_angles)):
            hex_vertices = hexagon_vertices(pos[0], pos[1], angle)
            if not check_containment(hex_vertices, outer_vertices):
                valid = False
                break

        if valid:
            best_radius = mid
            right = mid
        else:
            left = mid
        iterations += 1

    return best_radius

def evaluate_fitness(inner_positions, inner_angles):
    """Evaluate fitness: higher is better, maximize 1/radius"""
    # Create outer hexagon vertices
    outer_radius = compute_outer_hexagon_radius(inner_positions, inner_angles)

    # Check all constraints
    total_penalty = 0

    # Check containment for all inner hexagons
    outer_vertices = hexagon_vertices(0, 0, 0, outer_radius)
    for i, (pos, angle) in enumerate(zip(inner_positions, inner_angles)):
        hex_vertices = hexagon_vertices(pos[0], pos[1], angle)
        if not check_containment(hex_vertices, outer_vertices):
            total_penalty += 10000  # Large penalty for containment violation

    # Check overlaps between all pairs of inner hexagons
    for i in range(len(inner_positions)):
        for j in range(i+1, len(inner_positions)):
            hex1_vertices = hexagon_vertices(inner_positions[i][0], inner_positions[i][1], inner_angles[i])
            hex2_vertices = hexagon_vertices(inner_positions[j][0], inner_positions[j][1], inner_angles[j])
            if check_overlap(hex1_vertices, hex2_vertices):
                total_penalty += 10000  # Large penalty for overlap violation

    # Fitness is negative of the radius plus penalties
    # We want to minimize radius, so fitness = -radius
    fitness = -outer_radius - total_penalty

    return fitness, outer_radius

def create_hexagonal_lattice_config():
    """Create an initial configuration based on a mathematical hexagonal lattice"""
    # This follows the pattern of placing hexagons in a way that naturally minimizes empty space
    # We'll place them in layers according to hexagonal number sequence
    
    # Layer 1: center
    positions = [[0, 0]]
    
    # Layer 2: first ring (6 hexagons)
    for i in range(6):
        angle = i * np.pi / 3
        positions.append([2 * np.cos(angle), 2 * np.sin(angle)])
    
    # Layer 3: second ring (12 hexagons)
    for i in range(12):
        angle = i * np.pi / 6
        positions.append([3 * np.cos(angle), 3 * np.sin(angle)])
    
    # Trim to exactly 11 hexagons
    positions = positions[:11]
    
    # Convert to numpy array and add rotation values (0 for now)
    result = np.zeros((11, 3))
    for i in range(11):
        result[i][0] = positions[i][0]
        result[i][1] = positions[i][1]
        result[i][2] = 0
    
    # Add slight perturbations around these lattice positions
    for i in range(11):
        result[i][0] += random.uniform(-0.2, 0.2)
        result[i][1] += random.uniform(-0.2, 0.2)
        result[i][2] += random.uniform(-10, 10)
        result[i][2] = result[i][2] % 360
    
    return result

def generate_spatial_tree_config():
    """Generate initial configuration using spatial tree-based approach"""
    # Start with a hexagonal lattice pattern
    base_config = create_hexagonal_lattice_config()
    
    # Apply small random perturbations to break symmetry and allow optimization
    for i in range(11):
        base_config[i][0] += random.gauss(0, 0.1)
        base_config[i][1] += random.gauss(0, 0.1)
        base_config[i][2] += random.gauss(0, 5)
        base_config[i][2] = base_config[i][2] % 360
    
    return base_config

def get_hexagon_centroids_and_bounds(hex_data):
    """Get centroids and bounds for efficient spatial queries"""
    centroids = [(hex_data[i][0], hex_data[i][1]) for i in range(len(hex_data))]
    bounds = []
    for i in range(len(hex_data)):
        cx, cy, angle = hex_data[i]
        vertices = hexagon_vertices(cx, cy, angle)
        xs = vertices[:, 0]
        ys = vertices[:, 1]
        bounds.append((min(xs), min(ys), max(xs), max(ys)))
    return centroids, bounds

def get_overlapping_pairs(hex_data, centroids, bounds, tree=None):
    """Find overlapping pairs using spatial tree for efficiency"""
    overlapping_pairs = set()
    
    if tree is None:
        # Build spatial tree manually for efficiency
        points = np.array(centroids)
        tree = cKDTree(points)
    
    # For each hexagon, check nearby hexagons
    for i in range(len(hex_data)):
        cx, cy, angle = hex_data[i]
        # Use radius approximation for query
        radius = 2.0  # approximate distance between centers that could overlap
        
        # Query neighbors
        indices = tree.query_ball_point((cx, cy), radius)
        
        for j in indices:
            if i >= j:
                continue
                
            # Check actual overlap
            hex1_vertices = hexagon_vertices(hex_data[i][0], hex_data[i][1], hex_data[i][2])
            hex2_vertices = hexagon_vertices(hex_data[j][0], hex_data[j][1], hex_data[j][2])
            
            if check_overlap(hex1_vertices, hex2_vertices):
                overlapping_pairs.add((i, j))
    
    return overlapping_pairs

def geometric_relaxation_step(hex_data, max_iterations=50):
    """Apply geometric relaxation to resolve overlaps"""
    # Build initial spatial tree
    centroids, bounds = get_hexagon_centroids_and_bounds(hex_data)
    points = np.array(centroids)
    tree = cKDTree(points)
    
    # Try to resolve overlaps iteratively
    for iteration in range(max_iterations):
        overlapping_pairs = get_overlapping_pairs(hex_data, centroids, bounds, tree)
        
        if not overlapping_pairs:
            break
            
        # Resolve conflicts by moving hexagons apart
        for i, j in overlapping_pairs:
            # Get vectors between centers
            dx = hex_data[j][0] - hex_data[i][0]
            dy = hex_data[j][1] - hex_data[i][1]
            
            # Normalize
            dist = max(math.sqrt(dx*dx + dy*dy), 1e-6)
            dx /= dist
            dy /= dist
            
            # Move them apart (small distance)
            move_amount = 0.05
            
            hex_data[i][0] -= dx * move_amount * 0.5
            hex_data[i][1] -= dy * move_amount * 0.5
            hex_data[j][0] += dx * move_amount * 0.5
            hex_data[j][1] += dy * move_amount * 0.5
            
            # Update spatial tree for next iteration
            centroids, bounds = get_hexagon_centroids_and_bounds(hex_data)
            points = np.array(centroids)
            tree = cKDTree(points)
    
    return hex_data

def apply_boundary_constraints(hex_data, outer_radius):
    """Ensure all hexagons stay within outer boundary"""
    outer_vertices = hexagon_vertices(0, 0, 0, outer_radius)
    outer_polygon = Polygon(outer_vertices)
    
    for i in range(len(hex_data)):
        cx, cy, angle = hex_data[i]
        vertices = hexagon_vertices(cx, cy, angle)
        
        # Create hexagon polygon
        inner_polygon = Polygon(vertices)
        
        # If hexagon is outside, push it back in
        if not outer_polygon.contains(inner_polygon):
            # Find centroid of hexagon
            centroid_x = np.mean(vertices[:, 0])
            centroid_y = np.mean(vertices[:, 1])
            
            # Calculate vector from center to centroid
            dx = centroid_x - 0
            dy = centroid_y - 0
            
            # Normalize and scale
            dist = max(math.sqrt(dx*dx + dy*dy), 1e-6)
            dx /= dist
            dy /= dist
            
            # Push inward by a small amount
            push_distance = 0.1
            hex_data[i][0] = 0 - dx * (dist - push_distance)
            hex_data[i][1] = 0 - dy * (dist - push_distance)
            
    return hex_data

def optimize_hexagon_positions(initial_config):
    """Main optimization routine using geometric approaches"""
    current_config = deepcopy(initial_config)
    best_config = deepcopy(current_config)
    best_fitness, best_radius = evaluate_fitness(current_config[:, :2], current_config[:, 2])
    
    # Stage 1: Geometric relaxation to remove overlaps
    current_config = geometric_relaxation_step(current_config)
    
    # Stage 2: Boundary adjustment
    temp_fitness, temp_radius = evaluate_fitness(current_config[:, :2], current_config[:, 2])
    current_config = apply_boundary_constraints(current_config, temp_radius)
    
    # Stage 3: Local optimization with multiple strategies
    for iteration in range(100):
        # Strategy 1: Random perturbations with geometric checks
        mutated = deepcopy(current_config)
        for i in range(11):
            if random.random() < 0.3:
                # Perturb position
                mutated[i][0] += random.uniform(-0.1, 0.1)
                mutated[i][1] += random.uniform(-0.1, 0.1)
                # Perturb rotation
                mutated[i][2] += random.uniform(-5, 5)
                mutated[i][2] = mutated[i][2] % 360
        
        # Apply relaxation and boundaries
        mutated = geometric_relaxation_step(mutated)
        temp_fitness, temp_radius = evaluate_fitness(mutated[:, :2], mutated[:, 2])
        mutated = apply_boundary_constraints(mutated, temp_radius)
        
        # Evaluate new configuration
        mutated_fitness, mutated_radius = evaluate_fitness(mutated[:, :2], mutated[:, 2])
        
        if mutated_fitness > best_fitness:
            best_fitness = mutated_fitness
            best_config = deepcopy(mutated)
            current_config = deepcopy(mutated)
        elif mutated_fitness > temp_fitness:
            current_config = deepcopy(mutated)
    
    return best_config, best_fitness

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Generate initial configuration using spatial tree-based approach
    initial_config = generate_spatial_tree_config()
    
    # Optimize the configuration
    optimized_config, final_fitness = optimize_hexagon_positions(initial_config)
    
    # Final evaluation
    _, outer_radius = evaluate_fitness(optimized_config[:, :2], optimized_config[:, 2])
    
    # Format output
    inner_hex_data = optimized_config
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    
    elapsed = time.time() - start_time
    print(f"Eval time: {elapsed:.4f}s")
    
    return inner_hex_data, outer_hex_data, outer_radius

# EVOLVE-BLOCK-END