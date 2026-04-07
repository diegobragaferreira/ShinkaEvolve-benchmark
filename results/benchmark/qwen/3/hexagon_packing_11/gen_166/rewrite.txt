# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
import math
from scipy.optimize import differential_evolution, minimize
import time
from itertools import combinations
from collections import defaultdict

def create_regular_hexagon(center_x, center_y, side_length=1, rotation_deg=0):
    """Create a regular hexagon as a Shapely polygon"""
    rotation_rad = math.radians(rotation_deg)
    points = []
    for i in range(6):
        angle = rotation_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        points.append((x, y))
    return Polygon(points)

def compute_outer_hexagon_radius(inner_hexagons, padding=0.01):
    """Compute minimum radius needed to contain all inner hexagons with some padding"""
    # Get all vertices of all hexagons
    all_vertices = []
    for hex_poly in inner_hexagons:
        all_vertices.extend(list(hex_poly.exterior.coords))

    # Find center of bounding box
    xs = [p[0] for p in all_vertices]
    ys = [p[1] for p in all_vertices]
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2

    # Compute max distance from center to any vertex
    max_dist = 0
    for x, y in all_vertices:
        dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = max(max_dist, dist)

    # Add padding and convert to side length
    return max_dist + padding

def check_containment_and_overlap_fast(inner_hexagons, outer_hexagon):
    """Fast containment and overlap check with early termination"""
    # Check containment with early exit
    for hex_poly in inner_hexagons:
        if not outer_hexagon.contains(hex_poly):
            return False

    # Simple AABB-based overlap detection for quick rejection
    def get_aabb(hex_poly):
        coords = list(hex_poly.exterior.coords)
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        return min(xs), max(xs), min(ys), max(ys)
    
    # Quick AABB checking first
    aabbs = [get_aabb(hex_poly) for hex_poly in inner_hexagons]
    
    # Check pairwise AABB collisions (fast)
    for i in range(len(aabbs)):
        for j in range(i+1, len(aabbs)):
            ax1, ax2, ay1, ay2 = aabbs[i]
            bx1, bx2, by1, by2 = aabbs[j]
            if not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2):
                # AABB overlap detected, do detailed check
                if inner_hexagons[i].intersects(inner_hexagons[j]):
                    return False
    
    return True

def fast_hex_grid_placement():
    """Generate initial hexagon positions on a hexagonal grid"""
    # Create a structured hexagonal grid pattern that naturally minimizes overlaps
    hex_spacing = math.sqrt(3)  # Distance between centers of adjacent hexagons
    
    # Center hexagon
    positions = [[0, 0, 0]]
    
    # Place hexagons in layers around center
    layers = [
        # Layer 1 (6 hexagons)
        [(-hex_spacing, 0, 0), (hex_spacing, 0, 0), (0, hex_spacing, 0), (0, -hex_spacing, 0),
         (-hex_spacing/2, hex_spacing/2, 0), (hex_spacing/2, hex_spacing/2, 0)],
        # Layer 2 (6 hexagons)  
        [(-hex_spacing*1.5, 0, 0), (hex_spacing*1.5, 0, 0), (0, hex_spacing*1.5, 0), 
         (0, -hex_spacing*1.5, 0), (-hex_spacing*1.5/2, hex_spacing*1.5/2, 0), 
         (hex_spacing*1.5/2, hex_spacing*1.5/2, 0)]
    ]
    
    # Select first 11 positions from structured grid (ensuring compactness)
    for layer in layers:
        positions.extend(layer)
        if len(positions) >= 11:
            break
    
    # Fill remaining positions with slight perturbations to ensure 11 total
    while len(positions) < 11:
        positions.append([0, 0, 0])
    
    return np.array(positions[:11])

def polar_representations(hex_positions):
    """Convert Cartesian positions to polar representations for more efficient optimization"""
    # Use center of first 5 hexagons as reference point
    ref_x, ref_y = np.mean([pos[:2] for pos in hex_positions[:5]], axis=0)
    
    polar_coords = []
    for pos in hex_positions:
        x, y = pos[0], pos[1]
        # Calculate polar coordinates relative to reference point
        r = math.sqrt((x - ref_x)**2 + (y - ref_y)**2)
        theta = math.atan2(y - ref_y, x - ref_x)
        polar_coords.append([r, theta, pos[2]])  # r, theta, angle
    
    return np.array(polar_coords), (ref_x, ref_y)

def cartesian_from_polar(polar_coords, ref_point):
    """Convert polar representations back to Cartesian coordinates"""
    ref_x, ref_y = ref_point
    cartesian = []
    for r, theta, angle in polar_coords:
        x = ref_x + r * math.cos(theta)
        y = ref_y + r * math.sin(theta)
        cartesian.append([x, y, angle])
    return np.array(cartesian)

def evaluate_fitness_fast(polar_coords, ref_point, max_radius=20.0):
    """Fast fitness evaluation based on polar representation"""
    try:
        # Convert back to Cartesian
        positions = cartesian_from_polar(polar_coords, ref_point)
        
        # Create hexagon polygons
        hexagons = [create_regular_hexagon(pos[0], pos[1], 1, pos[2]) for pos in positions]
        
        # Compute outer radius
        outer_radius = compute_outer_hexagon_radius(hexagons, 0.01)
        
        # Check constraints (fast version)
        outer_hex = create_regular_hexagon(0, 0, outer_radius, 0)
        valid = check_containment_and_overlap_fast(hexagons, outer_hex)
        
        # Return fitness (negative inverse radius if valid, otherwise very bad)
        if valid:
            return -(1.0 / outer_radius)
        else:
            return -1e10
            
    except Exception:
        return -1e10

def adaptive_mutation(polar_coords, mutation_rate=0.1):
    """Adaptive mutation that adjusts based on solution quality"""
    mutated = polar_coords.copy()
    
    # Randomly mutate components with different rates
    for i in range(len(mutated)):
        # Mutate position (r, theta)
        if np.random.random() < mutation_rate:
            mutated[i][0] *= np.random.normal(1, 0.05)  # r mutation
            mutated[i][0] = max(0.1, mutated[i][0])  # Ensure positive
        if np.random.random() < mutation_rate:
            mutated[i][1] += np.random.normal(0, 0.1)  # theta mutation
        
        # Mutate angle
        if np.random.random() < mutation_rate:
            mutated[i][2] += np.random.normal(0, 5)  # Angle mutation
            mutated[i][2] = mutated[i][2] % 360  # Keep angle in [0,360)
            
    return mutated

def tournament_selection(population, fitnesses, k=3):
    """Tournament selection with better diversity"""
    selected = []
    for _ in range(len(population)):
        tournament_indices = np.random.choice(len(population), k)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        selected.append(population[winner_idx])
    return selected

def crossover(parent1, parent2, crossover_rate=0.8):
    """Uniform crossover with geometric awareness"""
    if np.random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()
    
    # Simple uniform crossover on polar coordinates
    child1, child2 = parent1.copy(), parent2.copy()
    for i in range(len(child1)):
        if np.random.random() < 0.5:
            # Swap components
            child1[i], child2[i] = child2[i], child1[i]
    return child1, child2

def hexgrid_evolution():
    """Main evolutionary algorithm using hexagonal grid principles"""
    np.random.seed(42)
    population_size = 15
    generations = 30
    elite_size = 3
    
    # Initial population generation using structured hexagonal grid
    initial_configs = []
    for _ in range(population_size):
        # Start with good hexagonal grid placement
        base_positions = fast_hex_grid_placement()
        # Add small random perturbations to create diverse initial population
        perturbed = base_positions.copy().astype(float)
        for i in range(len(perturbed)):
            perturbed[i][0] += np.random.normal(0, 0.1)
            perturbed[i][1] += np.random.normal(0, 0.1)
            perturbed[i][2] += np.random.normal(0, 5)
        initial_configs.append(perturbed)
    
    # Convert to polar representation for efficient evolution
    populations = []
    for config in initial_configs:
        polar_rep, ref_point = polar_representations(config)
        populations.append((polar_rep, ref_point))
    
    best_fitness = -float('inf')
    best_solution = None
    
    for gen in range(generations):
        # Evaluate population
        fitnesses = []
        for polar_coords, ref_point in populations:
            fitness = evaluate_fitness_fast(polar_coords, ref_point)
            fitnesses.append(fitness)
        
        # Track best solution
        max_idx = np.argmax(fitnesses)
        if fitnesses[max_idx] > best_fitness:
            best_fitness = fitnesses[max_idx]
            best_solution = populations[max_idx]
        
        # Selection
        selected_pop = tournament_selection(populations, fitnesses)
        
        # Elitism
        elite_indices = np.argsort(fitnesses)[-elite_size:]
        elites = [populations[i] for i in elite_indices]
        
        # Crossover and mutation
        new_pop = elites[:]
        while len(new_pop) < population_size:
            parent1, parent2 = np.random.choice(selected_pop, 2, replace=False)
            child1, child2 = crossover(parent1[0], parent2[0])
            
            # Apply mutations
            child1 = adaptive_mutation(child1, 0.15 if gen < 15 else 0.05)
            child2 = adaptive_mutation(child2, 0.15 if gen < 15 else 0.05)
            
            # Add to population
            new_pop.append((child1, parent1[1]))  # Keep reference point
            if len(new_pop) < population_size:
                new_pop.append((child2, parent2[1]))
        
        populations = new_pop[:population_size]
    
    # Return best solution
    best_polar, best_ref = best_solution
    final_positions = cartesian_from_polar(best_polar, best_ref)
    return final_positions

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Run hexgrid evolution
    start_time = time.time()
    
    try:
        # Use the specialized evolutionary approach
        inner_positions = hexgrid_evolution()
        
        # Final validation and refinement
        inner_hexagons = []
        for pos in inner_positions:
            x, y, angle = pos
            hex_poly = create_regular_hexagon(x, y, 1, angle)
            inner_hexagons.append(hex_poly)
        
        # Compute outer hexagon size
        outer_radius = compute_outer_hexagon_radius(inner_hexagons, 0.01)
        outer_hexagon = create_regular_hexagon(0, 0, outer_radius, 0)
        
        # Validate constraints
        if not check_containment_and_overlap_fast(inner_hexagons, outer_hexagon):
            # Fallback to simple arrangement if validation fails
            inner_positions = fast_hex_grid_placement()
            inner_hexagons = []
            for pos in inner_positions:
                x, y, angle = pos
                hex_poly = create_regular_hexagon(x, y, 1, angle)
                inner_hexagons.append(hex_poly)
            outer_radius = compute_outer_hexagon_radius(inner_hexagons, 0.01)
            
    except Exception as e:
        # Fallback to basic configuration
        print(f"Evolution failed: {e}")
        inner_positions = fast_hex_grid_placement()
        inner_hexagons = []
        for pos in inner_positions:
            x, y, angle = pos
            hex_poly = create_regular_hexagon(x, y, 1, angle)
            inner_hexagons.append(hex_poly)
        outer_radius = compute_outer_hexagon_radius(inner_hexagons, 0.01)

    # Format output
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    
    # Return results
    return inner_positions, outer_hex_data, outer_radius

# EVOLVE-BLOCK-END