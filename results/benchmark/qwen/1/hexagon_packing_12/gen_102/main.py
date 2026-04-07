# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
from scipy.spatial import cKDTree
import time
from functools import partial

def generate_hexagon_vertices(center_x, center_y, angle_degrees, side_length=1):
    """Generate vertices of a regular hexagon."""
    angle_rad = np.radians(angle_degrees)
    angles = np.linspace(0, 2*np.pi, 7) + angle_rad  # 6 sides + closing vertex
    vertices = []
    for angle in angles:
        x = center_x + side_length * np.cos(angle)
        y = center_y + side_length * np.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

def check_containment(hexagon_vertices, outer_hexagon_vertices):
    """Check if all vertices of inner hexagon are within outer hexagon."""
    inner_polygon = Polygon(hexagon_vertices)
    outer_polygon = Polygon(outer_hexagon_vertices)
    return outer_polygon.contains(inner_polygon)

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap."""
    polygon1 = Polygon(hex1_vertices)
    polygon2 = Polygon(hex2_vertices)
    return polygon1.intersects(polygon2)

def fast_check_overlap_pair(hex1_vertices, hex2_vertices):
    """Fast overlap check using bounding circles for early rejection."""
    # Compute centroids
    cx1, cy1 = np.mean(hex1_vertices, axis=0)
    cx2, cy2 = np.mean(hex2_vertices, axis=0)
    
    # Compute approximate radii (distance from centroid to farthest vertex)
    r1 = max(np.linalg.norm(v - [cx1, cy1]) for v in hex1_vertices)
    r2 = max(np.linalg.norm(v - [cx2, cy2]) for v in hex2_vertices)
    
    # Fast circle overlap test
    dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    return dist < (r1 + r2)

def evaluate_configuration_fast(params, use_kdtree=True):
    """
    Fast evaluation function optimized for speed with spatial indexing.
    """
    # Extract parameters
    # First 36 params: 12 hexagons * (x, y, angle)
    hex_params = params[:36].reshape(12, 3)

    # Last 3 params: outer hexagon center (x, y) and angle
    outer_center_x, outer_center_y, outer_angle = params[36:]

    # Create inner hexagons
    inner_hexagons = []
    positions = []
    for i in range(12):
        center_x, center_y, angle = hex_params[i]
        vertices = generate_hexagon_vertices(center_x, center_y, angle)
        inner_hexagons.append(vertices)
        positions.append([center_x, center_y])

    # Build KDTree for fast overlap detection if requested
    kdtree = None
    if use_kdtree:
        kdtree = cKDTree(positions)

    # Check containment and overlap efficiently
    total_penalty = 0
    overlap_count = 0
    
    # Find maximum distance from center for outer hexagon sizing
    max_dist = 0
    for hex_vertices in inner_hexagons:
        for vertex in hex_vertices:
            dist = np.sqrt(vertex[0]**2 + vertex[1]**2)
            max_dist = max(max_dist, dist)

    # Calculate outer hexagon side length needed
    outer_radius = max_dist * 1.01  # Add buffer for numerical stability

    # Create outer hexagon
    outer_vertices = generate_hexagon_vertices(outer_center_x, outer_center_y, outer_angle, outer_radius)

    # Check containment
    for hex_vertices in inner_hexagons:
        if not check_containment(hex_vertices, outer_vertices):
            total_penalty += 10000

    # Optimized overlap checking
    if use_kdtree:
        # Use spatial indexing for efficient neighbor detection
        for i in range(12):
            # Find neighbors within a certain distance using KDTree
            indices = kdtree.query_ball_point(positions[i], 2.5)  # Approximate hex radius * 2.5
            for j in indices:
                if i >= j:
                    continue
                # Only check if they're actually nearby (faster than full O(n^2))
                if fast_check_overlap_pair(inner_hexagons[i], inner_hexagons[j]):
                    if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                        total_penalty += 10000
                        overlap_count += 1
    else:
        # Classic pairwise check for smaller populations
        for i in range(12):
            for j in range(i+1, 12):
                if fast_check_overlap_pair(inner_hexagons[i], inner_hexagons[j]):
                    if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                        total_penalty += 10000
                        overlap_count += 1

    # Return negative inverse of outer radius plus penalties
    return -(1.0 / (outer_radius + total_penalty + 1e-8))

def create_initial_population(pop_size=20):
    """Create diverse initial population with symmetric and random components."""
    population = []
    
    # Base symmetric configuration (good starting point)
    base_config = np.array([
        [0, 0, 0],           # center
        [-2.0, 0, 0],        # left
        [2.0, 0, 0],         # right
        [0, 2.0, 0],         # top
        [0, -2.0, 0],        # bottom
        [-1.5, 1.5, 0],      # top-left
        [1.5, 1.5, 0],       # top-right
        [-1.5, -1.5, 0],     # bottom-left
        [1.5, -1.5, 0],      # bottom-right
        [-2.5, 0, 0],        # far left
        [2.5, 0, 0],         # far right
        [0, -2.5, 0],        # far bottom
    ])
    
    # Create variations with different rotations and positions
    for _ in range(pop_size):
        # Start with base config
        individual = base_config.copy().astype(float)
        
        # Apply slight random perturbations
        for i in range(12):
            # Randomize positions slightly
            individual[i, 0] += np.random.normal(0, 0.3)
            individual[i, 1] += np.random.normal(0, 0.3)
            # Randomize angles
            individual[i, 2] += np.random.normal(0, 20)
        
        # Add outer hexagon parameters (centered with small variance)
        individual_flat = individual.flatten()
        outer_params = np.array([np.random.normal(0, 0.5), np.random.normal(0, 0.5), np.random.normal(0, 30)])
        
        # Combine into full parameter vector
        full_params = np.concatenate([individual_flat, outer_params])
        population.append(full_params)
        
    return population

def mutate_individual(individual, mutation_rate=0.3, std_dev=0.5):
    """Custom mutation operator for hexagon packing."""
    mutated = individual.copy()
    
    # Mutate hexagon positions and angles
    for i in range(12):
        if np.random.rand() < mutation_rate:
            # Slightly move hexagon
            mutated[i*3] += np.random.normal(0, std_dev)
            mutated[i*3+1] += np.random.normal(0, std_dev)
            # Slight rotation change
            mutated[i*3+2] += np.random.normal(0, 30)
    
    # Mutate outer hexagon parameters
    if np.random.rand() < mutation_rate:
        mutated[36] += np.random.normal(0, 1.0)  # outer center x
    if np.random.rand() < mutation_rate:
        mutated[37] += np.random.normal(0, 1.0)  # outer center y
    if np.random.rand() < mutation_rate:
        mutated[38] += np.random.normal(0, 30)   # outer angle
    
    return mutated

def crossover_parents(parent1, parent2, crossover_rate=0.8):
    """Custom crossover that preserves hexagon packing properties."""
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    if np.random.rand() < crossover_rate:
        # Swap segments between parents for better structural diversity
        segment_start = np.random.randint(0, 12)
        segment_end = np.random.randint(segment_start + 1, 13)
        
        # Swap hexagon positions and angles in segment
        child1[segment_start*3 : segment_end*3] = parent2[segment_start*3 : segment_end*3]
        child2[segment_start*3 : segment_end*3] = parent1[segment_start*3 : segment_end*3]
        
        # Also swap outer hexagon parameters
        child1[36:] = parent2[36:]
        child2[36:] = parent1[36:]
    
    return child1, child2

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses evolutionary algorithm with geometric-aware operators for superior performance.
    """
    # Initialize population
    population_size = 20
    population = create_initial_population(population_size)
    
    # Track best solution
    best_fitness = float('-inf')
    best_individual = None
    
    # Evolutionary algorithm parameters
    generations = 50
    mutation_rate = 0.3
    crossover_rate = 0.8
    elite_size = 2
    
    # Evaluation function with optimized spatial indexing
    eval_func = partial(evaluate_configuration_fast, use_kdtree=True)
    
    # Evolution loop
    for gen in range(generations):
        # Evaluate fitness for entire population
        fitness_scores = []
        for ind in population:
            try:
                fit = eval_func(ind)
                fitness_scores.append(fit)
            except Exception as e:
                fitness_scores.append(float('-inf'))
        
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]
        
        # Update best solution
        if sorted_fitness[0] > best_fitness:
            best_fitness = sorted_fitness[0]
            best_individual = sorted_population[0].copy()
        
        # Create new population
        new_population = []
        
        # Elitism: keep top individuals
        for i in range(elite_size):
            new_population.append(sorted_population[i])
        
        # Generate rest through selection, crossover, and mutation
        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = 3
            selected_indices = np.random.choice(len(sorted_population), tournament_size)
            selected_fitness = [sorted_fitness[i] for i in selected_indices]
            winner_idx = selected_indices[np.argmax(selected_fitness)]
            
            parent1 = sorted_population[winner_idx]
            
            # Select second parent
            selected_indices = np.random.choice(len(sorted_population), tournament_size)
            selected_fitness = [sorted_fitness[i] for i in selected_indices]
            winner_idx = selected_indices[np.argmax(selected_fitness)]
            parent2 = sorted_population[winner_idx]
            
            # Crossover
            child1, child2 = crossover_parents(parent1, parent2, crossover_rate)
            
            # Mutation
            child1 = mutate_individual(child1, mutation_rate, 0.5)
            child2 = mutate_individual(child2, mutation_rate, 0.5)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        population = new_population[:population_size]
        
        # Adaptive mutation rate based on convergence
        if gen > 10 and gen % 5 == 0:
            improvement = best_fitness - sorted_fitness[0]
            if improvement < 1e-6:
                mutation_rate = min(mutation_rate * 1.1, 0.7)  # Increase mutation if stuck
    
    # If we have a good solution, validate it properly
    if best_individual is not None and best_fitness != float('-inf'):
        try:
            # Evaluate final solution with strict checking
            final_fitness = evaluate_configuration_fast(best_individual, use_kdtree=False)
            
            # Extract final parameters
            hex_params = best_individual[:36].reshape(12, 3)
            outer_center_x, outer_center_y, outer_angle = best_individual[36:]
            
            # Calculate exact outer hexagon side length
            inner_hexagons = []
            for i in range(12):
                center_x, center_y, angle = hex_params[i]
                vertices = generate_hexagon_vertices(center_x, center_y, angle)
                inner_hexagons.append(vertices)
            
            # Find required outer hexagon radius
            max_dist = 0
            for hex_vertices in inner_hexagons:
                for vertex in hex_vertices:
                    dist = np.sqrt(vertex[0]**2 + vertex[1]**2)
                    max_dist = max(max_dist, dist)
            
            outer_side_length = max_dist * 1.01  # Add buffer
            
            # Generate outer hexagon vertices for validation
            outer_hex_vertices = generate_hexagon_vertices(outer_center_x, outer_center_y, outer_angle, outer_side_length)
            
            # Final validation 
            valid = True
            for hex_vertices in inner_hexagons:
                if not check_containment(hex_vertices, outer_hex_vertices):
                    valid = False
                    break
                    
            if not valid:
                # Fallback to simple configuration if validation fails
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
                outer_hex_side_length = 8
                return inner_hex_data, outer_hex_data, outer_hex_side_length
            
            # Format output
            inner_hex_data = hex_params.copy()
            outer_hex_data = np.array([outer_center_x, outer_center_y, outer_angle])
            
            return inner_hex_data, outer_hex_data, outer_side_length
            
        except Exception as e:
            pass
    
    # Fallback to original configuration if everything fails
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
    outer_hex_side_length = 8
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
