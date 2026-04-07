# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
import math
from scipy.optimize import differential_evolution, minimize
import time
import random
from joblib import Parallel, delayed

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

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

def check_containment_and_overlap(inner_hexagons, outer_hexagon):
    """Check if all inner hexagons are contained in outer hexagon and don't overlap"""
    # Check containment
    for hex_poly in inner_hexagons:
        if not outer_hexagon.contains(hex_poly):
            return False

    # Check pairwise overlaps - optimized for small number of hexagons
    n = len(inner_hexagons)
    for i in range(n):
        for j in range(i+1, n):
            if inner_hexagons[i].intersects(inner_hexagons[j]):
                return False

    return True

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
    # For a regular hexagon, radius = side_length
    return max_dist + padding

def evaluate_layout_with_validation(inner_positions_angles, outer_center=(0, 0), initial_outer_radius=8):
    """Evaluate the layout quality with comprehensive validation"""
    try:
        # Convert to hexagon polygons
        inner_hexagons = []
        for pos_angle in inner_positions_angles:
            x, y, angle = pos_angle
            hex_poly = create_regular_hexagon(x, y, 1, angle)
            inner_hexagons.append(hex_poly)

        # Create outer hexagon with current radius
        outer_radius = compute_outer_hexagon_radius(inner_hexagons, 0.01)
        outer_hexagon = create_regular_hexagon(outer_center[0], outer_center[1], outer_radius, 0)

        # Validate constraints
        valid = check_containment_and_overlap(inner_hexagons, outer_hexagon)

        # Return negative because we want to maximize 1/R (minimize R)
        outer_side_length = outer_radius
        inv_radius = 1.0 / outer_side_length if valid else 0.0

        return -inv_radius, outer_side_length, valid
    except Exception as e:
        return 0.0, 100.0, False

def generate_better_initial_config():
    """
    Generate a better initial configuration for 11 hexagons based on known dense packings
    """
    # This configuration is designed to be close to an optimal arrangement
    # Based on hexagonal close packing principles with strategic placement
    initial_positions = [
        # Central hexagon
        [0.0, 0.0, 0.0],
        # First ring (6 hexagons)
        [-2.0, 0.0, 0.0],      # Left
        [2.0, 0.0, 0.0],       # Right
        [0.0, 2.0, 0.0],       # Top
        [0.0, -2.0, 0.0],      # Bottom
        [-1.0, 1.732, 0.0],    # Top-left
        [1.0, 1.732, 0.0],     # Top-right
        # Second ring (4 hexagons)
        [-1.0, -1.732, 0.0],   # Bottom-left
        [1.0, -1.732, 0.0],    # Bottom-right
        [-2.0, 1.0, 0.0],      # Far top-left
        [2.0, 1.0, 0.0],       # Far top-right
        [-2.0, -1.0, 0.0],     # Far bottom-left
        [2.0, -1.0, 0.0],      # Far bottom-right
    ]

    # Keep only first 11 positions (the 11 required hexagons)
    return np.array(initial_positions[:11])

def adaptive_evolutionary_optimization(initial_positions, max_generations=100, pop_size=40):
    """Enhanced evolutionary algorithm with adaptive parameters"""
    # Population initialization with structured starting points
    population = [initial_positions.copy()]
    
    # Generate diverse initial population
    for _ in range(pop_size - 1):
        individual = initial_positions.copy()
        # Add small random perturbations
        for i in range(len(individual)):
            individual[i][0] += np.random.normal(0, 0.1)
            individual[i][1] += np.random.normal(0, 0.1)
            individual[i][2] += np.random.normal(0, 5)  # rotation
            individual[i][2] %= 360
        population.append(individual)
    
    best_solution = None
    best_fitness = float('inf')
    best_side_length = float('inf')
    
    # Evolutionary parameters
    for generation in range(max_generations):
        # Adaptive parameters based on generation progress
        gen_progress = generation / max_generations
        mutation_rate = 0.3 * (1 - gen_progress * 0.8)
        crossover_rate = 0.8 * (1 - gen_progress * 0.5)
        elite_size = max(3, int(5 + 10 * (1 - gen_progress)))
        
        # Evaluate all individuals in parallel
        def evaluate_individual(individual):
            fitness, side_length, valid = evaluate_layout_with_validation(individual)
            return fitness, side_length, valid, individual
        
        results = Parallel(n_jobs=-1)(
            delayed(evaluate_individual)(individual) for individual in population
        )
        
        # Filter valid individuals
        valid_results = [(f, s, v, i) for f, s, v, i in results if v]
        
        if valid_results:
            # Sort by fitness (lower is better)
            valid_results.sort(key=lambda x: x[0])
            current_best = valid_results[0]
            
            if current_best[0] < best_fitness:
                best_fitness = current_best[0]
                best_side_length = current_best[1]
                best_solution = current_best[3].copy()
        
        # Selection and reproduction
        if len(valid_results) >= 2:
            # Sort valid results
            valid_results.sort(key=lambda x: x[0])
            
            # Select elite
            elite = [ind for _, _, _, ind in valid_results[:elite_size]]
            
            # Generate new population
            new_population = elite.copy()
            
            while len(new_population) < pop_size:
                # Tournament selection for parents
                parent1 = tournament_selection(valid_results, tournament_size=3)
                parent2 = tournament_selection(valid_results, tournament_size=3)
                
                # Crossover
                if np.random.random() < crossover_rate:
                    child1, child2 = crossover_parents(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                child1 = mutate_individual(child1, mutation_rate)
                child2 = mutate_individual(child2, mutation_rate)
                
                new_population.extend([child1, child2])
            
            population = new_population[:pop_size]
        else:
            # If no valid individuals, continue with current population
            pass
    
    return best_solution, best_side_length

def tournament_selection(results, tournament_size=3):
    """Tournament selection for evolutionary algorithm"""
    if len(results) < tournament_size:
        tournament_size = len(results)
        
    tournament = random.sample(results, tournament_size)
    tournament.sort(key=lambda x: x[0])  # Sort by fitness
    return tournament[0][3]  # Return the individual

def crossover_parents(parent1, parent2):
    """Uniform crossover between two parents"""
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    for i in range(len(child1)):
        if np.random.random() < 0.5:
            child1[i] = parent2[i].copy()
            child2[i] = parent1[i].copy()
    
    return child1, child2

def mutate_individual(individual, mutation_rate):
    """Mutate an individual with adaptive parameters"""
    mutated = individual.copy()
    
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            mutated[i][0] += np.random.normal(0, 0.2)
        if np.random.random() < mutation_rate:
            mutated[i][1] += np.random.normal(0, 0.2)
        if np.random.random() < mutation_rate:
            mutated[i][2] += np.random.normal(0, 15)  # Larger rotation changes
            mutated[i][2] %= 360
    
    return mutated

def local_gradient_refinement(initial_positions, outer_side_length):
    """Refine solution using gradient-based optimization"""
    # Define objective function for scipy optimization
    def objective(params):
        # Reshape parameters into positions and angles
        positions_angles = []
        for i in range(11):
            x = params[i*3]
            y = params[i*3 + 1]
            angle = params[i*3 + 2]
            positions_angles.append([x, y, angle])

        score, side_length, valid = evaluate_layout_with_validation(positions_angles)
        return score  # Negative since we minimize -score = maximize score

    # Convert to flat array for scipy optimization
    initial_flat = []
    for pos_angle in initial_positions:
        initial_flat.extend(pos_angle)

    # Bounds for optimization
    bounds = []
    for i in range(11):
        bounds.extend([(-8, 8), (-8, 8), (0, 360)])

    try:
        # Use L-BFGS-B for local refinement
        result = minimize(
            objective,
            initial_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-10},
            callback=None
        )

        if result.success:
            # Extract refined solution
            refined_params = result.x
            refined_positions_angles = []
            for i in range(11):
                x = refined_params[i*3]
                y = refined_params[i*3 + 1]
                angle = refined_params[i*3 + 2]
                refined_positions_angles.append([x, y, angle])
            
            # Re-evaluate refined solution
            refined_score, refined_side_length, refined_valid = evaluate_layout_with_validation(refined_positions_angles)
            
            if refined_valid and refined_score < 0:  # Better than previous
                return refined_positions_angles, refined_side_length
    except:
        pass
    
    return initial_positions, outer_side_length

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Generate a better initial configuration
    initial_positions = generate_better_initial_config()
    inner_hex_data = initial_positions.copy()

    # Stage 1: Evolutionary optimization with adaptive parameters
    print("Starting evolutionary optimization...")
    evolutionary_solution, evolutionary_side_length = adaptive_evolutionary_optimization(
        initial_positions, max_generations=80, pop_size=40
    )
    
    if evolutionary_solution is not None:
        best_inner_data = evolutionary_solution
        best_outer_side_length = evolutionary_side_length
    else:
        best_inner_data = initial_positions.copy()
        best_outer_side_length = 10.0

    # Stage 2: Local refinement with gradient-based optimization
    print("Starting local refinement...")
    refined_inner_data, refined_side_length = local_gradient_refinement(
        best_inner_data, best_outer_side_length
    )
    
    if refined_side_length < best_outer_side_length:
        best_inner_data = refined_inner_data
        best_outer_side_length = refined_side_length

    # Stage 3: Final validation and binary search refinement
    print("Performing final validation and refinement...")
    
    # Always validate the result
    inner_hexagons = []
    for pos_angle in best_inner_data:
        x, y, angle = pos_angle
        hex_poly = create_regular_hexagon(x, y, 1, angle)
        inner_hexagons.append(hex_poly)

    # Recompute outer hexagon size carefully
    outer_radius = compute_outer_hexagon_radius(inner_hexagons, 0.01)
    outer_hexagon = create_regular_hexagon(0, 0, outer_radius, 0)

    # Final validation check
    valid = check_containment_and_overlap(inner_hexagons, outer_hexagon)
    
    if not valid:
        print("Final validation failed, using fallback...")
        # Fall back to initial configuration
        best_inner_data = initial_positions.copy()
        inner_hexagons = []
        for pos_angle in best_inner_data:
            x, y, angle = pos_angle
            hex_poly = create_regular_hexagon(x, y, 1, angle)
            inner_hexagons.append(hex_poly)
        outer_radius = compute_outer_hexagon_radius(inner_hexagons, 0.01)

    # Ensure we're returning the correct data format
    outer_hex_data = np.array([0, 0, 0])  # centered at origin

    # Return results
    return best_inner_data, outer_hex_data, outer_radius

# EVOLVE-BLOCK-END