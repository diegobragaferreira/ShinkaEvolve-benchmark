# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from scipy.optimize import minimize_scalar, differential_evolution
import time
import random
from numba import jit

# Constants for hexagon geometry
HEX_RADIUS = 1.0  # Unit hexagon radius
HEX_APOGEE = HEX_RADIUS * np.sqrt(3) / 2  # Distance from center to side midpoint
HEX_HEIGHT = 2 * HEX_APOGEE  # Height of hexagon
HEX_WIDTH = 2 * HEX_RADIUS  # Width of hexagon

@jit(nopython=True)
def hexagon_vertices(x, y, angle_rad, radius=1.0):
    """Generate vertices of a hexagon given center, angle, and radius"""
    vertices = np.zeros((6, 2))
    for i in range(6):
        angle = angle_rad + i * np.pi / 3
        vertices[i, 0] = x + radius * np.cos(angle)
        vertices[i, 1] = y + radius * np.sin(angle)
    return vertices

def get_hexagon_polygon(x, y, angle_rad, radius=1.0):
    """Get shapely polygon representation of hexagon"""
    vertices = hexagon_vertices(x, y, angle_rad, radius)
    return Polygon(vertices)

def check_inner_hexagon_containment(hexagon_polygon, outer_hex_polygon):
    """Check if inner hexagon is fully contained in outer hexagon"""
    return outer_hex_polygon.contains(hexagon_polygon)

def check_hexagon_overlap(hex1, hex2):
    """Check if two hexagons overlap using Shapely"""
    return hex1.intersects(hex2)

def compute_outer_hexagon_radius(inner_hex_data, initial_guess=5.0):
    """
    Compute the minimum outer hexagon radius that can contain all inner hexagons
    Uses binary search approach
    """
    # Create a rough estimate of total span
    centers = inner_hex_data[:, :2]
    max_dist_from_origin = np.max(np.sqrt(centers[:, 0]**2 + centers[:, 1]**2)) + HEX_RADIUS
    
    # Binary search bounds
    low = max_dist_from_origin
    high = max_dist_from_origin * 3
    
    def can_fit(radius):
        # Create outer hexagon
        outer_poly = get_hexagon_polygon(0, 0, 0, radius)
        
        # Check containment for each inner hexagon
        for i in range(len(inner_hex_data)):
            x, y, angle = inner_hex_data[i]
            inner_poly = get_hexagon_polygon(x, y, np.radians(angle), HEX_RADIUS)
            
            if not check_inner_hexagon_containment(inner_poly, outer_poly):
                return False
                
        return True
    
    # Binary search for optimal radius
    tolerance = 1e-6
    while high - low > tolerance:
        mid = (low + high) / 2
        if can_fit(mid):
            high = mid
        else:
            low = mid
            
    return (low + high) / 2

def evaluate_layout(inner_hex_data):
    """Evaluate the current layout and return inverse radius"""
    try:
        # Compute minimal outer hexagon radius
        outer_radius = compute_outer_hexagon_radius(inner_hex_data)
        inv_radius = 1.0 / outer_radius
        
        # Check for overlaps between hexagons
        hex_polygons = []
        for i in range(len(inner_hex_data)):
            x, y, angle = inner_hex_data[i]
            poly = get_hexagon_polygon(x, y, np.radians(angle), HEX_RADIUS)
            hex_polygons.append(poly)
        
        # Check for overlaps
        for i in range(len(hex_polygons)):
            for j in range(i+1, len(hex_polygons)):
                if check_hexagon_overlap(hex_polygons[i], hex_polygons[j]):
                    # Return very small value if overlapping
                    return 0.0
        
        return inv_radius
        
    except Exception as e:
        return 0.0

def create_random_individual():
    """Create a random valid individual"""
    # Start with a reasonable configuration
    # Place center hexagon at origin
    individual = np.zeros((11, 3))  # (x, y, angle)
    individual[0] = [0, 0, 0]  # Center hexagon
    
    # Place others in a circular pattern around the center
    angles = np.linspace(0, 2*np.pi, 10, endpoint=False)
    radii = [1.5, 2.5, 3.5, 4.5]  # Different radii for layers
    
    idx = 1
    for i, angle in enumerate(angles):
        if idx >= 11:
            break
        r = radii[i % len(radii)]
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        individual[idx] = [x, y, 0]
        idx += 1
    
    # Add some random noise and ensure valid placements
    for i in range(11):
        individual[i, 0] += np.random.uniform(-0.5, 0.5)
        individual[i, 1] += np.random.uniform(-0.5, 0.5)
        individual[i, 2] += np.random.uniform(-30, 30)
    
    return individual

def crossover(parent1, parent2):
    """Perform crossover between two parents"""
    child = parent1.copy()
    # Single point crossover
    crossover_point = np.random.randint(1, 11)
    
    for i in range(crossover_point, 11):
        child[i] = parent2[i]
    
    return child

def mutate(individual, mutation_rate=0.1):
    """Mutate an individual"""
    mutated = individual.copy()
    
    for i in range(11):  # Skip center hexagon
        if np.random.random() < mutation_rate:
            # Mutate position
            mutated[i, 0] += np.random.normal(0, 0.2)
            mutated[i, 1] += np.random.normal(0, 0.2)
            # Mutate angle
            mutated[i, 2] += np.random.normal(0, 15)
    
    return mutated

def local_optimization_step(individual):
    """Perform local optimization on individual"""
    # Convert to optimization-friendly format
    def objective(params):
        # Reconstruct individual from flattened params
        temp_individual = individual.copy()
        idx = 0
        for i in range(11):
            temp_individual[i, 0] = params[idx]
            temp_individual[i, 1] = params[idx + 1]
            temp_individual[i, 2] = params[idx + 2]
            idx += 3
        
        return -evaluate_layout(temp_individual)  # Negative because we want to maximize
    
    # Use differential evolution for local optimization
    bounds = []
    for _ in range(11):
        bounds.extend([(-10, 10), (-10, 10), (-180, 180)])
    
    try:
        result = differential_evolution(objective, bounds, maxiter=30, popsize=10)
        if result.success:
            # Update individual with optimized values
            idx = 0
            for i in range(11):
                individual[i, 0] = result.x[idx]
                individual[i, 1] = result.x[idx + 1]
                individual[i, 2] = result.x[idx + 2]
                idx += 3
    except:
        pass  # If optimization fails, keep original
    
    return individual

def evolutionary_search(max_generations=200, population_size=50):
    """Perform evolutionary search to find optimal packing"""
    # Initialize population
    population = [create_random_individual() for _ in range(population_size)]
    
    best_fitness = 0.0
    best_individual = None
    
    for generation in range(max_generations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            fitness = evaluate_layout(individual)
            fitness_scores.append(fitness)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual.copy()
        
        # Select top individuals (tournament selection)
        sorted_indices = np.argsort(fitness_scores)[::-1][:population_size//2]
        selected_parents = [population[i] for i in sorted_indices]
        
        # Generate new population through crossover and mutation
        new_population = selected_parents[:]
        
        while len(new_population) < population_size:
            # Tournament selection
            parent1_idx = np.random.choice(len(selected_parents))
            parent2_idx = np.random.choice(len(selected_parents))
            
            child = crossover(selected_parents[parent1_idx], selected_parents[parent2_idx])
            child = mutate(child)
            
            # Local optimization on child
            child = local_optimization_step(child)
            
            new_population.append(child)
        
        population = new_population
        
        # Print progress
        if generation % 20 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")
    
    return best_individual, best_fitness

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    start_time = time.time()
    
    # Run evolutionary search
    try:
        final_individual, fitness = evolutionary_search(max_generations=150, population_size=40)
    except Exception as e:
        print(f"Evolutionary search failed: {e}")
        # Fallback to random attempt
        final_individual = create_random_individual()
        fitness = evaluate_layout(final_individual)
    
    # Final local optimization
    final_individual = local_optimization_step(final_individual)
    
    # Calculate actual outer hexagon radius
    outer_radius = compute_outer_hexagon_radius(final_individual)
    
    # Ensure we have valid results
    if fitness <= 0 or outer_radius <= 0:
        # Fallback to simple heuristic
        final_individual = np.array([
            [0, 0, 0],      # center
            [-2.5, 0, 0],   # left
            [2.5, 0, 0],    # right
            [-1.25, 2.17, 0],  # top-left
            [1.25, 2.17, 0],   # top-right
            [-1.25, -2.17, 0], # bottom-left
            [1.25, -2.17, 0],  # bottom-right
            [-3.75, 2.17, 0],  # far top-left
            [3.75, 2.17, 0],   # far top-right
            [-3.75, -2.17, 0], # far bottom-left
            [3.75, -2.17, 0],  # far bottom-right
        ])
        outer_radius = 8.0
    
    end_time = time.time()
    
    # Format output
    inner_hex_data = final_individual
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = outer_radius
    
    # Report metrics
    inv_outer_hex_side_length = 1.0 / outer_hex_side_length
    benchmark_ratio = inv_outer_hex_side_length / 0.2544
    
    print(f"Final Results:")
    print(f"  Inverse outer hex side length: {inv_outer_hex_side_length:.6f}")
    print(f"  Benchmark ratio: {benchmark_ratio:.6f}")
    print(f"  Eval time: {end_time - start_time:.2f}s")
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
