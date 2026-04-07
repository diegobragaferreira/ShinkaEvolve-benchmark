# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
import math
import random
import time
from collections import defaultdict

# Quadtree-based spatial indexing for efficient collision detection
class QuadTree:
    def __init__(self, boundary, capacity=4):
        self.boundary = boundary  # (x, y, width, height)
        self.capacity = capacity
        self.points = []
        self.divided = False
        self.nw = None
        self.ne = None
        self.sw = None
        self.se = None
        
    def insert(self, point, data):
        if not self._in_boundary(point):
            return False
            
        if len(self.points) < self.capacity and not self.divided:
            self.points.append((point, data))
            return True
            
        if not self.divided:
            self._subdivide()
            
        return (self.nw.insert(point, data) or 
                self.ne.insert(point, data) or 
                self.sw.insert(point, data) or 
                self.se.insert(point, data))
    
    def _in_boundary(self, point):
        x, y = point
        bx, by, w, h = self.boundary
        return bx <= x <= bx + w and by <= y <= by + h
        
    def _subdivide(self):
        x, y, w, h = self.boundary
        half_w, half_h = w / 2, h / 2
        
        self.nw = QuadTree((x, y, half_w, half_h), self.capacity)
        self.ne = QuadTree((x + half_w, y, half_w, half_h), self.capacity)
        self.sw = QuadTree((x, y + half_h, half_w, half_h), self.capacity)
        self.se = QuadTree((x + half_w, y + half_h, half_w, half_h), self.capacity)
        
        for point, data in self.points:
            self.nw.insert(point, data) or \
            self.ne.insert(point, data) or \
            self.sw.insert(point, data) or \
            self.se.insert(point, data)
        
        self.points = []
        self.divided = True
        
    def query_range(self, range_rect):
        """Query all points within a rectangular range"""
        result = []
        x, y, w, h = range_rect
        bx, by, bw, bh = self.boundary
        
        # Check if ranges intersect
        if not (x + w < bx or x > bx + bw or y + h < by or y > by + bh):
            if self.divided:
                result.extend(self.nw.query_range(range_rect))
                result.extend(self.ne.query_range(range_rect))
                result.extend(self.sw.query_range(range_rect))
                result.extend(self.se.query_range(range_rect))
            else:
                for point, data in self.points:
                    if x <= point[0] <= x + w and y <= point[1] <= y + h:
                        result.append((point, data))
        return result

def generate_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, rotation, and side length."""
    angle_rad = math.radians(angle_deg)
    # Vertices of a regular hexagon with side_length=1 centered at origin
    base_vertices = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        x = math.cos(theta)
        y = math.sin(theta)
        base_vertices.append((x, y))

    # Scale and translate
    vertices = [(center_x + side_length * vx, center_y + side_length * vy) for vx, vy in base_vertices]
    return vertices

def check_containment(hexagon_vertices, outer_hexagon_vertices):
    """Check if all vertices of inner hexagon are inside outer hexagon."""
    inner_poly = Polygon(hexagon_vertices)
    outer_poly = Polygon(outer_hexagon_vertices)
    return outer_poly.contains(inner_poly)

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def compute_voronoi_hex_centers(n_points=11, bounds=(-5, 5)):
    """Generate initial hexagon centers using Voronoi diagram."""
    # Generate random points
    np.random.seed(42)
    points = np.random.uniform(bounds[0], bounds[1], size=(n_points, 2))

    # Compute Voronoi diagram
    vor = Voronoi(points)

    # Take the centroids of the finite Voronoi cells
    centroids = []
    for region in vor.regions:
        if len(region) > 0 and -1 not in region:
            points_in_region = [vor.vertices[i] for i in region if i >= 0]
            if points_in_region:
                centroid = np.mean(points_in_region, axis=0)
                # Only keep centroids within bounds
                if bounds[0] <= centroid[0] <= bounds[1] and bounds[0] <= centroid[1] <= bounds[1]:
                    centroids.append(centroid)

    # If we don't have enough centroids, add some random ones
    while len(centroids) < n_points:
        centroids.append(np.random.uniform(bounds[0], bounds[1], size=2))

    # Take first n_points
    return np.array(centroids[:n_points])

def initialize_population(pop_size, n_hexagons=11, bounds=(-5, 5)):
    """Initialize population with Voronoi-based configurations."""
    population = []
    for _ in range(pop_size):
        # Get Voronoi-based centers
        centers = compute_voronoi_hex_centers(n_hexagons, bounds)

        # Add some randomness
        individual = []
        for i in range(n_hexagons):
            x, y = centers[i]
            # Add some noise to positions
            x += np.random.normal(0, 0.3)
            y += np.random.normal(0, 0.3)
            # Random rotation
            angle = np.random.uniform(-180, 180)
            individual.extend([x, y, angle])

        # Add outer hexagon parameters (center, angle, side_length)
        individual.extend([0.0, 0.0, 0.0, 4.0])  # reasonable starting side length

        population.append(individual)

    return population

def evaluate_individual_with_spatial_index(params, spatial_tree=None, return_penalties=False):
    """Evaluate a single individual with spatial indexing for faster collision detection."""
    # Extract inner hexagon positions and rotations
    inner_params = params[:-4]
    outer_center_x, outer_center_y, outer_angle, outer_side_length = params[-4:]

    # Create inner hexagons
    inner_hexagons = []
    hex_positions = []
    for i in range(11):
        x, y, theta = inner_params[3*i:3*i+3]
        vertices = generate_hexagon_vertices(x, y, theta, 1.0)
        inner_hexagons.append(vertices)
        hex_positions.append((x, y))

    # Create outer hexagon
    outer_vertices = generate_hexagon_vertices(outer_center_x, outer_center_y, outer_angle, outer_side_length)

    # Check constraints
    penalty = 0
    penalties = []

    # Check containment
    for vertices in inner_hexagons:
        if not check_containment(vertices, outer_vertices):
            penalty += 1000000
            penalties.append("containment")

    # Check overlaps using spatial indexing if provided
    if spatial_tree is not None:
        # Use spatial tree for faster neighbor detection
        for i in range(len(inner_hexagons)):
            pos_i = hex_positions[i]
            # Query nearby points to see if we should check for collision
            x, y = pos_i
            nearby_range = (x - 2.5, y - 2.5, 5, 5)  # Search area around hexagon
            neighbors = spatial_tree.query_range(nearby_range)
            for j, _ in neighbors:
                if i != j:
                    if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                        penalty += 1000000
                        penalties.append("overlap")
    else:
        # Fallback to brute-force (used for very early stages)
        for i in range(len(inner_hexagons)):
            for j in range(i+1, len(inner_hexagons)):
                if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                    penalty += 1000000
                    penalties.append("overlap")

    # Return negative of inverse side length plus penalties
    if penalty > 0:
        if return_penalties:
            return penalty + 1.0 / outer_side_length, penalties
        return penalty + 1.0 / outer_side_length
    else:
        if return_penalties:
            return -1.0 / outer_side_length, []
        return -1.0 / outer_side_length

def evaluate_individual(params, return_penalties=False):
    """Wrapped evaluation function for compatibility."""
    return evaluate_individual_with_spatial_index(params, return_penalties=return_penalties)

def tournament_selection(population, fitnesses, tournament_size=3):
    """Select an individual using tournament selection."""
    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmin(tournament_fitnesses)]
    return population[winner_index]

def crossover(parent1, parent2):
    """Crossover operation for hexagon packing."""
    # Uniform crossover for positions and rotations
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Crossover for inner hexagons
    for i in range(11):
        if random.random() < 0.5:
            child1[3*i:3*i+3] = parent2[3*i:3*i+3]
            child2[3*i:3*i+3] = parent1[3*i:3*i+3]

    # Crossover for outer hexagon parameters
    if random.random() < 0.5:
        child1[-4:] = parent2[-4:]
        child2[-4:] = parent1[-4:]

    return child1, child2

def mutate(individual, mutation_rate=0.1, bounds=(-10, 10), adaptive_mutation=False, generation=None, max_generations=None):
    """Mutation operation for hexagon packing with adaptive mutation rate."""
    mutated = individual.copy()

    # Adaptive mutation rate - decrease over time
    if adaptive_mutation and generation is not None and max_generations is not None:
        # Start with higher mutation rate, decrease over generations
        effective_mutation_rate = mutation_rate * (1.0 - generation/max_generations)
        # Ensure it doesn't go too low
        effective_mutation_rate = max(effective_mutation_rate, 0.01)
    else:
        effective_mutation_rate = mutation_rate

    # Mutate inner hexagons
    for i in range(11):
        for j in range(3):  # x, y, angle
            if random.random() < effective_mutation_rate:
                if j < 2:  # x or y
                    mutated[3*i+j] += np.random.normal(0, 0.5)
                    # Keep within bounds
                    mutated[3*i+j] = np.clip(mutated[3*i+j], bounds[0], bounds[1])
                else:  # angle
                    mutated[3*i+j] += np.random.normal(0, 30)
                    # Normalize angle to [-180, 180]
                    mutated[3*i+j] = ((mutated[3*i+j] + 180) % 360) - 180

    # Mutate outer hexagon
    if random.random() < effective_mutation_rate:
        mutated[-1] += np.random.normal(0, 0.5)  # mutate side_length
        mutated[-1] = max(1.0, mutated[-1])  # ensure positive side_length

    return mutated

def adaptive_genetic_algorithm_optimization():
    """Perform adaptive genetic algorithm optimization with dynamic parameters."""
    pop_size = 30
    max_generations = 50
    initial_mutation_rate = 0.8  # High initial mutation for exploration

    # Initialize population
    population = initialize_population(pop_size)

    best_fitness_history = []

    for gen in range(max_generations):
        # Evaluate fitness
        fitnesses = [evaluate_individual(ind) for ind in population]

        # Find best individual
        best_idx = np.argmin(fitnesses)
        best_fitness = fitnesses[best_idx]
        best_fitness_history.append(best_fitness)

        # Print progress
        if gen % 10 == 0:
            print(f"Generation {gen}: Best fitness = {-best_fitness}")

        # Early stopping if no improvement for last 10 generations
        if len(best_fitness_history) > 10:
            recent_improvement = best_fitness_history[-10] - best_fitness_history[-1]  # Should be positive if improving
            if recent_improvement < 1e-8 and gen > 20:  # Very small improvement, stop early
                print(f"Early stopping at generation {gen}")
                break

        # Create new population
        new_population = []

        # Elitism: keep best individual
        new_population.append(population[best_idx].copy())

        # Generate offspring
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            child1, child2 = crossover(parent1, parent2)

            child1 = mutate(child1, initial_mutation_rate, adaptive_mutation=True, generation=gen, max_generations=max_generations)
            child2 = mutate(child2, initial_mutation_rate, adaptive_mutation=True, generation=gen, max_generations=max_generations)

            new_population.extend([child1, child2])

        # Trim to exact population size
        population = new_population[:pop_size]

    # Return best individual
    fitnesses = [evaluate_individual(ind) for ind in population]
    best_idx = np.argmin(fitnesses)
    return population[best_idx]

def local_refinement_improved(params, max_evaluations=500):
    """Improved local refinement using multiple optimization techniques."""
    # Try differential evolution first (global optimization)
    try:
        # Create a wrapper for the objective function that takes the right parameters
        def wrapped_de_objective(x):
            # Reshape x to match the full parameters (add back outer hexagon)
            full_params = list(x) + [params[-4], params[-3], params[-2], params[-1]]
            return evaluate_individual(full_params)

        # Bounds for inner hexagons only (33 parameters)
        bounds = [(-10, 10), (-10, 10), (-180, 180)] * 11

        # Run differential evolution on inner parameters
        result_de = differential_evolution(
            wrapped_de_objective,
            bounds,
            maxiter=20,
            popsize=5,
            seed=42,
            disp=False
        )

        if result_de.success:
            # Update params with DE result
            refined_inner_params = result_de.x
            refined_params = list(refined_inner_params) + [params[-4], params[-3], params[-2], params[-1]]
            return refined_params
    except Exception as e:
        print(f"Differential evolution failed: {e}")

    # Fall back to Nelder-Mead if DE fails
    try:
        def wrapped_nm_objective(x):
            full_params = list(x) + [params[-4], params[-3], params[-2], params[-1]]
            return evaluate_individual(full_params)

        # Optimize just the inner hexagons using Nelder-Mead
        x0 = params[:-4]  # Remove outer hexagon parameters

        result_nm = minimize(
            wrapped_nm_objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'disp': False}
        )

        if result_nm.success:
            refined_inner_params = result_nm.x
            refined_params = list(refined_inner_params) + [params[-4], params[-3], params[-2], params[-1]]
            return refined_params
    except Exception as e:
        print(f"Nelder-Mead failed: {e}")

    # If all fail, return original params
    return params

def generate_diverse_initial_configs():
    """Generate multiple diverse initial configurations."""
    configs = []
    
    # Configuration 1: Hexagonal close-packed pattern
    config1 = [
        [0, 0, 0],  # center
        [-2.0, 0, 0],  # left
        [2.0, 0, 0],  # right
        [0, 2.0, 0],  # top
        [0, -2.0, 0],  # bottom
        [-1.0, 1.0, 0],  # top-left
        [1.0, 1.0, 0],  # top-right
        [-1.0, -1.0, 0],  # bottom-left
        [1.0, -1.0, 0],  # bottom-right
        [-2.5, 1.5, 0],  # far top-left
        [2.5, 1.5, 0],  # far top-right
    ]
    configs.append(config1)
    
    # Configuration 2: Spiral pattern
    config2 = [
        [0, 0, 0],
        [0, 2.5, 0],
        [2.165, 1.25, 0],
        [2.165, -1.25, 0],
        [0, -2.5, 0],
        [-2.165, -1.25, 0],
        [-2.165, 1.25, 0],
        [0, 4.0, 0],
        [3.464, 2.0, 0],
        [3.464, -2.0, 0],
        [0, -4.0, 0],
    ]
    configs.append(config2)
    
    # Configuration 3: Linear pattern
    config3 = [
        [0, 0, 0],
        [-2.5, 0, 0],
        [2.5, 0, 0],
        [0, 2.5, 0],
        [0, -2.5, 0],
        [-2.5, 2.5, 0],
        [2.5, 2.5, 0],
        [-2.5, -2.5, 0],
        [2.5, -2.5, 0],
        [-4.0, 0, 0],
        [4.0, 0, 0],
    ]
    configs.append(config3)
    
    # Configuration 4: Honeycomb pattern
    config4 = [
        [0, 0, 0],
        [-1.732, 0, 0],
        [1.732, 0, 0],
        [0, 1.732, 0],
        [0, -1.732, 0],
        [-0.866, 0.866, 0],
        [0.866, 0.866, 0],
        [-0.866, -0.866, 0],
        [0.866, -0.866, 0],
        [-1.732, 1.732, 0],
        [1.732, 1.732, 0],
    ]
    configs.append(config4)
    
    # Configuration 5: Random perturbed version of hexagonal pattern
    config5 = [
        [0, 0, 0],
        [-2.1, 0.1, 0],
        [2.2, -0.1, 0],
        [0.1, 2.1, 0],
        [-0.1, -2.1, 0],
        [-1.0, 1.0, 0],
        [1.1, 1.2, 0],
        [-0.9, -1.1, 0],
        [1.0, -0.9, 0],
        [-2.4, 1.4, 0],
        [2.6, 1.6, 0],
    ]
    configs.append(config5)
    
    return configs

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses adaptive hybrid genetic algorithm with Voronoi initialization and local refinement.
    """
    start_time = time.time()
    max_time_seconds = 175  # Leave some margin for cleanup
    
    best_fitness = float('inf')
    best_params = None
    
    # Multi-start optimization: try multiple initial configurations
    initial_configs = generate_diverse_initial_configs()
    
    # Also run the standard GA for comparison
    ga_configs = []
    
    for i, config in enumerate(initial_configs):
        # Convert to proper format
        individual = []
        for x, y, angle in config:
            individual.extend([x, y, angle])
        individual.extend([0.0, 0.0, 0.0, 4.0])  # outer hexagon params
        ga_configs.append(individual)
    
    # Run GA on each initial configuration
    for i, individual in enumerate(ga_configs):
        if time.time() - start_time > max_time_seconds - 5:
            break
            
        try:
            # Use a more aggressive approach for the initial configs
            # Run a shorter GA on this specific starting point with enhanced settings
            pop_size = 20
            max_generations = 30
            initial_mutation_rate = 0.5
            
            # Initialize population starting with this individual
            population = [individual] + initialize_population(pop_size - 1)
            
            for gen in range(max_generations):
                # Evaluate fitness
                fitnesses = [evaluate_individual(ind) for ind in population]
                
                # Find best individual
                best_idx = np.argmin(fitnesses)
                current_best_fitness = fitnesses[best_idx]
                
                if current_best_fitness < best_fitness:
                    best_fitness = current_best_fitness
                    best_params = population[best_idx].copy()
                
                # Early stopping if we're close to a good solution
                if abs(current_best_fitness) < 0.255:  # Close to the benchmark
                    break
                
                # Create new population with this individual as elite
                new_population = [population[best_idx].copy()]
                
                # Generate offspring
                while len(new_population) < pop_size:
                    parent1 = tournament_selection(population, fitnesses)
                    parent2 = tournament_selection(population, fitnesses)
                    
                    child1, child2 = crossover(parent1, parent2)
                    
                    child1 = mutate(child1, initial_mutation_rate, adaptive_mutation=True, generation=gen, max_generations=max_generations)
                    child2 = mutate(child2, initial_mutation_rate, adaptive_mutation=True, generation=gen, max_generations=max_generations)
                    
                    new_population.extend([child1, child2])
                
                # Trim to exact population size
                population = new_population[:pop_size]
                
                # Early stopping if no improvement in last 10 generations
                if gen > 10 and abs(fitnesses[best_idx] - current_best_fitness) < 1e-6:
                    break
                    
        except Exception as e:
            print(f"GA run {i} failed: {e}")
            continue
    
    # If we didn't get a good solution from GA, try direct optimization
    if best_params is None:
        try:
            # Run the main optimizer as fallback
            best_params = adaptive_genetic_algorithm_optimization()
        except Exception as e:
            print(f"Main GA failed: {e}")
            # Fallback to a good initial configuration
            best_params = [
                0.0, 0.0, 0.0,      # center hexagon
                -2.0, 0.0, 0.0,     # left
                2.0, 0.0, 0.0,      # right
                0.0, 2.0, 0.0,      # top
                0.0, -2.0, 0.0,     # bottom
                -1.0, 1.0, 0.0,     # top-left
                1.0, 1.0, 0.0,      # top-right
                -1.0, -1.0, 0.0,    # bottom-left
                1.0, -1.0, 0.0,     # bottom-right
                -2.0, 1.5, 0.0,     # far top-left
                2.0, 1.5, 0.0,      # far top-right
                0.0, 0.0, 0.0, 4.0  # outer hexagon parameters
            ]

    # Local refinement with adaptive optimization
    if time.time() - start_time < max_time_seconds - 5:
        try:
            best_params = local_refinement_improved(best_params)
        except Exception as e:
            print(f"Local refinement failed: {e}")

    # Extract final results
    inner_params = best_params[:-4]
    outer_center_x, outer_center_y, outer_angle, outer_side_length = best_params[-4:]

    # Format inner hexagon data
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i] = [inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]]

    outer_hex_data = np.array([outer_center_x, outer_center_y, outer_angle])

    return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END