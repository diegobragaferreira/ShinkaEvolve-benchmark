# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import SphericalVoronoi
import time
from itertools import combinations
import math
from deap import base, creator, tools, algorithms
from functools import partial
import random

def create_spherical_voronoi_initialization(n_points: int = 14) -> np.ndarray:
    """
    Create initial point configuration based on spherical Voronoi tiling principles.
    This provides a more structured starting point than random initialization.
    """
    # Generate points on a sphere using fibonacci-like method
    points = []
    
    # Use a modified fibonacci approach for better distribution
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    for i in range(n_points):
        # Distribute points more evenly on sphere
        y = 1 - (i / (n_points - 1)) * 2  # y from 1 to -1
        radius = np.sqrt(1 - y*y)
        
        theta = np.arctan2(y, radius) + (i * 2 * np.pi / n_points) 
        
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        points.append([x, y, z])
    
    points = np.array(points)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    return points

def calculate_min_max_ratio(points: np.ndarray) -> tuple[float, float, float]:
    """
    Calculate minimum and maximum distances between all point pairs.
    Returns (min_distance, max_distance, ratio).
    """
    if len(points) < 2:
        return 0.0, 0.0, 0.0
    
    distances = pdist(points)
    
    if len(distances) == 0:
        return 0.0, 0.0, 0.0
        
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    if max_dist <= 0:
        return 0.0, 0.0, 0.0
    
    ratio = min_dist / max_dist
    return min_dist, max_dist, ratio

def spherical_constraint(points: np.ndarray) -> np.ndarray:
    """Keep points on the unit sphere by normalizing them."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    return points / norms

def objective_function_for_minmax(points_flat: np.ndarray) -> float:
    """
    Objective function to maximize the min/max distance ratio.
    Returns negative ratio since optimizers minimize by default.
    """
    n, d = 14, 3
    points = points_flat.reshape(n, d)
    
    # Keep points on sphere
    points = spherical_constraint(points)
    
    # Ensure points are within reasonable bounds
    points = np.clip(points, -1.0, 1.0)
    
    # Calculate distances
    distances = pdist(points)
    
    if len(distances) == 0:
        return float('inf')
    
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    # Avoid division by zero
    if max_dist <= 0:
        return float('inf')
    
    # Prefer configurations with higher ratios, so return negative
    return -min_dist / max_dist

def calculate_fitness(points_flat: np.ndarray) -> float:
    """Calculate fitness as the inverse of negative ratio (higher is better)"""
    return -objective_function_for_minmax(points_flat)

def mutate_point(point, mu=0, sigma=0.05):
    """Mutate a single 3D point"""
    new_point = point + np.random.normal(mu, sigma, 3)
    return spherical_constraint(new_point.reshape(1, 3))[0]

def crossover_points(ind1, ind2):
    """Crossover two individuals by averaging their coordinates"""
    child1 = (ind1 + ind2) / 2.0
    child2 = (ind1 + ind2) / 2.0
    return spherical_constraint(child1.reshape(1, 3))[0], spherical_constraint(child2.reshape(1, 3))[0]

# Custom DEAP toolbox for 3D point optimization
def create_evolutionary_toolbox():
    # Define the fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Attribute generator - create a 3D point on sphere
    def create_individual():
        # Generate random point on unit sphere
        point = np.random.randn(3)
        norm = np.linalg.norm(point)
        if norm > 0:
            point = point / norm
        return creator.Individual(point.tolist())
    
    # Attribute generator for 14 points
    def create_full_individual():
        individual = []
        for _ in range(14):
            point = np.random.randn(3)
            norm = np.linalg.norm(point)
            if norm > 0:
                point = point / norm
            individual.extend(point.tolist())
        return creator.Individual(individual)
    
    toolbox.register("individual", create_full_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register evaluation function
    toolbox.register("evaluate", calculate_fitness)
    
    # Register selection operators
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Register crossover operator (uniform crossover for 3D points)
    def cx_uniform(ind1, ind2):
        # Convert to numpy arrays for easier manipulation
        arr1 = np.array(ind1).reshape(14, 3)
        arr2 = np.array(ind2).reshape(14, 3)
        
        # For each pair of points, swap with 50% probability
        mask = np.random.random((14, 3)) > 0.5
        child1 = np.where(mask, arr1, arr2)
        child2 = np.where(mask, arr2, arr1)
        
        return creator.Individual(child1.flatten().tolist()), creator.Individual(child2.flatten().tolist())
    
    toolbox.register("mate", cx_uniform)
    
    # Register mutation operator
    def mut_uniform(individual, indpb=0.05):
        for i in range(len(individual)):
            if random.random() < indpb:
                # Mutate this point coordinate
                idx = i // 3
                coord = i % 3
                individual[i] += np.random.normal(0, 0.03)  # Small Gaussian perturbation
        
        # Normalize all points to sphere
        arr = np.array(individual).reshape(14, 3)
        for i in range(14):
            norm = np.linalg.norm(arr[i])
            if norm > 0:
                arr[i] = arr[i] / norm
        return creator.Individual(arr.flatten().tolist()),
    
    toolbox.register("mutate", mut_uniform)
    
    return toolbox

def adaptive_evolutionary_optimization(initial_points: np.ndarray, max_time: float) -> np.ndarray:
    """
    Perform evolutionary optimization with adaptive parameters
    """
    # Initialize DEAP toolbox
    toolbox = create_evolutionary_toolbox()
    
    # Create initial population
    population = toolbox.population(n=20)
    
    # Initialize population with our best solution
    for i in range(len(population)):
        if i == 0:
            # First individual is our initial solution
            population[i] = toolbox.individual()
            # Copy initial points to first individual
            arr = np.array(initial_points).flatten()
            population[i][:] = arr.tolist()
        else:
            # Other individuals are random on sphere
            population[i] = toolbox.individual()
    
    # Statistics tracking
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Evolution parameters
    CXPB = 0.7  # Crossover probability
    MUTPB = 0.2  # Mutation probability
    NGEN = 100  # Number of generations
    
    # Run evolution
    start_time = time.time()
    best_ind = None
    best_fitness = float('-inf')
    
    for gen in range(NGEN):
        if time.time() - start_time > max_time:
            break
            
        # Select parents
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation
        for i in range(0, len(offspring)-1, 2):
            if random.random() < CXPB:
                child1, child2 = toolbox.mate(offspring[i], offspring[i+1])
                del offspring[i].fitness.values
                del offspring[i+1].fitness.values
                offspring[i] = child1
                offspring[i+1] = child2
                
        for i in range(len(offspring)):
            if random.random() < MUTPB:
                mutated, = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values
                offspring[i] = mutated
                
        # Evaluate fitness of new individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)
            
        # Replace population
        population[:] = offspring
        
        # Track best individual
        best_in_gen = tools.selBest(population, 1)[0]
        if best_in_gen.fitness.values[0] > best_fitness:
            best_fitness = best_in_gen.fitness.values[0]
            best_ind = toolbox.clone(best_in_gen)
    
    # Return best solution found
    if best_ind is not None:
        return np.array(best_ind).reshape(14, 3)
    else:
        # Fallback to initial points
        return initial_points

def local_geometric_refinement(points: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """
    Apply geometric refinement using local search with distance constraints.
    """
    current_points = points.copy()
    best_points = current_points.copy()
    _, _, best_ratio = calculate_min_max_ratio(current_points)
    
    for iteration in range(max_iter):
        # Try small random perturbations
        for i in range(len(current_points)):
            # Perturb each point slightly
            perturbation = np.random.normal(0, 0.001, 3)
            new_points = current_points.copy()
            new_points[i] += perturbation
            new_points[i] = spherical_constraint(new_points[i:i+1])[0]
            
            # Check if this improves the ratio
            _, _, ratio = calculate_min_max_ratio(new_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = new_points.copy()
                current_points = new_points.copy()
    
    return best_points

def symmetric_rotation(points: np.ndarray, num_rotations: int = 8) -> np.ndarray:
    """
    Generate rotated versions of point set using quaternion-based rotations.
    """
    # Simple rotation around z-axis
    angles = np.linspace(0, 2*np.pi, num_rotations, endpoint=False)
    rotated_sets = []
    
    for angle in angles:
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        rotated_points = points @ rotation_matrix.T
        rotated_sets.append(rotated_points)
    
    return np.vstack(rotated_sets)

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses evolutionary computation with spherical Voronoi initialization.
    
    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Phase 1: Initial configuration using spherical Voronoi principles
    initial_points = create_spherical_voronoi_initialization(14)
    
    # Phase 2: Evolutionary optimization with adaptive parameters
    optimized_points = adaptive_evolutionary_optimization(initial_points, max_time=280.0)
    
    # Phase 3: Local geometric refinement
    optimized_points = local_geometric_refinement(optimized_points, max_iter=100)
    
    # Phase 4: Generate symmetric variations for exploration
    symmetric_candidates = symmetric_rotation(optimized_points, num_rotations=6)
    
    # Evaluate all candidates and select best
    best_points = optimized_points.copy()
    best_ratio = 0.0
    
    # Check current optimized version
    _, _, current_ratio = calculate_min_max_ratio(optimized_points)
    if current_ratio > best_ratio:
        best_ratio = current_ratio
        best_points = optimized_points.copy()
    
    # Check symmetric variants (if any are better)
    num_candidates = len(symmetric_candidates) // 14
    for i in range(num_candidates):
        candidate_points = symmetric_candidates[i*14:(i+1)*14]
        _, _, ratio = calculate_min_max_ratio(candidate_points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = candidate_points.copy()
    
    # Phase 5: Final local optimization for polishing
    try:
        # Apply one more round of local refinement
        final_points = local_geometric_refinement(best_points, max_iter=50)
        _, _, final_ratio = calculate_min_max_ratio(final_points)
        
        if final_ratio > best_ratio:
            best_points = final_points
            
    except:
        pass
    
    # Ensure final result is properly bounded and normalized
    best_points = spherical_constraint(best_points)
    
    return best_points

# EVOLVE-BLOCK-END