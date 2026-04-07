# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import SphericalVoronoi
import math
from deap import base, creator, tools, algorithms
import random

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def distance_ratio(points):
        """Calculate the ratio of minimum to maximum distance"""
        distances = squareform(pdist(points))
        # Set diagonal to large value so it doesn't affect min/max
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return min_dist / max_dist
    
    def uniformity_score(points):
        """Calculate uniformity score based on variance of distances"""
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        mean_dist = np.mean(distances)
        var_dist = np.var(distances)
        return 1.0 / (1.0 + var_dist / (mean_dist * mean_dist + 1e-10))
    
    def combined_objective(points):
        """Combined objective: maximize ratio and uniformity"""
        ratio = distance_ratio(points)
        uniformity = uniformity_score(points)
        # Weighted sum favoring ratio but penalizing poor uniformity
        return ratio * 0.7 + uniformity * 0.3
    
    # Create individual and population types
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    def init_individual():
        """Initialize individual with improved starting configuration"""
        # Start with icosahedron vertices
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        vertices = [
            (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
            (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
            (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1)
        ]
        
        points = np.array(vertices, dtype=float)
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        
        # Add remaining points using Fibonacci-like distribution
        remaining = 14 - len(points)
        for i in range(remaining):
            theta = math.acos(1 - 2 * (i / (remaining - 1)))
            phi_coord = (i * 2.414213562) % (2 * math.pi)
            
            x = math.sin(theta) * math.cos(phi_coord)
            y = math.sin(theta) * math.sin(phi_coord)
            z = math.cos(theta)
            points = np.vstack([points, [x, y, z]])
        
        # Apply controlled jittering
        np.random.seed(42)
        noise = np.random.normal(0, 0.01, points.shape)
        points += noise
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        
        # Flatten for DEAP representation
        return list(points.flatten())
    
    def evaluate(individual):
        """Evaluate fitness of individual"""
        points = np.array(individual).reshape(-1, 3)
        # Ensure points remain on unit sphere via normalization
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        return (combined_objective(points),)
    
    def mutate(individual):
        """Custom mutation operator"""
        points = np.array(individual).reshape(-1, 3)
        # Apply small random perturbations with adaptive scale
        scale = 0.005  # Smaller mutation for fine-tuning
        noise = np.random.normal(0, scale, points.shape)
        points += noise
        
        # Project back onto sphere
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        
        return tuple(points.flatten()),
    
    def crossover(ind1, ind2):
        """Custom crossover operator - blend two individuals"""
        points1 = np.array(ind1).reshape(-1, 3)
        points2 = np.array(ind2).reshape(-1, 3)
        
        # Blend points with 50% probability
        mask = np.random.rand(*points1.shape) > 0.5
        offspring = np.where(mask, points1, points2)
        
        # Normalize offspring points
        norms = np.linalg.norm(offspring, axis=1)
        offspring = offspring / norms[:, np.newaxis]
        
        return tuple(offspring.flatten()), tuple(offspring.flatten())
    
    # Register functions with toolbox
    toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Set up evolutionary parameters
    np.random.seed(42)
    random.seed(42)
    
    # Initialize population
    pop = toolbox.population(n=30)
    
    # Hall of fame to keep best individual
    hof = tools.HallOfFame(1)
    
    # Statistics monitoring
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Run evolution
    try:
        algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=50, 
                           stats=stats, halloffame=hof, verbose=False)
    except Exception:
        pass
    
    # Return best individual
    if len(hof) > 0:
        best_individual = hof[0]
        points = np.array(best_individual).reshape(-1, 3)
        # Final normalization for safety
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        return points
    else:
        # Fallback to simple initialization
        return init_individual().reshape(-1, 3)

# EVOLVE-BLOCK-END