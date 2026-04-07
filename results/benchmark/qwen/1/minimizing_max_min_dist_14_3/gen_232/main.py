# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.spatial import SphericalVoronoi
import warnings
from deap import base, creator, tools, algorithms
import random

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses evolutionary spherical Voronoi optimization approach.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Set fixed seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    def normalize_to_unit_sphere(points: np.ndarray) -> np.ndarray:
        """Normalize points to lie on unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        safe_norms = np.where(norms == 0, 1, norms)
        return points / safe_norms
    
    def calculate_min_max_ratio(points: np.ndarray) -> float:
        """Calculate the minimum-to-maximum distance ratio"""
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist <= 0:
            return 0.0
        return min_dist / max_dist
    
    def voronoi_initialization(n_points: int) -> np.ndarray:
        """Generate initial points using spherical Voronoi tessellation approach"""
        # Start with vertices of regular icosahedron
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
        ])
        
        # Normalize to unit sphere
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        vertices = vertices / np.where(norms > 0, norms, 1)
        
        # For 14 points, start with 12 icosahedron vertices and add 2 more
        if n_points <= 12:
            return vertices[:n_points]
        else:
            # Use Voronoi-based distribution for remaining points
            points = vertices.copy()
            remaining = n_points - 12
            
            # Distribute remaining points using spherical Voronoi sampling
            # Sample points on the sphere and compute their Voronoi cells
            for i in range(remaining):
                # Simple heuristic: place points near existing vertices with perturbation
                base_idx = i % 12
                base_point = vertices[base_idx].copy()
                
                # Add small random perturbation
                perturbation = np.random.normal(0, 0.1, 3)
                new_point = base_point + perturbation
                
                # Project back to sphere
                norm = np.linalg.norm(new_point)
                if norm > 0:
                    new_point = new_point / norm
                    
                points = np.vstack([points, new_point])
            
            return points[:n_points]
    
    def objective_ratio(points_flat: np.ndarray) -> float:
        """Objective function that maximizes min/max distance ratio"""
        points = points_flat.reshape(-1, 3)
        # Normalize points to unit sphere
        points = normalize_to_unit_sphere(points)
        
        # Compute distance matrix
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist <= 0:
            return -1.0
            
        # We want to maximize min_dist / max_dist, so minimize -min_dist / max_dist
        return -min_dist / max_dist
    
    def constraint_sphere(x):
        """Constraint function ensuring all points lie on unit sphere"""
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        # Return difference from unit radius (should be close to 0)
        return norms - 1.0
    
    def evaluate_individual(individual):
        """Evaluate fitness of an individual (point configuration)"""
        try:
            points = np.array(individual).reshape(-1, 3)
            points = normalize_to_unit_sphere(points)
            ratio = calculate_min_max_ratio(points)
            return (ratio,)
        except:
            return (-1.0,)
    
    def mutate_individual(individual):
        """Mutate an individual by adding small random perturbations"""
        mutated = individual[:]
        # Mutate ~20% of coordinates
        num_mutations = max(1, len(mutated) // 5)
        indices = random.sample(range(len(mutated)), num_mutations)
        for i in indices:
            mutated[i] += random.gauss(0, 0.02)
        return tuple(mutated)
    
    def crossover_individuals(ind1, ind2):
        """Crossover two individuals"""
        # Uniform crossover
        child1 = list(ind1)
        child2 = list(ind2)
        for i in range(len(ind1)):
            if random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
        return tuple(child1), tuple(child2)
    
    # Initialize evolutionary algorithm components
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -1.0, 1.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=42)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", crossover_individuals)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Generate initial population with diverse strategies
    population = []
    
    # Strategy 1: Voronoi-based initialization
    voronoi_points = voronoi_initialization(14)
    voronoi_points = normalize_to_unit_sphere(voronoi_points)
    population.append(toolbox.individual(voronoi_points.flatten().tolist()))
    
    # Strategy 2: Fibonacci spiral
    phi = np.pi * (3 - np.sqrt(5))
    fib_points = []
    for i in range(14):
        y = 1 - (i / 13) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        fib_points.append([x, y, z])
    fib_points = np.array(fib_points)
    fib_points = normalize_to_unit_sphere(fib_points)
    population.append(toolbox.individual(fib_points.flatten().tolist()))
    
    # Strategy 3: Random on sphere
    random_points = np.random.randn(14, 3)
    norms = np.linalg.norm(random_points, axis=1, keepdims=True)
    random_points = random_points / np.where(norms > 0, norms, 1)
    population.append(toolbox.individual(random_points.flatten().tolist()))
    
    # Strategy 4: Perturbed Voronoi points
    perturbed_voronoi = voronoi_points + np.random.normal(0, 0.05, voronoi_points.shape)
    perturbed_voronoi = normalize_to_unit_sphere(perturbed_voronoi)
    population.append(toolbox.individual(perturbed_voronoi.flatten().tolist()))
    
    # Strategy 5: Perturbed Fibonacci points
    perturbed_fib = fib_points + np.random.normal(0, 0.05, fib_points.shape)
    perturbed_fib = normalize_to_unit_sphere(perturbed_fib)
    population.append(toolbox.individual(perturbed_fib.flatten().tolist()))
    
    # Initialize population with diverse starting points
    for i in range(10-len(population)):
        random_init = np.random.randn(14, 3)
        norms = np.linalg.norm(random_init, axis=1, keepdims=True)
        random_init = random_init / np.where(norms > 0, norms, 1)
        population.append(toolbox.individual(random_init.flatten().tolist()))
    
    # Run evolutionary algorithm
    CXPB, MUTPB, NGEN = 0.5, 0.2, 20
    
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Run the evolutionary algorithm
    try:
        pop, logbook = algorithms.eaSimple(population, toolbox, CXPB, MUTPB, NGEN, 
                                         stats=stats, halloffame=hof, verbose=False)
        
        # Get the best individual found
        best_individual = hof[0]
        best_points = np.array(best_individual).reshape(-1, 3)
        best_points = normalize_to_unit_sphere(best_points)
        
        # Apply final local optimization to the best result
        try:
            cons = {'type': 'eq', 'fun': constraint_sphere}
            result = minimize(
                objective_ratio,
                best_points.flatten(),
                method='SLSQP',
                constraints=cons,
                options={'ftol': 1e-12, 'maxiter': 500},
                tol=1e-12
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 3)
                final_points = normalize_to_unit_sphere(final_points)
                return final_points
        except:
            pass
            
        return best_points
        
    except Exception as e:
        warnings.warn(f"Evolutionary algorithm failed: {str(e)}")
    
    # Fallback to Voronoi initialization with local optimization
    try:
        initial_points = voronoi_initialization(14)
        initial_points = normalize_to_unit_sphere(initial_points)
        
        cons = {'type': 'eq', 'fun': constraint_sphere}
        result = minimize(
            objective_ratio,
            initial_points.flatten(),
            method='SLSQP',
            constraints=cons,
            options={'ftol': 1e-12, 'maxiter': 500},
            tol=1e-12
        )
        
        if result.success:
            final_points = result.x.reshape(-1, 3)
            final_points = normalize_to_unit_sphere(final_points)
            return final_points
            
    except Exception as e:
        warnings.warn(f"Fallback optimization failed: {str(e)}")
    
    # Final fallback to Voronoi points
    final_points = voronoi_initialization(14)
    final_points = normalize_to_unit_sphere(final_points)
    return final_points

# EVOLVE-BLOCK-END