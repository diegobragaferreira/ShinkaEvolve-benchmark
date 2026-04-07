# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

def generate_icosahedron():
    """Generate vertices of a regular icosahedron"""
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    # Normalize to unit sphere
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    return vertices / norms

def project_to_sphere(points):
    """Project points onto unit sphere"""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return points / norms

def tetrahedral_subdivision(vertices, depth=1):
    """Subdivide tetrahedra to create more points"""
    if depth <= 0:
        return vertices
    
    # Create a simple subdivision by adding midpoints
    new_vertices = vertices.copy()
    
    # For 14 points, we'll build a structure that maintains the right number
    # Start with icosahedron vertices and subdivide appropriately
    edges = []
    # Connect adjacent vertices of icosahedron (simplified approach)
    
    # Create a simpler subdivision pattern
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            # Simple connection rule - connect nearby points
            dist = np.linalg.norm(vertices[i] - vertices[j])
            if dist < 1.5:  # approximate adjacency
                edges.append((i, j))
    
    # Add midpoints of edges
    midpoints = []
    for i, j in edges[:len(edges)//2]:  # take half the edges
        midpoint = (vertices[i] + vertices[j]) / 2
        midpoints.append(midpoint)
    
    # Combine original vertices with new ones, but keep only enough to reach 14
    if len(midpoints) > 0:
        selected_midpoints = np.array(midpoints[:14-len(vertices)])
        new_vertices = np.vstack([vertices, selected_midpoints])
    
    # Project all to sphere
    return project_to_sphere(new_vertices[:14])

def spherical_similarity_score(points):
    """Compute a score based on how evenly distributed the points are on sphere"""
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    
    # Calculate minimum and maximum distances
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    if max_dist == 0:
        return 0
        
    # Ratio of min to max distances (our primary metric)
    ratio = min_dist / max_dist
    
    # Bonus for symmetry - check if points form symmetric pattern
    # Simple check: compute average distance from center and variance
    center_distances = np.linalg.norm(points, axis=1)
    center_variance = np.var(center_distances)
    
    # Penalize if points are not uniformly distributed in radial direction
    if center_variance > 0.01:
        ratio *= (1 - center_variance * 100)
    
    return ratio

def evolve_tessellation_population(pop_size=20, generations=100):
    """Evolve a population of point configurations"""
    
    # Start with icosahedron as base structure
    base_points = generate_icosahedron()
    
    # Generate initial population around the icosahedron
    population = []
    for i in range(pop_size):
        # Start with icosahedron points
        points = base_points.copy()
        
        # Add small random perturbations
        np.random.seed(i * 42)
        perturbation = np.random.normal(0, 0.05, points.shape)
        points += perturbation
        
        # Project to sphere
        points = project_to_sphere(points)
        
        population.append(points)
    
    best_individual = None
    best_fitness = 0
    
    # Evolutionary process
    for gen in range(generations):
        fitness_scores = []
        
        # Evaluate fitness of all individuals
        for individual in population:
            fitness = spherical_similarity_score(individual)
            fitness_scores.append(fitness)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual.copy()
        
        # Selection - keep top 50%
        sorted_indices = np.argsort(fitness_scores)[::-1]
        top_indices = sorted_indices[:pop_size//2]
        
        # Create new generation through crossover and mutation
        new_population = []
        for i in range(pop_size):
            parent1_idx = top_indices[i % len(top_indices)]
            parent2_idx = top_indices[(i + 1) % len(top_indices)]
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover: blend points
            alpha = np.random.random()
            child = alpha * parent1 + (1 - alpha) * parent2
            
            # Mutation: add small random noise
            np.random.seed(gen * pop_size + i)
            mutation = np.random.normal(0, 0.01, child.shape)
            child += mutation
            
            # Project to sphere
            child = project_to_sphere(child)
            
            new_population.append(child)
        
        population = new_population
    
    return best_individual if best_individual is not None else population[0]

def adaptive_optimization(points, max_iter=500):
    """Refine the point configuration using adaptive optimization"""
    
    def objective(x):
        points_arr = x.reshape(-1, 3)
        distances = cdist(points_arr, points_arr)
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 1e10
        return -min_dist / max_dist
    
    def constraint_sphere(x):
        points_arr = x.reshape(-1, 3)
        norms = np.linalg.norm(points_arr, axis=1)
        return 1 - norms
    
    # Multi-stage optimization to avoid local minima
    current_points = points.copy()
    
    # Stage 1: Differential Evolution for global search
    n_vars = len(current_points) * 3
    bounds = [(-1, 1) for _ in range(n_vars)]
    
    try:
        de_result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7,
            disp=False
        )
        
        if de_result.success:
            current_points = de_result.x.reshape(-1, 3)
            current_points = project_to_sphere(current_points)
    except:
        pass
    
    # Stage 2: Local optimization
    try:
        cons = [{'type': 'ineq', 'fun': constraint_sphere}]
        result = minimize(
            objective,
            current_points.flatten(),
            method='L-BFGS-B',
            constraints=cons,
            options={'maxiter': 300, 'ftol': 1e-12},
            tol=1e-12
        )
        
        if result.success:
            current_points = result.x.reshape(-1, 3)
            current_points = project_to_sphere(current_points)
    except:
        pass
    
    return current_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Strategy: Use tessellation-based evolutionary approach
    # 1. Start with an icosahedron as base structure (known good starting point)
    # 2. Evolve population of configurations using genetic operators
    # 3. Refine using adaptive optimization
    
    # Phase 1: Evolutionary search
    evolved_points = evolve_tessellation_population(pop_size=20, generations=100)
    
    # Phase 2: Adaptive refinement  
    refined_points = adaptive_optimization(evolved_points, max_iter=500)
    
    # Final validation and cleanup
    final_points = project_to_sphere(refined_points)
    
    # Ensure we have exactly 14 points
    if len(final_points) != 14:
        # Fall back to a good initialization
        base_points = generate_icosahedron()
        # Add more points by subdividing
        final_points = tetrahedral_subdivision(base_points, depth=2)
        if len(final_points) < 14:
            # Fill with random points on sphere
            np.random.seed(42)
            extra_points = np.random.randn(14 - len(final_points), 3)
            extra_points = extra_points / np.linalg.norm(extra_points, axis=1, keepdims=True)
            final_points = np.vstack([final_points, extra_points])
        elif len(final_points) > 14:
            final_points = final_points[:14]
    
    # Final projection to unit sphere
    final_points = project_to_sphere(final_points)
    
    return final_points

# EVOLVE-BLOCK-END