# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import itertools

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a sphere packing inspired evolutionary approach for optimal point distribution.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    np.random.seed(42)
    n = 14
    
    # Phase 1: Generate initial symmetric configuration based on icosahedral symmetry
    def generate_icosahedral_initial():
        # Vertices of regular icosahedron scaled to unit sphere
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ])
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        # Add additional points by distributing them uniformly
        # Start with icosahedron vertices
        points = vertices.tolist()
        
        # Add 2 more points to make 14 total
        # Place them at poles for initial symmetric structure
        points.append([0, 0, 1])      # north pole
        points.append([0, 0, -1])     # south pole
        
        # Use rotationally symmetric initialization
        return np.array(points)
    
    # Phase 2: Evolutionary optimization on sphere
    def calculate_ratio(points):
        """Calculate min/max distance ratio"""
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return min_dist / max_dist
    
    def project_to_sphere(points):
        """Project points to unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1
        return points / norms
    
    def rotate_point(point, axis, angle):
        """Rotate a point around an axis by angle ( Rodrigues' rotation formula )"""
        axis = axis / np.linalg.norm(axis)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return point * cos_a + np.cross(axis, point) * sin_a + axis * np.dot(axis, point) * (1 - cos_a)
    
    def evolutionary_step(current_points, population_size=20, mutation_rate=0.1):
        """Perform evolutionary step on spherical points"""
        best_points = current_points.copy()
        best_ratio = calculate_ratio(best_points)
        
        # Generate population of mutated solutions
        population = [best_points]
        
        for _ in range(population_size - 1):
            # Create mutated version
            mutated = best_points.copy()
            
            # Select random points to mutate
            indices = np.random.choice(n, size=max(1, int(n * mutation_rate)), replace=False)
            
            # Apply rotation to selected points
            for idx in indices:
                # Generate random rotation axis and angle
                axis = np.random.randn(3)
                angle = np.random.uniform(-0.5, 0.5)  # Small rotation
                
                # Apply rotation
                mutated[idx] = rotate_point(mutated[idx], axis, angle)
            
            # Project back to sphere
            mutated = project_to_sphere(mutated)
            
            population.append(mutated)
        
        # Evaluate population and select best
        ratios = [calculate_ratio(p) for p in population]
        best_idx = np.argmax(ratios)
        
        return population[best_idx]
    
    # Phase 3: Initialize and optimize
    points = generate_icosahedral_initial()
    points = project_to_sphere(points)
    
    # Multiple evolutionary steps
    current_points = points.copy()
    best_ratio = calculate_ratio(current_points)
    best_solution = current_points.copy()
    
    # Evolutionary optimization loop
    for iteration in range(100):
        # Perform evolutionary step
        new_points = evolutionary_step(current_points)
        new_ratio = calculate_ratio(new_points)
        
        # Accept better solutions
        if new_ratio > best_ratio:
            best_ratio = new_ratio
            best_solution = new_points.copy()
        
        current_points = new_points
    
    # Phase 4: Local refinement with constrained optimization
    def objective(x):
        points = x.reshape(-1, 3)
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return -min_dist / max_dist
    
    def constraint_func(x):
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0
    
    # Final refinement using optimization
    cons = {'type': 'eq', 'fun': constraint_func}
    
    try:
        # Try multiple local optimization approaches
        x0 = best_solution.flatten()
        
        # Try L-BFGS-B
        result = minimize(objective, x0, method='L-BFGS-B', constraints=cons, 
                         options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 300})
        
        if result.success:
            optimized_points = result.x.reshape(-1, 3)
            optimized_points = project_to_sphere(optimized_points)
            final_ratio = calculate_ratio(optimized_points)
            
            if final_ratio > best_ratio:
                best_solution = optimized_points
                
    except Exception:
        pass
    
    return best_solution

# EVOLVE-BLOCK-END