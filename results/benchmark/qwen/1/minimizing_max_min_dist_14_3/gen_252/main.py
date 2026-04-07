# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy.spatial import SphericalVoronoi
import warnings

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a sphere tiling inspired evolutionary approach with geometric constraint preservation.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    np.random.seed(42)
    n = 14
    
    # Phase 1: Generate initial configuration using spherical tiling principles
    def generate_tiling_initial():
        """Generate initial configuration based on geometric tiling principles"""
        # Start with icosahedral symmetry points (12 vertices)
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ])
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        # Add 2 more points to make 14 total - place them at poles
        points = vertices.tolist()
        points.append([0, 0, 1])      # north pole
        points.append([0, 0, -1])     # south pole
        
        # Apply controlled perturbation to break symmetries
        points = np.array(points)
        noise = np.random.normal(0, 0.01, points.shape)
        points += noise
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        return points
    
    # Phase 2: Specialized spherical rotation operator
    def spherical_rotation(point, axis, angle):
        """Rotate a point on the unit sphere using Rodrigues' rotation formula"""
        axis = axis / np.linalg.norm(axis)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return point * cos_a + np.cross(axis, point) * sin_a + axis * np.dot(axis, point) * (1 - cos_a)
    
    # Phase 3: Geometrically informed evolutionary operators
    def calculate_ratio(points):
        """Calculate min/max distance ratio with numerical stability"""
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist <= 1e-12:
            return 0
        return min_dist / max_dist
    
    def project_to_sphere(points):
        """Project points to unit sphere with numerical stability"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Handle zero vectors safely
        norms = np.where(norms == 0, 1.0, norms)
        return points / norms
    
    def spherical_evolution_step(current_points, population_size=25, current_iteration=0):
        """
        Evolutionary step using spherical geometry-aware operators
        """
        best_points = current_points.copy()
        best_ratio = calculate_ratio(best_points)
        
        # Determine adaptive mutation rate based on iteration
        base_mutation_rate = max(0.05, 0.2 * np.exp(-current_iteration * 0.02))
        
        # Create population of candidates
        population = [best_points]
        
        for _ in range(population_size - 1):
            # Create new candidate through geometric operations
            mutated = best_points.copy()
            
            # Select points for modification
            num_modify = max(1, int(len(mutated) * base_mutation_rate))
            indices = np.random.choice(len(mutated), size=num_modify, replace=False)
            
            # Apply spherical-aware mutations
            for idx in indices:
                # Generate rotation axis (perpendicular to current point)
                axis = np.random.randn(3)
                # Ensure axis is perpendicular to point vector
                axis = axis - np.dot(axis, mutated[idx]) * mutated[idx]
                # Normalize the axis
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-10:
                    axis = axis / axis_norm
                else:
                    # If axis is parallel to point, choose arbitrary perpendicular
                    axis = np.cross(mutated[idx], np.array([1, 0, 0]))
                    if np.linalg.norm(axis) < 1e-10:
                        axis = np.cross(mutated[idx], np.array([0, 1, 0]))
                    axis = axis / np.linalg.norm(axis)
                
                # Apply rotation with adaptive angle
                angle = np.random.uniform(-0.3, 0.3) * (1.0 - current_iteration * 0.005)
                mutated[idx] = spherical_rotation(mutated[idx], axis, angle)
            
            # Project back to sphere
            mutated = project_to_sphere(mutated)
            
            population.append(mutated)
        
        # Evaluate fitness of all candidates
        ratios = [calculate_ratio(p) for p in population]
        
        # Selection based on geometric diversity and ratio
        best_idx = np.argmax(ratios)
        
        return population[best_idx], ratios[best_idx]
    
    # Phase 4: Multi-stage optimization
    # Initialize with tiling-based configuration
    points = generate_tiling_initial()
    
    # Stage 1: Global evolutionary search
    current_points = points.copy()
    best_ratio = calculate_ratio(current_points)
    best_solution = current_points.copy()
    
    # Evolutionary optimization loop
    for iteration in range(150):
        new_points, new_ratio = spherical_evolution_step(current_points, current_iteration=iteration)
        
        # Accept better solutions
        if new_ratio > best_ratio:
            best_ratio = new_ratio
            best_solution = new_points.copy()
        
        current_points = new_points
    
    # Stage 2: Local refinement with specialized constraint handling
    def objective(x):
        points = x.reshape(-1, 3)
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist <= 1e-12:
            return 0
        return -min_dist / max_dist
    
    def constraint_func(x):
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0
    
    # Final local optimization with multiple strategies
    cons = {'type': 'eq', 'fun': constraint_func}
    
    try:
        # Try multiple local optimization approaches with different tolerances
        x0 = best_solution.flatten()
        
        # Strategy 1: L-BFGS-B with strict tolerances
        result1 = minimize(objective, x0, method='L-BFGS-B', constraints=cons, 
                          options={'ftol': 1e-14, 'gtol': 1e-14, 'maxiter': 500})
        
        if result1.success:
            optimized_points = result1.x.reshape(-1, 3)
            optimized_points = project_to_sphere(optimized_points)
            final_ratio = calculate_ratio(optimized_points)
            
            if final_ratio > best_ratio:
                best_solution = optimized_points
        
        # Strategy 2: SLSQP as backup
        if not result1.success:
            result2 = minimize(objective, x0, method='SLSQP', constraints=cons, 
                              options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 400})
            
            if result2.success:
                optimized_points = result2.x.reshape(-1, 3)
                optimized_points = project_to_sphere(optimized_points)
                final_ratio = calculate_ratio(optimized_points)
                
                if final_ratio > best_ratio:
                    best_solution = optimized_points
                    
    except Exception as e:
        warnings.warn(f"Local optimization failed: {e}")
        pass
    
    # Final validation and cleanup
    try:
        # Ensure final points are on sphere
        best_solution = project_to_sphere(best_solution)
        # Verify ratio calculation
        final_ratio = calculate_ratio(best_solution)
        
        # In case of degenerate solution, fall back to initial configuration
        if final_ratio <= 0:
            best_solution = generate_tiling_initial()
            
    except Exception:
        # Last resort fallback
        best_solution = generate_tiling_initial()
    
    return best_solution

# EVOLVE-BLOCK-END