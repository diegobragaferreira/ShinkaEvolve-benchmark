# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import SphericalVoronoi
import time
import warnings

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    
    def calculate_min_max_ratio(points):
        """Calculate the min/max distance ratio"""
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        distances = distances[np.isfinite(distances)]
        if len(distances) == 0:
            return 0.0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max <= 0:
            return 0.0
        return d_min / d_max

    def create_voronoi_initialization(n_points):
        """Create initial points using spherical Voronoi diagram approach"""
        # Generate random points on sphere
        np.random.seed(42)
        points = np.random.randn(n_points, 3)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        try:
            # Create spherical Voronoi diagram
            sv = SphericalVoronoi(points)
            
            # Use Voronoi cell centers as new candidate points
            voronoi_centers = sv.vertices
            
            # Normalize to unit sphere and return
            if len(voronoi_centers) >= n_points:
                selected = voronoi_centers[:n_points]
            else:
                # If insufficient points, supplement with original points
                selected = np.vstack([voronoi_centers, points[:n_points-len(voronoi_centers)]])
                
            selected = selected / np.linalg.norm(selected, axis=1, keepdims=True)
            return selected
        except:
            # Fallback to fibonacci if spherical voronoi fails
            return create_fibonacci_initialization(n_points)

    def create_fibonacci_initialization(n_points):
        """Create initial points using Fibonacci spiral on sphere"""
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle

        for i in range(n_points):
            y = 1 - (i / (n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def create_symmetric_initialization():
        """Create initial configuration with known good geometric properties"""
        # Points from icosahedron-based construction
        # Vertices of regular icosahedron (normalized)
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = np.array([
            [-1,  phi,  0],
            [ 1,  phi,  0],
            [-1, -phi,  0],
            [ 1, -phi,  0],
            [ 0, -1,  phi],
            [ 0,  1,  phi],
            [ 0, -1, -phi],
            [ 0,  1, -phi],
            [ phi,  0, -1],
            [ phi,  0,  1],
            [-phi,  0, -1],
            [-phi,  0,  1]
        ])

        # Normalize vertices to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

        # For 14 points, use 12 vertices plus 2 more strategically placed
        # Add 2 points at poles for better distribution
        poles = np.array([[0, 0, 1], [0, 0, -1]])
        points = np.vstack([vertices, poles])
        
        # Scale to [0,1]^3
        points = (points + 1) / 2
        
        # Add small perturbation to escape local optima
        np.random.seed(42)
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        
        return points

    def adaptive_population_evolution(objective_func, bounds, max_time):
        """Evolutionary optimization with adaptive population sizing"""
        start_time = time.time()
        best_solution = None
        best_value = float('inf')
        
        # Multiple initial populations
        populations = []
        
        # Population 1: Voronoi-based initialization
        pop1 = create_voronoi_initialization(14)
        populations.append(("voronoi", pop1))
        
        # Population 2: Fibonacci-based
        pop2 = create_fibonacci_initialization(14)
        pop2 = (pop2 + 1) / 2
        populations.append(("fibonacci", pop2))
        
        # Population 3: Symmetric
        pop3 = create_symmetric_initialization()
        populations.append(("symmetric", pop3))
        
        # Population 4: Random
        np.random.seed(42)
        pop4 = np.random.rand(14, 3)
        populations.append(("random", pop4))
        
        # Try multiple evolutionary strategies
        for i, (name, pop) in enumerate(populations):
            if time.time() - start_time > max_time - 30:
                break
                
            try:
                # Start with larger population for better exploration
                popsize = 20 if i == 0 else 15
                
                # Run differential evolution
                result = differential_evolution(
                    objective_func,
                    bounds,
                    seed=42 + i,
                    maxiter=80,
                    popsize=popsize,
                    mutation=(0.7, 1.0),
                    recombination=0.8,
                    disp=False,
                    polish=True
                )
                
                if result.success:
                    current_value = result.fun
                    if current_value < best_value:
                        best_value = current_value
                        best_solution = result.x
                        
            except Exception as e:
                warnings.warn(f"Evolutionary optimization failed for {name}: {e}")
                continue
        
        # Return best found solution or fallback
        if best_solution is None:
            # Fallback to symmetric initialization
            return create_symmetric_initialization().flatten()
            
        return best_solution

    def geometric_local_refinement(points, max_iterations=10):
        """Apply geometric refinement using constrained optimization"""
        def objective(x):
            points_new = x.reshape(-1, 3)
            
            # Calculate distances
            distances = pdist(points_new)
            if len(distances) == 0:
                return 1e10
            distances = distances[np.isfinite(distances)]
            if len(distances) == 0:
                return 1e10
                
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            if d_max <= 0:
                return 1e10
                
            # Minimize negative ratio (maximize ratio)
            return -d_min / d_max
        
        # Use trust-constr optimization for better geometric control
        try:
            # Apply multiple refinement attempts with different tolerances
            tolerance_levels = [
                {'ftol': 1e-8, 'gtol': 1e-8},
                {'ftol': 1e-10, 'gtol': 1e-10}
            ]
            
            current_solution = points.copy()
            
            for tol_params in tolerance_levels:
                if time.time() - start_time > 345 - 10:
                    break
                    
                result = minimize(
                    objective,
                    current_solution,
                    method='trust-constr',
                    bounds=[(0, 1) for _ in range(42)],
                    options={'maxiter': 50, **tol_params}
                )
                
                if result.success:
                    current_solution = result.x
                else:
                    # Try L-BFGS-B as fallback
                    try:
                        result = minimize(
                            objective,
                            current_solution,
                            method='L-BFGS-B',
                            bounds=[(0, 1) for _ in range(42)],
                            options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-10}
                        )
                        if result.success:
                            current_solution = result.x
                    except:
                        pass
                        
            return current_solution
            
        except Exception as e:
            warnings.warn(f"Local refinement failed: {e}")
            return points.flatten()

    def create_symmetric_variants(points):
        """Create useful symmetric variants of point configuration"""
        variants = [points]
        
        # Basic transformations: reflections along axes
        transforms = [
            np.eye(3),  # Identity
            np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # Reflect x-axis
            np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # Reflect y-axis
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),  # Reflect z-axis
        ]
        
        # Apply transformations
        for transform in transforms:
            transformed = points @ transform.T
            variants.append(transformed)
            
        return variants

    def objective_with_penalties(x):
        """Objective function with boundary penalty terms"""
        points = x.reshape(-1, 3)
        
        # Apply boundary penalties
        penalty = 0.0
        for i in range(14):
            for j in range(3):
                if points[i,j] < 0:
                    penalty += 1e6 * (0 - points[i,j])**2
                elif points[i,j] > 1:
                    penalty += 1e6 * (points[i,j] - 1)**2
                    
        # Calculate main objective
        distances = pdist(points)
        if len(distances) == 0:
            return 1e10 + penalty
        distances = distances[np.isfinite(distances)]
        if len(distances) == 0:
            return 1e10 + penalty
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max <= 0:
            return 1e10 + penalty
            
        return -(d_min / d_max) + penalty

    # Main execution
    start_time = time.time()
    max_time = 345  # seconds
    
    # Phase 1: Create multiple initial configurations
    initial_configs = []
    
    # Create Voronoi-based initial points
    voronoi_points = create_voronoi_initialization(14)
    initial_configs.append(("voronoi", voronoi_points))
    
    # Create Fibonacci-based initial points
    fib_points = create_fibonacci_initialization(14)
    fib_points = (fib_points + 1) / 2
    initial_configs.append(("fibonacci", fib_points))
    
    # Create symmetric initial points
    symmetric_points = create_symmetric_initialization()
    initial_configs.append(("symmetric", symmetric_points))
    
    # Evaluate initial configurations
    best_ratio = -float('inf')
    best_points = None
    
    for name, points in initial_configs:
        ratio = calculate_min_max_ratio(points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()
    
    # Phase 2: Adaptive evolutionary optimization
    bounds = [(0, 1) for _ in range(42)]
    optimized_flat = adaptive_population_evolution(objective_with_penalties, bounds, max_time)
    
    # Extract optimized points
    optimized_points = optimized_flat.reshape(-1, 3)
    
    # Phase 3: Geometric local refinement
    refined_flat = geometric_local_refinement(optimized_points.flatten(), 10)
    refined_points = refined_flat.reshape(-1, 3)
    
    # Phase 4: Try symmetric variants for better solution
    variants = create_symmetric_variants(refined_points)
    best_variant = refined_points.copy()
    best_variant_ratio = calculate_min_max_ratio(refined_points)
    
    for variant in variants:
        ratio = calculate_min_max_ratio(variant)
        if ratio > best_variant_ratio:
            best_variant_ratio = ratio
            best_variant = variant.copy()
    
    # Final validation
    final_points = np.clip(best_variant, 0, 1)
    
    # Final check
    if calculate_min_max_ratio(final_points) <= 0:
        # Fallback to best initial configuration
        return best_points
    
    return final_points

# EVOLVE-BLOCK-END