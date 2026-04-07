# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import SphericalVoronoi
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def compute_distance_ratio(points):
        """Compute the minimum/maximum distance ratio"""
        if len(points) < 2:
            return 0.0
            
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max <= 1e-12:
            return 0.0
            
        return d_min / d_max

    def spherical_voronoi_initialization(n_points):
        """Generate points using spherical Voronoi distribution for better uniformity"""
        # Generate points using Fibonacci method with spherical Voronoi properties
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle
        
        # Generate points that naturally form good Voronoi cells
        for i in range(n_points):
            # More uniform distribution using Fibonacci spiral
            y = 1 - (i / float(n_points - 1)) * 2  # y from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
            
        return np.array(points)

    def project_to_unit_cube(points):
        """Project points to [0,1]^3 bounds"""
        return np.clip(points, 0, 1)

    def objective_function(x):
        """Objective function to maximize distance ratio"""
        points = x.reshape((14, 3))
        points = project_to_unit_cube(points)
        
        distances = pdist(points)
        if len(distances) == 0:
            return -np.inf
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max <= 1e-12:
            return -np.inf
            
        # Return negative since we're minimizing for maximization
        return -d_min / d_max

    def penalty_objective(x, penalty_weight=1e6):
        """Objective with penalty for boundary violations"""
        points = x.reshape((14, 3))
        
        # Calculate base objective
        distances = pdist(points)
        if len(distances) == 0:
            return -np.inf
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max <= 1e-12:
            return -np.inf
            
        ratio = -d_min / d_max
        
        # Add penalty for boundary violations
        penalty = 0
        for i in range(14):
            for j in range(3):
                coord = points[i, j]
                if coord < 0:
                    penalty += penalty_weight * (0 - coord) ** 2
                elif coord > 1:
                    penalty += penalty_weight * (coord - 1) ** 2
                    
        return ratio - penalty

    def adaptive_differential_evolution(bounds, maxiter=300):
        """Enhanced differential evolution with adaptive population sizing"""
        # Test multiple population sizes to find optimal one
        pop_sizes = [15, 20, 25, 30]
        best_result = None
        best_ratio = -np.inf
        
        for popsize in pop_sizes:
            try:
                result = differential_evolution(
                    penalty_objective,
                    bounds,
                    seed=42,
                    maxiter=min(maxiter // len(pop_sizes), 100),
                    popsize=popsize,
                    mutation=(0.5, 1.0),
                    recombination=0.9,
                    tol=1e-12,
                    disp=False
                )
                
                # Evaluate the result
                points = result.x.reshape((14, 3))
                ratio = compute_distance_ratio(points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_result = result
                    
            except Exception:
                continue
                
        return best_result

    def improved_local_refinement(points, max_iter=100):
        """Enhanced local refinement using multiple optimization approaches"""
        # Try L-BFGS-B first for fine-tuning
        try:
            x0 = points.flatten()
            bounds = [(0, 1)] * 42
            
            result = minimize(
                objective_function,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': max_iter},
                tol=1e-12
            )
            
            if result.success:
                refined_points = result.x.reshape((14, 3))
                refined_points = project_to_unit_cube(refined_points)
                return refined_points
                
        except Exception:
            pass
            
        # Fallback to simple coordinate-wise optimization if needed
        try:
            refined_points = points.copy()
            last_ratio = compute_distance_ratio(refined_points)
            improved = True
            
            for iteration in range(max_iter):
                if not improved:
                    break
                    
                improved = False
                current_ratio = compute_distance_ratio(refined_points)
                
                # Try small perturbations in each dimension
                for i in range(14):
                    for j in range(3):
                        # Try both directions with adaptive step size
                        step = 0.001 * (1.0 + iteration * 0.001)  # Gradually decrease step size
                        
                        # Positive perturbation
                        test_points = refined_points.copy()
                        test_points[i, j] += step
                        test_points[i, j] = np.clip(test_points[i, j], 0, 1)
                        
                        test_ratio = compute_distance_ratio(test_points)
                        if test_ratio > current_ratio:
                            refined_points = test_points.copy()
                            improved = True
                            current_ratio = test_ratio
                            
                        # Negative perturbation  
                        test_points = refined_points.copy()
                        test_points[i, j] -= step
                        test_points[i, j] = np.clip(test_points[i, j], 0, 1)
                        
                        test_ratio = compute_distance_ratio(test_points)
                        if test_ratio > current_ratio:
                            refined_points = test_points.copy()
                            improved = True
                            current_ratio = test_ratio
                
            return refined_points
            
        except Exception:
            return points

    def spherical_voronoi_refinement(initial_points):
        """Refine using spherical Voronoi-inspired approach"""
        # Start with Voronoi-based initialization
        voronoi_points = spherical_voronoi_initialization(14)
        
        # Normalize to unit sphere and project to cube
        norms = np.linalg.norm(voronoi_points, axis=1, keepdims=True)
        voronoi_points = voronoi_points / np.maximum(norms, 1e-10)
        voronoi_points = (voronoi_points + 1) / 2  # Map to [0,1]^3
        
        # Apply multiple refinement steps
        best_points = voronoi_points.copy()
        best_ratio = compute_distance_ratio(voronoi_points)
        
        # Step 1: Differential Evolution
        bounds = [(0, 1)] * 42
        de_result = adaptive_differential_evolution(bounds, maxiter=200)
        if de_result is not None:
            de_points = de_result.x.reshape((14, 3))
            de_ratio = compute_distance_ratio(de_points)
            if de_ratio > best_ratio:
                best_points = de_points.copy()
                best_ratio = de_ratio
        
        # Step 2: Local refinement
        refined_points = improved_local_refinement(best_points)
        refined_ratio = compute_distance_ratio(refined_points)
        if refined_ratio > best_ratio:
            best_points = refined_points.copy()
            best_ratio = refined_ratio
        
        # Step 3: Multiple restarts with different initializations
        for restart in range(3):
            np.random.seed(42 + restart)
            # Random perturbation of best solution
            perturbed = best_points + np.random.normal(0, 0.01, best_points.shape)
            perturbed = np.clip(perturbed, 0, 1)
            
            # Refine again
            restart_points = improved_local_refinement(perturbed)
            restart_ratio = compute_distance_ratio(restart_points)
            if restart_ratio > best_ratio:
                best_points = restart_points.copy()
                best_ratio = restart_ratio
        
        return best_points

    # Main optimization loop
    best_points = None
    best_ratio = -np.inf
    
    # Try multiple spherical Voronoi approaches with different seeds
    for attempt in range(5):
        np.random.seed(42 + attempt * 10)
        
        # Generate initial points using spherical Voronoi approach
        try:
            initial_points = spherical_voronoi_initialization(14)
            
            # Normalize and project
            norms = np.linalg.norm(initial_points, axis=1, keepdims=True)
            initial_points = initial_points / np.maximum(norms, 1e-10)
            initial_points = (initial_points + 1) / 2  # To [0,1]^3
            
            # Add some random perturbations to break symmetries
            initial_points += np.random.normal(0, 0.02, initial_points.shape)
            initial_points = np.clip(initial_points, 0, 1)
            
            # Refine using our specialized approach
            refined_points = spherical_voronoi_refinement(initial_points)
            ratio = compute_distance_ratio(refined_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined_points.copy()
                
        except Exception as e:
            continue
    
    # Final check with standard approach if needed
    if best_points is None:
        # Fallback to basic initialization and optimization
        np.random.seed(42)
        initial_points = np.random.rand(14, 3)
        bounds = [(0, 1)] * 42
        
        # Apply differential evolution
        de_result = adaptive_differential_evolution(bounds, maxiter=300)
        if de_result is not None:
            final_points = de_result.x.reshape((14, 3))
            final_points = improved_local_refinement(final_points)
            best_points = final_points
    
    # Ensure we return a valid result
    if best_points is None:
        np.random.seed(42)
        best_points = np.random.rand(14, 3)
    
    return best_points

# EVOLVE-BLOCK-END