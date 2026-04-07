# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi
import time
import random
from scipy.spatial.transform import Rotation as R

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)
    random.seed(42)

    n = 14
    d = 3

    # Generate initial points using a known good spherical configuration
    def generate_initial_points():
        # Use a spherical code approach - start with a configuration that has good symmetry
        # We'll use a modified icosahedral arrangement with perturbations
        
        # Create vertices of an icosahedron and then add 4 more points
        # Icosahedron vertices scaled to unit sphere
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = []
        
        # Generate 12 vertices of icosahedron
        for i in range(12):
            if i < 4:
                x = (-1)**i * 1
                y = 0
                z = 0
            elif i < 8:
                x = 0
                y = (-1)**(i-4) * 1
                z = 0
            else:
                x = 0
                y = 0
                z = (-1)**(i-8) * 1
            
            # Adjust for icosahedron symmetry
            if i == 0:
                x, y, z = 0, -1, 0
            elif i == 1:
                x, y, z = 0, 1, 0
            elif i == 2:
                x, y, z = 0, 0, -1
            elif i == 3:
                x, y, z = 0, 0, 1
            elif i == 4:
                x, y, z = 1, 0, 0
            elif i == 5:
                x, y, z = -1, 0, 0
            elif i == 6:
                x, y, z = 0, 0, 0  # placeholder
            elif i == 7:
                x, y, z = 0, 0, 0  # placeholder
            elif i == 8:
                x, y, z = 0, 0, 0  # placeholder
            elif i == 9:
                x, y, z = 0, 0, 0  # placeholder
            elif i == 10:
                x, y, z = 0, 0, 0  # placeholder
            elif i == 11:
                x, y, z = 0, 0, 0  # placeholder
            
            # Normalize to unit sphere
            norm = np.sqrt(x*x + y*y + z*z)
            if norm > 0:
                x, y, z = x/norm, y/norm, z/norm
            vertices.append([x, y, z])
        
        # Replace placeholders with actual vertices
        vertices = [[0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1], [1, 0, 0], [-1, 0, 0]]
        
        # Add some additional points for total of 14
        # Use random points on sphere but biased towards the icosahedral symmetry
        additional_points = []
        for i in range(8):
            # Create points using spherical coordinates with some structure
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.arccos(2*np.random.random() - 1)
            
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            additional_points.append([x, y, z])
            
        # Combine and return
        points = vertices + additional_points
        return np.array(points[:14])

    # Alternative initialization: Fibonacci-like spherical distribution
    def fibonacci_sphere(samples=14):
        points = []
        phi = np.pi * (3. - np.sqrt(5.))
        
        for i in range(samples):
            # Distribute points more evenly
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            # Use golden angle for even angular distribution
            theta = phi * i
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)

    # Even better: use spherical codes from literature
    def spherical_code_initialization():
        # A known good configuration for 14 points on sphere
        # These are approximations based on research on optimal point distributions
        points = np.array([
            [0.000000, 0.000000, 1.000000],
            [0.000000, 0.000000, -1.000000],
            [0.955573, 0.000000, 0.294755],
            [-0.955573, 0.000000, 0.294755],
            [0.000000, 0.955573, 0.294755],
            [0.000000, -0.955573, 0.294755],
            [0.000000, 0.955573, -0.294755],
            [0.000000, -0.955573, -0.294755],
            [0.955573, 0.000000, -0.294755],
            [-0.955573, 0.000000, -0.294755],
            [0.707107, 0.707107, 0.000000],
            [0.707107, -0.707107, 0.000000],
            [-0.707107, 0.707107, 0.000000],
            [-0.707107, -0.707107, 0.000000]
        ])
        return points

    # Constraint-preserving rotation optimization
    def rotate_points(points, angles):
        """Apply rotation to points"""
        rot = R.from_euler('xyz', angles)
        return rot.apply(points)
    
    # Objective function that works with spherical constraint
    def objective_with_constraints(x_flat):
        points = x_flat.reshape((n, d))
        
        # Normalize to sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized_points = points / norms
        
        # Calculate distances
        distances = pdist(normalized_points)
        distances = distances[distances > 1e-12]
        
        if len(distances) == 0:
            return np.inf
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max == 0:
            return np.inf
            
        # Return negative ratio (minimize)
        return -d_min / d_max

    # Multi-start optimization with spherical Voronoi structure preservation
    best_ratio = -np.inf
    best_points = None

    # Different initialization strategies
    init_strategies = [
        spherical_code_initialization,
        lambda s: fibonacci_sphere(s),
        lambda s: generate_initial_points(),
        lambda s: np.random.rand(s, 3) * 2 - 1  # Random in [-1,1]^3
    ]

    num_starts = 30
    base_perturbation = 0.05
    perturbation_decay = 0.95
    min_perturbation = 0.001

    start_time = time.time()
    
    for start_idx in range(num_starts):
        if time.time() - start_time > 350:  # Time limit
            break
            
        # Adaptive perturbation
        current_perturbation = max(base_perturbation * (perturbation_decay ** start_idx),
                                 min_perturbation)
        
        # Select initialization strategy
        init_func = init_strategies[start_idx % len(init_strategies)]
        
        # Get initial points
        initial_points = init_func(n)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(initial_points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        initial_points = initial_points / norms
        
        # Add small perturbation for diversity
        if start_idx > 0:
            perturbation = np.random.normal(0, current_perturbation, initial_points.shape)
            initial_points += perturbation
            # Re-normalize to maintain sphere constraint
            norms = np.linalg.norm(initial_points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            initial_points = initial_points / norms

        # Flatten for optimization
        initial_flat = initial_points.flatten()

        # Optimization with constraints to preserve sphere structure
        def constraint_func(x_flat):
            points = x_flat.reshape((n, d))
            norms = np.linalg.norm(points, axis=1)
            return norms - 1.0  # Should equal zero for points on unit sphere

        # Constraints for optimization
        constraints = {'type': 'eq', 'fun': constraint_func}

        # Bounds for optimization (not really used due to constraints, but included for safety)
        bounds = [(-2, 2) for _ in range(n * d)]

        # Set optimization options
        options = {'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}

        # Try different optimization approaches based on success
        try:
            # First try L-BFGS-B with constraints
            result = minimize(
                objective_with_constraints,
                initial_flat,
                method='L-BFGS-B',
                bounds=bounds,
                constraints=constraints,
                options=options
            )
            
            # If L-BFGS-B doesn't work, try trust-constr
            if not result.success:
                result = minimize(
                    objective_with_constraints,
                    initial_flat,
                    method='trust-constr',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 800, 'gtol': 1e-12}
                )
                
            # If still failing, try SLSQP
            if not result.success:
                result = minimize(
                    objective_with_constraints,
                    initial_flat,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-12}
                )

        except Exception:
            continue

        # Extract and validate solution
        try:
            optimized_points = result.x.reshape((n, d))
            
            # Ensure points are on sphere
            norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            optimized_points = optimized_points / norms
            
            # Calculate actual ratio
            distances = pdist(optimized_points)
            distances = distances[distances > 1e-12]
            
            if len(distances) > 0:
                d_min = np.min(distances)
                d_max = np.max(distances)
                
                if d_max > 0:
                    ratio = d_min / d_max
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
            
        except Exception:
            continue

    # Final refinement using a hybrid approach
    # If no good solution was found, use default initialization
    if best_points is None:
        best_points = spherical_code_initialization()
    
    # Perform final fine-tuning
    final_points = best_points.copy()
    
    # Try to improve using direct optimization without constraints (for fine tuning)
    def simple_objective(x_flat):
        points = x_flat.reshape((n, d))
        # Simple distance-based objective
        distances = pdist(points)
        distances = distances[distances > 1e-12]
        
        if len(distances) == 0:
            return np.inf
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max == 0:
            return np.inf
            
        return -d_min / d_max

    # Fine-tune using different methods
    try:
        # Method 1: L-BFGS-B without constraints
        final_flat = final_points.flatten()
        result = minimize(
            simple_objective,
            final_flat,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14}
        )
        
        if result.success:
            candidate_points = result.x.reshape((n, d))
            # Normalize to sphere again
            norms = np.linalg.norm(candidate_points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            candidate_points = candidate_points / norms
            
            # Validate and update if better
            distances = pdist(candidate_points)
            distances = distances[distances > 1e-12]
            
            if len(distances) > 0:
                d_min = np.min(distances)
                d_max = np.max(distances)
                if d_max > 0:
                    ratio = d_min / d_max
                    if ratio > best_ratio:
                        final_points = candidate_points
                        
    except Exception:
        pass
    
    # Return the final result
    return final_points

# EVOLVE-BLOCK-END