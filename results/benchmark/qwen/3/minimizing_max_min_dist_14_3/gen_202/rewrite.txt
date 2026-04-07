# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import SphericalVoronoi
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """

    np.random.seed(42)

    def compute_min_max_ratio(points):
        """Compute the min/max distance ratio for given points."""
        if len(points) < 2:
            return 0.0

        # Compute pairwise distances efficiently
        distances = pdist(points)

        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Avoid division by zero
        if d_max == 0:
            return 0.0

        return d_min / d_max

    def fibonacci_sphere(n):
        """Generate points on sphere using Fibonacci spiral."""
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))

        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = golden_angle * i  # Golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def spherical_constraint(points):
        """Normalize points to lie on the unit sphere."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return points / norms

    def objective_function(points_flat):
        """Objective function to maximize - negative of min/max ratio."""
        # Reshape flat array to 2D points array
        points = points_flat.reshape(-1, 3)

        # Apply spherical constraint to keep points on unit sphere
        points = spherical_constraint(points)

        # Compute ratio
        ratio = compute_min_max_ratio(points)

        # Return negative because we want to maximize ratio, but optimizers minimize
        return -ratio

    def generate_diverse_initializations():
        """Generate multiple diverse initial point sets."""
        initial_sets = []

        # Strategy 1: Fibonacci sphere distribution with small perturbation
        fib_points = fibonacci_sphere(14)
        perturbed = fib_points + np.random.normal(0, 0.03, fib_points.shape)
        initial_sets.append(spherical_constraint(perturbed))

        # Strategy 2: Random points on sphere with structured distribution
        random_points = np.random.randn(14, 3)
        initial_sets.append(spherical_constraint(random_points))

        # Strategy 3: Axis-based structured distribution
        struct_points = np.zeros((14, 3))
        for i in range(14):
            if i < 3:
                # Along positive axes
                struct_points[i] = [1 if j==i else 0 for j in range(3)]
            elif i < 6:
                # Along negative axes
                struct_points[i] = [-1 if j==i-3 else 0 for j in range(3)]
            elif i < 9:
                # Diagonal combinations
                j = i - 6
                struct_points[i] = [1 if k==j else -1 if k==(j+1)%3 else 0 for k in range(3)]
            else:
                # Random points on sphere
                struct_points[i] = np.random.randn(3)
        initial_sets.append(spherical_constraint(struct_points))

        # Strategy 4: Perturbed Fibonacci with larger variance
        fib_perturbed = fib_points + np.random.normal(0, 0.08, fib_points.shape)
        initial_sets.append(spherical_constraint(fib_perturbed))

        # Strategy 5: Spherical Voronoi distribution (more evenly spread)
        voronoi_points = np.random.randn(14, 3)
        voronoi_points = spherical_constraint(voronoi_points)
        initial_sets.append(voronoi_points)

        # Strategy 6: Multiple Fibonacci configurations with different scalings
        for scale in [0.7, 0.85, 1.15, 1.3]:
            scaled_fib = fib_points * scale
            scaled_perturbed = scaled_fib + np.random.normal(0, 0.04, scaled_fib.shape)
            initial_sets.append(spherical_constraint(scaled_perturbed))

        return initial_sets

    def adaptive_differential_evolution(initial_points, maxiter=50):
        """Enhanced differential evolution with adaptive population sizing."""
        points = initial_points.copy()
        
        # Track convergence for adaptive population sizing
        history = []
        stagnation_count = 0
        max_stagnation = 10
        min_popsize = 10
        max_popsize = 30
        popsize = 15  # Reduced population size with adaptive scaling
        
        bounds = [(-1, 1)] * (14 * 3)
        
        def adaptive_objective(x_flat):
            points = x_flat.reshape(-1, 3)
            points = spherical_constraint(points)
            ratio = compute_min_max_ratio(points)
            return -ratio

        try:
            # Run multiple rounds of DE with adaptive population sizing
            for iteration in range(3):
                # Adjust population size based on convergence history
                current_popsize = popsize
                
                # If we've seen improvement recently, reduce population size for faster convergence
                if len(history) >= 2:
                    improvement = -history[-1] - (-history[-2])
                    if improvement < 1e-6:
                        stagnation_count += 1
                        if stagnation_count >= 3 and current_popsize < max_popsize:
                            # Increase population size when stagnating to boost exploration
                            current_popsize = min(current_popsize + 5, max_popsize)
                    else:
                        stagnation_count = 0
                        if current_popsize > min_popsize and iteration > 0:
                            # Decrease population size when making progress to speed up convergence
                            current_popsize = max(current_popsize - 3, min_popsize)
                
                # Run differential evolution with current parameters
                result = differential_evolution(
                    adaptive_objective,
                    bounds,
                    maxiter=maxiter//3,
                    popsize=current_popsize,
                    seed=42 + iteration,
                    disp=False,
                    polish=True,
                    strategy='best1bin'
                )

                if result.success:
                    temp_points = result.x.reshape(-1, 3)
                    temp_points = spherical_constraint(temp_points)
                    current_ratio = compute_min_max_ratio(temp_points)
                    history.append(current_ratio)
                    
                    # Update best solution
                    if len(history) == 1 or current_ratio > history[-2]:
                        points = temp_points.copy()
                        
                    # Check for stagnation and early stopping
                    if len(history) >= 2:
                        improvement = current_ratio - history[-2]
                        if improvement < 1e-8:
                            stagnation_count += 1
                            if stagnation_count >= max_stagnation:
                                break
                        else:
                            stagnation_count = 0
                else:
                    # If optimization fails, continue with current points
                    break
                    
        except Exception:
            pass
            
        return points

    def enhanced_hill_climbing(initial_points, maxiter=200):
        """Enhanced hill climbing with adaptive step sizes and improved neighborhood search."""
        points = initial_points.copy()
        last_ratio = compute_min_max_ratio(points)
        patience = 0
        max_patience = 15
        step_size = 0.01
        
        # Track improvement history for adaptive behaviors
        improvement_history = []
        max_improvement_history = 5
        
        for iteration in range(maxiter):
            current_ratio = compute_min_max_ratio(points)
            best_points = points.copy()
            best_ratio = current_ratio
            
            # Adaptive step size based on recent improvement trends
            if len(improvement_history) >= 2:
                avg_improvement = np.mean(improvement_history[-min(len(improvement_history), max_improvement_history):])
                if avg_improvement > 1e-6:
                    step_size = min(0.02, step_size * 1.1)  # Increase step size if improving consistently
                elif avg_improvement < 1e-8:
                    step_size = max(0.0001, step_size * 0.9)  # Decrease step size if improving slowly
            
            # Track improvement for adaptive behavior
            improvement = current_ratio - last_ratio
            improvement_history.append(improvement)
            if len(improvement_history) > max_improvement_history:
                improvement_history.pop(0)
            
            # Try perturbations with adaptive step sizes
            for i in range(14):
                for dim in range(3):
                    # Try moving in positive direction
                    test_points = points.copy()
                    test_points[i, dim] += step_size
                    test_points = spherical_constraint(test_points)
                    test_ratio = compute_min_max_ratio(test_points)
                    
                    if test_ratio > best_ratio:
                        best_ratio = test_ratio
                        best_points = test_points.copy()
                    
                    # Try moving in negative direction
                    test_points = points.copy()
                    test_points[i, dim] -= step_size
                    test_points = spherical_constraint(test_points)
                    test_ratio = compute_min_max_ratio(test_points)
                    
                    if test_ratio > best_ratio:
                        best_ratio = test_ratio
                        best_points = test_points.copy()
            
            # Early stopping condition
            if best_ratio <= current_ratio:
                patience += 1
                if patience > max_patience:
                    break
            else:
                patience = 0
                points = best_points
                
            last_ratio = current_ratio

        return points

    def hybrid_optimization(initial_points, maxiter=100):
        """Perform hybrid optimization combining global and local methods."""
        points = initial_points.copy()
        
        # Stage 1: Adaptive global optimization with differential evolution
        points = adaptive_differential_evolution(points, maxiter=maxiter//3)
        
        # Stage 2: Local refinement with L-BFGS-B
        def local_obj(x_flat):
            points = x_flat.reshape(-1, 3)
            points = spherical_constraint(points)
            ratio = compute_min_max_ratio(points)
            return -ratio  # Negative for minimization

        try:
            x0 = points.flatten()
            bounds = [(-1, 1)] * (14 * 3)
            result = minimize(
                local_obj,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 60, 'ftol': 1e-10, 'gtol': 1e-10, 'disp': False}
            )
            if result.success:
                points = result.x.reshape(-1, 3)
                points = spherical_constraint(points)
        except:
            pass
            
        # Stage 3: Enhanced hill climbing for additional refinement
        points = enhanced_hill_climbing(points, maxiter=maxiter//3)
        
        return points

    # Multi-start optimization with diverse initializations
    best_solution = None
    best_ratio = 0.0

    # Generate multiple initial sets
    initial_sets = generate_diverse_initializations()

    # Try each initialization with enhanced optimization
    for i, initial_points in enumerate(initial_sets):
        # Perform hybrid optimization
        optimized_points = hybrid_optimization(initial_points, maxiter=60)
        ratio = compute_min_max_ratio(optimized_points)

        if ratio > best_ratio:
            best_ratio = ratio
            best_solution = optimized_points.copy()

    # Additional refinement with multiple restarts
    if best_solution is not None:
        # Try more focused optimization from the best solution
        refined_points = hybrid_optimization(best_solution, maxiter=40)
        ratio = compute_min_max_ratio(refined_points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_solution = refined_points.copy()

    # If nothing worked, return the best initialization
    if best_solution is None:
        # Fallback to Fibonacci with small perturbation
        fib_points = fibonacci_sphere(14)
        fib_points = fib_points + np.random.normal(0, 0.05, fib_points.shape)
        best_solution = spherical_constraint(fib_points)

    return best_solution

# EVOLVE-BLOCK-END