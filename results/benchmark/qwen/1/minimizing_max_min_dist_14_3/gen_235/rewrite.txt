# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from scipy.stats import qmc
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def objective_ratio(x):
        """Objective function to maximize min/max distance ratio"""
        points = x.reshape(-1, 3)
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0.0
        return min_dist / max_dist

    def objective_voronoi_regularized(x):
        """Regularized objective using Voronoi-inspired principles"""
        points = x.reshape(-1, 3)
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        mean_dist = np.mean(distances)
        var_dist = np.var(distances)
        
        if max_dist == 0:
            return 0.0
            
        # Regularized objective: maximize ratio + penalize distance variance for better distribution
        ratio = min_dist / max_dist
        # Penalize large variance in distances to encourage uniform distribution
        variance_penalty = 0.1 * (var_dist / (mean_dist**2 + 1e-12))
        return -(ratio - variance_penalty)  # Negative because we minimize

    def constraint_bounds(x):
        # Ensure all points are within [0,1]^3 bounds
        points = x.reshape(-1, 3)
        return np.concatenate([
            points.flatten() - 0.0,      # lower bound
            1.0 - points.flatten()       # upper bound
        ])

    def generate_sobol_points(n):
        """Generate points using 3D Sobol sequence for better space-filling"""
        sampler = qmc.Sobol(d=3, scramble=True, seed=42)
        points = sampler.random(n)
        # Map to unit sphere by normalizing
        points = points * 2 - 1  # map to [-1,1]^3
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        points = points / norms
        return points

    def generate_fibonacci_points(n):
        """Generate points on sphere using Fibonacci spiral method"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            theta = np.arccos(1 - 2*(i/(n-1)))
            phi = i * 2 * np.pi / golden_ratio
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            points.append([x, y, z])
        return np.array(points)

    def generate_voronoi_like_initial(n):
        """Generate initial points with Voronoi-like spacing properties"""
        # Start with Sobol points for good distribution
        points = generate_sobol_points(n)
        
        # Add some Voronoi-inspired perturbations
        # Distribute points more evenly by slightly repelling neighbors
        for _ in range(20):  # Iterative relaxation
            distances = cdist(points, points)
            np.fill_diagonal(distances, np.inf)
            
            # Calculate repulsion forces (inverse of distances)
            forces = np.zeros_like(points)
            for i in range(n):
                diff = points[i] - points
                dists = distances[i]
                mask = (dists > 1e-8) & (dists < 10.0)  # Avoid very close or very far
                if np.any(mask):
                    # Inverse force proportional to distance squared
                    inv_dists = 1.0 / (dists[mask]**2 + 1e-12)
                    forces[i] = np.sum(diff[mask] * inv_dists[:, np.newaxis], axis=0)
            
            # Apply forces with damping
            points += 0.01 * forces
            # Normalize to sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            points = points / norms
        
        return points

    def voronoi_relaxation(points, iterations=30):
        """Apply iterative Voronoi relaxation to improve point distribution"""
        for _ in range(iterations):
            # Compute pairwise distances
            distances = cdist(points, points)
            np.fill_diagonal(distances, np.inf)
            
            # Compute repulsion forces
            forces = np.zeros_like(points)
            for i in range(len(points)):
                diff = points[i] - points
                dists = distances[i]
                # Only consider nearby points for efficiency
                mask = dists < 3.0  # Some reasonable threshold
                if np.any(mask):
                    # Repulsion force inversely proportional to distance squared
                    inv_dists = 1.0 / (dists[mask]**2 + 1e-12)
                    forces[i] = np.sum(diff[mask] * inv_dists[:, np.newaxis], axis=0)
            
            # Apply force with damping
            points += 0.02 * forces
            
            # Normalize to sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            points = points / norms
            
        return points

    # Generate initial diverse configurations
    initial_configs = []
    
    # Configuration 1: Sobol sequence points
    initial_configs.append(generate_sobol_points(14))
    
    # Configuration 2: Fibonacci points
    initial_configs.append(generate_fibonacci_points(14))
    
    # Configuration 3: Voronoi-like initial points
    initial_configs.append(generate_voronoi_like_initial(14))
    
    # Configuration 4: Random points on sphere
    np.random.seed(100)
    random_points = np.random.randn(14, 3)
    norms = np.linalg.norm(random_points, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    initial_configs.append(random_points / norms)
    
    # Configuration 5: Perturbed Fibonacci
    np.random.seed(200)
    fib_points = generate_fibonacci_points(14)
    noise = np.random.normal(0, 0.03, fib_points.shape)
    perturbed_fib = fib_points + noise
    norms = np.linalg.norm(perturbed_fib, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    initial_configs.append(perturbed_fib / norms)
    
    # Configuration 6: Another Voronoi-like with different perturbation
    np.random.seed(300)
    initial_configs.append(generate_voronoi_like_initial(14) + np.random.normal(0, 0.02, (14, 3)))

    # Multi-stage optimization approach
    best_final_points = None
    best_ratio = 0
    
    # Stage 1: Coarse global optimization with regularized objective
    print("Stage 1: Coarse global optimization")
    for i, initial_config in enumerate(initial_configs):
        try:
            x0 = initial_config.flatten()
            bounds = [(0, 1) for _ in range(14 * 3)]
            
            # Use differential evolution with regularized objective for broad search
            result = differential_evolution(
                objective_voronoi_regularized,
                bounds,
                seed=42 + i,
                maxiter=500,
                popsize=20,
                tol=1e-7,
                mutation=(0.5, 1),
                recombination=0.7,
                disp=False
            )
            
            # Evaluate this solution
            ratio = objective_ratio(result.x)
            if ratio > best_ratio:
                best_ratio = ratio
                best_final_points = result.x.reshape(-1, 3).copy()
                
        except Exception as e:
            continue
    
    # Stage 2: Local refinement with Voronoi relaxation
    print("Stage 2: Local refinement with Voronoi relaxation")
    if best_final_points is not None:
        # Apply Voronoi relaxation to the best solution
        relaxed_points = voronoi_relaxation(best_final_points.copy(), iterations=50)
        
        # Evaluate and update if better
        ratio = objective_ratio(relaxed_points.flatten())
        if ratio > best_ratio:
            best_ratio = ratio
            best_final_points = relaxed_points.copy()
    
    # Stage 3: Fine-grained local optimization
    print("Stage 3: Fine-grained local optimization")
    if best_final_points is not None:
        try:
            x0 = best_final_points.flatten()
            cons = [{'type': 'ineq', 'fun': constraint_bounds}]
            
            # First stage: SLSQP with moderate tolerance
            result1 = minimize(
                objective_voronoi_regularized,
                x0,
                method='SLSQP',
                constraints=cons,
                options={'ftol': 1e-10, 'maxiter': 300}
            )
            
            if result1.success:
                refined_points = result1.x.reshape(-1, 3)
                ratio = objective_ratio(refined_points.flatten())
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_final_points = refined_points.copy()
            
            # Second stage: L-BFGS-B with tighter tolerance  
            result2 = minimize(
                objective_voronoi_regularized,
                best_final_points.flatten(),
                method='L-BFGS-B',
                constraints=cons,
                options={'ftol': 1e-12, 'maxiter': 500}
            )
            
            if result2.success:
                final_points = result2.x.reshape(-1, 3)
                ratio = objective_ratio(final_points.flatten())
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_final_points = final_points.copy()
                    
        except Exception as e:
            pass
    
    # Stage 4: Final Voronoi relaxation with improved points
    print("Stage 4: Final Voronoi relaxation")
    if best_final_points is not None:
        final_points = voronoi_relaxation(best_final_points.copy(), iterations=30)
        ratio = objective_ratio(final_points.flatten())
        if ratio > best_ratio:
            best_ratio = ratio
            best_final_points = final_points.copy()
    
    # If still no solution, return best of initial configurations
    if best_final_points is None:
        print("Fallback to initial configurations")
        best_initial_ratio = 0
        for config in initial_configs:
            ratio = objective_ratio(config.flatten())
            if ratio > best_initial_ratio:
                best_initial_ratio = ratio
                best_final_points = config.copy()
    
    # Ensure final points are within bounds
    if best_final_points is not None:
        best_final_points = np.clip(best_final_points, 0, 1)
        
    # Return the best solution found
    return best_final_points if best_final_points is not None else initial_configs[0]

# EVOLVE-BLOCK-END