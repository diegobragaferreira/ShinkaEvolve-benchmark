# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def objective_with_penalty(x, penalty_weight=1000.0):
        """Objective function with penalty for constraint violations."""
        points = x.reshape(-1, 2)
        
        # Calculate pairwise distances
        distances = pdist(points)
        
        # Calculate min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max == 0:
            return penalty_weight * 1000  # Large penalty for invalid configuration
            
        # Compute ratio
        ratio = d_min / d_max
        
        # Add penalty for boundary violations
        penalty = 0
        for i in range(16):
            # Penalty for going out of bounds
            if points[i, 0] < 0 or points[i, 0] > 1 or points[i, 1] < 0 or points[i, 1] > 1:
                penalty += penalty_weight * 100
                
        # Return negative ratio plus penalty (since we want to maximize)
        return -ratio + penalty

    def create_multi_resolution_initialization():
        """Create diverse initial configurations across multiple resolutions."""
        configs = []
        
        # Resolution 1: Coarse 4x4 grid
        np.random.seed(42)
        grid_x = np.linspace(0.1, 0.9, 4)
        grid_y = np.linspace(0.1, 0.9, 4)
        coarse_grid = np.array([[x, y] for x in grid_x for y in grid_y])
        # Add perturbations
        noise = np.random.normal(0, 0.08, (16, 2))
        coarse_config = np.clip(coarse_grid + noise, 0, 1)
        configs.append(coarse_config)
        
        # Resolution 2: Fine grid
        np.random.seed(123)
        fine_grid_x = np.linspace(0.1, 0.9, 4)
        fine_grid_y = np.linspace(0.1, 0.9, 4)
        fine_grid = np.array([[x, y] for x in fine_grid_x for y in fine_grid_y])
        # Add small perturbations to make more varied
        noise = np.random.normal(0, 0.02, (16, 2))
        fine_config = np.clip(fine_grid + noise, 0, 1)
        configs.append(fine_config)
        
        # Resolution 3: Fibonacci spiral
        np.random.seed(456)
        phi = (1 + np.sqrt(5)) / 2
        angles = np.arange(16) * 2 * np.pi / phi
        radii = np.sqrt(np.linspace(0.05, 0.45, 16))
        spiral_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        spiral_points = np.clip((spiral_points + 1) / 2, 0.05, 0.95)
        configs.append(spiral_points)
        
        # Resolution 4: Hexagonal pattern
        np.random.seed(789)
        hex_x = np.array([0.15, 0.45, 0.75, 0.3, 0.6, 0.15, 0.45, 0.75, 0.225, 0.525, 0.825, 0.375, 0.675, 0.225, 0.525, 0.825])
        hex_y = np.array([0.15, 0.15, 0.15, 0.45, 0.45, 0.75, 0.75, 0.75, 0.3, 0.3, 0.3, 0.6, 0.6, 0.9, 0.9, 0.9])
        hex_points = np.column_stack([hex_x, hex_y])
        noise = np.random.normal(0, 0.03, (16, 2))
        hex_config = np.clip(hex_points + noise, 0, 1)
        configs.append(hex_config)

        return configs

    def progressive_optimization(x0, max_iter=1000):
        """Perform progressive optimization with increasing precision."""
        points = x0.reshape(-1, 2)
        current_x = x0.copy()
        
        # Progressive refinement with varying tolerances
        tolerances = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
        max_iters = [100, 100, 100, 100, 100, 100]
        
        for i, (tol, max_it) in enumerate(zip(tolerances, max_iters)):
            try:
                bounds = [(0, 1) for _ in range(32)]
                
                # Use L-BFGS-B with progressive tightening
                result = minimize(
                    objective_with_penalty,
                    current_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max_it, 'ftol': tol, 'gtol': tol}
                )
                
                if result.success:
                    current_x = result.x
                else:
                    # If first stage fails, try with different approach
                    result = minimize(
                        objective_with_penalty,
                        current_x,
                        method='TNC',
                        bounds=bounds,
                        options={'maxiter': max_it, 'ftol': tol, 'gtol': tol}
                    )
                    if result.success:
                        current_x = result.x
                        
            except Exception:
                continue
                
        return current_x

    def symmetry_breaking_mutation(points, mutation_rate=0.3):
        """Apply symmetry breaking mutations to prevent symmetric solutions."""
        mutated_points = points.copy()
        
        # Identify potential symmetric patterns and break them
        np.random.seed(int(time.time()) % 1000000)
        
        # Break symmetry by randomly selecting some points and perturbing them
        num_mutate = int(len(points) * mutation_rate)
        indices_to_mutate = np.random.choice(len(points), num_mutate, replace=False)
        
        for idx in indices_to_mutate:
            # Apply different types of mutations to break symmetry
            mutation_type = np.random.choice(['random', 'directional'])
            
            if mutation_type == 'random':
                # Random small perturbation
                noise = np.random.normal(0, 0.01, 2)
            else:
                # Directional perturbation toward center to promote spread
                center = np.mean(points, axis=0)
                direction = points[idx] - center
                # If point is near center, move away; otherwise move towards center
                if np.linalg.norm(direction) < 0.1:
                    noise = np.random.normal(0, 0.01, 2)
                else:
                    # Move towards center to avoid too much clustering
                    noise = -0.1 * direction + np.random.normal(0, 0.005, 2)
            
            mutated_points[idx] += noise
            mutated_points[idx] = np.clip(mutated_points[idx], 0, 1)
            
        return mutated_points

    def enhanced_evolutionary_search():
        """Enhanced evolutionary search to find promising starting points."""
        # Run differential evolution with multiple trials
        best_solution = None
        best_fitness = float('inf')
        
        for trial in range(3):
            try:
                bounds = [(0, 1) for _ in range(32)]
                de_result = differential_evolution(
                    objective_with_penalty,
                    bounds,
                    maxiter=50,
                    popsize=12,
                    seed=42 + trial,
                    tol=1e-6,
                    mutation=(0.5, 1),
                    recombination=0.7
                )
                
                if de_result.success:
                    fitness = de_result.fun
                    if fitness < best_fitness:
                        best_fitness = fitness
                        best_solution = de_result.x
                        
            except Exception:
                continue
                
        if best_solution is not None:
            return best_solution.reshape(-1, 2)
        else:
            return None

    # Main optimization routine
    best_ratio = -np.inf
    best_points = None
    
    # Create multi-resolution initial configurations
    initial_configs = create_multi_resolution_initialization()
    
    # Add evolutionary result as additional starting point
    try:
        evol_result = enhanced_evolutionary_search()
        if evol_result is not None:
            initial_configs.append(evol_result)
    except Exception:
        pass
    
    # Progressive optimization with symmetry breaking
    start_time = time.time()
    max_time = 170
    
    for i, initial_points in enumerate(initial_configs):
        # Check if time limit is approaching
        if time.time() - start_time > max_time:
            break
            
        try:
            # Apply symmetry breaking to initial configuration
            symmetrical_points = symmetry_breaking_mutation(initial_points)
            
            # Progressive optimization
            optimized_x = progressive_optimization(symmetrical_points.flatten(), 800)
            
            # Extract final points
            final_points = optimized_x.reshape(-1, 2)
            
            # Compute ratio
            distances = pdist(final_points)
            if len(distances) > 0:
                d_min = np.min(distances)
                d_max = np.max(distances)
                
                if d_max > 0:
                    ratio = d_min / d_max
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()
                        
        except Exception as e:
            continue
    
    # Fallback to best initial configuration if no optimization successful
    if best_points is None:
        if len(initial_configs) > 0:
            best_points = initial_configs[0]
        else:
            np.random.seed(42)
            best_points = np.random.rand(16, 2)
    
    return best_points

# EVOLVE-BLOCK-END