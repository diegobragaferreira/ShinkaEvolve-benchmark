# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def objective(x):
        # Reshape x into points array
        points = x.reshape(-1, 2)

        # Calculate pairwise distances
        distances = pdist(points)

        # Calculate min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Avoid division by zero
        if d_max == 0:
            return -np.inf

        # Return negative ratio (since we want to maximize)
        return -d_min / d_max

    def constraint(x):
        # Ensure points stay within [0,1] x [0,1]
        points = x.reshape(-1, 2)
        return np.concatenate([
            points[:, 0],           # x coordinates >= 0
            1 - points[:, 0],       # x coordinates <= 1
            points[:, 1],           # y coordinates >= 0
            1 - points[:, 1]        # y coordinates <= 1
        ])

    def compute_ratio(points):
        """Compute the min/max distance ratio for given points."""
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return 0.0
        return d_min / d_max

    def adaptive_perturbation(initial_points, current_ratio):
        """Adaptively scale perturbations based on current distance distribution"""
        # Scale perturbation inversely with the current ratio
        base_perturbation = 0.12
        adaptive_scale = max(0.1, 1.0 / (current_ratio + 0.01))  # Prevent extreme scaling
        perturbation_magnitude = base_perturbation * adaptive_scale
        
        # Apply noise with adaptive magnitude
        noise = np.random.normal(0, perturbation_magnitude/3, (16, 2))
        perturbed_points = np.clip(initial_points + noise, 0, 1)
        return perturbed_points

    def create_initial_configurations():
        """Create multiple high-quality initial configurations."""
        configs = []
        
        # Configuration 1: Structured 4x4 grid with adaptive perturbations
        np.random.seed(42)
        grid_x = np.linspace(0.1, 0.9, 4)
        grid_y = np.linspace(0.1, 0.9, 4)
        grid_points = np.array([[x, y] for x in grid_x for y in grid_y])
        config1 = adaptive_perturbation(grid_points, 0.1)
        configs.append(config1)
        
        # Configuration 2: Golden ratio spiral for better distribution
        np.random.seed(123)
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        angles = np.arange(16) * 2 * np.pi / phi
        radii = np.sqrt(np.linspace(0.05, 0.45, 16))
        spiral_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        spiral_points = np.clip((spiral_points + 1) / 2, 0, 1)
        configs.append(spiral_points)
        
        # Configuration 3: Random with clustering avoidance
        np.random.seed(456)
        config3 = np.random.uniform(0.05, 0.95, (16, 2))
        # Add some structure to avoid very tight clusters
        for i in range(0, 16, 4):  # Group every 4 points
            group_center = np.mean(config3[i:i+4], axis=0)
            config3[i:i+4] += np.random.normal(0, 0.03, (4, 2))
            config3[i:i+4] = np.clip(config3[i:i+4], 0, 1)
        configs.append(config3)
        
        # Configuration 4: Improved hexagonal grid approximation
        np.random.seed(789)
        hex_x = np.array([0.15, 0.45, 0.75, 0.3, 0.6, 0.15, 0.45, 0.75, 0.225, 0.525, 0.825, 0.375, 0.675, 0.225, 0.525, 0.825])
        hex_y = np.array([0.15, 0.15, 0.15, 0.45, 0.45, 0.75, 0.75, 0.75, 0.3, 0.3, 0.3, 0.6, 0.6, 0.9, 0.9, 0.9])
        hex_points = np.column_stack([hex_x, hex_y])
        config4 = adaptive_perturbation(hex_points, 0.1)
        configs.append(config4)
        
        return configs

    def optimize_with_refinement(x0):
        """Perform sequential optimization with refinement stages."""
        # Stage 1: Fast optimization with L-BFGS-B
        bounds = [(0, 1) for _ in range(32)]
        result1 = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
        )
        
        if not result1.success:
            return None
            
        # Stage 2: Precise optimization with SLSQP
        result2 = minimize(
            objective,
            result1.x,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint},
            options={'maxiter': 1500, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        if result2.success:
            return result2.x
        return None

    def hierarchical_optimization(initial_points_list, max_time=170):
        """Run optimization with coarse-to-fine strategy to improve efficiency"""
        start_time = time.time()
        best_ratio = -np.inf
        best_points = None
        
        # Phase 1: Coarse optimization with fewer iterations
        phase1_configs = []
        for i, initial_points in enumerate(initial_points_list[:2]):  # Use first 2 configs for phase 1
            if time.time() - start_time > max_time * 0.6:  # Reserve time for phase 2
                break
            try:
                # Quick optimization with fewer iterations
                bounds = [(0, 1) for _ in range(32)]
                result = minimize(
                    objective,
                    initial_points.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-5, 'gtol': 1e-5}
                )
                
                if result.success:
                    phase1_configs.append(result.x.reshape(-1, 2))
                    
            except Exception:
                continue
        
        # Phase 2: Detailed optimization on promising configurations
        configs_to_optimize = phase1_configs if phase1_configs else initial_points_list[:2]
        
        for i, initial_points in enumerate(configs_to_optimize):
            if time.time() - start_time > max_time:
                break
                
            try:
                # Optimize using our refined two-stage approach
                optimized_x = optimize_with_refinement(initial_points.flatten())
                
                if optimized_x is not None:
                    optimized_points = optimized_x.reshape(-1, 2)
                    final_ratio = compute_ratio(optimized_points)
                    
                    if final_ratio > best_ratio:
                        best_ratio = final_ratio
                        best_points = optimized_points.copy()
                        
            except Exception as e:
                continue
        
        return best_points if best_points is not None else initial_points_list[0]

    # Multi-start optimization with improved initializations and evolutionary restarts
    best_ratio = -np.inf
    best_points = None
    
    # Generate multiple initial configurations
    initial_configs = create_initial_configurations()
    
    # Integrate evolutionary algorithm restarts with enhanced parameters
    try:
        bounds = [(0, 1) for _ in range(32)]
        de_result = differential_evolution(
            objective,
            bounds,
            maxiter=50,
            popsize=12,  # Increased population size for better exploration
            seed=42,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7
        )

        if de_result.success:
            de_points = de_result.x.reshape(-1, 2)
            # Add evolutionary result as another initial configuration
            initial_configs.append(de_points)
    except Exception:
        pass
    
    # Run hierarchical optimization for better efficiency
    best_points = hierarchical_optimization(initial_configs, max_time=170)
    
    # Final refinement of the best result
    if best_points is not None:
        try:
            final_x = optimize_with_refinement(best_points.flatten())
            if final_x is not None:
                final_points = final_x.reshape(-1, 2)
                final_ratio = compute_ratio(final_points)
                if final_ratio > best_ratio:
                    best_points = final_points
        except Exception:
            pass
    
    # If no successful optimization, return the best initial configuration
    if best_points is None:
        best_points = initial_configs[0] if initial_configs else np.random.uniform(0, 1, (16, 2))
        
    return best_points

# EVOLVE-BLOCK-END