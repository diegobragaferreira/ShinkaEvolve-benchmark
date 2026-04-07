# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
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
        
        # Calculate all pairwise distances efficiently
        distances = cdist(points, points, metric='euclidean')
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Return negative ratio to maximize the ratio (negative because we minimize)
        if max_dist <= 0:
            return 0
        return -min_dist / max_dist
    
    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0

        # Efficiently compute all pairwise distances
        distances = cdist(points, points, metric='euclidean')

        # Set diagonal to infinity to ignore self-distances
        np.fill_diagonal(distances, np.inf)

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        if max_dist <= 0:
            return 0

        return min_dist / max_dist
    
    def create_initial_configurations():
        """Create multiple high-quality initial configurations."""
        configs = []
        
        # Configuration 1: Hexagonal grid with adaptive perturbations  
        np.random.seed(42)
        hex_points = []
        for i in range(4):
            for j in range(4):
                x_offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + x_offset) / 3.0
                y = i / 3.0
                hex_points.append([x, y])
        
        # Add small adaptive perturbations
        hex_points = np.array(hex_points) + np.random.uniform(-0.015, 0.015, (16, 2))
        hex_points = np.clip(hex_points, 0, 1)
        configs.append(hex_points)
        
        # Configuration 2: Spiral pattern for good coverage
        np.random.seed(123)
        spiral_points = []
        angles = np.linspace(0, 4 * 2 * np.pi, 16)
        radii = np.linspace(0.1, 0.45, 16)
        
        for angle, radius in zip(angles, radii):
            x = 0.5 + radius * np.cos(angle) * 0.4
            y = 0.5 + radius * np.sin(angle) * 0.4
            spiral_points.append([x, y])
        
        spiral_points = np.array(spiral_points)
        spiral_points = np.clip(spiral_points, 0, 1)
        configs.append(spiral_points)
        
        # Configuration 3: Perturbed regular grid
        np.random.seed(456)
        grid_points = np.array([[i, j] for i in range(4) for j in range(4)])
        grid_points = grid_points.astype(float) / 3.0
        
        # Add position-based perturbations
        for i in range(16):
            row, col = i // 4, i % 4
            scale = 0.02 if (row + col) % 2 == 0 else 0.015
            grid_points[i] += np.random.uniform(-scale, scale, 2)
            
        grid_points = np.clip(grid_points, 0, 1)
        configs.append(grid_points)
        
        return configs
    
    def optimize_with_refinement(x0):
        """Perform sequential optimization with refinement stages."""
        # Stage 1: Fast optimization with L-BFGS-B
        bounds = [(0, 1) for _ in range(32)]
        try:
            result1 = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if result1.success:
                # Stage 2: Precise optimization with SLSQP
                result2 = minimize(
                    objective,
                    result1.x,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
                )
                
                if result2.success:
                    return result2.x
        except Exception:
            pass
        
        return None

    def adaptive_hill_climbing(points, max_iterations=500):
        """Improve solution using adaptive hill climbing with decreasing step size"""
        current_ratio = calculate_min_max_ratio(points)
        step_size = 0.01
        stagnation_count = 0
        max_stagnation = 50
        
        for iteration in range(max_iterations):
            best_improvement = 0
            best_new_points = None
            
            # Try moving each point
            for i in range(len(points)):
                temp_points = points.copy()
                
                # Try multiple random moves for this point
                for _ in range(5):  # Multiple attempts per point
                    move = np.random.uniform(-step_size, step_size, 2)
                    temp_points[i] = points[i] + move
                    
                    # Keep within bounds
                    temp_points[i] = np.clip(temp_points[i], 0, 1)
                    
                    new_ratio = calculate_min_max_ratio(temp_points)
                    
                    if new_ratio > current_ratio:
                        improvement = new_ratio - current_ratio
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_new_points = temp_points.copy()
            
            # If improvement found, accept it
            if best_improvement > 0:
                points = best_new_points
                current_ratio = calculate_min_max_ratio(points)
                stagnation_count = 0
            else:
                stagnation_count += 1
                # Reduce step size if no progress
                if stagnation_count > 10:
                    step_size *= 0.8
                    stagnation_count = 0
                    
            # Early stopping if improvement becomes negligible
            if best_improvement < 1e-12:
                break
                
        return points

    # Multi-start optimization with improved initializations
    best_ratio = -np.inf
    best_points = None
    
    # Generate multiple initial configurations
    initial_configs = create_initial_configurations()
    
    # Add evolutionary algorithm restarts for better exploration
    try:
        bounds = [(0, 1) for _ in range(32)]
        de_result = differential_evolution(
            objective,
            bounds,
            maxiter=30,
            popsize=8,
            seed=42,
            tol=1e-5,
            mutation=(0.5, 1),
            recombination=0.7
        )

        if de_result.success:
            de_points = de_result.x.reshape(-1, 2)
            # Add evolutionary result as another initial configuration
            initial_configs.append(de_points)
    except Exception:
        pass
    
    # Try each initial configuration with optimization
    for i, initial_points in enumerate(initial_configs):
        try:
            # Optimize using our refined two-stage approach
            optimized_x = optimize_with_refinement(initial_points.flatten())
            
            if optimized_x is not None:
                optimized_points = optimized_x.reshape(-1, 2)
                final_ratio = calculate_min_max_ratio(optimized_points)
                
                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_points = optimized_points.copy()
                    
        except Exception as e:
            continue

    # If no successful optimization, apply hill climbing to the best initial config
    if best_points is None:
        best_points = initial_configs[0] if initial_configs else np.random.uniform(0, 1, (16, 2))
    
    # Final refinement with hill climbing
    best_points = adaptive_hill_climbing(best_points)
    
    # Final L-BFGS optimization
    try:
        bounds = [(0, 1) for _ in range(32)]
        final_result = minimize(
            objective,
            best_points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if final_result.success:
            final_points = final_result.x.reshape(-1, 2)
            final_ratio = calculate_min_max_ratio(final_points)
            
            if final_ratio > best_ratio:
                best_points = final_points
    except Exception:
        pass
    
    return best_points

# EVOLVE-BLOCK-END