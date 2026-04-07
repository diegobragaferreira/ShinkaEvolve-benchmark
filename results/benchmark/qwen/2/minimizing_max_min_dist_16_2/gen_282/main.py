# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from numba import jit
import warnings
import time

@jit(nopython=True)
def fast_pdist_squared(points):
    """Fast computation of squared pairwise distances using numba"""
    n = points.shape[0]
    distances_squared = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dist_sq = dx*dx + dy*dy
            distances_squared[i, j] = dist_sq
            distances_squared[j, i] = dist_sq
    return distances_squared

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the minimum to maximum distance ratio for given points."""
        if len(points) < 2:
            return 0.0

        # Use faster distance calculation
        distances_squared = fast_pdist_squared(points)
        distances = np.sqrt(distances_squared[np.triu_indices_from(distances_squared, k=1)])
        
        # Find min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return 0.0

        return min_dist / max_dist

    def objective_with_regularization(x):
        """Objective with regularization to avoid numerical issues"""
        points = x.reshape(-1, 2)
        distances_squared = fast_pdist_squared(points)
        distances = np.sqrt(distances_squared[np.triu_indices_from(distances_squared, k=1)])

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Add small epsilon to avoid division by zero
        eps = 1e-12
        if max_dist < eps:
            return -1.0  # Return worst possible value

        ratio = min_dist / (max_dist + eps)
        return -ratio

    def golden_spiral_2d(n_points):
        """Generate points on a 2D golden spiral"""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio

        for i in range(n_points):
            angle = 2 * np.pi * i / phi
            radius = np.sqrt(i / (n_points - 1)) if n_points > 1 else 0
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])

        return np.array(points)

    def hexagonal_lattice_2d(n_points):
        """Generate points on a 2D hexagonal lattice"""
        # Calculate grid parameters
        rows = int(np.ceil(np.sqrt(n_points)))
        cols = int(np.ceil(n_points / rows))

        # Create hexagonal grid
        points = []
        spacing = 1.0 / max(rows, cols)

        for i in range(rows):
            for j in range(cols):
                if len(points) >= n_points:
                    break
                # Offset every other row
                x_offset = (i % 2) * spacing / 2
                x = (j * spacing) + x_offset
                y = i * spacing
                points.append([x, y])

        return np.array(points[:n_points])

    def create_grid_initialization():
        """Create a structured 4x4 grid with adaptive perturbations to improve spacing"""
        # Start with regular 4x4 grid
        grid_points = []
        for i in range(4):
            for j in range(4):
                x = i / 3.0  # Normalized to [0,1] range
                y = j / 3.0
                grid_points.append([x, y])
        
        points = np.array(grid_points)
        
        # Apply adaptive perturbations based on location
        # Emphasize corners and edges to encourage better spreading
        for i in range(16):
            row, col = i // 4, i % 4
            
            # More aggressive perturbations for corners to break symmetry
            if (row in [0, 3] and col in [0, 3]):
                std = 0.035
            elif (row in [0, 3] or col in [0, 3]):
                std = 0.02
            else:
                std = 0.01
                
            # Apply perturbation
            points[i, 0] += np.random.normal(0, std)
            points[i, 1] += np.random.normal(0, std)
        
        # Ensure points stay within bounds
        points = np.clip(points, 0, 1)
        return points

    def distance_aware_local_search(points, max_iterations=50):
        """Perform a local search that specifically targets improving minimum distance"""
        current_points = points.copy()
        best_points = points.copy()
        best_ratio = compute_min_max_ratio(best_points)
        
        for iteration in range(max_iterations):
            improved = False
            
            # Try moving each point to improve the minimum distance
            for i in range(len(current_points)):
                original_point = current_points[i].copy()
                best_move = original_point.copy()
                best_improvement = 0.0
                
                # Test small movements in different directions
                movements = [
                    [-0.01, -0.01], [-0.01, 0], [-0.01, 0.01],
                    [0, -0.01], [0, 0.01],
                    [0.01, -0.01], [0.01, 0], [0.01, 0.01]
                ]
                
                # Test each movement
                for dx, dy in movements:
                    test_point = original_point.copy()
                    test_point[0] += dx
                    test_point[1] += dy
                    
                    # Clip to bounds
                    test_point[0] = np.clip(test_point[0], 0.001, 0.999)
                    test_point[1] = np.clip(test_point[1], 0.001, 0.999)
                    
                    # Temporarily update this point
                    temp_points = current_points.copy()
                    temp_points[i] = test_point
                    
                    # Check ratio improvement
                    ratio = compute_min_max_ratio(temp_points)
                    improvement = ratio - best_ratio
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = test_point.copy()
                
                # Apply the best move if it improves the solution
                if best_improvement > 0:
                    current_points[i] = best_move
                    improved = True
                    
                    # Update best solution if this is better
                    ratio = compute_min_max_ratio(current_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = current_points.copy()
            
            # Early stopping if no improvement
            if not improved:
                break
                
        return best_points

    def progressive_optimization(initial_points, max_time=170):
        """Apply multi-stage optimization with increasing precision"""
        start_time = time.time()
        current_points = initial_points.copy()
        
        # Stage 1: Coarse optimization with relaxed tolerances
        if time.time() - start_time < max_time * 0.3:
            result = minimize(
                objective_with_regularization,
                current_points.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(32)],
                options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if result.success:
                current_points = result.x.reshape(-1, 2)
        
        # Stage 2: Refinement with medium precision
        if time.time() - start_time < max_time * 0.6:
            # Apply distance-aware local search
            current_points = distance_aware_local_search(current_points, max_iterations=30)
            
            # Fine optimization
            result = minimize(
                objective_with_regularization,
                current_points.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(32)],
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.success:
                current_points = result.x.reshape(-1, 2)
        
        # Stage 3: Final high precision optimization
        if time.time() - start_time < max_time * 0.9:
            # Apply final local search
            current_points = distance_aware_local_search(current_points, max_iterations=50)
            
            # Very tight optimization
            result = minimize(
                objective_with_regularization,
                current_points.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(32)],
                options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.success:
                current_points = result.x.reshape(-1, 2)
        
        return current_points

    def optimize_with_restarts():
        """Run optimization with multiple enhanced initial configurations"""
        best_ratio = -np.inf
        best_points = None

        # Try multiple different initial configurations with enhanced strategies
        initial_configs = []

        # 1. Structured grid with adaptive perturbations (from progressive approach)
        grid_points = create_grid_initialization()
        # Fix corner and some interior points to break symmetry
        grid_points[0] = [0.0, 0.0]  # Bottom-left corner
        grid_points[3] = [1.0, 0.0]  # Bottom-right corner
        grid_points[12] = [0.0, 1.0] # Top-left corner
        grid_points[15] = [1.0, 1.0] # Top-right corner
        grid_points[5] = [0.25, 0.25]  # Interior point
        grid_points[10] = [0.75, 0.75] # Interior point
        initial_configs.append(grid_points)

        # 2. Golden spiral pattern
        spiral_points = golden_spiral_2d(16)
        # Scale and center the spiral
        spiral_points = (spiral_points - np.min(spiral_points, axis=0)) / (
            np.max(spiral_points, axis=0) - np.min(spiral_points, axis=0) + 1e-12)
        spiral_points = spiral_points * 0.8 + 0.1  # Scale to [0.1, 0.9]
        initial_configs.append(spiral_points.copy())

        # 3. Hexagonal lattice pattern
        hex_points = hexagonal_lattice_2d(16)
        # Normalize to [0.1, 0.9] range
        hex_points = (hex_points - np.min(hex_points, axis=0)) / (
            np.max(hex_points, axis=0) - np.min(hex_points, axis=0) + 1e-12)
        hex_points = hex_points * 0.8 + 0.1
        initial_configs.append(hex_points.copy())

        # 4. Perturbed grid with fixed corners
        grid_points = np.array([[i/4, j/4] for i in range(4) for j in range(4)])[:16]
        grid_points += np.random.normal(0, 0.05, (16, 2))
        grid_points = np.clip(grid_points, 0, 1)
        # Fix corners and interior points
        grid_points[0] = [0.0, 0.0]
        grid_points[3] = [1.0, 0.0]
        grid_points[12] = [0.0, 1.0]
        grid_points[15] = [1.0, 1.0]
        grid_points[5] = [0.25, 0.25]
        grid_points[10] = [0.75, 0.75]
        initial_configs.append(grid_points)

        # 5. Random uniform points with fixed seed
        np.random.seed(42)
        random_points = np.random.rand(16, 2)
        initial_configs.append(random_points)

        # 6. Enhanced corner-based initialization
        corner_points = np.array([
            [0.05, 0.05], [0.95, 0.05], [0.05, 0.95], [0.95, 0.95],
            [0.5, 0.05], [0.5, 0.95], [0.05, 0.5], [0.95, 0.5],
            [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75],
            [0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]
        ])
        initial_configs.append(corner_points)

        # Run optimization for each configuration with progressive approach
        for i, init_points in enumerate(initial_configs):
            try:
                # Apply progressive optimization
                final_points = progressive_optimization(init_points, max_time=165)
                
                # Apply final local search
                final_points = distance_aware_local_search(final_points, max_iterations=100)
                
                # Compute actual ratio
                ratio = compute_min_max_ratio(final_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()

            except Exception as e:
                warnings.warn(f"Optimization failed for initial config {i}: {e}")
                continue

        # If no successful optimization, return the best initial configuration
        if best_points is None:
            return initial_configs[0]

        return best_points

    # Try optimized approach first
    try:
        final_points = optimize_with_restarts()
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}")
        # Fallback to simple approach if something fails
        np.random.seed(42)
        final_points = np.random.rand(16, 2)

    return final_points

# EVOLVE-BLOCK-END