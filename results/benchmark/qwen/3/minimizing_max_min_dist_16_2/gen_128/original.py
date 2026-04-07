# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import time
from scipy.spatial.distance import cdist


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Computes the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0
            
        # Compute pairwise distances
        distances = pdist(points)
        dmin = np.min(distances)
        dmax = np.max(distances)
        
        # Handle edge case where all points are identical
        if dmax == 0:
            return 0.0
            
        return dmin / dmax
    
    def compute_boundary_penalty(points, penalty_weight=10.0):
        """Computes penalty for points near boundaries."""
        penalty = 0
        for point in points:
            # Penalty for being close to any boundary
            dist_to_boundaries = [
                point[0],  # distance to left boundary
                1 - point[0],  # distance to right boundary
                point[1],  # distance to bottom boundary
                1 - point[1]   # distance to top boundary
            ]
            min_dist = min(dist_to_boundaries)
            if min_dist < 0.01:  # Only penalize if very close to boundary
                penalty += penalty_weight * (0.01 - min_dist)**2
        return penalty
    
    def evaluate_with_penalty(points, penalty_weight=10.0):
        """Evaluate ratio with boundary penalty applied."""
        ratio = compute_min_max_ratio(points)
        penalty = compute_boundary_penalty(points, penalty_weight)
        return ratio - penalty
    
    def generate_hexagonal_grid():
        """Generate a hexagonal grid pattern."""
        points = []
        sqrt3 = np.sqrt(3)
        
        # 4x4 hexagonal pattern
        for i in range(4):
            for j in range(4):
                x = j + 0.5 * (i % 2)
                y = i * sqrt3 / 2
                points.append([x, y])
        
        points = np.array(points)
        
        # Normalize to [0,1] x [0,1]
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        
        if x_range > 0:
            points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
        if y_range > 0:
            points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
            
        return points
    
    def initialize_multiple_strategies():
        """Initialize multiple starting configurations."""
        strategies = []
        
        # Strategy 1: Standard hexagonal grid
        base_grid = generate_hexagonal_grid()
        np.random.seed(42)
        perturbed = base_grid + np.random.normal(0, 0.01, base_grid.shape)
        perturbed = np.clip(perturbed, 0, 1)
        strategies.append(("hex", perturbed))
        
        # Strategy 2: Hexagonal with higher noise
        np.random.seed(123)
        perturbed_high = base_grid + np.random.normal(0, 0.02, base_grid.shape)
        perturbed_high = np.clip(perturbed_high, 0, 1)
        strategies.append(("hex_high", perturbed_high))
        
        # Strategy 3: Random initialization
        np.random.seed(456)
        random_points = np.random.rand(16, 2)
        strategies.append(("random", random_points))
        
        # Strategy 4: Triangular lattice
        triangular_points = []
        sqrt3 = np.sqrt(3)
        for i in range(4):
            for j in range(4):
                x = j + 0.5 * (i % 2)
                y = i * sqrt3 / 2
                triangular_points.append([x, y])
        triangular_points = np.array(triangular_points[:16])
        
        # Normalize triangular points
        x_range = np.max(triangular_points[:, 0]) - np.min(triangular_points[:, 0])
        y_range = np.max(triangular_points[:, 1]) - np.min(triangular_points[:, 1])
        if x_range > 0:
            triangular_points[:, 0] = (triangular_points[:, 0] - np.min(triangular_points[:, 0])) / x_range
        if y_range > 0:
            triangular_points[:, 1] = (triangular_points[:, 1] - np.min(triangular_points[:, 1])) / y_range
        strategies.append(("triangular", triangular_points))
        
        # Strategy 5: Hexagonal with lower noise
        np.random.seed(789)
        perturbed_low = base_grid + np.random.normal(0, 0.005, base_grid.shape)
        perturbed_low = np.clip(perturbed_low, 0, 1)
        strategies.append(("hex_low", perturbed_low))
        
        return strategies
    
    def neighborhood_move(current_points, point_indices, step_size=0.01):
        """Performs a coordinated move on a cluster of points."""
        new_points = current_points.copy()
        
        # Calculate centroid of selected points
        centroid = np.mean(current_points[point_indices], axis=0)
        
        # Generate movement vector (same for all selected points)
        move_vector = np.random.normal(0, step_size, 2)
        
        # Apply movement to selected points
        for idx in point_indices:
            new_points[idx] = current_points[idx] + move_vector
            
            # Boundary handling with reflection
            for dim in range(2):
                if new_points[idx, dim] < 0:
                    new_points[idx, dim] = -new_points[idx, dim]
                elif new_points[idx, dim] > 1:
                    new_points[idx, dim] = 2 - new_points[idx, dim]
                    
        return new_points
    
    def adaptive_simulated_annealing(initial_points, max_iterations=5000, initial_temp=0.1):
        """Enhanced simulated annealing with adaptive cooling and neighborhood moves."""
        current_points = initial_points.copy()
        current_ratio = evaluate_with_penalty(current_points)
        
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        temperature = initial_temp
        cooling_rate = 0.9995
        min_temp = 1e-8
        
        # Track recent improvements for adaptive cooling
        recent_improvements = []
        improvement_threshold = 50
        stagnation_counter = 0
        
        for iteration in range(max_iterations):
            # Decide between single point or neighborhood move
            if np.random.random() < 0.7:  # 70% chance of neighborhood move
                # Select random subset of points for neighborhood move
                num_selected = np.random.randint(2, 5)  # 2 to 4 points
                point_indices = np.random.choice(len(current_points), size=num_selected, replace=False)
                
                # Perform neighborhood move
                new_points = neighborhood_move(current_points, point_indices, step_size=temperature * 0.1)
            else:
                # Single point move (traditional approach)
                new_points = current_points.copy()
                point_idx = np.random.randint(len(current_points))
                delta = np.random.uniform(-temperature * 0.1, temperature * 0.1, 2)
                new_points[point_idx] = current_points[point_idx] + delta
                
                # Boundary handling
                new_points[point_idx] = np.clip(new_points[point_idx], 0, 1)
            
            # Evaluate new solution
            new_ratio = evaluate_with_penalty(new_points)
            
            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio:
                current_points = new_points
                current_ratio = new_ratio
                
                if new_ratio > best_ratio:
                    best_points = new_points.copy()
                    best_ratio = new_ratio
                    stagnation_counter = 0
            else:
                if np.random.random() < np.exp((new_ratio - current_ratio) / temperature):
                    current_points = new_points
                    current_ratio = new_ratio
                else:
                    stagnation_counter += 1
            
            # Adaptive cooling: If no improvement for a while, cool faster
            if stagnation_counter > 100:
                cooling_rate = min(cooling_rate * 0.99, 0.9999)
                stagnation_counter = 0
            
            # Cool down temperature
            temperature *= cooling_rate
            if temperature < min_temp:
                temperature = min_temp
            
            # Early stopping if we're not improving much
            recent_improvements.append(1 if new_ratio > current_ratio else 0)
            if len(recent_improvements) > improvement_threshold:
                recent_improvements.pop(0)
                if sum(recent_improvements) < 2 and iteration > 1000:
                    break
        
        return best_points, best_ratio
    
    # Generate multiple initializations
    strategies = initialize_multiple_strategies()
    
    # Run optimization from each starting point
    best_result = None
    best_score = -np.inf
    
    for strategy_name, initial_points in strategies:
        try:
            optimized_points, score = adaptive_simulated_annealing(
                initial_points, max_iterations=3000, initial_temp=0.05
            )
            
            if score > best_score:
                best_score = score
                best_result = optimized_points
                
        except Exception as e:
            continue  # Skip failed runs
    
    # Final refinement with the best result
    if best_result is not None:
        try:
            final_points, _ = adaptive_simulated_annealing(
                best_result, max_iterations=2000, initial_temp=0.02
            )
            return final_points
        except:
            pass
    
    # Fallback to best found if optimization fails
    if best_result is not None:
        return best_result
    
    # Last resort: return a basic hexagonal grid with small perturbation
    base_grid = generate_hexagonal_grid()
    np.random.seed(42)
    fallback = base_grid + np.random.normal(0, 0.005, base_grid.shape)
    return np.clip(fallback, 0, 1)

# EVOLVE-BLOCK-END