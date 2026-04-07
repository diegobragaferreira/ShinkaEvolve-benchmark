# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import itertools

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
        # Avoid division by zero
        if len(distances) == 0:
            return 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        # Minimize negative of min/max ratio (equivalent to maximizing min/max ratio)
        return -min_dist / max_dist
    
    def analyze_distance_distribution(points):
        """Analyze distance distribution to inform optimization strategy"""
        distances = pdist(points)
        if len(distances) == 0:
            return 0, 1, 0.0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        ratio = min_dist / max_dist if max_dist > 0 else 0.0
        return min_dist, max_dist, ratio
    
    def create_hierarchical_grid():
        """Create a hierarchical grid structure for initial placement"""
        # Start with a 2x2 grid (4 points) and gradually expand
        points = []
        
        # 2x2 grid in corners
        for i in range(2):
            for j in range(2):
                x = 0.25 + i * 0.5
                y = 0.25 + j * 0.5
                points.append([x, y])
        
        # Add middle points to form 4x4 structure
        for i in range(4):
            for j in range(4):
                if (i, j) not in [(0,0), (0,1), (1,0), (1,1)]:
                    x = (j + 0.5) / 4.0
                    y = (i + 0.5) / 4.0
                    points.append([x, y])
        
        # Ensure we have exactly 16 points
        if len(points) < 16:
            # Fill remaining points with random distribution
            for i in range(16 - len(points)):
                points.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
        elif len(points) > 16:
            points = points[:16]
            
        return np.array(points)
    
    def adaptive_perturbation(base_points, iteration=0):
        """Apply adaptive perturbations based on current solution quality"""
        # Analyze current configuration
        min_dist, max_dist, ratio = analyze_distance_distribution(base_points)
        
        # Determine perturbation magnitude based on distance quality
        # If ratio is poor, allow larger perturbations; if good, use smaller ones
        if ratio < 0.1:
            # Poor configuration - allow larger perturbations
            base_magnitude = 0.08
        elif ratio < 0.2:
            base_magnitude = 0.05
        elif ratio < 0.3:
            base_magnitude = 0.03
        else:
            # Good configuration - small perturbations for refinement
            base_magnitude = 0.015
            
        # Apply adaptive perturbation
        perturbation = np.random.normal(0, base_magnitude, base_points.shape)
        perturbed_points = base_points + perturbation
        # Clip to valid range
        perturbed_points = np.clip(perturbed_points, 0.01, 0.99)
        return perturbed_points
    
    def create_symmetry_breaking_initial():
        """Create initial configuration with explicit symmetry breaking"""
        # Start with structured 4x4 grid
        points = []
        for i in range(4):
            for j in range(4):
                x = (j + 0.5) / 4.0
                y = (i + 0.5) / 4.0
                points.append([x, y])
        
        points = np.array(points)
        
        # Add controlled perturbations with symmetry breaking
        np.random.seed(42)
        # Break symmetry by perturbing corner points differently
        points[0] += np.random.normal(0, 0.01, 2)  # top-left
        points[15] += np.random.normal(0, 0.01, 2) # bottom-right
        
        # Apply adaptive perturbations to all points
        points = adaptive_perturbation(points, 0)
        
        return points
    
    def optimize_with_refinement(initial_points, max_iterations=100):
        """Perform optimization with iterative refinement"""
        points = initial_points.copy()
        best_points = points.copy()
        best_ratio = -np.inf
        
        # First, try a quick global optimization
        bounds = [(0.01, 0.99) for _ in range(32)]
        try:
            result = minimize(
                objective,
                points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            if result.success:
                final_points = result.x.reshape(-1, 2)
                _, _, ratio = analyze_distance_distribution(final_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()
        except:
            pass
        
        # Iterative refinement process
        for iter_num in range(max_iterations):
            # Adaptive perturbation based on current configuration
            perturbed_points = adaptive_perturbation(best_points, iter_num)
            
            # Local optimization on perturbed configuration
            try:
                result = minimize(
                    objective,
                    perturbed_points.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 30, 'ftol': 1e-10, 'gtol': 1e-10}
                )
                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    _, _, ratio = analyze_distance_distribution(final_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()
                        
            except:
                continue
                
        return best_points
    
    # Main optimization loop with hierarchical approach
    np.random.seed(42)
    
    # Stage 1: Generate diverse initial configurations using hierarchical approach
    initial_configs = []
    
    # Configuration 1: Symmetry breaking grid
    config1 = create_symmetry_breaking_initial()
    initial_configs.append(config1)
    
    # Configuration 2: Hierarchical grid structure  
    config2 = create_hierarchical_grid()
    initial_configs.append(config2)
    
    # Configuration 3: Random with structured elements
    config3 = np.random.uniform(0.1, 0.9, (16, 2))
    initial_configs.append(config3)
    
    # Add diversified variants
    for config in initial_configs[:2]:  # Only enhance the first two
        for i in range(2):
            np.random.seed(42 + i)
            perturbed = adaptive_perturbation(config, i)
            initial_configs.append(perturbed)
    
    # Stage 2: Optimize each configuration and keep the best
    best_ratio = -np.inf
    best_points = None
    
    for i, config in enumerate(initial_configs):
        try:
            # Apply optimization with refinement
            optimized_points = optimize_with_refinement(config, max_iterations=20)
            
            # Analyze result
            _, _, ratio = analyze_distance_distribution(optimized_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
                
        except Exception as e:
            continue
    
    # Stage 3: Final refinement if needed
    if best_points is not None:
        # Perform one final optimization with highest precision
        bounds = [(0.01, 0.99) for _ in range(32)]
        try:
            result = minimize(
                objective,
                best_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if result.success:
                final_points = result.x.reshape(-1, 2)
                _, _, ratio = analyze_distance_distribution(final_points)
                if ratio > best_ratio:
                    best_points = final_points
        except:
            pass
    
    # Fallback to structured grid if everything fails
    if best_points is None:
        fallback_points = create_symmetry_breaking_initial()
        bounds = [(0.01, 0.99) for _ in range(32)]
        try:
            result = minimize(
                objective,
                fallback_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            if result.success:
                best_points = result.x.reshape(-1, 2)
            else:
                best_points = fallback_points
        except:
            best_points = fallback_points
    
    return best_points

# EVOLVE-BLOCK-END