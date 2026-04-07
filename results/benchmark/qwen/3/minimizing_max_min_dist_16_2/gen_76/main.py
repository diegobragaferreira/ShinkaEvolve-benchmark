# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
from itertools import product

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distances between all point pairs."""
        if len(points) < 2:
            return 0.0
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return 0.0
            
        return min_dist / max_dist
    
    def create_grid_initial():
        """Create initial configuration on a structured grid."""
        # Create a 4x4 grid pattern for 16 points
        grid_points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + (i % 2) * 0.125  # Offset every other row
                y = i * 0.25
                grid_points.append([x, y])
        return np.array(grid_points)
    
    def perturb_points(points, magnitude=0.01):
        """Add small random perturbations to points."""
        return points + np.random.normal(0, magnitude, points.shape)
    
    def refine_grid(points, scale=0.001, max_iter=50):
        """Refine point positions using local optimization."""
        current_points = points.copy()
        best_ratio = compute_min_max_ratio(current_points)
        
        for _ in range(max_iter):
            # Try small perturbations to each point
            for i in range(len(current_points)):
                test_points = current_points.copy()
                # Perturb one point at a time
                test_points[i] += np.random.normal(0, scale, 2)
                test_points = np.clip(test_points, 0, 1)
                
                ratio = compute_min_max_ratio(test_points)
                if ratio > best_ratio:
                    current_points = test_points
                    best_ratio = ratio
                    
        return current_points
    
    def multi_scale_refinement(initial_points, scales=[0.01, 0.005, 0.001]):
        """Perform refinement at multiple scales."""
        best_points = initial_points.copy()
        best_ratio = compute_min_max_ratio(best_points)
        
        for scale in scales:
            # Refine at current scale
            refined = refine_grid(best_points, scale=scale, max_iter=100)
            ratio = compute_min_max_ratio(refined)
            
            if ratio > best_ratio:
                best_points = refined
                best_ratio = ratio
                
        return best_points
    
    def grid_search_space_exploration():
        """Perform systematic grid-based exploration with local refinement."""
        # Start with structured grid
        base_points = create_grid_initial()
        
        # Add some randomness to avoid symmetric traps
        np.random.seed(42)
        base_points = perturb_points(base_points, 0.005)
        base_points = np.clip(base_points, 0, 1)
        
        best_points = base_points.copy()
        best_ratio = compute_min_max_ratio(best_points)
        
        # Multi-scale refinement
        best_points = multi_scale_refinement(best_points)
        best_ratio = compute_min_max_ratio(best_points)
        
        # Additional refinement steps
        for _ in range(10):
            # Try different random perturbations
            test_points = perturb_points(best_points, 0.002)
            test_points = np.clip(test_points, 0, 1)
            
            ratio = compute_min_max_ratio(test_points)
            if ratio > best_ratio:
                best_points = test_points
                best_ratio = ratio
                
            # Apply local refinement
            refined = refine_grid(test_points, 0.0005, 50)
            ratio = compute_min_max_ratio(refined)
            if ratio > best_ratio:
                best_points = refined
                best_ratio = ratio
                
        return best_points
    
    def adaptive_local_search(initial_points, max_time=175):
        """Adaptive local search with progressive refinement."""
        start_time = time.time()
        
        points = initial_points.copy()
        ratio = compute_min_max_ratio(points)
        
        best_points = points.copy()
        best_ratio = ratio
        
        # Progressive refinement approach
        scales = [0.02, 0.01, 0.005, 0.002, 0.001]
        iter_count = 0
        
        while (time.time() - start_time) < max_time and iter_count < 500:
            # Try different refinement strategies
            for scale in scales:
                # Local optimization at different scales
                test_points = points.copy()
                
                # Add small perturbations
                for i in range(len(test_points)):
                    if np.random.random() < 0.3:  # 30% chance to perturb
                        test_points[i] += np.random.normal(0, scale, 2)
                
                test_points = np.clip(test_points, 0, 1)
                test_ratio = compute_min_max_ratio(test_points)
                
                if test_ratio > ratio:
                    points = test_points
                    ratio = test_ratio
                    
                    if ratio > best_ratio:
                        best_points = points.copy()
                        best_ratio = ratio
                        
                # Early termination if improvement is minimal
                if (time.time() - start_time) > max_time * 0.9:
                    break
                    
            iter_count += 1
            
        return best_points
    
    # Main algorithm
    np.random.seed(42)
    
    # Step 1: Start with structured grid
    initial_points = create_grid_initial()
    
    # Step 2: Add slight perturbations
    initial_points = perturb_points(initial_points, 0.005)
    initial_points = np.clip(initial_points, 0, 1)
    
    # Step 3: Multi-scale refinement
    refined_points = multi_scale_refinement(initial_points)
    
    # Step 4: Adaptive local search
    final_points = adaptive_local_search(refined_points)
    
    # Final validation and improvement
    final_ratio = compute_min_max_ratio(final_points)
    
    # Try a few more random refinements
    for _ in range(20):
        test_points = perturb_points(final_points, 0.001)
        test_points = np.clip(test_points, 0, 1)
        test_ratio = compute_min_max_ratio(test_points)
        
        if test_ratio > final_ratio:
            final_points = test_points
            final_ratio = test_ratio
    
    return final_points

# EVOLVE-BLOCK-END