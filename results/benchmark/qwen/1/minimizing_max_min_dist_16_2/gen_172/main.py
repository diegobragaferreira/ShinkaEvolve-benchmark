# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the min/max distance ratio for given points"""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max <= 1e-12:
            return 0.0
        return d_min / d_max
    
    def create_hexagonal_grid():
        """Create an initial hexagonal grid configuration"""
        # Create a 4x4 grid with hexagonal offset
        points = []
        rows, cols = 4, 4
        
        for i in range(rows):
            for j in range(cols):
                # Hexagonal grid with alternating row offset
                x = (j + 0.5 * (i % 2)) / (cols - 1) if cols > 1 else 0.5
                y = i / (rows - 1) if rows > 1 else 0.5
                
                # Add slight perturbation to break symmetries
                x += (np.random.rand() - 0.5) * 0.05
                y += (np.random.rand() - 0.5) * 0.05
                
                points.append([x, y])
        
        # Clip to valid range and ensure we have exactly 16 points
        points = np.array(points[:16])
        points = np.clip(points, 0, 1)
        return points
    
    def voronoi_based_perturbation(points, step_size=0.005):
        """Apply perturbations based on Voronoi cell analysis"""
        try:
            # Compute Voronoi diagram
            vor = Voronoi(points)
            
            # For each point, determine if it's in a good position
            new_points = points.copy()
            
            # Perturb each point toward increasing minimum distances
            for i in range(len(points)):
                # Get nearby points (excluding itself)
                neighbors = []
                for j in range(len(points)):
                    if i != j:
                        neighbors.append(j)
                
                # Simple heuristic: move point to maximize its minimum distance
                current_distance = np.min([np.linalg.norm(points[i] - points[j]) 
                                         for j in neighbors if j != i])
                
                # Try small moves in various directions
                best_move = None
                best_ratio = compute_min_max_ratio(new_points)
                
                # Test several small moves
                for _ in range(10):
                    # Generate random perturbation
                    dx = (np.random.rand() - 0.5) * step_size
                    dy = (np.random.rand() - 0.5) * step_size
                    
                    # Apply perturbation
                    test_points = new_points.copy()
                    test_points[i] += [dx, dy]
                    
                    # Keep within bounds
                    test_points[i] = np.clip(test_points[i], 0, 1)
                    
                    # Check if this improves the ratio
                    test_ratio = compute_min_max_ratio(test_points)
                    if test_ratio > best_ratio:
                        best_ratio = test_ratio
                        best_move = [dx, dy]
                
                # Apply best move if found
                if best_move is not None:
                    new_points[i] += best_move
                    
            return new_points
            
        except Exception:
            # Fall back to simple random perturbations if Voronoi fails
            new_points = points.copy()
            for i in range(len(points)):
                dx = (np.random.rand() - 0.5) * step_size
                dy = (np.random.rand() - 0.5) * step_size
                new_points[i] += [dx, dy]
                new_points[i] = np.clip(new_points[i], 0, 1)
            return new_points
    
    def geometric_optimization(points, max_iter=100):
        """Optimize using geometric principles and Voronoi analysis"""
        current_points = points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Iteratively improve using geometric heuristics
        for iteration in range(max_iter):
            # Apply Voronoi-based perturbations
            new_points = voronoi_based_perturbation(current_points, 0.01)
            
            # Evaluate new configuration
            new_ratio = compute_min_max_ratio(new_points)
            
            # Accept improvement
            if new_ratio > current_ratio:
                current_points = new_points
                current_ratio = new_ratio
                
                # Update best if this is better
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = new_points.copy()
            
            # Occasionally reset to best if stuck
            if iteration % 20 == 0 and iteration > 0:
                if current_ratio < best_ratio * 0.99:
                    current_points = best_points.copy()
                    current_ratio = best_ratio
                    
        return best_points
    
    def get_theoretical_optimal_16():
        """Get a theoretically informed initial configuration"""
        # Using the idea of hexagonal packing with proper spacing
        # For 16 points, arrange in a roughly circular pattern with good spacing
        
        # Generate points in a circle and then spread them out
        theta = np.linspace(0, 2*np.pi, 16)
        r = 0.4 * np.ones(16)  # Radius around center
        
        # Add some variation to avoid perfect symmetry
        r += np.random.normal(0, 0.05, 16)
        r = np.clip(r, 0.1, 0.45)
        
        x = 0.5 + r * np.cos(theta)
        y = 0.5 + r * np.sin(theta)
        
        points = np.column_stack([x, y])
        return points
    
    # Main algorithm
    # Phase 1: Generate multiple initial configurations
    np.random.seed(42)
    
    # Try multiple initialization strategies
    initial_configs = [
        create_hexagonal_grid(),
        get_theoretical_optimal_16(),
        np.random.rand(16, 2) * 0.8 + 0.1,  # Centered random
    ]
    
    # Add some variation to initial configurations
    for i in range(len(initial_configs)):
        config = initial_configs[i]
        # Add some noise
        noise = np.random.normal(0, 0.02, config.shape)
        config += noise
        config = np.clip(config, 0, 1)
        initial_configs[i] = config
    
    # Phase 2: Optimize each configuration using geometric approach
    best_points = None
    best_ratio = 0.0
    
    for initial_config in initial_configs:
        try:
            optimized_points = geometric_optimization(initial_config, max_iter=50)
            ratio = compute_min_max_ratio(optimized_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
        except Exception:
            continue
    
    # If no optimization worked, return the best initial configuration
    if best_points is None:
        # Return the first valid initial configuration with reasonable ratio
        for initial_config in initial_configs:
            try:
                ratio = compute_min_max_ratio(initial_config)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = initial_config.copy()
            except Exception:
                continue
    
    # Final refinement using a simple local optimization approach
    if best_points is not None:
        # Apply one final geometric optimization
        final_points = geometric_optimization(best_points, max_iter=10)
        return final_points
    
    # Fallback to a default configuration
    return create_hexagonal_grid()

# EVOLVE-BLOCK-END