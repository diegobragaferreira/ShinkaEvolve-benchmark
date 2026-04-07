# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0.0
            
        # Compute pairwise distances
        distances = cdist(points, points)
        
        # Mask diagonal elements (distance to self is 0)
        np.fill_diagonal(distances, np.inf)
        
        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return 0.0
            
        return min_dist / max_dist
    
    def voronoi_relaxation(points, max_iterations=100, tolerance=1e-6):
        """Apply Lloyd relaxation using Voronoi diagrams."""
        prev_points = points.copy()
        
        for iteration in range(max_iterations):
            # Create Voronoi diagram
            try:
                vor = Voronoi(points)
            except:
                # Fallback for degenerate cases
                return points
            
            # Compute centroids of Voronoi regions
            new_points = []
            
            # For each point, find corresponding Voronoi cell centroid
            for i in range(len(points)):
                # Find vertices of Voronoi cell for point i
                regions = [r for r in vor.point_region if r != -1]
                if len(regions) <= i:
                    new_points.append(points[i])
                    continue
                    
                # Simple approach: use the point itself for non-boundary cases
                # In practice, we'd compute actual Voronoi cell centroids
                new_points.append(points[i])
            
            # More practical approach: regular Lloyd relaxation with boundary handling
            new_points = points.copy()
            
            # For each point, compute the centroid of its Voronoi cell
            # Since direct Voronoi centroid calculation is complex, we'll do simple projection to maintain constraints
            for i in range(len(points)):
                # Simple heuristic: add small displacement towards average of neighbors
                # but keep within bounds
                neighbor_indices = []
                if len(points) > 1:
                    # Find 5 closest neighbors
                    dists = cdist([points[i]], points)[0]
                    dists[i] = np.inf  # Exclude self
                    neighbor_indices = np.argsort(dists)[:5]  # Top 5 neighbors
                
                if len(neighbor_indices) > 0:
                    # Move towards centroid of neighbors, but clipped to bounds
                    neighbors = points[neighbor_indices]
                    centroid = np.mean(neighbors, axis=0)
                    # Small displacement towards centroid
                    displacement = (centroid - points[i]) * 0.1
                    new_points[i] = points[i] + displacement
                    
            # Apply boundary constraints
            new_points = np.clip(new_points, 0.01, 0.99)
            
            # Check for convergence
            diff = np.sum(np.abs(new_points - prev_points))
            if diff < tolerance:
                break
                
            points = new_points.copy()
            prev_points = new_points.copy()
        
        return points
    
    def create_initial_configuration():
        """Create a good initial configuration using hexagonal pattern with noise."""
        # Create hexagonal grid pattern
        points = []
        rows = 4
        cols = 4
        
        for i in range(rows):
            for j in range(cols):
                if len(points) < 16:
                    # Hexagonal grid with offset rows
                    x = j + (i % 2) * 0.5
                    y = i * np.sqrt(3) / 2
                    points.append([x, y])
        
        # Normalize to unit square and add noise
        points = np.array(points[:16])
        points[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min()) * 0.9 + 0.05
        points[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min()) * 0.9 + 0.05
        
        # Add small random noise for diversity
        points += np.random.normal(0, 0.005, points.shape)
        
        # Ensure bounds
        points = np.clip(points, 0.01, 0.99)
        
        return points
    
    def local_refinement(points, max_iterations=100):
        """Perform local optimization around the current solution."""
        best_points = points.copy()
        best_ratio = compute_min_max_ratio(best_points)
        
        for iteration in range(max_iterations):
            # Try small random perturbations
            test_points = best_points.copy()
            
            # Perturb one random point
            idx = np.random.randint(0, 16)
            perturbation = np.random.normal(0, 0.001, 2)
            test_points[idx] += perturbation
            test_points[idx] = np.clip(test_points[idx], 0.01, 0.99)
            
            # Evaluate
            test_ratio = compute_min_max_ratio(test_points)
            
            if test_ratio > best_ratio:
                best_ratio = test_ratio
                best_points = test_points.copy()
        
        return best_points
    
    # Step 1: Create initial configuration
    current_points = create_initial_configuration()
    
    # Step 2: Apply Voronoi-based relaxation
    relaxed_points = voronoi_relaxation(current_points, max_iterations=50)
    
    # Step 3: Local refinement
    final_points = local_refinement(relaxed_points, max_iterations=200)
    
    # Step 4: Additional refinement with multiple approaches
    best_points = final_points.copy()
    best_ratio = compute_min_max_ratio(best_points)
    
    # Try different refinements
    for _ in range(3):
        # Random restart
        restart_points = create_initial_configuration()
        # Apply relaxation
        relaxed = voronoi_relaxation(restart_points, max_iterations=30)
        # Local refinement
        refined = local_refinement(relaxed, max_iterations=100)
        ratio = compute_min_max_ratio(refined)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = refined.copy()
    
    return best_points

# EVOLVE-BLOCK-END
