# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy.spatial import Voronoi
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0

        # Compute pairwise distances with enhanced numerical stability
        distance_matrix = squareform(pdist(points))
        
        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(distance_matrix, np.inf)
        
        # Get all finite distances (excluding NaN and inf values)
        finite_distances = distance_matrix[np.isfinite(distance_matrix)]
        
        if len(finite_distances) == 0:
            return 0

        # Get min and max distances
        dmin = np.min(finite_distances)
        dmax = np.max(finite_distances)

        # Avoid division by zero
        if dmax == 0:
            return 0

        return dmin / dmax

    def compute_voronoi_energy(points):
        """Compute energy based on Voronoi cell areas to encourage uniform distribution."""
        if len(points) < 2:
            return 0
            
        try:
            vor = Voronoi(points)
            # Calculate area of each Voronoi cell (excluding infinite regions)
            areas = []
            for region in vor.regions:
                if len(region) > 0 and -1 not in region:
                    # Approximate area using centroid method
                    vertices = np.array([vor.vertices[i] for i in region])
                    if len(vertices) >= 3:
                        # Simple polygon area calculation
                        x = vertices[:, 0]
                        y = vertices[:, 1]
                        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                        areas.append(area)
                        
            if not areas:
                return 0
                
            # Return inverse of variance of areas (higher = more uniform)
            return 1.0 / (np.var(areas) + 1e-12)
        except:
            return 0

    def generate_triangular_lattice():
        """Generate points arranged in a triangular lattice pattern."""
        # Create a triangular lattice in a unit square
        points = []
        rows = 4
        cols = 4
        
        # Base spacing based on triangular lattice
        spacing = 1.0 / (cols - 1) if cols > 1 else 1.0
        height = np.sqrt(3) / 2 * spacing
        
        for i in range(rows):
            for j in range(cols):
                x = j * spacing
                y = i * height
                # Offset every other row
                if i % 2 == 1:
                    x += spacing / 2
                points.append([x, y])
                
        return np.array(points[:16])  # Ensure exactly 16 points

    def generate_discrete_initial_placement():
        """Generate initial placement using a hybrid discrete approach."""
        # Start with triangular lattice
        base_points = generate_triangular_lattice()
        
        # Add some randomness to break symmetry while maintaining good structure
        np.random.seed(42)
        
        # Mix of structured and random elements
        # Primary structure: triangular lattice
        points = base_points.copy()
        
        # Add strategic perturbation to avoid degenerate configurations
        perturbations = np.random.uniform(-0.02, 0.02, (16, 2))
        points += perturbations
        
        # Ensure points remain within bounds
        points = np.clip(points, 0, 1)
        
        return points

    def adaptive_local_search(initial_points, max_iter=100):
        """Perform adaptive local search with hybrid approach."""
        current_points = initial_points.copy()
        
        # Adaptive parameters
        learning_rate = 0.01
        momentum = 0.9
        velocity = np.zeros_like(current_points)
        
        best_points = current_points.copy()
        best_ratio = compute_min_max_ratio(current_points)
        
        for iteration in range(max_iter):
            # Compute current gradient approximation using finite differences
            current_ratio = compute_min_max_ratio(current_points)
            
            # Small perturbations to estimate gradient
            gradient = np.zeros_like(current_points)
            eps = 1e-6
            
            for i in range(len(current_points)):
                for j in range(2):  # x and y coordinates
                    # Forward difference
                    test_points = current_points.copy()
                    test_points[i, j] += eps
                    test_points = np.clip(test_points, 0, 1)
                    
                    forward_ratio = compute_min_max_ratio(test_points)
                    
                    # Backward difference
                    test_points[i, j] -= 2 * eps
                    test_points = np.clip(test_points, 0, 1)
                    
                    backward_ratio = compute_min_max_ratio(test_points)
                    
                    # Central difference
                    gradient[i, j] = (forward_ratio - backward_ratio) / (2 * eps)
            
            # Update with momentum and adaptive learning rate
            velocity = momentum * velocity - learning_rate * gradient
            current_points += velocity
            
            # Project back to feasible region
            current_points = np.clip(current_points, 0, 1)
            
            # Evaluate new solution
            new_ratio = compute_min_max_ratio(current_points)
            
            # Accept better solutions
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_points = current_points.copy()
            
            # Adaptive learning rate decay
            if iteration > 0 and iteration % 20 == 0:
                learning_rate *= 0.8
                
            # Occasionally add discrete perturbations to escape local minima
            if iteration % 10 == 0:
                np.random.seed(iteration)
                # Add small random displacement
                noise = np.random.normal(0, 0.001, current_points.shape)
                current_points += noise
                current_points = np.clip(current_points, 0, 1)
        
        return best_points

    def multi_scale_optimization(initial_points):
        """Apply multi-scale optimization approach."""
        # Scale 1: Coarse optimization with reduced dimensionality
        coarse_points = initial_points.copy()
        
        # Apply adaptive local search with fewer iterations for coarse stage
        coarse_points = adaptive_local_search(coarse_points, max_iter=50)
        
        # Scale 2: Fine optimization with full resolution
        fine_points = coarse_points.copy()
        fine_points = adaptive_local_search(fine_points, max_iter=100)
        
        # Scale 3: Final refinement with detailed local search
        final_points = fine_points.copy()
        final_points = adaptive_local_search(final_points, max_iter=150)
        
        return final_points

    # Generate initial discrete placement
    initial_points = generate_discrete_initial_placement()
    
    # Perform multi-scale optimization
    optimized_points = multi_scale_optimization(initial_points)
    
    # Final refinement with additional local search
    final_points = adaptive_local_search(optimized_points, max_iter=100)
    
    # Evaluate and return the best among several attempts
    best_points = final_points
    best_ratio = compute_min_max_ratio(final_points)
    
    # Try a few more variations to ensure we haven't missed anything
    for attempt in range(3):
        np.random.seed(attempt + 1000)
        # Add small random perturbations to explore nearby solutions
        perturbed = final_points + np.random.normal(0, 0.005, final_points.shape)
        perturbed = np.clip(perturbed, 0, 1)
        
        # Local refine the perturbed version
        refined = adaptive_local_search(perturbed, max_iter=50)
        refined_ratio = compute_min_max_ratio(refined)
        
        if refined_ratio > best_ratio:
            best_ratio = refined_ratio
            best_points = refined.copy()
    
    return best_points

# EVOLVE-BLOCK-END