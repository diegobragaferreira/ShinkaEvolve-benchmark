# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    n = 26
    circles = np.zeros((n, 3))
    
    # Step 1: Initialize using Voronoi-inspired spreading
    # Create a grid of potential positions and spread them
    positions = []
    grid_size = int(math.ceil(math.sqrt(n)))
    
    # Generate points on a grid with some randomness
    for i in range(grid_size):
        for j in range(grid_size):
            if len(positions) >= n:
                break
            x = (i + 0.5 + np.random.normal(0, 0.1)) / grid_size
            y = (j + 0.5 + np.random.normal(0, 0.1)) / grid_size
            # Ensure point is within bounds
            x = max(0.01, min(0.99, x))
            y = max(0.01, min(0.99, y))
            positions.append([x, y])
    
    # If we don't have enough points, add random ones
    while len(positions) < n:
        x = np.random.uniform(0.01, 0.99)
        y = np.random.uniform(0.01, 0.99)
        positions.append([x, y])
    
    positions = np.array(positions[:n])
    
    # Step 2: Initialize with small radii
    # Start with equal-sized circles, then optimize
    radii = np.ones(n) * 0.02
    
    # Step 3: Refinement using constrained optimization
    def objective(radii):
        return -np.sum(radii)  # Negative because we want to maximize
    
    def constraint_func(radii, i, j):
        # Distance constraint: circles must not overlap
        pos_i = positions[i]
        pos_j = positions[j]
        dist = np.linalg.norm(pos_i - pos_j)
        return dist - (radii[i] + radii[j])  # Should be >= 0
    
    def containment_constraint(radii, i):
        # Containment constraint: circle must fit in unit square
        pos = positions[i]
        return min(pos[0] - radii[i], pos[1] - radii[i], 1 - pos[0] - radii[i], 1 - pos[1] - radii[i])
    
    # Create bounds and constraints
    bounds = [(0.001, 0.4) for _ in range(n)]
    
    # Run optimization using scipy minimize
    try:
        # First, let's do a simpler approach with coordinate-based optimization
        optimized_radii = np.copy(radii)
        
        # Simple iterative improvement approach
        for iteration in range(100):
            improved = False
            # Try to increase all radii one by one
            for i in range(n):
                original_radius = optimized_radii[i]
                # Find maximum possible radius for this circle
                max_radius = min(
                    positions[i][0], 
                    positions[i][1], 
                    1 - positions[i][0], 
                    1 - positions[i][1]
                )
                
                # Check if other circles allow larger radius
                for j in range(n):
                    if i != j:
                        # Distance to other circle center
                        dist = np.linalg.norm(positions[i] - positions[j])
                        # Maximum radius this circle can have without overlapping
                        max_radius = min(max_radius, dist - optimized_radii[j])
                
                # Limit the radius to be positive
                max_radius = max(0.001, max_radius)
                
                if max_radius > optimized_radii[i]:
                    optimized_radii[i] = max_radius
                    improved = True
            
            # If no improvements were made, we're done
            if not improved:
                break
                
        # Final refinement using projection onto constraints
        final_positions = positions.copy()
        final_radii = optimized_radii.copy()
        
        # Apply hard constraints
        for i in range(n):
            # Ensure containment
            max_containment_radius = min(
                final_positions[i][0],
                final_positions[i][1],
                1 - final_positions[i][0],
                1 - final_positions[i][1]
            )
            
            # Ensure no overlaps with others
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(final_positions[i] - final_positions[j])
                    max_overlap_radius = dist - final_radii[j]
                    max_containment_radius = min(max_containment_radius, max_overlap_radius)
                    
            final_radii[i] = min(final_radii[i], max_containment_radius)
            final_radii[i] = max(final_radii[i], 0.001)
            
        # Store results
        circles = np.column_stack([final_positions, final_radii])
        
        # Verify constraints
        total_sum = np.sum(circles[:, 2])
        if total_sum < 0.5:  # If not very good, try another approach
            # Fall back to a more systematic approach
            circles = fallback_strategy()
        else:
            # Validate the configuration
            if not validate_circles(circles):
                circles = fallback_strategy()
                
    except Exception as e:
        # If anything goes wrong, fall back to grid approach
        circles = fallback_strategy()
    
    return circles

def validate_circles(circles):
    """Validate that all circles satisfy constraints."""
    n = circles.shape[0]
    
    # Check containment
    for i in range(n):
        x, y, r = circles[i]
        if r > x or r > y or r > (1 - x) or r > (1 - y):
            return False
    
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if distance < (r1 + r2):
                return False
    
    return True

def fallback_strategy():
    """A robust fallback strategy for circle placement."""
    np.random.seed(42)
    n = 26
    circles = np.zeros((n, 3))
    
    # Create a more careful grid-based initialization
    positions = []
    grid_size = 5  # 5x5 grid gives 25 points, plus one extra
    
    # Fill grid with spacing
    spacing = 1.0 / (grid_size + 1)
    for i in range(grid_size):
        for j in range(grid_size):
            x = (i + 1) * spacing
            y = (j + 1) * spacing
            positions.append([x, y])
    
    # Add one more point
    positions.append([0.5, 0.5])
    positions = np.array(positions[:n])
    
    # Assign initial radii based on proximity to edges
    radii = np.zeros(n)
    for i in range(n):
        pos = positions[i]
        min_dist_to_edge = min(pos[0], pos[1], 1-pos[0], 1-pos[1])
        # Radius is limited by distance to nearest edge and neighbors
        radii[i] = min_dist_to_edge * 0.8
        
    # Try to resolve overlaps through shrinking
    for _ in range(100):  # Allow multiple iterations to resolve overlaps
        # Calculate pairwise distances between centers
        dist_matrix = cdist(positions, positions)
        changed = False
        
        for i in range(n):
            # Find minimum distance to neighbors (excluding self)
            neighbor_distances = dist_matrix[i]
            neighbor_distances[i] = float('inf')  # Exclude self
            min_neighbor_distance = np.min(neighbor_distances)
            
            # Compute max allowable radius
            max_radius = min(
                positions[i][0], 
                positions[i][1], 
                1 - positions[i][0], 
                1 - positions[i][1],
                min_neighbor_distance / 2
            )
            
            # Update radius if needed
            if max_radius > 0.001 and radii[i] > max_radius:
                radii[i] = max_radius
                changed = True
        
        if not changed:
            break
    
    # Ensure all radii are positive and reasonable
    radii = np.maximum(radii, 0.001)
    
    # Final cleanup to ensure no overlaps
    for i in range(n):
        # Make sure we don't exceed distance to neighbors
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(positions[i] - positions[j])
                max_radius = dist - radii[j]  # max radius to avoid overlap
                if max_radius > 0:
                    radii[i] = min(radii[i], max_radius)
        radii[i] = max(radii[i], 0.001)
    
    circles = np.column_stack([positions, radii])
    
    return circles

# EVOLVE-BLOCK-END
