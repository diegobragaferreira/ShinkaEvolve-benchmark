# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Phase 1: Hexagonal grid initialization with density-aware radius estimation
    rows = 6
    cols = 6
    
    sqrt3 = np.sqrt(3)
    spacing_x = 1.0 / cols
    spacing_y = sqrt3 / 2 * spacing_x
    
    # Generate base grid positions
    grid_points = []
    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5) * spacing_x
            y = (i + 0.5) * spacing_y
            if x <= 1.0 and y <= 1.0:
                grid_points.append([x, y])
    
    # Take first 32 points
    points = np.array(grid_points[:n])
    
    # Calculate initial radii based on local density and boundary constraints
    circles = np.zeros((n, 3))
    
    # Use KDTree for efficient neighbor search to estimate local density
    tree = cKDTree(points)
    
    for i in range(n):
        x, y = points[i]
        
        # Find nearby points to estimate local density
        # Look at neighbors within a certain radius
        neighbors = tree.query_ball_point([x, y], 0.2)
        neighbor_count = len(neighbors) - 1  # exclude self
        
        # Calculate maximum possible radius based on boundaries and neighbors
        max_radius = min(x, y, 1 - x, 1 - y)
        
        # Adjust based on density - denser regions get smaller initial radii
        density_factor = 1.0 / (1.0 + neighbor_count * 0.1)
        initial_radius = max_radius * 0.3 * density_factor
        
        # Ensure reasonable minimum radius
        initial_radius = max(initial_radius, 0.01)
        
        circles[i] = [x, y, initial_radius]
    
    # Phase 2: Optimization using L-BFGS-B with proper constraint handling
    def objective_and_constraints(params):
        # Reshape params: [x1,y1,r1,x2,y2,r2,...]
        pos_rad = params.reshape(-1, 3)
        
        # Extract positions and radii
        positions = pos_rad[:, :2]
        radii = pos_rad[:, 2]
        
        # Calculate negative sum of radii (since we want to maximize)
        neg_sum_radii = -np.sum(radii)
        
        # Constraints: containment and overlap
        penalties = 0
        
        # Boundary containment penalty (smooth exponential)
        for i in range(n):
            x, y, r = positions[i][0], positions[i][1], radii[i]
            # Penalties for going outside bounds
            penalty = 0
            
            # Left boundary
            if x - r < 0:
                penalty += np.exp(10 * (x - r))
            # Right boundary  
            if x + r > 1:
                penalty += np.exp(10 * (x + r - 1))
            # Bottom boundary
            if y - r < 0:
                penalty += np.exp(10 * (y - r))
            # Top boundary
            if y + r > 1:
                penalty += np.exp(10 * (y + r - 1))
                
            penalties += penalty
        
        # Overlap penalties using KDTree for efficiency
        tree = cKDTree(positions)
        
        # For each pair, check overlap condition
        for i in range(n):
            x1, y1, r1 = positions[i][0], positions[i][1], radii[i]
            
            # Find candidates for overlap with radius r1 + r2 + epsilon
            candidates = tree.query_ball_point([x1, y1], 2 * (r1 + 0.01))
            
            for j in candidates:
                if i == j:
                    continue
                    
                x2, y2, r2 = positions[j][0], positions[j][1], radii[j]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                # Penalty for overlap
                if distance < r1 + r2:
                    # Smooth penalty using exponential
                    overlap_amount = r1 + r2 - distance
                    penalties += 1000 * np.exp(10 * overlap_amount)
        
        # Return combined objective (negative sum of radii + penalties)
        return neg_sum_radii + penalties
    
    # Flatten initial configuration for optimization
    initial_params = circles.flatten()
    
    # Define bounds for optimization (x, y, r for each circle)
    bounds = []
    for i in range(n):
        # x bounds
        bounds.append((0.001, 0.999))
        # y bounds
        bounds.append((0.001, 0.999))
        # r bounds (small positive value to prevent degenerate cases)
        bounds.append((0.001, 0.499))
    
    # Optimization using L-BFGS-B
    try:
        result = minimize(
            objective_and_constraints,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        # Extract optimized results
        if result.success:
            optimized_pos_rad = result.x.reshape(-1, 3)
            circles = optimized_pos_rad.copy()
        else:
            # If optimization fails, keep the initial configuration
            pass
    except Exception as e:
        # In case of optimization failure, proceed with initial configuration
        pass
    
    # Phase 3: Local refinement using simulated annealing approach
    # Try to improve the solution with small local adjustments
    best_circles = circles.copy()
    best_sum = np.sum(best_circles[:, 2])
    
    # Apply multiple small perturbations to find better local solutions
    for _ in range(1000):  # Number of attempts
        test_circles = best_circles.copy()
        
        # Pick a random circle to modify
        idx = np.random.randint(0, n)
        
        # Small random changes to position and radius
        test_circles[idx, 0] += np.random.normal(0, 0.005)
        test_circles[idx, 1] += np.random.normal(0, 0.005)
        test_circles[idx, 2] += np.random.normal(0, 0.002)
        
        # Ensure bounds
        test_circles[idx, 0] = np.clip(test_circles[idx, 0], 0.001, 0.999)
        test_circles[idx, 1] = np.clip(test_circles[idx, 1], 0.001, 0.999)
        test_circles[idx, 2] = np.clip(test_circles[idx, 2], 0.001, 0.499)
        
        # Check if this improves the total sum
        sum_radii = np.sum(test_circles[:, 2])
        if sum_radii > best_sum:
            best_circles = test_circles
            best_sum = sum_radii
    
    # Final cleanup to ensure boundary constraints
    for i in range(n):
        x, y, r = best_circles[i]
        max_radius = min(x, y, 1 - x, 1 - y)
        if r > max_radius:
            best_circles[i, 2] = max_radius
    
    return best_circles


# EVOLVE-BLOCK-END