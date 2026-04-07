# EVOLVE-BLOCK-START
import numpy as np
import math
from typing import Tuple, List
import time

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses gradient-based optimization with smooth constraint approximation.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    
    # Initialize 26 circles with random positions and small radii
    n = 26
    circles = np.zeros((n, 3))
    
    # Initialize with random positions and small radii
    for i in range(n):
        circles[i] = [
            np.random.uniform(0.1, 0.9),
            np.random.uniform(0.1, 0.9),
            np.random.uniform(0.01, 0.1)
        ]
    
    # Smooth overlap constraint using error function approximation
    def smooth_overlap_constraint(xi, yi, ri, xj, yj, rj, epsilon=1e-3):
        """Smooth approximation of overlap constraint with gradient"""
        dx = xi - xj
        dy = yi - yj
        distance = math.sqrt(dx*dx + dy*dy)
        # Smooth transition around the boundary
        return 0.5 * (1 - math.erf((distance - ri - rj) / epsilon))
    
    # Compute total radius gradient
    def compute_radius_gradient(circles):
        """Gradient of sum of radii w.r.t. circle parameters"""
        grad = np.zeros_like(circles)
        # Gradient of sum of radii is just unit vectors for radii components
        grad[:, 2] = 1.0  # d/dr_sum = 1 for each radius
        return grad
    
    # Compute overlap penalty gradient
    def compute_overlap_gradient(circles):
        """Gradient of smooth overlap penalty"""
        n = len(circles)
        grad = np.zeros_like(circles)
        epsilon = 1e-3
        
        # For each pair of circles
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                
                dx = xi - xj
                dy = yi - yj
                distance = max(epsilon, math.sqrt(dx*dx + dy*dy))
                
                # Gradient of the smooth overlap function
                diff = distance - ri - rj
                erf_val = math.erf(diff / epsilon)
                exp_term = math.exp(-(diff/epsilon)**2)
                
                # Derivative of smooth overlap constraint
                overlap_grad = -0.5 * (1 - erf_val) * exp_term * (2/epsilon) / distance
                
                # Apply to both circles
                grad[i, 0] += overlap_grad * dx
                grad[i, 1] += overlap_grad * dy
                grad[i, 2] += overlap_grad
                
                grad[j, 0] -= overlap_grad * dx
                grad[j, 1] -= overlap_grad * dy
                grad[j, 2] += overlap_grad
        
        return grad
    
    # Compute boundary penalty gradient
    def compute_boundary_gradient(circles):
        """Gradient of boundary penalty"""
        grad = np.zeros_like(circles)
        
        # Penalty for boundary violations
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Penalties for being too close to boundaries
            penalty_x1 = max(0, r - x)  # Too close to left
            penalty_x2 = max(0, x + r - 1)  # Too close to right
            penalty_y1 = max(0, r - y)  # Too close to bottom
            penalty_y2 = max(0, y + r - 1)  # Too close to top
            
            # Gradients of penalties
            grad[i, 0] += 2 * penalty_x1 - 2 * penalty_x2  # d/dx
            grad[i, 1] += 2 * penalty_y1 - 2 * penalty_y2  # d/dy
            grad[i, 2] += 2 * penalty_x1 + 2 * penalty_x2 + 2 * penalty_y1 + 2 * penalty_y2  # d/dr
        
        return grad
    
    # Project circles to stay within valid bounds
    def project_to_bounds(circles):
        """Project circles to ensure they fit in unit square"""
        result = circles.copy()
        for i in range(len(result)):
            x, y, r = result[i]
            # Ensure circle fits in unit square
            max_radius = min(x, 1-x, y, 1-y)
            r = min(r, max_radius)
            r = max(0.001, r)
            
            # Clamp coordinates to valid range
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            
            result[i] = [x, y, r]
        return result
    
    # Line search to find optimal step size
    def line_search(circles, direction, max_alpha=1.0):
        """Simple backtracking line search"""
        alpha = max_alpha
        current_radius = np.sum(circles[:, 2])
        
        # Try decreasing step sizes until improvement
        while alpha > 1e-8:
            test_circles = circles + alpha * direction
            test_circles = project_to_bounds(test_circles)
            test_radius = np.sum(test_circles[:, 2])
            
            if test_radius > current_radius * 1.001:  # Allow small decrease for stability
                return alpha
            alpha *= 0.5
            
        return alpha
    
    # Main optimization loop
    max_iter = 2000
    tolerance = 1e-6
    
    for iteration in range(max_iter):
        # Compute gradients
        radius_grad = compute_radius_gradient(circles)
        overlap_grad = compute_overlap_gradient(circles)
        boundary_grad = compute_boundary_gradient(circles)
        
        # Combine gradients (weighted appropriately)
        total_grad = radius_grad - 0.1 * overlap_grad - 0.05 * boundary_grad
        
        # Apply line search for step size
        step_size = line_search(circles, total_grad)
        
        # Update circles
        circles = circles + step_size * total_grad
        
        # Project to bounds
        circles = project_to_bounds(circles)
        
        # Early stopping based on improvement
        if iteration > 10 and iteration % 50 == 0:
            current_radius = np.sum(circles[:, 2])
            print(f"Iteration {iteration}: Sum of radii = {current_radius}")
            
            # Check if improvement is negligible
            if iteration > 100:
                # Simple check for stagnation
                pass
    
    # Final projection
    circles = project_to_bounds(circles)
    
    return circles

# EVOLVE-BLOCK-END