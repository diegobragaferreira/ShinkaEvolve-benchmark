# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions - perimeter = 4, so width + height = 2
    # Using a 1:1 ratio for simplicity (square-like shape)
    rect_width = 1.0
    rect_height = 1.0
    
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    n_circles = 21
    
    # Objective function to maximize (negative because minimize)
    def objective(params):
        # Reshape params into circles array
        circles = params.reshape(-1, 3)
        # Return negative sum of radii to maximize sum of radii
        return -np.sum(circles[:, 2])
    
    # Constraint functions
    def boundary_constraints(params):
        circles = params.reshape(-1, 3)
        constraints = []
        
        # Boundary constraints: x - r >= 0, x + r <= width, y - r >= 0, y + r <= height
        for i in range(n_circles):
            x, y, r = circles[i]
            # x - r >= 0
            constraints.append(x - r)
            # x + r <= width
            constraints.append(rect_width - x - r)
            # y - r >= 0
            constraints.append(y - r)
            # y + r <= height
            constraints.append(rect_height - y - r)
            
        return np.array(constraints)
    
    def overlap_constraints(params):
        circles = params.reshape(-1, 3)
        constraints = []
        
        # Overlap constraints: distance between centers >= sum of radii
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                
                # Distance between centers
                dx = x1 - x2
                dy = y1 - y2
                dist = np.sqrt(dx*dx + dy*dy)
                
                # Constraint: dist >= r1 + r2 (or equivalently: r1 + r2 - dist <= 0)
                # We add a small epsilon to make it work with inequality constraints
                constraints.append(r1 + r2 - dist + 1e-8)
                
        return np.array(constraints)
    
    # Create bounds for parameters (x, y, r) for each circle
    bounds = []
    for i in range(n_circles):
        # x bounds
        bounds.append((0.01, rect_width - 0.01))
        # y bounds  
        bounds.append((0.01, rect_height - 0.01))
        # r bounds
        bounds.append((0.01, 0.3))  # Reasonable upper bound
        
    # Initial guess - start with a random configuration
    initial_params = []
    for i in range(n_circles):
        # Random positions within bounds
        x = np.random.uniform(0.05, rect_width - 0.05)
        y = np.random.uniform(0.05, rect_height - 0.05)
        # Radius - smaller than the smallest dimension, with some margin
        r = np.random.uniform(0.02, 0.15)
        initial_params.extend([x, y, r])
    
    # Apply simple force-based relaxation first
    def force_relaxation(params, iterations=1000):
        circles = params.reshape(-1, 3)
        learning_rate = 0.01
        damping_factor = 0.99
        
        for _ in range(iterations):
            forces = np.zeros_like(circles)
            
            # Compute forces between overlapping circles
            for i in range(n_circles):
                for j in range(i+1, n_circles):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    
                    dx = x1 - x2
                    dy = y1 - y2
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist < r1 + r2 and dist > 0:
                        # Repulsion force
                        force_magnitude = (r1 + r2 - dist) / (dist + 1e-8)
                        forces[i, 0] += force_magnitude * dx / dist
                        forces[i, 1] += force_magnitude * dy / dist
                        forces[j, 0] -= force_magnitude * dx / dist
                        forces[j, 1] -= force_magnitude * dy / dist
            
            # Apply forces with boundary constraints
            for i in range(n_circles):
                # Boundary repulsion
                x, y, r = circles[i]
                if x - r < 0.01:
                    forces[i, 0] += 0.1 * (0.01 - (x - r))
                elif x + r > rect_width - 0.01:
                    forces[i, 0] -= 0.1 * ((x + r) - (rect_width - 0.01))
                
                if y - r < 0.01:
                    forces[i, 1] += 0.1 * (0.01 - (y - r))
                elif y + r > rect_height - 0.01:
                    forces[i, 1] -= 0.1 * ((y + r) - (rect_height - 0.01))
            
            # Update positions
            circles[:, 0] += learning_rate * forces[:, 0]
            circles[:, 1] += learning_rate * forces[:, 1]
            
            # Apply bounds after update
            circles[:, 0] = np.clip(circles[:, 0], 0.01, rect_width - 0.01)
            circles[:, 1] = np.clip(circles[:, 1], 0.01, rect_height - 0.01)
            
            # Reduce learning rate
            learning_rate *= damping_factor
            
        return circles.flatten()
    
    # First apply force relaxation to get a better starting point
    relaxed_params = force_relaxation(np.array(initial_params))
    
    # Define constraints
    # Boundary constraints (g(x) >= 0)
    boundary_con = {'type': 'ineq', 'fun': lambda x: boundary_constraints(x)}
    
    # Overlap constraints (g(x) <= 0)
    overlap_con = {'type': 'ineq', 'fun': lambda x: overlap_constraints(x)}
    
    # Set up constraints list
    constraints = [boundary_con, overlap_con]
    
    # Optimize using SLSQP method
    try:
        result = minimize(objective, 
                         relaxed_params,
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints,
                         options={'maxiter': 1000}, 
                         tol=1e-6)
        
        if result.success:
            final_circles = result.x.reshape(-1, 3)
            # Ensure all radii are positive
            final_circles[:, 2] = np.maximum(final_circles[:, 2], 0.01)
            return final_circles
    except Exception as e:
        pass
    
    # Fallback to the relaxed configuration
    return relaxed_params.reshape(-1, 3)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
