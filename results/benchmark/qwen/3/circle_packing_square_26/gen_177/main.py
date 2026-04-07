# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
from typing import Tuple
import math

# Global constants for quadratic optimization approach
MAX_ITERATIONS = 1000
OPTIMIZATION_TOLERANCE = 1e-6
INITIAL_GRID_DENSITY = 4
BARRIER_PARAMETER = 1e6
BOUNDARY_MARGIN = 0.01

def generate_initial_grid_points(n_circles: int, density: int = INITIAL_GRID_DENSITY) -> np.ndarray:
    """Generate initial points using a structured grid approach"""
    # Create a structured grid pattern with some randomness for variation
    grid_size = max(2, int(np.ceil(np.sqrt(n_circles))))
    
    # Create regular grid points
    points = []
    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) >= n_circles:
                break
            # Add some jitter to make it less regular
            x = (i + 0.5 + np.random.uniform(-0.2, 0.2)) / grid_size
            y = (j + 0.5 + np.random.uniform(-0.2, 0.2)) / grid_size
            # Ensure points stay within bounds
            x = np.clip(x, BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
            y = np.clip(y, BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
            points.append([x, y])
    
    # If we don't have enough points, add random ones
    if len(points) < n_circles:
        additional = n_circles - len(points)
        for _ in range(additional):
            x = np.random.uniform(BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
            y = np.random.uniform(BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
            points.append([x, y])
    
    return np.array(points[:n_circles])

def calculate_initial_radii(points: np.ndarray, n_circles: int) -> np.ndarray:
    """Calculate initial radii based on distance to nearest neighbors"""
    if n_circles <= 1:
        return np.full(n_circles, 0.05)
    
    # Compute pairwise distances
    distances = cdist(points, points, 'euclidean')
    
    # Set diagonal to large value to ignore self-distance
    np.fill_diagonal(distances, np.inf)
    
    # Find minimum distances to neighbors
    min_distances = np.min(distances, axis=1)
    
    # Calculate initial radii - inversely proportional to neighbor distances
    # but bounded by boundary constraints
    initial_radii = np.minimum(
        min_distances / 3.0,  # Half of min distance to nearest neighbor
        np.minimum(
            points[:, 0],  # Distance to left boundary
            np.minimum(
                1 - points[:, 0],  # Distance to right boundary
                np.minimum(
                    points[:, 1],  # Distance to bottom boundary
                    1 - points[:, 1]   # Distance to top boundary
                )
            )
        )
    )
    
    # Ensure minimum radius
    initial_radii = np.maximum(initial_radii, 0.005)
    
    # Add some randomness to break symmetry
    initial_radii *= (1 + np.random.normal(0, 0.1, n_circles))
    initial_radii = np.maximum(initial_radii, 0.005)
    
    return initial_radii

def create_quadratic_objective_and_constraints(n_circles: int, points: np.ndarray, 
                                             radii: np.ndarray, barrier_param: float = BARRIER_PARAMETER):
    """
    Create the quadratic optimization parameters for sum of radii with barrier constraints.
    """
    def objective(x):
        # x contains [x1, y1, r1, x2, y2, r2, ..., xn, yn, rn]
        # Extract radii
        radii_vals = x[n_circles*2::3]
        return -np.sum(radii_vals)  # Negative because we want to maximize
    
    def constraint_func(x):
        # x contains [x1, y1, r1, x2, y2, r2, ..., xn, yn, rn]
        # Extract positions and radii
        positions = x[:n_circles*2].reshape((-1, 2))
        radii_vals = x[n_circles*2::3]
        
        # Check boundary constraints
        boundary_violations = []
        for i in range(n_circles):
            x_pos, y_pos = positions[i]
            r_val = radii_vals[i]
            if x_pos - r_val < 0 or x_pos + r_val > 1 or y_pos - r_val < 0 or y_pos + r_val > 1:
                boundary_violations.append(1)  # Violation detected
            else:
                boundary_violations.append(0)
        
        # Check overlap constraints with barrier method
        overlap_violations = []
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                r1 = radii_vals[i]
                r2 = radii_vals[j]
                
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                overlap = max(0, r1 + r2 - distance)
                
                # Add barrier term for overlaps
                if overlap > 0:
                    # Apply barrier function: log(1/(overlap + epsilon))
                    overlap_violations.append(barrier_param / (overlap + 1e-10))
                else:
                    overlap_violations.append(0)
        
        return np.concatenate([boundary_violations, overlap_violations])
    
    # Create initial guess
    x0 = np.zeros(n_circles * 3)
    x0[:n_circles*2:2] = points[:, 0]  # x coordinates
    x0[1:n_circles*2:2] = points[:, 1]  # y coordinates
    x0[n_circles*2:] = radii  # radii
    
    return objective, constraint_func, x0

def solve_quadratic_optimization(n_circles: int, initial_points: np.ndarray, 
                               initial_radii: np.ndarray) -> np.ndarray:
    """
    Solve the quadratic optimization problem for circle packing.
    """
    # Create optimization variables
    x0 = np.zeros(n_circles * 3)
    x0[:n_circles*2:2] = initial_points[:, 0]  # x coordinates
    x0[1:n_circles*2:2] = initial_points[:, 1]  # y coordinates
    x0[n_circles*2:] = initial_radii  # radii
    
    # Define bounds (positions within unit square, radii positive)
    bounds = [(0, 1) if i % 3 < 2 else (1e-6, 0.5) for i in range(n_circles * 3)]
    
    # Create bounds for positions and radii separately
    bounds = []
    for i in range(n_circles):
        bounds.extend([(BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN),  # x bound
                      (BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN),  # y bound
                      (1e-6, 0.5)])  # radius bound
    
    # Objective function: maximize sum of radii (minimize negative sum)
    def obj_func(x):
        # Extract radii
        radii_vals = x[n_circles*2:]
        return -np.sum(radii_vals)
    
    # Constraints function
    def constraint_func(x):
        positions = x[:n_circles*2].reshape((-1, 2))
        radii_vals = x[n_circles*2:]
        
        # Boundary violations
        violations = []
        for i in range(n_circles):
            x_pos, y_pos = positions[i]
            r_val = radii_vals[i]
            # Boundary violations (penalty if outside bounds)
            if x_pos - r_val < 0:
                violations.append((0 - (x_pos - r_val))**2)
            if x_pos + r_val > 1:
                violations.append(((x_pos + r_val) - 1)**2)
            if y_pos - r_val < 0:
                violations.append((0 - (y_pos - r_val))**2)
            if y_pos + r_val > 1:
                violations.append(((y_pos + r_val) - 1)**2)
        
        # Overlap violations
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                r1 = radii_vals[i]
                r2 = radii_vals[j]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                overlap = max(0, r1 + r2 - distance)
                violations.append(overlap**2)
        
        return np.array(violations)
    
    # Create constraint dictionary
    constraints = {'type': 'ineq', 'fun': lambda x: np.array([0.0]) - constraint_func(x)}
    
    # Optimize
    try:
        result = minimize(obj_func, x0, method='SLSQP', bounds=bounds, 
                         constraints=constraints, tol=OPTIMIZATION_TOLERANCE, 
                         options={'maxiter': MAX_ITERATIONS})
        if result.success:
            optimal_x = result.x
            return optimal_x.reshape((n_circles, 3))
    except:
        pass
    
    # Fallback: return initial solution if optimization fails
    circles = np.zeros((n_circles, 3))
    circles[:, 0] = initial_points[:, 0]  # x
    circles[:, 1] = initial_points[:, 1]  # y
    circles[:, 2] = initial_radii          # r
    return circles

def geometric_refinement(circles: np.ndarray, max_iterations: int = 100) -> np.ndarray:
    """Refine solution using geometric optimization"""
    n_circles = len(circles)
    
    # Get initial configuration
    positions = circles[:, :2]
    radii = circles[:, 2]
    
    # Try to locally optimize by increasing radii where possible
    for iteration in range(max_iterations):
        improved = False
        
        # For each circle, try to increase radius while maintaining constraints
        for i in range(n_circles):
            x, y, r = circles[i]
            
            # Try to increase radius
            max_possible_r = min(
                x, 1-x, y, 1-y  # Boundary constraints
            )
            
            # Check overlap constraints with all other circles
            new_r = r
            for j in range(n_circles):
                if i != j:
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                    max_r_from_this_circle = distance - r2
                    max_possible_r = min(max_possible_r, max_r_from_this_circle)
            
            # Increase radius if beneficial and valid
            new_r = min(max_possible_r, r + 0.005)
            if new_r > r:
                circles[i, 2] = new_r
                improved = True
        
        # If no improvements made, break early
        if not improved:
            break
    
    return circles

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    n_circles = 26
    
    # Step 1: Generate initial grid points
    initial_points = generate_initial_grid_points(n_circles, INITIAL_GRID_DENSITY)
    
    # Step 2: Calculate initial radii
    initial_radii = calculate_initial_radii(initial_points, n_circles)
    
    # Step 3: Solve quadratic optimization problem
    circles = solve_quadratic_optimization(n_circles, initial_points, initial_radii)
    
    # Step 4: Apply geometric refinement
    refined_circles = geometric_refinement(circles)
    
    # Final validation and adjustment
    final_circles = np.zeros((n_circles, 3))
    
    for i in range(n_circles):
        x, y, r = refined_circles[i]
        # Ensure boundary constraints
        x = np.clip(x, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
        y = np.clip(y, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
        
        # Recalculate radii to ensure no overlaps
        # Check if this causes overlaps with others
        final_radii = []
        for j in range(n_circles):
            if i != j:
                x2, y2, r2 = refined_circles[j]
                distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                max_r = distance - r2
                if max_r > 0:
                    final_radii.append(max_r)
        
        if final_radii:
            # Use minimum of all constraints
            effective_radius = min(r, min(final_radii))
        else:
            effective_radius = r
            
        # Ensure effective radius is reasonable
        effective_radius = max(effective_radius, 0.001)
        effective_radius = min(effective_radius, 0.5)
        
        final_circles[i] = [x, y, effective_radius]
    
    return final_circles

# EVOLVE-BLOCK-END