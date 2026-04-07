# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import random
from typing import Tuple, List
import math

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def validate_circles(circles: np.ndarray) -> bool:
    """
    Validates that all circles are within bounds and don't overlap.
    Optimized with early termination and efficient spatial queries.
    """
    n = len(circles)

    # Check containment constraints first
    for i in range(n):
        x, y, r = circles[i]
        if x < r or x > 1 - r or y < r or y > 1 - r:
            return False

    # Check overlap constraints using KDTree for efficiency
    points = circles[:, :2]  # Get (x, y) coordinates
    tree = cKDTree(points)

    # For each circle, check overlap with others
    for i in range(n):
        x1, y1, r1 = circles[i]

        # Find nearby circles (within 2*(r1+r2) distance)
        nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))

        # Check overlap with each nearby circle
        for j in nearby_indices:
            if i != j:
                x2, y2, r2 = circles[j]
                distance_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_distance_sq = (r1 + r2)**2

                if distance_sq < min_distance_sq:
                    return False

    return True

def calculate_sum_radii(circles: np.ndarray) -> float:
    """Calculate the sum of all radii"""
    return np.sum(circles[:, 2])

def create_initial_solution() -> np.ndarray:
    """Create an initial solution using a deterministic hexagonal packing approach"""
    # Start with a regular hexagonal grid pattern which tends to be efficient
    # We'll place circles in a hexagonal arrangement then slightly perturb for optimization
    
    # Hexagonal packing parameters
    n_circles = 26
    grid_size = int(np.ceil(np.sqrt(n_circles)))
    
    # Create hexagonal grid
    circles = []
    hex_radius = 0.1  # Starting guess for hex packing
    hex_height = hex_radius * np.sqrt(3)
    hex_width = hex_radius * 1.5
    
    # Generate hexagonal grid
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if idx >= n_circles:
                break
                
            # Hexagonal offset pattern
            offset = (j % 2) * 0.5
            x = (i + offset) * hex_width
            y = j * hex_height
            
            # Adjust to fit in unit square
            if x > 1 - hex_radius:
                x = 1 - hex_radius
            if y > 1 - hex_radius:
                y = 1 - hex_radius
                
            # Ensure we're within bounds
            if x >= hex_radius and y >= hex_radius:
                circles.append([x, y, hex_radius])
                idx += 1
                
        if idx >= n_circles:
            break
    
    # Fill remaining spots randomly if needed
    while len(circles) < n_circles:
        x = np.random.uniform(0.01, 0.99)
        y = np.random.uniform(0.01, 0.99)
        r = np.random.uniform(0.005, 0.05)
        circles.append([x, y, r])
    
    return np.array(circles)

def compute_constraints(circles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute constraint violations for all pairs of circles"""
    n = len(circles)
    constraints = []
    
    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        # Distance to boundaries
        dist_to_left = x - r
        dist_to_right = 1 - x - r
        dist_to_bottom = y - r
        dist_to_top = 1 - y - r
        
        # Violations (negative means violation)
        if dist_to_left < 0:
            constraints.append(dist_to_left)
        if dist_to_right < 0:
            constraints.append(dist_to_right)
        if dist_to_bottom < 0:
            constraints.append(dist_to_bottom)
        if dist_to_top < 0:
            constraints.append(dist_to_top)
    
    # Check overlap constraints
    points = circles[:, :2]
    tree = cKDTree(points)
    
    for i in range(n):
        x1, y1, r1 = circles[i]
        nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))
        
        for j in nearby_indices:
            if i != j:
                x2, y2, r2 = circles[j]
                distance_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_distance_sq = (r1 + r2)**2
                
                # Overlap violation (negative means overlap)
                overlap_violation = min_distance_sq - distance_sq
                if overlap_violation > 0:
                    constraints.append(-overlap_violation)
    
    return np.array(constraints)

def penalty_function(circles: np.ndarray, penalty_weight: float = 1000.0) -> float:
    """Calculate penalty based on constraint violations"""
    constraints = compute_constraints(circles)
    # Only penalize negative constraints (violations)
    violation_penalty = np.sum(np.maximum(0, -constraints))
    return penalty_weight * violation_penalty

def objective_with_penalty(circles: np.ndarray, penalty_weight: float = 1000.0) -> float:
    """Objective function with penalty for constraint violations"""
    # Negative because we want to maximize sum of radii, but optimizer minimizes
    objective_value = -calculate_sum_radii(circles)
    
    # Add penalty term
    penalty = penalty_function(circles, penalty_weight)
    
    return objective_value + penalty

def project_to_feasible(circles: np.ndarray) -> np.ndarray:
    """Project circles to satisfy constraints via geometric operations"""
    projected = circles.copy()
    
    # First, fix boundary constraints
    for i in range(len(projected)):
        x, y, r = projected[i]
        # Clamp to boundary constraints
        projected[i, 0] = np.clip(x, r, 1 - r)
        projected[i, 1] = np.clip(y, r, 1 - r)
    
    # Then fix overlap constraints using a simple repulsion approach
    points = projected[:, :2]
    tree = cKDTree(points)
    
    # Try to resolve overlaps iteratively
    for _ in range(5):
        overlap_found = False
        for i in range(len(projected)):
            x1, y1, r1 = projected[i]
            nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))
            
            for j in nearby_indices:
                if i != j:
                    x2, y2, r2 = projected[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    min_distance = r1 + r2
                    
                    if distance < min_distance:
                        overlap_found = True
                        # Push circles apart
                        if distance > 0.001:
                            dx = (x1 - x2) / distance
                            dy = (y1 - y2) / distance
                            move_amount = (min_distance - distance) * 0.5
                            
                            # Move both circles away from each other
                            projected[i, 0] += dx * move_amount * 0.5
                            projected[i, 1] += dy * move_amount * 0.5
                            projected[j, 0] -= dx * move_amount * 0.5
                            projected[j, 1] -= dy * move_amount * 0.5
                            
                            # Clamp to bounds
                            projected[i, 0] = np.clip(projected[i, 0], r1, 1 - r1)
                            projected[i, 1] = np.clip(projected[i, 1], r1, 1 - r1)
                            projected[j, 0] = np.clip(projected[j, 0], r2, 1 - r2)
                            projected[j, 1] = np.clip(projected[j, 1], r2, 1 - r2)
        
        if not overlap_found:
            break
            
    return projected

def quadratic_optimization_approach() -> np.ndarray:
    """Use quadratic programming approach with gradient-based optimization"""
    # Start with a good initial solution
    circles = create_initial_solution()
    
    # Create bounds for optimization
    bounds = []
    for i in range(len(circles)):
        # Bounds for x, y, r
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Use L-BFGS-B for optimization with bounds
    def objective(params):
        # Reshape params back into circles format (x, y, r)
        circles_local = circles.copy()
        for i in range(len(circles_local)):
            circles_local[i] = [params[i*3], params[i*3+1], params[i*3+2]]
        
        return objective_with_penalty(circles_local, penalty_weight=1000.0)
    
    def grad_objective(params):
        # Simple finite difference gradient estimation
        eps = 1e-6
        grad = np.zeros_like(params)
        base_obj = objective(params)
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            grad[i] = (objective(params_plus) - base_obj) / eps
            
        return grad
    
    # Flatten initial circles for optimization
    initial_params = []
    for x, y, r in circles:
        initial_params.extend([x, y, r])
    
    # Optimize using scipy minimize with L-BFGS-B method
    result = minimize(
        objective,
        initial_params,
        method='L-BFGS-B',
        bounds=bounds,
        jac=grad_objective,
        options={'maxiter': 1000, 'ftol': 1e-8}
    )
    
    # Convert back to circles format
    optimized_circles = circles.copy()
    for i in range(len(optimized_circles)):
        optimized_circles[i] = [result.x[i*3], result.x[i*3+1], result.x[i*3+2]]
    
    # Project to feasible region
    optimized_circles = project_to_feasible(optimized_circles)
    
    return optimized_circles

def hybrid_optimization() -> np.ndarray:
    """Combine global and local optimization strategies"""
    best_circles = None
    best_sum = -np.inf
    
    # Run multiple optimization attempts with different starting points
    for attempt in range(10):
        try:
            # Create a slightly different initial solution each time
            circles = create_initial_solution()
            
            # Add small random perturbation to initial solution
            for i in range(len(circles)):
                circles[i, 0] += np.random.normal(0, 0.01)
                circles[i, 1] += np.random.normal(0, 0.01)
                circles[i, 2] += np.random.normal(0, 0.005)
            
            # Optimize using our quadratic approach
            optimized_circles = quadratic_optimization_approach()
            
            # Validate and evaluate
            if validate_circles(optimized_circles):
                current_sum = calculate_sum_radii(optimized_circles)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = optimized_circles.copy()
                    
        except Exception as e:
            continue
    
    # If we haven't found anything, fall back to a decent initial solution
    if best_circles is None:
        best_circles = create_initial_solution()
    
    return best_circles

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Use our advanced quadratic optimization approach
    return hybrid_optimization()

# EVOLVE-BLOCK-END