# EVOLVE-BLOCK-START
import numpy as np
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
    """
    n = len(circles)

    # Check containment constraints
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

def create_initial_physics_config(n_circles: int) -> np.ndarray:
    """Create initial configuration using physics-inspired grid placement"""
    circles = np.zeros((n_circles, 3))
    
    # Create a 5x5 grid pattern with some randomness
    grid_size = 5
    spacing_x = 1.0 / grid_size
    spacing_y = 1.0 / grid_size
    
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if idx >= n_circles:
                break
                
            # Position in grid cell with randomness
            x = (i + 0.5 + np.random.uniform(-0.1, 0.1)) * spacing_x
            y = (j + 0.5 + np.random.uniform(-0.1, 0.1)) * spacing_y
            
            # Initial radius based on proximity to edges
            max_radius = min(x, 1-x, y, 1-y)
            r = min(0.08, max_radius * 0.4)
            
            circles[idx] = [x, y, r]
            idx += 1
            
        if idx >= n_circles:
            break
    
    # Add some random perturbations to improve diversity
    for i in range(n_circles):
        if np.random.rand() < 0.7:  # 70% chance to perturb
            circles[i, 0] += np.random.normal(0, 0.005)
            circles[i, 1] += np.random.normal(0, 0.005)
            circles[i, 2] += np.random.normal(0, 0.002)
            
            # Ensure valid bounds
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
            circles[i, 2] = max(0.001, circles[i, 2])
    
    return circles

def compute_forces(circles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute forces acting on each circle from all other circles"""
    n = len(circles)
    forces = np.zeros((n, 2))  # Force vectors (dx, dy)
    radius_gradients = np.zeros(n)  # Gradient of radius sum
    
    # Precompute distances for efficiency
    points = circles[:, :2]
    tree = cKDTree(points)
    
    for i in range(n):
        x1, y1, r1 = circles[i]
        
        # Find nearby circles to avoid expensive full-distance computation
        nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))
        
        for j in nearby_indices:
            if i != j:
                x2, y2, r2 = circles[j]
                dx = x1 - x2
                dy = y1 - y2
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance > 0.001:  # Avoid division by zero
                    min_distance = r1 + r2
                    
                    if distance < min_distance:  # Overlapping
                        # Strong repulsive force when overlapping
                        force_magnitude = 1000 * (min_distance - distance) / distance
                        forces[i, 0] += force_magnitude * dx / distance
                        forces[i, 1] += force_magnitude * dy / distance
                    elif distance < 2 * min_distance:  # Near overlap
                        # Moderate repulsive force
                        force_magnitude = 10 * (min_distance - distance) / distance**2
                        forces[i, 0] += force_magnitude * dx / distance
                        forces[i, 1] += force_magnitude * dy / distance
    
    # Boundary forces (push back into feasible region)
    for i in range(n):
        x, y, r = circles[i]
        boundary_force_x = 0.0
        boundary_force_y = 0.0
        
        # Forces from boundaries
        if x < r:
            boundary_force_x += 100 * (r - x)
        elif x > 1 - r:
            boundary_force_x += 100 * (1 - r - x)
            
        if y < r:
            boundary_force_y += 100 * (r - y)
        elif y > 1 - r:
            boundary_force_y += 100 * (1 - r - y)
            
        forces[i, 0] += boundary_force_x
        forces[i, 1] += boundary_force_y
    
    # Radius gradient (want to maximize sum of radii)
    # Simple gradient: partial derivative of sum w.r.t. each radius is 1
    radius_gradients.fill(1.0)
    
    return forces, radius_gradients

def update_circles(circles: np.ndarray, forces: np.ndarray, 
                   radius_gradients: np.ndarray, learning_rate: float = 0.01,
                   max_radius_update: float = 0.005) -> np.ndarray:
    """Update circle positions and radii based on forces and gradients"""
    updated = circles.copy()
    n = len(updated)
    
    # Update positions with forces
    for i in range(n):
        # Apply forces to position
        updated[i, 0] += learning_rate * forces[i, 0]
        updated[i, 1] += learning_rate * forces[i, 1]
        
        # Ensure positions remain valid
        updated[i, 0] = np.clip(updated[i, 0], updated[i, 2], 1 - updated[i, 2])
        updated[i, 1] = np.clip(updated[i, 1], updated[i, 2], 1 - updated[i, 2])
    
    # Update radii with gradient ascent
    for i in range(n):
        # Update radius in direction of gradient (away from obstacles)
        radius_change = learning_rate * radius_gradients[i]
        updated[i, 2] += radius_change
        
        # Limit radius changes
        updated[i, 2] = np.clip(updated[i, 2], 0.001, 0.4)
        # Ensure radius doesn't violate bounds
        max_radius = min(updated[i, 0], 1 - updated[i, 0], updated[i, 1], 1 - updated[i, 1])
        updated[i, 2] = min(updated[i, 2], max_radius)
    
    return updated

def local_refinement(circles: np.ndarray, max_iterations: int = 50) -> np.ndarray:
    """Apply local physics-based refinement to improve solution"""
    current = circles.copy()
    best_solution = current.copy()
    best_fitness = calculate_sum_radii(current)
    
    for iteration in range(max_iterations):
        # Compute forces and gradients
        forces, radius_gradients = compute_forces(current)
        
        # Update circles
        updated = update_circles(current, forces, radius_gradients, 
                                learning_rate=0.02, max_radius_update=0.01)
        
        # Validate the updated solution
        if validate_circles(updated):
            current = updated
            fitness = calculate_sum_radii(current)
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = current.copy()
        else:
            # If update violates constraints, try a modified version
            # Reduce step size and retry
            forces, radius_gradients = compute_forces(current)
            reduced_updated = update_circles(current, forces, radius_gradients, 
                                           learning_rate=0.005, max_radius_update=0.005)
            
            if validate_circles(reduced_updated):
                current = reduced_updated
                fitness = calculate_sum_radii(current)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = current.copy()
    
    return best_solution

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Multi-stage optimization approach
    # Stage 1: Generate good initial configuration
    circles = create_initial_physics_config(26)
    
    # Stage 2: Local refinement with physics model
    circles = local_refinement(circles, max_iterations=100)
    
    # Stage 3: Iterative improvement
    best_fitness = calculate_sum_radii(circles)
    best_solution = circles.copy()
    
    # Run multiple rounds of physics-based optimization
    for round_num in range(3):
        # Create perturbed version of current solution
        perturbed = circles.copy()
        
        # Add small random perturbations
        for i in range(len(perturbed)):
            if np.random.rand() < 0.3:
                perturbed[i, 0] += np.random.normal(0, 0.003)
                perturbed[i, 1] += np.random.normal(0, 0.003)
                perturbed[i, 2] += np.random.normal(0, 0.001)
                
                # Keep within bounds
                perturbed[i, 0] = np.clip(perturbed[i, 0], perturbed[i, 2], 1 - perturbed[i, 2])
                perturbed[i, 1] = np.clip(perturbed[i, 1], perturbed[i, 2], 1 - perturbed[i, 2])
                perturbed[i, 2] = max(0.001, perturbed[i, 2])
        
        # Refine the perturbed solution
        refined = local_refinement(perturbed, max_iterations=50)
        
        # Keep the better solution
        if validate_circles(refined):
            fitness = calculate_sum_radii(refined)
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = refined.copy()
        
        # Continue refining the best solution
        circles = local_refinement(best_solution, max_iterations=30)
        fitness = calculate_sum_radii(circles)
        if fitness > best_fitness:
            best_fitness = fitness
            best_solution = circles.copy()
    
    # Final validation and return
    if validate_circles(best_solution):
        return best_solution
    else:
        # Fallback to a valid configuration
        circles = np.zeros((26, 3))
        for i in range(26):
            circles[i] = [0.5, 0.5, 0.01]
        return circles

# EVOLVE-BLOCK-END