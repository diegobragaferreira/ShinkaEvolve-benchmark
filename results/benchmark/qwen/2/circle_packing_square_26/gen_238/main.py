# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.optimize import minimize
import random
import time
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

    # Check overlap constraints
    for i in range(n):
        x1, y1, r1 = circles[i]
        for j in range(i+1, n):
            x2, y2, r2 = circles[j]
            dist_sq = (x1 - x2)**2 + (y1 - y2)**2
            min_dist_sq = (r1 + r2)**2
            
            if dist_sq < min_dist_sq:
                return False
                
    return True

def calculate_sum_radii(circles: np.ndarray) -> float:
    """Calculate the sum of all radii"""
    return np.sum(circles[:, 2])

def generate_voronoi_initialization(n_circles: int) -> np.ndarray:
    """
    Generate initial circle positions using Voronoi tessellation for even distribution.
    """
    # Generate random points for Voronoi diagram
    points = np.random.random((n_circles * 3, 2))
    
    # Adjust points to avoid extreme edge cases
    points = np.clip(points, 0.05, 0.95)
    
    try:
        vor = Voronoi(points)
        # Get Voronoi vertices as candidate circle centers
        candidates = vor.vertices
        
        # Filter valid candidates inside unit square
        valid_candidates = []
        for vertex in candidates:
            if 0.05 <= vertex[0] <= 0.95 and 0.05 <= vertex[1] <= 0.95:
                valid_candidates.append(vertex)
        
        # If we don't have enough candidates, fall back to random points
        if len(valid_candidates) < n_circles:
            valid_candidates = points[:n_circles]
        
        # Take first n_circles candidates
        positions = np.array(valid_candidates[:n_circles])
        
    except:
        # Fallback to simple random initialization if Voronoi fails
        positions = np.random.random((n_circles, 2))
        positions = np.clip(positions, 0.05, 0.95)
    
    # Create circles with initial radii based on proximity to edges
    circles = np.zeros((n_circles, 3))
    for i in range(n_circles):
        x, y = positions[i]
        # Maximum possible radius at this point
        max_radius = min(x, 1-x, y, 1-y)
        # Start with a reasonable initial radius
        r = np.random.uniform(0.02, max_radius * 0.5)
        circles[i] = [x, y, r]
        
    return circles

def create_physics_simulation(circles: np.ndarray) -> np.ndarray:
    """
    Create a physics-based simulation of the circle packing configuration.
    """
    n = len(circles)
    # Copy circles for simulation
    sim_circles = circles.copy()
    
    # Physics parameters
    dt = 0.01
    friction = 0.95
    repulsion_strength = 10.0
    boundary_strength = 50.0
    
    # Define forces function
    def compute_forces(circs):
        forces = np.zeros((n, 2))
        
        # Repulsion forces between circles
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circs[i]
                x2, y2, r2 = circs[j]
                
                dx = x2 - x1
                dy = y2 - y1
                dist = max(1e-8, np.sqrt(dx*dx + dy*dy))
                min_dist = r1 + r2
                
                if dist < min_dist:
                    # Overlapping - repel with stronger force
                    force_magnitude = repulsion_strength * (min_dist - dist) / (dist + 1e-8)
                    forces[i] += force_magnitude * np.array([dx, dy]) / dist
                    forces[j] -= force_magnitude * np.array([dx, dy]) / dist
        
        # Boundary forces (push towards center)
        for i in range(n):
            x, y, r = circs[i]
            # Force away from boundaries
            fx = 0.0
            fy = 0.0
            
            # Left/right boundaries
            if x < r:
                fx += boundary_strength * (r - x)
            elif x > 1 - r:
                fx -= boundary_strength * (x - (1 - r))
                
            # Top/bottom boundaries  
            if y < r:
                fy += boundary_strength * (r - y)
            elif y > 1 - r:
                fy -= boundary_strength * (y - (1 - r))
            
            forces[i] += np.array([fx, fy])
            
        return forces
    
    # Run physics simulation for several steps
    for step in range(50):
        forces = compute_forces(sim_circles)
        
        # Update positions
        for i in range(n):
            x, y, r = sim_circles[i]
            fx, fy = forces[i]
            
            # Apply forces
            new_x = x + fx * dt
            new_y = y + fy * dt
            
            # Apply friction
            new_x = x + (new_x - x) * friction
            new_y = y + (new_y - y) * friction
            
            # Keep within bounds
            new_x = np.clip(new_x, r, 1 - r)
            new_y = np.clip(new_y, r, 1 - r)
            
            sim_circles[i] = [new_x, new_y, r]
            
    return sim_circles

def optimize_radii(circles: np.ndarray, max_iterations: int = 50) -> np.ndarray:
    """
    Optimize radii using local search with constraint satisfaction.
    """
    n = len(circles)
    optimized = circles.copy()
    
    for iteration in range(max_iterations):
        improved = False
        
        # Try to increase each radius
        for i in range(n):
            x, y, r = optimized[i]
            
            # Calculate maximum possible radius at current position
            max_possible = min(x, 1-x, y, 1-y)
            
            # Try to increase radius
            test_r = min(r + 0.005, max_possible * 0.99)
            
            if test_r > r + 1e-6:
                # Check overlap with all others
                valid = True
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = optimized[j]
                        dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                        if dist < test_r + r2:
                            valid = False
                            break
                
                if valid:
                    optimized[i, 2] = test_r
                    improved = True
                    
        if not improved:
            break
            
    return optimized

def local_gradient_refinement(circles: np.ndarray, max_iterations: int = 30) -> np.ndarray:
    """
    Apply gradient-based refinement to improve the configuration.
    """
    n = len(circles)
    
    # Convert to flattened parameter array for optimization
    def objective(params):
        # Reshape into circles
        circs = params.reshape(n, 3)
        
        # Calculate sum of radii (negative because we want to maximize)
        objective_value = -np.sum(circs[:, 2])
        
        # Add penalty for constraint violations
        penalty = 0
        
        # Boundary constraints
        for i in range(n):
            x, y, r = circs[i]
            if x < r or x > 1-r or y < r or y > 1-r:
                penalty += 1000
        
        # Overlap constraints
        for i in range(n):
            x1, y1, r1 = circs[i]
            for j in range(i+1, n):
                x2, y2, r2 = circs[j]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < r1 + r2:
                    penalty += 100 * (r1 + r2 - dist)
                    
        return objective_value + penalty
    
    # Run optimization
    try:
        # Initialize with current solution
        flat_params = circles.flatten()
        
        # Use L-BFGS-B for optimization
        result = minimize(
            objective, 
            flat_params, 
            method='L-BFGS-B',
            bounds=[(0.001, 0.999) if i % 3 != 2 else (0.001, 0.49) for i in range(len(flat_params))],
            options={'maxiter': max_iterations, 'ftol': 1e-6}
        )
        
        if result.success:
            refined = result.x.reshape(n, 3)
            # Ensure bounds are respected after optimization
            for i in range(n):
                x, y, r = refined[i]
                refined[i] = [np.clip(x, r, 1-r), np.clip(y, r, 1-r), max(0.001, r)]
            return refined
    except:
        pass
    
    return circles

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # Phase 1: Generate initial configuration using Voronoi-based method
    initial_circles = generate_voronoi_initialization(n)
    
    # Phase 2: Apply physics simulation to distribute circles
    physics_result = create_physics_simulation(initial_circles)
    
    # Phase 3: Optimize radii locally
    radius_optimized = optimize_radii(physics_result, 30)
    
    # Phase 4: Apply gradient-based refinement
    final_solution = local_gradient_refinement(radius_optimized, 20)
    
    # Phase 5: Final validation and refinement
    if not validate_circles(final_solution):
        # If initial solution is invalid, start over with different approach
        fallback = generate_voronoi_initialization(n)
        # Simple optimization: try to maximize individual radii
        for i in range(100):
            for j in range(n):
                # Try to maximize radius of circle j
                x, y, r = final_solution[j]
                max_possible = min(x, 1-x, y, 1-y)
                # Try to increase up to maximum possible
                test_r = min(r + 0.001, max_possible)
                
                # Check if this increases total sum without violating constraints
                temp_circles = final_solution.copy()
                temp_circles[j, 2] = test_r
                
                # Check validity
                if validate_circles(temp_circles):
                    final_solution[j, 2] = test_r
            
            # Occasionally reapply physics simulation to help with local minima
            if i % 10 == 0:
                physics_result = create_physics_simulation(final_solution)
                final_solution = optimize_radii(physics_result, 10)
    
    # Final validation
    if not validate_circles(final_solution):
        # Resort to simpler approach if complex optimization fails
        circles = np.zeros((n, 3))
        positions = np.random.random((n, 2))
        positions = np.clip(positions, 0.05, 0.95)
        for i in range(n):
            x, y = positions[i]
            max_radius = min(x, 1-x, y, 1-y)
            r = min(0.05, max_radius * 0.5)
            circles[i] = [x, y, r]
        final_solution = circles
    
    return final_solution

# EVOLVE-BLOCK-END