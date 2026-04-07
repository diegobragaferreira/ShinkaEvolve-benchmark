# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List, Optional
import heapq
from collections import deque

# Fixed seed for reproducibility
np.random.seed(42)
random.seed(42)

def is_valid_placement(circles: np.ndarray, idx: int) -> bool:
    """Check if circle at index idx is valid (within bounds and not overlapping)."""
    x, y, r = circles[idx]

    # Check containment constraints
    if x < r or x > 1 - r or y < r or y > 1 - r:
        return False

    # Check overlap constraints with existing circles
    for i in range(len(circles)):
        if i == idx:
            continue
        x_i, y_i, r_i = circles[i]
        distance = np.sqrt((x - x_i)**2 + (y - y_i)**2)
        if distance < r + r_i:
            return False

    return True

def evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate fitness as sum of radii."""
    return np.sum(circles[:, 2])

def compute_max_radius(circles: np.ndarray, x: float, y: float, exclude_idx: int = -1) -> float:
    """Compute maximum possible radius for a circle at (x,y) without overlapping existing circles."""
    max_radius = min(x, 1-x, y, 1-y)  # Bound by edges
    
    for i in range(len(circles)):
        if i == exclude_idx:
            continue
        x_i, y_i, r_i = circles[i]
        distance = np.sqrt((x - x_i)**2 + (y - y_i)**2)
        max_radius = min(max_radius, distance - r_i)
    
    return max_radius

def greedy_initial_placement(n_circles: int) -> np.ndarray:
    """Create initial placement using greedy approach prioritizing corner regions."""
    circles = np.zeros((n_circles, 3))
    
    # Start with corner and edge strategies
    positions = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]  # Four corners
    if n_circles > 4:
        # Add edge centers
        positions.extend([(0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5)])
    if n_circles > 8:
        # Add center region
        positions.extend([(0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)])
    
    # Fill remaining positions randomly with careful consideration
    while len(positions) < n_circles:
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        positions.append((x, y))
    
    # Place circles greedily considering spatial distribution
    for i in range(n_circles):
        x, y = positions[i]
        
        # Compute maximum radius at this position
        candidate_circles = circles.copy()
        candidate_circles[i] = [x, y, 0.001]  # Temporarily place
        
        # Calculate max radius avoiding overlap
        max_r = compute_max_radius(candidate_circles, x, y, i)
        if max_r <= 0:
            max_r = 0.05
            
        # Prefer larger radii in corner areas
        if i < 4:  # Corners
            max_r = min(max_r, 0.15)
        elif i < 8:  # Edges
            max_r = min(max_r, 0.1)
            
        circles[i] = [x, y, max_r]
    
    # Refine initial positions to reduce overlap
    for attempt in range(100):
        improved = False
        for i in range(n_circles):
            if not is_valid_placement(circles, i):
                # Try to move circle to a better position
                original_x, original_y, original_r = circles[i]
                
                # Search neighborhood for better position
                best_pos = (original_x, original_y, original_r)
                best_radius = original_r
                
                for dx in [-0.1, -0.05, 0, 0.05, 0.1]:
                    for dy in [-0.1, -0.05, 0, 0.05, 0.1]:
                        new_x = original_x + dx
                        new_y = original_y + dy
                        
                        if 0.01 <= new_x <= 0.99 and 0.01 <= new_y <= 0.99:
                            # Check if this position allows larger radius
                            temp_circles = circles.copy()
                            temp_circles[i] = [new_x, new_y, 0.001]
                            max_r = compute_max_radius(temp_circles, new_x, new_y, i)
                            
                            if max_r > best_radius and max_r > 0.001:
                                best_radius = max_r
                                best_pos = (new_x, new_y, best_radius)
                
                if best_radius > original_r:
                    circles[i] = list(best_pos)
                    improved = True
                    
        if not improved:
            break
    
    return circles

def propagate_constraints(circles: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Use constraint propagation to refine circle placements."""
    refined = circles.copy()
    
    for iteration in range(max_iter):
        improved = False
        
        # For each circle, try to increase radius while maintaining feasibility
        for i in range(len(refined)):
            if i >= len(refined):
                break
                
            x, y, r = refined[i]
            
            # Compute new maximum radius
            max_r_new = compute_max_radius(refined, x, y, i)
            if max_r_new > r and max_r_new > 0.001:
                # Check if we can actually increase radius without breaking other constraints
                # by temporarily setting it and checking all neighbors
                test_circles = refined.copy()
                test_circles[i] = [x, y, max_r_new]
                
                valid = True
                for j in range(len(test_circles)):
                    if i != j:
                        x_j, y_j, r_j = test_circles[j]
                        dist = np.sqrt((x - x_j)**2 + (y - y_j)**2)
                        if dist < max_r_new + r_j:
                            valid = False
                            break
                
                if valid:
                    refined[i] = [x, y, max_r_new]
                    improved = True
                    
        if not improved:
            break
    
    return refined

def local_improvement_step(circles: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """Perform local improvements using gradient-like approaches."""
    improved = circles.copy()
    
    for _ in range(max_iter):
        improved_iter = False
        
        # Try to increase each circle's radius
        for i in range(len(improved)):
            x, y, r = improved[i]
            
            # Compute max possible radius at this position
            max_r = compute_max_radius(improved, x, y, i)
            
            if max_r > r and max_r > 0.001 and max_r - r > 0.001:
                # Try to increase radius with small step
                step_size = min(0.01, (max_r - r) * 0.5)
                new_r = min(max_r, r + step_size)
                
                # Check validity after change
                test_circles = improved.copy()
                test_circles[i] = [x, y, new_r]
                
                valid = True
                for j in range(len(test_circles)):
                    if i != j:
                        x_j, y_j, r_j = test_circles[j]
                        dist = np.sqrt((x - x_j)**2 + (y - y_j)**2)
                        if dist < new_r + r_j:
                            valid = False
                            break
                
                if valid and new_r <= 1 - max(x, 1-x) and new_r <= 1 - max(y, 1-y):
                    improved[i] = [x, y, new_r]
                    improved_iter = True
                    
        if not improved_iter:
            break
            
    return improved

def constraint_propagation_optimization(n_circles: int = 26, max_iterations: int = 1000) -> np.ndarray:
    """Main optimization using constraint propagation approach."""
    
    # Phase 1: Greedy initial placement
    circles = greedy_initial_placement(n_circles)
    
    # Phase 2: Constraint propagation
    circles = propagate_constraints(circles)
    
    # Phase 3: Local improvements
    circles = local_improvement_step(circles)
    
    # Phase 4: Systematic refinement
    best_fitness = evaluate_fitness(circles)
    best_solution = circles.copy()
    
    # Try multiple random restarts to escape local optima
    for restart in range(20):
        # Perturb solution slightly and re-optimize
        perturbed = circles.copy()
        for i in range(min(n_circles, 15)):  # Only perturb first 15 circles
            if np.random.random() < 0.3:
                perturbed[i, 0] += np.random.uniform(-0.02, 0.02)
                perturbed[i, 1] += np.random.uniform(-0.02, 0.02)
                # Clamp to valid range
                perturbed[i, 0] = np.clip(perturbed[i, 0], 0.01, 0.99)
                perturbed[i, 1] = np.clip(perturbed[i, 1], 0.01, 0.99)
                
        # Apply constraint propagation to perturbed solution
        refined = propagate_constraints(perturbed)
        refined = local_improvement_step(refined)
        
        fitness = evaluate_fitness(refined)
        if fitness > best_fitness:
            best_fitness = fitness
            best_solution = refined.copy()
    
    # Final local optimization
    final_result = local_improvement_step(best_solution, 100)
    
    return final_result

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    try:
        circles = constraint_propagation_optimization(26)
        return circles
    except Exception as e:
        print(f"Error during optimization: {e}")
        # Fallback to improved heuristic
        circles = np.zeros((26, 3))
        
        # Try to create a more organized pattern
        grid_size = int(np.ceil(np.sqrt(26)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        radius = spacing_x / 3.0

        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= 26:
                    break
                x = spacing_x * (i + 1)
                y = spacing_y * (j + 1)
                # Slightly randomize to avoid perfect grid issues
                x += np.random.uniform(-spacing_x/10, spacing_x/10)
                y += np.random.uniform(-spacing_y/10, spacing_y/10)
                circles[count] = [x, y, radius]
                count += 1
            if count >= 26:
                break

        # Ensure constraints are satisfied
        for i in range(count):
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])

        return circles

# EVOLVE-BLOCK-END