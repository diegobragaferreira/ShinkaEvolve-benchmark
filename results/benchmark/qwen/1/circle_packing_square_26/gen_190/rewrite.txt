# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
import time
from typing import Tuple
import cvxpy as cp

# Constants for the optimization
BOUNDARY_MARGIN = 0.01
MAX_LOCAL_SEARCH_ITERATIONS = 50

def is_valid_configuration(circles: np.ndarray) -> bool:
    """Check if a configuration of circles is valid (no overlaps, fully contained)."""
    n = len(circles)

    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0 or x < r + BOUNDARY_MARGIN or x > 1-r - BOUNDARY_MARGIN or y < r + BOUNDARY_MARGIN or y > 1-r - BOUNDARY_MARGIN:
            return False

    # Check overlap constraints with early termination
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance_squared = (x1-x2)**2 + (y1-y2)**2
            min_distance_squared = (r1 + r2)**2
            if distance_squared < min_distance_squared:
                return False

    return True

def evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate fitness as the sum of radii."""
    return np.sum(circles[:, 2])

def create_geometric_initialization(n_circles: int) -> np.ndarray:
    """Create initial configuration using geometric constraint programming approach."""
    # Generate points using a honeycomb-inspired pattern for better spatial distribution
    grid_size = int(np.ceil(np.sqrt(n_circles)))
    points = []
    
    # Create honeycomb pattern with staggered rows
    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) < n_circles:
                # Staggered rows for better coverage
                offset = (i % 2) * 0.5
                x = (j + offset) / grid_size
                y = i / grid_size
                points.append([x, y])

    points = np.array(points)
    
    # Add boundary points for better coverage
    boundary_points = []
    for _ in range(10):
        side = np.random.randint(0, 4)
        if side == 0:  # Top
            boundary_points.append([np.random.rand(), 1.0 - BOUNDARY_MARGIN])
        elif side == 1:  # Bottom
            boundary_points.append([np.random.rand(), BOUNDARY_MARGIN])
        elif side == 2:  # Left
            boundary_points.append([BOUNDARY_MARGIN, np.random.rand()])
        else:  # Right
            boundary_points.append([1.0 - BOUNDARY_MARGIN, np.random.rand()])

    points = np.vstack([points, boundary_points])
    points = points[:n_circles]

    # Create circles with initial radii based on local density estimation
    circles = np.zeros((n_circles, 3))
    
    # Use Voronoi-based radius estimation for better initial values
    try:
        vor = Voronoi(points)
        # Estimate radii based on Voronoi cell areas
        for i in range(n_circles):
            # Find minimum distance to neighbors for radius estimation
            x, y = points[i]
            distances = np.sqrt(np.sum((points - [x, y])**2, axis=1))
            distances = distances[distances > 0]  # Exclude self-distance
            
            if len(distances) > 0:
                # Use inverse relationship to neighbors for sensible radius
                avg_distance = np.min(distances) * 0.4
                radius = min(avg_distance, 0.25)
            else:
                radius = 0.1

            # Ensure it's within bounds
            radius = min(radius, x - BOUNDARY_MARGIN, 1 - x - BOUNDARY_MARGIN,
                        y - BOUNDARY_MARGIN, 1 - y - BOUNDARY_MARGIN)

            circles[i] = [x, y, max(radius, 0.001)]
    except:
        # Fallback to simpler initialization if Voronoi fails
        for i in range(n_circles):
            x, y = points[i]
            # Conservative radius
            radius = min(0.15, x - BOUNDARY_MARGIN, 1 - x - BOUNDARY_MARGIN,
                       y - BOUNDARY_MARGIN, 1 - y - BOUNDARY_MARGIN)
            circles[i] = [x, y, max(radius, 0.001)]

    return circles

def geometric_constraint_optimizer(circles: np.ndarray, max_iterations: int = MAX_LOCAL_SEARCH_ITERATIONS) -> np.ndarray:
    """
    Geometric constraint programming approach that formulates circle packing as 
    constrained optimization problem and solves it using geometric reasoning.
    """
    # Clone input to avoid modifying original
    solution = circles.copy()
    n = len(solution)
    
    # Precompute neighbor relationships for efficiency
    def get_neighbors(circle_idx):
        neighbors = []
        for j in range(n):
            if i != j:
                x1, y1, r1 = solution[i]
                x2, y2, r2 = solution[j]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < (r1 + r2):
                    neighbors.append(j)
        return neighbors
    
    # Iterative constraint satisfaction with geometric reasoning
    for iteration in range(max_iterations):
        improved = False
        
        # Process each circle to optimize both position and radius
        for i in range(n):
            # Get current state
            x, y, r = solution[i]
            
            # Store original values for comparison
            original_x, original_y, original_r = x, y, r
            
            # Geometric constraint processing
            # 1. First, analyze constraints and compute feasible region
            max_radius = min(x - BOUNDARY_MARGIN, 1 - x - BOUNDARY_MARGIN,
                           y - BOUNDARY_MARGIN, 1 - y - BOUNDARY_MARGIN)
            
            # 2. Find minimum distance to all neighbors to constrain radius increase
            min_neighbor_dist = float('inf')
            neighbor_info = []
            
            for j in range(n):
                if i != j:
                    x2, y2, r2 = solution[j]
                    dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if dist < (r + r2):
                        # Record constraint violation
                        neighbor_info.append((j, dist, r2))
                    min_neighbor_dist = min(min_neighbor_dist, dist)
            
            # 3. Determine optimal radius and position adjustments
            if max_radius > r:
                # Try to expand radius respecting neighbor constraints
                target_radius = min(max_radius, min_neighbor_dist - 0.001)
                
                if target_radius > r:
                    # Binary search for optimal expansion within constraints
                    low = 0.0
                    high = target_radius - r
                    best_expansion = 0.0
                    
                    for _ in range(10):
                        test_expansion = (low + high) / 2
                        test_radius = r + test_expansion
                        
                        # Check if expansion violates any constraints
                        valid = True
                        for j in range(n):
                            if i != j:
                                x2, y2, r2 = solution[j]
                                dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                                if dist < (test_radius + r2):
                                    valid = False
                                    break
                        
                        if valid:
                            best_expansion = test_expansion
                            low = test_expansion
                        else:
                            high = test_expansion
                    
                    if best_expansion > 0:
                        solution[i, 2] = r + best_expansion
                        improved = True
            
            # 4. Position adjustment to reduce overlaps
            if neighbor_info:
                # Compute repulsive forces from overlapping neighbors
                total_dx, total_dy = 0.0, 0.0
                for j, dist, r2 in neighbor_info:
                    if dist < (r + r2):
                        x2, y2, r2 = solution[j]
                        dx = x2 - x
                        dy = y2 - y
                        if dx != 0 or dy != 0:
                            length = np.sqrt(dx*dx + dy*dy)
                            # Normalize and scale by overlap amount
                            force_magnitude = (r + r2 - dist) / length
                            total_dx += dx * force_magnitude / length
                            total_dy += dy * force_magnitude / length
                
                if abs(total_dx) > 1e-8 or abs(total_dy) > 1e-8:
                    # Apply small adjustment to reduce overlap
                    adjustment_scale = 0.01
                    new_x = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, x + total_dx * adjustment_scale))
                    new_y = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, y + total_dy * adjustment_scale))
                    
                    # Check that adjustment is still valid
                    valid_adjustment = True
                    for j in range(n):
                        if i != j:
                            x2, y2, r2 = solution[j]
                            dist = np.sqrt((new_x - x2)**2 + (new_y - y2)**2)
                            if dist < (r + r2):
                                valid_adjustment = False
                                break
                    
                    if valid_adjustment:
                        solution[i, 0] = new_x
                        solution[i, 1] = new_y
                        improved = True
        
        # Break early if no significant improvement
        if not improved:
            break
    
    # Final validation and cleanup
    for i in range(n):
        # Ensure boundary constraints
        x, y, r = solution[i]
        solution[i, 0] = np.clip(x, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
        solution[i, 1] = np.clip(y, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
    
    return solution

def constraint_satisfaction_approach(circles: np.ndarray, max_iterations: int = MAX_LOCAL_SEARCH_ITERATIONS) -> np.ndarray:
    """
    Constraint satisfaction approach using mathematical programming concepts
    to iteratively resolve conflicts through geometric reasoning.
    """
    # Create a constraint-based optimization model
    solution = circles.copy()
    n = len(solution)
    
    # Track improvement to detect convergence
    prev_fitness = evaluate_fitness(solution)
    
    for iteration in range(max_iterations):
        improved = False
        current_fitness = evaluate_fitness(solution)
        
        # Try to increase all radii simultaneously using geometric constraints
        for i in range(n):
            # Find neighbors that would cause overlap if we increase radius
            x, y, r = solution[i]
            
            # Constraint analysis
            max_radius_allowed = min(
                x - BOUNDARY_MARGIN,
                1 - x - BOUNDARY_MARGIN,
                y - BOUNDARY_MARGIN,
                1 - y - BOUNDARY_MARGIN
            )
            
            # Check neighbor constraints more carefully
            max_radius_for_overlap = float('inf')
            for j in range(n):
                if i != j:
                    x2, y2, r2 = solution[j]
                    d = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if d < (r + r2):
                        # If overlapping, we can only increase radius if we move
                        # But for now, we'll just see what the max radius would be
                        max_radius_for_overlap = min(max_radius_for_overlap, d - r2 - 0.001)
            
            # Determine safe maximum radius
            effective_max_radius = min(max_radius_allowed, max_radius_for_overlap)
            
            if effective_max_radius > r:
                # Increase radius conservatively
                new_radius = min(effective_max_radius, r + 0.005)
                if new_radius > r:
                    solution[i, 2] = new_radius
                    improved = True
        
        # Position fine-tuning to resolve conflicts
        for i in range(n):
            x, y, r = solution[i]
            
            # Check all overlaps
            overlaps = []
            for j in range(n):
                if i != j:
                    x2, y2, r2 = solution[j]
                    d = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if d < (r + r2):
                        overlaps.append((j, d))
            
            if overlaps:
                # Apply geometric adjustment to reduce overlaps
                total_adjustment = np.array([0.0, 0.0])
                for j, dist in overlaps:
                    x2, y2, r2 = solution[j]
                    dx = x2 - x
                    dy = y2 - y
                    if abs(dx) > 1e-10 or abs(dy) > 1e-10:
                        length = np.sqrt(dx*dx + dy*dy)
                        # Repulsion force (inverse distance)
                        force = (r + r2 - dist) / (length + 1e-10)  # Avoid division by zero
                        total_adjustment[0] += dx * force / length
                        total_adjustment[1] += dy * force / length
                
                # Apply adjustment
                if np.linalg.norm(total_adjustment) > 1e-8:
                    # Scale adjustment
                    adjustment_magnitude = min(0.01, np.linalg.norm(total_adjustment))
                    normalized_adjustment = total_adjustment / (np.linalg.norm(total_adjustment) + 1e-10) * adjustment_magnitude
                    
                    new_x = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, x + normalized_adjustment[0]))
                    new_y = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, y + normalized_adjustment[1]))
                    
                    # Verify adjustment
                    is_valid = True
                    for j in range(n):
                        if i != j:
                            x2, y2, r2 = solution[j]
                            dist = np.sqrt((new_x - x2)**2 + (new_y - y2)**2)
                            if dist < (r + r2):
                                is_valid = False
                                break
                    
                    if is_valid:
                        solution[i, 0] = new_x
                        solution[i, 1] = new_y
                        improved = True
        
        # Check for convergence
        if not improved and current_fitness < prev_fitness + 1e-6:
            break
            
        prev_fitness = current_fitness
    
    return solution

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses geometric constraint programming approach instead of evolutionary methods.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    n = 26
    best_solution = None
    best_fitness = -np.inf
    
    # Generate initial configurations using geometric approach
    initial_candidates = []
    for _ in range(10):  # Generate several candidates
        circles = create_geometric_initialization(n)
        if is_valid_configuration(circles):
            initial_candidates.append(circles)
    
    # If no valid candidates, fall back to grid
    if not initial_candidates:
        circles = np.zeros((n, 3))
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing = 1.0 / grid_size
        r = spacing * 0.3
        
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count < n:
                    x = (j + 0.5) * spacing
                    y = (i + 0.5) * spacing
                    x = np.clip(x, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                    y = np.clip(y, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                    circles[count] = [x, y, r]
                    count += 1
        return circles
    
    # Select best initial candidate
    initial_fitnesses = [evaluate_fitness(c) for c in initial_candidates]
    best_initial_idx = np.argmax(initial_fitnesses)
    current_solution = initial_candidates[best_initial_idx].copy()
    
    # Apply geometric constraint programming optimization
    refined_solution = geometric_constraint_optimizer(current_solution)
    current_fitness = evaluate_fitness(refined_solution)
    
    # Track best solution so far
    if current_fitness > best_fitness:
        best_fitness = current_fitness
        best_solution = refined_solution.copy()
    
    # Apply constraint satisfaction approach for refinement
    for _ in range(3):  # Multiple refinement passes
        constraint_refined = constraint_satisfaction_approach(refined_solution)
        constraint_fitness = evaluate_fitness(constraint_refined)
        
        if constraint_fitness > best_fitness:
            best_fitness = constraint_fitness
            best_solution = constraint_refined.copy()
            refined_solution = constraint_refined.copy()
        else:
            # Try random perturbations to escape local optima
            perturbed = refined_solution.copy()
            for i in range(len(perturbed)):
                if np.random.random() < 0.2:  # 20% chance to perturb
                    # Small random movement
                    perturbed[i, 0] += np.random.uniform(-0.005, 0.005)
                    perturbed[i, 1] += np.random.uniform(-0.005, 0.005)
                    
                    # Keep within bounds
                    r = perturbed[i, 2]
                    perturbed[i, 0] = np.clip(perturbed[i, 0], r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                    perturbed[i, 1] = np.clip(perturbed[i, 1], r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
            
            if is_valid_configuration(perturbed):
                perturbed_fitness = evaluate_fitness(perturbed)
                if perturbed_fitness > best_fitness:
                    best_fitness = perturbed_fitness
                    best_solution = perturbed.copy()
                    refined_solution = perturbed.copy()
    
    # Final validation
    if best_solution is not None:
        if not is_valid_configuration(best_solution):
            # Reconstruct with basic validation if necessary
            final_solution = np.zeros_like(best_solution)
            for i in range(n):
                x, y, r = best_solution[i]
                final_solution[i] = [x, y, r]
                # Ensure bounding
                final_solution[i, 0] = np.clip(final_solution[i, 0], 
                                             final_solution[i, 2] + BOUNDARY_MARGIN, 
                                             1 - final_solution[i, 2] - BOUNDARY_MARGIN)
                final_solution[i, 1] = np.clip(final_solution[i, 1], 
                                             final_solution[i, 2] + BOUNDARY_MARGIN, 
                                             1 - final_solution[i, 2] - BOUNDARY_MARGIN)
            best_solution = final_solution
    
    return best_solution if best_solution is not None else create_geometric_initialization(n)

# EVOLVE-BLOCK-END