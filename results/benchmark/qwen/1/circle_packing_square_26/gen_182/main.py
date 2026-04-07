# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import time
from typing import Tuple

# Constants
BOUNDARY_MARGIN = 0.01
MAX_LOCAL_SEARCH_ITERATIONS = 100

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

def create_voronoi_initialization(n_circles: int) -> np.ndarray:
    """Create initial configuration using Voronoi-based spreading with improved spatial distribution."""
    # Create a structured grid pattern with controlled randomness
    grid_size = int(np.ceil(np.sqrt(n_circles)))
    grid_points = []
    
    # Generate regular grid points
    for i in range(grid_size):
        for j in range(grid_size):
            if len(grid_points) < n_circles:
                x = (j + 0.5) / grid_size
                y = (i + 0.5) / grid_size
                grid_points.append([x, y])

    points = np.array(grid_points)

    # Add controlled noise to avoid perfect grid artifacts
    noise_level = 0.025
    points += np.random.uniform(-noise_level, noise_level, points.shape)

    # Clip points to ensure they're within bounds
    points = np.clip(points, BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)

    # Add boundary points for better coverage
    boundary_points = []
    for _ in range(8):
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

    # Limit points to exactly what we need
    points = points[:n_circles]

    # Compute Voronoi diagram and use centroids
    try:
        vor = Voronoi(points)
        
        # Use Voronoi cell centers as initial circle positions
        # We'll use the original points as centroids to start with
        centroids = points[:n_circles]

        # Create circles with initial radii based on Voronoi cell properties
        circles = np.zeros((n_circles, 3))
        for i in range(n_circles):
            x, y = centroids[i]
            
            # Find nearest neighbor to estimate appropriate radius
            distances = np.sqrt(np.sum((centroids - [x, y])**2, axis=1))
            distances = distances[distances > 0]  # Exclude self-distance
            if len(distances) > 0:
                avg_distance = np.min(distances) * 0.35
                radius = min(avg_distance, 0.25)
            else:
                radius = 0.1

            # Ensure it's within bounds
            radius = min(radius, x - BOUNDARY_MARGIN, 1 - x - BOUNDARY_MARGIN,
                        y - BOUNDARY_MARGIN, 1 - y - BOUNDARY_MARGIN)

            circles[i] = [x, y, max(radius, 0.001)]

        return circles
    except:
        # Fallback to grid-based initialization if Voronoi fails
        return generate_grid_initialization(n_circles)

def generate_grid_initialization(n_circles: int) -> np.ndarray:
    """Generate grid-based initial configuration."""
    circles = np.zeros((n_circles, 3))
    grid_size = int(np.ceil(np.sqrt(n_circles)))
    spacing = 1.0 / grid_size
    r = spacing * 0.3
    
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if count < n_circles:
                x = (j + 0.5) * spacing
                y = (i + 0.5) * spacing
                # Adjust for boundary constraints
                x = np.clip(x, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                y = np.clip(y, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                circles[count] = [x, y, r]
                count += 1
    
    return circles

def constraint_aware_local_search(circles: np.ndarray, max_iterations: int = MAX_LOCAL_SEARCH_ITERATIONS) -> np.ndarray:
    """
    A novel constraint-aware local search that simultaneously optimizes positions and radii
    by using a hybrid approach of geometric constraint solving and binary search optimization.
    """
    # Clone input to avoid modifying original
    current_solution = circles.copy()
    
    # Track improvements to detect convergence
    last_best_fitness = evaluate_fitness(current_solution)
    improvement_threshold = 1e-6
    
    for iteration in range(max_iterations):
        improved = False
        current_fitness = evaluate_fitness(current_solution)
        
        # Phase 1: Optimize each circle's radius independently
        for i in range(len(current_solution)):
            original_r = current_solution[i, 2]
            x, y, r = current_solution[i]
            
            # Calculate maximum possible radius at this position
            max_radius = min(x - BOUNDARY_MARGIN, 1 - x - BOUNDARY_MARGIN,
                           y - BOUNDARY_MARGIN, 1 - y - BOUNDARY_MARGIN)
            
            # Try to expand radius using binary search
            if max_radius > r:
                # Binary search for maximum safe expansion using high precision
                low = 0.0
                high = max_radius - r
                best_expansion = 0.0
                precision = 1e-6
                
                # Binary search iterations
                for _ in range(20):  # More iterations for better precision
                    if high - low < precision:
                        break
                    test_expansion = (low + high) / 2
                    test_radius = r + test_expansion
                    
                    # Check if expansion violates any constraints
                    valid = True
                    for j in range(len(current_solution)):
                        if i != j:
                            pos_j = current_solution[j, :2]
                            r_j = current_solution[j, 2]
                            dist = np.sqrt((x - pos_j[0])**2 + (y - pos_j[1])**2)
                            if dist < (test_radius + r_j):
                                valid = False
                                break
                                
                    if valid:
                        best_expansion = test_expansion
                        low = test_expansion
                    else:
                        high = test_expansion
                
                if best_expansion > 0:
                    current_solution[i, 2] = r + best_expansion
                    improved = True
        
        # Phase 2: Optimize positions to reduce overlap and potentially increase radii
        for i in range(len(current_solution)):
            x, y, r = current_solution[i]
            original_pos = [x, y]
            
            # Collect overlapping neighbors efficiently
            neighbors_to_move = []
            for j in range(len(current_solution)):
                if i != j:
                    pos_j = current_solution[j, :2]
                    r_j = current_solution[j, 2]
                    dist = np.sqrt((x - pos_j[0])**2 + (y - pos_j[1])**2)
                    if dist < (r + r_j):
                        neighbors_to_move.append(j)
            
            # If there are overlaps, try to adjust position 
            if neighbors_to_move:
                # Try systematic position adjustments to resolve overlaps
                best_move = [0.0, 0.0]
                best_score = -10000
                
                # Test multiple adjustment directions
                adjustment_steps = np.linspace(-0.02, 0.02, 9)
                for dx in adjustment_steps:
                    for dy in adjustment_steps:
                        test_x = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, x + dx))
                        test_y = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, y + dy))
                        
                        # Check if move helps by evaluating overlap improvement
                        score = 0
                        valid = True
                        
                        # Check overlap with others
                        for j in range(len(current_solution)):
                            if i != j:
                                pos_j = current_solution[j, :2]
                                r_j = current_solution[j, 2]
                                dist = np.sqrt((test_x - pos_j[0])**2 + (test_y - pos_j[1])**2)
                                if dist < (r + r_j):
                                    valid = False
                                    break
                                # Score based on improvement of separation
                                if dist < (r + r_j + 0.01):
                                    score -= 10  # Penalty for remaining overlap
                                else:
                                    score += 1   # Reward for better separation
                                
                        if valid and score > best_score:
                            best_score = score
                            best_move = [dx, dy]
                
                # Apply best move if beneficial
                if best_score > -10000:
                    current_solution[i, 0] = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, x + best_move[0]))
                    current_solution[i, 1] = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, y + best_move[1]))
                    improved = True
        
        # Phase 3: Aggressive repositioning if needed
        if not improved:
            # Try aggressive adjustment by moving all circles slightly
            for i in range(len(current_solution)):
                x, y, r = current_solution[i]
                # Try to shift the circle to reduce overlaps
                moved = False
                for attempt in range(3):
                    dx = np.random.uniform(-0.01, 0.01)
                    dy = np.random.uniform(-0.01, 0.01)
                    
                    test_x = np.clip(x + dx, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                    test_y = np.clip(y + dy, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                    
                    # Check if this leads to fewer overlaps or better fitness
                    valid = True
                    for j in range(len(current_solution)):
                        if i != j:
                            pos_j = current_solution[j, :2]
                            r_j = current_solution[j, 2]
                            dist = np.sqrt((test_x - pos_j[0])**2 + (test_y - pos_j[1])**2)
                            if dist < (r + r_j):
                                valid = False
                                break
                    
                    if valid:
                        current_solution[i, 0] = test_x
                        current_solution[i, 1] = test_y
                        moved = True
                        improved = True
                        break
        
        # Check if we've made significant improvement
        new_fitness = evaluate_fitness(current_solution)
        if new_fitness > last_best_fitness + improvement_threshold:
            last_best_fitness = new_fitness
        elif not improved:
            # If no improvement, stop early
            break
    
    return current_solution

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    n = 26
    best_solution = None
    best_fitness = -np.inf
    
    # Step 1: Generate multiple initial configurations using Voronoi-based approach
    initial_candidates = []
    for _ in range(15):  # Generate several good candidates
        circles = create_voronoi_initialization(n)
        if is_valid_configuration(circles):
            initial_candidates.append(circles)
    
    # If no valid candidates, fall back to grid
    if not initial_candidates:
        circles = generate_grid_initialization(n)
        return circles
    
    # Step 2: Select the best initial candidate
    initial_fitnesses = [evaluate_fitness(c) for c in initial_candidates]
    best_initial_idx = np.argmax(initial_fitnesses)
    current_solution = initial_candidates[best_initial_idx].copy()
    
    # Step 3: Apply our novel local search optimization
    refined_solution = constraint_aware_local_search(current_solution)
    current_fitness = evaluate_fitness(refined_solution)
    
    # Track best solution so far
    if current_fitness > best_fitness:
        best_fitness = current_fitness
        best_solution = refined_solution.copy()
    
    # Step 4: Multi-pass refinement using our advanced local search
    for round_num in range(3):  # Multiple rounds of refinement
        local_refined = constraint_aware_local_search(refined_solution, 150)
        local_fitness = evaluate_fitness(local_refined)
        
        if local_fitness > best_fitness:
            best_fitness = local_fitness
            best_solution = local_refined.copy()
            refined_solution = local_refined.copy()
        else:
            # Try a different approach for more diverse exploration
            perturbed = refined_solution.copy()
            # Apply more aggressive random perturbations
            for i in range(len(perturbed)):
                # Randomly decide whether to perturb or not
                if np.random.random() < 0.3:  # 30% chance to perturb
                    # Larger perturbations
                    perturbed[i, 0] += np.random.uniform(-0.01, 0.01)
                    perturbed[i, 1] += np.random.uniform(-0.01, 0.01)
                    # Keep within bounds
                    perturbed[i, 0] = np.clip(perturbed[i, 0], 
                                            perturbed[i, 2] + BOUNDARY_MARGIN, 
                                            1 - perturbed[i, 2] - BOUNDARY_MARGIN)
                    perturbed[i, 1] = np.clip(perturbed[i, 1], 
                                            perturbed[i, 2] + BOUNDARY_MARGIN, 
                                            1 - perturbed[i, 2] - BOUNDARY_MARGIN)
            
            # Validate and update if better
            if is_valid_configuration(perturbed):
                perturbed_fitness = evaluate_fitness(perturbed)
                if perturbed_fitness > best_fitness:
                    best_fitness = perturbed_fitness
                    best_solution = perturbed.copy()
                    refined_solution = perturbed.copy()
    
    # Step 5: Final refinement pass
    if best_solution is not None:
        final_refinement = constraint_aware_local_search(best_solution, 100)
        final_fitness = evaluate_fitness(final_refinement)
        if final_fitness > best_fitness:
            best_solution = final_refinement
    
    return best_solution if best_solution is not None else generate_grid_initialization(n)

# EVOLVE-BLOCK-END