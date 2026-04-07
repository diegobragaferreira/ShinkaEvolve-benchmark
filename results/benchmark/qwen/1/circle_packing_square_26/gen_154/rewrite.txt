# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
from typing import Tuple, List

# Global constants for the optimization
BOUNDARY_MARGIN = 0.01
MAX_LOCAL_SEARCH_ITERATIONS = 200

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
    """Create initial configuration using Voronoi-based initialization with Lloyd's algorithm."""
    # Start with random points
    points = np.random.rand(n_circles, 2)
    
    # Apply Lloyd's algorithm for better distribution
    for _ in range(10):
        try:
            vor = Voronoi(points)
            # Compute centroids of Voronoi cells
            new_points = []
            for i in range(len(points)):
                region = vor.point_region[i]
                if region != -1:  # Not infinite region
                    vertices = vor.vertices[vor.regions[region]]
                    if len(vertices) > 0:
                        centroid = np.mean(vertices, axis=0)
                        # Keep centroid within bounds
                        centroid = np.clip(centroid, BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
                        new_points.append(centroid)
                    else:
                        # Fallback to original point if no vertices
                        new_points.append(points[i])
                else:
                    # Fallback to original point for infinite region
                    new_points.append(points[i])
            points = np.array(new_points)
        except:
            # If Voronoi fails, use random points with boundary adjustments
            points = np.random.rand(n_circles, 2)
            points = np.clip(points, BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
    
    # Create initial circles with radii based on Voronoi proximity
    circles = np.zeros((n_circles, 3))
    
    # Use a greedy approach to assign radii
    for i in range(n_circles):
        x, y = points[i]
        
        # Find nearest neighbors to estimate appropriate radius
        distances = np.sqrt(np.sum((points - [x, y])**2, axis=1))
        distances = distances[distances > 0]  # Exclude self-distance
        
        if len(distances) > 0:
            avg_distance = np.min(distances) * 0.4
            radius = min(avg_distance, 0.3)
        else:
            radius = 0.1
            
        # Ensure it's within bounds
        radius = min(radius, x - BOUNDARY_MARGIN, 1 - x - BOUNDARY_MARGIN,
                    y - BOUNDARY_MARGIN, 1 - y - BOUNDARY_MARGIN)
        
        circles[i] = [x, y, max(radius, 0.001)]
    
    return circles

def optimize_single_circle(circles: np.ndarray, index: int, max_iter: int = 50) -> np.ndarray:
    """Optimize a single circle's position and radius using constrained optimization."""
    current_circles = circles.copy()
    x, y, r = current_circles[index]
    
    # Define bounds for optimization
    bounds = [
        (r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN),  # x bound
        (r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN),  # y bound
        (0.001, 0.5)  # r bound (max radius)
    ]
    
    # Optimization variables: [delta_x, delta_y, delta_r]
    def objective(params):
        dx, dy, dr = params
        new_x = max(bounds[0][0], min(bounds[0][1], x + dx))
        new_y = max(bounds[1][0], min(bounds[1][1], y + dy))
        new_r = max(bounds[2][0], min(bounds[2][1], r + dr))
        
        # Calculate the change in total fitness (negative because we want to maximize)
        # We only care about change in this circle's radius
        return -new_r
    
    def constraint_overlap(params):
        dx, dy, dr = params
        new_x = max(bounds[0][0], min(bounds[0][1], x + dx))
        new_y = max(bounds[1][0], min(bounds[1][1], y + dy))
        new_r = max(bounds[2][0], min(bounds[2][1], r + dr))
        
        # Check for overlap with all other circles
        penalty = 0
        for i in range(len(current_circles)):
            if i != index:
                cx, cy, cr = current_circles[i]
                dist = np.sqrt((new_x - cx)**2 + (new_y - cy)**2)
                if dist < (new_r + cr):
                    # Penalty proportional to overlap amount
                    overlap = (new_r + cr) - dist
                    penalty += overlap**2
        return 1.0 - penalty  # Positive if constraint satisfied
    
    # Set up constraints
    constraints = [{'type': 'ineq', 'fun': constraint_overlap}]
    
    # Initial guess
    initial_guess = [0.0, 0.0, 0.0]
    
    try:
        # Optimize
        result = minimize(objective, initial_guess, method='SLSQP', 
                         bounds=[(-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)],
                         constraints=constraints, options={'maxiter': max_iter})
        
        if result.success:
            dx, dy, dr = result.x
            new_x = max(bounds[0][0], min(bounds[0][1], x + dx))
            new_y = max(bounds[1][0], min(bounds[1][1], y + dy))
            new_r = max(bounds[2][0], min(bounds[2][1], r + dr))
            
            # Update the circle
            current_circles[index] = [new_x, new_y, new_r]
    except:
        pass  # If optimization fails, keep current configuration
    
    return current_circles

def voronoi_guided_optimization(circles: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """
    Perform Voronoi-guided optimization by iteratively improving each circle.
    """
    current_circles = circles.copy()
    
    # Group circles by Voronoi region influence
    for iteration in range(max_iter):
        improved = False
        
        # Shuffle circle order for better convergence
        circle_order = list(range(len(current_circles)))
        np.random.shuffle(circle_order)
        
        # Optimize each circle
        for i in circle_order:
            # Get current configuration
            old_config = current_circles[i].copy()
            
            # Optimize this specific circle
            current_circles = optimize_single_circle(current_circles, i)
            
            # Check if there was improvement
            if not np.allclose(old_config, current_circles[i]):
                improved = True
        
        # If no improvement after a full cycle, stop
        if not improved:
            break
    
    return current_circles

def constraint_aware_local_search(circles: np.ndarray, max_iterations: int = MAX_LOCAL_SEARCH_ITERATIONS) -> np.ndarray:
    """
    Apply a constrained local search optimization to improve solution quality.
    """
    # Clone input to avoid modifying original
    current_solution = circles.copy()
    
    # Keep track of best solution found during local search
    best_solution = current_solution.copy()
    best_fitness = evaluate_fitness(current_solution)

    for iteration in range(max_iterations):
        improved = False
        
        # Strategy 1: Try to expand each circle's radius while maintaining constraints
        for i in range(len(current_solution)):
            original_r = current_solution[i, 2]
            x, y, r = current_solution[i]
            
            # Calculate maximum possible radius at this position
            max_radius = min(x - BOUNDARY_MARGIN, 1 - x - BOUNDARY_MARGIN,
                           y - BOUNDARY_MARGIN, 1 - y - BOUNDARY_MARGIN)
            
            # Try to expand radius
            if max_radius > r:
                # Binary search for maximum safe expansion
                low = 0.0
                high = max_radius - r
                best_expansion = 0.0
                
                for _ in range(15):  # More iterations for better precision
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
        
        # Strategy 2: Try position adjustments to resolve overlaps
        for i in range(len(current_solution)):
            x, y, r = current_solution[i]
            
            # Collect overlapping neighbors
            overlapping_pairs = []
            for j in range(len(current_solution)):
                if i != j:
                    pos_j = current_solution[j, :2]
                    r_j = current_solution[j, 2]
                    dist = np.sqrt((x - pos_j[0])**2 + (y - pos_j[1])**2)
                    if dist < (r + r_j):
                        overlapping_pairs.append((j, dist))
            
            # If there are overlaps, try adjusting position
            if overlapping_pairs:
                # Try several small moves
                best_move = [0.0, 0.0]
                best_score = -1000
                
                # Use a more thorough search around current position
                moves = []
                for dx in [-0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02]:
                    for dy in [-0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02]:
                        moves.append((dx, dy))
                        
                for dx, dy in moves:
                    test_x = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, x + dx))
                    test_y = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, y + dy))
                    
                    # Check if move helps
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
                            # Score based on how much we improve overlap (closer to minimum distance)
                            if dist < (r + r_j + 0.005):
                                score -= 10  # Penalty for remaining overlap
                            else:
                                score += (dist - (r + r_j))  # Reward for better separation
                    
                    if valid and score > best_score:
                        best_score = score
                        best_move = [dx, dy]
                
                # Apply best move if beneficial
                if best_score > -1000 and best_score > 0:
                    current_solution[i, 0] = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, x + best_move[0]))
                    current_solution[i, 1] = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, y + best_move[1]))
                    improved = True
        
        # Strategy 3: Global improvement by adjusting all positions
        if not improved and iteration % 5 == 0:  # Only do occasionally
            # Try small random adjustments for all circles
            for i in range(len(current_solution)):
                x, y, r = current_solution[i]
                dx = np.random.normal(0, 0.005)  # Smaller perturbation
                dy = np.random.normal(0, 0.005)
                
                test_x = np.clip(x + dx, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                test_y = np.clip(y + dy, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                
                # Check overlap - be more conservative here
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
                    improved = True
        
        # Update best solution if current is better
        current_fitness = evaluate_fitness(current_solution)
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_solution = current_solution.copy()
        
        if not improved:
            break
    
    return best_solution

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

    # Use Voronoi-guided initialization
    circles = create_voronoi_initialization(n)
    
    # Ensure it's valid
    if not is_valid_configuration(circles):
        # Fall back to grid initialization
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
                    circles[count] = [x, y, r]
                    count += 1

    # Apply Voronoi-guided optimization
    optimized_circles = voronoi_guided_optimization(circles, 50)
    
    # Apply local search refinements
    refined_circles = constraint_aware_local_search(optimized_circles)
    
    # Final evaluation
    final_fitness = evaluate_fitness(refined_circles)
    
    if final_fitness > best_fitness:
        best_fitness = final_fitness
        best_solution = refined_circles.copy()

    # Additional refinement rounds
    for _ in range(3):
        optimized_circles = voronoi_guided_optimization(best_solution, 30)
        refined_circles = constraint_aware_local_search(optimized_circles)
        current_fitness = evaluate_fitness(refined_circles)
        
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_solution = refined_circles.copy()
        else:
            # Try perturbation if no improvement
            perturbed = best_solution.copy()
            for i in range(len(perturbed)):
                # Small random perturbation
                perturbed[i, 0] += np.random.normal(0, 0.005)
                perturbed[i, 1] += np.random.normal(0, 0.005)
                # Keep within bounds
                perturbed[i, 0] = np.clip(perturbed[i, 0], 
                                        perturbed[i, 2] + BOUNDARY_MARGIN, 
                                        1 - perturbed[i, 2] - BOUNDARY_MARGIN)
                perturbed[i, 1] = np.clip(perturbed[i, 1], 
                                        perturbed[i, 2] + BOUNDARY_MARGIN, 
                                        1 - perturbed[i, 2] - BOUNDARY_MARGIN)
            
            # Validate and update
            if is_valid_configuration(perturbed):
                perturbed_fitness = evaluate_fitness(perturbed)
                if perturbed_fitness > best_fitness:
                    best_fitness = perturbed_fitness
                    best_solution = perturbed.copy()

    # Return the best solution found
    if best_solution is not None:
        return best_solution
    else:
        # Fallback to grid initialization if no good solution was found
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
                    circles[count] = [x, y, r]
                    count += 1
        return circles


# EVOLVE-BLOCK-END