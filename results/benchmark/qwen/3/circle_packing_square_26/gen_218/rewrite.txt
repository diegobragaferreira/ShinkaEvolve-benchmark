# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance_matrix
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import math

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_poisson_points(n_points: int, max_iter: int = 10000) -> np.ndarray:
    """Generate Poisson-distributed points using rejection sampling."""
    points = []
    attempts = 0
    
    while len(points) < n_points and attempts < max_iter:
        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)
        
        # Check minimum distance to existing points
        if not points:
            points.append([x, y])
        else:
            min_dist = min(math.sqrt((x - px)**2 + (y - py)**2) for px, py in points)
            if min_dist > 0.1:  # Minimum distance threshold
                points.append([x, y])
                
        attempts += 1
    
    # If we didn't get enough points, add random ones
    while len(points) < n_points:
        points.append([random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)])
        
    return np.array(points[:n_points])

def voronoi_circle_initialization(n_circles: int) -> np.ndarray:
    """Initialize circles using Voronoi diagram approach."""
    # Generate Poisson points for Voronoi seeds
    seed_points = generate_poisson_points(n_circles)
    
    # Create Voronoi diagram
    vor = Voronoi(seed_points)
    
    # Get Voronoi vertices and regions
    circles = np.zeros((n_circles, 3))
    
    # For each Voronoi cell, place a circle at its centroid
    for i in range(n_circles):
        region = vor.point_region[i]
        if region < len(vor.regions) and vor.regions[region]:
            # Get vertices of this region
            vertices = np.array(vor.vertices[vor.regions[region]])
            
            # Skip infinite regions
            if len(vertices) > 2:
                # Compute centroid of finite region
                centroid_x = np.mean(vertices[:, 0])
                centroid_y = np.mean(vertices[:, 1])
                
                # Calculate approximate radius based on region size
                # Use the minimum distance to any vertex as proxy for radius
                if len(vertices) > 0:
                    distances = np.sqrt((vertices[:, 0] - centroid_x)**2 + (vertices[:, 1] - centroid_y)**2)
                    avg_distance = np.mean(distances)
                    
                    # Scale radius based on area and constrain to unit square
                    max_radius = min(
                        centroid_x, 
                        1 - centroid_x, 
                        centroid_y, 
                        1 - centroid_y
                    )
                    radius = min(avg_distance * 0.5, max_radius * 0.8)
                    radius = max(0.001, min(0.4, radius))
                else:
                    radius = 0.05
                    
                circles[i] = [centroid_x, centroid_y, radius]
            else:
                # Fallback for degenerate cases
                circles[i] = [seed_points[i][0], seed_points[i][1], 0.05]
        else:
            # Fallback for missing regions
            circles[i] = [seed_points[i][0], seed_points[i][1], 0.05]
    
    # Ensure all circles are valid (within bounds)
    for i in range(n_circles):
        x, y, r = circles[i]
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        circles[i] = [x, y, r]
    
    return circles

def is_valid_configuration(circles: np.ndarray) -> bool:
    """Check if all circles are fully contained and non-overlapping."""
    n = len(circles)
    
    # Check containment
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    
    # Check overlaps using distance matrix
    if n > 1:
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Compute pairwise distances
        dist_matrix = distance_matrix(positions, positions)
        
        # Check all pairs for overlap
        for i in range(n):
            for j in range(i+1, n):
                distance = dist_matrix[i, j]
                if distance < (radii[i] + radii[j]):
                    return False
    
    return True

def calculate_sum_radii(circles: np.ndarray) -> float:
    """Calculate the sum of all radii."""
    return np.sum(circles[:, 2])

def compute_overlap_penalty(circles: np.ndarray) -> float:
    """Compute penalty based on overlap amount."""
    penalty = 0.0
    n = len(circles)
    
    if n > 1:
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        dist_matrix = distance_matrix(positions, positions)
        
        for i in range(n):
            for j in range(i+1, n):
                distance = dist_matrix[i, j]
                if distance < (radii[i] + radii[j]):
                    overlap = (radii[i] + radii[j]) - distance
                    penalty += overlap**2 * 100000
    
    return penalty

def evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate fitness of a solution (higher is better)."""
    if not is_valid_configuration(circles):
        # Apply penalty for constraint violations
        penalty = 0
        
        # Boundary penalty
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0:
                penalty += (r - x)**2 * 100000
            if x + r > 1:
                penalty += (x + r - 1)**2 * 100000
            if y - r < 0:
                penalty += (r - y)**2 * 100000
            if y + r > 1:
                penalty += (y + r - 1)**2 * 100000
                
        penalty += compute_overlap_penalty(circles)
        return -penalty - 1000000
    
    return calculate_sum_radii(circles)

def get_voronoi_neighbors(circles: np.ndarray, idx: int, vor: Voronoi) -> List[int]:
    """Get Voronoi neighbors for a specific circle."""
    neighbors = []
    # This is a simplified approach - in practice you'd use Voronoi topology
    # For now, we'll find nearby circles via Euclidean distance
    positions = circles[:, :2]
    distances = np.sqrt(np.sum((positions - positions[idx])**2, axis=1))
    # Get indices of 5 nearest neighbors (excluding self)
    nearest_indices = np.argsort(distances)[1:6]
    neighbors = [int(i) for i in nearest_indices if i != idx]
    return neighbors

def local_search_step(circles: np.ndarray, max_iterations: int = 50) -> np.ndarray:
    """Perform local search optimization on circle configuration."""
    current = circles.copy()
    best = current.copy()
    best_fitness = evaluate_fitness(best)
    
    # Simulated annealing parameters
    temperature = 1.0
    cooling_rate = 0.95
    
    for iteration in range(max_iterations):
        # Try to make small random adjustments
        new_circles = current.copy()
        
        # Pick a random circle to modify
        circle_idx = random.randint(0, len(new_circles) - 1)
        
        # Make small random changes to position and radius
        delta_x = random.uniform(-0.01, 0.01)
        delta_y = random.uniform(-0.01, 0.01)
        delta_r = random.uniform(-0.005, 0.005)
        
        # Apply changes
        new_circles[circle_idx, 0] += delta_x
        new_circles[circle_idx, 1] += delta_y
        new_circles[circle_idx, 2] += delta_r
        
        # Ensure constraints
        x, y, r = new_circles[circle_idx]
        r = max(0.001, min(0.49, r))
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        new_circles[circle_idx] = [x, y, r]
        
        # Evaluate new configuration
        new_fitness = evaluate_fitness(new_circles)
        
        # Accept or reject based on simulated annealing
        if new_fitness > best_fitness:
            current = new_circles.copy()
            best = current.copy()
            best_fitness = new_fitness
        elif random.random() < math.exp((new_fitness - best_fitness) / max(1e-8, temperature)):
            current = new_circles.copy()
            
        # Cool down
        temperature *= cooling_rate
    
    return best

def constrained_radius_adjustment(circles: np.ndarray) -> np.ndarray:
    """Adjust radii to maximize sum while respecting constraints."""
    adjusted = circles.copy()
    
    # Iteratively adjust radii while maintaining validity
    for _ in range(20):
        valid = True
        
        # First check if valid
        if not is_valid_configuration(adjusted):
            valid = False
        
        if valid:
            break
            
        # If invalid, reduce all radii slightly
        for i in range(len(adjusted)):
            adjusted[i, 2] = max(0.001, adjusted[i, 2] * 0.95)
    
    # Then try to increase radii where possible
    for _ in range(20):
        improved = False
        for i in range(len(adjusted)):
            x, y, r = adjusted[i]
            
            # Try to increase radius while staying valid
            max_radius = min(x, 1-x, y, 1-y)
            new_r = min(r * 1.05, max_radius * 0.95)
            
            if new_r > r:
                # Temporarily test with increased radius
                temp_circles = adjusted.copy()
                temp_circles[i, 2] = new_r
                
                if is_valid_configuration(temp_circles):
                    adjusted[i, 2] = new_r
                    improved = True
        
        if not improved:
            break
    
    return adjusted

def voronoi_evolve_optimize(max_iterations: int = 1000) -> np.ndarray:
    """Main optimization function using Voronoi-based approach."""
    n = 26
    
    # Start with Voronoi initialization
    circles = voronoi_circle_initialization(n)
    
    # Refine initial configuration
    circles = constrained_radius_adjustment(circles)
    
    # Better local search
    best_circles = local_search_step(circles, max_iterations=200)
    best_fitness = evaluate_fitness(best_circles)
    
    print(f"Initial fitness: {calculate_sum_radii(circles):.6f}")
    print(f"After initial refinement: {calculate_sum_radii(best_circles):.6f}")
    
    # Progressive refinement
    for iteration in range(100):
        # Apply local search multiple times
        refined = local_search_step(best_circles, max_iterations=100)
        new_fitness = evaluate_fitness(refined)
        
        if new_fitness > best_fitness:
            best_circles = refined
            best_fitness = new_fitness
            print(f"Iteration {iteration}: New best fitness = {best_fitness:.6f}")
        
        # Break if no improvement
        if abs(new_fitness - best_fitness) < 0.001:
            break
    
    # Final adjustment with constrained optimization
    final_circles = constrained_radius_adjustment(best_circles)
    
    # Ensure final validity
    if not is_valid_configuration(final_circles):
        # Apply final geometric correction
        final_circles = correct_positions_and_radii(final_circles)
    
    return final_circles

def correct_positions_and_radii(circles: np.ndarray) -> np.ndarray:
    """Correct positions and radii to meet all constraints."""
    corrected = circles.copy()
    
    # First handle containment
    for i in range(len(corrected)):
        x, y, r = corrected[i]
        # Adjust for boundaries
        r = min(r, x, 1-x, y, 1-y)
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        corrected[i] = [x, y, r]
    
    # Resolve overlaps by iterative adjustment
    for _ in range(50):
        any_changes = False
        for i in range(len(corrected)):
            for j in range(i+1, len(corrected)):
                x1, y1, r1 = corrected[i]
                x2, y2, r2 = corrected[j]
                
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                if distance < (r1 + r2):
                    # Move circles apart
                    if distance > 0.001:
                        dx = (x2 - x1) / distance
                        dy = (y2 - y1) / distance
                        move_amount = (r1 + r2 - distance) * 0.5
                        
                        # Apply movement with damping
                        corrected[i, 0] -= dx * move_amount * 0.3
                        corrected[i, 1] -= dy * move_amount * 0.3
                        corrected[j, 0] += dx * move_amount * 0.3
                        corrected[j, 1] += dy * move_amount * 0.3
                        any_changes = True
        
        if not any_changes:
            break
    
    # Final boundary correction
    for i in range(len(corrected)):
        x, y, r = corrected[i]
        r = min(r, x, 1-x, y, 1-y)
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        corrected[i] = [x, y, r]
    
    return corrected

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Run the Voronoi-evolve optimization
    circles = voronoi_evolve_optimize(max_iterations=1000)
    
    # Validate final result
    if not is_valid_configuration(circles):
        # If somehow still invalid, use fallback
        circles = np.zeros((26, 3))
        rows = 5
        cols = 5
        spacing_x = 0.9 / (cols + 1)
        spacing_y = 0.9 / (rows + 1)
        radius = min(spacing_x, spacing_y) * 0.35
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= 26:
                    break
                x = 0.05 + (j + 1) * spacing_x
                y = 0.05 + (i + 1) * spacing_y
                circles[idx] = [x, y, radius]
                idx += 1
        
        # Adjust last few circles to fit
        for i in range(idx, 26):
            circles[i] = [0.5, 0.5, 0.015]
    
    return circles

# EVOLVE-BLOCK-END