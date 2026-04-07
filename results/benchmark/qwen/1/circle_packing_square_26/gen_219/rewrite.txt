# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import time
from numba import jit
import random
from collections import defaultdict

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

@jit(nopython=True)
def validate_solution_fast(circles):
    """Fast validation using numba for the critical constraint checks"""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        # Check containment
        if r > x or r > y or r > 1-x or r > 1-y:
            return False
        # Check overlap with all previous circles
        for j in range(i):
            x2, y2, r2 = circles[j]
            dx = x - x2
            dy = y - y2
            dist_sq = dx*dx + dy*dy
            min_dist_sq = (r+r2)*(r+r2)
            if dist_sq < min_dist_sq:
                return False
    return True

def _compute_voronoi_cell_areas(points):
    """Compute approximate Voronoi cell areas for given points"""
    try:
        vor = Voronoi(points)
        areas = []
        
        # For each point, compute the area of its Voronoi cell
        for i in range(len(points)):
            region_idx = vor.point_region[i]
            if region_idx != -1 and region_idx < len(vor.regions):
                region = vor.regions[region_idx]
                if len(region) > 2 and -1 not in region:
                    # Extract vertices
                    vertices = vor.vertices[region]
                    if len(vertices) >= 3:
                        # Simple area calculation using shoelace formula
                        x = vertices[:, 0]
                        y = vertices[:, 1]
                        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                        areas.append(area)
                    else:
                        areas.append(1.0)
                else:
                    areas.append(1.0)
            else:
                areas.append(1.0)
        return np.array(areas)
    except:
        return np.ones(len(points))

def _create_voronoi_initialization(n_circles: int, seed: int = 42) -> np.ndarray:
    """Create initial configuration using Voronoi cell area-weighted placement"""
    np.random.seed(seed)
    
    # Generate initial candidate points using a multi-phase approach
    points = []
    
    # Phase 1: Regular grid points
    grid_size = max(5, int(np.ceil(np.sqrt(n_circles * 1.5))))
    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) < n_circles * 3:
                x = (j + 0.5) / grid_size
                y = (i + 0.5) / grid_size
                points.append([x, y])
    
    # Phase 2: Random interior points
    for _ in range(n_circles * 2):
        points.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
    
    # Phase 3: Boundary points for edge coverage
    for _ in range(n_circles // 2):
        side = np.random.randint(0, 4)
        if side == 0:  # Top edge
            points.append([np.random.uniform(0.05, 0.95), 0.95])
        elif side == 1:  # Bottom edge
            points.append([np.random.uniform(0.05, 0.95), 0.05])
        elif side == 2:  # Left edge
            points.append([0.05, np.random.uniform(0.05, 0.95)])
        else:  # Right edge
            points.append([0.95, np.random.uniform(0.05, 0.95)])
    
    points = np.array(points[:n_circles * 5])
    
    # Clip to valid bounds
    points = np.clip(points, 0.05, 0.95)
    
    # Use Voronoi to get well-distributed centers
    try:
        vor = Voronoi(points)
        
        # Calculate cell areas for each point
        cell_areas = _compute_voronoi_cell_areas(points)
        
        # Sort points by cell area (larger area = better for circle placement)
        sorted_indices = np.argsort(cell_areas)[::-1]
        
        # Select top candidates, but also add some random ones for diversity
        selected_indices = sorted_indices[:n_circles]
        if len(selected_indices) < n_circles:
            # Fill remaining slots with random selection
            remaining = n_circles - len(selected_indices)
            random_indices = np.random.choice(len(points), remaining, replace=False)
            selected_indices = np.concatenate([selected_indices, random_indices])
        
        # Get the actual points
        selected_points = points[selected_indices[:n_circles]]
        
    except Exception:
        # Fallback to direct sampling
        selected_indices = np.random.choice(len(points), n_circles, replace=False)
        selected_points = points[selected_indices]
    
    # Add jitter to prevent symmetry issues
    jitter_magnitude = 0.015
    selected_points += np.random.uniform(-jitter_magnitude, jitter_magnitude, selected_points.shape)
    
    # Ensure all points are within bounds
    selected_points[:, 0] = np.clip(selected_points[:, 0], 0.05, 0.95)
    selected_points[:, 1] = np.clip(selected_points[:, 1], 0.05, 0.95)
    
    return selected_points

def _compute_overlap_penalty(circles):
    """Efficiently compute overlap penalty using spatial indexing"""
    n = len(circles)
    penalty = 0
    
    # Use KDTree for efficient neighbor search
    positions = circles[:, :2]
    radii = circles[:, 2]
    
    try:
        # Create KDTree for fast nearest neighbor queries
        from scipy.spatial import cKDTree
        tree = cKDTree(positions)
        
        # Query pairs within radius sum
        pairs = tree.query_pairs(2.0, p=2)  # Query pairs with distance < 2.0
        
        for i, j in pairs:
            if i < j:  # Only process each pair once
                r_i = radii[i]
                r_j = radii[j]
                pos_i = positions[i]
                pos_j = positions[j]
                dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                
                if dist < (r_i + r_j):
                    overlap = (r_i + r_j) - dist
                    penalty += 1000 * overlap  # Strong penalty for overlaps
                    
    except:
        # Fallback to brute force for edge cases
        for i in range(n):
            for j in range(i+1, n):
                r_i = radii[i]
                r_j = radii[j]
                pos_i = positions[i]
                pos_j = positions[j]
                dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                
                if dist < (r_i + r_j):
                    overlap = (r_i + r_j) - dist
                    penalty += 1000 * overlap
                    
    return penalty

def _evaluate_fitness(circles):
    """Evaluate fitness with proper constraint handling"""
    # Extract positions and radii
    positions = circles[:, :2]
    radii = circles[:, 2]
    
    # Calculate objective (sum of radii)
    total_radius = np.sum(radii)
    
    # Compute penalty for constraint violations
    penalty = _compute_overlap_penalty(circles)
    
    # Compute containment penalty (weak penalty for being near boundaries)
    containment_penalty = 0
    margin = 0.01
    for i, (pos, r) in enumerate(zip(positions, radii)):
        x, y = pos
        # Add weak penalty if circle is too close to boundary
        if x - r < margin or x + r > 1 - margin or y - r < margin or y + r > 1 - margin:
            containment_penalty += 50 * (margin - min(x-r, 1-x-r, y-r, 1-y-r))
    
    return total_radius - penalty - containment_penalty

def _get_safe_radius(pos, radii, positions, max_radius):
    """Calculate maximum safe radius for a position considering overlaps"""
    x, y = pos
    # Maximum radius based on boundaries
    max_safe = min(x, y, 1-x, 1-y)
    max_safe = min(max_safe, max_radius)
    
    # Check overlaps with existing circles
    for i, (other_pos, other_r) in enumerate(zip(positions, radii)):
        if i < len(positions) and i < len(radii):
            dist = np.sqrt((x - other_pos[0])**2 + (y - other_pos[1])**2)
            if dist > 0:  # Avoid self-interaction
                max_safe = min(max_safe, dist - other_r)
    
    return max(0.001, max_safe)

def _improve_radius_assignment(circles):
    """Iteratively improve radius assignments"""
    improved = True
    iterations = 0
    max_iterations = 20
    
    while improved and iterations < max_iterations:
        improved = False
        new_circles = circles.copy()
        
        # Process in random order for better convergence
        indices = list(range(len(circles)))
        np.random.shuffle(indices)
        
        for i in indices:
            x, y, r = new_circles[i]
            
            # Try to increase radius
            positions = new_circles[:, :2]
            radii = new_circles[:, 2]
            
            # Compute maximum possible radius
            max_radius = _get_safe_radius([x, y], radii, positions, 0.5)
            
            if max_radius > r + 0.001:  # Allow small improvements only
                # Binary search for optimal radius
                low = r
                high = max_radius
                best_radius = r
                
                # Try to find maximum safe radius
                for _ in range(10):
                    test_r = (low + high) / 2
                    # Check if this radius works with all others
                    valid = True
                    for j, (other_pos, other_r) in enumerate(zip(positions, radii)):
                        if i != j:
                            dist = np.sqrt((x - other_pos[0])**2 + (y - other_pos[1])**2)
                            if dist < (test_r + other_r):
                                valid = False
                                break
                    
                    if valid:
                        best_radius = test_r
                        low = test_r
                    else:
                        high = test_r
                
                if best_radius > r:
                    new_circles[i, 2] = best_radius
                    improved = True
        
        if improved:
            circles = new_circles
        iterations += 1
        
    return circles

def _refine_positions(circles, max_iterations=100):
    """Refine positions using a gradient-like approach"""
    positions = circles[:, :2]
    radii = circles[:, 2]
    
    # Simple force-based refinement
    for iteration in range(max_iterations):
        forces = np.zeros_like(positions)
        
        # Compute forces from neighbors
        for i in range(len(positions)):
            x, y = positions[i]
            r = radii[i]
            
            # Attraction to center
            center_force = np.array([0.5, 0.5]) - np.array([x, y])
            force_magnitude = 0.002 / (np.linalg.norm(center_force) + 1e-8)
            forces[i] += center_force * force_magnitude
            
            # Repulsion from neighbors
            for j in range(len(positions)):
                if i != j:
                    x2, y2 = positions[j]
                    r2 = radii[j]
                    dx = x2 - x
                    dy = y2 - y
                    dist = np.sqrt(dx*dx + dy*dy) + 1e-8
                    
                    if dist < r + r2 + 0.001:
                        force_magnitude = 0.03 / (dist * dist + 1e-6)
                        forces[i] -= np.array([dx, dy]) * force_magnitude
        
        # Apply forces
        for i in range(len(positions)):
            x, y = positions[i]
            fx, fy = forces[i]
            
            # Apply boundary constraints
            new_x = np.clip(x + fx * 0.1, radii[i], 1 - radii[i])
            new_y = np.clip(y + fy * 0.1, radii[i], 1 - radii[i])
            
            positions[i] = [new_x, new_y]
            
        # Update circles
        circles[:, :2] = positions
        
        # Stop early if no significant changes
        if iteration > 10:
            # Check if changes are small
            max_change = np.max(np.abs(positions - circles[:, :2]))
            if max_change < 1e-5:
                break
                
    return circles

def _construct_solution_from_voronoi(n_circles):
    """Construct a good initial solution based on Voronoi analysis"""
    # Generate initial points using Voronoi approach
    initial_points = _create_voronoi_initialization(n_circles, seed=42)
    
    # Create circles with initial radii based on Voronoi cell properties  
    circles = np.zeros((n_circles, 3))
    
    # Get Voronoi information
    try:
        vor = Voronoi(initial_points)
        cell_areas = _compute_voronoi_cell_areas(initial_points)
        
        # Assign radii inversely proportional to cell areas (smaller cells get larger radii)
        normalized_areas = cell_areas / np.max(cell_areas)
        # Map areas to radii: smaller areas = larger radii
        radii = 0.15 * (1 - normalized_areas) + 0.02
        radii = np.clip(radii, 0.01, 0.2)  # Bound radii
        
    except:
        # Fallback to simple uniform radii
        radii = np.full(n_circles, 0.08)
    
    # Ensure radii are compatible with positions
    for i, (point, r) in enumerate(zip(initial_points, radii)):
        x, y = point
        # Make sure radius doesn't violate physical constraints
        max_radius = min(x, y, 1-x, 1-y)
        circles[i] = [x, y, min(r, max_radius)]
    
    return circles

def _local_search_optimization(circles):
    """Perform local search optimization for better solutions"""
    best_circles = circles.copy()
    best_fitness = _evaluate_fitness(best_circles)
    
    # Multiple local search rounds
    for round_num in range(10):
        # Try different neighborhood operations
        
        # 1. Position refinement
        refined = _refine_positions(best_circles.copy())
        refined_fitness = _evaluate_fitness(refined)
        if refined_fitness > best_fitness:
            best_circles = refined
            best_fitness = refined_fitness
        
        # 2. Radius improvement
        improved_radii = _improve_radius_assignment(best_circles.copy())
        improved_fitness = _evaluate_fitness(improved_radii)
        if improved_fitness > best_fitness:
            best_circles = improved_radii
            best_fitness = improved_fitness
            
        # 3. Hybrid operation: combine position and radius changes
        hybrid = best_circles.copy()
        # Slight perturbation to positions
        for i in range(len(hybrid)):
            if np.random.random() < 0.3:
                hybrid[i, 0] += np.random.normal(0, 0.01)
                hybrid[i, 1] += np.random.normal(0, 0.01)
                # Clamp to bounds
                hybrid[i, 0] = np.clip(hybrid[i, 0], hybrid[i, 2], 1 - hybrid[i, 2])
                hybrid[i, 1] = np.clip(hybrid[i, 1], hybrid[i, 2], 1 - hybrid[i, 2])
        
        hybrid_fitness = _evaluate_fitness(hybrid)
        if hybrid_fitness > best_fitness:
            best_circles = hybrid
            best_fitness = hybrid_fitness
            
    return best_circles

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # Phase 1: Construct initial solution using Voronoi-based approach
    circles = _construct_solution_from_voronoi(n)
    
    # Phase 2: Optimize using local search
    circles = _local_search_optimization(circles)
    
    # Phase 3: Final validation and refinement
    if not validate_solution_fast(circles):
        # If validation fails, construct an alternative solution
        circles = _construct_solution_from_voronoi(n)
        circles = _local_search_optimization(circles)
    
    # Ensure final solution is valid
    final_circles = circles.copy()
    final_circles = _refine_positions(final_circles, max_iterations=50)
    final_circles = _improve_radius_assignment(final_circles)
    
    # Final validation
    if not validate_solution_fast(final_circles):
        # Last resort: create a simple grid-based solution
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing = 1.0 / grid_size
        r = spacing * 0.3
        count = 0
        final_circles = np.zeros((n, 3))
        for i in range(grid_size):
            for j in range(grid_size):
                if count < n:
                    x = (j + 0.5) * spacing
                    y = (i + 0.5) * spacing
                    final_circles[count] = [x, y, r]
                    count += 1
    
    return final_circles

# EVOLVE-BLOCK-END