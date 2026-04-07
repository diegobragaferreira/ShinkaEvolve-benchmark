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
    """Validates that all circles are within bounds and don't overlap."""
    n = len(circles)
    
    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if x < r or x > 1 - r or y < r or y > 1 - r:
            return False

    # Check overlap constraints using KDTree for efficiency
    points = circles[:, :2]
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

def quadrant_guided_initialization(n_circles: int) -> np.ndarray:
    """
    Creates initial configuration using quadrant-guided placement
    """
    circles = np.zeros((n_circles, 3))
    
    # Define 9 quadrants
    quadrants = []
    for i in range(3):
        for j in range(3):
            quadrants.append((i, j))
    
    # Distribute circles across quadrants
    circles_per_quadrant = [0] * 9
    for i in range(n_circles):
        circles_per_quadrant[i % 9] += 1
    
    # Fill each quadrant
    idx = 0
    for q_idx, (qi, qj) in enumerate(quadrants):
        # Quadrant boundaries
        x_min = qj * 0.333
        x_max = (qj + 1) * 0.333
        y_min = qi * 0.333
        y_max = (qi + 1) * 0.333
        
        # Number of circles in this quadrant
        count = circles_per_quadrant[q_idx]
        
        # Place circles in this quadrant with strategic positioning
        for i in range(count):
            if idx >= n_circles:
                break
                
            # Strategic placement avoiding corners
            if count == 1:
                # Center the circle
                x = (x_min + x_max) / 2
                y = (y_min + y_max) / 2
                # Max radius based on distance to edges
                min_dist = min(x - x_min, x_max - x, y - y_min, y_max - y)
                r = min_dist * 0.8
            elif count == 2:
                # Place along diagonal
                ratio = (i + 1) / (count + 1)
                x = x_min + (x_max - x_min) * ratio
                y = y_min + (y_max - y_min) * ratio
                min_dist = min(x - x_min, x_max - x, y - y_min, y_max - y)
                r = min_dist * 0.6
            elif count >= 3:
                # Place in triangular arrangement (more complex distribution)
                if i == 0:
                    x = (x_min + x_max) / 2
                    y = (y_min + y_max) / 2
                    min_dist = min(x - x_min, x_max - x, y - y_min, y_max - y)
                    r = min_dist * 0.7
                else:
                    # Place around center with increasing distance from center
                    angle = i * 2 * np.pi / (count - 1) if count > 1 else 0
                    distance = 0.2 + (i % 2) * 0.15
                    x = (x_min + x_max) / 2 + distance * np.cos(angle) * 0.2
                    y = (y_min + y_max) / 2 + distance * np.sin(angle) * 0.2
                    # Ensure within bounds
                    x = max(x_min + 0.01, min(x_max - 0.01, x))
                    y = max(y_min + 0.01, min(y_max - 0.01, y))
                    min_dist = min(x - x_min, x_max - x, y - y_min, y_max - y)
                    r = min_dist * 0.4
            
            # Clip to ensure bounds and positive radius
            x = np.clip(x, 0.01, 0.99)
            y = np.clip(y, 0.01, 0.99)
            r = max(0.001, min(r, 0.1))
            
            circles[idx] = [x, y, r]
            idx += 1
            
            if idx >= n_circles:
                break
    
    # Ensure all circles are valid
    for i in range(n_circles):
        x, y, r = circles[i]
        # Check boundaries
        circles[i, 0] = np.clip(x, r, 1 - r)
        circles[i, 1] = np.clip(y, r, 1 - r)
        circles[i, 2] = max(0.001, circles[i, 2])
    
    return circles

def optimize_quadrant_placement(circles: np.ndarray) -> np.ndarray:
    """
    Optimize circle placement within each quadrant
    """
    optimized = circles.copy()
    
    # Group circles by quadrants (0-8)
    quadrant_groups = [[] for _ in range(9)]
    for i in range(len(optimized)):
        x, y = optimized[i][0], optimized[i][1]
        # Determine which quadrant
        q_row = int(y // 0.333)
        q_col = int(x // 0.333)
        q_idx = q_row * 3 + q_col
        if q_idx < 9:
            quadrant_groups[q_idx].append(i)
    
    # Optimize within each quadrant
    for q_idx, circle_indices in enumerate(quadrant_groups):
        if len(circle_indices) > 0:
            # Get quadrant boundaries
            row = q_idx // 3
            col = q_idx % 3
            x_min = col * 0.333
            x_max = (col + 1) * 0.333
            y_min = row * 0.333
            y_max = (row + 1) * 0.333
            
            if len(circle_indices) == 1:
                # Single circle optimization
                i = circle_indices[0]
                x, y, r = optimized[i]
                # Move towards center while staying within bounds
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                dx = center_x - x
                dy = center_y - y
                # Move only if beneficial
                if abs(dx) > 0.001 or abs(dy) > 0.001:
                    # Try to move closer to center but maintain radius
                    new_x = x + 0.1 * dx
                    new_y = y + 0.1 * dy
                    # Ensure it's still valid
                    new_x = np.clip(new_x, r, 1 - r)
                    new_y = np.clip(new_y, r, 1 - r)
                    # Check if this improves distance to edges
                    min_edge_dist = min(new_x - x_min, x_max - new_x, new_y - y_min, y_max - new_y)
                    if min_edge_dist > r:
                        optimized[i] = [new_x, new_y, r]
            else:
                # Multiple circles in quadrant - position them optimally
                for i in circle_indices:
                    x, y, r = optimized[i]
                    # Try to optimize position
                    best_x, best_y, best_r = x, y, r
                    best_radius = r
                    
                    # Try small movements in 8 directions
                    moves = [(0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01),
                             (0.007, 0.007), (-0.007, -0.007), (0.007, -0.007), (-0.007, 0.007)]
                    
                    # Try to increase radius slightly while maintaining distances
                    for mx, my in moves:
                        test_x = max(x_min + r, min(x_max - r, x + mx))
                        test_y = max(y_min + r, min(y_max - r, y + my))
                        
                        # Try to increase radius
                        test_r = r
                        # Keep exploring
                        if test_r > best_radius:
                            best_x, best_y, best_r = test_x, test_y, test_r
                            best_radius = test_r
                    
                    optimized[i] = [best_x, best_y, best_r]
    
    return optimized

def adaptive_local_search(circles: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """
    Apply adaptive local search with simulated annealing inspired approach
    """
    current = circles.copy()
    current_fitness = calculate_sum_radii(current)
    
    # Temperature schedule
    temp = 1.0
    min_temp = 0.001
    cooling_rate = 0.95
    
    for iteration in range(max_iter):
        # Decrease temperature
        if temp > min_temp:
            temp *= cooling_rate
            
        # Try to improve solution
        improved = False
        
        # Try to increase radii of all circles
        for i in range(len(current)):
            x, y, r = current[i]
            
            # Try to increase radius slightly
            test_r = min(r + 0.001, min(x, 1-x, y, 1-y) * 0.9)
            
            if test_r > r + 1e-6:
                # Test new configuration
                temp_circles = current.copy()
                temp_circles[i, 2] = test_r
                
                if validate_circles(temp_circles):
                    # Accept with probability based on temperature
                    new_fitness = calculate_sum_radii(temp_circles)
                    delta = new_fitness - current_fitness
                    
                    if delta > 0 or np.random.random() < np.exp(delta / temp):
                        current = temp_circles
                        current_fitness = new_fitness
                        improved = True
        
        # If no improvement made in this iteration, stop
        if not improved and iteration > 10:
            break
            
    return current

def geometric_constraint_analysis(circles: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Analyze geometric constraints and suggest improvements
    """
    analysis = {
        'total_overlap': 0,
        'max_overlap': 0,
        'constraint_violations': []
    }
    
    # Build KDTree for fast neighbor queries
    points = circles[:, :2]
    tree = cKDTree(points)
    
    total_overlap = 0
    max_overlap = 0
    
    # Check overlaps
    for i in range(len(circles)):
        x1, y1, r1 = circles[i]
        nearby = tree.query_ball_point([x1, y1], 2*(r1 + 0.01))
        
        for j in nearby:
            if i != j:
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                min_distance = r1 + r2
                
                if distance < min_distance:
                    overlap = min_distance - distance
                    total_overlap += overlap
                    max_overlap = max(max_overlap, overlap)
                    analysis['constraint_violations'].append((i, j, overlap))
    
    analysis['total_overlap'] = total_overlap
    analysis['max_overlap'] = max_overlap
    
    return circles, analysis

def global_optimization_refinement(circles: np.ndarray) -> np.ndarray:
    """
    Perform global refinement using a combination of strategies:
    1. Constraint relaxation and repositioning
    2. Geometric optimization
    3. Radius maximization
    """
    refined = circles.copy()
    
    # Strategy 1: Reposition circles to reduce overlap conflicts
    # Try to move circles away from each other using repulsion forces
    points = refined[:, :2]
    tree = cKDTree(points)
    
    for _ in range(30):  # Limit iterations
        any_changes = False
        for i in range(len(refined)):
            x1, y1, r1 = refined[i]
            
            # Find nearby circles
            nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))
            
            for j in nearby_indices:
                if i != j:
                    x2, y2, r2 = refined[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    min_distance = r1 + r2
                    
                    if distance < min_distance:
                        # Apply repulsion force
                        if distance > 0.001:
                            dx = (x1 - x2) / distance
                            dy = (y1 - y2) / distance
                            
                            # Move them apart
                            move_amount = (min_distance - distance) * 0.3
                            refined[i, 0] += dx * move_amount
                            refined[i, 1] += dy * move_amount
                            refined[j, 0] -= dx * move_amount
                            refined[j, 1] -= dy * move_amount
                            
                            # Clamp to bounds
                            refined[i, 0] = np.clip(refined[i, 0], r1, 1 - r1)
                            refined[i, 1] = np.clip(refined[i, 1], r1, 1 - r1)
                            refined[j, 0] = np.clip(refined[j, 0], r2, 1 - r2)
                            refined[j, 1] = np.clip(refined[j, 1], r2, 1 - r2)
                        
                        any_changes = True
                        
        if not any_changes:
            break
    
    # Strategy 2: Try to maximize radii
    for _ in range(50):
        improved = False
        # Process in shuffled order
        indices = list(range(len(refined)))
        np.random.shuffle(indices)
        
        for i in indices:
            x, y, r = refined[i]
            # Maximum possible radius
            max_r = min(x, 1-x, y, 1-y)
            
            # Try to increase radius
            test_r = min(r + 0.001, max_r * 0.98)
            
            if test_r > r + 1e-6:
                temp_circles = refined.copy()
                temp_circles[i, 2] = test_r
                
                if validate_circles(temp_circles):
                    refined = temp_circles
                    improved = True
        
        if not improved:
            break
    
    return refined


def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Primary approach: quadrant-guided initialization followed by refinement
    
    # Step 1: Create initial configuration using quadrant guidance
    circles = quadrant_guided_initialization(26)
    
    # Step 2: Optimize within quadrants
    circles = optimize_quadrant_placement(circles)
    
    # Step 3: Apply adaptive local search for further improvement
    circles = adaptive_local_search(circles)
    
    # Step 4: Global optimization refinement
    circles = global_optimization_refinement(circles)
    
    # Final validation
    if not validate_circles(circles):
        # If invalid, try repair approach
        circles = repair_invalid_configuration(circles)
    
    # Final refinement pass
    circles = adaptive_local_search(circles, 50)
    
    return circles

def repair_invalid_configuration(circles: np.ndarray) -> np.ndarray:
    """Repair invalid configuration using constraint satisfaction approach"""
    repaired = circles.copy()
    
    # Fix boundary issues first
    for i in range(len(repaired)):
        x, y, r = repaired[i]
        repaired[i, 0] = np.clip(x, r, 1 - r)
        repaired[i, 1] = np.clip(y, r, 1 - r)
        repaired[i, 2] = max(0.001, repaired[i, 2])
    
    # Apply overlap resolution via iterative repulsion
    points = repaired[:, :2]
    tree = cKDTree(points)
    
    # Try up to 50 iterations
    for _ in range(50):
        any_changes = False
        for i in range(len(repaired)):
            x1, y1, r1 = repaired[i]
            nearby = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))
            
            for j in nearby:
                if i != j:
                    x2, y2, r2 = repaired[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    min_distance = r1 + r2
                    
                    if distance < min_distance:
                        # Repel circles apart
                        if distance > 0.001:
                            dx = (x1 - x2) / distance
                            dy = (y1 - y2) / distance
                            
                            move_amount = (min_distance - distance) * 0.5
                            repaired[i, 0] += dx * move_amount
                            repaired[i, 1] += dy * move_amount
                            repaired[j, 0] -= dx * move_amount
                            repaired[j, 1] -= dy * move_amount
                            
                            # Clamp to bounds
                            repaired[i, 0] = np.clip(repaired[i, 0], r1, 1 - r1)
                            repaired[i, 1] = np.clip(repaired[i, 1], r1, 1 - r1)
                            repaired[j, 0] = np.clip(repaired[j, 0], r2, 1 - r2)
                            repaired[j, 1] = np.clip(repaired[j, 1], r2, 1 - r2)
                        
                        any_changes = True
                        
        if not any_changes:
            break
            
    return repaired

# EVOLVE-BLOCK-END