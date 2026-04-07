# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
import math
from typing import Tuple, List

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def validate_circles(circles: np.ndarray) -> bool:
    """
    Validates that all circles are within bounds and don't overlap.
    Uses efficient spatial indexing for overlap checking.
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

def create_adaptive_grid_initialization(n_circles: int) -> np.ndarray:
    """Create initial configuration using an adaptive grid strategy."""
    circles = np.zeros((n_circles, 3))
    
    # Try multiple grid configurations to find best starting point
    grid_configs = [(2, 13), (3, 9), (4, 7), (5, 6), (6, 5), (7, 4), (9, 3), (13, 2)]
    
    # Use the configuration that best fits our target count
    best_config = None
    min_diff = float('inf')
    
    for height, width in grid_configs:
        total_cells = height * width
        diff = abs(total_cells - n_circles)
        if diff < min_diff:
            min_diff = diff
            best_config = (height, width)
    
    grid_height, grid_width = best_config
    
    # Ensure we don't exceed n_circles
    actual_cols = min(grid_width, n_circles // grid_height + (1 if n_circles % grid_height else 0))
    actual_rows = min(grid_height, (n_circles + actual_cols - 1) // actual_cols)  # Ceiling division

    spacing_x = 1.0 / (actual_cols + 1)
    spacing_y = 1.0 / (actual_rows + 1)

    # Place circles on the grid with some randomness
    idx = 0
    for i in range(actual_rows):
        for j in range(actual_cols):
            if idx >= n_circles:
                break

            # Position in grid cell with adaptive perturbation  
            x = (j + 1) * spacing_x + np.random.uniform(-spacing_x/6, spacing_x/6)
            y = (i + 1) * spacing_y + np.random.uniform(-spacing_y/6, spacing_y/6)

            # Ensure within bounds
            x = np.clip(x, 0.02, 0.98)
            y = np.clip(y, 0.02, 0.98)

            # Assign initial radius based on proximity to edges
            min_dist_to_edge = min(x, 1-x, y, 1-y)
            # Distribute radii more strategically - central areas can have larger initial radii
            if min_dist_to_edge > 0.3:
                # Central region: larger initial radii
                r = min(0.08, min_dist_to_edge * np.random.uniform(0.7, 0.9))
            else:
                # Edge regions: smaller initial radii
                r = min(0.05, min_dist_to_edge * np.random.uniform(0.5, 0.8))
                
            circles[idx] = [x, y, r]
            idx += 1

        if idx >= n_circles:
            break
    
    # Fill remaining circles strategically
    for i in range(idx, n_circles):
        # Prefer corner/edge placement for better spread initially
        if np.random.rand() < 0.4:  # 40% chance for strategic placement
            # Place near edges/corners
            edge_positions = [
                (0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9),
                (0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5)
            ]
            edge_x, edge_y = edge_positions[np.random.randint(0, len(edge_positions))]
            x = edge_x + np.random.uniform(-0.07, 0.07)
            y = edge_y + np.random.uniform(-0.07, 0.07)
        else:
            # Random placement but with better bounds enforcement
            x = np.random.uniform(0.15, 0.85)
            y = np.random.uniform(0.15, 0.85)

        x = np.clip(x, 0.02, 0.98)
        y = np.clip(y, 0.02, 0.98)

        min_dist_to_edge = min(x, 1-x, y, 1-y)
        r = min(0.06, min_dist_to_edge * np.random.uniform(0.4, 0.7))

        circles[i] = [x, y, r]
    
    # Apply preliminary improvement through local optimizations
    improve_initial_configuration(circles, 3)
    
    return circles

def improve_initial_configuration(circles: np.ndarray, max_passes: int = 3) -> None:
    """Improve initial configuration by locally optimizing radii."""
    n_circles = len(circles)
    
    for _ in range(max_passes):
        improved = False
        # Process in shuffled order to avoid bias
        indices = list(range(n_circles))
        np.random.shuffle(indices)
        
        for i in indices:
            original_r = circles[i, 2]
            # Try to increase radius safely
            potential_r = original_r * 1.1
            
            # Check if we can increase the radius
            can_increase = True
            for j in range(n_circles):
                if i != j:
                    distance = np.sqrt((circles[i, 0] - circles[j, 0])**2 +
                                     (circles[i, 1] - circles[j, 1])**2)
                    if distance < (potential_r + circles[j, 2]):
                        can_increase = False
                        break

            if can_increase:
                # Check boundary constraints
                min_edge_dist = min(circles[i, 0], 1-circles[i, 0],
                                  circles[i, 1], 1-circles[i, 1])
                if potential_r <= min_edge_dist * 0.95:
                    circles[i, 2] = potential_r
                    improved = True
        
        if not improved:
            break

def apply_forces(circles: np.ndarray, dt: float = 0.02) -> np.ndarray:
    """
    Apply physics-based forces to circles to resolve overlaps.
    Each circle experiences repulsion from others, bounded by edges.
    """
    n = len(circles)
    forces = np.zeros((n, 2))  # Force vectors for each circle
    
    # Repulsion forces between circles
    for i in range(n):
        x1, y1, r1 = circles[i]
        
        # Apply repulsion from other circles
        for j in range(n):
            if i != j:
                x2, y2, r2 = circles[j]
                dx = x1 - x2
                dy = y1 - y2
                distance = np.sqrt(dx*dx + dy*dy)
                
                # Only repel if circles are too close
                if distance > 0 and distance < (r1 + r2):
                    # Inverse-square law for repulsion (stronger when closer)
                    force_magnitude = 100.0 / (distance * distance + 0.01)
                    # Normalize direction
                    fx = force_magnitude * dx / distance
                    fy = force_magnitude * dy / distance
                    
                    forces[i, 0] += fx
                    forces[i, 1] += fy
        
        # Edge repulsion - make circles stay within bounds
        edge_force_scale = 50.0
        
        # Left edge
        if x1 < r1:
            forces[i, 0] += edge_force_scale * (r1 - x1)
        # Right edge
        elif x1 > 1 - r1:
            forces[i, 0] += edge_force_scale * (1 - r1 - x1)
            
        # Bottom edge
        if y1 < r1:
            forces[i, 1] += edge_force_scale * (r1 - y1)
        # Top edge
        elif y1 > 1 - r1:
            forces[i, 1] += edge_force_scale * (1 - r1 - y1)
    
    # Apply forces to update positions
    updated_circles = circles.copy()
    for i in range(n):
        # Limit maximum velocity to prevent instability
        max_velocity = 0.02
        
        # Update positions using Verlet integration with damping
        velocity_x = forces[i, 0] * dt
        velocity_y = forces[i, 1] * dt
        
        # Clamp velocity
        velocity_x = np.clip(velocity_x, -max_velocity, max_velocity)
        velocity_y = np.clip(velocity_y, -max_velocity, max_velocity)
        
        # Apply position updates
        updated_circles[i, 0] += velocity_x
        updated_circles[i, 1] += velocity_y
        
        # Keep within bounds with margin
        updated_circles[i, 0] = np.clip(updated_circles[i, 0], 
                                       updated_circles[i, 2], 
                                       1 - updated_circles[i, 2])
        updated_circles[i, 1] = np.clip(updated_circles[i, 1], 
                                       updated_circles[i, 2], 
                                       1 - updated_circles[i, 2])
    
    return updated_circles

def adaptive_local_search(circles: np.ndarray, max_iterations: int = 200) -> np.ndarray:
    """
    Apply physics-guided local search with multi-scale approach.
    """
    current = circles.copy()
    best = circles.copy()
    best_fitness = calculate_sum_radii(best)
    
    # Multi-scale optimization - coarse to fine
    scales = [0.8, 0.5, 0.2, 0.1]
    
    for scale in scales:
        # Reduce time step for finer resolution
        dt = 0.005 * scale
        
        # Apply physics simulation with reduced scale
        for iteration in range(max_iterations // len(scales)):
            # Apply physics forces with appropriate scale
            updated = apply_forces(current, dt)
            
            # Check if we've improved
            current_fitness = calculate_sum_radii(updated)
            if current_fitness > best_fitness:
                best = updated.copy()
                best_fitness = current_fitness
                
            current = updated
            
            # Occasionally restore best solution if quality degrades
            if iteration % 50 == 0 and current_fitness < best_fitness * 0.99:
                current = best.copy()
    
    return best

def physics_based_optimization(initial_circles: np.ndarray, max_iterations: int = 500) -> np.ndarray:
    """
    Main optimization routine using physics-based approach.
    """
    # Start with improved initial configuration
    circles = initial_circles.copy()
    
    # Multi-stage optimization
    stages = [
        {"iterations": 100, "dt": 0.03, "scale": 0.8},
        {"iterations": 150, "dt": 0.02, "scale": 0.5},  
        {"iterations": 200, "dt": 0.01, "scale": 0.2}
    ]
    
    best_circles = circles.copy()
    best_fitness = calculate_sum_radii(best_circles)
    
    for stage in stages:
        iterations = stage["iterations"]
        dt = stage["dt"] 
        scale = stage["scale"]
        
        # Apply physics simulation for this stage
        for i in range(iterations):
            # Apply forces with current parameters
            updated = apply_forces(circles, dt)
            
            # Check if we've improved
            current_fitness = calculate_sum_radii(updated)
            if current_fitness > best_fitness:
                best_circles = updated.copy()
                best_fitness = current_fitness
                
            circles = updated
            
            # Occasionally apply additional refinement
            if i % 20 == 0:
                circles = adaptive_local_search(circles, 10)
    
    return best_circles

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Create initial configuration using adaptive grid
    initial_config = create_adaptive_grid_initialization(26)
    
    # Apply physics-based optimization
    solution = physics_based_optimization(initial_config, max_iterations=500)
    
    # Final validation and improvement
    if not validate_circles(solution):
        # If invalid, fix it
        solution = fix_invalid_configuration(solution)
    
    # Apply final local refinement
    solution = adaptive_local_search(solution, 50)
    
    # Final validation
    if not validate_circles(solution):
        # If still invalid, perform one last fix
        solution = fix_invalid_configuration(solution)
    
    return solution

def fix_invalid_configuration(circles: np.ndarray) -> np.ndarray:
    """Fix invalid configurations through constraint-aware repulsion."""
    # First ensure boundary conditions are met
    repaired = circles.copy()
    
    for i in range(len(repaired)):
        x, y, r = repaired[i]
        repaired[i, 0] = np.clip(x, r, 1 - r)
        repaired[i, 1] = np.clip(y, r, 1 - r)
        repaired[i, 2] = max(0.001, repaired[i, 2])
    
    # Iteratively resolve overlaps using repulsion
    points = repaired[:, :2]
    tree = cKDTree(points)
    
    # Try several rounds of overlap resolution
    for _ in range(15):
        changes = 0
        for i in range(len(repaired)):
            x1, y1, r1 = repaired[i]
            
            # Find nearby circles
            nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))
            
            for j in nearby_indices:
                if i != j:
                    x2, y2, r2 = repaired[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    min_distance = r1 + r2
                    
                    if distance < min_distance:
                        # Repel circles apart
                        if distance > 0.001:
                            dx = (x1 - x2) / distance
                            dy = (y1 - y2) / distance
                            
                            move_amount = (min_distance - distance) * 0.3
                            repaired[i, 0] += dx * move_amount
                            repaired[i, 1] += dy * move_amount
                            repaired[j, 0] -= dx * move_amount
                            repaired[j, 1] -= dy * move_amount
                            
                            # Clamp to bounds
                            repaired[i, 0] = np.clip(repaired[i, 0], r1, 1 - r1)
                            repaired[i, 1] = np.clip(repaired[i, 1], r1, 1 - r1)
                            repaired[j, 0] = np.clip(repaired[j, 0], r2, 1 - r2)
                            repaired[j, 1] = np.clip(repaired[j, 1], r2, 1 - r2)
                            
                        changes += 1
                        
        if changes == 0:
            break
    
    # Final boundary check
    for i in range(len(repaired)):
        x, y, r = repaired[i]
        repaired[i, 0] = np.clip(x, r, 1 - r)
        repaired[i, 1] = np.clip(y, r, 1 - r)
        repaired[i, 2] = max(0.001, repaired[i, 2])
    
    return repaired

# EVOLVE-BLOCK-END