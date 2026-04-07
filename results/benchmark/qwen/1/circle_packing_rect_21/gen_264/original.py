# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import time
from typing import Tuple, List, Optional
import math

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    n = 21
    # Optimized rectangle dimensions - slightly wider than taller for better packing
    rect_width = 1.2
    rect_height = 0.8
    
    # Phase 1: Advanced multi-pattern initialization with smart seeding
    circles = np.zeros((n, 3))
    
    # Strategy 1: Hexagonal pattern (dense initial configuration)
    hex_circles = initialize_hexagonal(n, rect_width, rect_height)
    
    # Strategy 2: Spiral pattern (better edge coverage)
    spiral_circles = initialize_spiral(n, rect_width, rect_height)
    
    # Strategy 3: Square grid pattern (balanced distribution)
    square_circles = initialize_square_grid(n, rect_width, rect_height)
    
    # Strategy 4: Random with smart spacing (diversity boost)
    random_circles = initialize_smart_random(n, rect_width, rect_height)
    
    # Evaluate and select best initialization
    initial_strategies = [
        ("hex", hex_circles),
        ("spiral", spiral_circles), 
        ("square", square_circles),
        ("random", random_circles)
    ]
    
    best_initialization = hex_circles.copy()
    best_sum = np.sum(hex_circles[:, 2])
    
    for name, candidate in initial_strategies:
        candidate_sum = np.sum(candidate[:, 2])
        if candidate_sum > best_sum:
            best_sum = candidate_sum
            best_initialization = candidate.copy()
    
    circles = best_initialization.copy()
    
    # Phase 2: Multi-stage optimization with progressive constraint relaxation
    max_iterations_phase1 = 200  # Increased iterations for better exploration
    max_iterations_phase2 = 150  # Reduced for faster convergence
    
    # Phase 1: Progressive constraint relaxation with Voronoi awareness
    for iteration in range(max_iterations_phase1):
        improved = False
        # Start with 10% constraint violations tolerance, decrease to 0%
        violation_tolerance = max(0.0, 1.0 - (iteration / max_iterations_phase1) * 0.9)
        
        # Shuffle circle indices for diverse optimization
        indices = list(range(n))
        np.random.shuffle(indices)
        
        # Precompute Voronoi areas for this iteration (only when needed)
        if iteration % 10 == 0:  # Update Voronoi every 10 iterations for efficiency
            try:
                points = circles[:, :2]
                vor = Voronoi(points)
                voronoi_areas = np.zeros(n)
                for i in range(n):
                    region_idx = vor.point_region[i]
                    if region_idx < len(vor.regions) and -1 not in vor.regions[region_idx]:
                        region = vor.regions[region_idx]
                        if len(region) > 2:
                            vertices = np.array([vor.vertices[j] for j in region])
                            if len(vertices) >= 3:
                                x = vertices[:, 0]
                                y = vertices[:, 1]
                                area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                                voronoi_areas[i] = area
                            else:
                                voronoi_areas[i] = 100.0
                        else:
                            voronoi_areas[i] = 100.0
                    else:
                        voronoi_areas[i] = 100.0
            except:
                voronoi_areas = np.ones(n) * 100.0
        else:
            # Reuse previous Voronoi data for efficiency
            pass
        
        # Process circles in shuffled order
        for i in indices:
            # Calculate maximum allowable radius for this circle
            max_radius = min(
                circles[i][0],  # Distance to left edge
                rect_width - circles[i][0],  # Distance to right edge
                circles[i][1],  # Distance to bottom edge
                rect_height - circles[i][1]   # Distance to top edge
            ) - 0.001
            
            # Consider collision constraints with neighbors
            collision_violations = 0
            for j in range(n):
                if i != j:
                    dx = circles[i][0] - circles[j][0]
                    dy = circles[i][1] - circles[j][1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    collision_radius = distance - circles[j][2] - 0.001
                    if collision_radius > 0:
                        # Apply progressive constraint relaxation
                        if np.random.random() > violation_tolerance:
                            max_radius = min(max_radius, collision_radius)
                        else:
                            collision_violations += 1
            
            # Adaptive expansion based on local density and constraint status
            if max_radius > circles[i][2] and max_radius > 0.001:
                # Determine expansion rate based on Voronoi area
                voronoi_area = voronoi_areas[i] if 'voronoi_areas' in locals() else 100.0
                
                # Normalize Voronoi area relative to total area
                normalized_area = voronoi_area / (rect_width * rect_height) if voronoi_area > 0 else 0.01
                
                # Determine expansion factor based on local constraint density
                # Higher Voronoi area = more space = larger expansion
                expansion_factor = max(0.1, min(1.0, 1.0 - normalized_area * 0.7))
                
                # Apply different expansion rates depending on constraint status
                if collision_violations > 4:
                    # Highly constrained region - very small expansion
                    delta = min(0.01, max_radius - circles[i][2]) * expansion_factor * 0.2
                elif collision_violations > 2:
                    # Moderately constrained - small expansion
                    delta = min(0.015, max_radius - circles[i][2]) * expansion_factor * 0.5
                elif collision_violations > 0:
                    # Slightly constrained - medium expansion
                    delta = min(0.02, max_radius - circles[i][2]) * expansion_factor * 0.8
                else:
                    # Less constrained - full expansion
                    delta = min(0.025, max_radius - circles[i][2]) * expansion_factor * 1.0
                
                if delta > 0.001:
                    circles[i][2] += delta
                    improved = True
        
        if not improved and iteration > 50:  # Early stopping condition
            break
    
    # Phase 3: Strict constraint enforcement with refined boundary optimization
    for iteration in range(max_iterations_phase2):
        improved = False
        
        # Shuffle circle indices for diverse optimization
        indices = list(range(n))
        np.random.shuffle(indices)
        
        # Process circles in shuffled order
        for i in indices:
            # Calculate maximum allowable radius for this circle
            max_radius = min(
                circles[i][0],  # Distance to left edge
                rect_width - circles[i][0],  # Distance to right edge
                circles[i][1],  # Distance to bottom edge
                rect_height - circles[i][1]   # Distance to top edge
            ) - 0.001
            
            # Consider collision constraints with neighbors
            for j in range(n):
                if i != j:
                    dx = circles[i][0] - circles[j][0]
                    dy = circles[i][1] - circles[j][1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    collision_radius = distance - circles[j][2] - 0.001
                    if collision_radius > 0:
                        max_radius = min(max_radius, collision_radius)
            
            # Increase radius if beneficial and strictly feasible
            if max_radius > circles[i][2] and max_radius > 0.001:
                # Conservative but effective increments for final refinement
                delta = min(0.015, max_radius - circles[i][2]) * 0.7
                if delta > 0.0005:
                    circles[i][2] += delta
                    improved = True
        
        if not improved:
            break
    
    # Phase 4: Boundary-aware final refinement and validation
    # Enhanced boundary handling for edge circles
    for _ in range(50):  # Additional refinement passes
        improved = False
        for i in range(n):
            # Focus on boundary conditions
            boundary_margin = 0.01
            
            # Check if circle is near boundary and adjust appropriately
            near_left = circles[i][0] < circles[i][2] + boundary_margin
            near_right = circles[i][0] > rect_width - circles[i][2] - boundary_margin
            near_bottom = circles[i][1] < circles[i][2] + boundary_margin
            near_top = circles[i][1] > rect_height - circles[i][2] - boundary_margin
            
            # Boundary-aware radius adjustment
            if near_left or near_right or near_bottom or near_top:
                # Try to adjust position to maintain boundaries while maximizing radius
                max_radius = min(
                    circles[i][0] if not near_left else rect_width - circles[i][0],
                    rect_width - circles[i][0] if not near_right else circles[i][0],
                    circles[i][1] if not near_bottom else rect_height - circles[i][1],
                    rect_height - circles[i][1] if not near_top else circles[i][1]
                ) - 0.001
                
                # Only adjust if beneficial
                if max_radius > circles[i][2] and max_radius > 0.001:
                    delta = min(0.005, max_radius - circles[i][2])
                    if delta > 0.0005:
                        circles[i][2] += delta
                        improved = True
        
        if not improved:
            break
    
    # Final validation and cleanup
    circles = validate_and_fix_constraints(circles, rect_width, rect_height)
    
    return circles

def initialize_hexagonal(n: int, width: float, height: float) -> np.ndarray:
    """Initialize circles using hexagonal packing pattern."""
    circles = np.zeros((n, 3))
    
    # Hexagonal packing arrangement
    rows = 4
    cols = 6
    
    # Calculate spacing
    spacing_x = width / (cols + 1)
    spacing_y = height / (rows + 1)
    
    # Place circles in hexagonal pattern
    idx = 0
    for i in range(rows):
        offset = spacing_x * (i % 2) * 0.5  # Offset every other row
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 1) * spacing_x + offset
            y = (i + 1) * spacing_y
            
            # Ensure position is within bounds
            x = max(0.01, min(width - 0.01, x))
            y = max(0.01, min(height - 0.01, y))
            
            # Set initial radius to a small value
            circles[idx] = [x, y, 0.05]
            idx += 1
            
        if idx >= n:
            break
    
    # Fill remaining circles if needed
    while idx < n:
        x = np.random.uniform(0.01, width - 0.01)
        y = np.random.uniform(0.01, height - 0.01)
        circles[idx] = [x, y, 0.05]
        idx += 1
    
    return circles

def initialize_spiral(n: int, width: float, height: float) -> np.ndarray:
    """Initialize circles using a spiral pattern for better edge coverage."""
    circles = np.zeros((n, 3))
    
    # Center of the rectangle
    center_x, center_y = width / 2, height / 2
    
    # Spiral parameters
    angle_step = 2 * np.pi / 5
    radius_step = min(width, height) * 0.2 / n
    max_radius = min(width, height) * 0.4
    
    # Generate spiral points
    for i in range(n):
        # Spiral position
        angle = i * angle_step
        radius = min(max_radius, i * radius_step)
        
        # Convert to cartesian coordinates
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        
        # Ensure within bounds
        x = max(0.01, min(width - 0.01, x))
        y = max(0.01, min(height - 0.01, y))
        
        # Set initial radius
        circles[i] = [x, y, 0.04]
    
    return circles

def initialize_square_grid(n: int, width: float, height: float) -> np.ndarray:
    """Initialize circles using square grid pattern."""
    circles = np.zeros((n, 3))
    
    # Square grid arrangement
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    spacing_x = width / (cols + 1)
    spacing_y = height / (rows + 1)
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 1) * spacing_x
            y = (i + 1) * spacing_y
            
            # Ensure position is within bounds
            x = max(0.01, min(width - 0.01, x))
            y = max(0.01, min(height - 0.01, y))
            
            # Set initial radius to a small value
            circles[idx] = [x, y, 0.05]
            idx += 1
            
        if idx >= n:
            break
    
    # Fill remaining circles if needed
    while idx < n:
        x = np.random.uniform(0.01, width - 0.01)
        y = np.random.uniform(0.01, height - 0.01)
        circles[idx] = [x, y, 0.05]
        idx += 1
    
    return circles

def initialize_smart_random(n: int, width: float, height: float) -> np.ndarray:
    """Initialize circles using random placement with intelligent spacing."""
    circles = np.zeros((n, 3))
    
    # Start with larger initial radii to encourage better packing
    for i in range(n):
        # Attempt placement with more intelligent distribution
        placed = False
        attempts = 0
        
        while not placed and attempts < 100:
            # Prefer placing near edges and corners for better spread
            if np.random.random() < 0.3:  # 30% chance to place near boundary
                # Near boundary placement
                boundary_edge = np.random.choice(['left', 'right', 'top', 'bottom'])
                if boundary_edge == 'left':
                    x = np.random.uniform(0.01, 0.1)
                    y = np.random.uniform(0.01, height - 0.01)
                elif boundary_edge == 'right':
                    x = np.random.uniform(width - 0.1, width - 0.01)
                    y = np.random.uniform(0.01, height - 0.01)
                elif boundary_edge == 'top':
                    x = np.random.uniform(0.01, width - 0.01)
                    y = np.random.uniform(height - 0.1, height - 0.01)
                else:  # bottom
                    x = np.random.uniform(0.01, width - 0.01)
                    y = np.random.uniform(0.01, 0.1)
            else:
                # Regular random placement
                x = np.random.uniform(0.01, width - 0.01)
                y = np.random.uniform(0.01, height - 0.01)
            
            # Check if this location is compatible with existing circles
            valid = True
            for j in range(i):
                dx = x - circles[j, 0]
                dy = y - circles[j, 1]
                distance = np.sqrt(dx*dx + dy*dy)
                # Require more space between new and existing circles
                if distance < (circles[j, 2] + 0.04) * 1.2:
                    valid = False
                    break
            
            if valid:
                circles[i] = [x, y, 0.04]
                placed = True
            else:
                attempts += 1
        
        # Fallback to random if failed
        if not placed:
            x = np.random.uniform(0.01, width - 0.01)
            y = np.random.uniform(0.01, height - 0.01)
            circles[i] = [x, y, 0.04]
    
    return circles

def validate_and_fix_constraints(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Ensure all circles satisfy boundary and collision constraints."""
    n = len(circles)
    
    # Check boundary violations and fix them
    for i in range(n):
        # Ensure circles are within bounds
        circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], width - circles[i, 2])
        circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], height - circles[i, 2])
    
    # Resolve collision violations iteratively
    for _ in range(100):
        improved = False
        for i in range(n):
            # Check collisions with all others
            for j in range(n):
                if i != j:
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    required_distance = circles[i, 2] + circles[j, 2] + 0.001
                    
                    if distance < required_distance:
                        # Move circles apart (simple approach)
                        if distance > 0.001:
                            move_distance = (required_distance - distance) / 2.0
                            direction_x = dx / distance
                            direction_y = dy / distance
                            
                            # Move both circles away from each other
                            circles[i, 0] += direction_x * move_distance * 0.5
                            circles[i, 1] += direction_y * move_distance * 0.5
                            circles[j, 0] -= direction_x * move_distance * 0.5
                            circles[j, 1] -= direction_y * move_distance * 0.5
                            
                            # Clip back to bounds
                            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], width - circles[i, 2])
                            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], height - circles[i, 2])
                            circles[j, 0] = np.clip(circles[j, 0], circles[j, 2], width - circles[j, 2])
                            circles[j, 1] = np.clip(circles[j, 1], circles[j, 2], height - circles[j, 2])
                        
                        improved = True
                        
        if not improved:
            break
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")