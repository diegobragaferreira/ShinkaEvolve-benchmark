# EVOLVE-BLOCK-START
import numpy as np
import random
from typing import Tuple, List
import math

# Physics-based optimization parameters
MAX_ITERATIONS = 10000
TEMPERATURE_START = 1.0
TEMPERATURE_END = 0.01
COOLING_RATE = 0.9995
FORCE_CONSTANT = 100.0
BOUNDARY_FORCE_CONSTANT = 1000.0
TIME_STEP = 0.01
VELOCITY_DAMPING = 0.95

def poisson_disk_sampling(n_points: int, min_distance: float = 0.1) -> List[Tuple[float, float]]:
    """Generate points using Poisson disk sampling for better uniformity."""
    points = []
    active_list = []
    
    # Start with a random point
    points.append((random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)))
    active_list.append(0)
    
    while len(points) < n_points:
        if not active_list:
            break
            
        # Pick a random active point
        idx = random.choice(active_list)
        x, y = points[idx]
        
        # Try to generate a new point
        found = False
        for _ in range(30):  # Limit attempts
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(min_distance, 2 * min_distance)
            
            new_x = x + radius * math.cos(angle)
            new_y = y + radius * math.sin(angle)
            
            # Check bounds
            if new_x < 0.05 or new_x > 0.95 or new_y < 0.05 or new_y > 0.95:
                continue
                
            # Check distance to existing points
            too_close = False
            for px, py in points:
                dist = math.sqrt((new_x - px)**2 + (new_y - py)**2)
                if dist < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                points.append((new_x, new_y))
                active_list.append(len(points) - 1)
                found = True
                break
        
        if not found:
            active_list.remove(idx)
    
    # If we didn't get enough points, fill with random ones
    while len(points) < n_points:
        points.append((random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)))
    
    return points[:n_points]

def initialize_circles(n: int) -> np.ndarray:
    """Initialize circles using Poisson disk sampling for good spatial distribution."""
    # Generate points using Poisson disk sampling
    sample_points = poisson_disk_sampling(n, 0.15)
    
    circles = np.zeros((n, 3))  # x, y, r
    
    # Distribute circles using the sample points
    for i in range(min(n, len(sample_points))):
        x_base, y_base = sample_points[i]
        
        # Add jitter for diversity
        x = max(0.01, min(0.99, x_base + random.uniform(-0.03, 0.03)))
        y = max(0.01, min(0.99, y_base + random.uniform(-0.03, 0.03)))
        
        # Initial radius - start with moderately large values
        circles[i] = [x, y, 0.06]
    
    # Fill remaining circles
    for i in range(len(sample_points), n):
        # Place remaining circles more randomly but still with some structure
        if random.random() < 0.4:
            # Near an existing circle
            idx = random.randint(0, min(i-1, len(sample_points)-1))
            x_base, y_base = sample_points[idx]
            x = max(0.01, min(0.99, x_base + random.uniform(-0.08, 0.08)))
            y = max(0.01, min(0.99, y_base + random.uniform(-0.08, 0.08)))
        else:
            # Completely random
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
        
        circles[i] = [x, y, 0.025]
    
    # Ensure circles don't overlap by resolving initial conflicts
    circles = resolve_initial_overlaps(circles)
    
    return circles

def resolve_initial_overlaps(circles: np.ndarray) -> np.ndarray:
    """Resolve overlaps in initial configuration using force-based approach."""
    resolved = circles.copy()
    
    # Iteratively resolve overlaps
    for _ in range(10):
        changed = False
        for i in range(len(resolved)):
            for j in range(i+1, len(resolved)):
                xi, yi, ri = resolved[i]
                xj, yj, rj = resolved[j]
                dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                
                if dist < (ri + rj - 1e-6):
                    # Move circles apart
                    dx = xj - xi
                    dy = yj - yi
                    distance = max(1e-6, dist)
                    
                    # Normalize
                    dx /= distance
                    dy /= distance
                    
                    # Move based on inverse radius ratio
                    move_amount = (ri + rj - dist) * 0.5
                    
                    # Apply movement in opposite directions
                    resolved[i, 0] -= dx * move_amount * 0.4
                    resolved[i, 1] -= dy * move_amount * 0.4
                    resolved[j, 0] += dx * move_amount * 0.4
                    resolved[j, 1] += dy * move_amount * 0.4
                    changed = True
        
        # Ensure bounds
        for i in range(len(resolved)):
            x, y, r = resolved[i]
            # Clamp to valid range
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            resolved[i] = [x, y, r]
            
        if not changed:
            break
    
    return resolved

def check_containment(circles: np.ndarray) -> bool:
    """Check if all circles are fully contained in the unit square."""
    for x, y, r in circles:
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    return True

def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def check_overlap(circles: np.ndarray) -> bool:
    """Check if any circles overlap."""
    n = len(circles)
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = calculate_distance((xi, yi), (xj, yj))
            if dist < (ri + rj - 1e-6):
                return True
    return False

def compute_forces(circles: np.ndarray, temperature: float) -> np.ndarray:
    """Compute net forces on each circle including boundary forces."""
    n = len(circles)
    forces = np.zeros((n, 2))  # x_force, y_force
    
    # Compute pairwise repulsive forces between overlapping circles
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = calculate_distance((xi, yi), (xj, yj))
            
            if dist < (ri + rj - 1e-6):
                # Repulsive force (overlap)
                dx = xj - xi
                dy = yj - yi
                distance = max(1e-6, dist)
                
                # Normalize
                dx /= distance
                dy /= distance
                
                # Force magnitude inversely proportional to distance
                force_magnitude = FORCE_CONSTANT * (ri + rj - dist) / distance
                forces[i, 0] += dx * force_magnitude
                forces[i, 1] += dy * force_magnitude
                forces[j, 0] -= dx * force_magnitude
                forces[j, 1] -= dy * force_magnitude
    
    # Compute boundary forces (hard walls)
    for i in range(n):
        x, y, r = circles[i]
        
        # Left boundary force
        if x - r < 0:
            forces[i, 0] += BOUNDARY_FORCE_CONSTANT * (r - x) / (r * r)
        
        # Right boundary force  
        if x + r > 1:
            forces[i, 0] += BOUNDARY_FORCE_CONSTANT * (1 - r - x) / (r * r)
        
        # Bottom boundary force
        if y - r < 0:
            forces[i, 1] += BOUNDARY_FORCE_CONSTANT * (r - y) / (r * r)
        
        # Top boundary force
        if y + r > 1:
            forces[i, 1] += BOUNDARY_FORCE_CONSTANT * (1 - r - y) / (r * r)
    
    # Add some gradient-based attraction towards optimality
    # This helps move solutions toward better configurations
    total_radius = np.sum(circles[:, 2])
    for i in range(n):
        # Simple gradient-based attraction - pull towards center of mass
        # but weighted by radius (larger circles get more attention)
        x, y, r = circles[i]
        # Attract to center (weighted by relative radius)
        center_attraction = np.array([0.5 - x, 0.5 - y])
        attraction_magnitude = 0.1 * r  # Weight by radius
        forces[i] += center_attraction * attraction_magnitude
    
    # Scale forces by temperature to control movement
    forces *= temperature
    
    return forces

def update_positions(circles: np.ndarray, forces: np.ndarray) -> np.ndarray:
    """Update circle positions using forces and velocity damping."""
    updated = circles.copy()
    
    # Simplified physics: assume velocity proportional to force
    # Update positions using Euler integration with time step
    for i in range(len(updated)):
        x, y, r = updated[i]
        fx, fy = forces[i]
        
        # Update position
        new_x = x + fx * TIME_STEP
        new_y = y + fy * TIME_STEP
        
        # Apply velocity damping to prevent oscillation
        new_x = x + (new_x - x) * VELOCITY_DAMPING
        new_y = y + (new_y - y) * VELOCITY_DAMPING
        
        # Clamp to valid range
        new_x = max(r, min(1-r, new_x))
        new_y = max(r, min(1-r, new_y))
        
        updated[i, 0] = new_x
        updated[i, 1] = new_y
    
    return updated

def compute_penalty(circles: np.ndarray) -> float:
    """Compute penalty based on constraint violations."""
    penalty = 0.0
    
    # Check containment violations with scaled penalties
    for x, y, r in circles:
        # Boundary violations
        if x - r < 0:
            penalty += (abs(x - r) ** 2) * 10000
        elif x + r > 1:
            penalty += (abs(x + r - 1) ** 2) * 10000
        if y - r < 0:
            penalty += (abs(y - r) ** 2) * 10000
        elif y + r > 1:
            penalty += (abs(y + r - 1) ** 2) * 10000
    
    # Check overlap violations with scaled penalties
    if check_overlap(circles):
        penalty += 10000000.0
    
    return penalty

def evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate fitness of a circle configuration."""
    # If invalid, heavily penalize
    if not check_containment(circles) or check_overlap(circles):
        penalty = compute_penalty(circles)
        return -penalty
    
    # Otherwise, return total radius
    total_radius = np.sum(circles[:, 2])
    return total_radius

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    # Initialize circles
    circles = initialize_circles(26)
    
    # Physics-based optimization
    current_temp = TEMPERATURE_START
    best_solution = circles.copy()
    best_fitness = evaluate_fitness(best_solution)
    
    # Main optimization loop
    for iteration in range(MAX_ITERATIONS):
        # Compute forces on all circles
        forces = compute_forces(circles, current_temp)
        
        # Update positions
        circles = update_positions(circles, forces)
        
        # Evaluate current configuration
        current_fitness = evaluate_fitness(circles)
        
        # Accept or reject based on fitness improvement
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_solution = circles.copy()
        
        # Cool down temperature
        current_temp *= COOLING_RATE
        
        # Occasionally print progress
        if iteration % 1000 == 0:
            print(f"Iteration {iteration}: Best fitness = {best_fitness}")
    
    return best_solution

# EVOLVE-BLOCK-END