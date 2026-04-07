# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, cKDTree
from scipy.spatial.distance import cdist
import time
import math

def compute_voronoi_forces(circles, rect_width, rect_height):
    """Compute Voronoi-based repulsive forces between circles."""
    n = len(circles)
    if n <= 1:
        return np.zeros((n, 2))
    
    # Get circle centers
    centers = circles[:, :2]
    
    # Add boundary points for proper Voronoi calculation
    boundary_points = [
        [0, 0], [rect_width, 0], [0, rect_height], [rect_width, rect_height],
        [rect_width/2, 0], [rect_width/2, rect_height],
        [0, rect_height/2], [rect_width, rect_height/2]
    ]
    all_points = np.vstack([centers, boundary_points])
    
    try:
        vor = Voronoi(all_points)
        
        # Compute Voronoi cell areas for each circle
        voronoi_areas = []
        for i in range(n):
            region_idx = np.where(vor.point_region == i)[0][0] if i in vor.point_region else -1
            if region_idx != -1 and region_idx < len(vor.regions):
                region = vor.regions[region_idx]
                if -1 not in region and len(region) >= 3:
                    vertices = np.array([vor.vertices[j] for j in region])
                    if len(vertices) >= 3:
                        # Compute area using shoelace formula
                        x = vertices[:, 0]
                        y = vertices[:, 1]
                        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                        voronoi_areas.append(area)
                    else:
                        voronoi_areas.append(1.0)
                else:
                    voronoi_areas.append(1.0)
            else:
                voronoi_areas.append(1.0)
        
        voronoi_areas = np.array(voronoi_areas)
        
        # Normalize Voronoi areas (smaller areas = more constrained regions)
        normalized_areas = 1.0 / (voronoi_areas + 1e-8)
        normalized_areas = normalized_areas / (np.max(normalized_areas) + 1e-8)
        
        # Compute forces
        forces = np.zeros((n, 2))
        
        # Pairwise repulsion forces
        for i in range(n):
            for j in range(i+1, n):
                pos_i = centers[i]
                pos_j = centers[j]
                dist_vec = pos_i - pos_j
                distance = np.linalg.norm(dist_vec)
                
                if distance > 1e-8:
                    # Base repulsion force (inverse square law)
                    force_magnitude = 1.0 / (distance * distance)
                    
                    # Modify by Voronoi density (denser regions get stronger forces)
                    density_factor = (normalized_areas[i] + normalized_areas[j]) / 2.0
                    force_magnitude *= (1.0 + 5.0 * density_factor)
                    
                    # Apply force direction
                    force_direction = dist_vec / distance
                    forces[i] += force_magnitude * force_direction
                    forces[j] -= force_magnitude * force_direction
        
        return forces
        
    except Exception as e:
        # Fallback to basic repulsion if Voronoi fails
        forces = np.zeros((n, 2))
        for i in range(n):
            for j in range(i+1, n):
                pos_i = centers[i]
                pos_j = centers[j]
                dist_vec = pos_i - pos_j
                distance = np.linalg.norm(dist_vec)
                
                if distance > 1e-8:
                    force_magnitude = 1.0 / (distance * distance)
                    force_direction = dist_vec / distance
                    forces[i] += force_magnitude * force_direction
                    forces[j] -= force_magnitude * force_direction
        
        return forces

def compute_boundary_forces(circles, rect_width, rect_height):
    """Compute boundary forces to keep circles within rectangle."""
    n = len(circles)
    forces = np.zeros((n, 2))
    
    # Boundary constants
    boundary_stiffness = 100.0
    boundary_threshold = 0.1
    
    for i in range(n):
        x, y, r = circles[i]
        
        # Left boundary
        if x - r < boundary_threshold:
            forces[i, 0] += boundary_stiffness * (boundary_threshold - (x - r))
            
        # Right boundary  
        if x + r > rect_width - boundary_threshold:
            forces[i, 0] -= boundary_stiffness * (x + r - (rect_width - boundary_threshold))
            
        # Bottom boundary
        if y - r < boundary_threshold:
            forces[i, 1] += boundary_stiffness * (boundary_threshold - (y - r))
            
        # Top boundary
        if y + r > rect_height - boundary_threshold:
            forces[i, 1] -= boundary_stiffness * (y + r - (rect_height - boundary_threshold))
    
    return forces

def compute_radius_constraints(circles, rect_width, rect_height):
    """Compute constraint forces that prevent radius from becoming negative."""
    n = len(circles)
    forces = np.zeros((n, 2))
    
    # Prevent negative radii
    for i in range(n):
        x, y, r = circles[i]
        if r < 0.001:
            # Push towards positive radius
            forces[i, 0] -= 100.0 * (0.001 - r)
            forces[i, 1] -= 100.0 * (0.001 - r)
    
    return forces

def simulate_force_system(circles, rect_width, rect_height, steps=200):
    """Simulate physics-based force system."""
    n = len(circles)
    current_circles = circles.copy()
    
    # Initialize velocities
    velocities = np.zeros((n, 2))
    
    # Simulation parameters
    dt = 0.01
    damping = 0.95
    max_velocity = 0.1
    
    for step in range(steps):
        # Compute all force components
        force_voronoi = compute_voronoi_forces(current_circles, rect_width, rect_height)
        force_boundary = compute_boundary_forces(current_circles, rect_width, rect_height)
        force_radius = compute_radius_constraints(current_circles, rect_width, rect_height)
        
        # Total forces
        total_forces = force_voronoi + force_boundary + force_radius
        
        # Update velocities and positions
        for i in range(n):
            # Apply forces
            accelerations = total_forces[i] / 1.0  # Assume unit mass
            
            # Update velocity
            velocities[i] += accelerations * dt
            
            # Apply damping
            velocities[i] *= damping
            
            # Limit velocity
            vel_mag = np.linalg.norm(velocities[i])
            if vel_mag > max_velocity:
                velocities[i] = velocities[i] * max_velocity / vel_mag
            
            # Update position
            current_circles[i, 0] += velocities[i, 0] * dt
            current_circles[i, 1] += velocities[i, 1] * dt
            
            # Update radius (with small increase to encourage growth)
            if step % 5 == 0:
                # Increase radius slightly (adaptive to avoid overlap)
                current_circles[i, 2] = min(current_circles[i, 2] + 0.0001, rect_width/2)
        
        # Enforce boundary constraints after updates
        for i in range(n):
            x, y, r = current_circles[i]
            current_circles[i, 0] = np.clip(x, r, rect_width - r)
            current_circles[i, 1] = np.clip(y, r, rect_height - r)
            current_circles[i, 2] = np.clip(current_circles[i, 2], 0.001, rect_width/2)
    
    return current_circles

def get_voronoi_density(circles, rect_width, rect_height):
    """Get Voronoi density measure for constraint awareness."""
    n = len(circles)
    if n <= 1:
        return np.array([1.0] * n)
    
    centers = circles[:, :2]
    
    # Add boundary points
    boundary_points = [
        [0, 0], [rect_width, 0], [0, rect_height], [rect_width, rect_height],
        [rect_width/2, 0], [rect_width/2, rect_height],
        [0, rect_height/2], [rect_width, rect_height/2]
    ]
    all_points = np.vstack([centers, boundary_points])
    
    try:
        vor = Voronoi(all_points)
        
        # Compute Voronoi cell areas for each circle
        areas = []
        for i in range(n):
            region_idx = np.where(vor.point_region == i)[0][0] if i in vor.point_region else -1
            if region_idx != -1 and region_idx < len(vor.regions):
                region = vor.regions[region_idx]
                if -1 not in region and len(region) >= 3:
                    vertices = np.array([vor.vertices[j] for j in region])
                    if len(vertices) >= 3:
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
        return np.ones(n)

def is_valid_solution(circles, rect_width=1.0, rect_height=1.0):
    """Fast validity check for circles."""
    n = len(circles)
    
    # Check bounds
    if np.any(circles[:, 0] - circles[:, 2] < 0) or np.any(circles[:, 0] + circles[:, 2] > rect_width) or \
       np.any(circles[:, 1] - circles[:, 2] < 0) or np.any(circles[:, 1] + circles[:, 2] > rect_height):
        return False

    # Fast collision detection
    if n > 1:
        coords = circles[:, :2]
        radii = circles[:, 2]
        distances = cdist(coords, coords)
        
        # Check all pairs
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        
        min_distances = distances[mask]
        required_distances = (radii + radii[:, None])[mask]
        
        if np.any(min_distances < required_distances):
            return False
    
    return True

def generate_hexagonal_pattern(n, rect_width=1.0, rect_height=1.0):
    """Generate hexagonal starting pattern."""
    circles = np.zeros((n, 3))
    
    rows = int(np.sqrt(n)) + 1
    cols = int(np.ceil(n / rows))
    
    spacing_x = rect_width / (cols + 1)
    spacing_y = rect_height / (rows + 1)
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 1) * spacing_x
            y = (i + 1) * spacing_y
            
            if i % 2 == 1:
                x += spacing_x / 2
                
            circles[idx] = [x, y, 0.02]
            idx += 1
        if idx >= n:
            break
            
    return circles

def generate_triangular_pattern(n, rect_width=1.0, rect_height=1.0):
    """Generate triangular starting pattern."""
    circles = np.zeros((n, 3))
    
    sqrt_n = int(np.ceil(np.sqrt(n))) + 1
    spacing_x = rect_width / (sqrt_n + 1)
    spacing_y = rect_height / (sqrt_n + 1)
    
    idx = 0
    for i in range(sqrt_n):
        for j in range(sqrt_n):
            if idx >= n:
                break
            x = (j + 1) * spacing_x
            y = (i + 1) * spacing_y
            if i % 2 == 1:
                x += spacing_x / 2
            circles[idx] = [x, y, 0.02]
            idx += 1
        if idx >= n:
            break
            
    return circles

def generate_random_pattern(n, rect_width=1.0, rect_height=1.0):
    """Generate random starting pattern."""
    circles = np.zeros((n, 3))
    
    for i in range(n):
        x = np.random.uniform(0.05, rect_width - 0.05)
        y = np.random.uniform(0.05, rect_height - 0.05)
        r = 0.03
        circles[i] = [x, y, r]
    
    return circles

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle with perimeter = 4, so width + height = 2. Optimized ratio
    rect_width = 1.2
    rect_height = 0.8
    
    # Try multiple initialization strategies
    initial_strategies = [
        generate_hexagonal_pattern(21, rect_width, rect_height),
        generate_triangular_pattern(21, rect_width, rect_height), 
        generate_random_pattern(21, rect_width, rect_height)
    ]
    
    best_solution = None
    best_score = -float('inf')
    
    # Multi-start approach with force-based optimization
    for i, seed_pattern in enumerate(initial_strategies):
        # Apply force-based simulation
        optimized_pattern = simulate_force_system(seed_pattern, rect_width, rect_height, steps=100)
        
        # Refinement passes
        for _ in range(3):
            refined = simulate_force_system(optimized_pattern, rect_width, rect_height, steps=50)
            if is_valid_solution(refined, rect_width, rect_height):
                optimized_pattern = refined
            else:
                break
        
        # Score and validate
        if is_valid_solution(optimized_pattern, rect_width, rect_height):
            score = np.sum(optimized_pattern[:, 2])
            if score > best_score:
                best_score = score
                best_solution = optimized_pattern.copy()
    
    # Final validation and return
    if best_solution is not None:
        # Do one final refinement
        final_solution = simulate_force_system(best_solution, rect_width, rect_height, steps=30)
        if is_valid_solution(final_solution, rect_width, rect_height):
            return final_solution
    
    # Fallback to the best we could find
    if best_solution is not None:
        return best_solution
    
    # Last resort - generate random pattern and optimize
    fallback = generate_random_pattern(21, rect_width, rect_height)
    return simulate_force_system(fallback, rect_width, rect_height, steps=200)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")