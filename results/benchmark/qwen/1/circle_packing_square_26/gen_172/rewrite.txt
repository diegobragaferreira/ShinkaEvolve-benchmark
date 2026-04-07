# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple
import time

def _validate_circle_placement(circles: np.ndarray) -> bool:
    """Validate that circles are within bounds and don't overlap."""
    n = len(circles)

    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0 or x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False

    # Check non-overlap constraints using KDTree for efficiency
    points = circles[:, :2]
    tree = cKDTree(points)

    # Find all pairs within distance 2*r (minimum separation needed to avoid overlap)
    pairs = tree.query_pairs(2 * min(circles[:, 2]), output_type='ndarray')

    for i, j in pairs:
        x1, y1, r1 = circles[i]
        x2, y2, r2 = circles[j]
        distance_sq = (x1 - x2)**2 + (y1 - y2)**2
        min_distance_sq = (r1 + r2)**2
        if distance_sq < min_distance_sq:
            return False

    return True

def _evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate fitness as negative sum of radii (since we want to maximize)."""
    if not _validate_circle_placement(circles):
        return -float('inf')  # Invalid configuration gets very low fitness
    return float(np.sum(circles[:, 2]))

def _generate_voronoi_initialization(n_circles: int, seed: int = 42) -> np.ndarray:
    """Generate initial circle positions using a Voronoi-inspired spreading mechanism."""
    np.random.seed(seed)

    # Create a grid of candidate positions
    grid_size = max(3, int(np.ceil(np.sqrt(n_circles))))
    x_coords = np.linspace(0.05, 0.95, grid_size)
    y_coords = np.linspace(0.05, 0.95, grid_size)

    # Generate all grid points
    grid_points = []
    for x in x_coords:
        for y in y_coords:
            grid_points.append([x, y])

    # If we have more circles than grid points, add some random points
    if len(grid_points) < n_circles:
        extra_points = n_circles - len(grid_points)
        for _ in range(extra_points):
            grid_points.append([np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)])

    # Shuffle the points to avoid systematic bias
    random.shuffle(grid_points)

    # Take the first n_circles points
    points = np.array(grid_points[:n_circles])

    # Initialize circles with small radii
    circles = np.zeros((n_circles, 3))
    circles[:, 0] = points[:, 0]  # x coordinates
    circles[:, 1] = points[:, 1]  # y coordinates
    circles[:, 2] = 0.01         # initial small radii

    return circles

def _generate_spiral_initialization(n_circles: int, seed: int = 42) -> np.ndarray:
    """Generate initial circle positions using a spiral pattern."""
    np.random.seed(seed)
    
    circles = np.zeros((n_circles, 3))
    
    # Spiral parameters
    a = 0.05  # spiral parameter
    b = 0.05  # spiral parameter
    
    for i in range(n_circles):
        angle = 2 * np.pi * i / n_circles * 5  # spiral with 5 turns
        radius = a + b * angle
        radius = min(radius, 0.45)  # cap at reasonable value
        
        x = 0.5 + radius * np.cos(angle) * 0.4
        y = 0.5 + radius * np.sin(angle) * 0.4
        
        # Clip to valid range
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        
        circles[i, 0] = x
        circles[i, 1] = y
        circles[i, 2] = 0.01  # small initial radius

    return circles

def _greedy_fallback(n_circles: int) -> np.ndarray:
    """Fallback method to generate a feasible configuration."""
    # Simple greedy approach: place circles in order of decreasing radius
    circles = np.zeros((n_circles, 3))

    # Start with small radii and gradually increase
    # Place in a way that they don't overlap initially
    positions = []
    radii = []

    # Try to place circles greedily by spacing them out
    placed = 0
    radius = 0.05
    while placed < n_circles and radius > 0.005:
        # Try placing circles in a spiral pattern or grid
        attempt = 0
        while attempt < 100 and placed < n_circles:
            # Place in grid-like fashion
            rows = int(np.sqrt(n_circles)) + 1
            cols = n_circles // rows + 1

            for i in range(rows):
                for j in range(cols):
                    if placed >= n_circles:
                        break
                    x = 0.1 + j * 0.8 / cols
                    y = 0.1 + i * 0.8 / rows

                    # Check if this position is valid
                    valid = True
                    for pos, rad in zip(positions, radii):
                        dist_sq = (x - pos[0])**2 + (y - pos[1])**2
                        if dist_sq < (rad + radius)**2:
                            valid = False
                            break

                    if valid:
                        positions.append([x, y])
                        radii.append(radius)
                        placed += 1
            attempt += 1

        radius *= 0.9  # Decrease radius slightly

    # Fill remaining circles
    while placed < n_circles:
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        positions.append([x, y])
        radii.append(0.01)
        placed += 1

    circles[:, 0] = [pos[0] for pos in positions]
    circles[:, 1] = [pos[1] for pos in positions]
    circles[:, 2] = radii

    return circles

def _compute_forces(circles: np.ndarray, 
                   repulsion_strength: float = 100.0,
                   boundary_strength: float = 50.0) -> Tuple[np.ndarray, float]:
    """
    Compute forces acting on each circle including repulsion and boundary attraction.
    
    Returns:
        forces: array of shape (n_circles, 2) representing force vectors
        potential_energy: total potential energy of the system
    """
    n = len(circles)
    forces = np.zeros((n, 2))
    potential_energy = 0.0
    
    # Repulsion forces between circles
    for i in range(n):
        x1, y1, r1 = circles[i]
        for j in range(i+1, n):
            x2, y2, r2 = circles[j]
            
            dx = x1 - x2
            dy = y1 - y2
            distance_sq = dx*dx + dy*dy
            
            # Skip if too far apart
            if distance_sq > 4*(r1+r2)**2:  # Approximate cutoff
                continue
                
            distance = np.sqrt(distance_sq)
            
            # Avoid division by zero
            if distance < 1e-10:
                continue
                
            # Repulsive force (inverse square law)
            force_magnitude = repulsion_strength / (distance_sq + 1e-10) 
            
            # Apply force direction
            fx = force_magnitude * dx / distance
            fy = force_magnitude * dy / distance
            
            forces[i, 0] += fx
            forces[i, 1] += fy
            forces[j, 0] -= fx
            forces[j, 1] -= fy
            
            # Energy contribution (potential energy)
            potential_energy += force_magnitude * (1.0/distance)

    # Boundary forces (spring-like attraction to boundaries)
    for i in range(n):
        x, y, r = circles[i]
        
        # Force components towards boundaries
        fx = 0.0
        fy = 0.0
        
        # Left boundary
        if x - r < 0:
            fx += boundary_strength * (0 - (x - r))
        # Right boundary  
        if x + r > 1:
            fx += boundary_strength * (1 - (x + r))
            
        # Bottom boundary
        if y - r < 0:
            fy += boundary_strength * (0 - (y - r))
        # Top boundary
        if y + r > 1:
            fy += boundary_strength * (1 - (y + r))
            
        forces[i, 0] += fx
        forces[i, 1] += fy

    return forces, potential_energy

def _update_circles(circles: np.ndarray, forces: np.ndarray, 
                   dt: float = 0.01, max_velocity: float = 0.05) -> np.ndarray:
    """
    Update circle positions based on forces and velocity limits.
    """
    updated_circles = circles.copy()
    
    # Update velocities and positions
    for i in range(len(circles)):
        x, y, r = circles[i]
        
        # Update velocity (force gives acceleration)
        vx = forces[i, 0] * dt
        vy = forces[i, 1] * dt
        
        # Apply velocity limits
        vel_norm = np.sqrt(vx*vx + vy*vy)
        if vel_norm > max_velocity:
            vx = vx * max_velocity / vel_norm
            vy = vy * max_velocity / vel_norm
            
        # Update position
        new_x = x + vx
        new_y = y + vy
        
        # Ensure new positions respect boundaries
        new_x = np.clip(new_x, r, 1-r)
        new_y = np.clip(new_y, r, 1-r)
        
        updated_circles[i, 0] = new_x
        updated_circles[i, 1] = new_y
        
    return updated_circles

def _adaptive_simulation_step(circles: np.ndarray, 
                             target_energy: float = 1e-3,
                             max_steps: int = 1000) -> np.ndarray:
    """
    Perform adaptive simulation to reach equilibrium.
    """
    # Parameters for adaptive dynamics
    dt = 0.01
    damping_factor = 0.95
    energy_threshold = target_energy
    
    # Track energy evolution
    prev_energy = float('inf')
    energy_history = []
    
    for step in range(max_steps):
        # Compute forces
        forces, current_energy = _compute_forces(circles)
        
        # Check for convergence
        if len(energy_history) > 10:
            recent_avg = np.mean(energy_history[-10:])
            if abs(recent_avg - current_energy) < energy_threshold:
                break
                
        # Store energy for tracking
        energy_history.append(current_energy)
        
        # Update circles
        circles = _update_circles(circles, forces, dt)
        
        # Apply damping to prevent oscillation
        if step % 10 == 0:
            dt *= damping_factor
        
        # Occasionally reset time step
        if step % 50 == 0:
            dt = max(0.001, dt * 0.95)
    
    return circles

def _multi_scale_refinement(initial_circles: np.ndarray, 
                          max_iterations: int = 200) -> np.ndarray:
    """
    Refine the solution using multi-scale approach:
    1. Coarse-grained optimization
    2. Fine-grained refinement
    3. Local search improvement
    """
    circles = initial_circles.copy()
    
    # Phase 1: Coarse optimization using large time steps and strong forces
    circles = _adaptive_simulation_step(circles, target_energy=1e-2, max_steps=300)
    
    # Phase 2: Medium optimization with moderate forces
    circles = _adaptive_simulation_step(circles, target_energy=1e-3, max_steps=200)
    
    # Phase 3: Fine-tuning with very small forces and high precision
    circles = _adaptive_simulation_step(circles, target_energy=1e-4, max_steps=150)
    
    return circles

def _constraint_aware_local_search(circles: np.ndarray, max_iterations: int = 100) -> np.ndarray:
    """
    Apply constraint-aware local search to fine-tune the solution.
    This version focuses on finding local optima while respecting constraints.
    """
    improved_circles = circles.copy()
    n = len(improved_circles)
    
    for iteration in range(max_iterations):
        improved = False
        
        # Try to increase radii while maintaining constraints
        for i in range(n):
            x, y, r = improved_circles[i]
            
            # Calculate maximum possible radius at this position
            max_radius = min(x, 1-x, y, 1-y)
            
            # Find neighboring circles to check constraints
            neighbors = []
            for j in range(n):
                if i != j:
                    x2, y2, r2 = improved_circles[j]
                    dist_sq = (x - x2)**2 + (y - y2)**2
                    min_dist_sq = (r + r2)**2
                    neighbors.append((j, dist_sq, min_dist_sq))
            
            # Calculate maximum increase in radius
            max_incr = max_radius - r
            
            # Check overlap constraints more rigorously
            min_dist_to_neighbor = float('inf')
            for _, dist_sq, min_dist_sq in neighbors:
                if dist_sq < min_dist_sq:
                    min_dist_to_neighbor = min(min_dist_to_neighbor, dist_sq)
                    
            # If we have space to grow and can still satisfy constraints
            if max_incr > 0.001:
                # Binary search for maximum possible radius
                low, high = r, max_radius
                best_radius = r
                
                # Binary search for maximum safe radius
                for _ in range(20):  # Limit iterations
                    mid = (low + high) / 2
                    valid = True
                    
                    # Check if this radius creates overlap issues
                    for j, dist_sq, min_dist_sq in neighbors:
                        x2, y2, r2 = improved_circles[j]
                        dist_sq = (x - x2)**2 + (y - y2)**2
                        if dist_sq < (mid + r2)**2:
                            valid = False
                            break
                    
                    if valid:
                        best_radius = mid
                        low = mid
                    else:
                        high = mid
                
                # Apply the found safe radius if it's an improvement
                if best_radius > r + 1e-6:
                    improved_circles[i, 2] = best_radius
                    improved = True
        
        # If no improvement from radius increases, try position adjustments
        if not improved:
            for i in range(n):
                x, y, r = improved_circles[i]
                
                # Try small movements in 8 directions
                movements = [(-0.005, -0.005), (-0.005, 0), (-0.005, 0.005),
                           (0, -0.005),              (0, 0.005),
                           (0.005, -0.005), (0.005, 0), (0.005, 0.005)]
                
                best_x, best_y = x, y
                best_score = -float('inf')
                best_radius = r
                
                for dx, dy in movements:
                    new_x, new_y = x + dx, y + dy
                    
                    # Check bounds
                    if new_x - r < 0 or new_x + r > 1 or new_y - r < 0 or new_y + r > 1:
                        continue
                        
                    # Check overlap with neighbors
                    overlap_penalty = 0
                    valid = True
                    for j in range(n):
                        if i != j:
                            x2, y2, r2 = improved_circles[j]
                            dist_sq = (new_x - x2)**2 + (new_y - y2)**2
                            min_dist_sq = (r + r2)**2
                            if dist_sq < min_dist_sq:
                                overlap_penalty += (min_dist_sq - dist_sq) * 1000
                                valid = False
                    
                    if valid:
                        # Score based on overlap reduction and radius preservation
                        score = -overlap_penalty + r
                        if score > best_score:
                            best_score = score
                            best_x, best_y = new_x, new_y
                
                # Apply the best movement if it helps
                if best_x != x or best_y != y:
                    improved_circles[i, 0] = best_x
                    improved_circles[i, 1] = best_y
                    improved = True
        
        # If no improvement made, exit loop
        if not improved:
            break
    
    return improved_circles

def _improve_with_gradient_descent(circles: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """
    Apply gradient-based improvement focusing on increasing radii while respecting constraints.
    """
    improved_circles = circles.copy()
    n = len(improved_circles)
    
    for iteration in range(max_iter):
        # For each circle, compute how much we can increase radius
        updated = False
        
        for i in range(n):
            x, y, r = improved_circles[i]
            max_radius = min(x, 1-x, y, 1-y)
            
            # Find minimum distance to neighbors for overlap constraints
            min_dist = float('inf')
            for j in range(n):
                if i != j:
                    x2, y2, r2 = improved_circles[j]
                    dist_sq = (x - x2)**2 + (y - y2)**2
                    if dist_sq < min_dist:
                        min_dist = dist_sq
            
            # If we can increase radius
            current_max_radius = max_radius
            if min_dist < (r + 0.001)**2:
                # We're currently overlapping, reduce radius
                # Or find maximum safe radius
                safe_radius = min(r, max_radius)
                if safe_radius > r + 1e-6:
                    improved_circles[i, 2] = safe_radius
                    updated = True
            else:
                # Try to increase radius up to boundary
                if r < max_radius - 1e-6:
                    # Test if we can safely increase
                    test_radius = min(max_radius, r + 0.005)
                    valid = True
                    for j in range(n):
                        if i != j:
                            x2, y2, r2 = improved_circles[j]
                            dist_sq = (x - x2)**2 + (y - y2)**2
                            if dist_sq < (test_radius + r2)**2:
                                valid = False
                                break
                    
                    if valid:
                        improved_circles[i, 2] = test_radius
                        updated = True
        
        if not updated:
            break
    
    return improved_circles

def _initialize_population(n_circles: int, n_pop: int, seed: int = 42) -> list:
    """Initialize population with multiple strategies."""
    np.random.seed(seed)
    population = []
    
    # Add multiple initialization strategies
    # Voronoi initialization
    voronoi_init = _generate_voronoi_initialization(n_circles, seed)
    population.append(voronoi_init.flatten().tolist())
    
    # Spiral initialization
    spiral_init = _generate_spiral_initialization(n_circles, seed + 1)
    population.append(spiral_init.flatten().tolist())
    
    # Random initialization
    for i in range(n_pop - 2):
        individual = []
        for j in range(n_circles):
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = np.random.uniform(0.005, 0.45)
            individual.extend([x, y, r])
        population.append(individual)
    
    return population

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining physics simulation with evolutionary initialization strategies.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    seed = 42
    
    np.random.seed(seed)
    
    # Hybrid initialization: start with multiple strategies
    initial_strategies = []
    
    # Strategy 1: Voronoi-inspired
    voronoi_init = _generate_voronoi_initialization(n, seed)
    initial_strategies.append(voronoi_init)
    
    # Strategy 2: Spiral
    spiral_init = _generate_spiral_initialization(n, seed + 1)
    initial_strategies.append(spiral_init)
    
    # Strategy 3: Random
    random_init = []
    for _ in range(5):
        individual = []
        for j in range(n):
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = np.random.uniform(0.005, 0.45)
            individual.extend([x, y, r])
        random_init.append(np.array(individual).reshape(-1, 3))
    initial_strategies.extend(random_init)
    
    best_result = None
    best_fitness = -float('inf')
    
    # Try each initialization strategy
    for i, initial_circles in enumerate(initial_strategies):
        # Apply physics-based optimization to each starting point
        circles = initial_circles.copy()
        
        # Multi-scale refinement using physical simulation
        circles = _multi_scale_refinement(circles, max_iterations=200)
        
        # Apply constraint-aware local search for fine-tuning
        circles = _constraint_aware_local_search(circles, max_iterations=200)
        
        # Additional gradient-based refinement
        circles = _improve_with_gradient_descent(circles, max_iter=100)
        
        # Validate result
        if _validate_circle_placement(circles):
            fitness = _evaluate_fitness(circles)
            if fitness > best_fitness:
                best_fitness = fitness
                best_result = circles.copy()
        else:
            # Fallback for invalid configurations
            circles = _greedy_fallback(n)
            if _validate_circle_placement(circles):
                fitness = _evaluate_fitness(circles)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_result = circles.copy()
    
    # If still no good result, use default Voronoi initialization
    if best_result is None:
        best_result = _generate_voronoi_initialization(n, seed)
    
    return best_result

# EVOLVE-BLOCK-END