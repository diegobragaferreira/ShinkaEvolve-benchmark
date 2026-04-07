# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def initialize_physics_based_layout(n_circles: int, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """
    Initialize circle positions using physics-inspired layout that starts with evenly distributed points
    and applies repulsive forces to distribute them initially.
    """
    # Start with a simple grid pattern to get good initial spread
    circles = np.zeros((n_circles, 3))
    
    # Calculate grid dimensions based on number of circles
    sqrt_n = int(np.ceil(np.sqrt(n_circles)))
    grid_width = rect_width * 0.9
    grid_height = rect_height * 0.9
    
    # Grid spacing
    spacing_x = grid_width / (sqrt_n + 1)
    spacing_y = grid_height / (sqrt_n + 1)
    
    placed = 0
    
    # Fill grid with circles
    for i in range(sqrt_n):
        for j in range(sqrt_n):
            if placed >= n_circles:
                break
            x = (j + 1) * spacing_x + (rect_width - grid_width) / 2
            y = (i + 1) * spacing_y + (rect_height - grid_height) / 2
            
            # Adjust position slightly to avoid regular patterns
            x += random.uniform(-spacing_x*0.2, spacing_x*0.2)
            y += random.uniform(-spacing_y*0.2, spacing_y*0.2)
            
            # Ensure within bounds
            x = np.clip(x, 0.01, rect_width - 0.01)
            y = np.clip(y, 0.01, rect_height - 0.01)
            
            # Initial radius based on available space
            max_radius = min(x, y, rect_width - x, rect_height - y)
            r = np.clip(max_radius * 0.2, 0.01, 0.2)
            
            circles[placed] = [x, y, r]
            placed += 1
            
        if placed >= n_circles:
            break
    
    # Fill remaining circles with random placements away from edges
    for i in range(placed, n_circles):
        x = random.uniform(0.05, rect_width - 0.05)
        y = random.uniform(0.05, rect_height - 0.05)
        # Radius based on distance from edges and nearby circles
        max_radius = min(x, y, rect_width - x, rect_height - y)
        r = min(max_radius * 0.3, 0.3)
        circles[i] = [x, y, r]
    
    return circles

def compute_forces(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """
    Compute net forces on each circle based on repulsion from others and attraction to walls.
    Uses physics-based model where circles repel each other with inverse-square law and 
    are attracted to container boundaries.
    """
    n = len(circles)
    forces = np.zeros((n, 2))  # [fx, fy] for each circle
    
    # Parameters for physics model
    repulsion_strength = 100.0
    wall_attraction_strength = 50.0
    min_distance = 0.001
    
    # Repulsion forces between circles
    for i in range(n):
        x1, y1, r1 = circles[i]
        
        for j in range(n):
            if i == j:
                continue
                
            x2, y2, r2 = circles[j]
            
            dx = x2 - x1
            dy = y2 - y1
            distance = np.sqrt(dx*dx + dy*dy)
            
            # Avoid division by zero
            if distance < min_distance:
                distance = min_distance
                
            # Calculate repulsion force (inverse square law)
            # Force decreases with distance squared
            force_magnitude = repulsion_strength / (distance * distance + 1e-8)
            
            # Direction
            fx = force_magnitude * dx / distance
            fy = force_magnitude * dy / distance
            
            # Ensure we don't push circles beyond their mutual minimum distance
            min_dist = r1 + r2
            if distance < min_dist:
                # Push them apart when they're too close
                separation_force = (min_dist - distance) * 1000
                fx += separation_force * dx / (distance + 1e-8)
                fy += separation_force * dy / (distance + 1e-8)
            
            forces[i, 0] += fx
            forces[i, 1] += fy
    
    # Wall attraction forces (push circles towards center and away from edges)
    for i in range(n):
        x, y, r = circles[i]
        
        # Attract to center (weaker to maintain some boundary utilization)
        center_fx = -(x - rect_width/2) * 0.1
        center_fy = -(y - rect_height/2) * 0.1
        
        # Attract to walls (stronger for boundary circles)
        wall_fx = wall_attraction_strength * (
            max(0, r - x) * (1 if x < rect_width/2 else -1) +
            max(0, r - (rect_width - x)) * (-1 if x > rect_width/2 else 1)
        )
        wall_fy = wall_attraction_strength * (
            max(0, r - y) * (1 if y < rect_height/2 else -1) +
            max(0, r - (rect_height - y)) * (-1 if y > rect_height/2 else 1)
        )
        
        forces[i, 0] += center_fx + wall_fx
        forces[i, 1] += center_fy + wall_fy
    
    return forces

def update_positions(circles: np.ndarray, forces: np.ndarray, dt: float = 0.01, 
                    rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """
    Update circle positions based on forces using simple integration.
    """
    updated = circles.copy()
    n = len(updated)
    
    for i in range(n):
        x, y, r = updated[i]
        fx, fy = forces[i]
        
        # Velocity (for this simplified physics, just use force directly)
        vx = fx * dt
        vy = fy * dt
        
        # New position
        new_x = x + vx
        new_y = y + vy
        
        # Boundary constraints - reflect if hitting walls
        if new_x - r < 0:
            new_x = r
            # Reverse velocity component if moving inward
            # We won't reverse because our physics handles this naturally through forces
        elif new_x + r > rect_width:
            new_x = rect_width - r
            
        if new_y - r < 0:
            new_y = r
        elif new_y + r > rect_height:
            new_y = rect_height - r
            
        # Apply updated position
        updated[i, 0] = new_x
        updated[i, 1] = new_y
    
    return updated

def simulate_physics_equilibrium(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0, 
                               max_steps: int = 500, tolerance: float = 1e-5) -> np.ndarray:
    """
    Simulate physics-based equilibrium until convergence.
    """
    current = circles.copy()
    prev_energy = float('inf')
    
    for step in range(max_steps):
        forces = compute_forces(current, rect_width, rect_height)
        updated = update_positions(current, forces, 0.01, rect_width, rect_height)
        
        # Check for convergence based on change in energy or positions
        delta_positions = np.sum(np.abs(updated[:, :2] - current[:, :2]))
        if delta_positions < tolerance:
            break
            
        current = updated
        # Optional: check energy change but not necessary for this approach
        
    return current

def compute_energy(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> float:
    """
    Compute total energy of the system (sum of repulsion energies).
    """
    n = len(circles)
    total_energy = 0.0
    
    # For simplicity, we'll just compute sum of radii as a proxy for performance metric
    # In a more complex version, we'd compute true energy including interaction terms
    return np.sum(circles[:, 2])

def compute_constraint_violations(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> Tuple[int, float]:
    """
    Compute number of constraint violations and total violation amount.
    """
    n = len(circles)
    violations = 0
    total_violation = 0.0
    
    # Check boundary violations
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
            violations += 1
            # Add violation amount
            if x - r < 0:
                total_violation += abs(x - r)
            if x + r > rect_width:
                total_violation += abs(x + r - rect_width)
            if y - r < 0:
                total_violation += abs(y - r)
            if y + r > rect_height:
                total_violation += abs(y + r - rect_height)
    
    # Check overlap violations
    if n > 1:
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        distances = cdist(positions, positions)
        
        for i in range(n):
            for j in range(i+1, n):
                distance = distances[i, j]
                min_distance = radii[i] + radii[j]
                if distance < min_distance:
                    violations += 1
                    total_violation += min_distance - distance
    
    return violations, total_violation

def optimize_with_physics_simulation(n_circles: int = 21, 
                                   rect_width: float = 1.0, 
                                   rect_height: float = 1.0,
                                   max_iterations: int = 50) -> np.ndarray:
    """
    Main optimization function using physics-based approach.
    """
    best_solution = None
    best_sum_radii = -float('inf')
    
    # Multiple restarts with different initializations
    for restart in range(10):
        # Initialize with different starting configurations
        circles = initialize_physics_based_layout(n_circles, rect_width, rect_height)
        
        # Some randomness to avoid local minima
        for i in range(n_circles):
            circles[i, 0] += random.uniform(-0.05, 0.05)
            circles[i, 1] += random.uniform(-0.05, 0.05)
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], rect_width - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], rect_height - circles[i, 2])
        
        # Run physics simulation to reach equilibrium
        final_solution = simulate_physics_equilibrium(circles, rect_width, rect_height, max_steps=300)
        
        # Validate and compute fitness
        violations, _ = compute_constraint_violations(final_solution, rect_width, rect_height)
        
        # Only consider valid solutions
        if violations == 0:
            sum_radii = np.sum(final_solution[:, 2])
            if sum_radii > best_sum_radii:
                best_sum_radii = sum_radii
                best_solution = final_solution.copy()
        
        # If we got a good solution early, try refinements
        if best_sum_radii > 1.5 and restart < 5:
            # Apply additional local refinement
            for _ in range(10):
                circles = best_solution.copy()
                # Slightly tweak the best solution to improve it
                for i in range(n_circles):
                    circles[i, 0] += random.uniform(-0.01, 0.01)
                    circles[i, 1] += random.uniform(-0.01, 0.01)
                    circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], rect_width - circles[i, 2])
                    circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], rect_height - circles[i, 2])
                
                # Run another physics simulation
                refined = simulate_physics_equilibrium(circles, rect_width, rect_height, max_steps=100)
                refined_sum = np.sum(refined[:, 2])
                
                if refined_sum > best_sum_radii:
                    best_sum_radii = refined_sum
                    best_solution = refined.copy()
    
    # If no valid solution found, return initial solution
    if best_solution is None:
        best_solution = initialize_physics_based_layout(n_circles, rect_width, rect_height)
    
    return best_solution

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: perimeter = 4 => width + height = 2
    # Optimize rectangle dimensions for better packing
    rect_width = 1.3
    rect_height = 0.7

    # Run physics simulation based optimization
    circles = optimize_with_physics_simulation(
        n_circles=21,
        rect_width=rect_width,
        rect_height=rect_height,
        max_iterations=30
    )

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")