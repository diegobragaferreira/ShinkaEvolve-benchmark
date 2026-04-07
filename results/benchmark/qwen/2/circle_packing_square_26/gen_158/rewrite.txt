# EVOLVE-BLOCK-START
import numpy as np
import random
import time
from scipy.spatial import cKDTree
from typing import Tuple, List, Optional
import math

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

class PhysicsBasedCirclePacker:
    def __init__(self, n_circles: int = 26, swarm_size: int = 100, max_iterations: int = 500):
        self.n_circles = n_circles
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.best_solution = None
        self.best_fitness = -float('inf')
        
    def validate_circles(self, circles: np.ndarray) -> bool:
        """Validate that circles are within bounds and non-overlapping."""
        if len(circles) != self.n_circles:
            return False
            
        # Check containment constraints
        for i in range(self.n_circles):
            x, y, r = circles[i]
            if r <= 0 or x < r or x > 1 - r or y < r or y > 1 - r:
                return False
        
        # Check overlap constraints using KDTree for efficiency
        points = circles[:, :2]
        tree = cKDTree(points)
        
        for i in range(self.n_circles):
            x, y, r = circles[i]
            # Find nearby circles (within 2*r distance)
            nearby = tree.query_ball_point([x, y], 2 * r)
            for j in nearby:
                if i != j:
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if distance < r + r2:
                        return False
        
        return True

    def calculate_fitness(self, circles: np.ndarray) -> float:
        """Calculate total radius sum as fitness."""
        return np.sum(circles[:, 2])

    def initialize_swarm(self) -> List[np.ndarray]:
        """Initialize swarm with diverse configurations using spatial distribution."""
        swarm = []
        
        for _ in range(self.swarm_size):
            # Create an initial solution using a hybrid approach
            circles = np.zeros((self.n_circles, 3))
            
            # Strategy: place circles in grid pattern with some randomness
            grid_rows = max(1, int(np.ceil(np.sqrt(self.n_circles))))
            grid_cols = max(1, int(np.ceil(self.n_circles / grid_rows)))
            
            spacing_x = 1.0 / (grid_cols + 1)
            spacing_y = 1.0 / (grid_rows + 1)
            
            idx = 0
            for i in range(grid_rows):
                for j in range(grid_cols):
                    if idx >= self.n_circles:
                        break
                        
                    x = (j + 1) * spacing_x + np.random.uniform(-spacing_x*0.1, spacing_x*0.1)
                    y = (i + 1) * spacing_y + np.random.uniform(-spacing_y*0.1, spacing_y*0.1)
                    
                    # Initial radius based on proximity to edges and a bit of randomness
                    r = min(x, 1-x, y, 1-y) * np.random.uniform(0.2, 0.4)
                    r = max(0.005, r)
                    
                    circles[idx] = [x, y, r]
                    idx += 1
            
            # Fill remaining circles with more randomized positions
            for i in range(idx, self.n_circles):
                max_attempts = 1000
                placed = False
                attempts = 0
                
                while not placed and attempts < max_attempts:
                    x = np.random.uniform(0.05, 0.95)
                    y = np.random.uniform(0.05, 0.95)
                    r = np.random.uniform(0.005, 0.15)
                    
                    # Check if valid placement
                    valid_placement = True
                    if r <= x <= 1 - r and r <= y <= 1 - r:
                        # Check overlap with existing circles
                        for j in range(i):
                            existing_x, existing_y, existing_r = circles[j]
                            distance = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                            if distance < r + existing_r:
                                valid_placement = False
                                break
                    else:
                        valid_placement = False
                    
                    if valid_placement:
                        circles[i] = [x, y, r]
                        placed = True
                    attempts += 1
                
                if not placed:
                    # Fallback to simple position with small radius
                    circles[i] = [0.5 + np.random.normal(0, 0.1), 0.5 + np.random.normal(0, 0.1), 0.01]
            
            swarm.append(circles)
            
        return swarm

    def compute_forces(self, circles: np.ndarray, iteration: int) -> np.ndarray:
        """Compute forces acting on each circle based on physics principles."""
        forces = np.zeros_like(circles)
        n = len(circles)
        
        # Repulsion forces from overlaps
        points = circles[:, :2]
        tree = cKDTree(points)
        
        # For each circle, compute repulsion force from nearby circles
        for i in range(n):
            x, y, r = circles[i]
            nearby = tree.query_ball_point([x, y], 2 * r)
            
            for j in nearby:
                if i != j:
                    x2, y2, r2 = circles[j]
                    dx = x - x2
                    dy = y - y2
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    if distance < r + r2 and distance > 0:
                        # Compute force magnitude (inverse relationship with distance)
                        force_magnitude = 1.0 / (distance + 1e-8)  # Add small epsilon
                        # Normalize direction
                        dx_norm = dx / (distance + 1e-8)
                        dy_norm = dy / (distance + 1e-8)
                        
                        # Apply force away from overlapping circle
                        forces[i, 0] += force_magnitude * dx_norm
                        forces[i, 1] += force_magnitude * dy_norm
                        
                        # Add small attractive force to prevent degenerate cases
                        forces[i, 0] -= 0.1 * dx_norm
                        forces[i, 1] -= 0.1 * dy_norm
        
        # Attraction forces to center (encourage spreading)
        center_x, center_y = 0.5, 0.5
        for i in range(n):
            x, y, r = circles[i]
            dx = center_x - x
            dy = center_y - y
            distance = np.sqrt(dx*dx + dy*dy)
            if distance > 0:
                # Scale attraction force with distance from center
                force_magnitude = 0.05 * (distance / 0.7)  # Normalize by max distance
                forces[i, 0] += force_magnitude * dx / distance
                forces[i, 1] += force_magnitude * dy / distance
                
        # Boundary repulsion (stronger near edges)
        for i in range(n):
            x, y, r = circles[i]
            # Repel from boundaries
            boundary_force_x = 0
            boundary_force_y = 0
            
            # Left boundary
            if x < r:
                boundary_force_x += (r - x) * 20.0
            # Right boundary  
            if x > 1 - r:
                boundary_force_x += (1 - r - x) * 20.0
            # Bottom boundary
            if y < r:
                boundary_force_y += (r - y) * 20.0
            # Top boundary
            if y > 1 - r:
                boundary_force_y += (1 - r - y) * 20.0
                
            forces[i, 0] += boundary_force_x
            forces[i, 1] += boundary_force_y
            
        # Apply velocity damping
        for i in range(n):
            # Scale forces based on iteration (gradually decrease influence)
            damp_factor = 1.0 - (iteration / self.max_iterations)
            forces[i, 0] *= 0.1 * damp_factor
            forces[i, 1] *= 0.1 * damp_factor
            
        return forces

    def update_position(self, circles: np.ndarray, forces: np.ndarray, iteration: int) -> np.ndarray:
        """Update circle positions using computed forces."""
        updated = circles.copy()
        dt = 0.01
        
        for i in range(len(updated)):
            x, y, r = updated[i]
            
            # Apply velocity (force scaled by time step)
            new_x = x + forces[i, 0] * dt
            new_y = y + forces[i, 1] * dt
            
            # Constrain positions to be within bounds
            new_x = max(r, min(1-r, new_x))
            new_y = max(r, min(1-r, new_y))
            
            updated[i] = [new_x, new_y, r]
            
        return updated

    def shrink_radii(self, circles: np.ndarray, iteration: int) -> np.ndarray:
        """Gradually shrink radii to improve packing density."""
        updated = circles.copy()
        
        # Gradually reduce radii as iteration progresses
        shrink_factor = 1.0 - (iteration / self.max_iterations) * 0.3
        
        for i in range(len(updated)):
            x, y, r = updated[i]
            # Only shrink if it helps maintain non-overlap
            new_r = r * shrink_factor
            new_r = max(0.001, new_r)
            updated[i] = [x, y, new_r]
            
        return updated

    def local_refinement(self, circles: np.ndarray, max_iterations: int = 30) -> np.ndarray:
        """Perform local optimization to improve solution quality."""
        current = circles.copy()
        
        for iteration in range(max_iterations):
            improved = False
            
            # Try to improve each circle individually
            for i in range(self.n_circles):
                original = current[i].copy()
                original_fitness = self.calculate_fitness(current)
                
                # Try small adjustments to position and radius
                step_sizes = [0.005, 0.01, 0.02]
                
                for step in step_sizes:
                    # Test position changes
                    for dx in [-step, 0, step]:
                        for dy in [-step, 0, step]:
                            new_x = original[0] + dx
                            new_y = original[1] + dy
                            
                            # Ensure new position is valid
                            if (new_x - original[2] >= 0 and 
                                new_x + original[2] <= 1 and 
                                new_y - original[2] >= 0 and 
                                new_y + original[2] <= 1):
                                
                                # Create temporary configuration
                                temp_circles = current.copy()
                                temp_circles[i] = [new_x, new_y, original[2]]
                                
                                # Check if this improves overall fitness and validity
                                if self.validate_circles(temp_circles):
                                    new_fitness = self.calculate_fitness(temp_circles)
                                    if new_fitness > original_fitness:
                                        current = temp_circles
                                        improved = True
                                        break
                    
                    # Test radius changes
                    for dr in [-step, 0, step]:
                        new_r = original[2] + dr
                        if new_r > 0.001 and new_r < 0.5:  # Reasonable bounds
                            # Ensure new radius allows for valid positioning
                            new_r = min(new_r, original[0], 1-original[0], original[1], 1-original[1])
                            if new_r > 0.001:
                                # Create temporary configuration
                                temp_circles = current.copy()
                                temp_circles[i] = [original[0], original[1], new_r]
                                
                                # Check if this improves overall fitness and validity
                                if self.validate_circles(temp_circles):
                                    new_fitness = self.calculate_fitness(temp_circles)
                                    if new_fitness > original_fitness:
                                        current = temp_circles
                                        improved = True
                                        break
                
                if improved:
                    break
                    
            # Stop if no improvement was made in this iteration
            if not improved:
                break
                
        return current

    def evolve(self) -> np.ndarray:
        """Main physics-based optimization algorithm."""
        # Initialize swarm
        swarm = self.initialize_swarm()
        
        # Track best solution
        best_solution = None
        best_fitness = -float('inf')
        
        # Main evolution loop
        for iteration in range(self.max_iterations):
            # Evaluate all swarm members
            fitnesses = [self.calculate_fitness(swarm[i]) for i in range(self.swarm_size)]
            
            # Update global best
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > best_fitness:
                best_fitness = fitnesses[max_fitness_idx]
                best_solution = swarm[max_fitness_idx].copy()
                
                # Print progress every 50 iterations
                if iteration % 50 == 0:
                    print(f"Iteration {iteration}: Best fitness = {best_fitness:.6f}")
            
            # Process each member in the swarm
            for i in range(self.swarm_size):
                # Compute forces on this member
                forces = self.compute_forces(swarm[i], iteration)
                
                # Update positions
                swarm[i] = self.update_position(swarm[i], forces, iteration)
                
                # Apply gradual shrinking to improve density
                swarm[i] = self.shrink_radii(swarm[i], iteration)
                
                # Ensure validity after updates
                if not self.validate_circles(swarm[i]):
                    # Repair invalid solution
                    swarm[i] = self.repair_solution(swarm[i])
            
            # Occasionally apply local refinement to the best solution
            if iteration % 20 == 0 and best_solution is not None:
                best_solution = self.local_refinement(best_solution)
                # Update best fitness
                best_fitness = self.calculate_fitness(best_solution)
        
        # Final local refinement on best solution
        if best_solution is not None:
            best_solution = self.local_refinement(best_solution)
            
        return best_solution if best_solution is not None else swarm[0]

    def repair_solution(self, circles: np.ndarray) -> np.ndarray:
        """Repair any invalid solution by repositioning circles."""
        repaired = circles.copy()
        
        # Place circles back into valid positions
        for i in range(self.n_circles):
            x, y, r = repaired[i]
            
            # Ensure radius is valid
            r = max(0.001, min(0.5, r))
            
            # Ensure position is valid
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            
            repaired[i] = [x, y, r]
        
        # Resolve overlaps using iterative approach
        for attempt in range(100):
            valid = self.validate_circles(repaired)
            if valid:
                break
                
            # Reduce all radii slightly
            for i in range(self.n_circles):
                x, y, r = repaired[i]
                repaired[i] = [x, y, max(0.001, r * 0.95)]
        
        return repaired

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    start_time = time.time()
    
    try:
        # Create physics-based optimizer
        optimizer = PhysicsBasedCirclePacker(n_circles=26, swarm_size=100, max_iterations=500)
        
        # Run evolution
        circles = optimizer.evolve()
        
        # Validate result
        if not optimizer.validate_circles(circles):
            # If validation fails, try a simpler approach
            print("Validation failed on evolved solution, using fallback...")
            circles = np.zeros((26, 3))
            # Use a simple heuristic: distribute evenly with decreasing radii
            for i in range(26):
                circles[i] = [0.5, 0.5, 0.01]
        
        end_time = time.time()
        eval_time = end_time - start_time
        print(f"Physics-based evolution completed in {eval_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error during evolution: {e}")
        # Fallback to simple initialization
        circles = np.zeros((26, 3))
        print("Using fallback solution due to error")
    
    return circles

# EVOLVE-BLOCK-END