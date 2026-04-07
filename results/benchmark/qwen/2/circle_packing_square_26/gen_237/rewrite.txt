# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.spatial import cKDTree
from typing import Tuple, List
import time

# Set global random seed for reproducibility
random.seed(42)
np.random.seed(42)

class PhysicsCirclePacker:
    def __init__(self, num_circles: int = 26):
        self.num_circles = num_circles
        self.max_iterations = 5000
        self.force_iteration_limit = 1000
        self.convergence_threshold = 1e-6
        
    def is_valid_placement(self, circles: np.ndarray) -> bool:
        """Check if all circles are within bounds and don't overlap"""
        n = len(circles)
        
        # Check containment constraints
        for i in range(n):
            x, y, r = circles[i]
            if r <= 0 or x < r or x > 1-r or y < r or y > 1-r:
                return False

        # Check overlap constraints using KDTree for efficiency
        points = circles[:, :2]
        tree = cKDTree(points)

        for i in range(n):
            x, y, r = circles[i]
            # Find nearby circles (within 2*r distance)
            indices = tree.query_ball_point([x, y], 2*r)
            for j in indices:
                if i != j:
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if distance < r + r2:
                        return False

        return True

    def calculate_sum_radii(self, circles: np.ndarray) -> float:
        """Calculate sum of all radii"""
        return np.sum(circles[:, 2])
    
    def generate_initial_configuration(self) -> np.ndarray:
        """Generate initial configuration using improved heuristic"""
        circles = np.zeros((self.num_circles, 3))
        
        # Create a hexagonal-like grid pattern
        rows = 5
        cols = 6
        
        # Hexagonal spacing
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= self.num_circles:
                    break
                # Offset every other row for hexagonal packing
                x_offset = 0 if i % 2 == 0 else spacing_x / 2
                x = (j + 1) * spacing_x + x_offset + random.uniform(-spacing_x*0.1, spacing_x*0.1)
                y = (i + 1) * spacing_y + random.uniform(-spacing_y*0.1, spacing_y*0.1)
                
                # Initial radius based on proximity to boundaries
                r = min(x, 1-x, y, 1-y) * random.uniform(0.3, 0.5)
                circles[idx] = [x, y, r]
                idx += 1
                
        # Fill remaining positions with random distributions
        for i in range(idx, self.num_circles):
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            r = min(0.1, 0.5 * min(x, 1-x, y, 1-y))
            circles[i] = [x, y, r]
            
        return circles
    
    def apply_physics_repulsion(self, circles: np.ndarray, iteration: int) -> np.ndarray:
        """Apply physics-based repulsion forces to resolve overlaps"""
        # Create a copy to avoid modifying during computation
        updated_circles = circles.copy()
        n = len(updated_circles)
        
        # For each circle, compute net force from all other circles
        forces = np.zeros((n, 2))
        
        # Calculate forces between all pairs
        for i in range(n):
            x1, y1, r1 = updated_circles[i]
            for j in range(n):
                if i != j:
                    x2, y2, r2 = updated_circles[j]
                    dx = x2 - x1
                    dy = y2 - y1
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    # Only consider close enough neighbors
                    if distance < r1 + r2 + 0.01:  # Add small buffer
                        if distance > 0:
                            # Repulsive force (inverse square law)
                            force_magnitude = 0.01 / (distance * distance + 1e-8)
                            # Normalize force direction
                            fx = dx * force_magnitude / distance
                            fy = dy * force_magnitude / distance
                            forces[i, 0] += fx
                            forces[i, 1] += fy
        
        # Apply forces to positions with damping (velocity)
        damping = 0.5  # Damping factor to prevent oscillations
        for i in range(n):
            x, y, r = updated_circles[i]
            # Limit force magnitude to prevent extreme moves
            force_magnitude = np.sqrt(forces[i, 0]**2 + forces[i, 1]**2)
            if force_magnitude > 0.1:
                forces[i, 0] *= 0.1 / force_magnitude
                forces[i, 1] *= 0.1 / force_magnitude
                
            # Update position with damping
            new_x = x + forces[i, 0] * damping * (iteration / 100.0 + 1.0)
            new_y = y + forces[i, 1] * damping * (iteration / 100.0 + 1.0)
            
            # Keep within bounds
            new_x = np.clip(new_x, r, 1-r)
            new_y = np.clip(new_y, r, 1-r)
            
            updated_circles[i, 0] = new_x
            updated_circles[i, 1] = new_y
            
        return updated_circles
    
    def maximize_radii(self, circles: np.ndarray) -> np.ndarray:
        """Try to maximize radii of circles while keeping constraints"""
        n = len(circles)
        updated_circles = circles.copy()
        
        # Iteratively try to increase radii
        for _ in range(50):
            improved = False
            
            # Process circles in random order
            order = list(range(n))
            random.shuffle(order)
            
            for i in order:
                x, y, r = updated_circles[i]
                
                # Calculate maximum possible radius
                max_r = min(x, 1-x, y, 1-y)
                
                if max_r > r + 1e-6:
                    # Try to increase radius as much as possible
                    new_r = max_r
                    
                    # Check overlap with all neighbors
                    valid_radius = True
                    for j in range(n):
                        if i != j:
                            x2, y2, r2 = updated_circles[j]
                            distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                            if distance < new_r + r2:
                                valid_radius = False
                                break
                    
                    if valid_radius:
                        updated_circles[i, 2] = new_r
                        improved = True
                        
            if not improved:
                break
                
        return updated_circles
    
    def optimize(self) -> np.ndarray:
        """Main optimization loop combining physics simulation and local optimization"""
        # Start with good initial configuration
        circles = self.generate_initial_configuration()
        
        # Apply physics-based repulsion and refinement
        best_circles = circles.copy()
        best_fitness = self.calculate_sum_radii(best_circles)
        
        # Physics simulation with iterative improvements
        for iteration in range(self.max_iterations):
            # Apply repulsion forces
            circles = self.apply_physics_repulsion(circles, iteration)
            
            # Local optimization to maximize radii
            circles = self.maximize_radii(circles)
            
            # Check if current solution is better
            current_fitness = self.calculate_sum_radii(circles)
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_circles = circles.copy()
                
            # Early stopping if we're getting close to convergence
            if iteration > 100 and abs(current_fitness - best_fitness) < self.convergence_threshold:
                break
                
        # Final refinement through local optimization
        refined_circles = self.maximize_radii(best_circles)
        
        # Final validation
        if not self.is_valid_placement(refined_circles):
            # If invalid, fallback to a simpler approach
            fallback_circles = self.generate_initial_configuration()
            refined_circles = self.maximize_radii(fallback_circles)
            
        return refined_circles

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates
                 of the i-th circle of radius r.
    """
    packer = PhysicsCirclePacker(26)
    return packer.optimize()

# EVOLVE-BLOCK-END