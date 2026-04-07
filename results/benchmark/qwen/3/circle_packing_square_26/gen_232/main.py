# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
import time
from typing import Tuple, List

# Global constants for optimization
MAX_ITERATIONS = 10000
LEARNING_RATE = 0.01
FORCE_MAGNITUDE = 10.0
CENTRAL_ATTRACTION = 0.1
BOUNDARY_REPULSION = 100.0
OVERLAP_REPULSION = 1000.0
TOLERANCE = 1e-6
GRADIENT_THRESHOLD = 1e-4
INITIAL_RADIUS_SCALE = 0.05
MIN_RADIUS = 0.001
MAX_RADIUS = 0.49

class CirclePackingGravityOptimizer:
    def __init__(self):
        self.n_circles = 26
        
    def initialize_circles(self) -> np.ndarray:
        """Initialize circles with better spatial distribution using Poisson disk sampling"""
        circles = np.zeros((self.n_circles, 3))
        
        # Generate points using a simple grid-based approach with jitter for better distribution
        grid_size = int(np.ceil(np.sqrt(self.n_circles)))
        idx = 0
        
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= self.n_circles:
                    break
                x = 0.1 + (i / (grid_size - 1)) * 0.8 if grid_size > 1 else 0.5
                y = 0.1 + (j / (grid_size - 1)) * 0.8 if grid_size > 1 else 0.5
                
                # Add jitter for better distribution
                x += random.uniform(-0.05, 0.05)
                y += random.uniform(-0.05, 0.05)
                
                # Ensure within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                circles[idx] = [x, y, INITIAL_RADIUS_SCALE]
                idx += 1
                
        # Fill remaining circles
        for i in range(idx, self.n_circles):
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            circles[i] = [x, y, INITIAL_RADIUS_SCALE]
            
        return circles
    
    def calculate_forces(self, circles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate forces acting on each circle"""
        n = len(circles)
        forces = np.zeros((n, 2))
        torque = np.zeros(n)
        
        # Central attraction force
        for i in range(n):
            x, y, r = circles[i]
            dx = 0.5 - x
            dy = 0.5 - y
            distance = np.sqrt(dx*dx + dy*dy)
            if distance > 0:
                magnitude = CENTRAL_ATTRACTION / (distance + 0.01)
                forces[i, 0] += dx * magnitude / distance
                forces[i, 1] += dy * magnitude / distance
        
        # Pairwise repulsion forces (overlap prevention)
        positions = circles[:, :2]
        
        # Use efficient pairwise distance computation
        distances = cdist(positions, positions)
        
        for i in range(n):
            for j in range(i+1, n):
                distance = distances[i, j]
                if distance < (circles[i, 2] + circles[j, 2] + 0.001) and distance > 0:
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    # Repulsive force proportional to overlap
                    overlap = (circles[i, 2] + circles[j, 2] - distance)
                    force_magnitude = OVERLAP_REPULSION * overlap / (distance + 0.01)
                    forces[i, 0] += dx * force_magnitude / distance
                    forces[i, 1] += dy * force_magnitude / distance
                    forces[j, 0] -= dx * force_magnitude / distance
                    forces[j, 1] -= dy * force_magnitude / distance
        
        # Boundary repulsion
        for i in range(n):
            x, y, r = circles[i]
            # Repulsion from boundaries
            boundary_force_x = 0
            boundary_force_y = 0
            
            # Left boundary
            if x - r < 0:
                boundary_force_x += BOUNDARY_REPULSION * (0 - (x - r))
            # Right boundary
            if x + r > 1:
                boundary_force_x += BOUNDARY_REPULSION * ((x + r) - 1)
            # Bottom boundary
            if y - r < 0:
                boundary_force_y += BOUNDARY_REPULSION * (0 - (y - r))
            # Top boundary
            if y + r > 1:
                boundary_force_y += BOUNDARY_REPULSION * ((y + r) - 1)
                
            forces[i, 0] += boundary_force_x
            forces[i, 1] += boundary_force_y
        
        return forces, torque
    
    def update_circles(self, circles: np.ndarray, forces: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """Update circle positions based on forces"""
        updated = circles.copy()
        
        for i in range(len(updated)):
            x, y, r = updated[i]
            
            # Update position based on force
            dx = forces[i, 0] * dt
            dy = forces[i, 1] * dt
            
            # Apply movement
            new_x = x + dx
            new_y = y + dy
            
            # Keep within bounds
            new_x = max(r, min(1 - r, new_x))
            new_y = max(r, min(1 - r, new_y))
            
            updated[i, 0] = new_x
            updated[i, 1] = new_y
            
        return updated
    
    def enforce_constraints(self, circles: np.ndarray) -> np.ndarray:
        """Ensure all circles are within bounds and have valid radii"""
        constrained = circles.copy()
        
        for i in range(len(constrained)):
            x, y, r = constrained[i]
            
            # Adjust radius to fit within bounds
            max_radius = min(x, 1 - x, y, 1 - y)
            r = min(MAX_RADIUS, max(MIN_RADIUS, max_radius, r))
            
            # Adjust position to ensure circle fits
            x = max(r, min(1 - r, x))
            y = max(r, min(1 - r, y))
            
            constrained[i] = [x, y, r]
            
        return constrained
    
    def resolve_overlaps(self, circles: np.ndarray, iterations: int = 10) -> np.ndarray:
        """Iteratively resolve overlaps using force-based correction"""
        resolved = circles.copy()
        
        for _ in range(iterations):
            changed = False
            positions = resolved[:, :2]
            radii = resolved[:, 2]
            
            # Compute pairwise distances
            distances = cdist(positions, positions)
            
            for i in range(len(resolved)):
                for j in range(i+1, len(resolved)):
                    if distances[i, j] < (radii[i] + radii[j]):
                        # Resolve overlap
                        dx = positions[i, 0] - positions[j, 0]
                        dy = positions[i, 1] - positions[j, 1]
                        distance = distances[i, j]
                        
                        if distance > 0.001:
                            # Normalize
                            dx /= distance
                            dy /= distance
                            
                            # Calculate overlap
                            overlap = (radii[i] + radii[j] - distance)
                            
                            # Apply corrective moves
                            move_amount = overlap * 0.5
                            resolved[i, 0] += dx * move_amount * 0.3
                            resolved[i, 1] += dy * move_amount * 0.3
                            resolved[j, 0] -= dx * move_amount * 0.3
                            resolved[j, 1] -= dy * move_amount * 0.3
                            
                            changed = True
            
            # Enforce bounds
            for i in range(len(resolved)):
                x, y, r = resolved[i]
                x = max(r, min(1 - r, x))
                y = max(r, min(1 - r, y))
                resolved[i] = [x, y, r]
            
            if not changed:
                break
                
        return resolved
    
    def compute_total_radius(self, circles: np.ndarray) -> float:
        """Compute total sum of radii"""
        return np.sum(circles[:, 2])
    
    def is_valid_configuration(self, circles: np.ndarray) -> bool:
        """Check if configuration is valid (all circles within bounds)"""
        for x, y, r in circles:
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        return True
    
    def optimize_step(self, circles: np.ndarray) -> Tuple[np.ndarray, float]:
        """Perform one optimization step"""
        # Calculate forces
        forces, _ = self.calculate_forces(circles)
        
        # Update positions
        updated_circles = self.update_circles(circles, forces)
        
        # Enforce constraints
        constrained_circles = self.enforce_constraints(updated_circles)
        
        # Resolve overlaps
        resolved_circles = self.resolve_overlaps(constrained_circles)
        
        # Compute total radius
        total_radius = self.compute_total_radius(resolved_circles)
        
        return resolved_circles, total_radius

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    optimizer = CirclePackingGravityOptimizer()
    
    # Initialize circles
    circles = optimizer.initialize_circles()
    
    # Optimization loop
    start_time = time.time()
    
    best_total_radius = 0.0
    best_circles = circles.copy()
    
    # Run optimization steps
    for iteration in range(MAX_ITERATIONS):
        circles, total_radius = optimizer.optimize_step(circles)
        
        if total_radius > best_total_radius:
            best_total_radius = total_radius
            best_circles = circles.copy()
        
        # Simple convergence check
        if iteration > 100 and iteration % 100 == 0:
            pass  # Just for progress tracking
    
    elapsed = time.time() - start_time
    print(f"Final result: Best radius sum = {best_total_radius:.6f} Time: {elapsed:.2f}s")
    print(f"Benchmark ratio: {best_total_radius / 2.6358627564136983:.6f}")
    
    return best_circles

# EVOLVE-BLOCK-END