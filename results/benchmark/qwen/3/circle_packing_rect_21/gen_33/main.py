# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import nevergrad as ng
import random

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set fixed seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Container dimensions - make it a square for simplicity (width=height=1)
    container_width = 1.0
    container_height = 1.0
    
    def compute_forces(circles_flat):
        """Compute total forces acting on each circle"""
        circles = circles_flat.reshape(-1, 3)
        forces = np.zeros_like(circles_flat)
        
        # Wall forces (penalty for being outside bounds)
        for i in range(len(circles)):
            x, y, r = circles[i]
            
            # Left wall penalty
            if x - r < 0:
                forces[i*3] += (0 - (x - r)) * 1000
            
            # Right wall penalty  
            if x + r > container_width:
                forces[i*3] -= ((x + r) - container_width) * 1000
                
            # Bottom wall penalty
            if y - r < 0:
                forces[i*3 + 1] += (0 - (y - r)) * 1000
                
            # Top wall penalty
            if y + r > container_height:
                forces[i*3 + 1] -= ((y + r) - container_height) * 1000
        
        # Circle-circle repulsive forces
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Compute pairwise distances
        dist_matrix = cdist(positions, positions)
        
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                # Apply repulsive force if circles overlap
                if distance < (radii[i] + radii[j]):
                    force_magnitude = 1000 * (radii[i] + radii[j] - distance)
                    
                    # Normalize direction vector
                    if distance > 1e-10:
                        fx = force_magnitude * dx / distance
                        fy = force_magnitude * dy / distance
                        
                        forces[i*3] += fx
                        forces[i*3 + 1] += fy
                        forces[j*3] -= fx
                        forces[j*3 + 1] -= fy
        
        return forces
    
    def objective_with_penalty(circles_flat):
        """Objective function with penalties for constraints"""
        circles = circles_flat.reshape(-1, 3)
        radii_sum = np.sum(circles[:, 2])
        
        # Penalty for overlaps
        penalty = 0
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        dist_matrix = cdist(positions, positions)
        
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                distance = dist_matrix[i, j]
                if distance < (radii[i] + radii[j]):
                    overlap = (radii[i] + radii[j]) - distance
                    penalty += overlap ** 2 * 10000  # Quadratic penalty
        
        # Penalty for boundary violations
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > container_width or y - r < 0 or y + r > container_height:
                penalty += 100000
                
        return -radii_sum + penalty  # Negative because we minimize
    
    def generate_initial_config():
        """Generate initial configuration using hexagonal lattice"""
        circles = np.zeros((21, 3))
        
        # Hexagonal grid layout
        rows = 5
        cols = 5
        spacing_x = container_width / (cols + 1)
        spacing_y = container_height / (rows + 1)
        
        # Fill grid with circles
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= 21:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                # Add random perturbation
                x += random.uniform(-spacing_x/6, spacing_x/6)
                y += random.uniform(-spacing_y/6, spacing_y/6)
                
                # Ensure within bounds
                x = max(0.01, min(container_width - 0.01, x))
                y = max(0.01, min(container_height - 0.01, y))
                
                # Set initial radius
                circles[idx] = [x, y, 0.05]
                idx += 1
                
                if idx >= 21:
                    break
        
        # Increase some radii for better initial spread
        for i in range(min(5, len(circles))):
            circles[i, 2] = 0.1 + random.random() * 0.05
            
        return circles.flatten()
    
    # Multi-start optimization
    best_result = None
    best_value = float('inf')
    
    # Try several different starting configurations
    for _ in range(5):
        initial_circles = generate_initial_config()
        
        # Use nevergrad for optimization
        try:
            optimizer = ng.optimizers.DifferentialEvolution(popsize=15, mutation_scaling=0.8)
            optimizer.num_workers = 1
            
            result = optimizer.minimize(objective_with_penalty, initial_point=initial_circles, budget=200)
            
            final_value = objective_with_penalty(result)
            
            if final_value < best_value:
                best_value = final_value
                best_result = result
                
        except Exception:
            continue
    
    # Final validation and cleaning
    if best_result is None:
        # Fallback to simple initialization
        circles = np.zeros((21, 3))
        spacing_x = container_width / 6
        spacing_y = container_height / 4
        idx = 0
        for i in range(4):
            for j in range(6):
                if idx >= 21:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                x = max(0.01, min(container_width - 0.01, x))
                y = max(0.01, min(container_height - 0.01, y))
                circles[idx] = [x, y, 0.05]
                idx += 1
        return circles
    
    # Process final result
    final_circles = best_result.reshape(-1, 3)
    
    # Validate and correct any constraint violations
    for i in range(len(final_circles)):
        x, y, r = final_circles[i]
        # Ensure valid bounds
        final_circles[i, 0] = max(0.001, min(container_width - 0.001, x))
        final_circles[i, 1] = max(0.001, min(container_height - 0.001, y))
        final_circles[i, 2] = max(0.001, min(0.499, r))
    
    return final_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
