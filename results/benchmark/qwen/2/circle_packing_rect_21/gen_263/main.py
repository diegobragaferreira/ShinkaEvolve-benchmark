# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import math
import random
import time
from typing import Tuple, List

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Container setup (perimeter = 4, so width + height = 2)
    container_width = 1.3
    container_height = 0.7
    
    # Spiral-based initialization approach
    def initialize_spiral_placement(n_circles: int, width: float, height: float) -> np.ndarray:
        """Initialize circles using spiral pattern that systematically covers the space"""
        circles = np.zeros((n_circles, 3))
        
        # Parameters for spiral
        a = 0.05  # Spiral parameter
        b = 0.1   # Spiral parameter
        max_radius = min(width, height) * 0.15
        
        # Generate spiral points that cover the rectangle
        num_points = max(50, n_circles * 3)  # More points for better coverage
        angles = np.linspace(0, 4 * np.pi, num_points)
        
        spiral_points = []
        for angle in angles:
            r = a + b * angle
            x = width/2 + r * np.cos(angle) * width/4
            y = height/2 + r * np.sin(angle) * height/4
            if 0 <= x <= width and 0 <= y <= height:
                spiral_points.append([x, y])
        
        # Sample points for circle placement
        sample_indices = np.linspace(0, len(spiral_points)-1, n_circles).astype(int)
        sampled_points = [spiral_points[i] for i in sample_indices[:n_circles]]
        
        # Place circles with adaptive radii
        for i, (x, y) in enumerate(sampled_points):
            # Calculate maximum allowed radius for this position
            max_r = min(x, width - x, y, height - y)
            # Start with conservative radius and allow adjustment
            r = min(max_r * 0.5, max_radius)
            # Add some randomness to avoid symmetric solutions
            r *= random.uniform(0.7, 1.0)
            circles[i] = [x, y, r]
        
        return circles
    
    # Constraint validation functions
    def is_valid_position(x: float, y: float, r: float, width: float, height: float) -> bool:
        """Check if circle position is within bounds"""
        return (r <= x <= width - r and r <= y <= height - r)
    
    def check_overlap(pos1: Tuple[float, float], r1: float, pos2: Tuple[float, float], r2: float) -> bool:
        """Check if two circles overlap"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        distance_sq = dx*dx + dy*dy
        radius_sum = r1 + r2
        return distance_sq < radius_sum * radius_sum
    
    # Energy-based optimization approach
    def optimize_with_energy_model(circles: np.ndarray, width: float, height: float, max_iterations: int = 1000) -> np.ndarray:
        """Optimize using physics-inspired energy minimization"""
        # Create initial spatial tree for efficient overlap checking
        tree = cKDTree(circles[:, :2])
        
        best_circles = circles.copy()
        best_sum = np.sum(circles[:, 2])
        
        # Energy parameters
        overlap_penalty_scale = 1000.0
        boundary_penalty_scale = 10000.0
        repulsion_scale = 100.0
        attraction_scale = 1.0
        
        for iteration in range(max_iterations):
            improved = False
            # Process circles in shuffled order for better exploration
            indices = list(range(len(circles)))
            random.shuffle(indices)
            
            # For each circle, compute forces and update position/radius
            for i in indices:
                x, y, r = best_circles[i]
                
                # Compute forces from nearby circles
                forces = [0.0, 0.0]  # [dx, dy]
                overlap_penalty = 0.0
                boundary_penalty = 0.0
                
                # Find nearby circles using spatial tree
                nearby_indices = tree.query_ball_point([x, y], 2 * (r + 0.1))
                
                for j in nearby_indices:
                    if i != j:
                        x2, y2, r2 = best_circles[j]
                        dx = x2 - x
                        dy = y2 - y
                        distance = math.sqrt(dx*dx + dy*dy)
                        
                        if distance > 0:
                            # Overlap penalty (repulsion)
                            if distance < (r + r2):
                                overlap_penalty += (r + r2 - distance) * overlap_penalty_scale
                                # Repulsion force
                                force_mag = repulsion_scale * (1.0 - distance/(r + r2)) / (distance + 1e-8)
                                forces[0] += force_mag * dx / distance
                                forces[1] += force_mag * dy / distance
                            # Attraction towards center if too close to boundary
                            if distance < (r + r2) * 0.5:
                                # Attract to center of rectangle
                                center_attraction = attraction_scale * (1.0 - distance/(r + r2))
                                forces[0] -= center_attraction * dx / (distance + 1e-8)
                                forces[1] -= center_attraction * dy / (distance + 1e-8)
                
                # Penalty for boundary violations
                if x - r < 0.01:
                    boundary_penalty += (r - x) * boundary_penalty_scale
                    forces[0] += (0.01 - (x - r)) * boundary_penalty_scale
                if x + r > width - 0.01:
                    boundary_penalty += ((x + r) - (width - 0.01)) * boundary_penalty_scale
                    forces[0] -= ((x + r) - (width - 0.01)) * boundary_penalty_scale
                if y - r < 0.01:
                    boundary_penalty += (r - y) * boundary_penalty_scale
                    forces[1] += (0.01 - (y - r)) * boundary_penalty_scale
                if y + r > height - 0.01:
                    boundary_penalty += ((y + r) - (height - 0.01)) * boundary_penalty_scale
                    forces[1] -= ((y + r) - (height - 0.01)) * boundary_penalty_scale
                
                # Update position based on forces
                new_x = x + forces[0] * 0.001
                new_y = y + forces[1] * 0.001
                
                # Keep within bounds
                new_x = max(r + 0.01, min(width - r - 0.01, new_x))
                new_y = max(r + 0.01, min(height - r - 0.01, new_y))
                
                # Test if this improves the configuration
                test_circles = best_circles.copy()
                test_circles[i] = [new_x, new_y, r]
                
                # Check validity
                valid = True
                if not is_valid_position(new_x, new_y, r, width, height):
                    valid = False
                else:
                    for k in range(len(test_circles)):
                        if k != i:
                            x2, y2, r2 = test_circles[k]
                            if check_overlap([new_x, new_y], r, [x2, y2], r2):
                                valid = False
                                break
                
                if valid:
                    best_circles[i] = [new_x, new_y, r]
                    improved = True
                
                # Periodically recompute spatial tree for efficiency
                if iteration % 50 == 0:
                    tree = cKDTree(best_circles[:, :2])
            
            # Recompute total sum
            current_sum = np.sum(best_circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                improved = True
            
            # Early stopping if no significant improvement
            if not improved and iteration > max_iterations // 2:
                break
        
        return best_circles
    
    # Multi-resolution refinement to escape local optima
    def multi_resolution_refinement(initial_circles: np.ndarray, width: float, height: float) -> np.ndarray:
        """Perform optimization at multiple resolution scales"""
        current_config = initial_circles.copy()
        
        # Coarse resolution
        coarse_config = current_config.copy()
        # Make circles larger for coarse optimization to capture global structure
        for i in range(len(coarse_config)):
            coarse_config[i, 2] *= 1.2  # Slightly larger radii for coarse optimization
        
        coarse_config = optimize_with_energy_model(coarse_config, width, height, 300)
        
        # Medium resolution refinement
        medium_config = coarse_config.copy()
        # Slightly reduce radii to refine structure
        for i in range(len(medium_config)):
            medium_config[i, 2] *= 0.95
        
        medium_config = optimize_with_energy_model(medium_config, width, height, 500)
        
        # Fine resolution final optimization
        fine_config = medium_config.copy()
        fine_config = optimize_with_energy_model(fine_config, width, height, 700)
        
        return fine_config
    
    # Main optimization process
    # Step 1: Initialize using spiral method
    circles = initialize_spiral_placement(21, container_width, container_height)
    
    # Step 2: Multi-resolution optimization
    optimized_circles = multi_resolution_refinement(circles, container_width, container_height)
    
    # Step 3: Final refinement with constraint checking
    final_circles = optimized_circles.copy()
    
    # Validate and fix any constraint violations
    for _ in range(500):
        improved = False
        for i in range(len(final_circles)):
            x, y, r = final_circles[i]
            
            # Check if circle violates boundaries
            if x - r < 0.01:
                final_circles[i, 0] = r + 0.01
                improved = True
            elif x + r > container_width - 0.01:
                final_circles[i, 0] = container_width - r - 0.01
                improved = True
            if y - r < 0.01:
                final_circles[i, 1] = r + 0.01
                improved = True
            elif y + r > container_height - 0.01:
                final_circles[i, 1] = container_height - r - 0.01
                improved = True
            
            # Check for overlaps and resolve
            for j in range(len(final_circles)):
                if i != j:
                    x2, y2, r2 = final_circles[j]
                    dx = x - x2
                    dy = y - y2
                    distance = math.sqrt(dx*dx + dy*dy)
                    if distance < (r + r2) and distance > 0:
                        # Reduce radius to resolve overlap
                        new_r = max(0.001, (distance - 0.001) / 2)
                        if new_r < r:
                            final_circles[i, 2] = new_r
                            improved = True
                            break
        
        if not improved:
            break
    
    return final_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")