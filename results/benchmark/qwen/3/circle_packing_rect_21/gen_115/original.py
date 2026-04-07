# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import math
import random
from typing import Tuple, List, Optional

class CirclePacker:
    def __init__(self, n_circles: int = 21, perimeter: float = 4.0):
        self.n_circles = n_circles
        self.perimeter = perimeter
        self.width = 1.2
        self.height = 0.8
        self.max_iterations = 1000
        
    def initialize_layout(self) -> np.ndarray:
        """Initialize circles using enhanced hexagonal grid with strategic seeding."""
        circles = np.zeros((self.n_circles, 3))
        
        # Use a modified hexagonal pattern for better distribution
        rows = 5
        cols = 5
        
        # Calculate spacing for hexagonal packing
        spacing_x = self.width / (cols + 1)
        spacing_y = self.height / (rows + 1)
        
        # Hexagonal offset for alternate rows
        offset = spacing_x * 0.5
        
        # Generate hexagonal grid points
        hex_points = []
        
        for i in range(rows):
            for j in range(cols):
                x = (j + 1) * spacing_x
                if i % 2 == 1:
                    x += offset
                y = (i + 1) * spacing_y
                if 0 <= x <= self.width and 0 <= y <= self.height:
                    hex_points.append([x, y])
        
        # Add strategic corner and edge points for better coverage
        corner_points = [
            [0.1, 0.1], [self.width-0.1, 0.1], [0.1, self.height-0.1], [self.width-0.1, self.height-0.1],
            [self.width/2, 0.1], [self.width/2, self.height-0.1], [0.1, self.height/2], [self.width-0.1, self.height/2]
        ]
        
        # Add corner points if we have room
        available_points = hex_points.copy()
        for point in corner_points:
            if len(available_points) < self.n_circles and 0 <= point[0] <= self.width and 0 <= point[1] <= self.height:
                available_points.append(point)
        
        # Fill with initial positions
        for i in range(self.n_circles):
            if i < len(available_points):
                x, y = available_points[i]
            else:
                # Random placement with boundary constraints
                x = np.random.uniform(0.1, self.width - 0.1)
                y = np.random.uniform(0.1, self.height - 0.1)
                
            # Compute max radius that fits at this location
            max_radius = self.compute_max_radius(x, y, circles[:i])
            circles[i] = [x, y, max_radius]
        
        return circles
    
    def compute_max_radius(self, x: float, y: float, existing_circles: np.ndarray) -> float:
        """Compute maximum radius for a circle at (x,y) given existing circles and container boundaries."""
        # Boundary constraints
        min_dist_from_edge = min(x, self.width - x, y, self.height - y)
        
        if min_dist_from_edge <= 0:
            return 0
        
        # Overlap constraints with existing circles
        min_dist_from_others = float('inf')
        
        for circle in existing_circles:
            if circle[2] > 0:  # Only consider placed circles
                cx, cy, cr = circle
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                min_dist_from_others = min(min_dist_from_others, dist - cr)
        
        # Take minimum of boundary and overlap constraints
        max_radius = min(min_dist_from_edge, min_dist_from_others)
        
        return max(0.001, max_radius)
    
    def coarse_optimization(self, circles: np.ndarray) -> np.ndarray:
        """Coarse global optimization with large step sizes."""
        current_circles = circles.copy()
        improved = True
        iteration = 0
        
        while improved and iteration < 300:
            improved = False
            iteration += 1
            
            # Shuffle circle indices for better exploration
            indices = list(range(len(current_circles)))
            random.shuffle(indices)
            
            for i in indices:
                # Try to increase radius
                old_r = current_circles[i][2]
                max_radius = self.compute_max_radius(
                    current_circles[i][0], current_circles[i][1], 
                    np.vstack([current_circles[:i], current_circles[i+1:]])
                )
                
                if max_radius > old_r + 1e-6:
                    current_circles[i][2] = max_radius
                    improved = True
                    
                # Try moving circle to improve configuration
                old_x, old_y = current_circles[i][0], current_circles[i][1]
                step_size = 0.1
                
                # Try several positions
                best_pos = [old_x, old_y, old_r]
                best_radius = old_r
                
                # Grid search around current position
                for dx in [-step_size, -step_size/2, 0, step_size/2, step_size]:
                    for dy in [-step_size, -step_size/2, 0, step_size/2, step_size]:
                        new_x = old_x + dx
                        new_y = old_y + dy
                        
                        # Ensure within bounds
                        if 0.01 <= new_x <= self.width - 0.01 and 0.01 <= new_y <= self.height - 0.01:
                            # Compute max radius at new position
                            max_radius = self.compute_max_radius(
                                new_x, new_y,
                                np.vstack([current_circles[:i], current_circles[i+1:]])
                            )
                            
                            if max_radius > best_radius + 1e-6:
                                best_radius = max_radius
                                best_pos = [new_x, new_y, max_radius]
                                
                if best_pos[2] > current_circles[i][2] + 1e-6:
                    current_circles[i] = best_pos
                    improved = True
                    
        return current_circles
    
    def fine_grained_optimization(self, circles: np.ndarray) -> np.ndarray:
        """Fine-grained local optimization with smaller steps."""
        current_circles = circles.copy()
        improved = True
        iteration = 0
        
        while improved and iteration < 500:
            improved = False
            iteration += 1
            
            # Shuffle circle indices for better exploration
            indices = list(range(len(current_circles)))
            random.shuffle(indices)
            
            for i in indices:
                # Try to increase radius with more precise method
                old_r = current_circles[i][2]
                max_radius = self.compute_max_radius(
                    current_circles[i][0], current_circles[i][1],
                    np.vstack([current_circles[:i], current_circles[i+1:]])
                )
                
                if max_radius > old_r + 1e-6:
                    current_circles[i][2] = max_radius
                    improved = True
                
                # Try more refined position adjustments
                old_x, old_y = current_circles[i][0], current_circles[i][1]
                step_size = 0.05
                
                # Try several nearby positions
                best_pos = [old_x, old_y, old_r]
                best_radius = old_r
                
                # Grid search around current position with smaller steps
                for dx in [-step_size, -step_size/2, 0, step_size/2, step_size]:
                    for dy in [-step_size, -step_size/2, 0, step_size/2, step_size]:
                        new_x = old_x + dx
                        new_y = old_y + dy
                        
                        # Ensure within bounds
                        if 0.01 <= new_x <= self.width - 0.01 and 0.01 <= new_y <= self.height - 0.01:
                            # Compute max radius at new position
                            max_radius = self.compute_max_radius(
                                new_x, new_y,
                                np.vstack([current_circles[:i], current_circles[i+1:]])
                            )
                            
                            if max_radius > best_radius + 1e-6:
                                best_radius = max_radius
                                best_pos = [new_x, new_y, max_radius]
                                
                if best_pos[2] > current_circles[i][2] + 1e-6:
                    current_circles[i] = best_pos
                    improved = True
                    
        return current_circles
    
    def refinement_phase(self, circles: np.ndarray) -> np.ndarray:
        """Final refinement phase with very small steps and aggressive boundary handling."""
        current_circles = circles.copy()
        
        # Final intensive refinement
        for iteration in range(300):
            improved = False
            
            # Shuffle circle indices
            indices = list(range(len(current_circles)))
            random.shuffle(indices)
            
            for i in indices:
                # Try small random perturbations
                old_x, old_y, old_r = current_circles[i]
                
                # Perturb position
                dx = random.uniform(-0.02, 0.02)
                dy = random.uniform(-0.02, 0.02)
                new_x = old_x + dx
                new_y = old_y + dy
                
                # Clamp to valid range
                new_x = max(0.01, min(self.width - 0.01, new_x))
                new_y = max(0.01, min(self.height - 0.01, new_y))
                
                # Recalculate maximum radius at new position
                max_radius = self.compute_max_radius(
                    new_x, new_y,
                    np.vstack([current_circles[:i], current_circles[i+1:]])
                )
                
                # Update if improvement
                if max_radius > current_circles[i][2] + 1e-6:
                    current_circles[i] = [new_x, new_y, max_radius]
                    improved = True
            
            if not improved:
                break
        
        # Final boundary adjustment
        for i in range(len(current_circles)):
            x, y, r = current_circles[i]
            # Ensure circle stays within bounds with safety margin
            r = min(r, x - 0.01, self.width - x - 0.01, y - 0.01, self.height - y - 0.01)
            r = max(r, 0.001)
            current_circles[i] = [x, y, r]
            
        return current_circles
    
    def validate_solution(self, circles: np.ndarray) -> bool:
        """Validate that the solution satisfies all constraints."""
        # Check boundary constraints
        for circle in circles:
            x, y, r = circle
            if x - r < 0 or x + r > self.width or y - r < 0 or y + r > self.height:
                return False
        
        # Check overlap constraints
        for i in range(len(circles)):
            for j in range(i + 1, len(circles)):
                ci = circles[i]
                cj = circles[j]
                if ci[2] > 0 and cj[2] > 0:
                    dist = np.sqrt((ci[0] - cj[0])**2 + (ci[1] - cj[1])**2)
                    if dist < ci[2] + cj[2]:
                        return False
        
        return True

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    # Optimize width/height ratio - 3:1 ratio often works well for circle packing
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Create packer instance
    packer = CirclePacker(n_circles=21, perimeter=4.0)
    
    # Initialize layout
    circles = packer.initialize_layout()
    
    # Phase 1: Coarse global optimization
    circles = packer.coarse_optimization(circles)
    
    # Phase 2: Fine-grained local optimization
    circles = packer.fine_grained_optimization(circles)
    
    # Phase 3: Refinement and boundary optimization
    circles = packer.refinement_phase(circles)
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
