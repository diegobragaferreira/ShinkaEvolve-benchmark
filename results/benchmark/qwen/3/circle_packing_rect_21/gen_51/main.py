# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import math
import random
from typing import Tuple, List

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Rectangle dimensions: width + height = 2, using 1.5 x 0.5 for good aspect ratio
    rect_width, rect_height = 1.5, 0.5
    
    def get_hexagonal_grid_points(width: float, height: float, n_points: int) -> List[Tuple[float, float]]:
        """Generate hexagonal grid points for initial placement"""
        # Hexagon packing parameters
        side_length = 0.1  # Initial guess for side length
        hex_width = side_length * 2
        hex_height = side_length * math.sqrt(3)
        
        # Calculate grid dimensions
        cols = max(1, int(width / hex_width) + 2)
        rows = max(1, int(height / hex_height) + 2)
        
        # Generate hexagonal grid points
        points = []
        offset = 0.0
        
        for row in range(rows):
            for col in range(cols):
                x = (col * hex_width) + (offset if row % 2 == 1 else 0) + side_length
                y = row * hex_height + side_length
                
                # Only add if within bounds
                if 0 <= x <= width and 0 <= y <= height:
                    points.append((x, y))
                    
            offset = (offset + hex_width) % hex_width
            
        # If we have too few points, expand
        while len(points) < n_points:
            # Add random points to ensure enough
            for _ in range(n_points - len(points)):
                x = random.uniform(side_length, width - side_length)
                y = random.uniform(side_length, height - side_length)
                points.append((x, y))
                
        return points[:n_points]
    
    def compute_max_radius_at_position(x: float, y: float, existing_circles: np.ndarray, 
                                     rect_width: float, rect_height: float) -> float:
        """Compute maximum radius for a circle at given position"""
        # Boundary constraints
        min_bound = min(x, rect_width - x, y, rect_height - y)
        
        # Circle-to-circle constraints
        min_circle_dist = float('inf')
        for i in range(len(existing_circles)):
            ex, ey, er = existing_circles[i]
            dist = math.sqrt((x - ex)**2 + (y - ey)**2)
            min_circle_dist = min(min_circle_dist, dist - er)
            
        # Return minimum of all constraints
        max_radius = min(min_bound, min_circle_dist)
        return max(0.001, max_radius)
    
    def validate_and_correct(circles: np.ndarray, rect_width: float, rect_height: float) -> np.ndarray:
        """Ensure all circles are valid (within bounds, non-overlapping)"""
        corrected = circles.copy()
        
        # Apply boundary corrections
        for i in range(len(corrected)):
            x, y, r = corrected[i]
            corrected[i] = [max(r, min(rect_width - r, x)), 
                           max(r, min(rect_height - r, y)), r]
        
        # Handle overlaps through iterative correction
        for _ in range(50):  # Limited iterations to prevent infinite loops
            changed = False
            for i in range(len(corrected)):
                x, y, r = corrected[i]
                
                # Check for overlap
                for j in range(len(corrected)):
                    if i != j:
                        x2, y2, r2 = corrected[j]
                        dist = math.sqrt((x - x2)**2 + (y - y2)**2)
                        
                        if dist < (r + r2):
                            # Move away from overlapping circle
                            dx = x2 - x
                            dy = y2 - y
                            dist_total = math.sqrt(dx*dx + dy*dy)
                            
                            if dist_total > 0.001:
                                # Normalize direction
                                dx /= dist_total
                                dy /= dist_total
                                
                                # Move outward along separation vector
                                new_x = x - dx * 0.001
                                new_y = y - dy * 0.001
                                new_x = max(r, min(rect_width - r, new_x))
                                new_y = max(r, min(rect_height - r, new_y))
                                
                                corrected[i] = [new_x, new_y, r]
                                changed = True
                                break
            
            if not changed:
                break
                
        return corrected
    
    def compute_fitness(circles: np.ndarray) -> float:
        """Compute fitness as sum of radii (higher is better)"""
        return np.sum(circles[:, 2])
    
    def local_improve(circles: np.ndarray, rect_width: float, rect_height: float, 
                     iterations: int = 100) -> np.ndarray:
        """Local improvement using simulated annealing-like approach"""
        current = circles.copy()
        current_fitness = compute_fitness(current)
        
        for iter_num in range(iterations):
            # Cooling schedule
            temperature = 1.0 - (iter_num / iterations) * 0.9
            
            # Try to improve one circle at a time
            for i in range(len(current)):
                # Save original
                orig_x, orig_y, orig_r = current[i]
                
                # Try small perturbations
                new_x = max(0.001, min(rect_width - 0.001, orig_x + random.uniform(-0.05, 0.05)))
                new_y = max(0.001, min(rect_height - 0.001, orig_y + random.uniform(-0.05, 0.05)))
                
                # Compute new maximum radius
                temp_circles = current.copy()
                temp_circles[i] = [new_x, new_y, 0.01]
                max_radius = compute_max_radius_at_position(new_x, new_y, temp_circles, rect_width, rect_height)
                
                # Try to increase radius if beneficial
                new_r = min(max_radius, orig_r + random.uniform(-0.02, 0.02))
                new_r = max(0.001, new_r)
                
                temp_circles[i] = [new_x, new_y, new_r]
                
                # Validate if valid (not overlapping with others)
                temp_circles = validate_and_correct(temp_circles, rect_width, rect_height)
                new_fitness = compute_fitness(temp_circles)
                
                # Accept or reject based on fitness improvement
                if new_fitness > current_fitness:
                    current = temp_circles
                    current_fitness = new_fitness
                elif random.random() < math.exp((new_fitness - current_fitness) / (temperature + 1e-10)):
                    # Accept worse solution with probability
                    current = temp_circles
                    current_fitness = new_fitness
                    
        return current
    
    # Phase 1: Generate initial hexagonal arrangement
    points = get_hexagonal_grid_points(rect_width, rect_height, 21)
    
    # Initialize circles with base positions
    circles = np.zeros((21, 3))
    for i, (x, y) in enumerate(points):
        circles[i] = [x, y, 0.05]
    
    # Phase 2: Progressive refinement
    # Phase 2a: Global adjustment
    for i in range(21):
        x, y, r = circles[i]
        max_r = compute_max_radius_at_position(x, y, circles, rect_width, rect_height)
        circles[i] = [x, y, min(max_r, 0.2)]
    
    # Phase 2b: Local optimization with multiple rounds
    best_circles = circles.copy()
    best_fitness = compute_fitness(best_circles)
    
    # Multiple rounds of local optimization
    for round_num in range(5):
        improved = local_improve(best_circles, rect_width, rect_height, 300)
        improved_fitness = compute_fitness(improved)
        
        if improved_fitness > best_fitness:
            best_circles = improved
            best_fitness = improved_fitness
            
            # Early stopping if improvement is minimal
            if round_num > 0 and improved_fitness - best_fitness < 0.01:
                break
    
    # Phase 3: Fine-tuning
    final_circles = local_improve(best_circles, rect_width, rect_height, 200)
    final_circles = validate_and_correct(final_circles, rect_width, rect_height)
    
    return final_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")