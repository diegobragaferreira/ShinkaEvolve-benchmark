# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import random
from typing import Tuple, List
import time
import warnings
import math

# Global constants
RECT_PERIMETER = 4.0
RECT_WIDTH = 1.0
RECT_HEIGHT = 1.0
NUM_CIRCLES = 21
SEED = 42

class CirclePacker:
    def __init__(self, width: float = RECT_WIDTH, height: float = RECT_HEIGHT,
                 num_circles: int = NUM_CIRCLES):
        self.width = width
        self.height = height
        self.num_circles = num_circles
        self.rect_area = width * height

        # Initialize random seed for reproducibility
        np.random.seed(SEED)
        random.seed(SEED)

    def is_valid_position(self, x: float, y: float, r: float) -> bool:
        """Check if circle center is within bounds"""
        return (r <= x <= self.width - r and
                r <= y <= self.height - r)

    def is_valid_circle(self, x: float, y: float, r: float) -> bool:
        """Check if circle is valid (within bounds and positive radius)"""
        return (0 < r and
                self.is_valid_position(x, y, r))

    def check_overlap(self, circles: np.ndarray, idx1: int, idx2: int) -> bool:
        """Check if two circles overlap using Euclidean distance"""
        x1, y1, r1 = circles[idx1]
        x2, y2, r2 = circles[idx2]

        # Calculate squared distance to avoid sqrt computation
        dx = x1 - x2
        dy = y1 - y2
        dist_sq = dx*dx + dy*dy
        radius_sum = r1 + r2
        return dist_sq < radius_sum * radius_sum

    def calculate_total_radius_sum(self, circles: np.ndarray) -> float:
        """Calculate sum of all circle radii"""
        return np.sum(circles[:, 2])

    def calculate_fitness(self, circles: np.ndarray) -> Tuple[float, int]:
        """
        Calculate fitness: sum of radii with penalty for constraint violations

        Returns:
            Tuple of (fitness_score, number_of_violations)
        """
        total_radius = self.calculate_total_radius_sum(circles)

        # Count constraint violations
        violations = 0

        # Check boundary violations
        for i in range(self.num_circles):
            x, y, r = circles[i]
            if not self.is_valid_circle(x, y, r):
                violations += 100  # Heavy penalty for boundary violations

        # Check overlap violations using spatial indexing for efficiency
        try:
            # Build KDTree for fast neighbor search
            points = circles[:, :2]  # Only x,y coordinates
            tree = cKDTree(points)

            # Find neighbors within 2*max_radius distance (optimization)
            max_radius = np.max(circles[:, 2])
            if max_radius > 0:
                # Query pairs with distance threshold
                pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')

                for i, j in pairs:
                    if self.check_overlap(circles, i, j):
                        violations += 1

        except Exception as e:
            warnings.warn(f"Error in overlap checking: {e}")
            # Fallback to brute force when spatial indexing fails
            for i in range(self.num_circles):
                for j in range(i+1, self.num_circles):
                    if self.check_overlap(circles, i, j):
                        violations += 1

        # Return fitness score - higher is better
        # Adaptive penalty weight based on solution quality
        penalty_weight = 500.0 + 100.0 * max(0, total_radius - 1.0)
        return total_radius - (penalty_weight * violations), violations

    def generate_voronoi_grid_initialization(self) -> np.ndarray:
        """Create initial configuration using a Voronoi-inspired grid placement"""
        circles = np.zeros((self.num_circles, 3))
        
        # Calculate target area per circle based on rectangle area and number of circles
        target_area_per_circle = self.rect_area / self.num_circles
        
        # Estimate approximate radius based on circle area
        estimated_radius = np.sqrt(target_area_per_circle / np.pi)
        
        # Use a grid that adapts to the aspect ratio of the container
        aspect_ratio = self.width / self.height
        
        # Choose grid dimensions based on aspect ratio
        if aspect_ratio >= 1.0:  # Wide rectangle
            cols = int(np.ceil(np.sqrt(self.num_circles * aspect_ratio)))
            rows = int(np.ceil(self.num_circles / cols))
        else:  # Tall rectangle  
            rows = int(np.ceil(np.sqrt(self.num_circles / aspect_ratio)))
            cols = int(np.ceil(self.num_circles / rows))
            
        # Ensure we have enough grid points
        if cols * rows < self.num_circles:
            cols = max(cols, 1)
            rows = int(np.ceil(self.num_circles / cols))
            
        # Calculate spacing
        margin = 0.05
        cell_width = (self.width - 2 * margin) / cols
        cell_height = (self.height - 2 * margin) / rows
        
        # Place circles with Voronoi-like distribution
        circle_idx = 0
        for i in range(rows):
            for j in range(cols):
                if circle_idx >= self.num_circles:
                    break

                # Add some jitter to create more natural distribution
                jitter_x = np.random.uniform(-0.2 * cell_width, 0.2 * cell_width)
                jitter_y = np.random.uniform(-0.2 * cell_height, 0.2 * cell_height)
                
                x = margin + (j + 0.5) * cell_width + jitter_x
                y = margin + (i + 0.5) * cell_height + jitter_y
                
                # Adjust radius based on local density: decrease for dense areas
                radius_factor = 0.8 + np.random.uniform(0, 0.4)
                r = estimated_radius * radius_factor
                
                # Ensure circle fits within bounds
                x = np.clip(x, r, self.width - r)
                y = np.clip(y, r, self.height - r)

                circles[circle_idx] = [x, y, r]
                circle_idx += 1

            if circle_idx >= self.num_circles:
                break
                
        return circles

    def local_improvement_step(self, circles: np.ndarray, max_attempts: int = 50) -> np.ndarray:
        """Attempt to improve the solution by locally adjusting circle radii"""
        improved_circles = circles.copy()
        
        # Try to increase radii while maintaining constraints
        for attempt in range(max_attempts):
            # Select a random circle to potentially increase
            circle_idx = np.random.randint(0, self.num_circles)
            x, y, r = improved_circles[circle_idx]
            
            # Try to increase radius up to a reasonable limit
            new_r = min(r * 1.05, 0.3)  # 5% increase with ceiling at 0.3
            
            if new_r <= r:
                continue  # No improvement possible
                
            # Temporarily update the circle
            old_circle = improved_circles[circle_idx].copy()
            improved_circles[circle_idx] = [x, y, new_r]
            
            # Check if it's still valid and doesn't cause overlaps
            valid = self.is_valid_circle(x, y, new_r)
            
            if valid:
                # Check overlaps with all other circles
                has_overlap = False
                for i in range(self.num_circles):
                    if i != circle_idx:
                        temp_circles = np.vstack([improved_circles[:circle_idx], 
                                                 [x, y, new_r], 
                                                 improved_circles[circle_idx+1:]])
                        if self.check_overlap(temp_circles, circle_idx, i):
                            has_overlap = True
                            break
                
                if not has_overlap:
                    # Accept the change
                    continue
                else:
                    # Revert back if it causes overlap
                    improved_circles[circle_idx] = old_circle
            else:
                # Revert back if invalid
                improved_circles[circle_idx] = old_circle
                
        return improved_circles

    def constrained_local_search(self, circles: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Use constrained local optimization to refine the solution"""
        refined = circles.copy()
        
        # Simple iterative local search improvement
        for iteration in range(max_iter):
            # Try to improve radii
            improved = False
            
            # Try increasing each circle radius individually
            for i in range(self.num_circles):
                x, y, r = refined[i]
                
                # Try to increase radius slightly
                new_r = min(r * 1.02, 0.3)  # Small 2% increase with ceiling
                
                if new_r <= r:
                    continue
                    
                # Check if new radius is valid and doesn't cause overlaps
                if not self.is_valid_circle(x, y, new_r):
                    continue
                    
                # Check for overlaps with other circles
                overlap_found = False
                for j in range(self.num_circles):
                    if i != j:
                        temp_arr = np.vstack([refined[:i], [x, y, new_r], refined[i+1:]])
                        if self.check_overlap(temp_arr, i, j):
                            overlap_found = True
                            break
                
                if not overlap_found:
                    refined[i] = [x, y, new_r]
                    improved = True
                    
            # If no improvements were made, break early
            if not improved:
                break
                
        return refined

    def optimize(self) -> np.ndarray:
        """Main optimization using grid-based initialization and local refinement"""
        start_time = time.time()
        
        # Start with grid-based initialization
        circles = self.generate_voronoi_grid_initialization()
        
        # Apply local refinement and improvement 
        circles = self.constrained_local_search(circles, 100)
        
        # Continue with iterative improvement
        best_solution = circles.copy()
        best_fitness, _ = self.calculate_fitness(best_solution)
        
        # Try several local improvement attempts
        for attempt in range(50):  # Limited attempts to prevent infinite loops
            # Local improvement step
            improved = self.local_improvement_step(best_solution)
            
            # Evaluate new solution
            new_fitness, _ = self.calculate_fitness(improved)
            
            # Accept better solutions
            if new_fitness > best_fitness:
                best_solution = improved.copy()
                best_fitness = new_fitness
                # Occasionally print progress
                if attempt % 10 == 0:
                    print(f"Improvement attempt {attempt}: fitness = {best_fitness:.6f}")
        
        # Final local search
        final_solution = self.constrained_local_search(best_solution, 100)
        
        end_time = time.time()
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")
        print(f"Final fitness achieved: {self.calculate_fitness(final_solution)[0]:.6f}")

        return final_solution

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Create packer instance with rectangle dimensions
    packer = CirclePacker(width=1.0, height=1.0, num_circles=21)

    # Run optimization
    circles = packer.optimize()

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")