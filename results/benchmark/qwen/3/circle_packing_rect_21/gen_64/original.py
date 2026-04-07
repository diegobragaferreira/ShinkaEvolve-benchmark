# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
import random
from typing import Tuple, Optional

class CirclePackingOptimizer:
    def __init__(self, rect_width: float = 1.0, rect_height: float = 1.0, n_circles: int = 21):
        self.rect_width = rect_width
        self.rect_height = rect_height
        self.n_circles = n_circles
        self.circles = np.zeros((n_circles, 3))
        
    def initialize_circles(self) -> None:
        """Initialize circles using a hexagonal packing pattern with randomization"""
        rows = 4
        cols = 6
        
        x_spacing = self.rect_width / (cols + 1)
        y_spacing = self.rect_height / (rows + 1)
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= self.n_circles:
                    break
                x = (j + 1) * x_spacing
                y = (i + 1) * y_spacing
                if i % 2 == 1:
                    x += x_spacing * 0.5
                # Add slight randomization
                x += random.uniform(-0.005, 0.005)
                y += random.uniform(-0.005, 0.005)
                r = 0.02
                self.circles[idx] = [x, y, r]
                idx += 1
    
    def calculate_overlap_penalty(self, circles: np.ndarray, index: int) -> float:
        """Calculate overlap penalty for a specific circle"""
        x, y, r = circles[index]
        penalty = 0.0
        
        for i in range(len(circles)):
            if i != index:
                x2, y2, r2 = circles[i]
                distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                # Penalty increases quadratically with overlap depth
                if distance < r + r2:
                    overlap = (r + r2) - distance
                    penalty += overlap ** 2
                    
        return penalty
    
    def compute_objective(self, circles: np.ndarray, index: int) -> Tuple[float, float]:
        """
        Compute objective value and overlap penalty for a circle
        Returns: (objective_value, overlap_penalty)
        """
        x, y, r = circles[index]
        
        # Base radius contribution (negative because we want to maximize)
        base_objective = -r
        
        # Overlap penalty (positive because we want to minimize)
        overlap_penalty = self.calculate_overlap_penalty(circles, index)
        
        # Combined objective (weighted sum)
        # Weighting favors radius maximization but penalizes overlap
        total_objective = base_objective + 0.1 * overlap_penalty
        
        return total_objective, overlap_penalty
    
    def compute_max_radius(self, circles: np.ndarray, index: int) -> float:
        """Compute the maximum possible radius for a circle at given index without violating constraints"""
        x, y, _ = circles[index]

        # Minimum distance to boundaries
        min_dist_to_boundaries = min(x, self.rect_width - x, y, self.rect_height - y)

        # Check collisions with other circles
        min_dist_to_others = float('inf')
        for i in range(len(circles)):
            if i != index:
                x2, y2, r2 = circles[i]
                distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                # Distance should be at least the sum of radii for non-overlap
                if distance > 0:  # Avoid division by zero
                    min_dist_to_others = min(min_dist_to_others, distance - r2)

        # Return the minimum of all constraints, with safety margin
        max_radius = min(min_dist_to_boundaries, min_dist_to_others)
        
        return max(0.001, max_radius)
    
    def validate_configuration(self, circles: np.ndarray) -> bool:
        """Validate that all circles are within bounds and non-overlapping"""
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Check boundary conditions
            if x - r < 0 or x + r > self.rect_width or y - r < 0 or y + r > self.rect_height:
                return False

            # Check overlap with other circles
            for j in range(i + 1, len(circles)):
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                # Overlap occurs when distance < sum of radii
                if distance < r + r2 - 1e-8:
                    return False

        return True
    
    def local_search_step(self, circles: np.ndarray, step_size: float = 0.02) -> Tuple[np.ndarray, bool]:
        """Perform one local search step"""
        improved = False
        new_circles = circles.copy()
        
        # Shuffle circle order to avoid bias
        circle_indices = list(range(self.n_circles))
        random.shuffle(circle_indices)
        
        for i in circle_indices:
            # Compute current objective and maximum possible radius
            current_obj, _ = self.compute_objective(new_circles, i)
            max_radius = self.compute_max_radius(new_circles, i)
            
            # Try to increase radius
            if max_radius > new_circles[i, 2]:
                # Check if this change improves the objective
                old_radius = new_circles[i, 2]
                new_circles[i, 2] = max_radius
                
                # Calculate new objective
                new_obj, _ = self.compute_objective(new_circles, i)
                
                # If improvement, keep it; otherwise revert
                if new_obj < current_obj:  # Lower objective is better
                    improved = True
                else:
                    new_circles[i, 2] = old_radius  # Revert
                    
        return new_circles, improved
    
    def perturb_positions(self, circles: np.ndarray, step_size: float = 0.02) -> np.ndarray:
        """Apply random perturbations to positions to escape local minima"""
        new_circles = circles.copy()
        for _ in range(3):
            i = random.randint(0, self.n_circles - 1)
            # Apply perturbation to position
            new_circles[i, 0] += random.uniform(-step_size, step_size)
            new_circles[i, 1] += random.uniform(-step_size, step_size)

            # Clamp to rectangle bounds
            new_circles[i, 0] = np.clip(new_circles[i, 0], 0.01, self.rect_width - 0.01)
            new_circles[i, 1] = np.clip(new_circles[i, 1], 0.01, self.rect_height - 0.01)

            # Recompute max radius after perturbation
            max_radius = self.compute_max_radius(new_circles, i)
            new_circles[i, 2] = max_radius
            
        return new_circles
    
    def optimize(self) -> np.ndarray:
        """Main optimization loop"""
        # Initialize with a good starting point
        self.initialize_circles()
        
        best_circles = None
        best_radius_sum = 0
        
        # Multi-start approach
        for start_iter in range(5):
            current_circles = self.circles.copy()
            
            # Multi-scale optimization
            for phase in range(3):
                if phase == 0:
                    max_iterations = 100
                    step_size = 0.05
                elif phase == 1:
                    max_iterations = 150
                    step_size = 0.02
                else:
                    max_iterations = 200
                    step_size = 0.005

                for iteration in range(max_iterations):
                    # Perform local search step
                    current_circles, improved = self.local_search_step(current_circles, step_size)
                    
                    # Periodic perturbations to escape local minima
                    if iteration % 20 == 0 and iteration > 0:
                        current_circles = self.perturb_positions(current_circles, step_size)
                    
                    # Early stopping if no improvement
                    if not improved and iteration > 30:
                        break
                        
            # Validate and score configuration
            if self.validate_configuration(current_circles):
                radius_sum = np.sum(current_circles[:, 2])
                if radius_sum > best_radius_sum:
                    best_radius_sum = radius_sum
                    best_circles = current_circles.copy()
        
        # Fallback if no valid configuration found
        if best_circles is None:
            current_circles = self.circles.copy()
            for _ in range(300):
                current_circles, _ = self.local_search_step(current_circles)
            best_circles = current_circles
            
        # Final boundary correction
        for i in range(self.n_circles):
            x, y, r = best_circles[i]
            # Ensure circles are within bounds and radius is reasonable
            r = min(r, x, self.rect_width - x, y, self.rect_height - y)
            if r <= 0.001:
                r = 0.01
            best_circles[i] = [x, y, r]

        return best_circles

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions - since perimeter = 4, width + height = 2
    optimizer = CirclePackingOptimizer()
    return optimizer.optimize()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")