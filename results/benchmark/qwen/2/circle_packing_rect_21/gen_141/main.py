# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import KDTree
import random
import time
from typing import Tuple, Optional

class CirclePackingOptimizer:
    def __init__(self, width: float = 1.2, height: float = 0.8, n_circles: int = 21):
        self.width = width
        self.height = height
        self.n_circles = n_circles
        self.rect_width = width
        self.rect_height = height
        random.seed(42)
        np.random.seed(42)
        
    def _generate_hexagonal_grid(self) -> np.ndarray:
        """Generate initial circle placement using hexagonal grid pattern"""
        circles = []
        
        # Use approximately square grid with hexagonal offset
        rows = int(np.ceil(np.sqrt(self.n_circles)))
        cols = int(np.ceil(self.n_circles / rows))
        
        # Calculate spacing based on container dimensions
        margin = 0.1
        usable_width = self.width - 2 * margin
        usable_height = self.height - 2 * margin
        
        spacing_x = usable_width / (cols - 0.5) if cols > 1 else usable_width
        spacing_y = usable_height / (rows - 0.5) if rows > 1 else usable_height
        
        # Adjust for hexagonal packing
        spacing_x = min(spacing_x, spacing_y * 1.15)
        spacing_y = spacing_x * np.sqrt(3) / 2
        
        # Ensure reasonable spacing
        spacing_x = max(0.02, min(0.1, spacing_x))
        spacing_y = max(0.02, min(0.1, spacing_y))
        
        # Generate grid with hexagonal offset
        circle_count = 0
        y_offset = margin + spacing_y/2
        x_offset = margin + spacing_x/2
        
        for i in range(rows):
            y = y_offset + i * spacing_y
            x_start = x_offset + (i % 2) * spacing_x / 2
            for j in range(cols):
                if circle_count >= self.n_circles:
                    break
                x = x_start + j * spacing_x
                if x < self.width - margin and y < self.height - margin:
                    r = min(0.04, spacing_x / 4, spacing_y / 4)
                    circles.append([x, y, r])
                    circle_count += 1
            if circle_count >= self.n_circles:
                break
        
        # Fill remaining slots with random placement
        while len(circles) < self.n_circles:
            x = np.random.uniform(margin, self.width - margin)
            y = np.random.uniform(margin, self.height - margin)
            r = np.random.uniform(0.01, min(0.05, spacing_x/6, spacing_y/6))
            circles.append([x, y, r])
            
        return np.array(circles)
    
    def _calculate_fitness(self, circles: np.ndarray) -> float:
        """Calculate fitness with constraint penalties"""
        total_radius = np.sum(circles[:, 2])
        penalty = 0
        
        # Boundary penalties
        for i in range(self.n_circles):
            cx, cy, r = circles[i]
            if cx - r < 0.01:
                penalty += 100000 * (r - cx)**2
            if cx + r > self.rect_width - 0.01:
                penalty += 100000 * (cx + r - self.rect_width)**2
            if cy - r < 0.01:
                penalty += 100000 * (r - cy)**2
            if cy + r > self.rect_height - 0.01:
                penalty += 100000 * (cy + r - self.rect_height)**2
        
        # Overlap penalties using spatial indexing
        points = circles[:, :2]
        tree = KDTree(points)
        
        for i in range(self.n_circles):
            cx, cy, r = circles[i]
            neighbor_indices = tree.query_ball_point([cx, cy], 2*(r + 0.01) + 0.001)
            
            for j in neighbor_indices:
                if i != j:
                    other_cx, other_cy, other_r = circles[j]
                    dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                    overlap = (r + other_r) - dist
                    
                    if overlap > 0:
                        penalty += 500000 * overlap**2
        
        return total_radius - penalty
    
    def _validate_constraints(self, circles: np.ndarray) -> bool:
        """Validate all constraints are satisfied"""
        points = circles[:, :2]
        tree = KDTree(points)
        
        # Check boundary constraints
        for i in range(self.n_circles):
            cx, cy, r = circles[i]
            if cx - r < 0.01 or cx + r > self.rect_width - 0.01 or \
               cy - r < 0.01 or cy + r > self.rect_height - 0.01:
                return False
                
        # Check overlap constraints
        for i in range(self.n_circles):
            cx, cy, r = circles[i]
            neighbor_indices = tree.query_ball_point([cx, cy], 2*(r + 0.01) + 0.001)
            for j in neighbor_indices:
                if i != j:
                    other_cx, other_cy, other_r = circles[j]
                    dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                    if dist < r + other_r:
                        return False
                        
        return True
    
    def _refine_radii(self, circles: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Aggressive radius refinement with multiple strategies"""
        best_circles = circles.copy()
        best_fitness = self._calculate_fitness(best_circles)
        
        improvement_history = []
        
        for iteration in range(max_iter):
            improved = False
            
            # Process circles in random order for unbiased optimization
            indices = list(range(self.n_circles))
            random.shuffle(indices)
            
            # Try to increase radii of circles that have room
            for i in indices:
                cx, cy, r = best_circles[i]
                
                # Compute maximum allowable radius
                max_radius = float('inf')
                max_radius = min(max_radius, cx - 0.01)
                max_radius = min(max_radius, self.rect_width - cx - 0.01)
                max_radius = min(max_radius, cy - 0.01)
                max_radius = min(max_radius, self.rect_height - cy - 0.01)
                
                # Check overlap constraints
                points = best_circles[:, :2]
                tree = KDTree(points)
                neighbor_indices = tree.query_ball_point([cx, cy], 2*(r + 0.01) + 0.001)
                
                for j in neighbor_indices:
                    if i != j:
                        other_cx, other_cy, other_r = best_circles[j]
                        dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                        max_radius = min(max_radius, dist - other_r - 0.001)
                
                # Try various increments for aggressive improvement
                if max_radius > r and max_radius > 0.001:
                    increments = [0.005, 0.01, 0.015, 0.02]
                    for incr in increments:
                        new_r = min(r + incr, max_radius)
                        if new_r <= r + 0.001:
                            continue
                            
                        # Validate with spatial indexing
                        temp_circles = best_circles.copy()
                        temp_circles[i, 2] = new_r
                        
                        valid = True
                        temp_points = temp_circles[:, :2]
                        temp_tree = KDTree(temp_points)
                        temp_neighbor_indices = temp_tree.query_ball_point([cx, cy], 2*(new_r + 0.01) + 0.001)
                        
                        for k in temp_neighbor_indices:
                            if k != i:
                                other_cx, other_cy, other_r = temp_circles[k]
                                dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                                if dist < new_r + other_r:
                                    valid = False
                                    break
                        
                        if valid:
                            test_circles = best_circles.copy()
                            test_circles[i, 2] = new_r
                            test_fitness = self._calculate_fitness(test_circles)
                            
                            if test_fitness > best_fitness:
                                best_fitness = test_fitness
                                best_circles = test_circles
                                improved = True
                                break
            
            # Occasionally try position perturbations
            if not improved and iteration % 3 == 0:
                for _ in range(3):
                    i = random.randint(0, self.n_circles - 1)
                    x_old, y_old, r = best_circles[i]
                    
                    # Scale perturbation based on iteration
                    scale = max(0.01, 0.05 - iteration * 0.001)
                    dx = np.random.uniform(-scale, scale)
                    dy = np.random.uniform(-scale, scale)
                    
                    new_x = x_old + dx
                    new_y = y_old + dy
                    
                    # Check bounds
                    if (0.01 + r <= new_x <= self.rect_width - 0.01 - r and
                        0.01 + r <= new_y <= self.rect_height - 0.01 - r):
                        
                        temp_circles = best_circles.copy()
                        temp_circles[i, 0] = new_x
                        temp_circles[i, 1] = new_y
                        
                        # Validate overlap
                        valid = True
                        temp_points = temp_circles[:, :2]
                        temp_tree = KDTree(temp_points)
                        temp_neighbor_indices = temp_tree.query_ball_point([new_x, new_y], 2*(r + 0.01) + 0.001)
                        
                        for k in temp_neighbor_indices:
                            if k != i:
                                other_cx, other_cy, other_r = temp_circles[k]
                                dist = np.sqrt((new_x - other_cx)**2 + (new_y - other_cy)**2)
                                if dist < r + other_r:
                                    valid = False
                                    break
                        
                        if valid:
                            test_circles = best_circles.copy()
                            test_circles[i, 0] = new_x
                            test_circles[i, 1] = new_y
                            test_fitness = self._calculate_fitness(test_circles)
                            
                            if test_fitness > best_fitness:
                                best_fitness = test_fitness
                                best_circles = test_circles
                                improved = True
            
            # Track improvement
            improvement_history.append(best_fitness)
            if len(improvement_history) > 10:
                improvement_history.pop(0)
            
            # Early stopping if no significant improvement
            if len(improvement_history) >= 10:
                recent_improvements = [
                    improvement_history[i] - improvement_history[i-1]
                    for i in range(1, len(improvement_history))
                ]
                if all(improvement < 1e-6 for improvement in recent_improvements):
                    break
                    
        return best_circles
    
    def _multi_start_optimization(self, initial_solution: np.ndarray) -> np.ndarray:
        """Run multiple optimization starts to avoid local minima"""
        best_solution = initial_solution.copy()
        best_fitness = self._calculate_fitness(best_solution)
        
        # Multiple restarts
        for _ in range(5):
            # Generate new random initialization
            random_solution = self._generate_hexagonal_grid()
            refined_solution = self._refine_radii(random_solution, max_iter=50)
            refined_fitness = self._calculate_fitness(refined_solution)
            
            if refined_fitness > best_fitness:
                best_fitness = refined_fitness
                best_solution = refined_solution.copy()
                
        return best_solution

    def optimize(self) -> np.ndarray:
        """Main optimization pipeline"""
        # Phase 1: Initial placement
        initial_circles = self._generate_hexagonal_grid()
        
        # Phase 2: Multi-start optimization
        improved_initial = self._multi_start_optimization(initial_circles)
        
        # Phase 3: Enhanced refinement
        refined_solution = self._refine_radii(improved_initial, max_iter=100)
        
        # Final validation
        if not self._validate_constraints(refined_solution):
            refined_solution = self._refine_radii(refined_solution, max_iter=30)
            
        return refined_solution

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    optimizer = CirclePackingOptimizer()
    return optimizer.optimize()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")