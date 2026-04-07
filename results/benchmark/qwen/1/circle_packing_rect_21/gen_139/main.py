# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from typing import Tuple, List, Optional
import time
import random

# Set seeds for determinism
random.seed(42)
np.random.seed(42)

class CirclePackingOptimizer:
    def __init__(self, n_circles: int = 21, rect_width: float = 1.0, rect_height: float = 1.0):
        self.n_circles = n_circles
        self.rect_width = rect_width
        self.rect_height = rect_height
        self.best_solution = None
        self.best_score = -float('inf')
        
    def distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
    def is_valid_solution(self, circles: np.ndarray) -> bool:
        """Check if all circles are within bounds and non-overlapping."""
        n = len(circles)
        
        # Check bounds
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        radii = circles[:, 2]
        
        if np.any(x_coords - radii < 0) or np.any(x_coords + radii > self.rect_width) or \
           np.any(y_coords - radii < 0) or np.any(y_coords + radii > self.rect_height):
            return False
        
        # Check overlaps using vectorized approach
        x_diff = x_coords[:, np.newaxis] - x_coords[np.newaxis, :]
        y_diff = y_coords[:, np.newaxis] - y_coords[np.newaxis, :]
        dists = np.sqrt(x_diff**2 + y_diff**2)
        sums = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Set diagonal to infinity to avoid self-comparison
        np.fill_diagonal(dists, np.inf)
        
        # Check if any distances are less than sum of radii
        if np.any(dists < sums):
            return False
            
        return True
        
    def compute_voronoi_density(self, circles: np.ndarray) -> np.ndarray:
        """Compute Voronoi cell areas for each circle to estimate constraint density."""
        # Add boundary points for proper Voronoi calculation
        points = circles[:, :2].copy()
        
        # Add boundary points to make Voronoi more meaningful
        boundary_points = [
            [0, 0], [self.rect_width, 0], [0, self.rect_height], [self.rect_width, self.rect_height],
            [self.rect_width/2, 0], [self.rect_width/2, self.rect_height],
            [0, self.rect_height/2], [self.rect_width, self.rect_height/2]
        ]
        points = np.vstack([points, boundary_points])
        
        try:
            vor = Voronoi(points)
            
            # For each original point, compute Voronoi cell area
            areas = []
            for i in range(len(circles)):
                region_idx = np.where(vor.point_region == i)[0][0] if i in vor.point_region else -1
                
                if region_idx != -1 and region_idx < len(vor.regions):
                    region = vor.regions[region_idx]
                    if -1 not in region and len(region) >= 3:
                        # Compute area of polygon using shoelace formula
                        vertices = np.array([vor.vertices[i] for i in region])
                        if len(vertices) >= 3:
                            x = vertices[:, 0]
                            y = vertices[:, 1]
                            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                            areas.append(area)
                        else:
                            areas.append(1.0)
                    else:
                        areas.append(1.0)
                else:
                    areas.append(1.0)
                    
            return np.array(areas)
        except:
            # Fallback to uniform distribution if Voronoi fails
            return np.ones(len(circles))
            
    def evaluate_fitness(self, circles: np.ndarray) -> float:
        """Evaluate fitness as sum of radii."""
        return np.sum(circles[:, 2])
        
    def generate_initial_pattern(self, pattern_type: str = "hexagonal") -> np.ndarray:
        """Generate initial pattern with different strategies."""
        circles = np.zeros((self.n_circles, 3))
        
        if pattern_type == "hexagonal":
            return self._generate_hexagonal_pattern()
        elif pattern_type == "triangular":
            return self._generate_triangular_pattern()
        elif pattern_type == "square":
            return self._generate_square_pattern()
        else:  # random
            return self._generate_random_pattern()
            
    def _generate_hexagonal_pattern(self) -> np.ndarray:
        """Generate hexagonal pattern."""
        circles = np.zeros((self.n_circles, 3))
        
        # Hexagonal packing parameters
        rows = int(np.sqrt(self.n_circles))
        cols = int(self.n_circles / rows) + 1
        
        spacing_x = self.rect_width / (cols + 1)
        spacing_y = self.rect_height / (rows + 1)
        
        # Adjust spacing to fit better
        min_radius = 0.02
        
        for i in range(self.n_circles):
            row = i // cols
            col = i % cols
            x = (col + 1) * spacing_x
            y = (row + 1) * spacing_y
            
            # Offset every other row for hexagonal arrangement
            if row % 2 == 1:
                x += spacing_x / 2
                
            circles[i] = [x, y, min_radius]
            
        return circles
        
    def _generate_triangular_pattern(self) -> np.ndarray:
        """Generate triangular pattern."""
        circles = np.zeros((self.n_circles, 3))
        
        # Arrange in triangular pattern
        sqrt_n = int(np.ceil(np.sqrt(self.n_circles)))
        spacing_x = self.rect_width / (sqrt_n + 1)
        spacing_y = self.rect_height / (sqrt_n + 1)
        
        idx = 0
        for i in range(sqrt_n):
            for j in range(sqrt_n):
                if idx >= self.n_circles:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                # Slight offset for triangular pattern
                if i % 2 == 1:
                    x += spacing_x / 2
                circles[idx] = [x, y, 0.02]
                idx += 1
            if idx >= self.n_circles:
                break
                
        return circles
        
    def _generate_square_pattern(self) -> np.ndarray:
        """Generate square grid pattern."""
        circles = np.zeros((self.n_circles, 3))
        
        sqrt_n = int(np.ceil(np.sqrt(self.n_circles)))
        spacing_x = self.rect_width / (sqrt_n + 1)
        spacing_y = self.rect_height / (sqrt_n + 1)
        
        idx = 0
        for i in range(sqrt_n):
            for j in range(sqrt_n):
                if idx >= self.n_circles:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                circles[idx] = [x, y, 0.02]
                idx += 1
            if idx >= self.n_circles:
                break
                
        return circles
        
    def _generate_random_pattern(self) -> np.ndarray:
        """Generate random initial pattern."""
        circles = np.zeros((self.n_circles, 3))
        
        # Generate random positions and large initial radii
        for i in range(self.n_circles):
            x = np.random.uniform(0.05, self.rect_width - 0.05)
            y = np.random.uniform(0.05, self.rect_height - 0.05)
            r = 0.05  # Start with larger radius
            circles[i] = [x, y, r]
            
        return circles
        
    def mutate_radius(self, circles: np.ndarray, idx: int, density_scores: np.ndarray = None) -> np.ndarray:
        """Mutate radius with adaptive delta based on Voronoi density."""
        new_circles = circles.copy()
        old_r = new_circles[idx, 2]
        
        # Adaptive delta based on density - more sophisticated approach
        max_delta = 0.01
        if density_scores is not None and len(density_scores) > idx:
            # High density means more constraints, use smaller deltas
            # Normalize density to [0,1] range and invert for delta scaling
            normalized_density = min(1.0, density_scores[idx] / 10.0)  # Cap at 10 for stability
            delta_factor = 1.0 - 0.8 * normalized_density  # Reduce delta by up to 80% in dense regions
            delta = max_delta * delta_factor
        else:
            delta = max_delta
            
        # Apply adaptive delta with better distribution
        if np.random.random() < 0.7:  # 70% chance of small change
            delta_r = np.random.uniform(-delta*0.5, delta*0.5)
        else:  # 30% chance of larger change
            delta_r = np.random.uniform(-delta, delta)
            
        new_r = old_r + delta_r
        
        # Ensure positive radius with better clamping
        new_r = max(0.001, new_r)
        new_circles[idx, 2] = new_r
        
        return new_circles
        
    def mutate_position(self, circles: np.ndarray, idx: int) -> np.ndarray:
        """Mutate position with small random perturbation."""
        new_circles = circles.copy()
        old_x, old_y = new_circles[idx, 0], new_circles[idx, 1]
        
        # Adaptive step size based on proximity to boundary
        x, y, r = new_circles[idx]
        boundary_distance = min(x, self.rect_width - x, y, self.rect_height - y)
        max_delta = min(0.05, boundary_distance * 0.5)
        
        # Small random perturbation with boundary awareness
        delta_x = np.random.uniform(-max_delta, max_delta)
        delta_y = np.random.uniform(-max_delta, max_delta)
        
        new_x = old_x + delta_x
        new_y = old_y + delta_y
        
        # Ensure within bounds with margin
        margin = 0.01
        new_x = np.clip(new_x, margin, self.rect_width - margin)
        new_y = np.clip(new_y, margin, self.rect_height - margin)
        
        new_circles[idx, 0] = new_x
        new_circles[idx, 1] = new_y
        
        return new_circles
        
    def local_optimization_stage1(self, circles: np.ndarray, max_iter: int = 30) -> np.ndarray:
        """First stage of local optimization focusing on fast improvements."""
        current_circles = circles.copy()
        best_circles = current_circles.copy()
        best_fitness = self.evaluate_fitness(current_circles)
        
        # Get Voronoi densities for adaptive mutation
        density_scores = self.compute_voronoi_density(current_circles)
        
        for iteration in range(max_iter):
            improved = False
            
            # Mutate each circle with focus on the most constrained ones
            # Sort by density (high density = more constrained)
            if len(density_scores) > 0:
                sorted_indices = np.argsort(density_scores)[::-1]  # Descending order
                # Only mutate top 70% of circles
                mutate_indices = sorted_indices[:max(1, int(0.7 * len(sorted_indices)))]
            else:
                mutate_indices = range(len(current_circles))
                
            for i in mutate_indices:
                # Try position mutation
                mutated_pos = self.mutate_position(current_circles, i)
                
                # Try radius mutation
                mutated_rad = self.mutate_radius(current_circles, i, density_scores)
                
                # Evaluate both mutations
                pos_fitness = self.evaluate_fitness(mutated_pos)
                rad_fitness = self.evaluate_fitness(mutated_rad)
                
                # Choose the better one
                if pos_fitness > rad_fitness:
                    if self.is_valid_solution(mutated_pos):
                        current_circles = mutated_pos
                        improved = True
                else:
                    if self.is_valid_solution(mutated_rad):
                        current_circles = mutated_rad
                        improved = True
                        
            # Update best solution
            current_fitness = self.evaluate_fitness(current_circles)
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_circles = current_circles.copy()
                
        return best_circles
        
    def local_optimization_stage2(self, circles: np.ndarray, max_iter: int = 50) -> np.ndarray:
        """Second stage of local optimization for fine-tuning."""
        current_circles = circles.copy()
        best_circles = current_circles.copy()
        best_fitness = self.evaluate_fitness(current_circles)
        
        # Get Voronoi densities for adaptive mutation
        density_scores = self.compute_voronoi_density(current_circles)
        
        for iteration in range(max_iter):
            # Mutate each circle systematically
            for i in range(len(current_circles)):
                # Try to improve more frequently on boundary circles
                x, y, r = current_circles[i]
                boundary_close = (x <= r + 0.05 or x >= self.rect_width - r - 0.05 or
                                y <= r + 0.05 or y >= self.rect_height - r - 0.05)
                
                # Apply different mutation frequencies
                if boundary_close or np.random.random() < 0.6:
                    # Try position mutation
                    mutated_pos = self.mutate_position(current_circles, i)
                    
                    # Try radius mutation
                    mutated_rad = self.mutate_radius(current_circles, i, density_scores)
                    
                    # Evaluate both mutations
                    pos_fitness = self.evaluate_fitness(mutated_pos)
                    rad_fitness = self.evaluate_fitness(mutated_rad)
                    
                    # Choose the better one
                    if pos_fitness > rad_fitness:
                        if self.is_valid_solution(mutated_pos):
                            current_circles = mutated_pos
                    else:
                        if self.is_valid_solution(mutated_rad):
                            current_circles = mutated_rad
            
            # Update best solution
            current_fitness = self.evaluate_fitness(current_circles)
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_circles = current_circles.copy()
                
        return best_circles
        
    def multi_start_optimization(self) -> np.ndarray:
        """Run multi-start optimization with different initialization patterns."""
        initial_patterns = [
            ("hexagonal", self.generate_initial_pattern("hexagonal")),
            ("triangular", self.generate_initial_pattern("triangular")),
            ("square", self.generate_initial_pattern("square")),
            ("random", self.generate_initial_pattern("random"))
        ]
        
        for pattern_name, initial_pattern in initial_patterns:
            try:
                # Apply first stage optimization
                stage1_result = self.local_optimization_stage1(initial_pattern, max_iter=30)
                
                # Apply second stage optimization  
                stage2_result = self.local_optimization_stage2(stage1_result, max_iter=40)
                
                # Evaluate final result
                score = self.evaluate_fitness(stage2_result)
                if score > self.best_score and self.is_valid_solution(stage2_result):
                    self.best_score = score
                    self.best_solution = stage2_result.copy()
                    
            except Exception as e:
                # Continue with other patterns even if one fails
                continue
                
        # Fallback to the best found solution so far or random initialization
        if self.best_solution is None:
            random_pattern = self.generate_initial_pattern("random")
            self.best_solution = self.local_optimization_stage2(random_pattern, max_iter=100)
            
        return self.best_solution
        
    def optimize(self) -> np.ndarray:
        """Main optimization loop."""
        start_time = time.time()
        
        # Run multi-start optimization
        final_solution = self.multi_start_optimization()
        
        # Final validation
        if not self.is_valid_solution(final_solution):
            # Try to fix the solution if it's invalid
            final_solution = self._attempt_fix_solution(final_solution)
            
        # Final refinement if needed
        if self.best_score < self.evaluate_fitness(final_solution) * 0.99:
            final_solution = self.local_optimization_stage2(final_solution, max_iter=20)
            
        eval_time = time.time() - start_time
        print(f"Optimization completed in {eval_time:.2f} seconds")
        
        return final_solution
        
    def _attempt_fix_solution(self, circles: np.ndarray) -> np.ndarray:
        """Attempt to fix an invalid solution by adjusting positions."""
        fixed_circles = circles.copy()
        
        # Make sure all circles are within bounds
        for i in range(len(fixed_circles)):
            x, y, r = fixed_circles[i]
            # Clamp to valid range
            x = np.clip(x, r + 0.01, self.rect_width - r - 0.01)
            y = np.clip(y, r + 0.01, self.rect_height - r - 0.01)
            fixed_circles[i] = [x, y, r]
            
        # Try to fix overlaps by moving circles away from each other
        for _ in range(50):  # Limited iterations to prevent infinite loops
            # Try to move circles that are too close together
            moved_any = False
            for i in range(len(fixed_circles)):
                x1, y1, r1 = fixed_circles[i]
                
                # Find overlapping circles
                for j in range(i+1, len(fixed_circles)):
                    x2, y2, r2 = fixed_circles[j]
                    distance = self.distance([x1, y1], [x2, y2])
                    
                    if distance < r1 + r2:
                        # Move circle i away from j
                        dx = x2 - x1
                        dy = y2 - y1
                        dist = np.sqrt(dx*dx + dy*dy)
                        
                        if dist > 0:
                            # Normalize and move by small amount
                            dx = dx / dist * 0.01
                            dy = dy / dist * 0.01
                            
                            new_x1 = x1 - dx
                            new_y1 = y1 - dy
                            
                            # Ensure within bounds
                            new_x1 = np.clip(new_x1, r1 + 0.01, self.rect_width - r1 - 0.01)
                            new_y1 = np.clip(new_y1, r1 + 0.01, self.rect_height - r1 - 0.01)
                            
                            fixed_circles[i] = [new_x1, new_y1, r1]
                            moved_any = True
                            
            if not moved_any:
                break
                
        return fixed_circles

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    optimizer = CirclePackingOptimizer(n_circles=21, rect_width=1.0, rect_height=1.0)
    return optimizer.optimize()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")