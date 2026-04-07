# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import random
from typing import Tuple, List
import time

# Global constants
RECT_PERIMETER = 4.0
NUM_CIRCLES = 21
SEED = 42

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)

class AdaptiveEvolutionaryPacker:
    def __init__(self, width: float = 1.0, height: float = 1.0, num_circles: int = NUM_CIRCLES):
        self.width = width
        self.height = height
        self.num_circles = num_circles
        self.rect_area = width * height

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

    def efficient_overlap_check(self, circles: np.ndarray, tree: cKDTree = None) -> int:
        """Efficiently check all overlaps using spatial indexing"""
        violations = 0

        if tree is None:
            # Build KDTree for fast neighbor search
            points = circles[:, :2]  # Only x,y coordinates
            tree = cKDTree(points)

        # Get max radius to determine search radius
        max_radius = np.max(circles[:, 2])

        try:
            # Query pairs efficiently with safety margin
            pairs = tree.query_pairs(2.5 * max_radius, output_type='ndarray')

            for i, j in pairs:
                if self.check_overlap(circles, i, j):
                    violations += 1
        except Exception:
            # Fallback to brute force if spatial indexing fails
            for i in range(self.num_circles):
                for j in range(i+1, self.num_circles):
                    if self.check_overlap(circles, i, j):
                        violations += 1

        return violations

    def calculate_total_radius_sum(self, circles: np.ndarray) -> float:
        """Calculate sum of all circle radii"""
        return np.sum(circles[:, 2])

    def calculate_fitness_with_penalty(self, circles: np.ndarray) -> Tuple[float, int]:
        """
        Calculate fitness: sum of radii with penalty for constraint violations
        """
        total_radius = self.calculate_total_radius_sum(circles)

        # Count constraint violations
        violations = 0

        # Check boundary violations
        for i in range(self.num_circles):
            x, y, r = circles[i]
            if not self.is_valid_circle(x, y, r):
                violations += 100  # Heavy penalty for boundary violations

        # Check overlap violations using optimized spatial indexing
        violations += self.efficient_overlap_check(circles)

        # Return fitness score with adaptive penalty
        penalty_weight = 1000.0 + (violations * 500.0)
        return total_radius - (penalty_weight * violations), violations

    def generate_initial_grid_layout(self) -> np.ndarray:
        """Generate initial configuration using adaptive grid-based approach"""
        circles = np.zeros((self.num_circles, 3))
        
        # Determine optimal grid dimensions based on aspect ratio
        aspect_ratio = self.width / self.height
        
        if aspect_ratio >= 1.2:  # Landscape orientation
            cols = int(np.ceil(np.sqrt(self.num_circles * aspect_ratio * 1.3)))
            rows = int(np.ceil(self.num_circles / cols))
        elif aspect_ratio <= 0.8:  # Portrait orientation
            rows = int(np.ceil(np.sqrt(self.num_circles / aspect_ratio * 1.3)))
            cols = int(np.ceil(self.num_circles / rows))
        else:  # Balanced
            cols = int(np.ceil(np.sqrt(self.num_circles * aspect_ratio)))
            rows = int(np.ceil(self.num_circles / cols))

        # Ensure enough cells
        while cols * rows < self.num_circles:
            if aspect_ratio >= 1.2:
                cols += 1
            elif aspect_ratio <= 0.8:
                rows += 1
            else:
                cols += 1

        # Calculate spacing
        spacing_x = self.width / (cols + 1) if cols > 0 else self.width
        spacing_y = self.height / (rows + 1) if rows > 0 else self.height

        # Create hexagonal-like packing with better spacing
        placed_count = 0
        for i in range(rows):
            for j in range(cols):
                if placed_count >= self.num_circles:
                    break

                # Offset every other row for hexagonal packing
                offset_x = spacing_x * 0.5 if i % 2 == 1 else 0
                base_x = (j + 1) * spacing_x + offset_x
                base_y = (i + 1) * spacing_y

                # Add small random perturbation for diversity
                perturbation_factor = min(0.1, 0.15 * min(spacing_x, spacing_y))
                x = np.clip(base_x + np.random.uniform(-perturbation_factor, perturbation_factor),
                           0.01, self.width - 0.01)
                y = np.clip(base_y + np.random.uniform(-perturbation_factor, perturbation_factor),
                           0.01, self.height - 0.01)

                # Initial radius estimation with better heuristics
                max_r = min(x, self.width - x, y, self.height - y)
                # Estimate based on packing density considerations
                estimated_radius = min(0.15, max_r * 0.7)
                r = np.random.uniform(estimated_radius * 0.6, estimated_radius * 1.0)

                circles[placed_count] = [x, y, r]
                placed_count += 1

            if placed_count >= self.num_circles:
                break

        return circles

    def smart_refinement_step(self, circles: np.ndarray, max_iter: int = 20) -> np.ndarray:
        """Apply smart refinement to improve packing"""
        refined_circles = circles.copy()
        best_circles = refined_circles.copy()
        best_fitness, _ = self.calculate_fitness_with_penalty(best_circles)
        
        # Try to increase individual radii while maintaining feasibility
        for iteration in range(max_iter):
            improved = False
            # Try increasing each circle's radius
            for i in range(self.num_circles):
                test_circles = refined_circles.copy()
                
                # Increase radius slightly
                original_radius = test_circles[i, 2]
                delta_radius = min(0.01, 0.1 - original_radius)
                if delta_radius <= 0:
                    continue
                    
                test_circles[i, 2] = min(original_radius + delta_radius, 0.2)
                
                # Check if this modification is valid
                if self._is_valid_configuration(test_circles):
                    # If valid, check if fitness improved
                    test_fitness, _ = self.calculate_fitness_with_penalty(test_circles)
                    if test_fitness > best_fitness:
                        best_fitness = test_fitness
                        best_circles = test_circles.copy()
                        improved = True
                        
            if not improved:
                break
                
            refined_circles = best_circles.copy()
            
        return best_circles

    def _is_valid_configuration(self, circles: np.ndarray) -> bool:
        """Quick validity check for a configuration"""
        # Check boundary constraints
        for i in range(self.num_circles):
            x, y, r = circles[i]
            if not self.is_valid_circle(x, y, r):
                return False
                
        # Check overlap constraints with a quick spatial query
        try:
            tree = cKDTree(circles[:, :2])
            pairs = tree.query_pairs(2 * np.max(circles[:, 2]), output_type='ndarray')
            
            for i, j in pairs:
                if self.check_overlap(circles, i, j):
                    return False
        except:
            # Fallback to slow check if spatial indexing fails
            for i in range(self.num_circles):
                for j in range(i+1, self.num_circles):
                    if self.check_overlap(circles, i, j):
                        return False
                        
        return True

    def optimize_with_multi_stage_approach(self) -> np.ndarray:
        """Main optimization using multi-stage approach"""
        start_time = time.time()
        
        # Phase 1: Coarse optimization to find promising region
        initial_layout = self.generate_initial_grid_layout()
        
        # Refine initial layout
        refined_layout = self.smart_refinement_step(initial_layout, max_iter=15)
        
        # Phase 2: Local optimization around the best found solution
        # Use constrained optimization to fine-tune the arrangement
        try:
            # Convert to flat variable format for optimization
            initial_vars = self._circles_to_variables(refined_layout)
            
            # Define bounds
            bounds = []
            for i in range(self.num_circles):
                bounds.extend([(0.001, self.width - 0.001),   # x bounds
                               (0.001, self.height - 0.001),  # y bounds
                               (0.001, 0.3)])                # radius bounds
            
            # Objective function (minimize negative sum of radii)
            def objective(vars):
                circles = self._variables_to_circles(vars)
                return -np.sum(circles[:, 2])
                
            # Constraint function for boundary constraints
            def boundary_constraint(vars):
                circles = self._variables_to_circles(vars)
                violations = []
                for i in range(self.num_circles):
                    x, y, r = circles[i]
                    # x >= r
                    violations.append(x - r)
                    # width - x >= r
                    violations.append(self.width - x - r)
                    # y >= r
                    violations.append(y - r)
                    # height - y >= r
                    violations.append(self.height - y - r)
                return np.array(violations)
                
            # Constraint function for overlap constraints
            def overlap_constraint(vars):
                circles = self._variables_to_circles(vars)
                violations = []
                for i in range(self.num_circles):
                    for j in range(i+1, self.num_circles):
                        x1, y1, r1 = circles[i]
                        x2, y2, r2 = circles[j]
                        dx = x1 - x2
                        dy = y1 - y2
                        distance = np.sqrt(dx*dx + dy*dy)
                        # distance >= r1 + r2 (so we return distance - (r1 + r2) for inequality)
                        violations.append(distance - (r1 + r2))
                return np.array(violations)
                
            # Create constraints dictionary
            constraints = [
                {'type': 'ineq', 'fun': boundary_constraint},
                {'type': 'ineq', 'fun': overlap_constraint}
            ]
            
            # Optimize
            result = minimize(
                objective,
                initial_vars,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 50, 'ftol': 1e-6}
            )
            
            if result.success:
                optimized_circles = self._variables_to_circles(result.x)
            else:
                optimized_circles = refined_layout.copy()
        except:
            # Fallback to refined layout if optimization fails
            optimized_circles = refined_layout.copy()

        # Final refinement
        final_circles = self.smart_refinement_step(optimized_circles, max_iter=10)
        
        end_time = time.time()
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")
        
        return final_circles

    def _circles_to_variables(self, circles: np.ndarray) -> np.ndarray:
        """Convert circles array to flat variables array"""
        variables = []
        for i in range(self.num_circles):
            variables.extend([circles[i, 0], circles[i, 1], circles[i, 2]])
        return np.array(variables)

    def _variables_to_circles(self, variables: np.ndarray) -> np.ndarray:
        """Convert flat variables array back to circles array"""
        circles = np.zeros((self.num_circles, 3))
        for i in range(self.num_circles):
            circles[i, 0] = variables[3*i]
            circles[i, 1] = variables[3*i + 1]
            circles[i, 2] = variables[3*i + 2]
        return circles

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Try different rectangle sizes to find optimal aspect ratio
    best_circles = None
    best_sum = 0
    
    # Test several rectangle aspect ratios
    test_ratios = [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]
    
    for ratio in test_ratios:
        width = 2 * ratio / (1 + ratio)  # Ensure perimeter = 4
        height = 2 * 1 / (1 + ratio)
        
        # Create packer instance
        packer = AdaptiveEvolutionaryPacker(width=width, height=height, num_circles=21)
        
        # Optimize
        circles = packer.optimize_with_multi_stage_approach()
        
        # Check if this is better
        current_sum = np.sum(circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = circles.copy()
    
    return best_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
