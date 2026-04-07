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

class AdaptiveGravityPacker:
    def __init__(self, width: float = 1.0, height: float = 1.0, num_circles: int = NUM_CIRCLES):
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

    def efficient_overlap_check(self, circles: np.ndarray, tree: cKDTree = None) -> int:
        """Efficiently check all overlaps using spatial indexing"""
        violations = 0

        if tree is None:
            # Build KDTree for fast neighbor search
            points = circles[:, :2]  # Only x,y coordinates
            tree = cKDTree(points)

        # Get max radius to determine search radius
        max_radius = np.max(circles[:, 2])

        # Query pairs efficiently with safety margin
        try:
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

    def calculate_fitness_with_penalties(self, circles: np.ndarray) -> Tuple[float, int]:
        """
        Calculate fitness with carefully balanced penalties
        """
        total_radius = self.calculate_total_radius_sum(circles)

        # Count constraint violations
        violations = 0
        
        # Check boundary violations first (they are most critical)
        boundary_violations = 0
        for i in range(self.num_circles):
            x, y, r = circles[i]
            if not self.is_valid_circle(x, y, r):
                boundary_violations += 1
                
        violations += boundary_violations
        
        # Check overlap violations using optimized spatial indexing
        overlap_violations = self.efficient_overlap_check(circles)
        violations += overlap_violations

        # Apply penalties with proper weighting
        # Boundary violations are much more severe
        penalty = boundary_violations * 1000.0 + overlap_violations * 50.0
        
        # Return fitness (higher is better) 
        return total_radius - penalty, violations

    def generate_gravity_initialization(self) -> np.ndarray:
        """
        Generate initial configuration using a gravity-inspired approach:
        - Place circles in a grid pattern
        - Apply gentle repulsion forces to separate overlapping circles  
        - Use physics-like simulation to achieve natural distribution
        """
        circles = np.zeros((self.num_circles, 3))
        
        # Start with grid-based placement
        aspect_ratio = self.width / self.height
        
        # Determine grid dimensions
        cols = int(np.ceil(np.sqrt(self.num_circles * aspect_ratio)))
        rows = int(np.ceil(self.num_circles / cols))
        
        # Adjust to make sure we have enough cells
        while cols * rows < self.num_circles:
            if aspect_ratio >= 1:
                cols += 1
            else:
                rows += 1
                
        # Calculate grid spacing
        spacing_x = self.width / (cols + 1) if cols > 0 else self.width
        spacing_y = self.height / (rows + 1) if rows > 0 else self.height
        
        # Fill grid with circles
        placed_count = 0
        for i in range(rows):
            for j in range(cols):
                if placed_count >= self.num_circles:
                    break
                    
                # Hexagonal offset
                offset_x = spacing_x * 0.5 if i % 2 == 1 else 0
                base_x = (j + 1) * spacing_x + offset_x
                base_y = (i + 1) * spacing_y
                
                # Add small random perturbation
                perturbation_factor = min(0.1, 0.2 * min(spacing_x, spacing_y))
                x = np.clip(base_x + np.random.uniform(-perturbation_factor, perturbation_factor),
                           0.01, self.width - 0.01)
                y = np.clip(base_y + np.random.uniform(-perturbation_factor, perturbation_factor),
                           0.01, self.height - 0.01)
                
                # Initial radius estimation
                max_r = min(x, self.width - x, y, self.height - y)
                estimated_radius = min(0.15, max_r * 0.7)
                r = np.random.uniform(estimated_radius * 0.6, estimated_radius * 1.0)
                
                circles[placed_count] = [x, y, r]
                placed_count += 1
                
            if placed_count >= self.num_circles:
                break
                
        # Apply a simple gravity-based refinement
        # Repel overlapping circles gently
        for _ in range(50):  # Allow some iterations for refinement
            # Build spatial index for efficient neighbor search
            try:
                tree = cKDTree(circles[:, :2])
                pairs = tree.query_pairs(2.0 * np.max(circles[:, 2]), output_type='ndarray')
                
                # Apply gentle repulsion forces
                for i, j in pairs:
                    if i >= self.num_circles or j >= self.num_circles:
                        continue
                    if self.check_overlap(circles, i, j):
                        x1, y1, r1 = circles[i]
                        x2, y2, r2 = circles[j]
                        
                        # Calculate separation vector
                        dx = x1 - x2
                        dy = y1 - y2
                        dist = np.sqrt(dx*dx + dy*dy)
                        
                        if dist > 0.001:  # Avoid division by zero
                            # Normalize direction vector
                            dx /= dist
                            dy /= dist
                            
                            # Push circles apart (with small force)
                            force_magnitude = 0.005 * (r1 + r2 - dist)
                            circles[i, 0] += dx * force_magnitude * 0.5
                            circles[i, 1] += dy * force_magnitude * 0.5
                            circles[j, 0] -= dx * force_magnitude * 0.5
                            circles[j, 1] -= dy * force_magnitude * 0.5
                            
                        # Keep within bounds
                        circles[i, 0] = np.clip(circles[i, 0], r1, self.width - r1)
                        circles[i, 1] = np.clip(circles[i, 1], r1, self.height - r1)
                        circles[j, 0] = np.clip(circles[j, 0], r2, self.width - r2)
                        circles[j, 1] = np.clip(circles[j, 1], r2, self.height - r2)
            except:
                # Fallback if spatial indexing fails
                pass
                
        return circles

    def local_optimization_step(self, circles: np.ndarray, max_iter: int = 20) -> np.ndarray:
        """
        Local optimization focusing on improving individual radii while keeping constraints
        """
        optimized_circles = circles.copy()
        
        # Precompute spatial information
        max_radius = np.max(optimized_circles[:, 2])
        if max_radius <= 0:
            return optimized_circles
            
        try:
            tree = cKDTree(optimized_circles[:, :2])
            pairs = tree.query_pairs(2.5 * max_radius, output_type='ndarray')
        except:
            pairs = []
            
        improvement_count = 0
        iteration = 0
        
        # Iteratively try to increase radii
        while improvement_count < 5 and iteration < max_iter:
            improvement_count = 0
            iteration += 1
            
            # Shuffle order for fair treatment
            indices = list(range(self.num_circles))
            random.shuffle(indices)
            
            for i in indices:
                # Try to increase the radius of circle i
                current_radius = optimized_circles[i, 2]
                max_possible_radius = min(
                    optimized_circles[i, 0], 
                    self.width - optimized_circles[i, 0],
                    optimized_circles[i, 1], 
                    self.height - optimized_circles[i, 1]
                )
                
                if max_possible_radius <= current_radius:
                    continue
                    
                # Try different increment sizes
                increments = [0.005, 0.01, 0.015, 0.02]
                for inc in increments:
                    new_radius = min(current_radius + inc, max_possible_radius)
                    if new_radius <= current_radius:
                        continue
                        
                    # Temporarily update radius
                    temp_circles = optimized_circles.copy()
                    temp_circles[i, 2] = new_radius
                    
                    # Check constraints
                    valid = True
                    
                    # Check boundary constraints
                    if not self.is_valid_circle(temp_circles[i, 0], temp_circles[i, 1], new_radius):
                        valid = False
                        
                    if not valid:
                        continue
                        
                    # Check overlap constraints with all others
                    try:
                        temp_tree = cKDTree(temp_circles[:, :2])
                        temp_pairs = temp_tree.query_pairs(2.5 * new_radius, output_type='ndarray')
                        
                        for pi, pj in temp_pairs:
                            if pi == i and pj != i:
                                if self.check_overlap(temp_circles, pi, pj):
                                    valid = False
                                    break
                            elif pj == i and pi != i:
                                if self.check_overlap(temp_circles, pi, pj):
                                    valid = False
                                    break
                                
                    except:
                        # Fallback to brute force check
                        for j in range(self.num_circles):
                            if i != j:
                                if self.check_overlap(temp_circles, i, j):
                                    valid = False
                                    break
                                    
                    if valid:
                        optimized_circles = temp_circles.copy()
                        improvement_count += 1
                        break
                        
        return optimized_circles

    def hierarchical_optimization(self) -> np.ndarray:
        """
        Multi-scale optimization approach:
        1. Coarse grid-based initialization
        2. Local refinement with radius maximization
        3. Fine-grained constraint satisfaction
        """
        # Phase 1: Gravity-based initialization
        circles = self.generate_gravity_initialization()
        
        # Phase 2: Initial local optimization
        circles = self.local_optimization_step(circles, max_iter=30)
        
        # Phase 3: Scipy optimization for fine-tuning
        try:
            # Prepare optimization variables
            def objective(vars):
                # vars contains [x1, y1, r1, x2, y2, r2, ...]
                circles_temp = vars.reshape(-1, 3)
                # Maximize sum of radii (minimize negative sum)
                return -np.sum(circles_temp[:, 2])
            
            # Initial variables
            initial_vars = circles.flatten()
            
            # Bounds for each variable (x, y, r)
            bounds = []
            for i in range(self.num_circles):
                bounds.extend([(0.001, self.width - 0.001),   # x bounds
                               (0.001, self.height - 0.001),  # y bounds
                               (0.001, 0.3)])                # radius bounds

            # Constraints for optimization
            constraints = []

            # Boundary constraints
            for i in range(self.num_circles):
                def x_bound_left(vars, i=i):
                    circles_temp = vars.reshape(-1, 3)
                    x, y, r = circles_temp[i]
                    return x - r  # x >= r

                def x_bound_right(vars, i=i):
                    circles_temp = vars.reshape(-1, 3)
                    x, y, r = circles_temp[i]
                    return self.width - x - r  # width - x >= r

                def y_bound_bottom(vars, i=i):
                    circles_temp = vars.reshape(-1, 3)
                    x, y, r = circles_temp[i]
                    return y - r  # y >= r

                def y_bound_top(vars, i=i):
                    circles_temp = vars.reshape(-1, 3)
                    x, y, r = circles_temp[i]
                    return self.height - y - r  # height - y >= r

                constraints.append({'type': 'ineq', 'fun': x_bound_left})
                constraints.append({'type': 'ineq', 'fun': x_bound_right})
                constraints.append({'type': 'ineq', 'fun': y_bound_bottom})
                constraints.append({'type': 'ineq', 'fun': y_bound_top})

            # Overlap constraints
            for i in range(self.num_circles):
                for j in range(i+1, self.num_circles):
                    def overlap_constraint(vars, i=i, j=j):
                        circles_temp = vars.reshape(-1, 3)
                        x1, y1, r1 = circles_temp[i]
                        x2, y2, r2 = circles_temp[j]
                        dx = x1 - x2
                        dy = y1 - y2
                        distance = np.sqrt(dx*dx + dy*dy)
                        # distance >= r1 + r2 (so we return distance - (r1 + r2) for inequality)
                        return distance - (r1 + r2)
                        
                    constraints.append({'type': 'ineq', 'fun': overlap_constraint})

            # Optimize with SLSQP
            result = minimize(
                objective,
                initial_vars,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            
            if result.success:
                circles = result.x.reshape(-1, 3)
            else:
                # Fallback to previous state
                pass
                
        except Exception as e:
            # If optimization fails, continue with current circles
            pass
        
        # Phase 4: Final refinement
        circles = self.local_optimization_step(circles, max_iter=20)
        
        return circles

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Try multiple rectangle aspect ratios to find optimal configuration
    best_circles = None
    best_sum = 0
    
    # Test several aspect ratios
    test_ratios = [0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
    
    for ratio in test_ratios:
        width = 2 * ratio / (1 + ratio)  # Ensure perimeter = 4
        height = 2 * 1 / (1 + ratio)
        
        try:
            packer = AdaptiveGravityPacker(width=width, height=height, num_circles=21)
            circles = packer.hierarchical_optimization()
            
            current_sum = np.sum(circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = circles.copy()
                
        except Exception:
            # Skip failed configurations
            continue
    
    if best_circles is None:
        # Fallback to default setup
        packer = AdaptiveGravityPacker(width=1.0, height=1.0, num_circles=21)
        best_circles = packer.hierarchical_optimization()
        
    return best_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")