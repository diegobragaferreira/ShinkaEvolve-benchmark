# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import minimize
import random
import math

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Rectangle dimensions: width + height = 2
    # Optimize for 2:1 aspect ratio for better packing efficiency
    rect_width = 1.3333333333333333  # 2/3
    rect_height = 0.6666666666666666  # 1/3

    # Number of circles
    n = 21

    def generate_golden_hexagonal_grid(width, height, num_circles):
        """Generate initial configuration using golden ratio hexagonal packing"""
        # Golden ratio for optimal spacing
        phi = (1 + math.sqrt(5)) / 2
        
        # Determine grid dimensions
        rows = max(3, int(math.ceil(math.sqrt(num_circles * 1.5))))
        cols = max(3, int(math.ceil(num_circles / rows)))
        
        # Golden ratio spacing
        spacing_x = width / (cols + 1)
        spacing_y = height / (rows + 1)
        
        # Apply golden ratio adjustment for better packing
        hex_spacing_x = spacing_x * 0.75
        hex_spacing_y = spacing_y * (0.866 * 0.9)  # sqrt(3)/2 * adjustment
        
        circles = []
        idx = 0
        
        for i in range(rows):
            for j in range(cols):
                if idx >= num_circles:
                    break
                    
                # Hexagonal offset for odd rows
                x_offset = (i % 2) * (hex_spacing_x / 2)
                x = hex_spacing_x * j + x_offset + hex_spacing_x
                y = hex_spacing_y * i + hex_spacing_y
                
                # Ensure within bounds with safety margin
                x = max(hex_spacing_x, min(width - hex_spacing_x, x))
                y = max(hex_spacing_y, min(height - hex_spacing_y, y))
                
                # Initialize with radius based on spacing
                r = min(hex_spacing_x, hex_spacing_y) * 0.25
                
                circles.append([x, y, r])
                idx += 1
                
                if idx >= num_circles:
                    break
                    
        # Fill remaining slots if needed
        while len(circles) < num_circles:
            x = np.random.uniform(hex_spacing_x, width - hex_spacing_x)
            y = np.random.uniform(hex_spacing_y, height - hex_spacing_y)
            r = min(hex_spacing_x, hex_spacing_y) * 0.25
            circles.append([x, y, r])
            
        return np.array(circles)

    def compute_penalty(circles_array):
        """Compute penalty for constraint violations"""
        penalty = 0
        n = len(circles_array)
        
        # Boundary penalties
        for i in range(n):
            cx, cy, r = circles_array[i]
            if cx - r < 0:
                penalty += 10000 * (r - cx)**2
            if cx + r > rect_width:
                penalty += 10000 * (cx + r - rect_width)**2
            if cy - r < 0:
                penalty += 10000 * (r - cy)**2
            if cy + r > rect_height:
                penalty += 10000 * (cy + r - rect_height)**2

        # Overlap penalties using KDTree for efficiency
        points = circles_array[:, :2]
        tree = KDTree(points)
        
        for i in range(n):
            cx, cy, r = circles_array[i]
            # Get nearby circles efficiently
            neighbor_indices = tree.query_ball_point([cx, cy], 2*(r + 0.01))
            
            for j in neighbor_indices:
                if i != j:
                    other_cx, other_cy, other_r = circles_array[j]
                    dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                    overlap = (r + other_r) - dist
                    
                    if overlap > 0:
                        penalty += 100000 * overlap**2
                        
        return penalty

    def calculate_total_radius(circles_array):
        """Calculate sum of all radii"""
        return np.sum(circles_array[:, 2])

    def constraint_aware_fitness(circles_array):
        """Fitness function considering both objective and constraints"""
        total_radius = calculate_total_radius(circles_array)
        penalty = compute_penalty(circles_array)
        # Fitness is total radius minus penalty (negative penalty means violation)
        return total_radius - penalty

    def expand_radius(circles_array, i, max_radius):
        """Try to expand radius of circle i to max_radius, checking all constraints"""
        if max_radius <= 0:
            return False, circles_array[i, 2]
            
        current_cx, current_cy, current_r = circles_array[i]
        
        # Validate against current solution
        temp_circles = circles_array.copy()
        temp_circles[i, 2] = max_radius
        
        # Check overlaps with all circles using spatial indexing
        points = temp_circles[:, :2]
        tree = KDTree(points)
        neighbor_indices = tree.query_ball_point([current_cx, current_cy], 2 * max_radius + 0.001)
        
        for j in neighbor_indices:
            if i != j:
                other_cx, other_cy, other_r = temp_circles[j]
                dist = np.sqrt((current_cx - other_cx)**2 + (current_cy - other_cy)**2)
                if dist < max_radius + other_r:
                    return False, current_r  # Cannot expand
                    
        return True, max_radius

    def multi_scale_refinement(initial_circles, max_iterations=500):
        """Refine solution using multi-scale approach"""
        current_circles = initial_circles.copy()
        best_circles = current_circles.copy()
        best_fitness = constraint_aware_fitness(best_circles)
        
        # Phase 1: Coarse refinement with large steps
        phase1_steps = max_iterations // 3
        for iteration in range(phase1_steps):
            improved = False
            
            # Random shuffling for better exploration
            indices = list(range(n))
            random.shuffle(indices)
            
            for i in indices:
                cx, cy, r = current_circles[i]
                
                # Compute maximum allowable radius
                max_radius = float('inf')
                
                # Boundary constraints
                max_radius = min(max_radius, cx)
                max_radius = min(max_radius, rect_width - cx)
                max_radius = min(max_radius, cy)
                max_radius = min(max_radius, rect_height - cy)
                
                # Overlap constraints
                points = current_circles[:, :2]
                tree = KDTree(points)
                neighbor_indices = tree.query_ball_point([cx, cy], 2*(r + 0.01))
                
                for j in neighbor_indices:
                    if i != j:
                        other_cx, other_cy, other_r = current_circles[j]
                        dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                        max_radius = min(max_radius, dist - other_r - 0.001)
                
                # Try to expand radius with larger step
                if max_radius > r and max_radius > 0.001:
                    attempt_radius = min(r + 0.02, max_radius)  # Larger step
                    success, new_radius = expand_radius(current_circles, i, attempt_radius)
                    
                    if success:
                        current_circles[i, 2] = new_radius
                        improved = True
                        
            # Early stopping check
            current_fitness = constraint_aware_fitness(current_circles)
            if current_fitness > best_fitness:
                best_circles = current_circles.copy()
                best_fitness = current_fitness
                
            if not improved and iteration > 50:
                break
                
        # Phase 2: Medium refinement with moderate steps  
        phase2_steps = max_iterations // 3
        for iteration in range(phase2_steps):
            improved = False
            
            # Random shuffling for better exploration
            indices = list(range(n))
            random.shuffle(indices)
            
            for i in indices:
                cx, cy, r = current_circles[i]
                
                # Compute maximum allowable radius
                max_radius = float('inf')
                
                # Boundary constraints
                max_radius = min(max_radius, cx)
                max_radius = min(max_radius, rect_width - cx)
                max_radius = min(max_radius, cy)
                max_radius = min(max_radius, rect_height - cy)
                
                # Overlap constraints
                points = current_circles[:, :2]
                tree = KDTree(points)
                neighbor_indices = tree.query_ball_point([cx, cy], 2*(r + 0.01))
                
                for j in neighbor_indices:
                    if i != j:
                        other_cx, other_cy, other_r = current_circles[j]
                        dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                        max_radius = min(max_radius, dist - other_r - 0.001)
                
                # Try to expand radius with moderate step
                if max_radius > r and max_radius > 0.001:
                    attempt_radius = min(r + 0.01, max_radius)  # Moderate step
                    success, new_radius = expand_radius(current_circles, i, attempt_radius)
                    
                    if success:
                        current_circles[i, 2] = new_radius
                        improved = True
                        
            # Early stopping check
            current_fitness = constraint_aware_fitness(current_circles)
            if current_fitness > best_fitness:
                best_circles = current_circles.copy()
                best_fitness = current_fitness
                
            if not improved and iteration > 30:
                break
                
        # Phase 3: Fine refinement with small steps
        phase3_steps = max_iterations // 3
        for iteration in range(phase3_steps):
            improved = False
            
            # Random shuffling for better exploration
            indices = list(range(n))
            random.shuffle(indices)
            
            for i in indices:
                cx, cy, r = current_circles[i]
                
                # Compute maximum allowable radius
                max_radius = float('inf')
                
                # Boundary constraints
                max_radius = min(max_radius, cx)
                max_radius = min(max_radius, rect_width - cx)
                max_radius = min(max_radius, cy)
                max_radius = min(max_radius, rect_height - cy)
                
                # Overlap constraints
                points = current_circles[:, :2]
                tree = KDTree(points)
                neighbor_indices = tree.query_ball_point([cx, cy], 2*(r + 0.01))
                
                for j in neighbor_indices:
                    if i != j:
                        other_cx, other_cy, other_r = current_circles[j]
                        dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                        max_radius = min(max_radius, dist - other_r - 0.001)
                
                # Try to expand radius with small step
                if max_radius > r and max_radius > 0.001:
                    attempt_radius = min(r + 0.005, max_radius)  # Small step
                    success, new_radius = expand_radius(current_circles, i, attempt_radius)
                    
                    if success:
                        current_circles[i, 2] = new_radius
                        improved = True
                        
            # Early stopping check
            current_fitness = constraint_aware_fitness(current_circles)
            if current_fitness > best_fitness:
                best_circles = current_circles.copy()
                best_fitness = current_fitness
                
            if not improved and iteration > 10:
                break
                
        return best_circles

    def evolutionary_repair(initial_circles, generations=20):
        """Use evolutionary approach to repair and improve solution"""
        best_solution = initial_circles.copy()
        best_fitness = constraint_aware_fitness(best_solution)
        
        # Simple evolutionary approach: generate variants and keep best
        for gen in range(generations):
            # Create variant with small mutations
            variant = best_solution.copy()
            
            # Mutate some circles
            num_mutations = random.randint(1, 5)
            for _ in range(num_mutations):
                circle_idx = random.randint(0, n-1)
                # Slight position variation
                variant[circle_idx, 0] += random.uniform(-0.05, 0.05)
                variant[circle_idx, 1] += random.uniform(-0.05, 0.05)
                # Clamp to bounds
                variant[circle_idx, 0] = np.clip(variant[circle_idx, 0], 0.01, rect_width - 0.01)
                variant[circle_idx, 1] = np.clip(variant[circle_idx, 1], 0.01, rect_height - 0.01)
                
                # Slight radius variation
                variant[circle_idx, 2] *= np.random.uniform(0.9, 1.1)
                variant[circle_idx, 2] = max(0.001, variant[circle_idx, 2])
            
            # Evaluate variant
            variant_fitness = constraint_aware_fitness(variant)
            
            if variant_fitness > best_fitness:
                best_solution = variant
                best_fitness = variant_fitness
                
        return best_solution

    # Generate initial configuration
    initial_circles = generate_golden_hexagonal_grid(rect_width, rect_height, n)
    
    # Multi-scale refinement
    refined_circles = multi_scale_refinement(initial_circles, max_iterations=800)
    
    # Try evolutionary repair if needed
    repaired_circles = evolutionary_repair(refined_circles, generations=15)
    
    # Final validation and refinement
    final_circles = multi_scale_refinement(repaired_circles, max_iterations=200)
    
    return final_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")