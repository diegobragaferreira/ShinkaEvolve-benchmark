# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree, Voronoi
from scipy.spatial.distance import cdist
import random
import time
from typing import Tuple, List

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Rectangle dimensions (width + height = 2)
    # Optimize rectangle dimensions for better packing
    rect_width = 1.5
    rect_height = 0.5

    # Number of circles
    n = 21

    def generate_initial_patterns(width: float, height: float, n: int) -> List[np.ndarray]:
        """Generate multiple initial patterns for diverse starting points"""
        patterns = []
        
        # Pattern 1: Hexagonal packing
        try:
            hex_pattern = generate_hexagonal_pattern(width, height, n)
            patterns.append(hex_pattern)
        except:
            pass
            
        # Pattern 2: Grid-based packing  
        try:
            grid_pattern = generate_grid_pattern(width, height, n)
            patterns.append(grid_pattern)
        except:
            pass
            
        # Pattern 3: Spiral pattern
        try:
            spiral_pattern = generate_spiral_pattern(width, height, n)
            patterns.append(spiral_pattern)
        except:
            pass
            
        # Pattern 4: Random constrained pattern
        try:
            random_pattern = generate_random_constrained_pattern(width, height, n)
            patterns.append(random_pattern)
        except:
            pass
            
        # Pattern 5: Centered dense packing
        try:
            center_pattern = generate_centered_pattern(width, height, n)
            patterns.append(center_pattern)
        except:
            pass
            
        # If no patterns worked, fall back to simple grid
        if not patterns:
            patterns.append(generate_simple_grid_pattern(width, height, n))
            
        return patterns

    def generate_hexagonal_pattern(width: float, height: float, n: int) -> np.ndarray:
        """Generate initial hexagonal packing pattern"""
        circles = np.zeros((n, 3))

        # Determine grid parameters
        rows = int(np.sqrt(n))
        cols = int(np.ceil(n / rows))

        # Calculate spacing
        margin = 0.05
        max_radius = min(width, height) * 0.08

        # Create hexagonal grid
        x_spacing = max_radius * 2.5
        y_spacing = max_radius * 2.165  # sqrt(3)/2 * 2

        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = margin + j * x_spacing
                y = margin + i * y_spacing

                if i % 2 == 1:
                    x += x_spacing / 2

                # Adjust for bounds
                x = max(max_radius, min(width - max_radius, x))
                y = max(max_radius, min(height - max_radius, y))

                circles[idx] = [x, y, max_radius]
                idx += 1

        return circles

    def generate_grid_pattern(width: float, height: float, n: int) -> np.ndarray:
        """Generate initial grid pattern"""
        circles = np.zeros((n, 3))

        # Find grid dimensions
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        # Calculate spacing
        margin = 0.05
        cell_width = (width - 2 * margin) / cols
        cell_height = (height - 2 * margin) / rows
        max_radius = min(cell_width, cell_height) * 0.4

        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = margin + j * cell_width + cell_width / 2
                y = margin + i * cell_height + cell_height / 2
                circles[idx] = [x, y, max_radius]
                idx += 1

        return circles

    def generate_simple_grid_pattern(width: float, height: float, n: int) -> np.ndarray:
        """Generate simple grid pattern for fallback"""
        circles = np.zeros((n, 3))
        
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        
        margin = 0.05
        cell_width = (width - 2 * margin) / cols
        cell_height = (height - 2 * margin) / rows
        max_radius = min(cell_width, cell_height) * 0.3
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = margin + j * cell_width + cell_width / 2
                y = margin + i * cell_height + cell_height / 2
                circles[idx] = [x, y, max_radius]
                idx += 1
                
        return circles

    def generate_spiral_pattern(width: float, height: float, n: int) -> np.ndarray:
        """Generate initial spiral pattern"""
        circles = np.zeros((n, 3))
        center_x, center_y = width / 2, height / 2
        max_radius = min(width, height) * 0.1
        angle_step = 2 * np.pi / 5
        radius_step = 0.05

        for i in range(n):
            angle = i * angle_step
            radius = i * radius_step
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)

            # Keep within bounds
            x = max(max_radius, min(width - max_radius, x))
            y = max(max_radius, min(height - max_radius, y))

            circles[i] = [x, y, max_radius]

        return circles

    def generate_centered_pattern(width: float, height: float, n: int) -> np.ndarray:
        """Generate pattern with dense center and sparse periphery"""
        circles = np.zeros((n, 3))
        
        # Put 5 circles in center area
        center_radius = min(width, height) * 0.15
        center_x, center_y = width / 2, height / 2
        
        # Center cluster
        for i in range(min(5, n)):
            angle = i * 2 * np.pi / 5
            radius = center_radius * 0.5
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            circles[i] = [x, y, center_radius * 0.7]
        
        # Remaining with more spread out positions
        remaining = n - 5
        if remaining > 0:
            # Place remaining in outer area
            for i in range(remaining):
                x = np.random.uniform(center_radius, width - center_radius)
                y = np.random.uniform(center_radius, height - center_radius)
                r = np.random.uniform(0.005, center_radius * 0.8)
                circles[i + 5] = [x, y, r]
                
        return circles

    def generate_random_constrained_pattern(width: float, height: float, n: int) -> np.ndarray:
        """Generate random pattern with basic constraints"""
        circles = np.zeros((n, 3))
        max_radius = min(width, height) * 0.08
        attempts = 0

        for i in range(n):
            attempts = 0
            valid = False
            while not valid and attempts < 1000:
                x = np.random.uniform(max_radius, width - max_radius)
                y = np.random.uniform(max_radius, height - max_radius)
                radius = np.random.uniform(0.005, max_radius)

                # Check if this circle overlaps with existing ones
                valid = True
                for j in range(i):
                    existing_x, existing_y, existing_r = circles[j]
                    dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                    if dist < (radius + existing_r):
                        valid = False
                        break

                if valid:
                    circles[i] = [x, y, radius]
                attempts += 1

        return circles

    def check_constraints(circles: np.ndarray, width: float, height: float) -> bool:
        """Fast constraint checking using spatial indexing"""
        if len(circles) <= 1:
            return True
            
        # Check boundary constraints
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
                return False
        
        # Use KDTree for efficient collision detection
        coords = circles[:, :2]
        radii = circles[:, 2]
        
        try:
            tree = cKDTree(coords)
            
            # Query neighbors within sum of radii
            pairs = tree.query_pairs(radii.max() * 2, p=np.inf)
            
            # Check actual overlaps
            for i, j in pairs:
                if i < j:  # Only check each pair once
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if dist < (r1 + r2):
                        return False
                        
        except:
            # Fallback to brute-force if KDTree fails
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if dist < (r1 + r2):
                        return False
                        
        return True

    def evaluate_fitness(circles: np.ndarray) -> float:
        """Evaluate fitness as the sum of radii with constraint validation"""
        if not check_constraints(circles, rect_width, rect_height):
            return -np.inf
            
        return np.sum(circles[:, 2])

    def get_voronoi_criticality(circles: np.ndarray) -> np.ndarray:
        """Calculate criticality based on Voronoi diagram - more efficient version"""
        n = len(circles)
        if n <= 1:
            return np.ones(n) * 0.01
            
        try:
            # Use Voronoi for criticality assessment
            points = circles[:, :2]
            vor = Voronoi(points)
            
            # For each point, compute minimum distance to Voronoi edges
            # This gives us a measure of how much space is available for expansion
            criticality_scores = np.zeros(n)
            
            for i in range(n):
                # Find the Voronoi region for this point
                region_idx = vor.point_region[i]
                if region_idx != -1 and region_idx < len(vor.regions):
                    region = vor.regions[region_idx]
                    if len(region) > 0 and -1 not in region:
                        # Find minimum distance to Voronoi vertices
                        min_dist = float('inf')
                        for vertex_idx in region:
                            if vertex_idx < len(vor.vertices):
                                vertex = vor.vertices[vertex_idx]
                                dist = np.sqrt((circles[i,0] - vertex[0])**2 + (circles[i,1] - vertex[1])**2)
                                min_dist = min(min_dist, dist)
                        
                        if min_dist < float('inf'):
                            # Higher criticality means less room to grow
                            # This is inverse of available space
                            criticality_scores[i] = 1.0 / (min_dist + 0.001)
                            
            # Handle cases where Voronoi failed or gave invalid results
            if np.all(criticality_scores <= 0):
                # Fall back to neighbor-based method
                return get_neighbor_criticality(circles)
                
            # Normalize criticality scores
            if np.max(criticality_scores) > 0:
                criticality_scores = criticality_scores / np.max(criticality_scores) * 100
            else:
                criticality_scores.fill(1.0)
                
            # Ensure minimum positive criticality
            criticality_scores = np.maximum(criticality_scores, 0.01)
            
            return criticality_scores
            
        except:
            # Fallback to neighbor-based criticality calculation
            return get_neighbor_criticality(circles)

    def get_neighbor_criticality(circles: np.ndarray) -> np.ndarray:
        """Fallback criticality calculation using nearest neighbors"""
        n = len(circles)
        if n <= 1:
            return np.ones(n) * 0.01
            
        # Use spatial tree for fast nearest neighbor queries
        coords = circles[:, :2]
        tree = cKDTree(coords)
        
        criticality_scores = np.zeros(n)
        
        # For each circle, find distance to nearest neighbor and boundary
        for i in range(n):
            x, y, r = circles[i]
            
            # Find distance to nearest neighbor (excluding self)
            distances, indices = tree.query(coords[i], k=2)  # k=2 to exclude self
            nearest_neighbor_dist = distances[1] if len(distances) > 1 else float('inf')
            
            # Find distance to nearest boundary
            boundary_dists = [x, y, rect_width - x, rect_height - y]
            min_boundary_dist = min(boundary_dists)
            
            # Criticality is high when both distances are small
            # This means the circle is constrained by neighbors and/or boundaries
            neighbor_constraint = 1.0 / (nearest_neighbor_dist + 0.001) if nearest_neighbor_dist < float('inf') else 1000
            boundary_constraint = 0
            if min_boundary_dist < 0.05:
                boundary_constraint = 1000 * (0.05 - min_boundary_dist)
                
            criticality_scores[i] = neighbor_constraint + boundary_constraint
            
        # Normalize and ensure minimum positive value
        if np.max(criticality_scores) > 0:
            criticality_scores = criticality_scores / np.max(criticality_scores) * 100
        criticality_scores = np.maximum(criticality_scores, 0.01)
        
        return criticality_scores

    def is_valid_solution(circles: np.ndarray) -> bool:
        """Check if solution is valid"""
        return check_constraints(circles, rect_width, rect_height)

    def optimize_critical_circles(circles: np.ndarray, max_iterations: int = 200) -> np.ndarray:
        """Optimize the most critical circles using local search"""
        result = circles.copy()
        n = len(result)
        
        for iteration in range(max_iterations):
            # Get criticality scores
            criticality = get_voronoi_criticality(result)
            
            # Sort by criticality (highest first)
            sorted_indices = np.argsort(-criticality)
            
            # Optimize top 40% of circles
            num_optimize = max(1, int(n * 0.4))
            
            for i in range(min(num_optimize, n)):
                idx = sorted_indices[i]
                
                # Save current values
                old_x, old_y, old_r = result[idx]
                
                # Try small perturbations
                best_x, best_y, best_r = old_x, old_y, old_r
                best_fitness = evaluate_fitness(result)
                
                # Try several random moves
                for _ in range(20):
                    # Slightly modify position and radius
                    new_x = old_x + np.random.normal(0, 0.005)
                    new_y = old_y + np.random.normal(0, 0.005)
                    new_r = old_r + np.random.normal(0, 0.002)
                    
                    # Keep within bounds
                    new_x = np.clip(new_x, new_r, rect_width - new_r)
                    new_y = np.clip(new_y, new_r, rect_height - new_r)
                    new_r = max(0.001, new_r)
                    
                    # Test this modification
                    temp_result = result.copy()
                    temp_result[idx] = [new_x, new_y, new_r]
                    
                    if is_valid_solution(temp_result):
                        new_fitness = evaluate_fitness(temp_result)
                        if new_fitness > best_fitness:
                            best_fitness = new_fitness
                            best_x, best_y, best_r = new_x, new_y, new_r
                
                # Update if improvement was found
                if best_x != old_x or best_y != old_y or best_r != old_r:
                    result[idx] = [best_x, best_y, best_r]
            
            # Occasionally try to improve the overall solution with small global moves
            if iteration % 50 == 0 and iteration > 0:
                # Try to slightly adjust all circles to improve global fit
                for idx in range(n):
                    old_x, old_y, old_r = result[idx]
                    new_x = old_x + np.random.normal(0, 0.002)
                    new_y = old_y + np.random.normal(0, 0.002)
                    new_r = old_r + np.random.normal(0, 0.001)
                    
                    # Keep within bounds
                    new_x = np.clip(new_x, new_r, rect_width - new_r)
                    new_y = np.clip(new_y, new_r, rect_height - new_r)
                    new_r = max(0.001, new_r)
                    
                    # Test this modification
                    temp_result = result.copy()
                    temp_result[idx] = [new_x, new_y, new_r]
                    
                    if is_valid_solution(temp_result):
                        new_fitness = evaluate_fitness(temp_result)
                        if new_fitness > evaluate_fitness(result):
                            result = temp_result
        
        return result

    def boundary_refinement(circles: np.ndarray) -> np.ndarray:
        """Refine circles that are near boundaries"""
        result = circles.copy()
        
        # Process circles near boundaries
        for i in range(len(result)):
            x, y, r = result[i]
            
            # Check if near any boundary
            near_boundary = False
            boundary_distances = [
                x - r,  # Left
                y - r,  # Bottom  
                rect_width - x - r,  # Right
                rect_height - y - r   # Top
            ]
            
            for dist in boundary_distances:
                if dist < 0.02:  # Near boundary
                    near_boundary = True
                    break
            
            if near_boundary:
                # Try to slightly enlarge this circle if possible
                old_r = r
                test_r = min(old_r + 0.005, 0.2)  # Try to grow slightly
                
                # Keep within bounds
                test_x = np.clip(x, test_r, rect_width - test_r)
                test_y = np.clip(y, test_r, rect_height - test_y)
                
                temp_result = result.copy()
                temp_result[i] = [test_x, test_y, test_r]
                
                # If valid and improves fitness, keep it
                if is_valid_solution(temp_result):
                    new_fitness = evaluate_fitness(temp_result)
                    old_fitness = evaluate_fitness(result)
                    if new_fitness > old_fitness:
                        result = temp_result
                        
        return result

    # Main algorithm
    # Generate multiple initial solutions
    initial_patterns = generate_initial_patterns(rect_width, rect_height, n)
    
    # Evaluate all patterns and pick the best one
    best_pattern = None
    best_fitness = -float('inf')
    
    for pattern in initial_patterns:
        fitness = evaluate_fitness(pattern)
        if fitness > best_fitness:
            best_fitness = fitness
            best_pattern = pattern.copy()
    
    if best_pattern is None:
        # Fallback to simple grid
        best_pattern = generate_simple_grid_pattern(rect_width, rect_height, n)
    
    # Phase 1: Global optimization with criticality-guided refinement
    current_solution = best_pattern.copy()
    
    # Phase 2: Criticality-guided local search
    current_solution = optimize_critical_circles(current_solution, max_iterations=150)
    
    # Phase 3: Boundary refinement
    current_solution = boundary_refinement(current_solution)
    
    # Phase 4: Additional fine-tuning
    for _ in range(100):
        # Try optimizing all circles in a random order
        indices = list(range(len(current_solution)))
        random.shuffle(indices)
        
        for idx in indices[:10]:  # Optimize first 10 randomly selected circles
            old_x, old_y, old_r = current_solution[idx]
            
            # Small random perturbation
            new_x = old_x + np.random.normal(0, 0.003)
            new_y = old_y + np.random.normal(0, 0.003)
            new_r = old_r + np.random.normal(0, 0.001)
            
            # Keep within bounds
            new_x = np.clip(new_x, new_r, rect_width - new_r)
            new_y = np.clip(new_y, new_r, rect_height - new_r)
            new_r = max(0.001, new_r)
            
            temp_solution = current_solution.copy()
            temp_solution[idx] = [new_x, new_y, new_r]
            
            if is_valid_solution(temp_solution):
                new_fitness = evaluate_fitness(temp_solution)
                old_fitness = evaluate_fitness(current_solution)
                
                if new_fitness > old_fitness:
                    current_solution = temp_solution
    
    # Final validation and cleanup
    if not is_valid_solution(current_solution):
        # Revert to best valid pattern
        current_solution = best_pattern.copy()
        
    # Final boundary refinement
    current_solution = boundary_refinement(current_solution)
    
    return current_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")