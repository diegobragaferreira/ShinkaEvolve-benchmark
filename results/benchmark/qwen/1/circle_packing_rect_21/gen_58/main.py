# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import random
from typing import Tuple, List
import time

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    width, height = 1.0, 1.0
    
    def is_valid_position(x: float, y: float, r: float) -> bool:
        """Check if circle is within bounds"""
        return (r <= x <= width - r) and (r <= y <= height - r)
    
    def compute_distances(circles_array: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between circle centers"""
        centers = circles_array[:, :2]
        return cdist(centers, centers)
    
    def check_collisions(circles_array: np.ndarray) -> bool:
        """Fast collision detection using spatial indexing"""
        if len(circles_array) < 2:
            return False
            
        # Build KDTree for efficient nearest neighbor queries
        centers = circles_array[:, :2]
        tree = cKDTree(centers)
        
        # Query for neighbors within distance of (r1 + r2)
        radii = circles_array[:, 2]
        max_distance = radii[:, None] + radii[None, :]
        
        # Use tree query_pairs for efficient collision detection
        pairs = tree.query_pairs(max_distance, output_type='ndarray')
        
        # Check if any pair has overlapping circles
        for i, j in pairs:
            if i < j:
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist < (radii[i] + radii[j]):
                    return True
        return False
    
    def compute_total_radius(circles_array: np.ndarray) -> float:
        """Compute total sum of radii"""
        return np.sum(circles_array[:, 2])
    
    def initialize_hexagonal_layout(width: float, height: float, n_circles: int) -> np.ndarray:
        """Initialize circles using a hexagonal packing approach"""
        # Create a hexagonal grid of points
        rows = int(np.ceil(np.sqrt(n_circles)))
        cols = int(np.ceil(n_circles / rows))
        
        # Adjust grid to fit within rectangle
        padding = 0.05
        grid_width = width - 2 * padding
        grid_height = height - 2 * padding
        
        spacing_x = grid_width / (cols + 1)
        spacing_y = grid_height / (rows + 1)
        
        # Generate initial positions
        circles = np.zeros((n_circles, 3))
        idx = 0
        
        for i in range(rows):
            for j in range(cols):
                if idx >= n_circles:
                    break
                x = padding + (j + 1) * spacing_x
                y = padding + (i + 1) * spacing_y
                
                # Add hexagonal offset for odd rows
                if i % 2 == 1:
                    x += spacing_x / 2
                
                circles[idx] = [x, y, 0.0]
                idx += 1
            if idx >= n_circles:
                break
        
        # Distribute initial radii with some randomness
        base_radius = min(width, height) * 0.05
        for i in range(n_circles):
            # Start with base radius
            circles[i, 2] = base_radius
            
            # Apply small random adjustments
            adjustment = np.random.uniform(0.8, 1.2)
            circles[i, 2] *= adjustment
            
            # Enforce boundary constraints
            x, y, r = circles[i]
            r = min(r, x, width - x, y, height - y)
            circles[i, 2] = max(0.001, r)
            
        return circles
    
    def local_optimization_step(circles_array: np.ndarray, max_iterations: int = 50) -> np.ndarray:
        """Perform local optimization focusing on improving radii while maintaining constraints"""
        circles = circles_array.copy()
        
        for iteration in range(max_iterations):
            improved = False
            
            # For each circle, try to increase its radius
            for i in range(len(circles)):
                x, y, old_radius = circles[i]
                
                # Try to increase radius while respecting constraints
                max_radius = min(x, width - x, y, height - y)
                if max_radius <= old_radius:
                    continue
                    
                # Binary search for maximum radius
                left, right = old_radius, max_radius
                best_radius = old_radius
                
                # Test different radius values
                test_radii = np.linspace(left, right, 20)[1:-1]  # Skip endpoints
                
                for test_radius in test_radii:
                    # Temporarily update this circle's radius
                    circles[i, 2] = test_radius
                    
                    # Check constraints
                    if is_valid_position(x, y, test_radius) and not check_collisions(circles):
                        best_radius = test_radius
                        improved = True
                    else:
                        # Restore previous value
                        circles[i, 2] = old_radius
                        break  # No point testing larger values if this one doesn't work
                        
                # Update the radius if improvement was found
                if best_radius > old_radius:
                    circles[i, 2] = best_radius
            
            # If no improvement was made, stop early
            if not improved:
                break
                
        return circles
    
    def global_refinement_step(circles_array: np.ndarray) -> np.ndarray:
        """Apply global refinement to improve overall configuration"""
        circles = circles_array.copy()
        
        # First, try to expand all circles as much as possible
        circles = local_optimization_step(circles, max_iterations=20)
        
        # Then, apply some positional adjustments
        for _ in range(10):
            # Try moving circles to improve packing
            for i in range(len(circles)):
                x, y, r = circles[i]
                
                # Small random perturbations
                dx = np.random.uniform(-0.01, 0.01)
                dy = np.random.uniform(-0.01, 0.01)
                
                new_x = x + dx
                new_y = y + dy
                
                # Check if new position is valid
                if is_valid_position(new_x, new_y, r):
                    # Temporarily move circle
                    circles[i, 0] = new_x
                    circles[i, 1] = new_y
                    
                    # Check if collision was resolved
                    if not check_collisions(circles):
                        continue  # Keep the move
                    else:
                        # Restore original position
                        circles[i, 0] = x
                        circles[i, 1] = y
                else:
                    # Restore original position if out of bounds
                    circles[i, 0] = x
                    circles[i, 1] = y
                    
        return circles
    
    def generate_candidate_solutions() -> List[np.ndarray]:
        """Generate multiple candidate solutions using different initialization approaches"""
        candidates = []
        
        # Candidate 1: Standard hexagonal packing
        cand1 = initialize_hexagonal_layout(width, height, 21)
        candidates.append(cand1)
        
        # Candidate 2: Perturbed hexagonal packing
        cand2 = initialize_hexagonal_layout(width, height, 21)
        for i in range(len(cand2)):
            # Add some noise to positions
            cand2[i, 0] += np.random.uniform(-0.02, 0.02)
            cand2[i, 1] += np.random.uniform(-0.02, 0.02)
            # Keep original radius
        candidates.append(cand2)
        
        # Candidate 3: Random valid initialization
        cand3 = np.zeros((21, 3))
        for i in range(21):
            attempt = 0
            while attempt < 100:
                x = np.random.uniform(0.01, width - 0.01)
                y = np.random.uniform(0.01, height - 0.01)
                r = np.random.uniform(0.01, min(x, width - x, y, height - y) * 0.8)
                
                # Check if valid
                if is_valid_position(x, y, r):
                    cand3[i] = [x, y, r]
                    break
                attempt += 1
        candidates.append(cand3)
        
        return candidates
    
    # Generate initial candidates
    candidates = generate_candidate_solutions()
    
    best_solution = None
    best_radius_sum = -1
    
    # Evaluate each candidate
    for candidate in candidates:
        # Apply local optimization to each candidate
        optimized = local_optimization_step(candidate, max_iterations=30)
        
        # Apply global refinement
        refined = global_refinement_step(optimized)
        
        # Check if this is better than our current best
        radius_sum = compute_total_radius(refined)
        if radius_sum > best_radius_sum:
            best_radius_sum = radius_sum
            best_solution = refined.copy()
    
    # Final enhancement with more intensive local search
    if best_solution is not None:
        # Run a more intensive optimization pass
        final_solution = local_optimization_step(best_solution, max_iterations=50)
        final_solution = global_refinement_step(final_solution)
        
        # Final validation
        if not check_collisions(final_solution):
            best_solution = final_solution
    
    # If we still have no solution, fall back to a simple approach
    if best_solution is None:
        # Initialize with evenly spaced small circles
        best_solution = np.zeros((21, 3))
        base_radius = 0.05
        grid_size = 5
        spacing_x = width / (grid_size + 1)
        spacing_y = height / (grid_size + 1)
        
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= 21:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                r = min(base_radius, min(x, width-x), min(y, height-y))
                if r > 0:
                    best_solution[idx] = [x, y, r]
                    idx += 1
    
    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
