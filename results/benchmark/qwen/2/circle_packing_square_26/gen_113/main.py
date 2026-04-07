# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.spatial.distance import cdist
import heapq

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a greedy insertion algorithm with local optimization.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    
    # Initialize parameters
    n_circles = 26
    circles = np.zeros((n_circles, 3))
    
    # Sort candidate centers by priority (distance from edges matters)
    # We'll generate multiple candidate positions for each circle
    candidates_per_circle = 50
    
    # Helper function to check if a circle fits
    def is_valid_placement(x, y, r, existing_circles):
        # Check containment
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
        
        # Check overlap with existing circles
        for ox, oy, oradius in existing_circles:
            distance = math.sqrt((x - ox)**2 + (y - oy)**2)
            if distance < r + oradius:
                return False
        return True
    
    # Generate candidate positions for greedy insertion
    def generate_candidates(center_x, center_y, max_radius, existing_circles, count=10):
        candidates = []
        # Generate positions around the given center with various radii
        for i in range(count):
            # Randomize angle and distance from center
            angle = np.random.random() * 2 * np.pi
            distance = np.random.random() * max_radius * 0.8
            
            # Calculate candidate position
            x = center_x + distance * math.cos(angle)
            y = center_y + distance * math.sin(angle)
            
            # Try different radius values
            r = min(max_radius, 0.3 + np.random.random() * 0.3)  # Reasonable radius range
            
            # Check if valid
            if is_valid_placement(x, y, r, existing_circles):
                candidates.append((x, y, r))
        
        return candidates
    
    # Greedy insertion with backtracking
    inserted_circles = []
    
    # Strategy: place large circles first, then smaller ones
    # Start with a few large circles to establish structure
    
    # Initial attempt: place circles one by one using greedy approach
    # First, create a list of all potential placements (heuristic based)
    
    # Pre-generate a large pool of candidate positions
    pool_size = 1000
    candidate_pool = []
    
    # Fill candidate pool with various positions and radii
    for _ in range(pool_size):
        # Place near center with some randomness
        x = 0.5 + (np.random.random() - 0.5) * 0.8
        y = 0.5 + (np.random.random() - 0.5) * 0.8
        r = 0.05 + np.random.random() * 0.2  # Reasonable initial radii
        
        # Ensure reasonable bounds
        if x - r >= 0 and x + r <= 1 and y - r >= 0 and y + r <= 1:
            # Check if it doesn't overlap significantly with any existing
            candidate_pool.append((x, y, r))
    
    # Sort candidates by their potential value (larger radius preferred if valid)
    candidate_pool.sort(key=lambda c: c[2], reverse=True)
    
    # Insert circles greedily with smart positioning
    for i in range(n_circles):
        best_candidate = None
        best_score = -float('inf')
        
        # Try several candidates from our pool
        candidates_to_try = min(100, len(candidate_pool))
        
        # Sample candidates more intelligently
        sample_indices = list(range(min(100, len(candidate_pool))))
        np.random.shuffle(sample_indices)
        sample_indices = sample_indices[:candidates_to_try]
        
        for idx in sample_indices:
            x, y, r = candidate_pool[idx]
            
            # Check if this placement works with current insertion state
            if is_valid_placement(x, y, r, inserted_circles):
                # Score based on radius (higher is better) and potentially position quality
                # We're using a greedy approach: pick largest valid circle
                score = r
                
                if score > best_score:
                    best_score = score
                    best_candidate = (x, y, r)
        
        # If no valid candidate found, try to find a fallback
        if best_candidate is None:
            # Find a valid circle from remaining pool
            for x, y, r in candidate_pool:
                if is_valid_placement(x, y, r, inserted_circles):
                    best_candidate = (x, y, r)
                    break
        
        # If still no valid candidate, use default position
        if best_candidate is None:
            # Place at center with small radius
            best_candidate = (0.5, 0.5, 0.01)
        
        inserted_circles.append(best_candidate)
        
        # Remove this candidate from pool to encourage diversity
        try:
            candidate_pool.remove(best_candidate)
        except ValueError:
            pass  # Already removed or not in pool
    
    # Local optimization: try to improve the solution
    optimized_circles = optimize_solution(inserted_circles)
    
    # Final validation and cleanup
    final_circles = []
    for x, y, r in optimized_circles:
        # Ensure valid positions
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        final_circles.append([x, y, r])
    
    return np.array(final_circles)

def optimize_solution(original_circles):
    """Apply local optimization to improve solution quality"""
    circles = [list(circle) for circle in original_circles]
    n = len(circles)
    
    # Try improving each circle's position and radius
    improvement_count = 0
    max_iterations = 50
    
    for iteration in range(max_iterations):
        improved = False
        
        # Try moving each circle to a better position
        for i in range(n):
            original_x, original_y, original_r = circles[i]
            
            # Try nearby positions
            best_x, best_y, best_r = original_x, original_y, original_r
            best_total_radius = sum(c[2] for c in circles)
            
            # Test small movements
            steps = [-0.01, -0.005, 0, 0.005, 0.01]
            
            for dx in steps:
                for dy in steps:
                    # Only try positions that don't change radius too much
                    new_x = original_x + dx
                    new_y = original_y + dy
                    
                    # Check boundary constraints
                    if new_x - original_r >= 0 and new_x + original_r <= 1 and \
                       new_y - original_r >= 0 and new_y + original_r <= 1:
                        
                        # Check overlaps
                        valid = True
                        temp_circles = circles[:]
                        temp_circles[i] = [new_x, new_y, original_r]
                        
                        for j in range(n):
                            if i != j:
                                x1, y1, r1 = temp_circles[i]
                                x2, y2, r2 = temp_circles[j]
                                dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                                if dist < r1 + r2:
                                    valid = False
                                    break
                        
                        if valid:
                            # Measure improvement
                            new_total = sum(c[2] for c in temp_circles)
                            if new_total > best_total_radius:
                                best_total_radius = new_total
                                best_x, best_y, best_r = new_x, new_y, original_r
                                improved = True
        
        if improved:
            circles[i] = [best_x, best_y, best_r]
            improvement_count += 1
        else:
            continue
    
    return circles

# EVOLVE-BLOCK-END