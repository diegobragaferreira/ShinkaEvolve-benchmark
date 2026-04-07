# EVOLVE-BLOCK-START
import numpy as np
import random
import math
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import time
from copy import deepcopy
import heapq

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    
    Uses a grid-based optimization approach instead of evolutionary algorithms.
    Implements constraint-aware local search with simulated annealing.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    start_time = time.time()
    
    # Parameters
    n_circles = 26
    grid_resolution = 100  # High resolution grid for candidate positions
    max_attempts = 5000
    initial_temp = 1.0
    cooling_rate = 0.995
    min_temp = 0.001
    max_iter_per_temp = 100
    
    # Create grid of candidate positions
    grid_positions = []
    step = 1.0 / grid_resolution
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            x = (i + 0.5) * step
            y = (j + 0.5) * step
            # Only include positions that can accommodate reasonably sized circles
            if x >= 0.05 and x <= 0.95 and y >= 0.05 and y <= 0.95:
                grid_positions.append((x, y))
    
    # Precompute valid candidate combinations for efficient lookup
    candidates = []
    for i, (x, y) in enumerate(grid_positions):
        # For each position, find valid radius ranges based on boundary constraints
        max_r = min(x, 1-x, y, 1-y)
        if max_r > 0.01:  # Only keep positions that can support meaningful circles
            candidates.append((x, y, max_r, i))
    
    # Sort candidates by their maximum possible radius (descending)
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    # Greedy initialization with largest possible circles first
    circles = np.zeros((n_circles, 3))
    used_positions = set()
    
    def validate_position(x, y, r):
        """Check if a circle at (x,y) with radius r is valid"""
        return (0 <= x - r and x + r <= 1 and
                0 <= y - r and y + r <= 1)

    def check_overlap(x1, y1, r1, x2, y2, r2):
        """Check if two circles overlap"""
        dx = x1 - x2
        dy = y1 - y2
        distance_squared = dx*dx + dy*dy
        min_distance_squared = (r1 + r2) * (r1 + r2)
        return distance_squared < min_distance_squared

    def check_all_overlaps(positions, radii):
        """Efficiently check if any overlaps exist"""
        if len(positions) < 2:
            return False
        
        try:
            tree = cKDTree(positions)
            pairs = tree.query_pairs(0.0001, p=2)  # Very small threshold
            for i, j in pairs:
                if i < len(radii) and j < len(radii):
                    r_i = radii[i]
                    r_j = radii[j]
                    pos_i = positions[i]
                    pos_j = positions[j]
                    dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                    if dist < (r_i + r_j):
                        return True
        except:
            # Fall back to brute force
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    pos_i = positions[i]
                    pos_j = positions[j]
                    r_i = radii[i]
                    r_j = radii[j]
                    dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                    if dist < (r_i + r_j):
                        return True
        return False

    # Initial greedy placement
    for i in range(n_circles):
        best_x, best_y, best_r = 0.5, 0.5, 0.05
        best_score = 0
        
        # Try a subset of candidates for efficiency
        sample_size = min(200, len(candidates))
        sampled_candidates = random.sample(candidates, sample_size) if sample_size < len(candidates) else candidates
        
        for x, y, max_r, idx in sampled_candidates:
            if idx in used_positions:
                continue
                
            # Try to place with maximum possible radius
            r = min(max_r, 0.15)  # Cap at reasonable size
            
            # Check if this placement creates overlaps
            temp_positions = circles[:i, :2].tolist() + [[x, y]]
            temp_radii = circles[:i, 2].tolist() + [r]
            
            if not check_all_overlaps(temp_positions, temp_radii):
                # Score based on radius
                score = r
                if score > best_score:
                    best_score = score
                    best_x, best_y, best_r = x, y, r
                    
        if best_score > 0:
            circles[i] = [best_x, best_y, best_r]
            used_positions.add(next(idx for x, y, max_r, idx in candidates if abs(x-best_x)<0.001 and abs(y-best_y)<0.001))
        else:
            # Fallback to simple placement
            circles[i] = [0.5, 0.5, 0.05]
    
    # Ensure all circles satisfy constraints
    positions = circles[:, :2]
    radii = circles[:, 2]
    
    # Fix containment issues
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Adjust to stay within bounds
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        circles[i] = [x, y, r]
    
    # Simulated Annealing optimization
    current_circles = circles.copy()
    current_positions = current_circles[:, :2]
    current_radii = current_circles[:, 2]
    current_total = np.sum(current_radii)
    
    temperature = initial_temp
    iteration = 0
    
    try:
        while temperature > min_temp and iteration < max_attempts:
            for iter_in_temp in range(max_iter_per_temp):
                # Make a random modification
                modified_circles = current_circles.copy()
                modified_positions = modified_circles[:, :2]
                modified_radii = modified_circles[:, 2]
                
                # Choose random circle to modify
                circle_idx = random.randint(0, n_circles - 1)
                
                # Randomly decide what to change: position, radius, or both
                change_type = random.choice(['position', 'radius', 'both'])
                
                if change_type in ['position', 'both']:
                    # Perturb position
                    delta_x = random.uniform(-0.02, 0.02)
                    delta_y = random.uniform(-0.02, 0.02)
                    new_x = max(0.01, min(0.99, modified_positions[circle_idx][0] + delta_x))
                    new_y = max(0.01, min(0.99, modified_positions[circle_idx][1] + delta_y))
                    modified_positions[circle_idx] = [new_x, new_y]
                
                if change_type in ['radius', 'both']:
                    # Perturb radius
                    delta_r = random.uniform(-0.01, 0.01)
                    new_r = max(0.001, min(0.3, modified_radii[circle_idx] + delta_r))
                    modified_radii[circle_idx] = new_r
                
                # Check constraints
                positions_valid = True
                radii_valid = True
                
                # Check containment
                for i in range(len(modified_circles)):
                    x, y, r = modified_circles[i]
                    if not validate_position(x, y, r):
                        positions_valid = False
                        break
                
                # Check overlaps
                overlaps_exist = check_all_overlaps(modified_positions, modified_radii)
                
                if positions_valid and not overlaps_exist:
                    # Accept the modification
                    new_total = np.sum(modified_radii)
                    if new_total > current_total:
                        current_circles = modified_circles
                        current_total = new_total
                    else:
                        # Accept with probability based on temperature
                        delta_energy = new_total - current_total
                        if random.random() < math.exp(delta_energy / temperature):
                            current_circles = modified_circles
                            current_total = new_total
                
                iteration += 1
                if iteration >= max_attempts:
                    break
            
            # Cool down
            temperature *= cooling_rate
            
            if iteration >= max_attempts:
                break
    
    except Exception as e:
        print(f"Annealing error: {e}")
        pass
    
    # Final local optimization - more aggressive refinement
    final_circles = current_circles.copy()
    positions = final_circles[:, :2]
    radii = final_circles[:, 2]
    
    # Multi-phase local optimization
    for phase in range(3):
        improved = True
        phase_attempts = 0
        
        while improved and phase_attempts < 200:
            improved = False
            phase_attempts += 1
            
            # Try to improve each circle independently
            for i in range(len(final_circles)):
                original_pos = final_circles[i, :2].copy()
                original_r = final_circles[i, 2]
                
                # Try to increase radius
                max_increase = min(
                    final_circles[i, 0], 1 - final_circles[i, 0],
                    final_circles[i, 1], 1 - final_circles[i, 1]
                ) - original_r
                
                if max_increase > 0:
                    # Binary search for maximum safe increase
                    low = 0
                    high = max_increase
                    best_radius = original_r
                    
                    for _ in range(10):
                        test_r = (low + high) / 2
                        test_r = min(test_r, max_increase)
                        
                        # Check if this change is feasible
                        valid = True
                        test_pos = final_circles[i, :2]
                        test_r_new = original_r + test_r
                        
                        # Check overlap with other circles
                        for j in range(len(final_circles)):
                            if i != j:
                                pos_j = final_circles[j, :2]
                                r_j = final_circles[j, 2]
                                dist = np.sqrt(np.sum((test_pos - pos_j)**2))
                                if dist < (test_r_new + r_j):
                                    valid = False
                                    break
                        
                        if valid:
                            best_radius = original_r + test_r
                            low = test_r
                        else:
                            high = test_r
                    
                    if best_radius > original_r:
                        final_circles[i, 2] = best_radius
                        improved = True
                
                # Try to improve position
                best_pos = original_pos.copy()
                best_radius = final_circles[i, 2]
                best_score = best_radius
                
                # Try nearby positions
                for dx in [-0.02, -0.01, 0, 0.01, 0.02]:
                    for dy in [-0.02, -0.01, 0, 0.01, 0.02]:
                        test_x = max(0.01, min(0.99, final_circles[i, 0] + dx))
                        test_y = max(0.01, min(0.99, final_circles[i, 1] + dy))
                        
                        # Test if this position is valid
                        valid = True
                        test_r = final_circles[i, 2]
                        
                        # Check overlap with other circles
                        for j in range(len(final_circles)):
                            if i != j:
                                pos_j = final_circles[j, :2]
                                r_j = final_circles[j, 2]
                                dist = np.sqrt((test_x - pos_j[0])**2 + (test_y - pos_j[1])**2)
                                if dist < (test_r + r_j):
                                    valid = False
                                    break
                        
                        if valid:
                            score = test_r  # Just maximize radius for now
                            if score > best_score:
                                best_score = score
                                best_pos = [test_x, test_y]
                
                # Apply best improvement
                if best_score > final_circles[i, 2] or not np.array_equal(best_pos, original_pos):
                    final_circles[i, :2] = best_pos
                    final_circles[i, 2] = best_score
                    improved = True
            
            if not improved:
                break
    
    # Ensure final constraints
    for i in range(len(final_circles)):
        x, y, r = final_circles[i]
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        final_circles[i] = [x, y, r]
    
    end_time = time.time()
    sum_radii = np.sum(final_circles[:, 2])
    benchmark_ratio = sum_radii / 2.6358627564136983
    
    print(f"Total evaluation time: {end_time - start_time:.2f}s")
    print(f"Sum of radii: {sum_radii:.6f}")
    print(f"Benchmark ratio: {benchmark_ratio:.6f}")
    
    return final_circles

# EVOLVE-BLOCK-END