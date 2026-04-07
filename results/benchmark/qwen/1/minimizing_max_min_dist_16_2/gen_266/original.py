# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Phase 1: Initialize points using enhanced hexagonal grid pattern
    n = 16
    points = np.zeros((n, 2))
    
    # Create enhanced hexagonal grid pattern with better distribution
    rows = 4
    cols = 4
    spacing = 0.25
    
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx < n:
                # Offset every other row for hexagonal packing
                x = col * spacing + (row % 2) * spacing * 0.5
                y = row * spacing * math.sqrt(3) / 2
                points[idx] = [x, y]
                idx += 1
    
    # Scale and shift points to fit within [0.1, 0.9] x [0.1, 0.9] with some initial randomness
    points[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min()) * 0.8 + 0.1
    points[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min()) * 0.8 + 0.1
    
    # Phase 2: Enhanced Simulated Annealing optimization with adaptive cooling
    def compute_energy_and_ratio(points):
        # Compute pairwise distances
        distances = squareform(pdist(points))
        
        # Mask diagonal elements (distance to self is 0)
        np.fill_diagonal(distances, np.inf)
        
        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            ratio = 0
        else:
            ratio = min_dist / max_dist
            
        # Energy is negative ratio (we want to maximize ratio, so minimize negative ratio)
        # Add penalty for points outside bounds with epsilon padding
        penalty = 0
        epsilon = 1e-8
        for pt in points:
            if pt[0] < 0+epsilon or pt[0] > 1-epsilon or pt[1] < 0+epsilon or pt[1] > 1-epsilon:
                penalty += 1000
        
        return -ratio + penalty, ratio
    
    # Simulated Annealing parameters with intelligent adaptation
    temp = 1.0
    min_temp = 1e-8
    cooling_rate = 0.9999
    max_iter = 100000  # Increased iteration limit
    
    current_energy, current_ratio = compute_energy_and_ratio(points)
    best_points = points.copy()
    best_ratio = current_ratio
    
    # Track convergence for adaptive cooling
    last_improvement = 0
    patience = 2000
    improvement_count = 0
    recent_improvements = []
    
    # Main optimization loop
    for iteration in range(max_iter):
        # Generate neighbor solution (random perturbation)
        neighbor_points = points.copy()
        # Pick a random point to move
        move_idx = np.random.randint(0, n)
        # Small random displacement with adaptive magnitude based on temperature
        displacement_magnitude = 0.005 * temp
        displacement = np.random.normal(0, displacement_magnitude, 2)
        neighbor_points[move_idx] += displacement
        
        # Apply boundary constraints with epsilon padding
        epsilon = 1e-8
        neighbor_points[move_idx, 0] = np.clip(neighbor_points[move_idx, 0], 0+epsilon, 1-epsilon)
        neighbor_points[move_idx, 1] = np.clip(neighbor_points[move_idx, 1], 0+epsilon, 1-epsilon)
        
        # Compute energy of neighbor
        neighbor_energy, neighbor_ratio = compute_energy_and_ratio(neighbor_points)
        
        # Accept or reject move
        if neighbor_energy < current_energy:
            # Always accept better solutions
            points = neighbor_points
            current_energy = neighbor_energy
            current_ratio = neighbor_ratio
        else:
            # Accept worse solutions with probability based on temperature
            delta = neighbor_energy - current_energy
            if np.random.rand() < np.exp(-delta / temp):
                points = neighbor_points
                current_energy = neighbor_energy
                current_ratio = neighbor_ratio
        
        # Update best solution
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_points = points.copy()
            last_improvement = iteration
            improvement_count += 1
            recent_improvements.append(iteration)
            if len(recent_improvements) > 10:
                recent_improvements.pop(0)
        
        # Adaptive cooling schedule
        temp *= cooling_rate
        
        # Gradually cool down more aggressively if not improving
        if iteration % 1000 == 0 and temp > min_temp:
            temp *= 0.98
        
        # Early stopping based on patience and convergence
        if iteration - last_improvement > patience:
            # Check if recent improvements have stalled
            if len(recent_improvements) >= 5:
                if recent_improvements[-1] - recent_improvements[0] < 500:
                    break
            else:
                break
            
        # Very aggressive early stopping if barely improving
        if temp < min_temp * 10 and improvement_count < 5:
            break
    
    return best_points

# EVOLVE-BLOCK-END
