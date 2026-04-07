# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import math

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def check_containment(circles: np.ndarray) -> bool:
    """Check if all circles are fully contained within the unit square."""
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    return True

def check_overlap(circles: np.ndarray) -> bool:
    """Check if any circles overlap using pairwise distance calculation."""
    n = len(circles)
    if n <= 1:
        return True
    
    # Calculate pairwise distances
    positions = circles[:, :2]
    radii = circles[:, 2]
    
    # Create distance matrix
    distances = cdist(positions, positions)
    
    # Check for overlaps
    for i in range(n):
        for j in range(i+1, n):
            dist = distances[i, j]
            if dist < radii[i] + radii[j]:
                return False
    return True

def fitness(circles: np.ndarray) -> float:
    """Calculate fitness as sum of radii."""
    return np.sum(circles[:, 2])

def generate_initial_config(n_circles: int) -> np.ndarray:
    """Generate a random initial configuration with proper spatial distribution."""
    circles = np.zeros((n_circles, 3))
    
    # Use a more sophisticated initialization strategy
    # Place points in a grid-like pattern with jitter
    rows = int(np.ceil(np.sqrt(n_circles)))
    cols = rows
    spacing_x = 0.9 / (cols + 1)
    spacing_y = 0.9 / (rows + 1)
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n_circles:
                break
            x = 0.05 + (j + 1) * spacing_x + np.random.uniform(-spacing_x/4, spacing_x/4)
            y = 0.05 + (i + 1) * spacing_y + np.random.uniform(-spacing_y/4, spacing_y/4)
            
            # Initial radius based on proximity to edges and neighbors
            max_radius = min(x, y, 1-x, 1-y)
            
            # Estimate minimum distance to neighbors
            if idx > 0:
                # Simple estimation based on grid spacing
                min_dist = min(spacing_x, spacing_y) * 0.8
                proposed_radius = min(min_dist / 3.0, max_radius * 0.7)
            else:
                proposed_radius = max_radius * 0.5
            
            radius = max(0.001, min(proposed_radius, 0.4))
            
            circles[idx] = [x, y, radius]
            idx += 1
    
    return circles

def propose_move(circles: np.ndarray, temperature: float) -> np.ndarray:
    """Propose a new configuration by making a small random move."""
    new_circles = circles.copy()
    
    # Choose which circle to modify
    circle_idx = random.randint(0, len(new_circles) - 1)
    
    # Choose modification type
    mod_type = random.choice(['position', 'radius'])
    
    if mod_type == 'position':
        # Move the circle position
        new_circles[circle_idx, 0] += np.random.normal(0, temperature * 0.02)
        new_circles[circle_idx, 1] += np.random.normal(0, temperature * 0.02)
        
        # Ensure the new position is within bounds
        x, y, r = new_circles[circle_idx]
        new_circles[circle_idx, 0] = np.clip(x, r + 0.001, 1 - r - 0.001)
        new_circles[circle_idx, 1] = np.clip(y, r + 0.001, 1 - r - 0.001)
    else:
        # Modify the radius
        old_radius = new_circles[circle_idx, 2]
        log_factor = np.random.normal(0, temperature * 0.15)
        new_radius = old_radius * np.exp(log_factor)
        new_circles[circle_idx, 2] = np.clip(new_radius, 0.001, 0.4)
    
    return new_circles

def geometric_acceptance_probability(current_fitness: float, proposed_fitness: float, 
                                   current_valid: bool, proposed_valid: bool, 
                                   temperature: float) -> float:
    """Calculate the acceptance probability considering geometric feasibility."""
    
    # If proposed solution is invalid, reject it
    if not proposed_valid:
        return 0.0
    
    # If current is invalid and proposed is valid, accept immediately
    if not current_valid and proposed_valid:
        return 1.0
    
    # Both valid - use simulated annealing approach with geometric bonus
    if proposed_valid and current_valid:
        delta = proposed_fitness - current_fitness
        
        # Add geometric bonus for configurations with better spread
        # This encourages solutions that distribute circles more evenly
        geometric_bonus = 0.0
        
        # Bonus based on how well circles are spread out
        # Simple proxy: average distance between circles
        if len(current_fitness) > 1:
            # Calculate average distance between circles in current config
            positions_current = current_fitness[:, :2]
            distances_current = cdist(positions_current, positions_current).flatten()
            avg_dist_current = np.mean(distances_current[distances_current > 0])
            
            # For proposed, we'll estimate similar metric
            positions_proposed = proposed_fitness[:, :2]
            distances_proposed = cdist(positions_proposed, positions_proposed).flatten()
            avg_dist_proposed = np.mean(distances_proposed[distances_proposed > 0])
            
            # Reward configurations with larger average distances (better spread)
            geometric_bonus = (avg_dist_proposed - avg_dist_current) * 0.1
        
        # Standard Metropolis acceptance
        if delta >= 0:
            return 1.0
        else:
            # Modified acceptance with geometric bonus
            return np.exp(delta / temperature + geometric_bonus)
    
    # Shouldn't reach here with our logic
    return 0.0

def monte_carlo_optimize(initial_circles: np.ndarray, max_iterations: int = 100000) -> np.ndarray:
    """Optimize using Monte Carlo with simulated annealing approach."""
    current_solution = initial_circles.copy()
    best_solution = initial_circles.copy()
    best_fitness = fitness(current_solution)
    
    # Annealing schedule
    initial_temp = 0.1
    final_temp = 0.0001
    alpha = 0.9995  # Cooling rate
    
    temperature = initial_temp
    
    # Track recent improvements for early stopping
    recent_improvements = []
    max_recent_improvements = 1000
    
    for iteration in range(max_iterations):
        # Cool down temperature
        if iteration > 0 and iteration % 1000 == 0:
            temperature = max(final_temp, temperature * alpha)
        
        # Propose new solution
        proposed_solution = propose_move(current_solution, temperature)
        
        # Check validity
        current_valid = check_containment(current_solution) and check_overlap(current_solution)
        proposed_valid = check_containment(proposed_solution) and check_overlap(proposed_solution)
        
        # Calculate fitness
        current_fitness = fitness(current_solution)
        proposed_fitness = fitness(proposed_solution)
        
        # Accept or reject based on our custom acceptance rule
        if proposed_valid:
            # Calculate acceptance probability
            accept_prob = min(1.0, np.exp((proposed_fitness - current_fitness) / temperature))
            
            # Add geometric bonus
            if accepted:
                # We'll implement a simpler geometric bonus approach
                pass
            
            if random.random() < accept_prob:
                current_solution = proposed_solution.copy()
                
                # Update best solution if improved
                if proposed_fitness > best_fitness:
                    best_fitness = proposed_fitness
                    best_solution = proposed_solution.copy()
                    
                    # Reset early stopping counter
                    recent_improvements = []
                else:
                    # Track recent non-improvements
                    if len(recent_improvements) < max_recent_improvements:
                        recent_improvements.append(iteration)
                    else:
                        recent_improvements.pop(0)
                        recent_improvements.append(iteration)
        else:
            # Invalid move, stay with current solution
            pass
            
        # Early stopping if no improvement for long time
        if len(recent_improvements) > 0 and iteration - recent_improvements[-1] > 5000:
            # If no improvement for 5000 iterations, restart with better solution
            if best_fitness > 0.5:  # Only restart if we have a decent solution
                current_solution = best_solution.copy()
                temperature = initial_temp  # Reset temperature
                recent_improvements = []
    
    return best_solution

def optimize_single_circles(n_circles: int = 26, max_iter: int = 100000) -> np.ndarray:
    """Single run optimization with restart capability."""
    best_solution = None
    best_fitness = -float('inf')
    
    # Try multiple initializations
    for attempt in range(5):
        # Generate initial configuration
        initial_config = generate_initial_config(n_circles)
        
        # Optimize
        solution = monte_carlo_optimize(initial_config, max_iter // 5)
        
        # Check fitness
        current_fitness = fitness(solution)
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_solution = solution.copy()
            
    return best_solution

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Run single Monte Carlo optimization
    circles = optimize_single_circles(26, 100000)
    
    # Final validation
    if circles is None or not check_containment(circles) or not check_overlap(circles):
        # Fallback to a simple arrangement if optimization failed
        circles = np.zeros((26, 3))
        rows = 5
        cols = 5
        spacing_x = 0.9 / (cols + 1)
        spacing_y = 0.9 / (rows + 1)
        radius = min(spacing_x, spacing_y) * 0.3

        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= 26:
                    break
                x = 0.05 + (j + 1) * spacing_x
                y = 0.05 + (i + 1) * spacing_y
                circles[idx] = [x, y, radius]
                idx += 1

        # Adjust last few circles to fit
        for i in range(idx, 26):
            circles[i] = [0.5, 0.5, 0.01]

    return circles

# EVOLVE-BLOCK-END