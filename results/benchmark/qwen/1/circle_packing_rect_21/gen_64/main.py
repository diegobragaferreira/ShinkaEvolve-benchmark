# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
import random
import time
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import math
from typing import Tuple, List

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def is_valid_solution(circles: np.ndarray, container_width: float = 1.0, container_height: float = 1.0) -> bool:
    """Check if all circles are within bounds and non-overlapping using spatial hashing for efficiency."""
    n = len(circles)
    
    # Check if all circles are within container bounds
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > container_width or y - r < 0 or y + r > container_height:
            return False
    
    # For small n, brute force is fast enough and simpler
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            
            # Distance between centers
            dx = x1 - x2
            dy = y1 - y2
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Circles overlap if distance < sum of radii
            if distance < r1 + r2:
                return False
    
    return True

def calculate_voronoi_criticality(circles: np.ndarray, container_width: float = 1.0, container_height: float = 1.0) -> np.ndarray:
    """Calculate criticality score for each circle based on Voronoi cell area."""
    n = len(circles)
    
    # Add boundary points to ensure proper Voronoi diagram
    boundary_points = np.array([
        [-1, -1], [-1, container_height + 1], [container_width + 1, -1], [container_width + 1, container_height + 1],
        [container_width/2, -1], [container_width/2, container_height + 1],
        [-1, container_height/2], [container_width + 1, container_height/2]
    ])
    
    # Extract circle centers
    centers = circles[:, :2]
    all_points = np.vstack([centers, boundary_points])
    
    try:
        vor = Voronoi(all_points)
        
        # Get Voronoi regions for original points (not boundary points)
        voronoi_areas = []
        for i in range(n):
            region_indices = np.where(vor.point_region[:-4] == i)[0]  # Exclude boundary regions
            if len(region_indices) > 0:
                # Calculate area of Voronoi cell
                region = vor.regions[vor.point_region[i]]
                if -1 not in region and len(region) > 2:
                    polygon_points = vor.vertices[region]
                    # Calculate polygon area
                    area = 0.5 * abs(sum(polygon_points[j][0] * polygon_points[(j+1)%len(polygon_points)][1] 
                                       - polygon_points[(j+1)%len(polygon_points)][0] * polygon_points[j][1] 
                                       for j in range(len(polygon_points))))
                    voronoi_areas.append(area)
                else:
                    voronoi_areas.append(1000.0)  # Large area for invalid regions
            else:
                voronoi_areas.append(1000.0)  # Default large area
                
        # Normalize criticality scores (smaller areas = higher criticality)
        if len(voronoi_areas) > 0:
            max_area = max(voronoi_areas)
            if max_area > 0:
                criticality_scores = [1.0 - (area/max_area) for area in voronoi_areas]
            else:
                criticality_scores = [0.0] * n
        else:
            criticality_scores = [0.0] * n
            
    except:
        # Fallback if Voronoi fails
        criticality_scores = [0.0] * n
    
    return np.array(criticality_scores)

def evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate fitness as sum of radii (since we want to maximize)"""
    return np.sum(circles[:, 2])

def generate_hexagonal_packing(n_circles: int, width: float = 1.0, height: float = 1.0) -> np.ndarray:
    """Generate an approximate hexagonal packing for initial configuration."""
    circles = np.zeros((n_circles, 3))
    
    # Calculate approximate circle radius - using a more sophisticated approach
    max_radius = min(width, height) * 0.15
    spacing_factor = 0.9  # Allow some overlap for better packing
    
    # Use a hexagonal grid approach
    rows = int(height / (max_radius * 2 * spacing_factor)) + 1
    cols = int(width / (max_radius * 2 * spacing_factor * 0.866)) + 1  # sqrt(3)/2 for hex packing
    
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= n_circles:
                break
            x = max_radius + j * max_radius * 2 * spacing_factor * 0.866
            y = max_radius + i * max_radius * 2 * spacing_factor
            # Offset odd rows
            if i % 2 == 1:
                x += max_radius * spacing_factor * 0.866
            circles[count] = [x, y, max_radius * 0.8]
            count += 1
        if count >= n_circles:
            break
    
    # Fill remaining circles with random positions if needed
    for i in range(count, n_circles):
        circles[i] = [
            np.random.uniform(max_radius, width - max_radius),
            np.random.uniform(max_radius, height - max_radius),
            np.random.uniform(max_radius * 0.2, max_radius * 0.8)
        ]
    
    return circles

def generate_random_valid_configuration(n_circles: int, width: float = 1.0, height: float = 1.0) -> np.ndarray:
    """Generate a random valid configuration."""
    circles = np.zeros((n_circles, 3))
    
    attempts = 0
    max_attempts = 1000
    
    while attempts < max_attempts:
        # Generate random positions and radii
        for i in range(n_circles):
            circles[i] = [
                np.random.uniform(0.01, width - 0.01),
                np.random.uniform(0.01, height - 0.01),
                np.random.uniform(0.01, min(width, height) * 0.2)
            ]
        
        # If valid, return
        if is_valid_solution(circles, width, height):
            return circles
            
        attempts += 1
    
    # If no valid configuration found, return a basic one
    return generate_hexagonal_packing(n_circles, width, height)

def mutate_radius(circles: np.ndarray, criticality_scores: np.ndarray, 
                  container_width: float = 1.0, container_height: float = 1.0) -> np.ndarray:
    """Mutate radii based on criticality scores - high criticality means smaller changes."""
    mutated = circles.copy()
    n = len(mutated)
    
    # Calculate adaptive mutation step based on criticality
    for i in range(n):
        if np.random.random() < 0.3:
            criticality = criticality_scores[i]
            
            # Use criticality to determine mutation step size
            # High criticality (small Voronoi cells) = smaller steps
            # Low criticality (large Voronoi cells) = larger steps  
            max_step = 0.01 if criticality > 0.7 else 0.02
            step = max_step * (1.0 - criticality) + 0.001
            
            # Apply mutation
            delta = np.random.normal(0, step)
            mutated[i, 2] += delta
            
            # Ensure bounds
            mutated[i, 2] = max(0.001, mutated[i, 2])
    
    return mutated

def mutate_position(circles: np.ndarray, container_width: float = 1.0, container_height: float = 1.0) -> np.ndarray:
    """Mutate positions with boundary-aware constraints."""
    mutated = circles.copy()
    n = len(mutated)
    
    for i in range(n):
        if np.random.random() < 0.4:  # 40% chance to modify each circle
            # Apply small random movement
            mutated[i, 0] += np.random.uniform(-0.01, 0.01)
            mutated[i, 1] += np.random.uniform(-0.01, 0.01)
            
            # Enforce boundary constraints
            mutated[i, 0] = max(mutated[i, 2], min(container_width - mutated[i, 2], mutated[i, 0]))
            mutated[i, 1] = max(mutated[i, 2], min(container_height - mutated[i, 2], mutated[i, 1]))
    
    return mutated

def local_optimization(circles: np.ndarray, container_width: float = 1.0, container_height: float = 1.0, 
                      max_iterations: int = 100) -> Tuple[np.ndarray, float]:
    """Perform advanced local optimization with Voronoi guidance."""
    current = circles.copy()
    best_fitness = evaluate_fitness(current)
    best_solution = current.copy()
    
    for iteration in range(max_iterations):
        # Calculate criticality for current configuration
        criticality_scores = calculate_voronoi_criticality(current, container_width, container_height)
        
        # Create candidate by mutating
        candidate = current.copy()
        
        # Mutate positions
        candidate = mutate_position(candidate, container_width, container_height)
        
        # Mutate radii (based on criticality)
        candidate = mutate_radius(candidate, criticality_scores, container_width, container_height)
        
        # Ensure validity
        if not is_valid_solution(candidate, container_width, container_height):
            continue
            
        # Evaluate candidate
        candidate_fitness = evaluate_fitness(candidate)
        
        # Accept if better
        if candidate_fitness > best_fitness:
            best_fitness = candidate_fitness
            best_solution = candidate.copy()
            current = candidate.copy()
        elif np.random.random() < 0.1:  # Sometimes accept worse solutions to escape local minima
            current = candidate.copy()
    
    return best_solution, best_fitness

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    start_time = time.time()
    
    # Container dimensions (perimeter = 4, so width + height = 2)
    container_width = 1.0
    container_height = 1.0
    
    best_individual = None
    best_fitness = float('-inf')
    
    # Multi-start approach with different strategies
    strategies = [
        ("hexagonal", lambda: generate_hexagonal_packing(21, container_width, container_height)),
        ("random", lambda: generate_random_valid_configuration(21, container_width, container_height))
    ]
    
    # Multiple restarts to find better optima
    for strategy_name, strategy_func in strategies:
        for restart in range(5):  # 5 restarts per strategy
            # Start with specific initial configuration
            circles = strategy_func()
            
            # Apply multiple rounds of local optimization
            optimized, fitness = local_optimization(circles, container_width, container_height, 150)
            
            # Track best solution
            if fitness > best_fitness and is_valid_solution(optimized, container_width, container_height):
                best_fitness = fitness
                best_individual = optimized.copy()
    
    # Additional fine-tuning
    if best_individual is not None:
        # Do additional rounds with refined parameters
        refined, refined_fitness = local_optimization(best_individual, container_width, container_height, 200)
        if refined_fitness > best_fitness and is_valid_solution(refined, container_width, container_height):
            best_fitness = refined_fitness
            best_individual = refined.copy()
    
    # Ensure we have a valid result
    if best_individual is None:
        # Fallback to hexagonal packing
        best_individual = generate_hexagonal_packing(21, container_width, container_height)
    
    # Final validation
    if not is_valid_solution(best_individual, container_width, container_height):
        best_individual = generate_random_valid_configuration(21, container_width, container_height)
    
    elapsed_time = time.time() - start_time
    print(f"Optimization completed in {elapsed_time:.2f} seconds")
    
    return best_individual

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
