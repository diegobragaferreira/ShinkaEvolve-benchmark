# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, Delaunay
from scipy.spatial.distance import cdist
import random
from collections import defaultdict

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid Voronoi-evolutionary approach for optimal packing.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    width, height = 1.2, 0.8
    
    # Initialize circles array
    n = 21
    circles = np.zeros((n, 3))
    
    # Phase 1: Voronoi-based Initialization
    # Sample points using Voronoi tessellation for better spatial distribution
    np.random.seed(42)
    
    # Generate initial points using a combination of grid sampling and Voronoi sampling
    grid_points = []
    grid_size = 6
    
    # Create a regular grid for initial coverage
    for i in range(grid_size):
        for j in range(grid_size):
            x = 0.05 + (width - 0.1) * i / (grid_size - 1)
            y = 0.05 + (height - 0.1) * j / (grid_size - 1)
            grid_points.append([x, y])
    
    # Add boundary points for better edge coverage
    boundary_points = [
        [0.1, 0.1],           # Bottom-left
        [width-0.1, 0.1],     # Bottom-right
        [0.1, height-0.1],    # Top-left
        [width-0.1, height-0.1], # Top-right
        [width/2, 0.1],       # Bottom-middle
        [width/2, height-0.1], # Top-middle
        [0.1, height/2],      # Left-middle
        [width-0.1, height/2], # Right-middle
    ]
    
    # Combine all initial points
    all_init_points = grid_points + boundary_points
    
    # Generate additional points via Voronoi sampling if needed
    if len(all_init_points) < n:
        additional_points = []
        while len(additional_points) < (n - len(all_init_points)):
            x = np.random.uniform(0.05, width - 0.05)
            y = np.random.uniform(0.05, height - 0.05)
            additional_points.append([x, y])
        all_init_points.extend(additional_points[:n-len(all_init_points)])
    
    # Initialize with small radii at these positions
    for i in range(n):
        circles[i] = [all_init_points[i][0], all_init_points[i][1], 0.01]
    
    # Phase 2: Evolutionary Optimization with Multi-Scale Refinement
    # Set up evolutionary parameters
    max_generations = 200
    population_size = min(50, max(10, n * 2))
    elite_count = max(1, population_size // 5)
    
    # Store best solution found so far
    best_circles = circles.copy()
    best_sum_radii = np.sum(circles[:, 2])
    
    # Main evolutionary loop
    for generation in range(max_generations):
        # Create population
        population = []
        for _ in range(population_size):
            # Create offspring from current best solution
            individual = best_circles.copy()
            
            # Add small random perturbations
            for i in range(n):
                if random.random() < 0.3:  # 30% chance to mutate each circle
                    # Perturb position
                    individual[i][0] += np.random.normal(0, 0.02)
                    individual[i][1] += np.random.normal(0, 0.02)
                    # Keep within bounds
                    individual[i][0] = np.clip(individual[i][0], 0.01, width - 0.01)
                    individual[i][1] = np.clip(individual[i][1], 0.01, height - 0.01)
            
            population.append(individual)
        
        # Evaluate fitness of each individual
        fitness_scores = []
        for individual in population:
            try:
                # Calculate fitness (negative sum of radii since we want to maximize)
                fitness = -calculate_total_radius(individual, width, height)
                fitness_scores.append(fitness)
            except:
                # Handle invalid configurations with very poor fitness
                fitness_scores.append(-1e10)
        
        # Sort population by fitness (best first)
        sorted_indices = np.argsort(fitness_scores)
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]
        
        # Update best solution if found
        current_best = sorted_population[0]
        current_best_fitness = sorted_fitness[0]
        current_sum_radii = -current_best_fitness
        
        if current_sum_radii > best_sum_radii:
            best_circles = current_best.copy()
            best_sum_radii = current_sum_radii
            
        # Apply selection and reproduction
        elite_individuals = sorted_population[:elite_count]
        
        # Create next generation through crossover and mutation
        next_generation = elite_individuals[:]
        
        # Add some variation through crossover
        while len(next_generation) < population_size:
            parent1 = random.choice(elite_individuals)
            parent2 = random.choice(elite_individuals)
            
            # Crossover: combine positions from two parents
            child = parent1.copy()
            for i in range(n):
                if random.random() < 0.5:
                    child[i][0] = parent2[i][0]
                    child[i][1] = parent2[i][1]
            
            # Mutation
            for i in range(n):
                if random.random() < 0.2:  # 20% chance to mutate each circle
                    child[i][0] += np.random.normal(0, 0.01)
                    child[i][1] += np.random.normal(0, 0.01)
                    # Keep within bounds
                    child[i][0] = np.clip(child[i][0], 0.01, width - 0.01)
                    child[i][1] = np.clip(child[i][1], 0.01, height - 0.01)
            
            next_generation.append(child)
        
        population = next_generation[:population_size]
    
    # Phase 3: Local Fine-tuning with Improved Constraint Handling
    # Use a more sophisticated local search with neighbor lists
    circles = best_circles.copy()
    
    # Build neighbor list for efficient constraint checking
    def build_neighbor_list():
        neighbors = defaultdict(list)
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((circles[i][0] - circles[j][0])**2 + (circles[i][1] - circles[j][1])**2)
                if dist < (circles[i][2] + circles[j][2]):
                    neighbors[i].append(j)
                    neighbors[j].append(i)
        return neighbors
    
    # Refine solution with local optimization
    for refinement_step in range(100):
        # Build current neighbor list
        neighbors = build_neighbor_list()
        
        # For each circle, attempt to optimize radius and position
        for i in range(n):
            # Try to increase radius as much as possible
            old_radius = circles[i][2]
            max_radius = calculate_max_radius_fast(circles, i, width, height, neighbors)
            if max_radius > old_radius:
                circles[i][2] = max_radius
            
            # Try to slightly reposition to resolve overlaps
            current_x, current_y = circles[i][0], circles[i][1]
            best_x, best_y = current_x, current_y
            best_radius = circles[i][2]
            
            # Check a few nearby positions
            step_size = 0.02
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    new_x = current_x + dx
                    new_y = current_y + dy
                    
                    # Check bounds
                    if 0.01 <= new_x <= width - 0.01 and 0.01 <= new_y <= height - 0.01:
                        # Check if this position improves the radius
                        temp_circles = circles.copy()
                        temp_circles[i][0] = new_x
                        temp_circles[i][1] = new_y
                        
                        # Calculate max radius at new position
                        max_radius_at_new_pos = calculate_max_radius_fast(
                            temp_circles, i, width, height, neighbors
                        )
                        
                        if max_radius_at_new_pos > best_radius:
                            best_radius = max_radius_at_new_pos
                            best_x, best_y = new_x, new_y
            
            # Apply the best move if beneficial
            if best_x != current_x or best_y != current_y:
                circles[i][0] = best_x
                circles[i][1] = best_y
                circles[i][2] = best_radius
    
    # Final validation
    for i in range(n):
        # Ensure minimum radius
        circles[i][2] = max(circles[i][2], 0.001)
        
        # Ensure circles stay within bounds
        circles[i][0] = np.clip(circles[i][0], 0.001, width - 0.001)
        circles[i][1] = np.clip(circles[i][1], 0.001, height - 0.001)
    
    return circles

def calculate_total_radius(circles, width, height):
    """Calculate the total sum of all circle radii."""
    return np.sum(circles[:, 2])

def calculate_max_radius_fast(circles, index, width, height, neighbors):
    """Fast calculation of maximum radius for circle at given index, using neighbor information."""
    x, y, current_radius = circles[index]

    # Maximum radius based on container boundaries
    max_radius_bound = min(x, y, width - x, height - y)

    # Maximum radius based on other circles using neighbor list
    max_radius_overlap = float('inf')
    
    # Only check neighbors for efficiency
    if index < len(neighbors):
        for neighbor_idx in neighbors[index]:
            nx, ny, nr = circles[neighbor_idx]
            # Distance to other circle center
            dist = np.sqrt((x - nx)**2 + (y - ny)**2)
            # Max radius that avoids overlap
            max_radius_for_this_circle = dist - nr
            max_radius_overlap = min(max_radius_overlap, max_radius_for_this_circle)

    max_radius = min(max_radius_bound, max_radius_overlap)
    return max(max_radius, 0.001)  # Ensure minimum radius

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")