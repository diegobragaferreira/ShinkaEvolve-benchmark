# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, cKDTree
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time

# Set seeds for determinism
random.seed(42)
np.random.seed(42)

def is_valid_solution(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> bool:
    """
    Fast validity check using grid-based spatial indexing for collision detection.
    """
    n = len(circles)
    
    # Check boundary constraints first
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
            return False
    
    if n <= 1:
        return True
    
    # Use grid-based spatial indexing for efficient overlap detection
    coords = circles[:, :2]
    radii = circles[:, 2]
    
    try:
        # Build grid index
        avg_radius = np.mean(radii) if n > 0 else 0.1
        cell_size = avg_radius * 1.5  # Slightly larger than average radius
        
        # Calculate grid dimensions
        cols = max(1, int(rect_width / cell_size) + 1)
        rows = max(1, int(rect_height / cell_size) + 1)
        
        # Create cell dictionary mapping (row, col) to circle indices
        cell_dict = {}
        
        # Assign circles to grid cells
        for i in range(n):
            x, y, r = circles[i]
            # Find grid cell coordinates for circle center
            col = int(x / cell_size)
            row = int(y / cell_size)
            
            # Clamp to grid bounds
            col = max(0, min(col, cols - 1))
            row = max(0, min(row, rows - 1))
            
            # Store in dictionary
            key = (row, col)
            if key not in cell_dict:
                cell_dict[key] = []
            cell_dict[key].append(i)
        
        # Check for collisions
        for i in range(n):
            x1, y1, r1 = circles[i]
            
            # Find the grid cell that contains this circle
            col = int(x1 / cell_size)
            row = int(y1 / cell_size)
            col = max(0, min(col, cols - 1))
            row = max(0, min(row, rows - 1))
            
            # Check neighboring cells (including current cell)
            # Check 3x3 neighborhood around current cell
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    neighbor_row = row + dr
                    neighbor_col = col + dc
                    
                    # Check bounds
                    if 0 <= neighbor_row < rows and 0 <= neighbor_col < cols:
                        key = (neighbor_row, neighbor_col)
                        if key in cell_dict:
                            # Check all circles in this cell against current circle
                            for j in cell_dict[key]:
                                if i != j:  # Don't compare with self
                                    x2, y2, r2 = circles[j]
                                    
                                    # Fast distance check using squared distances
                                    dx = x1 - x2
                                    dy = y1 - y2
                                    distance_sq = dx*dx + dy*dy
                                    min_distance_sq = (r1 + r2) * (r1 + r2)
                                    
                                    if distance_sq < min_distance_sq:
                                        return False
        
        return True
    except:
        # Fallback to brute force if grid indexing fails
        for i in range(n):
            for j in range(i+1, n):
                distance = np.linalg.norm(coords[i] - coords[j])
                min_distance = radii[i] + radii[j]
                
                if distance < min_distance:
                    return False
        return True

def compute_voronoi_density(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """
    Compute refined constraint density for each circle based on Voronoi neighbors and geometric properties.
    """
    n = len(circles)
    if n < 2:
        return np.zeros(n)

    # Get circle centers
    centers = circles[:, :2]

    # Compute neighbor density using spatial indexing for efficiency
    constraint_density = np.zeros(n)

    # Use KDTree for efficient neighbor queries
    try:
        tree = cKDTree(centers)

        # For each circle, compute multiple constraint indicators
        for i in range(n):
            center_i = centers[i]

            # 1. Direct neighbor count within a reasonable distance
            # Use 3x the maximum radius as threshold for neighbors
            max_radius = np.max(circles[:, 2])
            threshold = max_radius * 3.0

            # Query neighbors efficiently
            neighbors = tree.query_ball_point(center_i, threshold)
            neighbor_count = len(neighbors) - 1  # Exclude self

            # 2. Consider Voronoi cell area (approximated using neighbor distances)
            # If we have enough neighbors, compute approximate Voronoi density
            if len(neighbors) >= 3:
                # Get distances to neighbors
                neighbor_distances = []
                for j in neighbors:
                    if i != j:
                        dist = np.linalg.norm(center_i - centers[j])
                        neighbor_distances.append(dist)

                if neighbor_distances:
                    avg_neighbor_dist = np.mean(neighbor_distances)
                    # Smaller average distances suggest denser packing
                    voronoi_density = 1.0 / (avg_neighbor_dist + 1e-8)
                else:
                    voronoi_density = 1.0
            else:
                voronoi_density = 1.0

            # Combine both measures with appropriate weights
            # Weight neighbor count more heavily for dense regions
            neighbor_weight = 0.7
            voronoi_weight = 0.3

            combined_density = (neighbor_weight * (neighbor_count / max(1, n - 1)) +
                              voronoi_weight * min(voronoi_density / 10.0, 1.0))

            # Normalize and bound the density value
            constraint_density[i] = min(combined_density, 3.0)  # Cap at reasonable level

    except Exception as e:
        # Fallback to simple neighbor counting if Voronoi fails
        for i in range(n):
            center_i = centers[i]
            nearby_count = 0

            # Check nearby circles (within 3x max radius)
            max_radius = np.max(circles[:, 2])
            threshold = 3 * max_radius

            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(center_i - centers[j])
                    if dist < threshold:
                        nearby_count += 1

            constraint_density[i] = nearby_count / max(1, n - 1)

    return constraint_density

def initialize_hexagonal_lattice(n_circles: int, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """
    Initialize circle positions using a hexagonal lattice pattern.
    """
    # Use hexagonal packing approach for initial placement
    # Estimate radius based on area
    total_area = rect_width * rect_height
    circle_area = total_area / n_circles * 0.9  # Leave some margin
    estimated_radius = np.sqrt(circle_area / np.pi)

    # Hexagon parameters
    side_length = 2 * estimated_radius

    # Determine grid dimensions
    cols = max(1, int(rect_width / side_length) + 1)
    rows = max(1, int(rect_height / (side_length * np.sqrt(3) / 2)) + 1)

    points = []
    for i in range(rows):
        for j in range(cols):
            x = (j + (i % 2) * 0.5) * side_length
            y = i * side_length * np.sqrt(3) / 2

            # Only include points that fit within the rectangle
            if x >= estimated_radius and x <= rect_width - estimated_radius and \
               y >= estimated_radius and y <= rect_height - estimated_radius:
                points.append([x, y])

    # If we have too few points, add more by expanding
    while len(points) < n_circles:
        # Add points at random locations within bounds
        x = random.uniform(estimated_radius, rect_width - estimated_radius)
        y = random.uniform(estimated_radius, rect_height - estimated_radius)
        points.append([x, y])

    # Trim to exact number needed
    points = points[:n_circles]

    # Create initial circles with estimated radii
    circles = np.zeros((n_circles, 3))
    for i, (x, y) in enumerate(points):
        circles[i] = [x, y, estimated_radius * 0.8]

    return circles

def initialize_spiral_pattern(n_circles: int, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """
    Initialize circles using a spiral pattern for better edge coverage.
    """
    circles = np.zeros((n_circles, 3))
    
    # Center of the rectangle
    center_x, center_y = rect_width / 2, rect_height / 2
    
    # Spiral parameters
    angle_step = 2 * np.pi / 5
    radius_step = min(rect_width, rect_height) * 0.2 / n_circles
    max_radius = min(rect_width, rect_height) * 0.4
    
    # Generate spiral points
    for i in range(n_circles):
        # Spiral position
        angle = i * angle_step
        radius = min(max_radius, i * radius_step)
        
        # Convert to cartesian coordinates
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        
        # Ensure within bounds
        x = max(0.01, min(rect_width - 0.01, x))
        y = max(0.01, min(rect_height - 0.01, y))
        
        # Set initial radius
        circles[i] = [x, y, 0.04]
    
    return circles

def initialize_square_grid(n_circles: int, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """
    Initialize circles using square grid pattern.
    """
    circles = np.zeros((n_circles, 3))
    
    # Square grid arrangement
    rows = int(np.ceil(np.sqrt(n_circles)))
    cols = int(np.ceil(n_circles / rows))
    
    spacing_x = rect_width / (cols + 1)
    spacing_y = rect_height / (rows + 1)
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n_circles:
                break
            x = (j + 1) * spacing_x
            y = (i + 1) * spacing_y
            
            # Ensure position is within bounds
            x = max(0.01, min(rect_width - 0.01, x))
            y = max(0.01, min(rect_height - 0.01, y))
            
            # Set initial radius to a small value
            circles[idx] = [x, y, 0.05]
            idx += 1
            
        if idx >= n_circles:
            break
    
    # Fill remaining circles if needed
    while idx < n_circles:
        x = np.random.uniform(0.01, rect_width - 0.01)
        y = np.random.uniform(0.01, rect_height - 0.01)
        circles[idx] = [x, y, 0.05]
        idx += 1
    
    return circles

def initialize_smart_random(n_circles: int, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """
    Initialize circles using random placement with intelligent spacing.
    """
    circles = np.zeros((n_circles, 3))
    
    # Start with larger initial radii to encourage better packing
    for i in range(n_circles):
        # Attempt placement with more intelligent distribution
        placed = False
        attempts = 0
        
        while not placed and attempts < 100:
            # Prefer placing near edges and corners for better spread
            if np.random.random() < 0.3:  # 30% chance to place near boundary
                # Near boundary placement
                boundary_edge = np.random.choice(['left', 'right', 'top', 'bottom'])
                if boundary_edge == 'left':
                    x = np.random.uniform(0.01, 0.1)
                    y = np.random.uniform(0.01, rect_height - 0.01)
                elif boundary_edge == 'right':
                    x = np.random.uniform(rect_width - 0.1, rect_width - 0.01)
                    y = np.random.uniform(0.01, rect_height - 0.01)
                elif boundary_edge == 'top':
                    x = np.random.uniform(0.01, rect_width - 0.01)
                    y = np.random.uniform(rect_height - 0.1, rect_height - 0.01)
                else:  # bottom
                    x = np.random.uniform(0.01, rect_width - 0.01)
                    y = np.random.uniform(0.01, 0.1)
            else:
                # Regular random placement
                x = np.random.uniform(0.01, rect_width - 0.01)
                y = np.random.uniform(0.01, rect_height - 0.01)
            
            # Check if this location is compatible with existing circles
            valid = True
            for j in range(i):
                dx = x - circles[j, 0]
                dy = y - circles[j, 1]
                distance = np.sqrt(dx*dx + dy*dy)
                # Require more space between new and existing circles
                if distance < (circles[j, 2] + 0.04) * 1.2:
                    valid = False
                    break
            
            if valid:
                circles[i] = [x, y, 0.04]
                placed = True
            else:
                attempts += 1
        
        # Fallback to random if failed
        if not placed:
            x = np.random.uniform(0.01, rect_width - 0.01)
            y = np.random.uniform(0.01, rect_height - 0.01)
            circles[i] = [x, y, 0.04]
    
    return circles

def calculate_fitness(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> Tuple[float, float]:
    """
    Calculate fitness of circle configuration with penalty for constraint violations.
    """
    n = len(circles)
    
    # Check boundary constraints
    penalty = 0.0
    for i in range(n):
        x, y, r = circles[i]
        # Circle must be fully contained within rectangle
        if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
            # Apply penalty based on how much it violates boundaries
            overlap = 0.0
            if x - r < 0:
                overlap += abs(x - r)
            if x + r > rect_width:
                overlap += abs(x + r - rect_width)
            if y - r < 0:
                overlap += abs(y - r)
            if y + r > rect_height:
                overlap += abs(y + r - rect_height)
            penalty += overlap * 1000

    # Check overlap constraints using efficient spatial indexing
    overlap_penalty = 0.0
    if n > 1:
        # Use spatial indexing with KDTree for efficient overlap detection
        coords = circles[:, :2]
        radii = circles[:, 2]
        
        try:
            tree = cKDTree(coords)
            max_radius = np.max(radii)
            
            # For each circle, find neighbors and check overlaps
            for i in range(n):
                # Query neighbors within 2 * max_radius distance
                neighbors = tree.query_ball_point(coords[i], 2 * max_radius)
                
                # Check overlaps with neighbors
                for j in neighbors:
                    if i != j:
                        distance = np.linalg.norm(coords[i] - coords[j])
                        min_distance = radii[i] + radii[j]
                        
                        if distance < min_distance:
                            # Overlap exists
                            overlap_amount = min_distance - distance
                            overlap_penalty += overlap_amount * 1000  # Heavy penalty for overlaps
        except:
            # Fallback to brute force if spatial indexing fails
            for i in range(n):
                for j in range(i+1, n):
                    distance = np.linalg.norm(coords[i] - coords[j])
                    min_distance = radii[i] + radii[j]
                    
                    if distance < min_distance:
                        # Overlap exists
                        overlap_amount = min_distance - distance
                        overlap_penalty += overlap_amount * 1000  # Heavy penalty for overlaps

    # Fitness is sum of radii minus penalties
    total_radius = np.sum(circles[:, 2])
    fitness = total_radius - penalty - overlap_penalty
    
    return fitness, overlap_penalty

def mutate_circles_adaptive(circles: np.ndarray, 
                          constraint_densities: np.ndarray,
                          rect_width: float = 1.0, 
                          rect_height: float = 1.0,
                          max_radius: float = 0.5) -> np.ndarray:
    """
    Mutate circle positions and radii with adaptive weights based on constraint density.
    """
    mutated = circles.copy()
    n = len(mutated)
    
    # Mutation parameters adapted based on constraint density
    for i in range(n):
        x, y, r = mutated[i]
        
        # Higher constraint density = more careful mutation
        density_weight = 1.0 + constraint_densities[i] * 2.0  # 1.0 to 3.0 range
        
        # Position mutation strength (smaller for high constraint areas)
        pos_strength = 0.02 / density_weight
        rad_strength = 0.01 / density_weight
        
        # Mutate position
        x += np.random.normal(0, pos_strength)
        y += np.random.normal(0, pos_strength)
        
        # Ensure position stays within bounds
        x = np.clip(x, r, rect_width - r)
        y = np.clip(y, r, rect_height - r)
        
        # Mutate radius
        r += np.random.normal(0, rad_strength)
        # Ensure radius remains positive and reasonable
        r = np.clip(r, 0.001, max_radius * 0.9)
        
        mutated[i] = [x, y, r]
        
    return mutated

def crossover_circles(parent1: np.ndarray, parent2: np.ndarray, 
                     crossover_rate: float = 0.8) -> np.ndarray:
    """
    Perform uniform crossover between two circle configurations.
    """
    if random.random() > crossover_rate:
        return parent1.copy()  # Return first parent if no crossover
    
    offspring = parent1.copy()
    n = len(parent1)
    
    # Uniform crossover
    for i in range(n):
        if random.random() < 0.5:
            offspring[i] = parent2[i].copy()
            
    return offspring

def refine_solution_fast(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0,
                        iterations: int = 100) -> np.ndarray:
    """
    Fast local refinement using gradient-like approach and constraint-aware moves.
    """
    refined = circles.copy()
    n = len(refined)
    
    if n <= 1:
        return refined
    
    # Precompute constraint information
    constraint_densities = compute_voronoi_constraints(refined, rect_width, rect_height)
    
    # Iterative refinement
    for iter_num in range(iterations):
        # Work on one circle at a time
        for i in range(n):
            # Save current state
            old_x, old_y, old_r = refined[i]
            
            # Adaptive mutation based on constraint density
            density_weight = 1.0 + constraint_densities[i] * 2.0
            pos_strength = 0.01 / density_weight
            rad_strength = 0.005 / density_weight
            
            # Try small random moves
            new_x = old_x + np.random.normal(0, pos_strength)
            new_y = old_y + np.random.normal(0, pos_strength)
            new_r = old_r + np.random.normal(0, rad_strength)
            
            # Clip to bounds
            new_x = np.clip(new_x, new_r, rect_width - new_r)
            new_y = np.clip(new_y, new_r, rect_height - new_r)
            new_r = np.clip(new_r, 0.001, rect_width / 2)
            
            # Test if this change improves fitness
            test_config = refined.copy()
            test_config[i] = [new_x, new_y, new_r]
            
            # Quick constraint check before full fitness evaluation
            if not is_valid_solution(test_config, rect_width, rect_height):
                continue
                
            # Check if this move improves fitness
            current_fitness, _ = calculate_fitness(refined, rect_width, rect_height)
            test_fitness, _ = calculate_fitness(test_config, rect_width, rect_height)
            
            if test_fitness > current_fitness:
                refined = test_config
                
            # Occasionally do larger moves to escape local optima
            elif random.random() < 0.05 and iter_num > iterations // 2:
                # Do bigger perturbation
                new_x = old_x + np.random.normal(0, pos_strength * 3)
                new_y = old_y + np.random.normal(0, pos_strength * 3)
                new_r = old_r + np.random.normal(0, rad_strength * 3)
                
                # Clip to bounds
                new_x = np.clip(new_x, new_r, rect_width - new_r)
                new_y = np.clip(new_y, new_r, rect_height - new_r)
                new_r = np.clip(new_r, 0.001, rect_width / 2)
                
                if is_valid_solution(test_config, rect_width, rect_height):
                    test_config[i] = [new_x, new_y, new_r]
                    test_fitness, _ = calculate_fitness(test_config, rect_width, rect_height)
                    if test_fitness > current_fitness:
                        refined = test_config
                        
    return refined

def compute_voronoi_constraints(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """
    Compute refined constraint density for each circle based on Voronoi neighbors and geometric properties.
    """
    n = len(circles)
    if n < 2:
        return np.zeros(n)

    # Get circle centers
    centers = circles[:, :2]

    # Compute neighbor density using spatial indexing for efficiency
    constraint_density = np.zeros(n)

    # Use KDTree for efficient neighbor queries
    try:
        tree = cKDTree(centers)

        # For each circle, compute multiple constraint indicators
        for i in range(n):
            center_i = centers[i]

            # 1. Direct neighbor count within a reasonable distance
            # Use 3x the maximum radius as threshold for neighbors
            max_radius = np.max(circles[:, 2])
            threshold = max_radius * 3.0

            # Query neighbors efficiently
            neighbors = tree.query_ball_point(center_i, threshold)
            neighbor_count = len(neighbors) - 1  # Exclude self

            # 2. Consider Voronoi cell area (approximated using neighbor distances)
            # If we have enough neighbors, compute approximate Voronoi density
            if len(neighbors) >= 3:
                # Get distances to neighbors
                neighbor_distances = []
                for j in neighbors:
                    if i != j:
                        dist = np.linalg.norm(center_i - centers[j])
                        neighbor_distances.append(dist)

                if neighbor_distances:
                    avg_neighbor_dist = np.mean(neighbor_distances)
                    # Smaller average distances suggest denser packing
                    voronoi_density = 1.0 / (avg_neighbor_dist + 1e-8)
                else:
                    voronoi_density = 1.0
            else:
                voronoi_density = 1.0

            # Combine both measures with appropriate weights
            # Weight neighbor count more heavily for dense regions
            neighbor_weight = 0.7
            voronoi_weight = 0.3

            combined_density = (neighbor_weight * (neighbor_count / max(1, n - 1)) +
                              voronoi_weight * min(voronoi_density / 10.0, 1.0))

            # Normalize and bound the density value
            constraint_density[i] = min(combined_density, 3.0)  # Cap at reasonable level

    except Exception as e:
        # Fallback to simple neighbor counting if Voronoi fails
        for i in range(n):
            center_i = centers[i]
            nearby_count = 0

            # Check nearby circles (within 3x max radius)
            max_radius = np.max(circles[:, 2])
            threshold = 3 * max_radius

            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(center_i - centers[j])
                    if dist < threshold:
                        nearby_count += 1

            constraint_density[i] = nearby_count / max(1, n - 1)

    return constraint_density

def optimize_with_voronoi_evolution(n_circles: int = 21, 
                                  rect_width: float = 1.0, 
                                  rect_height: float = 1.0,
                                  population_size: int = 100, 
                                  generations: int = 100) -> np.ndarray:
    """
    Optimize circle packing using Voronoi-enhanced evolutionary algorithm.
    """
    # Initialize population
    population = []
    for _ in range(population_size):
        circles = initialize_hexagonal_lattice(n_circles, rect_width, rect_height)
        # Add some randomness to initial positions
        for i in range(n_circles):
            circles[i][0] += random.uniform(-0.05, 0.05)
            circles[i][1] += random.uniform(-0.05, 0.05)
            circles[i][0] = np.clip(circles[i][0], circles[i][2], rect_width - circles[i][2])
            circles[i][1] = np.clip(circles[i][1], circles[i][2], rect_height - circles[i][2])
        population.append(circles)

    # Evolutionary loop
    best_fitness_history = []
    
    for gen in range(generations):
        # Evaluate fitness of population
        fitness_scores = []
        for circles in population:
            fitness, _ = calculate_fitness(circles, rect_width, rect_height)
            fitness_scores.append(fitness)
        
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]  # Descending order
        population = [population[i] for i in sorted_indices]
        fitness_scores.sort(reverse=True)
        
        best_fitness_history.append(fitness_scores[0])
        
        # Print progress
        if gen % 20 == 0:
            print(f"Generation {gen}, Best fitness: {fitness_scores[0]:.6f}")

        # Create new generation
        new_population = [population[0]]  # Elitism - keep best individual

        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = 5
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]

            # Select parent
            parent1 = population[winner_index]

            # Select second parent
            tournament_indices.remove(winner_index)
            tournament_fitness.remove(max(tournament_fitness))
            winner_index2 = tournament_indices[np.argmax(tournament_fitness)]
            parent2 = population[winner_index2]

            # Crossover
            offspring = crossover_circles(parent1, parent2)

            # Compute constraint densities and mutate adaptively
            constraint_densities = compute_voronoi_constraints(offspring, rect_width, rect_height)
            offspring = mutate_circles_adaptive(offspring, constraint_densities, 
                                              rect_width, rect_height, 
                                              max_radius=min(rect_width, rect_height) / 2)

            new_population.append(offspring)

        population = new_population[:population_size]
        
        # Early stopping if no improvement
        if len(best_fitness_history) >= 5:
            recent_improvement = best_fitness_history[-1] - best_fitness_history[-5]
            if recent_improvement < 1e-6 and gen > 30:
                break

    # Return best solution
    best_index = np.argmax([calculate_fitness(ind, rect_width, rect_height)[0] for ind in population])
    return population[best_index]

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: perimeter = 4 => width + height = 2
    # Optimize rectangle aspect ratio for better packing
    rect_width = 1.2
    rect_height = 0.8

    # Try multiple initialization strategies
    initial_patterns = [
        initialize_hexagonal_lattice(21, rect_width, rect_height),
        initialize_spiral_pattern(21, rect_width, rect_height),
        initialize_square_grid(21, rect_width, rect_height),
        initialize_smart_random(21, rect_width, rect_height)
    ]

    best_solution = None
    best_score = -float('inf')

    # Multi-start optimization
    for seed_pattern in initial_patterns:
        # Apply local optimization to get better starting points
        optimized_pattern = refine_solution_fast(seed_pattern, rect_width, rect_height, max_iter=30)

        # Further refine using a few rounds of local search
        final_circles = refine_solution_fast(optimized_pattern, rect_width, rect_height, max_iter=20)

        score = calculate_fitness(final_circles, rect_width, rect_height)[0]
        if score > best_score and is_valid_solution(final_circles, rect_width, rect_height):
            best_score = score
            best_solution = final_circles.copy()

    # Final fine-tuning with evolutionary algorithm
    if best_solution is not None:
        # Apply more extensive local optimization with Voronoi guidance
        evolved_solution = optimize_with_voronoi_evolution(
            n_circles=21,
            rect_width=rect_width,
            rect_height=rect_height,
            population_size=100,
            generations=100
        )
        
        # Compare with our previous best and use the better one
        evolved_score = calculate_fitness(evolved_solution, rect_width, rect_height)[0]
        if evolved_score > best_score:
            best_solution = evolved_solution

    # Ensure final validity
    if best_solution is None:
        # Fallback to simple initialization
        best_solution = initialize_smart_random(21, rect_width, rect_height)
        best_solution = refine_solution_fast(best_solution, rect_width, rect_height, max_iter=100)

    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")