# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, cKDTree
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time
from collections import defaultdict

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
        # Build grid index with adaptive cell sizing
        avg_radius = np.mean(radii) if n > 0 else 0.1
        cell_size = avg_radius * 1.8  # Increased safety margin for better performance
        
        # Calculate grid dimensions
        cols = max(1, int(rect_width / cell_size) + 2)
        rows = max(1, int(rect_height / cell_size) + 2)
        
        # Create cell dictionary mapping (row, col) to circle indices
        cell_dict = defaultdict(list)
        
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
            cell_dict[(row, col)].append(i)
        
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

def compute_voronoi_constraints(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """
    Compute refined constraint density for each circle based on Voronoi neighbors and geometric properties.
    """
    n = len(circles)
    if n < 2:
        return np.zeros(n)

    # Get circle centers
    centers = circles[:, :2]
    
    # For better constraint estimation, use both neighbor-based and Voronoi-style analysis
    constraint_density = np.zeros(n)
    
    try:
        # Use KDTree for neighbor queries
        tree = cKDTree(centers)
        
        # Compute density and geometric measures for each circle
        for i in range(n):
            center_i = centers[i]
            
            # 1. Neighbor-based density: count neighbors within 3x max radius
            max_radius = np.max(circles[:, 2])
            threshold = max_radius * 3.0
            
            neighbors = tree.query_ball_point(center_i, threshold)
            neighbor_count = len(neighbors) - 1  # Exclude self
            
            # 2. Compute average distance to neighbors (for Voronoi-like density)
            if neighbors and len(neighbors) > 1:
                neighbor_distances = []
                for j in neighbors:
                    if i != j:
                        dist = np.linalg.norm(center_i - centers[j])
                        neighbor_distances.append(dist)
                
                avg_neighbor_dist = np.mean(neighbor_distances) if neighbor_distances else 1.0
                # Inverse relationship with distance (closer = denser = more constrained)
                neighbor_density = 1.0 / (avg_neighbor_dist + 1e-8)
            else:
                neighbor_density = 1.0
            
            # 3. Boundary influence factor (how close to rectangle edges)
            x, y, r = circles[i]
            boundary_distances = [x, y, rect_width - x, rect_height - y]
            min_boundary_dist = min(boundary_distances)
            
            # The closer to boundary, the higher the constraint
            boundary_constraint = 1.0 / (min_boundary_dist + 1e-8) if min_boundary_dist < 0.3 else 0.0
            
            # Combine all factors with carefully tuned weights
            # neighbor_count (weighted lower since it's a discrete count)
            # neighbor_density (weighted higher as it's continuous)
            # boundary_constraint (weighted heavily since boundary effect is strong)
            
            weight_neighbor_count = 0.3
            weight_neighbor_density = 0.4
            weight_boundary = 0.3
            
            combined_density = (weight_neighbor_count * (neighbor_count / max(1, n-1)) +
                              weight_neighbor_density * min(neighbor_density, 10.0) +
                              weight_boundary * min(boundary_constraint, 10.0))
            
            # Normalize and bound the result
            constraint_density[i] = min(combined_density, 5.0)
            
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

    # Ensure reasonable constraint density values
    constraint_density = np.maximum(constraint_density, 0.01)
    return constraint_density

def initialize_hexagonal_lattice(n_circles: int, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """
    Initialize circle positions using a hexagonal lattice pattern with improved distribution.
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

def initialize_pattern_hybrid(n_circles: int, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """
    Initialize using a hybrid of multiple strategies for better diversity.
    """
    # Try multiple initial strategies and return the best
    strategies = []
    
    # Strategy 1: Hexagonal pattern
    try:
        hex_init = initialize_hexagonal_lattice(n_circles, rect_width, rect_height)
        strategies.append(("hex", hex_init))
    except:
        pass
    
    # Strategy 2: Grid pattern
    try:
        grid_points = []
        rows = int(np.sqrt(n_circles)) + 1
        cols = int(np.ceil(n_circles / rows))
        
        cell_width = rect_width / cols
        cell_height = rect_height / rows
        radius = min(cell_width, cell_height) * 0.3
        
        for i in range(rows):
            for j in range(cols):
                if len(grid_points) >= n_circles:
                    break
                x = j * cell_width + cell_width/2
                y = i * cell_height + cell_height/2
                if x >= radius and x <= rect_width - radius and y >= radius and y <= rect_height - radius:
                    grid_points.append([x, y, radius])
        
        while len(grid_points) < n_circles:
            grid_points.append([
                random.uniform(radius, rect_width - radius),
                random.uniform(radius, rect_height - radius),
                random.uniform(0.005, radius * 0.5)
            ])
            
        grid_init = np.array(grid_points[:n_circles])
        strategies.append(("grid", grid_init))
    except:
        pass

    # Strategy 3: Random with constraint checks
    try:
        random_init = np.zeros((n_circles, 3))
        for i in range(n_circles):
            success = False
            attempts = 0
            while not success and attempts < 1000:
                x = random.uniform(0.01, rect_width - 0.01)
                y = random.uniform(0.01, rect_height - 0.01)
                r = random.uniform(0.005, 0.2)
                
                # Check if it fits within bounds
                if x - r >= 0 and x + r <= rect_width and y - r >= 0 and y + r <= rect_height:
                    # Check no overlaps with already placed circles
                    valid = True
                    for j in range(i):
                        ox, oy, orad = random_init[j]
                        dist = np.sqrt((x - ox)**2 + (y - oy)**2)
                        if dist < (r + orad):
                            valid = False
                            break
                    if valid:
                        random_init[i] = [x, y, r]
                        success = True
                attempts += 1
            if not success:
                # Fallback to hex pattern for this circle if random fails
                fallback = initialize_hexagonal_lattice(1, rect_width, rect_height)[0]
                random_init[i] = [fallback[0], fallback[1], fallback[2]]
        strategies.append(("random", random_init))
    except:
        pass

    # Evaluate strategies and return best valid one
    if not strategies:
        # Fallback to hexagonal
        return initialize_hexagonal_lattice(n_circles, rect_width, rect_height)
    
    best_strategy = strategies[0]
    best_score = -np.inf
    
    for name, init in strategies:
        try:
            # Simple scoring: sum of radii with basic penalty for overlaps/boundary issues
            score = np.sum(init[:, 2])
            # Add penalty for boundary violations
            for i in range(len(init)):
                x, y, r = init[i]
                if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
                    score -= 1000
            
            # Add penalty for overlaps
            for i in range(len(init)):
                for j in range(i+1, len(init)):
                    x1, y1, r1 = init[i]
                    x2, y2, r2 = init[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if dist < (r1 + r2):
                        score -= 1000
                        
            if score > best_score:
                best_score = score
                best_strategy = (name, init)
        except:
            continue
    
    return best_strategy[1]

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
            penalty += overlap * 500  # Reduced penalty scale for better balance

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
                            overlap_penalty += overlap_amount * 500  # Reduced penalty scale
        except:
            # Fallback to brute force if spatial indexing fails
            for i in range(n):
                for j in range(i+1, n):
                    distance = np.linalg.norm(coords[i] - coords[j])
                    min_distance = radii[i] + radii[j]
                    
                    if distance < min_distance:
                        # Overlap exists
                        overlap_amount = min_distance - distance
                        overlap_penalty += overlap_amount * 500  # Reduced penalty scale

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
        pos_strength = 0.025 / density_weight
        rad_strength = 0.012 / density_weight
        
        # Add Gaussian noise with adaptive scales
        x += np.random.normal(0, pos_strength)
        y += np.random.normal(0, pos_strength)
        
        # Ensure position stays within bounds
        x = np.clip(x, r, rect_width - r)
        y = np.clip(y, r, rect_height - r)
        
        # Mutate radius with adaptive step size
        r += np.random.normal(0, rad_strength)
        # Ensure radius remains positive and reasonable
        r = np.clip(r, 0.0005, max_radius * 0.9)
        
        mutated[i] = [x, y, r]
        
    return mutated

def crossover_circles(parent1: np.ndarray, parent2: np.ndarray, 
                     crossover_rate: float = 0.8) -> np.ndarray:
    """
    Perform uniform crossover between two circle configurations with better mixing.
    """
    if random.random() > crossover_rate:
        return parent1.copy()  # Return first parent if no crossover
    
    offspring = parent1.copy()
    n = len(parent1)
    
    # Perform crossover in chunks for better structural coherence
    chunk_size = max(1, n // 4)  # Divide into roughly 4 chunks
    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        
        # With 50% probability, switch the entire chunk
        if random.random() < 0.5:
            offspring[chunk_start:chunk_end] = parent2[chunk_start:chunk_end].copy()
            
    return offspring

def refine_solution_fast(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0,
                        iterations: int = 150) -> np.ndarray:
    """
    Fast local refinement with enhanced constraint-aware moves and better convergence.
    """
    refined = circles.copy()
    n = len(refined)
    
    if n <= 1:
        return refined
    
    # Precompute constraint information
    constraint_densities = compute_voronoi_constraints(refined, rect_width, rect_height)
    
    # Iterative refinement with hybrid approaches
    for iter_num in range(iterations):
        # Work on one circle at a time with more sophisticated selection
        selected_indices = list(range(n))
        random.shuffle(selected_indices)  # Randomize order for better exploration
        
        for i in selected_indices:
            # Save current state
            old_x, old_y, old_r = refined[i]
            
            # Adaptive mutation based on constraint density
            density_weight = 1.0 + constraint_densities[i] * 2.0
            pos_strength = 0.015 / density_weight
            rad_strength = 0.008 / density_weight
            
            # Try small random moves with different probabilities
            mutation_type = random.choices(['small', 'medium', 'large'], weights=[0.7, 0.25, 0.05])[0]
            
            if mutation_type == 'small':
                # Standard small mutations
                new_x = old_x + np.random.normal(0, pos_strength)
                new_y = old_y + np.random.normal(0, pos_strength)
                new_r = old_r + np.random.normal(0, rad_strength)
                
            elif mutation_type == 'medium':
                # Larger moves for diversity
                new_x = old_x + np.random.normal(0, pos_strength * 2)
                new_y = old_y + np.random.normal(0, pos_strength * 2)
                new_r = old_r + np.random.normal(0, rad_strength * 2)
                
            else:  # large
                # Big moves occasionally to escape local optima
                new_x = old_x + np.random.normal(0, pos_strength * 5)
                new_y = old_y + np.random.normal(0, pos_strength * 5)
                new_r = old_r + np.random.normal(0, rad_strength * 5)
            
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
                
    return refined

def optimize_with_voronoi_evolution(n_circles: int = 21, 
                                  rect_width: float = 1.0, 
                                  rect_height: float = 1.0,
                                  population_size: int = 100, 
                                  generations: int = 150) -> np.ndarray:
    """
    Optimized version of Voronoi-enhanced evolutionary algorithm with improved strategies.
    """
    # Initialize population with better diversity
    population = []
    for _ in range(population_size):
        # Use hybrid initialization for better starting points
        circles = initialize_pattern_hybrid(n_circles, rect_width, rect_height)
        
        # Add some randomness to initial positions
        for i in range(n_circles):
            circles[i][0] += random.uniform(-0.03, 0.03)
            circles[i][1] += random.uniform(-0.03, 0.03)
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

        # Create new generation with improved elitism and selection
        new_population = [population[0]]  # Elitism - keep best individual
        
        # Generate offspring with improved strategy
        while len(new_population) < population_size:
            # Tournament selection with larger tournament size for better pressure
            tournament_size = 7
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

            # Crossover with better mixing
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
            if recent_improvement < 1e-6 and gen > 40:
                break

    # Return best solution with thorough validation
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
    rect_width = 1.25
    rect_height = 0.75

    # Run Voronoi-enhanced optimization with improved parameters
    best_solution = optimize_with_voronoi_evolution(
        n_circles=21,
        rect_width=rect_width,
        rect_height=rect_height,
        population_size=120,
        generations=150
    )

    # Apply fast refinement with increased iterations
    refined_solution = refine_solution_fast(best_solution, rect_width, rect_height, iterations=200)

    return refined_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")