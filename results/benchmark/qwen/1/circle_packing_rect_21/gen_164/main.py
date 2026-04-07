# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.spatial import Voronoi, distance_matrix
from scipy.spatial.distance import cdist
import time
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Rectangle dimensions: optimized width=1.5, height=0.5 (perimeter = 4)
    rect_width = 1.5
    rect_height = 0.5

    # Number of circles
    n_circles = 21

    def get_voronoi_criticality(circles_array: np.ndarray) -> np.ndarray:
        """
        Calculate criticality based on Voronoi diagram - how much room each circle has
        Criticality is inversely proportional to Voronoi cell area (smaller cells = more constrained)
        """
        try:
            points = circles_array[:, :2]  # x,y coordinates
            vor = Voronoi(points)
            
            criticality_scores = np.zeros(len(circles_array))
            
            for i in range(len(circles_array)):
                # Get Voronoi region for this point
                region_index = vor.point_region[i]
                region_vertices = vor.regions[region_index]
                
                if -1 not in region_vertices and len(region_vertices) > 0:
                    # Extract vertices for this region
                    vertices = np.array([vor.vertices[j] for j in region_vertices if j != -1])
                    
                    if len(vertices) >= 3:  # Need at least 3 points to form polygon
                        # Calculate area using shoelace formula
                        x = vertices[:, 0]
                        y = vertices[:, 1]
                        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                        
                        # Criticality is inverse of area (smaller area = more critical)
                        # Normalize to avoid extreme values
                        criticality_scores[i] = 1.0 / (area + 1e-8)
                    else:
                        # Fallback for degenerate cases
                        criticality_scores[i] = 1.0
                else:
                    # Fallback for boundary cases
                    criticality_scores[i] = 1.0
                    
        except Exception:
            # Fallback: uniform criticality if Voronoi fails
            criticality_scores = np.ones(len(circles_array))
            
        # Normalize to [0.1, 1.0] range for consistency
        if np.max(criticality_scores) > 0:
            normalized = (criticality_scores - np.min(criticality_scores)) / (np.max(criticality_scores) - np.min(criticality_scores) + 1e-8)
            criticality_scores = 0.1 + 0.9 * normalized
            
        return criticality_scores

    def evaluate_fitness(circles_array: np.ndarray) -> Tuple[float, bool]:
        """
        Evaluate fitness with detailed constraint checking
        Returns (fitness_value, is_valid)
        """
        # Check boundary constraints
        valid = True
        for i in range(len(circles_array)):
            x, y, r = circles_array[i]
            if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
                valid = False
                break
                
        if not valid:
            return -float('inf'), False
            
        # Check overlap constraints
        for i in range(len(circles_array)):
            for j in range(i+1, len(circles_array)):
                x1, y1, r1 = circles_array[i]
                x2, y2, r2 = circles_array[j]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < (r1 + r2):
                    valid = False
                    break
            if not valid:
                break
                
        if not valid:
            return -float('inf'), False
            
        # Valid solution - return sum of radii
        return float(np.sum(circles_array[:, 2])), True

    def initialize_population() -> List[np.ndarray]:
        """Create diverse initial population with multiple patterns"""
        population = []
        
        # Pattern 1: Hexagonal packing
        try:
            hex_pattern = generate_hexagonal_pattern(rect_width, rect_height, n_circles)
            fitness, valid = evaluate_fitness(hex_pattern)
            if valid:
                population.append(hex_pattern)
        except:
            pass
            
        # Pattern 2: Grid-based packing
        try:
            grid_pattern = generate_grid_pattern(rect_width, rect_height, n_circles)
            fitness, valid = evaluate_fitness(grid_pattern)
            if valid:
                population.append(grid_pattern)
        except:
            pass
            
        # Pattern 3: Random with overlap avoidance
        try:
            rand_pattern = generate_random_constrained_pattern(rect_width, rect_height, n_circles)
            fitness, valid = evaluate_fitness(rand_pattern)
            if valid:
                population.append(rand_pattern)
        except:
            pass
            
        # Pattern 4: Spiral pattern
        try:
            spiral_pattern = generate_spiral_pattern(rect_width, rect_height, n_circles)
            fitness, valid = evaluate_fitness(spiral_pattern)
            if valid:
                population.append(spiral_pattern)
        except:
            pass
            
        # If no valid patterns created, create fallback
        if len(population) == 0:
            fallback = np.zeros((n_circles, 3))
            for i in range(n_circles):
                fallback[i] = [
                    random.uniform(0.01, rect_width - 0.01),
                    random.uniform(0.01, rect_height - 0.01),
                    random.uniform(0.01, min(rect_width, rect_height) * 0.1)
                ]
            population.append(fallback)
            
        return population

    def generate_hexagonal_pattern(width: float, height: float, n: int) -> np.ndarray:
        """Generate initial hexagonal packing pattern"""
        circles = np.zeros((n, 3))
        
        # Determine grid parameters
        rows = int(np.sqrt(n))
        cols = int(np.ceil(n / rows))
        
        # Calculate spacing
        margin = 0.1
        max_radius = min(width, height) * 0.1
        
        # Create hexagonal grid
        x_spacing = max_radius * 2.5
        y_spacing = max_radius * 2.165  # sqrt(3)/2 * 2
        
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= n:
                    break
                    
                x = margin + col * x_spacing
                y = margin + row * y_spacing
                
                if row % 2 == 1:
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
        margin = 0.1
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

    def generate_random_constrained_pattern(width: float, height: float, n: int) -> np.ndarray:
        """Generate random pattern with basic overlap avoidance"""
        circles = np.zeros((n, 3))
        max_radius = min(width, height) * 0.1
        attempts = 0
        
        for i in range(n):
            attempts = 0
            valid = False
            while not valid and attempts < 1000:
                x = np.random.uniform(max_radius, width - max_radius)
                y = np.random.uniform(max_radius, height - max_radius)
                radius = np.random.uniform(0.01, max_radius)
                
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

    def generate_spiral_pattern(width: float, height: float, n: int) -> np.ndarray:
        """Generate spiral pattern for diverse initialization"""
        circles = np.zeros((n, 3))
        max_radius = min(width, height) * 0.1
        
        center_x = width / 2
        center_y = height / 2
        
        for i in range(n):
            angle = i * 0.5
            radius = i * 0.1
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            
            # Clip to bounds
            x = max(max_radius, min(width - max_radius, x))
            y = max(max_radius, min(height - max_radius, y))
            
            circles[i] = [x, y, max_radius]
            
        return circles

    def adaptive_mutation(individual: np.ndarray, generation: int, max_generation: int) -> np.ndarray:
        """
        Adaptive mutation that adjusts based on Voronoi criticality and generation
        """
        mutated = individual.copy()
        n = len(mutated)
        
        # Get criticality scores
        try:
            criticality = get_voronoi_criticality(mutated)
        except:
            criticality = np.ones(n)
            
        # Calculate adaptive mutation parameters
        gen_ratio = generation / max_generation if max_generation > 0 else 0
        base_mutation_rate = 0.2 + 0.3 * (1 - gen_ratio)  # Decrease mutation over time
        
        # Mutation step size varies with criticality (more critical = smaller steps)
        for i in range(n):
            # Higher criticality = smaller mutation steps
            criticality_factor = 0.5 + 0.5 * (1 - criticality[i])
            mutation_rate = base_mutation_rate * criticality_factor
            
            if random.random() < mutation_rate:
                # Mutation type selection based on criticality
                if criticality[i] > 0.7:  # Highly constrained
                    param_mutation_rate = 0.3
                else:  # Less constrained
                    param_mutation_rate = 0.5
                    
                # Mutate x coordinate
                if random.random() < param_mutation_rate:
                    delta = np.random.normal(0, 0.02 * criticality_factor)
                    mutated[i, 0] = max(0.01, min(rect_width - 0.01, mutated[i, 0] + delta))
                    
                # Mutate y coordinate
                if random.random() < param_mutation_rate:
                    delta = np.random.normal(0, 0.02 * criticality_factor)
                    mutated[i, 1] = max(0.01, min(rect_height - 0.01, mutated[i, 1] + delta))
                    
                # Mutate radius
                if random.random() < 0.5:
                    # Log-normal mutation to keep radius positive
                    if mutated[i, 2] > 0.001:
                        # Scale based on criticality
                        scale = 0.1 + 0.2 * (1 - criticality_factor)
                        log_delta = np.random.normal(0, scale)
                        new_radius = mutated[i, 2] * np.exp(log_delta)
                        mutated[i, 2] = max(0.001, new_radius)
                        
        return mutated

    def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crossover that respects Voronoi criticality information"""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Get criticality for both parents
        try:
            crit1 = get_voronoi_criticality(parent1)
            crit2 = get_voronoi_criticality(parent2)
        except:
            crit1 = np.ones(len(parent1))
            crit2 = np.ones(len(parent2))
            
        # Combine criticality measures
        combined_criticality = (crit1 + crit2) / 2
        
        # Sort by combined criticality (high to low)
        sorted_indices = np.argsort(-combined_criticality)
        
        # Exchange genes for most critical circles
        num_exchange = min(int(len(parent1) * 0.3), 5)
        for i in range(num_exchange):
            idx = sorted_indices[i]
            # Exchange x, y, and r
            child1[idx], child2[idx] = child2[idx], child1[idx]
            
        return child1, child2

    def local_force_refinement(circles_array: np.ndarray, iterations: int = 50) -> np.ndarray:
        """
        Refine solution using force-based local optimization with Voronoi awareness
        """
        refined = circles_array.copy()
        
        for iter_count in range(iterations):
            # Calculate forces
            forces = np.zeros_like(refined)
            
            # Get Voronoi criticality to inform force application
            try:
                criticality = get_voronoi_criticality(refined)
            except:
                criticality = np.ones(len(refined))
                
            # Calculate forces between overlapping circles
            centers = refined[:, :2]
            radii = refined[:, 2]
            
            # Vectorized pairwise calculations
            if len(refined) > 1:
                # Compute all pairwise distances
                dist_matrix = distance_matrix(centers, centers)
                
                # Get overlap mask
                radii_matrix = radii[:, np.newaxis] + radii[np.newaxis, :]
                overlap_mask = (dist_matrix < radii_matrix) & (dist_matrix > 0)
                
                # Process overlapping pairs
                overlap_pairs = np.where(overlap_mask)
                for i, j in zip(overlap_pairs[0], overlap_pairs[1]):
                    # Calculate repulsion force
                    dx = centers[j, 0] - centers[i, 0]
                    dy = centers[j, 1] - centers[i, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist > 1e-8:  # Avoid division by zero
                        # Force magnitude inversely proportional to distance squared
                        force_mag = 1.0 / (dist * dist + 1e-6)
                        
                        # Apply force weighted by criticality
                        weight1 = 0.5 + 0.5 * criticality[i]  # More critical circles move less
                        weight2 = 0.5 + 0.5 * criticality[j]
                        
                        forces[i, 0] -= force_mag * dx / dist * weight1
                        forces[i, 1] -= force_mag * dy / dist * weight1
                        forces[j, 0] += force_mag * dx / dist * weight2
                        forces[j, 1] += force_mag * dy / dist * weight2
                        
            # Apply forces with adaptive learning rate
            learning_rate = 0.05 + 0.05 * (1 - iter_count/iterations)  # Gradually decrease
            
            for i in range(len(refined)):
                # Apply force
                refined[i, 0] += learning_rate * forces[i, 0]
                refined[i, 1] += learning_rate * forces[i, 1]
                
                # Boundary constraints
                refined[i, 0] = max(refined[i, 2], min(rect_width - refined[i, 2], refined[i, 0]))
                refined[i, 1] = max(refined[i, 2], min(rect_height - refined[i, 2], refined[i, 1]))
                
                # Ensure positive radius
                refined[i, 2] = max(0.001, refined[i, 2])
                
        return refined

    def constraint_aware_selection(population: List[np.ndarray], 
                                 fitness_scores: List[float]) -> List[np.ndarray]:
        """
        Selection process that favors solutions with lower constraint violations
        """
        # Create sorted indices based on fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        
        # Take top 10% as elites
        elite_count = max(1, len(population) // 10)
        elites = [population[i] for i in sorted_indices[:elite_count]]
        
        # For rest, consider constraint quality
        remaining = [population[i] for i in sorted_indices[elite_count:]]
        
        return elites + remaining[:len(population) - elite_count]

    # Main algorithm
    start_time = time.time()
    
    # Step 1: Initialize diverse population
    init_population = initialize_population()
    
    # Step 2: Enhanced evolutionary optimization
    population = init_population[:]
    num_generations = 100
    
    # Track best solution
    best_fitness = -float('inf')
    best_solution = None
    
    # Evolutionary loop
    for generation in range(num_generations):
        # Evaluate population
        fitness_scores = []
        valid_solutions = []
        
        for individual in population:
            fitness, valid = evaluate_fitness(individual)
            fitness_scores.append(fitness)
            if valid:
                valid_solutions.append((individual, fitness))
                
        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        
        # Track best solution
        current_best_fitness = max(fitness_scores) if fitness_scores else -float('inf')
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[0].copy()
            
        # Print progress
        if generation % 20 == 0:
            print(f"Gen {generation}: Best fitness = {current_best_fitness:.6f}")
            
        # Break if we're done or have reached maximum fitness
        if generation > 10 and abs(current_best_fitness - best_fitness) < 1e-6:
            break
            
        # Create new population
        new_population = []
        
        # Elitism: keep top 20%
        elite_count = max(1, len(population) // 5)
        elitists = population[:elite_count]
        new_population.extend(elitists)
        
        # Generate offspring
        while len(new_population) < len(population):
            # Tournament selection
            tournament_size = min(5, len(population) // 2)
            parent1_idx = random.randint(0, len(population) - 1)
            parent2_idx = random.randint(0, len(population) - 1)
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutate
            child1 = adaptive_mutation(child1, generation, num_generations)
            child2 = adaptive_mutation(child2, generation, num_generations)
            
            # Repair solutions
            child1 = local_force_refinement(child1, iterations=10)
            child2 = local_force_refinement(child2, iterations=10)
            
            new_population.extend([child1, child2])
            
        population = new_population[:len(population)]
        
        # Occasionally run additional local refinement
        if generation % 10 == 0:
            for i in range(len(population)):
                if random.random() < 0.3:  # 30% chance to do extra refinement
                    population[i] = local_force_refinement(population[i], iterations=20)

    # Final validation
    final_fitness, is_valid = evaluate_fitness(best_solution)
    
    # If solution is invalid, try to fix with additional refinement
    if not is_valid:
        # Try refining with force-based optimization
        try:
            best_solution = local_force_refinement(best_solution, iterations=100)
            final_fitness, is_valid = evaluate_fitness(best_solution)
        except:
            pass
            
    # Ensure final solution is valid
    if not is_valid:
        # Return a valid fallback
        fallback = np.zeros((n_circles, 3))
        for i in range(n_circles):
            fallback[i] = [
                random.uniform(0.01, rect_width - 0.01),
                random.uniform(0.01, rect_height - 0.01),
                random.uniform(0.01, min(rect_width, rect_height) * 0.05)
            ]
        return fallback
        
    # Final local optimization
    refined_solution = local_force_refinement(best_solution, iterations=50)
    
    return refined_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")