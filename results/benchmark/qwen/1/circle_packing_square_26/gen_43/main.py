# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import time
from collections import defaultdict
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    n_circles = 26
    max_iterations = 2000
    population_size = 100
    elite_size = 10
    mutation_rate_initial = 0.3
    mutation_rate_min = 0.05
    crossover_rate = 0.8
    speciation_threshold = 0.15
    
    def calculate_sum_radii(circles):
        """Calculate the sum of radii for given circle configuration"""
        return np.sum(circles[:, 2])
    
    def is_valid_placement(circle, existing_circles):
        """Check if a circle is valid (within bounds and non-overlapping)"""
        x, y, r = circle
        
        # Check boundary constraints
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
            
        # Check overlap with existing circles
        for ex_circle in existing_circles:
            ex_x, ex_y, ex_r = ex_circle
            distance = np.sqrt((x - ex_x)**2 + (y - ex_y)**2)
            if distance < r + ex_r:
                return False
                
        return True
    
    def generate_initial_population(population_size, n_circles):
        """Generate initial population using Voronoi-inspired placement and grid fallback"""
        population = []
        
        for _ in range(population_size):
            circles = []
            # Try Voronoi-based initialization
            success = False
            try:
                # Generate candidate positions using a simple Voronoi-like approach
                candidates = []
                # Create a grid of potential positions
                grid_size = 6
                spacing = 1.0 / (grid_size + 1)
                
                # Add some randomness to avoid regular patterns
                for i in range(1, grid_size + 1):
                    for j in range(1, grid_size + 1):
                        x = i * spacing + np.random.uniform(-spacing/3, spacing/3)
                        y = j * spacing + np.random.uniform(-spacing/3, spacing/3)
                        candidates.append((x, y))
                
                # Place circles greedily
                placed_circles = []
                
                # Sort candidates by distance to center (prioritize center placement)
                center_distances = [np.sqrt((x-0.5)**2 + (y-0.5)**2) for x, y in candidates]
                sorted_indices = np.argsort(center_distances)
                
                # Place circles one by one
                for idx in sorted_indices:
                    x, y = candidates[idx]
                    
                    # Try different radii, starting from largest possible
                    max_radius = min(x, 1-x, y, 1-y)
                    if max_radius <= 0:
                        continue
                        
                    # Start with a reasonable initial radius
                    r = max_radius * 0.3
                    
                    # Binary search for maximum valid radius
                    low, high = 0.001, max_radius
                    best_radius = 0.001
                    
                    while high - low > 0.0001:
                        mid = (low + high) / 2
                        test_circle = (x, y, mid)
                        
                        if is_valid_placement(test_circle, placed_circles):
                            best_radius = mid
                            low = mid
                        else:
                            high = mid
                    
                    if best_radius > 0.001:
                        placed_circles.append((x, y, best_radius))
                        if len(placed_circles) >= n_circles:
                            break
                
                # If we didn't get enough circles, use fallback
                if len(placed_circles) < n_circles:
                    # Grid fallback approach
                    placed_circles = []
                    grid_size = int(np.ceil(np.sqrt(n_circles))) + 1
                    spacing = 1.0 / (grid_size + 1)
                    
                    for i in range(grid_size):
                        for j in range(grid_size):
                            if len(placed_circles) >= n_circles:
                                break
                            x = i * spacing + spacing/2
                            y = j * spacing + spacing/2
                            
                            # Adjust slightly for randomness
                            x += np.random.uniform(-spacing/6, spacing/6)
                            y += np.random.uniform(-spacing/6, spacing/6)
                            
                            max_radius = min(x, 1-x, y, 1-y)
                            if max_radius <= 0:
                                continue
                                
                            r = max_radius * 0.2
                            test_circle = (x, y, r)
                            
                            if is_valid_placement(test_circle, placed_circles):
                                placed_circles.append(test_circle)
                
                # If still insufficient, add random circles
                while len(placed_circles) < n_circles:
                    x = np.random.uniform(0.01, 0.99)
                    y = np.random.uniform(0.01, 0.99)
                    max_radius = min(x, 1-x, y, 1-y)
                    if max_radius <= 0:
                        continue
                    r = max_radius * np.random.uniform(0.1, 0.5)
                    test_circle = (x, y, r)
                    if is_valid_placement(test_circle, placed_circles):
                        placed_circles.append(test_circle)
                
                # Ensure exactly n_circles
                if len(placed_circles) > n_circles:
                    placed_circles = placed_circles[:n_circles]
                elif len(placed_circles) < n_circles:
                    # Fill remaining spots with random circles
                    for _ in range(n_circles - len(placed_circles)):
                        x = np.random.uniform(0.01, 0.99)
                        y = np.random.uniform(0.01, 0.99)
                        max_radius = min(x, 1-x, y, 1-y)
                        if max_radius <= 0:
                            continue
                        r = max_radius * np.random.uniform(0.05, 0.3)
                        test_circle = (x, y, r)
                        if is_valid_placement(test_circle, placed_circles):
                            placed_circles.append(test_circle)
                
                population.append(np.array(placed_circles))
                success = True
                
            except Exception as e:
                pass
            
            if not success:
                # Fallback: generate completely random valid circles
                circles = []
                while len(circles) < n_circles:
                    x = np.random.uniform(0.01, 0.99)
                    y = np.random.uniform(0.01, 0.99)
                    max_radius = min(x, 1-x, y, 1-y)
                    if max_radius <= 0:
                        continue
                    r = max_radius * np.random.uniform(0.05, 0.3)
                    test_circle = (x, y, r)
                    if is_valid_placement(test_circle, circles):
                        circles.append(test_circle)
                population.append(np.array(circles))
        
        return population
    
    def evaluate_fitness(population):
        """Evaluate fitness of population"""
        fitnesses = []
        for circles in population:
            fitness = calculate_sum_radii(circles)
            fitnesses.append(fitness)
        return fitnesses
    
    def reproduce(parent1, parent2):
        """Create offspring from two parents"""
        if np.random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
            
        # Uniform crossover
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for i in range(len(child1)):
            if np.random.random() < 0.5:
                child1[i] = parent2[i].copy()
                child2[i] = parent1[i].copy()
        
        return child1, child2
    
    def mutate(individual, mutation_rate, generation, max_generations):
        """Mutate individual with adaptive rate"""
        # Adapt mutation rate based on generation
        current_mutation_rate = max(mutation_rate_min, 
                                  mutation_rate_initial * (1 - generation/max_generations))
        
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if np.random.random() < current_mutation_rate:
                # Randomize position and radius
                mutated[i][0] = np.clip(mutated[i][0] + np.random.normal(0, 0.05), 0, 1)
                mutated[i][1] = np.clip(mutated[i][1] + np.random.normal(0, 0.05), 0, 1)
                mutated[i][2] = np.clip(mutated[i][2] + np.random.normal(0, 0.03), 0.001, 0.5)
                
                # Repair if needed
                if not is_valid_placement(mutated[i], mutated[:i] + mutated[i+1:]):
                    # Try to fix by reducing radius
                    mutated[i][2] = min(mutated[i][2], 
                                      min(mutated[i][0], 1-mutated[i][0], 
                                          mutated[i][1], 1-mutated[i][1]))
        return mutated
    
    def speciate(population, fitnesses):
        """Divide population into species based on geometric similarity"""
        if len(population) == 0:
            return [[]]
            
        # Calculate average centers for each individual
        centers = []
        for ind in population:
            avg_x = np.mean(ind[:, 0])
            avg_y = np.mean(ind[:, 1])
            centers.append([avg_x, avg_y])
            
        centers = np.array(centers)
        
        # Cluster similar individuals
        species = defaultdict(list)
        
        # Simple distance-based clustering
        for i, center in enumerate(centers):
            found_species = False
            for species_id, species_centers in species.items():
                distances = [np.linalg.norm(center - s_center) for s_center in species_centers]
                if min(distances) < speciation_threshold:
                    species[species_id].append(i)
                    found_species = True
                    break
            
            if not found_species:
                species[len(species)].append(i)
        
        return list(species.values())
    
    def repair_individual(individual):
        """Repair an individual to ensure all constraints are met"""
        repaired = individual.copy()
        
        # Apply constraints step by step
        for i in range(len(repaired)):
            # Adjust position and radius to fit within bounds
            x, y, r = repaired[i]
            
            # Adjust radius to fit in bounds
            r = min(r, x, 1-x, y, 1-y)
            
            # Adjust position to keep radius valid
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            
            repaired[i] = [x, y, r]
            
            # Resolve overlaps
            for j in range(len(repaired)):
                if i != j:
                    x1, y1, r1 = repaired[i]
                    x2, y2, r2 = repaired[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    
                    if distance < r1 + r2:
                        # Move circles apart
                        dx = x2 - x1
                        dy = y2 - y1
                        dist = max(distance, 0.001)
                        
                        # Calculate overlap amount
                        overlap = r1 + r2 - dist
                        
                        # Move them apart
                        move_factor = overlap / (dist * 2)
                        x1 -= dx * move_factor
                        y1 -= dy * move_factor
                        
                        # Clip back to bounds
                        x1 = np.clip(x1, r1, 1-r1)
                        y1 = np.clip(y1, r1, 1-r1)
                        
                        repaired[i] = [x1, y1, r1]
        
        return repaired
    
    # Main evolutionary loop
    population = generate_initial_population(population_size, n_circles)
    
    best_fitness = 0
    best_individual = None
    stagnation_counter = 0
    max_stagnation = 100
    
    for generation in range(max_iterations):
        # Evaluate fitness
        fitnesses = evaluate_fitness(population)
        
        # Track best solution
        current_best_idx = np.argmax(fitnesses)
        current_best_fitness = fitnesses[current_best_idx]
        
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[current_best_idx].copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        # Check for convergence
        if stagnation_counter > max_stagnation:
            break
            
        # Select elite
        elite_indices = np.argsort(fitnesses)[-elite_size:]
        elite = [population[i] for i in elite_indices]
        
        # Speciation
        species = speciate(population, fitnesses)
        
        # Create new population
        new_population = elite.copy()
        
        # Fill rest of population
        while len(new_population) < population_size:
            # Select parents using tournament selection
            tournament_size = 3
            parent1_idx = random.choice(
                np.argsort(fitnesses)[-tournament_size:].tolist()
            )
            parent2_idx = random.choice(
                np.argsort(fitnesses)[-tournament_size:].tolist()
            )
            
            # Recombine
            child1, child2 = reproduce(population[parent1_idx], population[parent2_idx])
            
            # Mutate
            child1 = mutate(child1, mutation_rate_initial, generation, max_iterations)
            child2 = mutate(child2, mutation_rate_initial, generation, max_iterations)
            
            # Repair
            child1 = repair_individual(child1)
            child2 = repair_individual(child2)
            
            new_population.extend([child1, child2])
            
        # Trim to exact population size
        population = new_population[:population_size]
        
        # Occasionally add some diversity with fresh individuals
        if generation % 50 == 0 and generation > 0:
            fresh_pop = generate_initial_population(min(20, population_size//5), n_circles)
            population = elite + fresh_pop + population[elite_size:-len(fresh_pop)]
    
    # Final repair of best individual
    if best_individual is not None:
        best_individual = repair_individual(best_individual)
        return best_individual
    else:
        # Return best from final population if nothing was tracked
        final_fitnesses = evaluate_fitness(population)
        final_best_idx = np.argmax(final_fitnesses)
        return population[final_best_idx]

# EVOLVE-BLOCK-END
