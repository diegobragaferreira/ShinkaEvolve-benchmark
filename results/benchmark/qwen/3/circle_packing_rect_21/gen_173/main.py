# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
import random
from sklearn.cluster import KMeans
import time

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    width, height = 1.0, 1.0

    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    def compute_max_radius_at_point(point, circles, rect_width, rect_height):
        """Compute maximum possible radius for a circle centered at 'point'"""
        x, y = point
        # Distance to rectangle edges
        dist_to_edges = [
            x,                    # distance to left edge
            rect_width - x,       # distance to right edge
            y,                    # distance to bottom edge
            rect_height - y       # distance to top edge
        ]
        
        # Distance to other circles (excluding self)
        min_dist_to_others = float('inf')
        for i, (other_x, other_y, other_r) in enumerate(circles):
            if not (abs(other_x - x) < 1e-10 and abs(other_y - y) < 1e-10):
                dist = distance.euclidean(point, (other_x, other_y))
                min_dist_to_others = min(min_dist_to_others, dist)
        
        # Maximum radius is limited by both edges and other circles
        max_radius = min(min(dist_to_edges), min_dist_to_others/2.0)
        return max(0.001, max_radius)
    
    def evaluate_fitness(circles, rect_width, rect_height):
        """Evaluate fitness of configuration - sum of radii"""
        total_radius = np.sum([r for _, _, r in circles])
        return total_radius
    
    def validate_configuration(circles, rect_width, rect_height):
        """Validate that all circles are within bounds and non-overlapping"""
        for i, (x, y, r) in enumerate(circles):
            # Check boundary conditions
            if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
                return False

            # Check overlap with other circles
            for j, (x2, y2, r2) in enumerate(circles):
                if i != j:
                    distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if distance < r + r2 - 1e-10:
                        return False

        return True
    
    def generate_voronoi_seed_points(rect_width, rect_height, n_points):
        """Generate seed points using Voronoi-based approach"""
        # Start with strategic corner placements
        seed_points = []
        
        # Add corners
        seed_points.extend([
            (0.1 * rect_width, 0.1 * rect_height),
            (0.9 * rect_width, 0.1 * rect_height),
            (0.1 * rect_width, 0.9 * rect_height),
            (0.9 * rect_width, 0.9 * rect_height),
            (rect_width/2, rect_height/2)
        ])
        
        # Add edge centers
        seed_points.extend([
            (rect_width/2, 0.1 * rect_height),
            (rect_width/2, 0.9 * rect_height),
            (0.1 * rect_width, rect_height/2),
            (0.9 * rect_width, rect_height/2)
        ])
        
        # Add more points in a structured way to ensure good spatial distribution
        additional_points = []
        grid_size = 4
        for i in range(grid_size):
            for j in range(grid_size):
                x = 0.1 + (i * 0.8 / (grid_size - 1)) if grid_size > 1 else 0.5
                y = 0.1 + (j * 0.8 / (grid_size - 1)) if grid_size > 1 else 0.5
                additional_points.append((x * rect_width, y * rect_height))
        
        # Combine and remove duplicates
        all_points = seed_points + additional_points
        unique_points = []
        seen = set()
        for point in all_points:
            point_tuple = (round(point[0], 6), round(point[1], 6))
            if point_tuple not in seen:
                seen.add(point_tuple)
                unique_points.append(point)
        
        # If we need more points, generate them with noise
        while len(unique_points) < n_points:
            x = np.random.uniform(0.05 * rect_width, 0.95 * rect_width)
            y = np.random.uniform(0.05 * rect_height, 0.95 * rect_height)
            unique_points.append((x, y))
        
        return unique_points[:n_points]
    
    def voronoi_evolution_step(population, rect_width, rect_height, elite_ratio=0.2):
        """Perform one evolution step on population"""
        n_pop = len(population)
        elite_size = int(elite_ratio * n_pop)
        
        # Evaluate fitness of entire population
        fitness_scores = []
        for indiv in population:
            fitness = evaluate_fitness(indiv, rect_width, rect_height)
            fitness_scores.append(fitness)
        
        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        
        # Keep elite
        new_population = sorted_population[:elite_size]
        
        # Generate offspring through crossover and mutation
        while len(new_population) < n_pop:
            # Tournament selection for parents
            parent1_idx = tournament_selection(sorted_population, fitness_scores)
            parent2_idx = tournament_selection(sorted_population, fitness_scores)
            
            parent1 = sorted_population[parent1_idx]
            parent2 = sorted_population[parent2_idx]
            
            # Crossover
            child = crossover(parent1, parent2, rect_width, rect_height)
            
            # Mutation
            child = mutate(child, rect_width, rect_height)
            
            new_population.append(child)
        
        return new_population
    
    def tournament_selection(population, fitness_scores, tournament_size=3):
        """Select best individual from tournament"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return winner_index
    
    def crossover(parent1, parent2, rect_width, rect_height):
        """Create offspring via uniform crossover"""
        offspring = []
        for i in range(len(parent1)):
            if random.random() < 0.5:
                offspring.append(parent1[i])
            else:
                offspring.append(parent2[i])
        return offspring
    
    def mutate(individual, rect_width, rect_height, mutation_rate=0.1, max_perturbation=0.05):
        """Apply random mutations to individual"""
        mutated = []
        for i, (x, y, r) in enumerate(individual):
            if random.random() < mutation_rate:
                # Apply perturbation to position
                delta_x = np.random.uniform(-max_perturbation, max_perturbation)
                delta_y = np.random.uniform(-max_perturbation, max_perturbation)
                
                new_x = max(0.05, min(rect_width - 0.05, x + delta_x))
                new_y = max(0.05, min(rect_height - 0.05, y + delta_y))
                
                # Keep radius within reasonable bounds
                new_r = max(0.001, min(0.4, r + np.random.normal(0, 0.01)))
                
                mutated.append((new_x, new_y, new_r))
            else:
                mutated.append((x, y, r))
        return mutated
    
    # Phase 1: Voronoi-guided initialization
    initial_seed_points = generate_voronoi_seed_points(width, height, 21)
    
    # Generate initial population using Voronoi-inspired seeding
    population = []
    for _ in range(30):  # 30 individuals in population
        # Start with seed points
        circles = []
        for x, y in initial_seed_points:
            # Add small noise to initial positions
            noisy_x = max(0.05, min(width - 0.05, x + np.random.normal(0, 0.02)))
            noisy_y = max(0.05, min(height - 0.05, y + np.random.normal(0, 0.02)))
            
            # Compute initial radius
            initial_radius = min(noisy_x, width - noisy_x, noisy_y, height - noisy_y)
            circles.append((noisy_x, noisy_y, initial_radius))
            
        # Refine each circle's radius to fit the configuration
        for i in range(len(circles)):
            circles[i] = (
                circles[i][0],
                circles[i][1],
                compute_max_radius_at_point((circles[i][0], circles[i][1]), circles, width, height)
            )
        
        population.append(circles)
    
    # Phase 2: Evolutionary optimization with Voronoi constraints
    best_fitness = 0
    best_individual = None
    
    # Evolution parameters
    generations = 100
    for gen in range(generations):
        population = voronoi_evolution_step(population, width, height)
        
        # Evaluate and track best
        for individual in population:
            fitness = evaluate_fitness(individual, width, height)
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual
    
    # Phase 3: Final refinement with local optimization
    if best_individual is not None:
        refined_individual = best_individual[:]
        
        # Do intensive local search on best solution
        for iter_refine in range(2000):
            improved = False
            
            # Try to improve each circle
            for i in range(len(refined_individual)):
                x, y, r = refined_individual[i]
                old_r = r
                old_x, old_y = x, y
                
                # Try several perturbation strategies
                best_x, best_y, best_r = x, y, r
                best_fitness_val = evaluate_fitness(refined_individual, width, height)
                
                # Strategy 1: Small local search around current position
                for dx in [-0.02, -0.01, 0, 0.01, 0.02]:
                    for dy in [-0.02, -0.01, 0, 0.01, 0.02]:
                        new_x = max(0.05, min(width - 0.05, x + dx))
                        new_y = max(0.05, min(height - 0.05, y + dy))
                        
                        # Create temporary configuration
                        temp_config = refined_individual[:]
                        temp_config[i] = (new_x, new_y, r)
                        
                        # Compute new radius
                        new_r = compute_max_radius_at_point((new_x, new_y), temp_config, width, height)
                        temp_config[i] = (new_x, new_y, new_r)
                        
                        # Check if valid and better
                        if validate_configuration(temp_config, width, height):
                            new_fitness = evaluate_fitness(temp_config, width, height)
                            if new_fitness > best_fitness_val:
                                best_fitness_val = new_fitness
                                best_x, best_y, best_r = new_x, new_y, new_r
                                improved = True
                
                # Apply the best improvement if found
                if improved:
                    refined_individual[i] = (best_x, best_y, best_r)
            
            # Early stopping if no improvement
            if not improved and iter_refine > 1000:
                break
        
        # Final validation and convert to output format
        if validate_configuration(refined_individual, width, height):
            result = np.array([[x, y, r] for x, y, r in refined_individual])
            return result
    
    # Fallback to simple grid if everything fails
    circles = np.zeros((21, 3))
    rows, cols = 3, 7
    spacing_x = width / (cols + 1)
    spacing_y = height / (rows + 1)
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= 21:
                break
            x = (j + 1) * spacing_x
            y = (i + 1) * spacing_y
            r = 0.025
            circles[idx] = [x, y, r]
            idx += 1
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")