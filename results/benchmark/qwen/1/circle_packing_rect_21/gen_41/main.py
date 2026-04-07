# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import random
from typing import Tuple
import time

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2, let's use 1x1 for simplicity (perimeter = 4)
    rect_width = 1.0
    rect_height = 1.0
    
    # Parameters
    n_circles = 21
    population_size = 50
    generations = 100
    mutation_rate = 0.3
    tournament_size = 5
    
    # Initialize random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    def create_initial_population(size: int) -> list:
        """Create initial population using hexagonal packing with random perturbations"""
        population = []
        
        # Hexagonal packing parameters
        radius_estimate = 0.1  # Initial radius estimate
        spacing = 2 * radius_estimate
        
        # Create hexagonal grid layout
        layout = []
        rows = int(np.sqrt(n_circles)) + 2
        cols = int(n_circles / rows) + 2
        
        for i in range(rows):
            for j in range(cols):
                x = (j + 0.5 * (i % 2)) * spacing + 0.1  # Add offset for hexagonal pattern
                y = i * spacing * np.sqrt(3)/2 + 0.1
                
                if len(layout) < n_circles:
                    layout.append((x, y, radius_estimate))
                else:
                    break
            if len(layout) >= n_circles:
                break
                
        # Adjust for rectangle bounds and add some randomness
        for _ in range(size):
            individual = []
            for i, (x, y, r) in enumerate(layout):
                # Apply small random perturbation
                new_x = max(0.01, min(rect_width - 0.01, x + np.random.normal(0, 0.02)))
                new_y = max(0.01, min(rect_height - 0.01, y + np.random.normal(0, 0.02)))
                new_r = max(0.001, min(0.2, r + np.random.normal(0, 0.01)))
                individual.append((new_x, new_y, new_r))
            
            population.append(individual)
        
        return population
    
    def compute_voronoi_constraint_density(circles: list) -> np.ndarray:
        """Compute constraint density using Voronoi diagram"""
        points = np.array([(c[0], c[1]) for c in circles])
        try:
            vor = Voronoi(points)
            # Calculate area of each Voronoi cell
            areas = []
            for i in range(len(vor.points)):
                region = vor.regions[vor.point_region[i]]
                if -1 not in region and len(region) > 0:
                    # Compute area of polygon
                    vertices = [vor.vertices[j] for j in region]
                    if len(vertices) >= 3:
                        # Simple polygon area calculation
                        area = 0
                        for k in range(len(vertices)):
                            j = (k + 1) % len(vertices)
                            area += vertices[k][0] * vertices[j][1]
                            area -= vertices[j][0] * vertices[k][1]
                        areas.append(abs(area) / 2)
                    else:
                        areas.append(1.0)
                else:
                    areas.append(1.0)
            return np.array(areas)
        except:
            # Fallback when Voronoi computation fails
            return np.ones(len(circles))
    
    def is_valid_solution(circles: list) -> bool:
        """Check if the configuration is valid (no overlaps, inside bounds)"""
        if len(circles) != n_circles:
            return False
            
        points = np.array([(c[0], c[1]) for c in circles])
        radii = np.array([c[2] for c in circles])
        
        # Check bounds
        for i, (x, y, r) in enumerate(circles):
            if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
                return False
        
        # Check overlaps
        distances = cdist(points, points)
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                if distances[i,j] < radii[i] + radii[j]:
                    return False
                    
        return True
    
    def evaluate_fitness(circles: list) -> float:
        """Evaluate fitness as sum of radii"""
        if not is_valid_solution(circles):
            return 0.0
        return sum(c[2] for c in circles)
    
    def mutate_individual(individual: list) -> list:
        """Mutate an individual with Voronoi-aware adjustments"""
        mutated = []
        constraint_densities = compute_voronoi_constraint_density(individual)
        
        for i, (x, y, r) in enumerate(individual):
            # Create new candidate with adaptive mutation based on constraint density
            if np.random.random() < mutation_rate:
                # Use constraint density to adjust mutation strength
                density_factor = np.clip(constraint_densities[i] / np.mean(constraint_densities), 0.1, 10.0)
                mutation_strength = 0.005 * density_factor
                
                # Mutate coordinates
                new_x = max(0.01, min(rect_width - 0.01, x + np.random.normal(0, mutation_strength)))
                new_y = max(0.01, min(rect_height - 0.01, y + np.random.normal(0, mutation_strength)))
                
                # Mutate radius with bounded adjustment
                new_r = max(0.001, min(0.3, r + np.random.normal(0, mutation_strength * 0.5)))
                mutated.append((new_x, new_y, new_r))
            else:
                mutated.append((x, y, r))
        return mutated
    
    def tournament_selection(population: list, fitnesses: list, k: int = tournament_size) -> list:
        """Select best individual from tournament"""
        tournament_indices = np.random.choice(len(population), size=k, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_index]
    
    # Main evolutionary loop
    population = create_initial_population(population_size)
    
    for gen in range(generations):
        # Evaluate fitness
        fitnesses = [evaluate_fitness(ind) for ind in population]
        
        # Keep track of best solution
        best_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_idx]
        
        # Create new population
        new_population = []
        
        # Elitism: keep best individual
        new_population.append(population[best_idx])
        
        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            parent = tournament_selection(population, fitnesses)
            
            # Mutation
            child = mutate_individual(parent)
            
            # Add to new population
            new_population.append(child)
        
        population = new_population
    
    # Get final best solution
    final_fitnesses = [evaluate_fitness(ind) for ind in population]
    final_best_idx = np.argmax(final_fitnesses)
    best_solution = population[final_best_idx]
    
    # Convert to numpy array
    result = np.zeros((n_circles, 3))
    for i, (x, y, r) in enumerate(best_solution):
        result[i] = [x, y, r]
    
    return result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
