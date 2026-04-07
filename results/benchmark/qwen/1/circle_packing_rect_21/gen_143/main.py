# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List, Optional
import time

# Set seeds for determinism
random.seed(42)
np.random.seed(42)

class VoronoiEvolutionaryPack:
    """Voronoi-based evolutionary circle packing algorithm."""
    
    def __init__(self, rect_width: float = 1.2, rect_height: float = 0.8, n_circles: int = 21):
        self.rect_width = rect_width
        self.rect_height = rect_height
        self.n_circles = n_circles
        self.population_size = 50
        self.generations = 100
        self.elite_size = 5
        self.tournament_size = 3
        
    def compute_voronoi_area(self, circles: np.ndarray) -> np.ndarray:
        """Compute Voronoi cell areas for each circle to identify constraint density."""
        try:
            # Add boundary points for proper Voronoi calculation
            points = circles[:, :2].copy()

            # Add boundary points to make Voronoi more meaningful
            boundary_points = [
                [0, 0], [self.rect_width, 0], [0, self.rect_height], [self.rect_width, self.rect_height],
                [self.rect_width/2, 0], [self.rect_width/2, self.rect_height],
                [0, self.rect_height/2], [self.rect_width, self.rect_height/2]
            ]
            points = np.vstack([points, boundary_points])

            vor = Voronoi(points)

            # For each original point, compute Voronoi cell area
            areas = []
            for i in range(len(circles)):
                region_idx = np.where(vor.point_region == i)[0][0] if i in vor.point_region else -1

                if region_idx != -1 and region_idx < len(vor.regions):
                    region = vor.regions[region_idx]
                    if -1 not in region and len(region) >= 3:
                        # Compute area of polygon using shoelace formula
                        vertices = np.array([vor.vertices[j] for j in region if j >= 0])
                        if len(vertices) >= 3:
                            x = vertices[:, 0]
                            y = vertices[:, 1]
                            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                            areas.append(area)
                        else:
                            areas.append(1.0)
                    else:
                        areas.append(1.0)
                else:
                    areas.append(1.0)

            return np.array(areas)
        except:
            # Fallback to uniform distribution if Voronoi fails
            return np.ones(len(circles))
    
    def evaluate_fitness(self, circles: np.ndarray) -> float:
        """Evaluate fitness as sum of radii with Voronoi-based penalty."""
        # Base fitness is sum of radii
        base_fitness = np.sum(circles[:, 2])
        
        # Compute Voronoi areas to detect constraint density
        voronoi_areas = self.compute_voronoi_area(circles)
        
        # Penalize areas that are too small (indicates high constraint density)
        # Smaller Voronoi cells suggest more constrained regions
        penalty = 0
        for area in voronoi_areas:
            if area < 0.01:  # Very small Voronoi cells cause penalty
                penalty -= 1000 * (0.01 - area)
                
        return base_fitness + penalty
    
    def is_valid_solution(self, circles: np.ndarray) -> bool:
        """Check if all circles are within bounds and non-overlapping."""
        n = len(circles)

        # Check bounds
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0 or x + r > self.rect_width or y - r < 0 or y + r > self.rect_height:
                return False

        # Check overlaps efficiently
        if n > 1:
            positions = circles[:, :2]
            radii = circles[:, 2]

            # Use distance matrix for overlap detection
            distances = cdist(positions, positions)

            for i in range(n):
                for j in range(i+1, n):
                    if distances[i, j] < (radii[i] + radii[j]):
                        return False

        return True
    
    def generate_random_individual(self) -> np.ndarray:
        """Generate a random valid individual."""
        circles = np.zeros((self.n_circles, 3))
        for i in range(self.n_circles):
            # Generate random valid position and radius
            x = np.random.uniform(0.05 * self.rect_width, self.rect_width * 0.95)
            y = np.random.uniform(0.05 * self.rect_height, self.rect_height * 0.95)
            
            # Use a reasonable starting radius
            max_radius = min(self.rect_width, self.rect_height) * 0.1
            r = np.random.uniform(0.01, max_radius * 0.5)
            
            circles[i] = [x, y, r]
        return circles
    
    def generate_initial_population(self) -> List[np.ndarray]:
        """Generate initial population."""
        population = []
        for _ in range(self.population_size):
            individual = self.generate_random_individual()
            # Ensure it's valid by adjusting if necessary
            if not self.is_valid_solution(individual):
                # Simple correction: reduce radii to avoid overlap
                for i in range(self.n_circles):
                    individual[i, 2] *= 0.5
                # Clamp positions to valid range
                individual[:, 0] = np.clip(individual[:, 0], individual[:, 2], self.rect_width - individual[:, 2])
                individual[:, 1] = np.clip(individual[:, 1], individual[:, 2], self.rect_height - individual[:, 2])
            population.append(individual)
        return population
    
    def mutate_position(self, circles: np.ndarray, idx: int, voronoi_areas: np.ndarray) -> np.ndarray:
        """Mutate position with adaptive delta based on Voronoi area."""
        new_circles = circles.copy()
        old_x, old_y = new_circles[idx, 0], new_circles[idx, 1]
        
        # Adaptive delta based on Voronoi area (smaller areas = more constrained)
        if len(voronoi_areas) > idx:
            area = voronoi_areas[idx]
            # Smaller Voronoi area = more constrained = smaller mutation
            delta = max(0.001, 0.05 / (1.0 + area * 10.0))
        else:
            delta = 0.05
            
        delta_x = np.random.uniform(-delta, delta)
        delta_y = np.random.uniform(-delta, delta)

        new_x = old_x + delta_x
        new_y = old_y + delta_y

        # Ensure within bounds
        new_x = np.clip(new_x, 0.01, self.rect_width - 0.01)
        new_y = np.clip(new_y, 0.01, self.rect_height - 0.01)

        new_circles[idx, 0] = new_x
        new_circles[idx, 1] = new_y
        return new_circles
    
    def mutate_radius(self, circles: np.ndarray, idx: int, voronoi_areas: np.ndarray) -> np.ndarray:
        """Mutate radius with adaptive delta based on Voronoi area."""
        new_circles = circles.copy()
        old_r = new_circles[idx, 2]
        
        # Adaptive delta based on Voronoi area (smaller areas = more constrained)
        if len(voronoi_areas) > idx:
            area = voronoi_areas[idx]
            # Smaller Voronoi area = more constrained = smaller mutation
            delta = max(0.001, 0.02 / (1.0 + area * 10.0))
        else:
            delta = 0.02
            
        delta_r = np.random.uniform(-delta, delta)
        new_r = old_r + delta_r

        # Ensure positive radius
        new_r = max(0.001, new_r)
        new_circles[idx, 2] = new_r
        return new_circles
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Single-point crossover of two parents."""
        child = parent1.copy()
        crossover_point = random.randint(1, self.n_circles - 1)
        
        # Swap portions of the two parents
        child[crossover_point:] = parent2[crossover_point:]
        return child
    
    def tournament_selection(self, population: List[np.ndarray], fitnesses: List[float]) -> np.ndarray:
        """Select an individual using tournament selection."""
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_index].copy()
    
    def evolve(self) -> np.ndarray:
        """Main evolution loop."""
        # Generate initial population
        population = self.generate_initial_population()
        
        for gen in range(self.generations):
            # Evaluate fitness for entire population
            fitnesses = []
            for individual in population:
                if self.is_valid_solution(individual):
                    fitness = self.evaluate_fitness(individual)
                else:
                    # Invalid individuals get low fitness
                    fitness = -1000000
                fitnesses.append(fitness)
            
            # Sort by fitness descending
            sorted_indices = np.argsort(fitnesses)[::-1]
            population = [population[i] for i in sorted_indices]
            fitnesses = [fitnesses[i] for i in sorted_indices]
            
            # Keep elite individuals
            elite = population[:self.elite_size]
            
            # Create new population
            new_population = elite.copy()
            
            # Generate offspring through crossover and mutation
            while len(new_population) < self.population_size:
                # Select parents
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Apply mutations with adaptive rates
                voronoi_areas = self.compute_voronoi_area(child)
                
                # Mutate some randomly selected circles
                num_mutations = random.randint(1, min(5, self.n_circles))
                mutation_indices = random.sample(range(self.n_circles), num_mutations)
                
                for idx in mutation_indices:
                    if random.random() < 0.5:
                        child = self.mutate_position(child, idx, voronoi_areas)
                    else:
                        child = self.mutate_radius(child, idx, voronoi_areas)
                
                # Ensure child is valid
                if not self.is_valid_solution(child):
                    # Simple fix: reduce all radii and reposition
                    for i in range(self.n_circles):
                        child[i, 2] *= 0.9
                    child[:, 0] = np.clip(child[:, 0], child[:, 2], self.rect_width - child[:, 2])
                    child[:, 1] = np.clip(child[:, 1], child[:, 2], self.rect_height - child[:, 2])
                
                new_population.append(child)
            
            population = new_population[:self.population_size]
            
            # Print progress
            if gen % 20 == 0:
                best_fitness = max(fitnesses)
                print(f"Generation {gen}: Best Fitness = {best_fitness:.4f}")
        
        # Return best individual
        final_fitnesses = [self.evaluate_fitness(ind) if self.is_valid_solution(ind) else -1000000 
                          for ind in population]
        best_index = np.argmax(final_fitnesses)
        return population[best_index]

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    # Using 1.2 width and 0.8 height for better packing efficiency
    packer = VoronoiEvolutionaryPack(1.2, 0.8, 21)
    
    # Evolve solution
    circles = packer.evolve()
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")