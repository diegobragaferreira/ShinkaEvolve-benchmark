# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List, Optional
import time
from collections import defaultdict

# Set seeds for determinism
random.seed(42)
np.random.seed(42)

class VoronoiGuidedEvolution:
    """Voronoi-guided evolutionary optimization for circle packing."""
    
    def __init__(self, rect_width: float = 1.0, rect_height: float = 1.0, n_circles: int = 21):
        self.rect_width = rect_width
        self.rect_height = rect_height
        self.n_circles = n_circles
        self.population_size = 50
        self.generations = 100
        self.mutation_rate = 0.3
        self.crossover_rate = 0.7
        
    def compute_voronoi_areas(self, circles: np.ndarray) -> np.ndarray:
        """Compute Voronoi cell areas for each circle to determine constraint density."""
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
                            areas.append(float('inf'))
                    else:
                        areas.append(float('inf'))
                else:
                    areas.append(float('inf'))

            # Convert to numpy array and handle inf values
            areas_array = np.array(areas)
            # Replace infinity with a large value so it doesn't interfere with ranking
            areas_array[areas_array == float('inf')] = 1e10
            return areas_array
        except:
            # Fallback to uniform distribution if Voronoi fails
            return np.ones(len(circles)) * 100

    def evaluate_fitness(self, circles: np.ndarray) -> float:
        """Evaluate fitness as sum of radii."""
        return np.sum(circles[:, 2])

    def is_valid_solution(self, circles: np.ndarray) -> bool:
        """Check if all circles are within bounds and non-overlapping."""
        n = len(circles)

        # Check bounds
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0 or x + r > self.rect_width or y - r < 0 or y + r > self.rect_height:
                return False

        # Check overlaps using efficient pairwise comparison
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

    def generate_individual(self) -> np.ndarray:
        """Generate a random individual with valid constraints."""
        circles = np.zeros((self.n_circles, 3))
        
        # Generate initial random positions and radii
        for i in range(self.n_circles):
            # Position with margin from boundaries
            x = np.random.uniform(0.05 * self.rect_width, self.rect_width * 0.95)
            y = np.random.uniform(0.05 * self.rect_height, self.rect_height * 0.95)
            
            # Initial radius - start with medium size to allow growth
            max_radius = min(self.rect_width, self.rect_height) * 0.1
            r = np.random.uniform(0.02 * max_radius, 0.3 * max_radius)
            
            circles[i] = [x, y, r]
            
        # Ensure validity through iterative refinement
        for _ in range(100):
            if self.is_valid_solution(circles):
                break
            # If invalid, slightly adjust positions and radii
            for i in range(self.n_circles):
                if np.random.random() < 0.5:
                    circles[i, 0] = np.clip(circles[i, 0] + np.random.normal(0, 0.01), 
                                          circles[i, 2], self.rect_width - circles[i, 2])
                if np.random.random() < 0.5:
                    circles[i, 1] = np.clip(circles[i, 1] + np.random.normal(0, 0.01), 
                                          circles[i, 2], self.rect_height - circles[i, 2])
        return circles

    def initialize_population(self) -> List[np.ndarray]:
        """Initialize population with diverse valid individuals."""
        population = []
        for _ in range(self.population_size):
            individual = self.generate_individual()
            # Ensure individual is valid
            while not self.is_valid_solution(individual) and len(population) < self.population_size:
                individual = self.generate_individual()
            if self.is_valid_solution(individual):
                population.append(individual)
        return population

    def crossover_two_parents(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Create offspring by combining two parents with Voronoi-aware crossover."""
        offspring = parent1.copy()
        
        # Select random circles to take from parent2 (weighted by Voronoi areas)
        voronoi_areas1 = self.compute_voronoi_areas(parent1)
        voronoi_areas2 = self.compute_voronoi_areas(parent2)
        
        # Higher probability to take circles with higher constraint density (smaller Voronoi areas)
        weights1 = 1.0 / (voronoi_areas1 + 1e-8)
        weights2 = 1.0 / (voronoi_areas2 + 1e-8)
        
        # Normalize weights
        weights1 = weights1 / np.sum(weights1)
        weights2 = weights2 / np.sum(weights2)
        
        # Select circles to swap based on weighted probability
        for i in range(self.n_circles):
            if np.random.random() < self.crossover_rate:
                # Weighted selection from parent2
                offspring[i] = parent2[i].copy()
                
        return offspring

    def mutate_individual(self, individual: np.ndarray) -> np.ndarray:
        """Apply mutation with Voronoi-based adaptive rates."""
        mutated = individual.copy()
        
        # Compute Voronoi areas to determine mutation intensity
        voronoi_areas = self.compute_voronoi_areas(individual)
        
        # Higher mutation rate for areas with larger Voronoi cells (less constrained)
        # Lower mutation rate for areas with smaller Voronoi cells (more constrained)
        weights = 1.0 / (voronoi_areas + 1e-8)
        weights = weights / np.sum(weights)
        
        for i in range(self.n_circles):
            if np.random.random() < self.mutation_rate:
                # Adaptive mutation based on Voronoi density
                mutation_intensity = 0.02 * (1.0 + np.random.random() * 0.5)
                
                # More sensitive mutation in less constrained areas
                mutation_factor = 1.0 + (1.0 - weights[i]) * 0.5
                
                # Mutate position
                mutated[i, 0] = np.clip(
                    mutated[i, 0] + np.random.normal(0, mutation_intensity * mutation_factor), 
                    mutated[i, 2], self.rect_width - mutated[i, 2]
                )
                mutated[i, 1] = np.clip(
                    mutated[i, 1] + np.random.normal(0, mutation_intensity * mutation_factor), 
                    mutated[i, 2], self.rect_height - mutated[i, 2]
                )
                
                # Mutate radius
                delta_r = np.random.normal(0, mutation_intensity * 0.5 * mutation_factor)
                mutated[i, 2] = max(0.001, mutated[i, 2] + delta_r)
                
        return mutated

    def tournament_selection(self, population: List[np.ndarray], k: int = 5) -> np.ndarray:
        """Select individual using tournament selection."""
        tournament = random.sample(population, min(k, len(population)))
        return max(tournament, key=self.evaluate_fitness)

    def evolve(self) -> np.ndarray:
        """Main evolutionary algorithm loop."""
        # Initialize population
        population = self.initialize_population()
        
        best_fitness_history = []
        
        for gen in range(self.generations):
            # Evaluate fitness for all individuals
            fitness_scores = [self.evaluate_fitness(ind) for ind in population]
            best_fitness = max(fitness_scores)
            best_fitness_history.append(best_fitness)
            
            # Create new population through selection, crossover, and mutation
            new_population = []
            
            # Elitism: keep best individual
            best_individual = max(population, key=self.evaluate_fitness)
            new_population.append(best_individual)
            
            # Generate rest of population
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                
                # Crossover
                if np.random.random() < self.crossover_rate:
                    offspring = self.crossover_two_parents(parent1, parent2)
                else:
                    offspring = parent1.copy()
                
                # Mutation
                mutated_offspring = self.mutate_individual(offspring)
                
                # Ensure validity and add to population
                if self.is_valid_solution(mutated_offspring):
                    new_population.append(mutated_offspring)
                else:
                    # If invalid, try to fix it
                    fixed = self.fix_invalid(mutated_offspring)
                    if self.is_valid_solution(fixed):
                        new_population.append(fixed)
                    else:
                        # Random replacement
                        new_population.append(self.generate_individual())
            
            population = new_population
            
            # Early stopping if improvement is minimal
            if len(best_fitness_history) > 10:
                recent_improvement = best_fitness_history[-1] - best_fitness_history[-10]
                if recent_improvement < 1e-6:
                    break
                    
        # Return best individual from final population
        return max(population, key=self.evaluate_fitness)

    def fix_invalid(self, circles: np.ndarray) -> np.ndarray:
        """Attempt to fix an invalid configuration."""
        fixed = circles.copy()
        
        # Reduce radii to resolve overlaps
        for i in range(len(fixed)):
            for j in range(i+1, len(fixed)):
                x1, y1, r1 = fixed[i]
                x2, y2, r2 = fixed[j]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                if distance < (r1 + r2):
                    # Reduce radii to eliminate overlap
                    reduction = (r1 + r2 - distance) * 0.5
                    fixed[i, 2] = max(0.001, fixed[i, 2] - reduction)
                    fixed[j, 2] = max(0.001, fixed[j, 2] - reduction)
                    
        # Ensure bounds
        for i in range(len(fixed)):
            x, y, r = fixed[i]
            fixed[i] = [np.clip(x, r, self.rect_width - r),
                       np.clip(y, r, self.rect_height - r),
                       r]
        
        return fixed

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    # Using 1.2 width and 0.8 height for better packing efficiency
    rect_width = 1.2
    rect_height = 0.8
    
    # Use Voronoi-guided evolution algorithm
    evolver = VoronoiGuidedEvolution(rect_width, rect_height, 21)
    circles = evolver.evolve()
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")