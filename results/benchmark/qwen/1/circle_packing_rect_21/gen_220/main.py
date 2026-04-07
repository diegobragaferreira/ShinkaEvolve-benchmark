# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi, cKDTree
import random
from typing import Tuple, List, Optional
import time
from collections import defaultdict
import math

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

class CircleInitializer:
    """Handles all circle initialization strategies."""
    
    @staticmethod
    def hexagonal_lattice(n_circles: int, rect_width: float, rect_height: float) -> np.ndarray:
        """Initialize circle positions using a hexagonal lattice pattern."""
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
    
    @staticmethod
    def random_distribution(n_circles: int, rect_width: float, rect_height: float) -> np.ndarray:
        """Create random circle distribution."""
        circles = np.zeros((n_circles, 3))
        for i in range(n_circles):
            x = random.uniform(0.01, rect_width - 0.01)
            y = random.uniform(0.01, rect_height - 0.01)
            r = random.uniform(0.01, min(rect_width, rect_height) * 0.1)
            circles[i] = [x, y, r]
        return circles

class FitnessCalculator:
    """Calculates fitness with optimized spatial checks."""
    
    @staticmethod
    def calculate_fitness(circles: np.ndarray, rect_width: float, rect_height: float) -> Tuple[float, float]:
        """
        Enhanced fitness calculation with optimized spatial checks.
        Returns (total_fitness, overlap_penalty)
        """
        n = len(circles)

        # Check boundary constraints efficiently
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

        # Check overlap constraints using spatial hashing for efficiency
        overlap_penalty = 0.0
        if n > 1:
            # Use optimized approach based on population size
            if n > 50:
                overlap_penalty = FitnessCalculator._calculate_fitness_with_spatial_hash(circles, rect_width, rect_height)
            elif n > 10:
                # Use optimized dense matrix approach for moderate populations
                overlap_penalty = FitnessCalculator._calculate_fitness_with_distance_matrix(circles)
            else:
                # Use brute force for very small populations
                overlap_penalty = FitnessCalculator._calculate_fitness_brute_force(circles)

        # Fitness is sum of radii minus penalties
        total_radius = np.sum(circles[:, 2])
        fitness = total_radius - penalty - overlap_penalty

        return fitness, overlap_penalty
    
    @staticmethod
    def _calculate_fitness_with_distance_matrix(circles: np.ndarray) -> float:
        """Calculate overlap penalty using distance matrix approach."""
        n = len(circles)
        coords = circles[:, :2]
        radii = circles[:, 2]
        
        try:
            dist_matrix = cdist(coords, coords)
            overlap_penalty = 0.0
            
            # For each pair of circles, check overlap
            for i in range(n):
                for j in range(i+1, n):
                    distance = dist_matrix[i, j]
                    min_distance = radii[i] + radii[j]

                    if distance < min_distance:
                        # Overlap exists
                        overlap_amount = min_distance - distance
                        overlap_penalty += overlap_amount * 1000  # Heavy penalty for overlaps
            
            return overlap_penalty
        except:
            return FitnessCalculator._calculate_fitness_brute_force(circles)
    
    @staticmethod
    def _calculate_fitness_brute_force(circles: np.ndarray) -> float:
        """Calculate overlap penalty using brute force approach."""
        n = len(circles)
        overlap_penalty = 0.0
        
        for i in range(n):
            for j in range(i+1, n):
                distance = np.linalg.norm(circles[i, :2] - circles[j, :2])
                min_distance = circles[i, 2] + circles[j, 2]

                if distance < min_distance:
                    # Overlap exists
                    overlap_amount = min_distance - distance
                    overlap_penalty += overlap_amount * 1000  # Heavy penalty for overlaps
                    
        return overlap_penalty
    
    @staticmethod
    def _calculate_fitness_with_spatial_hash(circles: np.ndarray, rect_width: float, rect_height: float) -> float:
        """Calculate overlap penalty using spatial hash approach."""
        class SpatialHashGrid:
            def __init__(self, width: float, height: float, cell_size: float = None):
                self.width = width
                self.height = height
                self.cell_size = cell_size if cell_size else 0.1
                self.grid = defaultdict(list)
                
            def _hash(self, x: float, y: float) -> Tuple[int, int]:
                """Convert world coordinates to grid cell coordinates."""
                return (int(x // self.cell_size), int(y // self.cell_size))
            
            def add_circle(self, idx: int, x: float, y: float, radius: float):
                """Add a circle to the spatial grid."""
                hash_key = self._hash(x, y)
                self.grid[hash_key].append((idx, x, y, radius))
                
            def get_neighbors(self, x: float, y: float, radius: float) -> List[Tuple[int, float, float, float]]:
                """Get all circles in neighboring cells that might collide."""
                neighbors = []
                center_cell = self._hash(x, y)
                
                # Check surrounding cells (3x3 grid)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        cell = (center_cell[0] + dx, center_cell[1] + dy)
                        if cell in self.grid:
                            for idx, cx, cy, cr in self.grid[cell]:
                                if idx != -1:  # Skip dummy entries
                                    neighbors.append((idx, cx, cy, cr))
                return neighbors
        
        # Calculate spatial hash overlap penalty
        n = len(circles)
        overlap_penalty = 0.0
        
        # Create spatial grid
        spatial_grid = SpatialHashGrid(rect_width, rect_height)
        
        # Add all circles to spatial grid
        for i in range(n):
            x, y, r = circles[i]
            spatial_grid.add_circle(i, x, y, r)
            
        # Check overlap constraints
        for i in range(n):
            x, y, r = circles[i]
            neighbors = spatial_grid.get_neighbors(x, y, r)
            for j, nx, ny, nr in neighbors:
                if i >= j:  # Avoid double counting
                    continue
                # Calculate distance between circle centers
                dx = x - nx
                dy = y - ny
                distance = math.sqrt(dx*dx + dy*dy)
                min_distance = r + nr
                
                if distance < min_distance:
                    # Overlap exists
                    overlap_amount = min_distance - distance
                    overlap_penalty += overlap_amount * 1000  # Heavy penalty for overlaps
                    
        return overlap_penalty

class MutationEngine:
    """Manages adaptive mutation strategies."""
    
    @staticmethod
    def compute_constraint_density(circles: np.ndarray) -> np.ndarray:
        """Compute constraint density for each circle."""
        n = len(circles)
        if n < 2:
            return np.zeros(n)
        
        # Simple density approximation using nearest neighbors
        centers = circles[:, :2]
        constraint_density = np.zeros(n)
        
        # For each circle, count nearby circles 
        for i in range(n):
            center_i = centers[i]
            nearby_count = 0
            max_radius = np.max(circles[:, 2])
            threshold = 3 * max_radius
            
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(center_i - centers[j])
                    if dist < threshold:
                        nearby_count += 1
                        
            constraint_density[i] = nearby_count / max(1, n - 1)
        
        return constraint_density
    
    @staticmethod
    def mutate_circles_adaptive(circles: np.ndarray, constraint_densities: np.ndarray, 
                               mutation_rate: float = 0.1, rect_width: float = 1.0, 
                               rect_height: float = 1.0) -> np.ndarray:
        """Mutate circle positions and radii with adaptive strategy."""
        mutated = circles.copy()

        for i in range(len(mutated)):
            x, y, r = mutated[i]
            
            # Use constraint density for adaptive mutation
            density_weight = 1.0 + constraint_densities[i] * 2.0  # Range 1.0 to 3.0
            
            pos_mutation_strength = 0.02 / density_weight
            rad_mutation_strength = 0.01 / density_weight
            
            # Mutate position
            if random.random() < mutation_rate:
                x += np.random.normal(0, pos_mutation_strength)
                y += np.random.normal(0, pos_mutation_strength)

                # Ensure position stays within bounds
                x = np.clip(x, r, rect_width - r)
                y = np.clip(y, r, rect_height - r)

            # Mutate radius
            if random.random() < mutation_rate:
                r += np.random.normal(0, rad_mutation_strength)
                # Ensure radius remains positive
                r = max(0.001, r)

            mutated[i] = [x, y, r]

        return mutated

class PopulationManager:
    """Handles evolution operations."""
    
    @staticmethod
    def tournament_selection(population: List[np.ndarray], fitness_scores: List[float], 
                            tournament_size: int = 7) -> int:
        """Select parent using tournament selection."""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        return winner_index
    
    @staticmethod
    def crossover_circles(parent1: np.ndarray, parent2: np.ndarray, crossover_rate: float = 0.8) -> np.ndarray:
        """Perform uniform crossover between two circle configurations."""
        if random.random() > crossover_rate:
            return parent1.copy()  # Return first parent if no crossover

        offspring = parent1.copy()

        # Uniform crossover with bias towards better parent
        for i in range(len(offspring)):
            if random.random() < 0.5:
                offspring[i] = parent2[i].copy()

        return offspring
    
    @staticmethod
    def evolve_generation(population: List[np.ndarray], fitness_scores: List[float],
                         mutation_rate: float = 0.1, crossover_rate: float = 0.8,
                         rect_width: float = 1.0, rect_height: float = 1.0) -> List[np.ndarray]:
        """Evolve the population for one generation."""
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]  # Descending order
        population = [population[i] for i in sorted_indices]
        fitness_scores.sort(reverse=True)

        # Create new generation
        new_population = [population[0]]  # Elitism - keep best individual

        # Generate offspring
        while len(new_population) < len(population):
            # Tournament selection
            winner_index = PopulationManager.tournament_selection(population, fitness_scores)
            parent1 = population[winner_index]

            # Select second parent
            tournament_indices = list(range(len(population)))
            tournament_indices.remove(winner_index)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index2 = tournament_indices[np.argmax(tournament_fitness)]
            parent2 = population[winner_index2]

            # Crossover
            offspring = PopulationManager.crossover_circles(parent1, parent2, crossover_rate)

            # Mutation
            constraint_densities = MutationEngine.compute_constraint_density(offspring)
            offspring = MutationEngine.mutate_circles_adaptive(
                offspring, constraint_densities, mutation_rate, rect_width, rect_height
            )

            new_population.append(offspring)

        return new_population[:len(population)]

class CirclePackingOptimizer:
    """Main optimization engine with improved structure."""
    
    def __init__(self, n_circles: int = 21, rect_width: float = 1.0, rect_height: float = 1.0):
        self.n_circles = n_circles
        self.rect_width = rect_width
        self.rect_height = rect_height
        self.population_size = 100
        self.generations = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        
    def initialize_population(self) -> List[np.ndarray]:
        """Create diverse initial population."""
        population = []
        for _ in range(self.population_size):
            # Use hexagonal initialization for most
            circles = CircleInitializer.hexagonal_lattice(self.n_circles, self.rect_width, self.rect_height)
            # Add some randomness to initial positions
            for i in range(self.n_circles):
                circles[i][0] += np.random.uniform(-0.05, 0.05)
                circles[i][1] += np.random.uniform(-0.05, 0.05)
                circles[i][0] = np.clip(circles[i][0], circles[i][2], self.rect_width - circles[i][2])
                circles[i][1] = np.clip(circles[i][1], circles[i][2], self.rect_height - circles[i][2])
            population.append(circles)
        return population
    
    def refine_solution(self, circles: np.ndarray, iterations: int = 150) -> np.ndarray:
        """Apply local refinement to improve final solution."""
        refined = circles.copy()

        # Multi-phase refinement
        # Phase 1: Aggressive refinement with large steps
        for _ in range(iterations // 3):
            for i in range(len(refined)):
                # Save current state
                old_x, old_y, old_r = refined[i]

                # Try larger random moves for quick improvements
                new_x = old_x + np.random.normal(0, 0.01)
                new_y = old_y + np.random.normal(0, 0.01)
                new_r = old_r + np.random.normal(0, 0.002)

                # Clip to bounds
                new_x = np.clip(new_x, new_r, self.rect_width - new_r)
                new_y = np.clip(new_y, new_r, self.rect_height - new_r)
                new_r = max(0.001, new_r)

                # Test if this change improves fitness
                test_config = refined.copy()
                test_config[i] = [new_x, new_y, new_r]

                # Check if this move improves fitness
                current_fitness, _ = FitnessCalculator.calculate_fitness(refined, self.rect_width, self.rect_height)
                test_fitness, _ = FitnessCalculator.calculate_fitness(test_config, self.rect_width, self.rect_height)

                if test_fitness > current_fitness:
                    refined = test_config

        # Phase 2: Fine-grained refinement with small steps
        for _ in range(iterations // 3):
            for i in range(len(refined)):
                # Save current state
                old_x, old_y, old_r = refined[i]

                # Try very small random moves for fine-tuning
                new_x = old_x + np.random.normal(0, 0.002)
                new_y = old_y + np.random.normal(0, 0.002)
                new_r = old_r + np.random.normal(0, 0.0005)

                # Clip to bounds
                new_x = np.clip(new_x, new_r, self.rect_width - new_r)
                new_y = np.clip(new_y, new_r, self.rect_height - new_r)
                new_r = max(0.001, new_r)

                # Test if this change improves fitness
                test_config = refined.copy()
                test_config[i] = [new_x, new_y, new_r]

                # Check if this move improves fitness
                current_fitness, _ = FitnessCalculator.calculate_fitness(refined, self.rect_width, self.rect_height)
                test_fitness, _ = FitnessCalculator.calculate_fitness(test_config, self.rect_width, self.rect_height)

                if test_fitness > current_fitness:
                    refined = test_config

        # Phase 3: Diversification phase to escape local optima
        for _ in range(iterations // 3):
            # Random perturbation for diversification
            if random.random() < 0.3:
                i = random.randint(0, len(refined) - 1)
                old_x, old_y, old_r = refined[i]
                new_x = old_x + np.random.uniform(-0.02, 0.02)
                new_y = old_y + np.random.uniform(-0.02, 0.02)
                new_r = old_r + np.random.uniform(-0.005, 0.005)

                # Clip to bounds
                new_x = np.clip(new_x, new_r, self.rect_width - new_r)
                new_y = np.clip(new_y, new_r, self.rect_height - new_r)
                new_r = max(0.001, new_r)

                # Test if this change improves fitness
                test_config = refined.copy()
                test_config[i] = [new_x, new_y, new_r]

                # Check if this move improves fitness
                current_fitness, _ = FitnessCalculator.calculate_fitness(refined, self.rect_width, self.rect_height)
                test_fitness, _ = FitnessCalculator.calculate_fitness(test_config, self.rect_width, self.rect_height)

                if test_fitness > current_fitness:
                    refined = test_config

        return refined
    
    def optimize(self) -> np.ndarray:
        """Main optimization routine with improved structure."""
        # Stage 1: Global exploration with diverse initial population
        population = self.initialize_population()

        # Evolutionary loop
        for gen in range(self.generations):
            # Evaluate fitness of population
            fitness_scores = []
            for circles in population:
                fitness, _ = FitnessCalculator.calculate_fitness(circles, self.rect_width, self.rect_height)
                fitness_scores.append(fitness)

            # Print progress
            if gen % 20 == 0:
                best_fitness = max(fitness_scores)
                print(f"Generation {gen}, Best fitness: {best_fitness:.6f}")

            # Evolve population
            population = PopulationManager.evolve_generation(
                population, fitness_scores, 
                self.mutation_rate, self.crossover_rate,
                self.rect_width, self.rect_height
            )

        # Stage 2: Local refinement of best solution
        best_index = np.argmax([FitnessCalculator.calculate_fitness(ind, self.rect_width, self.rect_height)[0] 
                               for ind in population])
        best_solution = population[best_index]
        
        # Apply intensive local refinement with higher iteration count
        refined_solution = self.refine_solution(best_solution, iterations=200)
        
        return refined_solution

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

    # Initialize optimizer
    optimizer = CirclePackingOptimizer(
        n_circles=21,
        rect_width=rect_width,
        rect_height=rect_height
    )
    
    # Run optimization
    best_solution = optimizer.optimize()

    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")