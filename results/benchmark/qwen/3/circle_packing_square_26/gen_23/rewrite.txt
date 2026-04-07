# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

class SpatialCirclePacker:
    def __init__(self, n_circles: int = 26):
        self.n_circles = n_circles
        self.grid_size = 20  # Grid size for spatial indexing
        
    def initialize_individual(self) -> np.ndarray:
        """Initialize a random individual with Voronoi-inspired seeding"""
        individual = np.zeros((self.n_circles, 3))
        
        # Create initial distribution using a grid-like pattern with some randomness
        rows = int(np.ceil(np.sqrt(self.n_circles)))
        cols = int(np.ceil(self.n_circles / rows))
        
        base_positions = []
        for i in range(rows):
            for j in range(cols):
                if len(base_positions) >= self.n_circles:
                    break
                x = 0.1 + 0.8 * i / (rows - 1) if rows > 1 else 0.5
                y = 0.1 + 0.8 * j / (cols - 1) if cols > 1 else 0.5
                base_positions.append((x, y))
        
        # Add randomness to positions
        for i in range(self.n_circles):
            individual[i, 0] = max(0.01, min(0.99, base_positions[i][0] + np.random.normal(0, 0.05)))
            individual[i, 1] = max(0.01, min(0.99, base_positions[i][1] + np.random.normal(0, 0.05)))
            
            # Set initial radius based on available space
            max_radius = min(0.5 - individual[i, 0], 0.5 - individual[i, 1], 
                           individual[i, 0], individual[i, 1])
            individual[i, 2] = np.random.uniform(0.001, max_radius * 0.5)
            
        # Adjust for overlaps
        self._adjust_for_overlaps(individual)
        return individual
    
    def check_containment_constraints(self, individual: np.ndarray) -> bool:
        """Check if all circles are contained within the unit square"""
        for i in range(len(individual)):
            x, y, r = individual[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        return True
    
    def get_overlaps(self, individual: np.ndarray) -> List[Tuple[int, int, float]]:
        """Get list of overlapping circle pairs using spatial indexing for efficiency"""
        if len(individual) <= 1:
            return []
            
        # Use KDTree for efficient neighbor search
        positions = individual[:, :2]
        tree = cKDTree(positions)
        
        # Find neighbors within a distance of 2*max_radius
        max_radius = np.max(individual[:, 2])
        pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')
        
        overlaps = []
        for i, j in pairs:
            if i < j:  # Ensure each pair is considered once
                x1, y1, r1 = individual[i]
                x2, y2, r2 = individual[j]
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if distance < r1 + r2:
                    overlap = (r1 + r2) - distance
                    overlaps.append((i, j, overlap))
                    
        return overlaps
    
    def check_overlap_constraints(self, individual: np.ndarray) -> bool:
        """Check if any circles overlap"""
        return len(self.get_overlaps(individual)) == 0
    
    def calculate_total_radius(self, individual: np.ndarray) -> float:
        """Calculate sum of all radii"""
        return np.sum(individual[:, 2])
    
    def evaluate_fitness(self, individual: np.ndarray) -> float:
        """Evaluate fitness with proper penalty system"""
        # Basic fitness is sum of radii
        total_radius = self.calculate_total_radius(individual)
        
        # Check constraints
        if not self.check_containment_constraints(individual):
            return -1000.0  # Heavy penalty for containment violations
            
        if not self.check_overlap_constraints(individual):
            # Calculate penalty based on overlap amounts
            overlaps = self.get_overlaps(individual)
            penalty = sum(overlap * 500 for _, _, overlap in overlaps)
            return total_radius - penalty
            
        return total_radius
    
    def _adjust_for_overlaps(self, individual: np.ndarray, iterations: int = 10):
        """Adjust positions to resolve overlaps"""
        for _ in range(iterations):
            overlaps = self.get_overlaps(individual)
            if not overlaps:
                break
                
            # Resolve each overlap
            for i, j, overlap in overlaps:
                x1, y1, r1 = individual[i]
                x2, y2, r2 = individual[j]
                
                dx = x2 - x1
                dy = y2 - y1
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance > 0:
                    push_amount = overlap * 0.5
                    push_x = dx / distance * push_amount
                    push_y = dy / distance * push_amount
                    
                    individual[i, 0] -= push_x
                    individual[i, 1] -= push_y
                    individual[j, 0] += push_x
                    individual[j, 1] += push_y
                else:
                    # Random movement if identical positions
                    angle = np.random.uniform(0, 2*np.pi)
                    push_dist = overlap * 0.5
                    individual[i, 0] -= push_dist * np.cos(angle)
                    individual[i, 1] -= push_dist * np.sin(angle)
                    individual[j, 0] += push_dist * np.cos(angle)
                    individual[j, 1] += push_dist * np.sin(angle)
                
                # Keep within bounds
                individual[i, 0] = np.clip(individual[i, 0], r1, 1-r1)
                individual[i, 1] = np.clip(individual[i, 1], r1, 1-r1)
                individual[j, 0] = np.clip(individual[j, 0], r2, 1-r2)
                individual[j, 1] = np.clip(individual[j, 1], r2, 1-r2)
    
    def tournament_selection(self, population: List[np.ndarray], 
                           fitness_scores: List[float], 
                           tournament_size: int = 3) -> np.ndarray:
        """Select an individual using tournament selection"""
        tournament_indices = random.sample(range(len(population)), 
                                          min(tournament_size, len(population)))
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_index]
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform crossover between two parents"""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Single point crossover on positions and radii
        crossover_point = random.randint(1, len(parent1) - 1)
        
        # Swap positions and radii for half the circles
        child1[crossover_point:, :2] = parent2[crossover_point:, :2]
        child1[crossover_point:, 2] = parent2[crossover_point:, 2]
        
        child2[crossover_point:, :2] = parent1[crossover_point:, :2]
        child2[crossover_point:, 2] = parent1[crossover_point:, 2]
        
        # Apply local refinement to keep children valid
        self._adjust_for_overlaps(child1)
        self._adjust_for_overlaps(child2)
        
        return child1, child2
    
    def mutate_individual(self, individual: np.ndarray, mutation_rate: float, 
                         generation: int, max_generations: int):
        """Mutate an individual with adaptive mutation rate"""
        # Adaptive mutation rate: decreases over time
        adaptive_rate = mutation_rate * (1 - generation / max_generations)
        adaptive_rate = max(adaptive_rate, 0.01)  # Minimum mutation rate
        
        for i in range(len(individual)):
            if random.random() < adaptive_rate:
                # Mutate position slightly
                individual[i, 0] += np.random.normal(0, 0.01)
                individual[i, 1] += np.random.normal(0, 0.01)
                
                # Keep within bounds
                individual[i, 0] = np.clip(individual[i, 0], 0.01, 0.99)
                individual[i, 1] = np.clip(individual[i, 1], 0.01, 0.99)
                
                # Mutate radius
                individual[i, 2] += np.random.normal(0, 0.005)
                individual[i, 2] = max(0.001, individual[i, 2])
                
                # Keep within bounds
                max_radius = min(0.5 - individual[i, 0], 0.5 - individual[i, 1], 
                               individual[i, 0], individual[i, 1])
                individual[i, 2] = min(individual[i, 2], max_radius * 0.9)
    
    def optimize_circles_evolutionary(self) -> np.ndarray:
        """Main evolutionary optimization loop"""
        # Parameters for evolutionary algorithm
        population_size = 100
        generations = 500
        mutation_rate = 0.1
        elite_size = 10
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = self.initialize_individual()
            population.append(individual)
        
        # Evolutionary loop
        best_fitness_history = []
        start_time = time.time()
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = [self.evaluate_fitness(individual) for individual in population]
            
            # Track best fitness
            best_fitness = max(fitness_scores)
            best_fitness_history.append(best_fitness)
            
            # Select top individuals (elitism)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            elite = [population[i] for i in sorted_indices[:elite_size]]
            
            # Generate new population
            new_population = elite.copy()
            
            # Fill rest of population through crossover and mutation
            while len(new_population) < population_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                if random.random() < 0.8:  # Crossover probability
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Apply mutation
                self.mutate_individual(child1, mutation_rate, generation, generations)
                self.mutate_individual(child2, mutation_rate, generation, generations)
                
                new_population.extend([child1, child2])
            
            # Trim to exact population size
            population = new_population[:population_size]
            
            # Print progress
            if generation % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Generation {generation}: Best fitness = {best_fitness:.6f} (Time: {elapsed:.2f}s)")
        
        # Return best solution
        final_fitness_scores = [self.evaluate_fitness(ind) for ind in population]
        best_index = np.argmax(final_fitness_scores)
        return population[best_index]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    packer = SpatialCirclePacker(26)
    return packer.optimize_circles_evolutionary()

# EVOLVE-BLOCK-END